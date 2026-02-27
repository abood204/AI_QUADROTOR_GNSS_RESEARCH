"""Deploy a trained PPO policy on the AirSim Quadrotor with safety monitor.

The safety monitor sits between the policy output and AirSim commands,
enforcing velocity limits, proximity scaling, and emergency stop.

Usage:
    python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip
    python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip --no_safety
    python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip --speed_scale 2.5
"""

import argparse
import math
import os

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environments.airsim_env import AirSimDroneEnv
from src.safety.monitor import SafetyMonitor


def make_env(cfg: dict):
    def _init():
        return Monitor(AirSimDroneEnv(cfg))
    return _init


def _get_center_roi_min_depth(raw_env: AirSimDroneEnv) -> float:
    """Extract minimum depth from center 30% ROI of depth image.

    Uses the last captured depth image from the environment state.
    """
    depth_img = raw_env.state["image"]
    if depth_img is None or depth_img.size == 0:
        return float("inf")

    h, w = depth_img.shape[:2]
    margin_h = int(h * 0.35)
    margin_w = int(w * 0.35)
    center_roi = depth_img[margin_h:h - margin_h, margin_w:w - margin_w]

    if center_roi.size == 0:
        return float("inf")

    # Depth is normalized [0, 1] — convert back to meters
    min_normalized = float(np.min(center_roi))
    return min_normalized * raw_env.depth_clip_m


def main():
    parser = argparse.ArgumentParser(description="Deploy PPO on AirSim")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model .zip",
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_ppo.yaml",
        help="Path to YAML config (must match training config)",
    )
    parser.add_argument(
        "--safety_config", type=str, default="configs/safety.yaml",
        help="Safety monitor config",
    )
    parser.add_argument(
        "--duration_s", type=float, default=60.0,
        help="Max deployment duration in seconds",
    )
    parser.add_argument(
        "--no_safety", action="store_true",
        help="Disable safety monitor (for ablation comparison)",
    )
    parser.add_argument(
        "--speed_scale", type=float, default=1.0,
        help="Multiply max_vx / max_vy / max_yaw_rate by this factor at deploy time. "
             "1.0 = training speed. 2.0 = double speed. Does not require retraining.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Scale velocities for deploy without affecting the observation space shape
    if args.speed_scale != 1.0:
        env_cfg = cfg.setdefault("env", {})
        env_cfg["max_vx"] = env_cfg.get("max_vx", 3.0) * args.speed_scale
        env_cfg["max_vy"] = env_cfg.get("max_vy", 1.0) * args.speed_scale
        env_cfg["max_yaw_rate_deg"] = env_cfg.get("max_yaw_rate_deg", 45) * args.speed_scale
        print(f"[deploy] Speed scale: {args.speed_scale}x  "
              f"(max_vx={env_cfg['max_vx']:.1f} m/s, "
              f"max_vy={env_cfg['max_vy']:.1f} m/s, "
              f"yaw={env_cfg['max_yaw_rate_deg']:.0f} deg/s)")

    frame_stack = cfg.get("frame_stack", 4)
    dt = cfg.get("env", {}).get("dt", 0.1)
    target_alt = cfg.get("env", {}).get("target_alt", 3.0)

    # Safety monitor
    safety = None
    if not args.no_safety and os.path.exists(args.safety_config):
        with open(args.safety_config, "r") as f:
            safety_cfg = yaml.safe_load(f)
        safety = SafetyMonitor.from_cfg(safety_cfg)
        print(f"[deploy] Safety monitor ENABLED from {args.safety_config}")
    else:
        print("[deploy] Safety monitor DISABLED")

    # Wrapper parity: same pipeline as training
    vec_env = DummyVecEnv([make_env(cfg)])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="last")

    model = PPO.load(args.model, env=vec_env)

    # Access underlying AirSim client
    raw_env = vec_env.envs[0].env  # Monitor wraps AirSimDroneEnv
    client = raw_env.client

    max_steps = int(args.duration_s / dt)
    print(f"[deploy] Model: {args.model}")
    print(f"[deploy] Duration: {args.duration_s}s ({max_steps} steps)")
    print("[deploy] Starting deployment...")

    obs = vec_env.reset()
    total_reward = 0.0
    step = 0
    safety_interventions = 0
    log_interval = 10

    try:
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)

            # Safety filter on the raw action before env.step processes it
            if safety is not None:
                min_depth = _get_center_roi_min_depth(raw_env)

                # Get current altitude from AirSim
                state = client.getMultirotorState()
                current_alt = -state.kinematics_estimated.position.z_val

                # Scale action from [-1,1] to physical units for safety check
                raw_vx = float(action[0][0]) * raw_env.max_vx
                raw_vy = float(action[0][1]) * raw_env.max_vy
                raw_yr = float(action[0][2]) * math.degrees(raw_env.max_yaw_rate)

                safe_vx, safe_vy, safe_yr, safety_info = safety(
                    raw_vx, raw_vy, raw_yr,
                    min_depth_m=min_depth,
                    current_alt=current_alt,
                    target_alt=target_alt,
                )

                # Track safety interventions
                if safety_info["prox_scale"] < 1.0 or safety_info["e_stop"]:
                    safety_interventions += 1

                # Convert back to [-1,1] for env.step
                action[0][0] = np.clip(safe_vx / raw_env.max_vx, -1, 1)
                action[0][1] = np.clip(safe_vy / raw_env.max_vy, -1, 1)
                action[0][2] = np.clip(safe_yr / math.degrees(raw_env.max_yaw_rate), -1, 1)

            obs, reward, done, infos = vec_env.step(action)
            total_reward += reward[0]
            step += 1

            if step % log_interval == 0:
                info = infos[0]
                safety_str = ""
                if safety is not None:
                    safety_str = f"  prox={safety_info['prox_scale']:.2f}"
                print(
                    f"  [{step:5d}] reward={reward[0]:+.3f}  "
                    f"vx={info.get('vx_body', 0):.2f}{safety_str}"
                )

            if done[0]:
                print(f"  [EPISODE END at step {step}] total_reward={total_reward:.2f}"
                      f"  safety_interventions={safety_interventions}")
                total_reward = 0.0

    except KeyboardInterrupt:
        print("\n[deploy] Interrupted by user.")
    finally:
        print(f"[deploy] Total safety interventions: {safety_interventions}")
        print("[deploy] Landing drone...")
        try:
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception as e:
            print(f"[deploy] Landing error: {e}")
        vec_env.close()
        print("[deploy] Done.")


if __name__ == "__main__":
    main()
