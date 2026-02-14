"""
Deploy a trained PPO policy on the AirSim Quadrotor.

Usage:
    python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip
    python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip --duration_s 30
"""

import argparse
import time

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environments.airsim_env import AirSimDroneEnv


def make_env(cfg: dict):
    def _init():
        return Monitor(AirSimDroneEnv(cfg))
    return _init


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
        "--duration_s", type=float, default=60.0,
        help="Max deployment duration in seconds",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    frame_stack = cfg.get("frame_stack", 4)
    dt = cfg.get("env", {}).get("dt", 0.1)

    # Wrapper parity: same pipeline as training
    vec_env = DummyVecEnv([make_env(cfg)])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="last")

    model = PPO.load(args.model, env=vec_env)

    # Access underlying AirSim client for emergency landing
    raw_env = vec_env.envs[0].env  # Monitor wraps AirSimDroneEnv
    client = raw_env.client

    max_steps = int(args.duration_s / dt)
    print(f"[deploy] Model: {args.model}")
    print(f"[deploy] Duration: {args.duration_s}s ({max_steps} steps)")
    print(f"[deploy] Starting deployment...")

    obs = vec_env.reset()
    total_reward = 0.0
    step = 0
    log_interval = 10

    try:
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = vec_env.step(action)

            total_reward += reward[0]
            step += 1

            if step % log_interval == 0:
                info = infos[0]
                print(
                    f"  [{step:5d}] reward={reward[0]:+.3f}  "
                    f"prog={info.get('r_progress', 0):+.3f}  "
                    f"col={info.get('r_collision', 0):+.1f}  "
                    f"smooth={info.get('r_smoothness', 0):+.3f}  "
                    f"vx={info.get('vx_body', 0):.2f}"
                )

            # VecEnv auto-resets on done; log episode boundary
            if done[0]:
                print(f"  [EPISODE END at step {step}] total_reward={total_reward:.2f}")
                total_reward = 0.0

    except KeyboardInterrupt:
        print("\n[deploy] Interrupted by user.")
    finally:
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
