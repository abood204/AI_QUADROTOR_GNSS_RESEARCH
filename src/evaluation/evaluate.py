"""
Full autonomous navigation test with trajectory logging.

Runs a trained PPO policy deterministically, recording position,
velocity, and reward at each step. Outputs:
  - trajectory.csv: per-step telemetry
  - summary.json: aggregate metrics
  - trajectory.png: 2D bird's-eye plot

Usage:
    python -m src.evaluation.evaluate --model logs/ppo/best_model/best_model.zip
    python -m src.evaluation.evaluate --model logs/ppo/best_model/best_model.zip --max_time_s 120
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime

import numpy as np
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
    parser = argparse.ArgumentParser(description="Full navigation test")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model .zip",
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_ppo.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--max_time_s", type=float, default=300.0,
        help="Max test duration in seconds",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: alongside model)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    frame_stack = cfg.get("frame_stack", 4)
    dt = cfg.get("env", {}).get("dt", 0.1)

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("logs", "nav_test", f"test_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Environment (same wrapping as training)
    vec_env = DummyVecEnv([make_env(cfg)])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="last")

    # Load trained model
    model = PPO.load(args.model, env=vec_env)

    # Access underlying AirSim client
    raw_env = vec_env.envs[0].env  # Monitor wraps AirSimDroneEnv
    client = raw_env.client

    max_steps = int(args.max_time_s / dt)
    print(f"[nav_test] Model: {args.model}")
    print(f"[nav_test] Max time: {args.max_time_s}s ({max_steps} steps)")
    print(f"[nav_test] Output: {out_dir}")

    # Trajectory data
    trajectory = []
    total_distance = 0.0
    num_collisions = 0
    num_episodes = 0
    yaw_changes = 0
    prev_yaw = None
    prev_pos = None
    YAW_TURN_THRESHOLD = math.radians(15)  # 15 deg yaw delta = a "turn"

    obs = vec_env.reset()
    num_episodes = 1
    step = 0

    try:
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = vec_env.step(action)
            step += 1

            info = infos[0]

            # Get world-frame position from AirSim
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            orient = state.kinematics_estimated.orientation
            yaw = float(np.degrees(
                np.arctan2(
                    2.0 * (orient.w_val * orient.z_val + orient.x_val * orient.y_val),
                    1.0 - 2.0 * (orient.y_val**2 + orient.z_val**2),
                )
            ))

            x, y, z = pos.x_val, pos.y_val, pos.z_val
            vx = info.get("vx_body", 0.0)

            # Distance tracking
            if prev_pos is not None:
                seg = math.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
                total_distance += seg
            prev_pos = (x, y, z)

            # Turn counting
            if prev_yaw is not None:
                yaw_delta = abs(yaw - prev_yaw)
                if yaw_delta > 180:
                    yaw_delta = 360 - yaw_delta
                if yaw_delta > math.degrees(YAW_TURN_THRESHOLD):
                    yaw_changes += 1
            prev_yaw = yaw

            # Collision tracking
            if info.get("r_collision", 0.0) < 0:
                num_collisions += 1

            trajectory.append({
                "step": step,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "yaw": round(yaw, 2),
                "vx": round(vx, 3),
                "reward": round(float(reward[0]), 4),
            })

            if step % 50 == 0:
                print(f"  [{step:5d}] pos=({x:.1f}, {y:.1f}, {z:.1f}) "
                      f"yaw={yaw:.1f}° vx={vx:.2f} dist={total_distance:.1f}m")

            if done[0]:
                num_episodes += 1
                print(f"  [EPISODE {num_episodes}] dist={total_distance:.1f}m "
                      f"collisions={num_collisions}")
                # VecEnv auto-resets; continue

    except KeyboardInterrupt:
        print("\n[nav_test] Interrupted by user.")
    finally:
        # Safe landing
        print("[nav_test] Landing drone...")
        try:
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception as e:
            print(f"[nav_test] Landing error: {e}")
        vec_env.close()

    if not trajectory:
        print("[nav_test] No data recorded.")
        return

    # --- Save trajectory.csv ---
    csv_path = os.path.join(out_dir, "trajectory.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
        writer.writeheader()
        writer.writerows(trajectory)
    print(f"[nav_test] Saved {csv_path} ({len(trajectory)} rows)")

    # --- Save summary.json ---
    duration_s = len(trajectory) * dt
    avg_speed = total_distance / duration_s if duration_s > 0 else 0.0

    summary = {
        "model": args.model,
        "total_distance_m": round(total_distance, 2),
        "num_turns": yaw_changes,
        "num_collisions": num_collisions,
        "num_episodes": num_episodes,
        "avg_speed_ms": round(avg_speed, 3),
        "duration_s": round(duration_s, 2),
        "total_steps": len(trajectory),
    }

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[nav_test] Saved {json_path}")

    # --- Save trajectory.png ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [row["x"] for row in trajectory]
        ys = [row["y"] for row in trajectory]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(ys, xs, linewidth=0.5, color="blue", alpha=0.7)
        ax.scatter([ys[0]], [xs[0]], c="green", s=80, zorder=5, label="Start")
        ax.scatter([ys[-1]], [xs[-1]], c="red", s=80, zorder=5, label="End")

        # Mark collisions
        col_steps = [row for row in trajectory if row["reward"] < -50]
        if col_steps:
            cx = [r["x"] for r in col_steps]
            cy = [r["y"] for r in col_steps]
            ax.scatter(cy, cx, c="orange", s=40, marker="x", zorder=6,
                       label=f"Collisions ({len(col_steps)})")

        ax.set_xlabel("Y (East)")
        ax.set_ylabel("X (North)")
        ax.set_title(f"Navigation Trajectory — {total_distance:.0f}m, "
                     f"{num_collisions} collisions, {yaw_changes} turns")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        png_path = os.path.join(out_dir, "trajectory.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[nav_test] Saved {png_path}")
    except ImportError:
        print("[nav_test] matplotlib not available — skipping trajectory.png")

    print(f"\n[nav_test] Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
