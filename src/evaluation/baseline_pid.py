"""PID-only baseline: hover + constant slow forward velocity.

No RL policy involved — establishes the "no AI" performance floor
for ablation comparison. Uses the same evaluation protocol as
evaluate.py for direct comparison.

Usage:
    python -m src.evaluation.baseline_pid
    python -m src.evaluation.baseline_pid --forward_vx 1.5 --episodes 20
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime

import airsim
import numpy as np

from src.evaluation.metrics import compute_episode_summary


def main():
    parser = argparse.ArgumentParser(description="PID-only baseline evaluation")
    parser.add_argument("--forward_vx", type=float, default=1.0, help="Forward speed (m/s)")
    parser.add_argument("--target_alt", type=float, default=3.0, help="Target altitude (m)")
    parser.add_argument("--dt", type=float, default=0.1, help="Control period (s)")
    parser.add_argument("--max_steps", type=int, default=1024, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("logs", "baseline", f"pid_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    episode_summaries = []

    for ep in range(args.episodes):
        print(f"\n[baseline_pid] Episode {ep + 1}/{args.episodes}")

        # Reset
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToZAsync(-args.target_alt, 1.0).join()

        trajectory = []
        collided = False
        prev_pos = None

        for step in range(args.max_steps):
            # Constant forward velocity, no lateral, no yaw
            client.moveByVelocityZBodyFrameAsync(
                args.forward_vx, 0.0, -args.target_alt, args.dt,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
            )
            client.simContinueForTime(args.dt)
            client.simPause(True)

            # Read state
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            x, y, z = pos.x_val, pos.y_val, pos.z_val

            # Check collision
            if client.simGetCollisionInfo().has_collided:
                collided = True
                trajectory.append({"x": x, "y": y, "z": z, "reward": -100.0})
                break

            trajectory.append({"x": x, "y": y, "z": z, "reward": 0.5})

            if step % 100 == 0:
                print(f"  [{step:5d}] pos=({x:.1f}, {y:.1f}, {z:.1f})")

        # Episode summary
        summary = compute_episode_summary(trajectory, dt=args.dt, collided=collided)
        summary["episode"] = ep + 1
        episode_summaries.append(summary)
        print(f"  DBC={summary['distance_before_collision_m']:.1f}m  "
              f"collided={collided}  steps={len(trajectory)}")

        # Save per-episode trajectory
        csv_path = os.path.join(out_dir, f"trajectory_ep{ep + 1:03d}.csv")
        if trajectory:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                writer.writeheader()
                writer.writerows(trajectory)

    # Aggregate summary
    avg_dbc = np.mean([s["distance_before_collision_m"] for s in episode_summaries])
    collision_rate = sum(1 for s in episode_summaries if s["collided"]) / len(episode_summaries)
    avg_speed = np.mean([s["average_speed_ms"] for s in episode_summaries])

    aggregate = {
        "baseline": "pid_only",
        "forward_vx": args.forward_vx,
        "episodes": args.episodes,
        "avg_distance_before_collision_m": round(float(avg_dbc), 2),
        "collision_rate": round(float(collision_rate), 3),
        "avg_speed_ms": round(float(avg_speed), 3),
        "episode_summaries": episode_summaries,
    }

    json_path = os.path.join(out_dir, "baseline_summary.json")
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n[baseline_pid] Results saved to {out_dir}")
    print(f"  Avg DBC: {avg_dbc:.1f}m")
    print(f"  Collision rate: {collision_rate:.1%}")
    print(f"  Avg speed: {avg_speed:.2f} m/s")

    # Cleanup
    try:
        client.armDisarm(False)
        client.enableApiControl(False)
        client.simPause(False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
