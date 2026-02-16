"""Benchmark AirSim environment step speed.

Runs N random steps and reports wall-clock FPS statistics.
Usage: python -m scripts.benchmark_fps --steps 500
"""
import argparse
import time

import numpy as np
import yaml

from src.environments.airsim_env import AirSimDroneEnv


def main():
    parser = argparse.ArgumentParser(description="Benchmark env FPS")
    parser.add_argument("--steps", type=int, default=500, help="Steps to run")
    parser.add_argument(
        "--config", type=str, default="configs/train_ppo.yaml",
        help="Config YAML (for env settings)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env = AirSimDroneEnv(cfg)
    env.reset()

    durations = []
    for i in range(args.steps):
        action = env.action_space.sample()
        t0 = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        durations.append(time.perf_counter() - t0)

        if terminated or truncated:
            env.reset()

        if (i + 1) % 100 == 0:
            fps = 1.0 / np.mean(durations[-100:])
            print(f"  Step {i+1}/{args.steps} — last 100 avg: {fps:.1f} FPS")

    env.close()

    durations = np.array(durations)
    print(f"\n--- Benchmark Results ({args.steps} steps) ---")
    print(f"  Mean: {1.0 / np.mean(durations):.1f} FPS ({np.mean(durations)*1000:.1f} ms/step)")
    print(f"  p50:  {1.0 / np.median(durations):.1f} FPS ({np.median(durations)*1000:.1f} ms/step)")
    print(f"  p95:  {1.0 / np.percentile(durations, 95):.1f} FPS ({np.percentile(durations, 95)*1000:.1f} ms/step)")
    print(f"  Min:  {1.0 / np.max(durations):.1f} FPS ({np.max(durations)*1000:.1f} ms/step)")


if __name__ == "__main__":
    main()
