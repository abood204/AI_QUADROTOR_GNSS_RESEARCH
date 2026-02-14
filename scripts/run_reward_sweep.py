"""Run reward weight sweep experiments sequentially.

Trains 3 PPO agents with different reward configurations:
  1. default   (w_progress=0.5, w_collision=-100, w_smoothness=-0.1)
  2. aggressive (w_progress=1.0, w_collision=-100, w_smoothness=-0.05)
  3. cautious  (w_progress=0.3, w_collision=-200, w_smoothness=-0.2)

Usage:
    python -m scripts.run_reward_sweep
    python -m scripts.run_reward_sweep --timesteps 500000
"""

import argparse
import subprocess
import sys


EXPERIMENTS = [
    {"name": "reward_default", "reward_config": "configs/rewards/default.yaml"},
    {"name": "reward_aggressive", "reward_config": "configs/rewards/aggressive.yaml"},
    {"name": "reward_cautious", "reward_config": "configs/rewards/cautious.yaml"},
]


def main():
    parser = argparse.ArgumentParser(description="Reward weight sweep")
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Training timesteps per experiment",
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_ppo.yaml",
        help="Base training config",
    )
    args = parser.parse_args()

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {exp['name']}")
        print(f"  Reward config: {exp['reward_config']}")
        print(f"  Timesteps: {args.timesteps}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "-m", "src.training.train",
            "--config", args.config,
            "--reward_config", exp["reward_config"],
            "--run_name", exp["name"],
            "--total_timesteps", str(args.timesteps),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[WARN] Experiment {exp['name']} exited with code {result.returncode}")

    print("\n[sweep] All experiments complete. Compare in TensorBoard:")
    print("  tensorboard --logdir logs/ppo/")


if __name__ == "__main__":
    main()
