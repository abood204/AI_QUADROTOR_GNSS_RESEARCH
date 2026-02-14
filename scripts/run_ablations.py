"""Run ablation experiments to isolate RL design choices.

Ablation suite:
  1. Reward components: Full (progress+collision+smoothness) vs no-smoothness vs progress-only
  2. Frame stack depth: 4-frame (temporal) vs 1-frame (Markovian)
  3. Domain randomization: With DR vs no-DR baseline
  4. Safety monitor: Evaluated at deployment time (--no_safety flag in deploy)

Each ablation trains a separate PPO agent and logs to TensorBoard.

Usage:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --timesteps 200000
    python scripts/run_ablations.py --timesteps 1000000 --skip abl3_frame_stack_1
"""
import argparse
import subprocess
import sys
from pathlib import Path


# Define ablation experiments
ABLATIONS = [
    # Ablation 1: Reward components
    {
        "name": "abl1_full_reward",
        "description": "Baseline: full reward (progress + collision + smoothness)",
        "base_config": "configs/train_ppo.yaml",
        "reward_config": "configs/rewards/default.yaml",
    },
    {
        "name": "abl1_no_smoothness",
        "description": "Ablation: disabled smoothness penalty (w_smoothness=0)",
        "base_config": "configs/train_ppo.yaml",
        "reward_config": "configs/ablations/no_smoothness.yaml",
    },
    {
        "name": "abl1_progress_only",
        "description": "Ablation: only forward progress reward (minimal signal)",
        "base_config": "configs/train_ppo.yaml",
        "reward_config": "configs/ablations/progress_only.yaml",
    },
    # Ablation 2: Frame stack depth
    {
        "name": "abl2_frames_4",
        "description": "Baseline: 4-frame stack (temporal history)",
        "base_config": "configs/train_ppo.yaml",
        "reward_config": None,
    },
    {
        "name": "abl2_frames_1",
        "description": "Ablation: 1-frame stack (Markovian observation)",
        "base_config": "configs/train_ppo.yaml",
        "reward_config": None,
        "overrides": {"frame_stack": 1},
    },
    # Ablation 3: Domain randomization
    {
        "name": "abl3_with_dr",
        "description": "Baseline: domain randomization enabled",
        "base_config": "configs/train_ppo.yaml",
        "reward_config": None,
    },
    {
        "name": "abl3_no_dr",
        "description": "Ablation: domain randomization disabled (nominal sim only)",
        "base_config": "configs/train_ppo.yaml",
        "reward_config": None,
        "overrides": {
            "domain_randomization": {
                "enabled": False,
                "depth_noise_std": 0.0,
                "spawn_radius_m": 0.0,
            }
        },
    },
]


def build_command(exp: dict, total_timesteps: int) -> list:
    """Build training command for ablation experiment."""
    cmd = [
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        exp["base_config"],
        "--run_name",
        exp["name"],
        "--total_timesteps",
        str(total_timesteps),
    ]

    # Add reward config if specified
    if exp.get("reward_config"):
        cmd.extend(["--reward_config", exp["reward_config"]])

    # Add CLI overrides for dynamic parameters (frame_stack, DR settings)
    # Note: This assumes the training script supports --override or similar
    # For now, we document this as a future enhancement
    if exp.get("overrides"):
        # Placeholder: future enhancement for CLI parameter overrides
        import json
        overrides_json = json.dumps(exp["overrides"])
        cmd.extend(["--overrides", overrides_json])

    return cmd


def run_experiment(exp: dict, total_timesteps: int, dry_run: bool = False) -> int:
    """Run a single ablation experiment. Returns exit code."""
    cmd = build_command(exp, total_timesteps)

    print(f"\n{'='*70}")
    print(f"  {exp['name']}")
    print(f"{'='*70}")
    print(f"  Description: {exp['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    if dry_run:
        print("  [DRY RUN] Command not executed.\n")
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run RL ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ablations.py
  python scripts/run_ablations.py --timesteps 200000
  python scripts/run_ablations.py --skip abl1_no_smoothness abl1_progress_only
  python scripts/run_ablations.py --dry-run
        """,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Training steps per experiment (default: 500_000)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Skip these ablations by name (e.g., --skip abl1_no_smoothness abl2_frames_1)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Run only these ablations by name (overrides --skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Filter ablations
    experiments = ABLATIONS

    if args.only:
        experiments = [exp for exp in ABLATIONS if exp["name"] in args.only]
        if not experiments:
            print(f"[ERROR] No experiments matched --only {args.only}")
            sys.exit(1)
    else:
        experiments = [exp for exp in ABLATIONS if exp["name"] not in args.skip]

    print(f"\n[ablations] Starting {len(experiments)} experiment(s) with {args.timesteps:,} timesteps each")
    if args.skip:
        print(f"[ablations] Skipping: {', '.join(args.skip)}")
    if args.dry_run:
        print("[ablations] DRY RUN MODE — no experiments will be executed\n")

    failed = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[ablations] [{i}/{len(experiments)}]")
        returncode = run_experiment(exp, args.timesteps, dry_run=args.dry_run)

        if returncode != 0:
            failed.append((exp["name"], returncode))
            print(f"[WARN] {exp['name']} exited with code {returncode}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  ABLATION SUITE COMPLETE")
    print(f"{'='*70}")
    if failed:
        print(f"\n  Failed experiments ({len(failed)}):")
        for name, code in failed:
            print(f"    - {name} (exit code {code})")
    else:
        print(f"\n  All {len(experiments)} experiment(s) completed successfully!")

    print(f"\n  View results in TensorBoard:")
    print(f"    tensorboard --logdir logs/ppo/")
    print(f"\n  Evaluate models:")
    print(f"    python -m src.evaluation.evaluate --model logs/ppo/abl1_full_reward/best_model/best_model.zip")
    print(f"\n  Test safety ablation at deployment:")
    print(f"    python -m src.deployment.deploy --model <model> --no_safety")
    print(f"{'='*70}\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
