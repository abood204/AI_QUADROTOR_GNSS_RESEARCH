"""Batch comparison: evaluate all trained models and produce a ranked summary table.

Discovers every best_model.zip under logs/ppo/, runs run_full_eval.py on each
(as a subprocess so AirSim state is clean between models), then aggregates all
eval_summary.json files into a single comparison table.

AirSim must be running before you start. The same AirSim instance handles every
model sequentially — each eval resets the drone on episode boundaries.

Usage:
    # Evaluate all discovered models (20 episodes each)
    python scripts/run_batch_comparison.py

    # Fewer episodes for a quick pass
    python scripts/run_batch_comparison.py --episodes 10

    # Only specific models
    python scripts/run_batch_comparison.py --only waypoint_v1 abl1_full_reward

    # Skip specific models
    python scripts/run_batch_comparison.py --skip reward_aggressive

    # Re-run even if eval results already exist
    python scripts/run_batch_comparison.py --force

    # Compare only (skip eval, just re-read existing results)
    python scripts/run_batch_comparison.py --compare_only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Model → eval config mapping
# ---------------------------------------------------------------------------

# Models that need a non-default config for eval (env params must match training).
# All others fall back to configs/train_ppo.yaml.
MODEL_CONFIGS: dict[str, str] = {
    "waypoint_v1": "configs/train_ppo_waypoint.yaml",
    "abl3_with_dr": "configs/train_ppo_dr.yaml",
}

# Old timestamped run directories to ignore
_SKIP_PATTERNS = ("ppo_2026", "ppo_2025", "smoke")

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(logs_dir: str = "logs/ppo") -> list[tuple[str, str]]:
    """Return [(name, model_zip_path), ...] for every best_model.zip found.

    Derives a clean experiment name from the directory structure:
      logs/ppo/<name>/best_model/best_model.zip  → name
      logs/ppo/ppo/<name>/best_model/best_model.zip → name  (nested ppo/ dir)
    """
    models = []
    for root, _dirs, files in os.walk(logs_dir):
        if "best_model.zip" not in files:
            continue

        model_zip = os.path.join(root, "best_model.zip")

        # Walk up from 'best_model' to get the experiment dir name
        # best_model/ → parent is experiment dir (or an intermediate 'ppo')
        exp_dir = Path(root).parent
        name = exp_dir.name

        # Handle logs/ppo/ppo/<name>/ nesting
        if name == "ppo":
            name = exp_dir.parent.name

        # Skip old timestamped or smoke runs
        if any(name.startswith(pat) for pat in _SKIP_PATTERNS):
            continue

        models.append((name, model_zip))

    return sorted(models, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    name: str,
    model_zip: str,
    episodes: int,
    out_dir: str,
    force: bool,
) -> str | None:
    """Run run_full_eval.py for one model. Returns path to eval_summary.json or None on failure."""
    summary_path = os.path.join(out_dir, "eval_summary.json")

    if not force and os.path.exists(summary_path):
        print(f"  [skip] {name} — results already exist at {summary_path}")
        return summary_path

    config = MODEL_CONFIGS.get(name, "configs/train_ppo.yaml")

    cmd = [
        sys.executable,
        "scripts/run_full_eval.py",
        "--model", model_zip,
        "--config", config,
        "--episodes", str(episodes),
        "--output_dir", out_dir,
    ]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Model:      {model_zip}")
    print(f"  Config:     {config}")
    print(f"  Episodes:   {episodes}")
    print(f"  Output:     {out_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n  [ERROR] Eval failed for {name} (exit code {result.returncode})")
        return None

    if not os.path.exists(summary_path):
        print(f"\n  [ERROR] eval_summary.json not found after eval for {name}")
        return None

    return summary_path


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def load_and_compare(results: dict[str, str]) -> dict:
    """Load eval_summary.json files and build comparison dict."""
    from src.evaluation.compare import compare_experiments

    loaded = {}
    for name, summary_path in results.items():
        try:
            with open(summary_path) as f:
                loaded[name] = json.load(f)
        except Exception as e:
            print(f"  [warn] Could not load {summary_path}: {e}")

    if not loaded:
        return {}

    return compare_experiments(loaded)


def print_table(comparison: dict, waypoint_data: dict[str, dict]):
    """Print a ranked comparison table to stdout."""
    if not comparison.get("experiments"):
        print("[compare] No results to display.")
        return

    # Determine column widths
    names = list(comparison["experiments"].keys())
    col_w = max(len(n) for n in names) + 2

    header = (
        f"\n{'Model':<{col_w}} {'Avg DBC (m)':>12} {'Col. Rate':>10} "
        f"{'Avg Speed':>10} {'Smoothness':>12}"
    )
    print("\n" + "=" * len(header))
    print("  BATCH COMPARISON RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Sort by avg DBC descending
    sorted_experiments = sorted(
        comparison["experiments"].items(),
        key=lambda x: x[1]["avg_dbc_m"],
        reverse=True,
    )

    for rank, (name, metrics) in enumerate(sorted_experiments, 1):
        smoothness = waypoint_data.get(name, {}).get("avg_path_smoothness_jerk", float("nan"))
        gcr = waypoint_data.get(name, {}).get("goal_completion_rate")
        gcr_str = f"  GCR={gcr:.1%}" if gcr is not None else ""
        print(
            f"  {rank}. {name:<{col_w - 4}} {metrics['avg_dbc_m']:>12.1f} "
            f"{metrics['collision_rate']:>10.1%} {metrics['avg_speed_ms']:>10.2f} "
            f"{smoothness:>12.4f}{gcr_str}"
        )

    print("-" * len(header))
    best_dbc = comparison["rankings"]["by_dbc"][0] if comparison["rankings"].get("by_dbc") else "N/A"
    best_col = comparison["rankings"]["by_collision_rate"][0] if comparison["rankings"].get("by_collision_rate") else "N/A"
    print(f"\n  Best DBC:            {best_dbc}")
    print(f"  Best collision rate: {best_col}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all models and produce comparison table")
    parser.add_argument("--logs_dir", default="logs/ppo", help="Root directory to search for models")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per model (default: 20)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Only evaluate these models by name")
    parser.add_argument("--skip", nargs="+", metavar="NAME", default=[], help="Skip these models by name")
    parser.add_argument("--force", action="store_true", help="Re-run eval even if results exist")
    parser.add_argument("--compare_only", action="store_true", help="Skip eval, only aggregate existing results")
    parser.add_argument("--output_dir", default=None, help="Root output dir (default: logs/eval/batch_<timestamp>)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.output_dir or os.path.join("logs", "eval", f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    # Discover models
    all_models = discover_models(args.logs_dir)
    if not all_models:
        print(f"[batch] No models found under {args.logs_dir}")
        return 1

    # Filter
    if args.only:
        all_models = [(n, p) for n, p in all_models if n in args.only]
    if args.skip:
        all_models = [(n, p) for n, p in all_models if n not in args.skip]

    print(f"\n[batch] Found {len(all_models)} models to evaluate:")
    for name, path in all_models:
        config = MODEL_CONFIGS.get(name, "configs/train_ppo.yaml")
        print(f"  - {name:<35} config: {config}")
    print(f"\n[batch] Episodes per model: {args.episodes}")
    print(f"[batch] Output dir:         {batch_dir}\n")

    # Run evaluations
    summary_paths: dict[str, str] = {}
    failed: list[str] = []

    for name, model_zip in all_models:
        model_out = os.path.join(batch_dir, name)
        os.makedirs(model_out, exist_ok=True)

        if args.compare_only:
            # Just look for existing results
            existing = os.path.join(model_out, "eval_summary.json")
            if os.path.exists(existing):
                summary_paths[name] = existing
            else:
                # Also check legacy location
                legacy = os.path.join("logs", "eval", name, "eval_summary.json")
                if os.path.exists(legacy):
                    summary_paths[name] = legacy
                else:
                    print(f"  [skip] {name} — no existing results found")
        else:
            result = run_eval(name, model_zip, args.episodes, model_out, args.force)
            if result:
                summary_paths[name] = result
            else:
                failed.append(name)

    if not summary_paths:
        print("\n[batch] No results to compare. Run without --compare_only first.")
        return 1

    # Build comparison
    print(f"\n[batch] Aggregating {len(summary_paths)} results...")
    comparison = load_and_compare(summary_paths)

    # Extract smoothness + waypoint metrics for display
    extra: dict[str, dict] = {}
    for name, path in summary_paths.items():
        try:
            with open(path) as f:
                data = json.load(f)
            extra[name] = {
                "avg_path_smoothness_jerk": data.get("avg_path_smoothness_jerk", float("nan")),
                "goal_completion_rate": data.get("goal_completion_rate"),
                "mission_success_rate": data.get("mission_success_rate"),
            }
        except Exception:
            pass

    print_table(comparison, extra)

    # Save comparison JSON
    comp_path = os.path.join(batch_dir, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "episodes_per_model": args.episodes,
            "models_evaluated": list(summary_paths.keys()),
            "models_failed": failed,
            "comparison": comparison,
            "extra_metrics": {
                k: {kk: vv for kk, vv in v.items() if vv is not None}
                for k, v in extra.items()
            },
        }, f, indent=2, default=str)

    print(f"[batch] Full results saved to: {batch_dir}")
    print(f"[batch] Comparison JSON:       {comp_path}")

    if failed:
        print(f"\n[batch] WARNING: {len(failed)} model(s) failed eval: {', '.join(failed)}")
        print("        Check AirSim is running and the model configs are correct.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
