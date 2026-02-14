"""Cross-environment evaluation and experiment comparison tools.

Loads JSON summaries from multiple experiments and produces
comparison tables and metrics like cross-env transfer ratio.

Usage (as library):
    from src.evaluation.compare import load_results, compare_experiments

Usage (as script):
    python -m src.evaluation.compare logs/ppo/reward_default/summary.json logs/ppo/reward_aggressive/summary.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load_results(json_path: str) -> dict:
    """Load a single experiment's JSON summary.

    Args:
        json_path: Path to summary JSON file.

    Returns:
        Parsed summary dict.

    Raises:
        FileNotFoundError: If json_path does not exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Summary file not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def cross_env_transfer_ratio(
    dbc_train_env: float,
    dbc_test_env: float,
) -> float:
    """Compute transfer ratio: performance on test env / train env.

    A ratio of 1.0 means perfect transfer. >0.5 is acceptable.

    Args:
        dbc_train_env: Distance before collision on training environment.
        dbc_test_env: Distance before collision on test environment.

    Returns:
        Transfer ratio (0 if train_env is non-positive).
    """
    if dbc_train_env <= 0:
        return 0.0
    return dbc_test_env / dbc_train_env


def compare_experiments(results: dict[str, dict]) -> dict:
    """Compare multiple experiments by key metrics.

    Extracts metrics from summaries that may contain either per-episode
    data or aggregated statistics. Produces rankings by key metrics.

    Args:
        results: {experiment_name: summary_dict}

    Returns:
        Comparison dict with structure:
            {
                "experiments": {
                    name: {"avg_dbc_m": float, "collision_rate": float, "avg_speed_ms": float},
                    ...
                },
                "rankings": {
                    "by_dbc": [name, ...],
                    "by_collision_rate": [name, ...]
                }
            }
    """
    comparison = {"experiments": {}, "rankings": {}}

    for name, summary in results.items():
        # Handle both per-episode summaries and aggregate summaries
        if "episode_summaries" in summary:
            episodes = summary["episode_summaries"]
            avg_dbc = np.mean([e.get("distance_before_collision_m", 0) for e in episodes])
            col_rate = sum(1 for e in episodes if e.get("collided", False)) / max(len(episodes), 1)
            avg_speed = np.mean([e.get("average_speed_ms", 0) for e in episodes])
        else:
            # Aggregate summary format
            avg_dbc = summary.get("avg_distance_before_collision_m", summary.get("distance_before_collision_m", 0))
            col_rate = summary.get("collision_rate", 0)
            avg_speed = summary.get("avg_speed_ms", summary.get("average_speed_ms", 0))

        comparison["experiments"][name] = {
            "avg_dbc_m": round(float(avg_dbc), 2),
            "collision_rate": round(float(col_rate), 3),
            "avg_speed_ms": round(float(avg_speed), 3),
        }

    # Rank by DBC (higher is better)
    ranked = sorted(
        comparison["experiments"].items(),
        key=lambda x: x[1]["avg_dbc_m"],
        reverse=True,
    )
    comparison["rankings"]["by_dbc"] = [name for name, _ in ranked]

    # Rank by collision rate (lower is better)
    ranked = sorted(
        comparison["experiments"].items(),
        key=lambda x: x[1]["collision_rate"],
    )
    comparison["rankings"]["by_collision_rate"] = [name for name, _ in ranked]

    return comparison


def print_comparison_table(comparison: dict):
    """Pretty-print comparison results as a formatted table.

    Args:
        comparison: Output of compare_experiments().
    """
    print(f"\n{'Experiment':<30} {'Avg DBC (m)':>12} {'Col. Rate':>10} {'Avg Speed':>10}")
    print("-" * 65)
    for name, metrics in comparison["experiments"].items():
        print(f"{name:<30} {metrics['avg_dbc_m']:>12.1f} {metrics['collision_rate']:>10.1%} {metrics['avg_speed_ms']:>10.2f}")

    if comparison["rankings"]["by_dbc"]:
        print(f"\nBest DBC: {comparison['rankings']['by_dbc'][0]}")
    if comparison["rankings"]["by_collision_rate"]:
        print(f"Best collision rate: {comparison['rankings']['by_collision_rate'][0]}")


def main():
    """CLI entry point for comparing experiments."""
    parser = argparse.ArgumentParser(
        description="Compare experiment results from multiple summary JSON files"
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        help="Paths to summary JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison JSON (optional)",
    )
    args = parser.parse_args()

    # Load all results, deriving names from file paths
    results = {}
    for path in args.summaries:
        # Try to use parent directory name, fall back to filename
        name = os.path.splitext(os.path.basename(os.path.dirname(path)))[0]
        if not name or name == "evaluation":
            name = os.path.splitext(os.path.basename(path))[0]

        try:
            results[name] = load_results(path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[compare] ERROR loading {path}: {e}")
            return 1

    if not results:
        print("[compare] No valid summary files loaded")
        return 1

    # Compare and display
    comparison = compare_experiments(results)
    print_comparison_table(comparison)

    # Optionally save comparison JSON
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved comparison to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
