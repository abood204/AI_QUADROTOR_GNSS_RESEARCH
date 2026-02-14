"""Plotting utilities for training visualization and evaluation.

All functions save to PNG files. Uses matplotlib with Agg backend
for headless environments.

Usage:
    from src.evaluation.plots import plot_trajectory, plot_ablation_comparison
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    csv_path: str,
    output_path: str | None = None,
    title: str | None = None,
):
    """Plot 2D bird's-eye trajectory from CSV telemetry.

    Expects CSV with columns: x, y, z, reward (at minimum).
    Marks start point (green), end point (red), and collisions (orange X).

    Args:
        csv_path: Path to telemetry CSV file.
        output_path: Output PNG path. Defaults to csv_path with .png extension.
        title: Plot title. Defaults to trajectory info.

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}")

    rows = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})
    except (ValueError, KeyError) as e:
        raise ValueError(f"CSV parsing error in {csv_path}: {e}")

    if not rows:
        print(f"[plots] Empty CSV: {csv_path}")
        return

    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(ys, xs, linewidth=0.8, color="blue", alpha=0.7)
    ax.scatter([ys[0]], [xs[0]], c="green", s=100, zorder=5, label="Start")
    ax.scatter([ys[-1]], [xs[-1]], c="red", s=100, zorder=5, label="End")

    # Mark collisions (reward < -50)
    collisions = [r for r in rows if r.get("reward", 0) < -50]
    if collisions:
        cx = [r["x"] for r in collisions]
        cy = [r["y"] for r in collisions]
        ax.scatter(cy, cx, c="orange", s=60, marker="x", zorder=6,
                   label=f"Collisions ({len(collisions)})")

    ax.set_xlabel("Y (East, m)")
    ax.set_ylabel("X (North, m)")
    ax.set_title(title or f"Navigation Trajectory ({len(rows)} steps)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if output_path is None:
        output_path = csv_path.replace(".csv", ".png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {output_path}")


def plot_ablation_comparison(
    comparison: dict,
    output_path: str = "ablation_comparison.png",
    title: str = "Ablation Comparison",
):
    """Plot grouped bar chart comparing experiments by key metrics.

    Creates a 1x3 subplot showing Distance Before Collision (DBC),
    collision rate, and average speed for each experiment.

    Args:
        comparison: Output of compare_experiments() with "experiments" key.
        output_path: Output PNG path.
        title: Figure title.

    Raises:
        ValueError: If comparison dict is malformed or empty.
    """
    if "experiments" not in comparison or not comparison["experiments"]:
        raise ValueError("Comparison dict must have non-empty 'experiments' key")

    experiments = comparison["experiments"]
    names = list(experiments.keys())

    # Shorten names for display (remove common prefixes)
    short_names = [
        n.replace("abl1_", "").replace("abl2_", "").replace("abl4_", "")
        for n in names
    ]

    dbc = [experiments[n]["avg_dbc_m"] for n in names]
    col_rate = [experiments[n]["collision_rate"] * 100 for n in names]  # percentage
    speed = [experiments[n]["avg_speed_ms"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(names))

    axes[0].bar(x, dbc, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=45, ha="right")
    axes[0].set_ylabel("Distance Before Collision (m)")
    axes[0].set_title("DBC (higher = better)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, col_rate, color="indianred")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=45, ha="right")
    axes[1].set_ylabel("Collision Rate (%)")
    axes[1].set_title("Collision Rate (lower = better)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, speed, color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=45, ha="right")
    axes[2].set_ylabel("Avg Speed (m/s)")
    axes[2].set_title("Average Speed")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {output_path}")


def plot_metric_history(
    metrics_list: list[dict],
    metric_key: str,
    output_path: str = "metric_history.png",
    title: str | None = None,
    xlabel: str = "Episode",
    ylabel: str | None = None,
):
    """Plot metric history from a list of episode dicts.

    Args:
        metrics_list: List of dicts, each with metric_key.
        metric_key: Key to extract from each dict (e.g., 'distance_before_collision_m').
        output_path: Output PNG path.
        title: Plot title. Defaults to metric_key.
        xlabel: X-axis label.
        ylabel: Y-axis label. Defaults to metric_key.

    Raises:
        ValueError: If metric_key not found in any metrics.
    """
    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    values = []
    for i, m in enumerate(metrics_list):
        if metric_key not in m:
            raise ValueError(f"Metric key '{metric_key}' not found in entry {i}")
        values.append(m[metric_key])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(values, linewidth=1.5, color="steelblue", marker="o", markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title or f"History of {metric_key}")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {output_path}")


def plot_reward_components(
    log_dir: str,
    output_path: str | None = None,
):
    """Print guidance for visualizing reward components.

    Note: Direct extraction from TensorBoard event files requires
    tensorboard.backend.event_processing and is environment-dependent.
    Use TensorBoard interactive UI for the best experience.

    Args:
        log_dir: Path to training log directory.
        output_path: Ignored (for API consistency).
    """
    print(f"[plots] For training curves, use:")
    print(f"[plots]   tensorboard --logdir {log_dir}")
    print(f"[plots] Automated extraction from tfevents not implemented")


if __name__ == "__main__":
    # Example usage (requires sample CSV file)
    print("[plots] This is a plotting library. Import functions to use them.")
    print("[plots] Example:")
    print("[plots]   from src.evaluation.plots import plot_trajectory")
    print("[plots]   plot_trajectory('telemetry.csv')")
