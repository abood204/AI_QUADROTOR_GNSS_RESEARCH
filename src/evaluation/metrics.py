"""Standardized evaluation metrics for navigation performance.

All functions take trajectory data (lists of dicts) and return scalar metrics.
Designed to be used without AirSim — pure computation on recorded telemetry.
"""
from __future__ import annotations

import math

import numpy as np


def distance_before_collision(trajectory: list[dict]) -> float:
    """Total distance traveled (meters) before first collision.

    If no collision occurred, returns total distance for the trajectory.
    """
    total = 0.0
    prev = None
    for row in trajectory:
        x, y = row["x"], row["y"]
        if prev is not None:
            total += math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2)
        prev = (x, y)
        # Collision detected by large negative reward
        if row.get("reward", 0) < -50:
            break
    return total


def collision_rate(episodes: list[dict]) -> float:
    """Fraction of episodes that ended in collision.

    Args:
        episodes: list of episode summary dicts with 'collided' bool key
    """
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep.get("collided", False)) / len(episodes)


def average_speed(trajectory: list[dict], dt: float = 0.1) -> float:
    """Average ground speed (m/s) over the trajectory."""
    total_dist = 0.0
    prev = None
    for row in trajectory:
        x, y = row["x"], row["y"]
        if prev is not None:
            total_dist += math.sqrt((x - prev[0]) ** 2 + (y - prev[1]) ** 2)
        prev = (x, y)
    duration = len(trajectory) * dt
    return total_dist / duration if duration > 0 else 0.0


def path_smoothness(trajectory: list[dict], dt: float = 0.1) -> float:
    """Mean absolute jerk (m/s^3) — lower is smoother.

    Computed from velocity changes across consecutive steps.
    """
    if len(trajectory) < 3:
        return 0.0

    velocities = []
    prev = None
    for row in trajectory:
        x, y = row["x"], row["y"]
        if prev is not None:
            vx = (x - prev[0]) / dt
            vy = (y - prev[1]) / dt
            velocities.append((vx, vy))
        prev = (x, y)

    if len(velocities) < 2:
        return 0.0

    jerks = []
    for i in range(1, len(velocities)):
        ax = (velocities[i][0] - velocities[i - 1][0]) / dt
        ay = (velocities[i][1] - velocities[i - 1][1]) / dt
        jerks.append(math.sqrt(ax ** 2 + ay ** 2))

    return float(np.mean(jerks))


def survival_time(trajectory: list[dict], dt: float = 0.1) -> float:
    """Time (seconds) before episode termination."""
    return len(trajectory) * dt


def goal_completion_rate(episodes: list[dict]) -> float:
    """Fraction of waypoints reached across all episodes.

    Args:
        episodes: list of episode summary dicts with 'goals_reached' and
                  'total_goals' int keys (added by waypoint evaluation).
                  Episodes without these keys contribute 0 numerator/denominator.
    """
    total_reached = sum(ep.get("goals_reached_count", 0) for ep in episodes)
    total_spawned = sum(ep.get("total_goals_count", 0) for ep in episodes)
    if total_spawned == 0:
        return 0.0
    return total_reached / total_spawned


def compute_episode_summary(
    trajectory: list[dict],
    dt: float = 0.1,
    collided: bool = False,
    goals_reached_count: int = 0,
    total_goals_count: int = 0,
    mission_success_flag: bool = False,
) -> dict:
    """Compute all metrics for a single episode.

    Backward-compatible: existing callers without waypoint args are unaffected.
    Returns a dict suitable for JSON serialization.
    """
    summary = {
        "distance_before_collision_m": round(distance_before_collision(trajectory), 2),
        "average_speed_ms": round(average_speed(trajectory, dt), 3),
        "path_smoothness_jerk": round(path_smoothness(trajectory, dt), 3),
        "survival_time_s": round(survival_time(trajectory, dt), 2),
        "collided": collided,
        "total_steps": len(trajectory),
    }
    # Waypoint metrics — only included when goal navigation is active
    if total_goals_count > 0:
        summary["goals_reached_count"] = goals_reached_count
        summary["total_goals_count"] = total_goals_count
        summary["mission_success"] = mission_success_flag
    return summary
