"""Tests for goal observation computation and goal navigation metrics.

All tests are pure math — no AirSim required.
Replicates the _get_goal_obs and _check_goal_reached logic in isolation.
"""
import math

import numpy as np
import pytest

from src.evaluation.metrics import compute_episode_summary, goal_completion_rate


# ---------------------------------------------------------------------------
# Pure-math helper replicating AirSimDroneEnv._get_goal_obs
# ---------------------------------------------------------------------------

def compute_goal_obs(px, py, yaw, gx, gy, max_goal_dist_m=30.0):
    """Replicate _get_goal_obs without AirSim.

    Returns [cos_theta, sin_theta, dist_norm].
    """
    dx_w = gx - px
    dy_w = gy - py
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx_b = c * dx_w - s * dy_w
    dy_b = s * dx_w + c * dy_w
    bearing = math.atan2(dy_b, dx_b)
    dist = math.hypot(dx_w, dy_w)
    dist_norm = float(np.clip(dist / max_goal_dist_m, 0.0, 1.0))
    return np.array([math.cos(bearing), math.sin(bearing), dist_norm])


class TestGoalObsGeometry:

    def test_goal_directly_ahead_yaw0(self):
        """Goal directly in front (yaw=0): cos=1, sin≈0."""
        obs = compute_goal_obs(px=0, py=0, yaw=0, gx=10, gy=0)
        assert obs[0] == pytest.approx(1.0, abs=1e-6)
        assert obs[1] == pytest.approx(0.0, abs=1e-6)

    def test_goal_directly_behind_yaw0(self):
        """Goal directly behind (yaw=0): cos=-1, sin≈0."""
        obs = compute_goal_obs(px=0, py=0, yaw=0, gx=-10, gy=0)
        assert obs[0] == pytest.approx(-1.0, abs=1e-6)
        assert obs[1] == pytest.approx(0.0, abs=1e-6)

    def test_goal_to_the_left_yaw0(self):
        """Goal to the left in AirSim NED (y-axis left): bearing=-pi/2."""
        # In AirSim NED: positive y is to the right, negative y to the left
        # Body frame: bearing = atan2(dy_b, dx_b); goal left => negative y
        obs = compute_goal_obs(px=0, py=0, yaw=0, gx=0, gy=-10)
        assert obs[0] == pytest.approx(0.0, abs=1e-6)
        assert obs[1] == pytest.approx(-1.0, abs=1e-6)

    def test_goal_to_the_right_yaw0(self):
        """Goal to the right (positive y): bearing=+pi/2."""
        obs = compute_goal_obs(px=0, py=0, yaw=0, gx=0, gy=10)
        assert obs[0] == pytest.approx(0.0, abs=1e-6)
        assert obs[1] == pytest.approx(1.0, abs=1e-6)

    def test_yaw_rotation_transforms_correctly(self):
        """Goal ahead in world frame but drone yawed 90° left => goal appears to the right."""
        # Drone at origin, yaw=+pi/2 (facing +y). Goal is at (10, 0) in world.
        # In body frame: goal should be to the right (negative y-bearing).
        obs = compute_goal_obs(px=0, py=0, yaw=math.pi / 2, gx=10, gy=0)
        assert obs[0] == pytest.approx(0.0, abs=1e-5)
        # sin < 0 means to the right in our convention
        assert obs[1] < 0

    def test_dist_norm_clipped_to_one(self):
        """Distance beyond max_goal_dist_m should clip to 1.0."""
        obs = compute_goal_obs(px=0, py=0, yaw=0, gx=100, gy=0, max_goal_dist_m=30.0)
        assert obs[2] == pytest.approx(1.0)

    def test_dist_norm_at_half_max(self):
        obs = compute_goal_obs(px=0, py=0, yaw=0, gx=15, gy=0, max_goal_dist_m=30.0)
        assert obs[2] == pytest.approx(0.5)

    def test_dist_norm_zero_at_origin(self):
        """Drone at goal: dist_norm=0."""
        obs = compute_goal_obs(px=5, py=5, yaw=0, gx=5, gy=5)
        assert obs[2] == pytest.approx(0.0)

    def test_cos_sin_continuous_near_pi(self):
        """cos/sin encoding is continuous through ±π: no discontinuity."""
        obs_plus = compute_goal_obs(px=0, py=0, yaw=0, gx=-10, gy=0.01)
        obs_minus = compute_goal_obs(px=0, py=0, yaw=0, gx=-10, gy=-0.01)
        # cos values should be very close (both ≈ -1)
        assert abs(obs_plus[0] - obs_minus[0]) < 0.01
        # sin values are close to 0 but opposite sign — that's correct and continuous


class TestGoalCompletionRate:

    def test_all_goals_reached(self):
        episodes = [
            {"goals_reached_count": 3, "total_goals_count": 3},
            {"goals_reached_count": 3, "total_goals_count": 3},
        ]
        assert goal_completion_rate(episodes) == pytest.approx(1.0)

    def test_no_goals_reached(self):
        episodes = [
            {"goals_reached_count": 0, "total_goals_count": 3},
        ]
        assert goal_completion_rate(episodes) == pytest.approx(0.0)

    def test_partial_completion(self):
        episodes = [
            {"goals_reached_count": 1, "total_goals_count": 3},
            {"goals_reached_count": 2, "total_goals_count": 3},
        ]
        # (1+2)/(3+3) = 3/6 = 0.5
        assert goal_completion_rate(episodes) == pytest.approx(0.5)

    def test_empty_episodes_list(self):
        assert goal_completion_rate([]) == pytest.approx(0.0)

    def test_episodes_without_waypoint_keys(self):
        """Backward-compatible: episodes without goal keys contribute nothing."""
        episodes = [{"collided": False}, {"collided": True}]
        assert goal_completion_rate(episodes) == pytest.approx(0.0)


class TestComputeEpisodeSummaryBackwardCompat:

    def test_baseline_call_unchanged(self):
        """Existing callers without waypoint args should get unchanged output."""
        traj = [{"x": 0, "y": i, "reward": 0.1} for i in range(10)]
        summary = compute_episode_summary(traj, dt=0.1, collided=False)
        assert "collided" in summary
        assert "distance_before_collision_m" in summary
        assert "goals_reached_count" not in summary
        assert "total_goals_count" not in summary

    def test_waypoint_fields_present_when_goals_provided(self):
        traj = [{"x": 0, "y": i, "reward": 0.1} for i in range(10)]
        summary = compute_episode_summary(
            traj, dt=0.1, collided=False,
            goals_reached_count=2, total_goals_count=3, mission_success_flag=False
        )
        assert summary["goals_reached_count"] == 2
        assert summary["total_goals_count"] == 3
        assert summary["mission_success"] is False

    def test_mission_success_recorded(self):
        traj = [{"x": 0, "y": i, "reward": 0.1} for i in range(5)]
        summary = compute_episode_summary(
            traj, dt=0.1, collided=False,
            goals_reached_count=3, total_goals_count=3, mission_success_flag=True
        )
        assert summary["mission_success"] is True
