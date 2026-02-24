"""Tests for WaypointRewardFunction.

Runs without AirSim — pure reward computation logic only.
"""
import numpy as np
import pytest

from src.environments.rewards import WaypointRewardFunction


@pytest.fixture
def reward_fn():
    return WaypointRewardFunction({
        "w_heading": 0.3,
        "w_dist": 0.5,
        "w_goal": 10.0,
        "w_mission": 50.0,
        "w_collision": -100.0,
        "w_smoothness": -0.1,
    })


class TestWaypointRewardFunction:

    def test_default_config(self):
        rf = WaypointRewardFunction()
        assert rf.w_heading == 0.3
        assert rf.w_dist == 0.5
        assert rf.w_goal == 10.0
        assert rf.w_mission == 50.0
        assert rf.w_collision == -100.0
        assert rf.w_smoothness == -0.1

    def test_heading_reward_facing_goal(self, reward_fn):
        """cos_theta=1 (facing goal directly) gives max heading reward."""
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            cos_theta=1.0, dist_norm=0.5,
        )
        assert info["r_heading"] == pytest.approx(0.3 * 1.0)

    def test_heading_reward_facing_away(self, reward_fn):
        """cos_theta=-1 (facing directly away from goal) gives min heading reward."""
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            cos_theta=-1.0, dist_norm=0.5,
        )
        assert info["r_heading"] == pytest.approx(-0.3)

    def test_dist_reward_zero_on_first_step(self, reward_fn):
        """First step after reset (prev_dist_norm=None) should have r_dist=0."""
        assert reward_fn._prev_dist_norm is None
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            dist_norm=0.8,
        )
        assert info["r_dist"] == pytest.approx(0.0)

    def test_dist_reward_positive_when_approaching(self, reward_fn):
        """Decreasing dist_norm should yield positive r_dist."""
        reward_fn._prev_dist_norm = 0.8
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            dist_norm=0.6,
        )
        # delta = 0.8 - 0.6 = 0.2; r_dist = 0.5 * 0.2 = 0.1
        assert info["r_dist"] == pytest.approx(0.1)

    def test_dist_reward_negative_when_receding(self, reward_fn):
        """Increasing dist_norm should yield negative r_dist (penalises drift)."""
        reward_fn._prev_dist_norm = 0.5
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            dist_norm=0.7,
        )
        assert info["r_dist"] == pytest.approx(-0.1)

    def test_prev_dist_norm_resets_after_goal_reached(self, reward_fn):
        """_prev_dist_norm must be None after goal_reached=True to prevent spurious delta."""
        reward_fn._prev_dist_norm = 0.3
        reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            goal_reached=True, dist_norm=0.1,
        )
        assert reward_fn._prev_dist_norm is None

    def test_goal_bonus(self, reward_fn):
        """goal_reached=True should fire the goal bonus."""
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            goal_reached=True,
        )
        assert info["r_goal"] == pytest.approx(10.0)

    def test_no_goal_bonus_when_not_reached(self, reward_fn):
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            goal_reached=False,
        )
        assert info["r_goal"] == pytest.approx(0.0)

    def test_mission_bonus(self, reward_fn):
        """all_goals_done=True should fire the mission bonus."""
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.zeros(3), prev_action=np.zeros(3),
            all_goals_done=True,
        )
        assert info["r_mission"] == pytest.approx(50.0)

    def test_collision_penalty(self, reward_fn):
        _, info = reward_fn(
            vx_body=0.0, has_collided=True,
            action=np.zeros(3), prev_action=np.zeros(3),
        )
        assert info["r_collision"] == pytest.approx(-100.0)

    def test_smoothness_penalty(self, reward_fn):
        _, info = reward_fn(
            vx_body=0.0, has_collided=False,
            action=np.array([1.0, 0.0, 0.0]),
            prev_action=np.array([-1.0, 0.0, 0.0]),
        )
        # ||[2,0,0]|| = 2.0; penalty = -0.1 * 2.0 = -0.2
        assert info["r_smoothness"] == pytest.approx(-0.2)

    def test_total_equals_sum_of_components(self, reward_fn):
        """Total reward must equal sum of all component rewards."""
        reward_fn._prev_dist_norm = 0.6
        total, info = reward_fn(
            vx_body=1.0, has_collided=False,
            action=np.array([0.4, 0.1, 0.2]),
            prev_action=np.array([0.3, 0.0, 0.1]),
            goal_reached=False, dist_norm=0.5, cos_theta=0.7,
        )
        expected = sum(info.values())
        assert total == pytest.approx(expected)
