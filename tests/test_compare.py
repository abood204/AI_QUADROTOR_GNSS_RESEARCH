"""Tests for experiment comparison tools."""
import pytest

from src.evaluation.compare import compare_experiments, cross_env_transfer_ratio


class TestTransferRatio:
    def test_perfect_transfer(self):
        assert cross_env_transfer_ratio(50.0, 50.0) == pytest.approx(1.0)

    def test_half_transfer(self):
        assert cross_env_transfer_ratio(100.0, 50.0) == pytest.approx(0.5)

    def test_zero_train_perf(self):
        assert cross_env_transfer_ratio(0.0, 50.0) == pytest.approx(0.0)


class TestCompareExperiments:
    def test_basic_comparison(self):
        results = {
            "exp_a": {
                "episode_summaries": [
                    {"distance_before_collision_m": 80.0, "collided": False, "average_speed_ms": 2.0},
                    {"distance_before_collision_m": 60.0, "collided": True, "average_speed_ms": 1.5},
                ]
            },
            "exp_b": {
                "episode_summaries": [
                    {"distance_before_collision_m": 100.0, "collided": False, "average_speed_ms": 2.5},
                    {"distance_before_collision_m": 90.0, "collided": False, "average_speed_ms": 2.0},
                ]
            },
        }
        comp = compare_experiments(results)
        assert comp["experiments"]["exp_a"]["avg_dbc_m"] == pytest.approx(70.0)
        assert comp["experiments"]["exp_b"]["avg_dbc_m"] == pytest.approx(95.0)
        assert comp["rankings"]["by_dbc"][0] == "exp_b"  # Better DBC
        assert comp["experiments"]["exp_a"]["collision_rate"] == pytest.approx(0.5)
        assert comp["experiments"]["exp_b"]["collision_rate"] == pytest.approx(0.0)

    def test_aggregate_format(self):
        """Compare module should handle flat aggregate summaries too."""
        results = {
            "baseline": {
                "avg_distance_before_collision_m": 30.0,
                "collision_rate": 0.8,
                "avg_speed_ms": 1.0,
            }
        }
        comp = compare_experiments(results)
        assert comp["experiments"]["baseline"]["avg_dbc_m"] == pytest.approx(30.0)
