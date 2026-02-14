"""Tests for SafetyMonitor.

Runs without AirSim — tests pure safety logic.
"""
import pytest

from src.safety.monitor import SafetyMonitor, SafetyLimits


class TestVelocityClamp:
    def setup_method(self):
        self.monitor = SafetyMonitor(SafetyLimits(max_vx=3.0, max_vy=1.0, max_yaw_rate_deg=45.0))

    def test_within_limits_unchanged(self):
        vx, vy, yr = self.monitor.clamp_velocity(2.0, 0.5, 30.0)
        assert vx == pytest.approx(2.0)
        assert vy == pytest.approx(0.5)
        assert yr == pytest.approx(30.0)

    def test_exceeds_max_vx(self):
        vx, vy, yr = self.monitor.clamp_velocity(5.0, 0.0, 0.0)
        assert vx == pytest.approx(3.0)

    def test_exceeds_negative_vy(self):
        vx, vy, yr = self.monitor.clamp_velocity(0.0, -2.0, 0.0)
        assert vy == pytest.approx(-1.0)

    def test_exceeds_yaw_rate(self):
        vx, vy, yr = self.monitor.clamp_velocity(0.0, 0.0, 90.0)
        assert yr == pytest.approx(45.0)


class TestProximityScale:
    def setup_method(self):
        self.monitor = SafetyMonitor(SafetyLimits(
            proximity_threshold_m=1.5,
            proximity_scale_min=0.2,
        ))

    def test_far_away_no_scaling(self):
        assert self.monitor.proximity_scale(5.0) == pytest.approx(1.0)

    def test_at_threshold_full_speed(self):
        assert self.monitor.proximity_scale(1.5) == pytest.approx(1.0)

    def test_at_zero_minimum_scale(self):
        assert self.monitor.proximity_scale(0.0) == pytest.approx(0.2)

    def test_halfway_interpolation(self):
        # At 0.75m (half of 1.5m threshold)
        # t = 0.75 / 1.5 = 0.5
        # scale = 0.2 + 0.5 * (1.0 - 0.2) = 0.2 + 0.4 = 0.6
        assert self.monitor.proximity_scale(0.75) == pytest.approx(0.6)

    def test_negative_depth_returns_minimum(self):
        assert self.monitor.proximity_scale(-1.0) == pytest.approx(0.2)


class TestAltitudeGuard:
    def setup_method(self):
        self.monitor = SafetyMonitor(SafetyLimits(altitude_tolerance_m=1.0))

    def test_within_tolerance(self):
        assert self.monitor.check_altitude(3.0, 3.5) is True

    def test_at_boundary(self):
        assert self.monitor.check_altitude(3.0, 4.0) is True

    def test_outside_tolerance(self):
        assert self.monitor.check_altitude(3.0, 4.5) is False


class TestEmergencyStop:
    def setup_method(self):
        self.monitor = SafetyMonitor()

    def test_estop_not_active_by_default(self):
        assert self.monitor.is_estopped is False

    def test_trigger_estop(self):
        self.monitor.trigger_estop()
        assert self.monitor.is_estopped is True

    def test_estop_zeros_output(self):
        self.monitor.trigger_estop()
        vx, vy, yr, info = self.monitor(3.0, 1.0, 45.0)
        assert vx == 0.0
        assert vy == 0.0
        assert yr == 0.0
        assert info["e_stop"] is True

    def test_clear_estop(self):
        self.monitor.trigger_estop()
        self.monitor.clear_estop()
        assert self.monitor.is_estopped is False


class TestFullPipeline:
    def test_clamp_and_proximity(self):
        monitor = SafetyMonitor(SafetyLimits(
            max_vx=3.0, max_vy=1.0, max_yaw_rate_deg=45.0,
            proximity_threshold_m=1.5, proximity_scale_min=0.2,
        ))
        # Request 5.0 m/s forward with obstacle at 0.75m
        vx, vy, yr, info = monitor(5.0, 0.0, 0.0, min_depth_m=0.75)
        # Clamped to 3.0, then scaled by 0.6 -> 1.8
        assert vx == pytest.approx(1.8)
        assert info["prox_scale"] == pytest.approx(0.6)

    def test_from_cfg(self):
        cfg = {"max_vx": 2.0, "proximity_threshold_m": 3.0}
        monitor = SafetyMonitor.from_cfg(cfg)
        assert monitor.limits.max_vx == 2.0
        assert monitor.limits.proximity_threshold_m == 3.0
