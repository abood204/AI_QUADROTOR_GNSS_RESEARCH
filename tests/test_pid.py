"""Tests for PID controller.

Runs without AirSim — tests pure control logic.
"""
import pytest

from src.control.pid import PID, PIDGains


class TestPIDBasic:
    def test_proportional_response(self):
        """P-only controller: output should be kp * error."""
        pid = PID(PIDGains(kp=1.0, ki=0.0, kd=0.0, limit=10.0))
        u = pid.update(error=5.0, dt=0.1)
        assert u == pytest.approx(5.0)

    def test_output_clamp(self):
        """Output should never exceed the limit."""
        pid = PID(PIDGains(kp=10.0, ki=0.0, kd=0.0, limit=3.0))
        u = pid.update(error=5.0, dt=0.1)
        assert u == pytest.approx(3.0)

    def test_negative_output_clamp(self):
        """Negative output should be clamped to -limit."""
        pid = PID(PIDGains(kp=10.0, ki=0.0, kd=0.0, limit=3.0))
        u = pid.update(error=-5.0, dt=0.1)
        assert u == pytest.approx(-3.0)

    def test_zero_error_zero_output(self):
        """Zero error should produce zero output (no integral buildup)."""
        pid = PID(PIDGains(kp=1.0, ki=0.0, kd=0.0, limit=10.0))
        u = pid.update(error=0.0, dt=0.1)
        assert u == pytest.approx(0.0)


class TestPIDIntegral:
    def test_integral_accumulation(self):
        """Integral should accumulate over time."""
        pid = PID(PIDGains(kp=0.0, ki=1.0, kd=0.0, limit=100.0))
        for _ in range(10):
            u = pid.update(error=1.0, dt=0.1)
        # Integral = 10 * 1.0 * 0.1 = 1.0
        assert u == pytest.approx(1.0)

    def test_anti_windup(self):
        """Integral should be clamped to prevent windup."""
        pid = PID(PIDGains(kp=0.0, ki=10.0, kd=0.0, limit=2.0))
        for _ in range(100):
            u = pid.update(error=1.0, dt=0.1)
        # Integral would be 100 * 1.0 * 0.1 = 10.0, but clamped to 2.0
        # ki * clamped_integral = 10.0 * 2.0 = 20.0, then output clamped to 2.0
        assert u == pytest.approx(2.0)


class TestPIDDerivative:
    def test_first_step_no_derivative(self):
        """First step should have zero derivative (protected)."""
        pid = PID(PIDGains(kp=0.0, ki=0.0, kd=1.0, limit=100.0))
        u = pid.update(error=5.0, dt=0.1)
        assert u == pytest.approx(0.0)  # d = 0 on first step

    def test_derivative_response(self):
        """Derivative should respond to error change rate."""
        pid = PID(PIDGains(kp=0.0, ki=0.0, kd=1.0, limit=100.0))
        pid.update(error=0.0, dt=0.1)  # first step
        u = pid.update(error=1.0, dt=0.1)  # d = (1.0 - 0.0) / 0.1 = 10.0
        assert u == pytest.approx(10.0)


class TestPIDReset:
    def test_reset_clears_state(self):
        """After reset, PID should behave as freshly initialized."""
        pid = PID(PIDGains(kp=0.0, ki=1.0, kd=0.0, limit=100.0))
        for _ in range(10):
            pid.update(error=1.0, dt=0.1)
        pid.reset()
        u = pid.update(error=0.0, dt=0.1)
        assert u == pytest.approx(0.0)

    def test_dt_zero_protection(self):
        """dt=0 should not cause division by zero."""
        pid = PID(PIDGains(kp=1.0, ki=0.0, kd=1.0, limit=100.0))
        u = pid.update(error=1.0, dt=0.0)  # Should handle gracefully
        assert abs(u) < 100.0  # Should not be inf or nan
