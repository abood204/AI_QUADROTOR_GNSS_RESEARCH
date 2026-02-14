"""Hard safety envelope around policy outputs.

The SafetyMonitor sits between the policy and the low-level controller,
clamping or overriding commands that violate physical safety bounds.
All parameters are loaded from YAML config.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyLimits:
    """Physical safety bounds loaded from config."""
    max_vx: float = 3.0
    max_vy: float = 1.0
    max_yaw_rate_deg: float = 45.0
    proximity_threshold_m: float = 1.5
    proximity_scale_min: float = 0.2
    altitude_tolerance_m: float = 1.0
    comms_timeout_ms: float = 500.0

    @classmethod
    def from_cfg(cls, cfg: dict | None = None) -> SafetyLimits:
        cfg = cfg or {}
        return cls(**{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})


class SafetyMonitor:
    """Enforces hard safety constraints on velocity commands.

    Components:
        1. Velocity clamp: enforce absolute speed limits
        2. Proximity scaling: reduce forward speed near obstacles
        3. Altitude guard: flag when altitude deviates too far
        4. Emergency stop: signal when collision/comms failure detected
    """

    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()
        self._e_stop = False

    @classmethod
    def from_cfg(cls, cfg: dict | None = None) -> SafetyMonitor:
        return cls(SafetyLimits.from_cfg(cfg))

    def clamp_velocity(self, vx: float, vy: float, yaw_rate_deg: float) -> tuple[float, float, float]:
        """Clamp raw velocity commands to physical limits."""
        vx = np.clip(vx, -self.limits.max_vx, self.limits.max_vx)
        vy = np.clip(vy, -self.limits.max_vy, self.limits.max_vy)
        yaw_rate_deg = np.clip(
            yaw_rate_deg,
            -self.limits.max_yaw_rate_deg,
            self.limits.max_yaw_rate_deg,
        )
        return float(vx), float(vy), float(yaw_rate_deg)

    def proximity_scale(self, min_depth_m: float) -> float:
        """Return a [proximity_scale_min, 1.0] scaling factor for forward velocity.

        If the nearest obstacle in the center ROI is closer than the
        threshold, linearly scale down vx to prevent collision.
        """
        if min_depth_m >= self.limits.proximity_threshold_m:
            return 1.0
        if min_depth_m <= 0.0:
            return self.limits.proximity_scale_min
        # Linear interpolation: 0 -> scale_min, threshold -> 1.0
        t = min_depth_m / self.limits.proximity_threshold_m
        return self.limits.proximity_scale_min + t * (1.0 - self.limits.proximity_scale_min)

    def check_altitude(self, current_alt: float, target_alt: float) -> bool:
        """Return True if altitude is within tolerance, False if guard triggered."""
        return abs(current_alt - target_alt) <= self.limits.altitude_tolerance_m

    def trigger_estop(self):
        """Activate emergency stop (collision or comms timeout)."""
        self._e_stop = True

    def clear_estop(self):
        """Clear emergency stop flag (e.g. after successful recovery)."""
        self._e_stop = False

    @property
    def is_estopped(self) -> bool:
        return self._e_stop

    def __call__(
        self,
        vx: float,
        vy: float,
        yaw_rate_deg: float,
        min_depth_m: float | None = None,
        current_alt: float | None = None,
        target_alt: float | None = None,
    ) -> tuple[float, float, float, dict]:
        """Full safety pipeline: clamp -> proximity -> altitude check.

        Returns:
            (safe_vx, safe_vy, safe_yaw_rate_deg, info_dict)
        """
        info: dict = {"e_stop": self._e_stop, "altitude_ok": True, "prox_scale": 1.0}

        if self._e_stop:
            return 0.0, 0.0, 0.0, info

        # 1. Velocity clamp
        vx, vy, yaw_rate_deg = self.clamp_velocity(vx, vy, yaw_rate_deg)

        # 2. Proximity scaling
        if min_depth_m is not None:
            scale = self.proximity_scale(min_depth_m)
            info["prox_scale"] = scale
            vx *= scale

        # 3. Altitude guard
        if current_alt is not None and target_alt is not None:
            alt_ok = self.check_altitude(current_alt, target_alt)
            info["altitude_ok"] = alt_ok

        return vx, vy, yaw_rate_deg, info
