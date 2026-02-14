"""Pluggable reward functions for the AirSim drone environment."""
from __future__ import annotations

import numpy as np


class RewardFunction:
    """Computes per-step reward from environment state.

    Loaded from YAML config under the `reward` key. All weights default
    to values proven in initial training runs.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.w_progress = cfg.get("w_progress", 0.5)
        self.w_collision = cfg.get("w_collision", -100.0)
        self.w_smoothness = cfg.get("w_smoothness", -0.1)

    def __call__(
        self,
        vx_body: float,
        has_collided: bool,
        action: np.ndarray,
        prev_action: np.ndarray,
    ) -> tuple[float, dict]:
        """Return (total_reward, info_dict) for a single step."""
        r_progress = self.w_progress * vx_body
        r_collision = self.w_collision if has_collided else 0.0
        r_smoothness = self.w_smoothness * float(np.linalg.norm(action - prev_action))

        total = r_progress + r_collision + r_smoothness

        info = {
            "r_progress": r_progress,
            "r_collision": r_collision,
            "r_smoothness": r_smoothness,
        }
        return total, info
