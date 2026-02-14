"""Tests for domain randomization noise injection.

Verifies that depth noise stays within [0, 1] bounds.
Runs without AirSim.
"""
import numpy as np
import pytest


class TestDepthNoise:
    """Test depth noise injection logic (extracted from airsim_env)."""

    def _apply_noise(self, img: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
        """Replicate the noise injection logic from _get_depth_image."""
        if noise_std > 0:
            noise = rng.normal(0, noise_std, img.shape)
            img = np.clip(img + noise, 0.0, 1.0).astype(np.float32)
        return img

    def test_noise_stays_in_bounds(self):
        """Even with large noise, output should be in [0, 1]."""
        rng = np.random.default_rng(42)
        img = rng.uniform(0, 1, (84, 84)).astype(np.float32)
        noisy = self._apply_noise(img, noise_std=0.1, rng=rng)
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0

    def test_zero_noise_unchanged(self):
        """Zero noise should leave image unchanged."""
        rng = np.random.default_rng(42)
        img = np.full((84, 84), 0.5, dtype=np.float32)
        noisy = self._apply_noise(img, noise_std=0.0, rng=rng)
        np.testing.assert_array_equal(img, noisy)

    def test_noise_changes_image(self):
        """Non-zero noise should actually modify the image."""
        rng = np.random.default_rng(42)
        img = np.full((84, 84), 0.5, dtype=np.float32)
        noisy = self._apply_noise(img, noise_std=0.05, rng=rng)
        assert not np.array_equal(img, noisy)

    def test_noise_mean_near_zero(self):
        """Average noise over many pixels should be near zero."""
        rng = np.random.default_rng(42)
        img = np.full((1000, 1000), 0.5, dtype=np.float32)
        noisy = self._apply_noise(img, noise_std=0.02, rng=rng)
        diff = (noisy - img).mean()
        assert abs(diff) < 0.01  # Mean noise should be near zero
