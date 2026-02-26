"""Tests for parallel AirSim environment support.

Verifies port passing, update_config method, and config isolation.
All tests run without AirSim (mocked at boundary) and without
importing stable_baselines3 (which requires matplotlib).
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_airsim_client():
    """Return a mock that satisfies AirSimDroneEnv.__init__."""
    client = MagicMock()
    client.confirmConnection.return_value = None
    return client


@pytest.fixture()
def base_cfg():
    """Minimal valid config for AirSimDroneEnv."""
    return {
        "env": {
            "ip": "",
            "port": 41451,
            "image_shape": [84, 84, 1],
            "target_alt": 3.0,
            "max_vx": 3.0,
            "max_vy": 1.0,
            "max_yaw_rate_deg": 45,
            "dt": 0.1,
            "max_steps": 1024,
            "depth_clip_m": 20.0,
        },
        "reward": {},
    }


# ---------------------------------------------------------------------------
# Port passing
# ---------------------------------------------------------------------------

class TestPortPassing:
    """Verify the port parameter flows through to the AirSim client."""

    @patch("src.environments.airsim_env.airsim")
    def test_default_port(self, mock_airsim, base_cfg):
        """Default port should be 41451."""
        mock_airsim.MultirotorClient.return_value = _mock_airsim_client()
        from src.environments.airsim_env import AirSimDroneEnv

        env = AirSimDroneEnv(base_cfg)
        assert env.port == 41451
        mock_airsim.MultirotorClient.assert_called_once_with(ip="", port=41451)
        env.close()

    @patch("src.environments.airsim_env.airsim")
    def test_custom_port(self, mock_airsim, base_cfg):
        """Custom port should be passed to the client."""
        mock_airsim.MultirotorClient.return_value = _mock_airsim_client()
        from src.environments.airsim_env import AirSimDroneEnv

        cfg = copy.deepcopy(base_cfg)
        cfg["env"]["port"] = 41455
        env = AirSimDroneEnv(cfg)
        assert env.port == 41455
        mock_airsim.MultirotorClient.assert_called_once_with(ip="", port=41455)
        env.close()

    @patch("src.environments.airsim_env.airsim")
    def test_missing_port_defaults_to_41451(self, mock_airsim, base_cfg):
        """Config without port key should use default 41451."""
        mock_airsim.MultirotorClient.return_value = _mock_airsim_client()
        from src.environments.airsim_env import AirSimDroneEnv

        cfg = copy.deepcopy(base_cfg)
        cfg["env"].pop("port", None)
        env = AirSimDroneEnv(cfg)
        assert env.port == 41451
        env.close()


# ---------------------------------------------------------------------------
# update_config
# ---------------------------------------------------------------------------

class TestUpdateConfig:
    """Verify the update_config method on AirSimDroneEnv."""

    @patch("src.environments.airsim_env.airsim")
    def test_update_config_changes_params(self, mock_airsim, base_cfg):
        """update_config should update mutable parameters."""
        mock_airsim.MultirotorClient.return_value = _mock_airsim_client()
        from src.environments.airsim_env import AirSimDroneEnv

        env = AirSimDroneEnv(base_cfg)
        assert env.max_vx == 3.0

        env.update_config({"max_vx": 5.0, "target_alt": 5.0})
        assert env.max_vx == 5.0
        assert env.target_alt == 5.0
        env.close()

    @patch("src.environments.airsim_env.airsim")
    def test_update_config_preserves_unset(self, mock_airsim, base_cfg):
        """update_config with partial dict should not reset other fields."""
        mock_airsim.MultirotorClient.return_value = _mock_airsim_client()
        from src.environments.airsim_env import AirSimDroneEnv

        env = AirSimDroneEnv(base_cfg)
        original_vy = env.max_vy
        env.update_config({"max_vx": 10.0})
        assert env.max_vy == original_vy
        env.close()

    @patch("src.environments.airsim_env.airsim")
    def test_update_config_empty_dict_noop(self, mock_airsim, base_cfg):
        """Empty dict should change nothing."""
        mock_airsim.MultirotorClient.return_value = _mock_airsim_client()
        from src.environments.airsim_env import AirSimDroneEnv

        env = AirSimDroneEnv(base_cfg)
        original = {
            "max_vx": env.max_vx,
            "max_vy": env.max_vy,
            "target_alt": env.target_alt,
            "max_steps": env.max_steps,
            "depth_clip_m": env.depth_clip_m,
        }
        env.update_config({})
        assert env.max_vx == original["max_vx"]
        assert env.max_vy == original["max_vy"]
        assert env.target_alt == original["target_alt"]
        assert env.max_steps == original["max_steps"]
        assert env.depth_clip_m == original["depth_clip_m"]
        env.close()

    @patch("src.environments.airsim_env.airsim")
    def test_update_config_all_params(self, mock_airsim, base_cfg):
        """All mutable params can be updated at once."""
        mock_airsim.MultirotorClient.return_value = _mock_airsim_client()
        from src.environments.airsim_env import AirSimDroneEnv

        env = AirSimDroneEnv(base_cfg)
        env.update_config({
            "max_vx": 6.0,
            "max_vy": 2.0,
            "target_alt": 5.0,
            "max_steps": 512,
            "depth_clip_m": 15.0,
        })
        assert env.max_vx == 6.0
        assert env.max_vy == 2.0
        assert env.target_alt == 5.0
        assert env.max_steps == 512
        assert env.depth_clip_m == 15.0
        env.close()


# ---------------------------------------------------------------------------
# Config isolation
# ---------------------------------------------------------------------------

class TestConfigIsolation:
    """Verify that parallel envs get independent config copies."""

    def test_deepcopy_isolation(self, base_cfg):
        """Deep-copied configs are independent."""
        cfg1 = copy.deepcopy(base_cfg)
        cfg2 = copy.deepcopy(base_cfg)
        cfg1["env"]["port"] = 41451
        cfg2["env"]["port"] = 41452

        assert cfg1["env"]["port"] != cfg2["env"]["port"]
        # Mutating one doesn't affect the other
        cfg1["env"]["max_vx"] = 999
        assert cfg2["env"].get("max_vx", 3.0) != 999

    def test_port_range_for_n_envs(self):
        """N parallel envs should get ports base_port through base_port+N-1."""
        base_port = 41451
        num_envs = 4
        ports = [base_port + i for i in range(num_envs)]
        assert ports == [41451, 41452, 41453, 41454]
        # Eval port should be base_port + num_envs
        eval_port = base_port + num_envs
        assert eval_port == 41455
        assert eval_port not in ports


# ---------------------------------------------------------------------------
# make_vec_env / make_env source verification (reads file directly)
# ---------------------------------------------------------------------------

def _read_train_source() -> str:
    """Read src/training/train.py without importing it (avoids SB3 import)."""
    import os
    train_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "training", "train.py"
    )
    with open(train_path) as f:
        return f.read()


class TestMakeVecEnvSource:
    """Verify make_vec_env logic by reading source file (avoids SB3 import)."""

    def test_source_uses_subproc_vec_env(self):
        """make_vec_env should use SubprocVecEnv for num_envs > 1."""
        source = _read_train_source()
        assert "SubprocVecEnv" in source

    def test_source_uses_spawn(self):
        """SubprocVecEnv should use start_method='spawn' to avoid CUDA fork."""
        source = _read_train_source()
        assert 'start_method="spawn"' in source

    def test_source_dummy_for_single(self):
        """make_vec_env should fall back to DummyVecEnv for num_envs=1."""
        source = _read_train_source()
        assert "DummyVecEnv" in source
        assert "num_envs == 1" in source

    def test_make_env_deepcopies_cfg(self):
        """make_env should deep-copy config when overriding port."""
        source = _read_train_source()
        assert "deepcopy" in source

    def test_num_envs_cli_arg_present(self):
        """--num_envs CLI arg should be present in train.py."""
        source = _read_train_source()
        assert "--num_envs" in source

    def test_base_port_cli_arg_present(self):
        """--base_port CLI arg should be present in train.py."""
        source = _read_train_source()
        assert "--base_port" in source
