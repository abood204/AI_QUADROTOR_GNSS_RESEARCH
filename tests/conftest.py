"""Pytest configuration: stub out AirSim and its broken dependencies.

All tests must run without AirSim connected. This conftest injects minimal
stubs for msgpackrpc, tornado, and airsim before any test module is imported,
allowing src.environments.airsim_env to be imported without a live simulator.

The AirSim *client* is still mocked per-test where needed via unittest.mock.
"""
import sys
import types
import unittest.mock


def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# ------------------------------------------------------------------
# Stub tornado.platform.auto (missing in Python 3.13 user site-packages)
# ------------------------------------------------------------------
_make_stub_module("tornado")
_make_stub_module("tornado.platform")
_make_stub_module("tornado.platform.auto")

# ------------------------------------------------------------------
# Stub msgpackrpc (depends on broken tornado)
# ------------------------------------------------------------------
_make_stub_module("msgpackrpc")
_make_stub_module("msgpackrpc.address")

# ------------------------------------------------------------------
# Stub airsim with just enough surface for AirSimDroneEnv to import
# ------------------------------------------------------------------
airsim_mod = _make_stub_module("airsim")
airsim_mod.MultirotorClient = unittest.mock.MagicMock
airsim_mod.ImageRequest = unittest.mock.MagicMock
airsim_mod.ImageType = unittest.mock.MagicMock()
airsim_mod.ImageType.DepthPerspective = 0
airsim_mod.Vector3r = unittest.mock.MagicMock
airsim_mod.Pose = unittest.mock.MagicMock
airsim_mod.YawMode = unittest.mock.MagicMock
airsim_mod.to_quaternion = unittest.mock.MagicMock(return_value=None)
airsim_mod.to_eularian_angles = unittest.mock.MagicMock(return_value=(0.0, 0.0, 0.0))
