"""Clean AirSim connection and flight control interface.

Extracts reusable patterns from the legacy controller into
standalone functions with proper error handling.
"""
from __future__ import annotations

import airsim


def connect(ip: str = "") -> airsim.MultirotorClient:
    """Connect to AirSim, enable API control, and arm the drone."""
    client = airsim.MultirotorClient(ip=ip)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def takeoff(client: airsim.MultirotorClient, altitude_m: float = 3.0):
    """Take off and climb to target altitude (NED: z = -altitude_m)."""
    client.takeoffAsync().join()
    client.moveToZAsync(z=-altitude_m, velocity=1.0).join()


def land(client: airsim.MultirotorClient):
    """Stop, land, disarm, and release API control."""
    try:
        client.moveByVelocityAsync(0, 0, 0, 1).join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception:
        # Best-effort landing — don't let errors propagate
        try:
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass


def safe_reset(
    client: airsim.MultirotorClient,
    altitude_m: float = 3.0,
    max_retries: int = 5,
) -> bool:
    """Reset the simulator and takeoff, retrying on spawn collisions.

    Returns True if reset succeeded without collision, False otherwise.
    """
    for attempt in range(max_retries):
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToZAsync(-altitude_m, 1.0).join()

        if not client.simGetCollisionInfo().has_collided:
            return True

        # Collision on spawn — teleport to safe pose
        client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(0, 0, -altitude_m),
                airsim.to_quaternion(0, 0, 0),
            ),
            ignore_collision=True,
        )

    return False
