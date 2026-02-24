"""Gymnasium environment for AirSim Quadrotor RL training.

Observation (Dict):
    - image: Depth (H, W, 1) normalised to [0, 1]
    - velocity: [vx, vy, yaw_rate] in body frame

Action (Box[-1, 1]):
    - [target_vx, target_vy, target_yaw_rate] scaled to physical limits

Uses simContinueForTime for lockstep simulation stepping (no wall-clock
sleeps) and moveByVelocityZBodyFrameAsync for active altitude hold.
"""
from __future__ import annotations

import math
import time

import airsim
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environments.rewards import RewardFunction, WaypointRewardFunction


class AirSimDroneEnv(gym.Env):

    metadata = {"render_modes": ["human"]}
    MAX_RESET_RETRIES = 5

    def __init__(self, cfg: dict | None = None):
        super().__init__()

        cfg = cfg or {}
        env_cfg = cfg.get("env", {})
        reward_cfg = cfg.get("reward", {})
        self._dr_cfg = cfg.get("domain_randomization", {})

        self.ip = env_cfg.get("ip", "")
        self.image_shape = tuple(env_cfg.get("image_shape", [84, 84, 1]))
        self.target_alt = env_cfg.get("target_alt", 3.0)
        self.max_vx = env_cfg.get("max_vx", 3.0)
        self.max_vy = env_cfg.get("max_vy", 1.0)
        self.max_yaw_rate = np.deg2rad(env_cfg.get("max_yaw_rate_deg", 45))
        self.dt = env_cfg.get("dt", 0.1)
        self.max_steps = env_cfg.get("max_steps", 1024)
        self.depth_clip_m = env_cfg.get("depth_clip_m", 20.0)

        # Goal navigation (backward-compatible — off by default)
        self.goal_navigation = env_cfg.get("goal_navigation", False)
        self.num_waypoints = env_cfg.get("num_waypoints", 3)
        self.goal_radius_m = env_cfg.get("goal_radius_m", 3.0)
        self.max_goal_dist_m = env_cfg.get("max_goal_dist_m", 30.0)
        self.waypoint_arena_half = env_cfg.get("waypoint_arena_half_m", 20.0)

        # Pluggable reward — select function based on mode
        if self.goal_navigation:
            self.reward_fn = WaypointRewardFunction(reward_cfg)
        else:
            self.reward_fn = RewardFunction(reward_cfg)

        # AirSim client
        self.client = airsim.MultirotorClient(ip=self.ip)
        self.client.confirmConnection()

        # Velocity observation dimension: 3 (baseline) or 6 (goal navigation)
        # Goal nav adds: [cos_theta, sin_theta, dist_norm] to [vx, vy, yaw_rate]
        vel_dim = 6 if self.goal_navigation else 3

        # Spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0, shape=self.image_shape, dtype=np.float32
            ),
            "velocity": spaces.Box(
                low=-np.inf, high=np.inf, shape=(vel_dim,), dtype=np.float32
            ),
        })

        # Internal state
        self.state = {
            "image": np.zeros(self.image_shape, dtype=np.float32),
            "velocity": np.zeros(vel_dim, dtype=np.float32),
        }
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self._last_col_ts = 0  # updated in reset()

        # Goal navigation internal state (initialised in reset)
        self._waypoint_queue: list[tuple[float, float]] = []
        self._goals_reached_this_episode = 0
        self._total_goals_this_episode = 0

    # ------------------------------------------------------------------
    # Domain Randomization
    # ------------------------------------------------------------------
    def _apply_domain_randomization(self):
        """Apply domain randomization on episode reset.

        Hooks for sensor noise, spawn position, and future texture variation.
        Controlled via `domain_randomization` key in config YAML.
        """
        if not self._dr_cfg.get("enabled", False):
            return

        # Depth noise: Gaussian noise injected per-step in _get_depth_image
        self._depth_noise_std = self._dr_cfg.get("depth_noise_std", 0.0)

        # Spawn position randomization
        spawn_radius = self._dr_cfg.get("spawn_radius_m", 0.0)
        if spawn_radius > 0:
            dx = float(self.np_random.uniform(-spawn_radius, spawn_radius))
            dy = float(self.np_random.uniform(-spawn_radius, spawn_radius))
            random_yaw = float(self.np_random.uniform(-math.pi, math.pi))
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(dx, dy, -self.target_alt),
                    airsim.to_quaternion(0, 0, random_yaw),
                ),
                ignore_collision=True,
            )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_depth_image(self) -> np.ndarray:
        """Capture and process depth image from AirSim."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        if not responses:
            return np.zeros(self.image_shape, dtype=np.float32)

        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img1d = img1d.reshape(responses[0].height, responses[0].width)
        img_depth = cv2.resize(img1d, (self.image_shape[1], self.image_shape[0]))
        img_depth = np.clip(img_depth, 0, self.depth_clip_m) / self.depth_clip_m

        # Domain randomization: sensor noise
        if hasattr(self, "_depth_noise_std") and self._depth_noise_std > 0:
            noise = self.np_random.normal(0, self._depth_noise_std, img_depth.shape)
            img_depth = np.clip(img_depth + noise, 0.0, 1.0).astype(np.float32)

        if len(self.image_shape) == 3:
            img_depth = np.expand_dims(img_depth, axis=-1)

        return img_depth

    def _get_body_velocity(self) -> np.ndarray:
        """Get body-frame velocity [vx, vy, yaw_rate]."""
        kin = self.client.getMultirotorState().kinematics_estimated
        v_global = np.array([
            kin.linear_velocity.x_val,
            kin.linear_velocity.y_val,
            kin.linear_velocity.z_val,
        ])
        yaw = airsim.to_eularian_angles(kin.orientation)[2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        v_body = R_yaw @ v_global

        return np.array([
            v_body[0],
            v_body[1],
            kin.angular_velocity.z_val,
        ], dtype=np.float32)

    def _sample_waypoints(self) -> list[tuple[float, float]]:
        """Sample num_waypoints positions with minimum spacing constraint.

        Returns a list of (x, y) world-frame tuples. Retries up to 200 times
        to satisfy the 2 * goal_radius_m minimum inter-waypoint spacing.
        """
        waypoints: list[tuple[float, float]] = []
        min_spacing = 2.0 * self.goal_radius_m
        half = self.waypoint_arena_half

        for _ in range(200 * self.num_waypoints):
            if len(waypoints) >= self.num_waypoints:
                break
            x = float(self.np_random.uniform(-half, half))
            y = float(self.np_random.uniform(-half, half))
            # Check spacing against all already-accepted waypoints
            if all(
                math.hypot(x - wx, y - wy) >= min_spacing
                for wx, wy in waypoints
            ):
                waypoints.append((x, y))

        return waypoints

    def _get_goal_obs(self) -> np.ndarray:
        """Compute goal-relative observation [cos_theta, sin_theta, dist_norm].

        Uses current AirSim state. cos/sin encoding avoids ±π discontinuity,
        providing a smooth 360° gradient for yaw control learning.

        Returns zeros if waypoint queue is empty (mission complete).
        """
        if not self._waypoint_queue:
            return np.zeros(3, dtype=np.float32)

        gx, gy = self._waypoint_queue[0]
        kin = self.client.getMultirotorState().kinematics_estimated
        px = kin.position.x_val
        py = kin.position.y_val
        yaw = airsim.to_eularian_angles(kin.orientation)[2]

        # World-frame delta
        dx_w = gx - px
        dy_w = gy - py

        # Rotate to body frame using negative yaw
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx_b = c * dx_w - s * dy_w
        dy_b = s * dx_w + c * dy_w

        bearing = math.atan2(dy_b, dx_b)
        dist = math.hypot(dx_w, dy_w)
        dist_norm = float(np.clip(dist / self.max_goal_dist_m, 0.0, 1.0))

        return np.array([math.cos(bearing), math.sin(bearing), dist_norm], dtype=np.float32)

    def _check_goal_reached(self) -> bool:
        """Check if the drone has reached the current waypoint.

        Pops the waypoint from the queue and increments the counter when reached.
        Returns True if a waypoint was reached this step.
        """
        if not self._waypoint_queue:
            return False

        gx, gy = self._waypoint_queue[0]
        kin = self.client.getMultirotorState().kinematics_estimated
        px = kin.position.x_val
        py = kin.position.y_val

        if math.hypot(px - gx, py - gy) <= self.goal_radius_m:
            self._waypoint_queue.pop(0)
            self._goals_reached_this_episode += 1
            return True
        return False

    def _get_obs(self) -> dict:
        depth = self._get_depth_image()
        vel = self._get_body_velocity()

        if self.goal_navigation:
            goal_obs = self._get_goal_obs()
            self.state["velocity"] = np.concatenate([vel, goal_obs])
        else:
            self.state["velocity"] = vel

        self.state["image"] = depth
        return self.state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Unpause sim — step() leaves it paused for lockstep,
        # and blocking calls (takeoff, moveToZ) hang on a paused sim.
        self.client.simPause(False)

        for attempt in range(self.MAX_RESET_RETRIES):
            self.client.reset()
            time.sleep(0.5)  # let physics settle after reset
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(-self.target_alt, 1.0).join()

            if not self.client.simGetCollisionInfo().has_collided:
                break

            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(0, 0, -self.target_alt),
                    airsim.to_quaternion(0, 0, 0),
                ),
                ignore_collision=True,
            )

        # Record collision timestamp to distinguish stale flags from
        # real in-episode collisions (AirSim doesn't reliably clear
        # has_collided on reset).
        self._last_col_ts = self.client.simGetCollisionInfo().time_stamp

        # Enter lockstep mode for deterministic training
        self.client.simPause(True)

        self._apply_domain_randomization()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        if self.goal_navigation:
            self._waypoint_queue = self._sample_waypoints()
            self._goals_reached_this_episode = 0
            self._total_goals_this_episode = len(self._waypoint_queue)
            # Clear prev_dist so first step gets no spurious delta reward
            self.reward_fn._prev_dist_norm = None

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step  (lockstep: fire command -> advance sim by dt -> read state)
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_vx = action[0] * self.max_vx
        target_vy = action[1] * self.max_vy
        target_yaw_rate = action[2] * self.max_yaw_rate

        self.client.moveByVelocityZBodyFrameAsync(
            float(target_vx),
            float(target_vy),
            float(-self.target_alt),
            self.dt,
            yaw_mode=airsim.YawMode(
                is_rate=True,
                yaw_or_rate=float(math.degrees(target_yaw_rate)),
            ),
        )

        self.client.simContinueForTime(self.dt)
        self.client.simPause(True)

        obs = self._get_obs()
        self.step_count += 1

        # Collision detection
        vx_body = obs["velocity"][0]
        col_info = self.client.simGetCollisionInfo()
        has_collided = (
            col_info.has_collided
            and col_info.time_stamp != self._last_col_ts
        )

        if self.goal_navigation:
            goal_reached = self._check_goal_reached()
            all_goals_done = (len(self._waypoint_queue) == 0)
            # dist_norm and cos_theta are in obs["velocity"][5] and [3]
            dist_norm = float(obs["velocity"][5]) if not all_goals_done else 0.0
            cos_theta = float(obs["velocity"][3])
            reward, reward_info = self.reward_fn(
                vx_body=vx_body,
                has_collided=has_collided,
                action=action,
                prev_action=self.prev_action,
                goal_reached=goal_reached,
                dist_norm=dist_norm,
                cos_theta=cos_theta,
                all_goals_done=all_goals_done,
            )
        else:
            goal_reached = False
            all_goals_done = False
            reward, reward_info = self.reward_fn(
                vx_body, has_collided, action, self.prev_action
            )

        self.prev_action = action.copy()

        terminated = has_collided or (self.goal_navigation and all_goals_done)
        truncated = self.step_count >= self.max_steps

        info = {
            **reward_info,
            "vx_body": vx_body,
            "step_count": self.step_count,
        }

        if self.goal_navigation:
            info.update({
                "goals_reached": self._goals_reached_this_episode,
                "total_goals": self._total_goals_this_episode,
                "mission_success": all_goals_done and not has_collided,
                "goal_reached_this_step": goal_reached,
            })

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            self.client.simPause(False)
        except RuntimeError:
            pass
        except Exception:
            pass
