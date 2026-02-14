import airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import math


class AirSimDroneEnv(gym.Env):
    """
    Gymnasium interface for AirSim Quadrotor RL training.

    Observation (Dict):
        - image: Depth (H, W, 1) normalised to [0, 1]
        - velocity: [vx, vy, yaw_rate] in body frame

    Action (Box[-1, 1]):
        - [target_vx, target_vy, target_yaw_rate] scaled to physical limits

    Uses simContinueForTime for lockstep simulation stepping (no wall-clock
    sleeps) and moveByVelocityZBodyFrameAsync for active altitude hold.
    """

    metadata = {"render_modes": ["human"]}
    MAX_RESET_RETRIES = 5

    def __init__(self, cfg: dict | None = None):
        super().__init__()

        cfg = cfg or {}
        env_cfg = cfg.get("env", {})
        reward_cfg = cfg.get("reward", {})

        self.ip = env_cfg.get("ip", "")
        self.image_shape = tuple(env_cfg.get("image_shape", [84, 84, 1]))
        self.target_alt = env_cfg.get("target_alt", 3.0)
        self.max_vx = env_cfg.get("max_vx", 3.0)
        self.max_vy = env_cfg.get("max_vy", 1.0)
        self.max_yaw_rate = np.deg2rad(env_cfg.get("max_yaw_rate_deg", 45))
        self.dt = env_cfg.get("dt", 0.1)
        self.max_steps = env_cfg.get("max_steps", 1024)

        # Reward weights
        self.w_progress = reward_cfg.get("w_progress", 0.5)
        self.w_collision = reward_cfg.get("w_collision", -100.0)
        self.w_smoothness = reward_cfg.get("w_smoothness", -0.1)

        # AirSim client
        self.client = airsim.MultirotorClient(ip=self.ip)
        self.client.confirmConnection()

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
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        })

        # Internal state
        self.state = {
            "image": np.zeros(self.image_shape, dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
        }
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        # Depth image
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        if responses:
            img1d = np.array(responses[0].image_data_float, dtype=np.float32)
            img1d = img1d.reshape(responses[0].height, responses[0].width)
            img_depth = cv2.resize(img1d, (self.image_shape[1], self.image_shape[0]))
            img_depth = np.clip(img_depth, 0, 20) / 20.0
            if len(self.image_shape) == 3:
                img_depth = np.expand_dims(img_depth, axis=-1)
            self.state["image"] = img_depth

        # Body-frame velocity
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

        self.state["velocity"] = np.array([
            v_body[0],
            v_body[1],
            kin.angular_velocity.z_val,
        ], dtype=np.float32)

        return self.state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for attempt in range(self.MAX_RESET_RETRIES):
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            self.client.moveToZAsync(-self.target_alt, 1.0).join()

            if not self.client.simGetCollisionInfo().has_collided:
                break

            # Collision on spawn — teleport to a safe pose and retry
            self.client.simSetVehiclePose(
                airsim.Pose(
                    airsim.Vector3r(0, 0, -self.target_alt),
                    airsim.to_quaternion(0, 0, 0),
                ),
                ignore_collision=True,
            )

        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step  (lockstep: fire command → advance sim by dt → read state)
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_vx = action[0] * self.max_vx
        target_vy = action[1] * self.max_vy
        target_yaw_rate = action[2] * self.max_yaw_rate

        # Fire-and-forget velocity command (do NOT .join())
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

        # Advance simulation exactly dt seconds (lockstep)
        self.client.simContinueForTime(self.dt)
        # simContinueForTime is non-blocking; simPause to wait for it
        self.client.simPause(True)

        obs = self._get_obs()
        self.step_count += 1

        # --- Reward ---
        vx_body = obs["velocity"][0]
        r_progress = self.w_progress * vx_body

        has_collided = self.client.simGetCollisionInfo().has_collided
        r_collision = self.w_collision if has_collided else 0.0

        action_delta = np.linalg.norm(action - self.prev_action)
        r_smoothness = self.w_smoothness * action_delta

        reward = r_progress + r_collision + r_smoothness
        self.prev_action = action.copy()

        terminated = has_collided
        truncated = self.step_count >= self.max_steps

        info = {
            "r_progress": r_progress,
            "r_collision": r_collision,
            "r_smoothness": r_smoothness,
            "vx_body": vx_body,
            "step_count": self.step_count,
        }

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
