"""Environment scheduler for multi-environment training.

Rotates between AirSim environment configurations every N episodes,
forcing the policy to generalize across different obstacle layouts.
The rotation is implemented as a callback that updates the env config
on episode boundaries.
"""
from __future__ import annotations


import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


class EnvironmentScheduler(BaseCallback):
    """Rotate environment configs every N episodes.

    This callback tracks episode completions and, at rotation boundaries,
    updates the environment configuration dict. Since AirSim runs a single
    instance, "rotation" means changing config parameters (max_vx, target_alt,
    etc.) that take effect on the next reset.

    For full environment switching (different UE maps), the user must
    restart AirSim with the appropriate settings.json — this callback
    logs which config should be active.
    """

    def __init__(
        self,
        env_configs: list[dict],
        rotate_every_episodes: int = 50,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_configs = env_configs
        self.rotate_every = rotate_every_episodes
        self._episode_count = 0
        self._current_idx = 0

    @classmethod
    def from_config_paths(
        cls,
        paths: list[str],
        rotate_every_episodes: int = 50,
    ) -> EnvironmentScheduler:
        """Load environment configs from YAML file paths."""
        configs = []
        for path in paths:
            with open(path, "r") as f:
                configs.append(yaml.safe_load(f))
        return cls(configs, rotate_every_episodes)

    @property
    def current_config(self) -> dict:
        return self.env_configs[self._current_idx]

    def _on_step(self) -> bool:
        # Detect episode ends from 'dones'
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._episode_count += 1

                if self._episode_count % self.rotate_every == 0:
                    self._current_idx = (self._current_idx + 1) % len(self.env_configs)
                    new_cfg = self.env_configs[self._current_idx]
                    self.logger.record("env/config_idx", self._current_idx)
                    if self.verbose > 0:
                        print(f"[env_scheduler] Episode {self._episode_count}: "
                              f"rotating to config {self._current_idx}")

                    # Update the underlying env's config parameters.
                    # Unwrap VecFrameStack (or any outer wrapper) to
                    # reach DummyVecEnv / SubprocVecEnv.
                    vec_env = self.training_env
                    while hasattr(vec_env, "venv"):
                        vec_env = vec_env.venv

                    env_cfg = new_cfg.get("env", {})

                    if isinstance(vec_env, SubprocVecEnv):
                        # SubprocVecEnv: call update_config inside each
                        # subprocess via the method added to AirSimDroneEnv.
                        vec_env.env_method("update_config", env_cfg)
                    elif hasattr(vec_env, "envs"):
                        # DummyVecEnv: direct attribute access
                        for env_wrapper in vec_env.envs:
                            raw = env_wrapper
                            while hasattr(raw, "env"):
                                raw = raw.env
                            if hasattr(raw, "update_config"):
                                raw.update_config(env_cfg)

        return True
