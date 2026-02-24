"""Custom SB3 callbacks for training infrastructure."""
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    """Log reward components to TensorBoard at each rollout end.

    Reads info dicts from the Monitor wrapper and logs per-component
    reward averages to TensorBoard for analysis.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            reward_entry = {k: v for k, v in info.items() if k.startswith("r_")}
            if reward_entry:
                self._episode_rewards.append(reward_entry)
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_rewards:
            return
        n = len(self._episode_rewards)
        # Collect all reward keys seen across steps
        all_keys: set[str] = set()
        for d in self._episode_rewards:
            all_keys.update(d.keys())
        for key in sorted(all_keys):
            avg = sum(d.get(key, 0.0) for d in self._episode_rewards) / n
            self.logger.record(f"reward/{key}", avg)
        self._episode_rewards.clear()
