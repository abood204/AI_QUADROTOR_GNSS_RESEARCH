"""
Train a PPO agent on the AirSim Quadrotor environment.

Usage:
    python -m src.training.train --config configs/train_ppo.yaml
    python -m src.training.train --config configs/train_ppo.yaml --total_timesteps 4096
"""

import argparse
import os
from datetime import datetime

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environments.airsim_env import AirSimDroneEnv
from src.training.callbacks import RewardLoggingCallback
from src.training.env_scheduler import EnvironmentScheduler


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def make_env(cfg: dict):
    """Return a factory that creates a Monitored AirSimDroneEnv."""
    def _init():
        return Monitor(AirSimDroneEnv(cfg))
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on AirSim")
    parser.add_argument(
        "--config", type=str, default="configs/train_ppo.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=None,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--reward_config", type=str, default=None,
        help="Override reward weights from a separate YAML (e.g. configs/rewards/aggressive.yaml)",
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Custom run name (default: ppo_TIMESTAMP)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .zip to resume training from",
    )
    parser.add_argument(
        "--overrides", type=str, default=None,
        help="JSON string of config overrides, e.g. '{\"reward\":{\"w_dist\":0.0}}'",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply JSON overrides (highest priority — applied before reward_config)
    if args.overrides:
        import json
        _deep_merge(cfg, json.loads(args.overrides))

    # Override reward weights if separate reward config provided
    if args.reward_config:
        with open(args.reward_config, "r") as f:
            reward_override = yaml.safe_load(f)
        cfg.setdefault("reward", {}).update(reward_override)

    ppo_cfg = cfg["ppo"]
    out_cfg = cfg["output"]
    frame_stack = cfg.get("frame_stack", 4)

    total_timesteps = args.total_timesteps or ppo_cfg["total_timesteps"]

    # Timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_{timestamp}"
    run_dir = os.path.join(out_cfg["log_dir"], run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Environments ---
    # Single AirSim instance => DummyVecEnv (not SubprocVecEnv)
    train_env = DummyVecEnv([make_env(cfg)])
    train_env = VecFrameStack(train_env, n_stack=frame_stack, channels_order="last")

    eval_env = DummyVecEnv([make_env(cfg)])
    eval_env = VecFrameStack(eval_env, n_stack=frame_stack, channels_order="last")

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=out_cfg.get("checkpoint_freq", 10000),
        save_path=ckpt_dir,
        name_prefix="ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=out_cfg.get("eval_freq", 5000),
        n_eval_episodes=out_cfg.get("eval_episodes", 5),
        deterministic=True,
    )

    # --- Model ---
    if args.resume:
        print(f"[train_ppo] Resuming from checkpoint: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=run_dir,
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            tensorboard_log=run_dir,
            verbose=1,
        )

    print(f"[train_ppo] Run directory: {run_dir}")
    print(f"[train_ppo] Total timesteps: {total_timesteps}")

    reward_cb = RewardLoggingCallback()
    callbacks = [checkpoint_cb, eval_cb, reward_cb]

    # Multi-environment rotation
    multi_env_cfg = cfg.get("multi_env", {})
    if multi_env_cfg.get("enabled", False):
        env_config_paths = multi_env_cfg.get("configs", [])
        rotate_every = multi_env_cfg.get("rotate_every_episodes", 50)
        if env_config_paths:
            env_scheduler = EnvironmentScheduler.from_config_paths(
                env_config_paths, rotate_every_episodes=rotate_every
            )
            callbacks.append(env_scheduler)
            print(f"[train_ppo] Multi-env rotation ENABLED ({len(env_config_paths)} configs, "
                  f"rotate every {rotate_every} episodes)")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        print("\n[train_ppo] Training interrupted by user.")
    finally:
        final_path = os.path.join(run_dir, "final_model")
        model.save(final_path)
        print(f"[train_ppo] Final model saved to {final_path}.zip")
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
