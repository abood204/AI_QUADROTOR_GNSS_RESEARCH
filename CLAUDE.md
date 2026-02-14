# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

**GNSS-Denied Quadrotor Autonomy via Reinforcement Learning**.
A depth-only PPO agent learns obstacle avoidance in AirSim, generalizing across environments via domain randomization and curriculum training. Simulation-first, designed for later sim-to-real transfer.

## Environment Setup

- **Python 3.11.x** — install via `pip install -e ".[dev]"` or `make install`
- **Core ML**: `torch`, `stable-baselines3`, `gymnasium`, `tensorboard`
- **Simulation**: `airsim==1.8.1`, `msgpack-rpc-python`
- **Dev**: `pytest`, `ruff`

## Commands

All modules are run as packages from the project root:

```bash
# Install (editable)
make install

# Verify AirSim environment and Gymnasium wrapper
make check-env

# Train PPO agent
make train                                        # uses configs/train_ppo.yaml
python -m src.training.train --total_timesteps 4096  # quick smoke test

# Evaluate trained policy
python -m src.evaluation.evaluate --model logs/ppo/best_model/best_model.zip

# Deploy (live inference)
python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip

# Visualize training
tensorboard --logdir logs/ppo/

# Lint + test
make lint
make test
```

## Source Layout

```
src/
  environments/     Gymnasium env wrapper (AirSimDroneEnv)
  training/         PPO training loop, callbacks, curriculum
  evaluation/       Standardized eval protocol, metrics, plots
  deployment/       Live inference with safety monitor
  safety/           Hard safety envelope (velocity clamp, proximity, e-stop)
  perception/       Observation pipeline (depth processing, frame stacking)
  control/          PID controller, AirSim interface
  utils/            Logging, camera helpers
scripts/            CLI utilities (check_env)
configs/            YAML configs (train, rewards, environments, control)
tests/              Unit and integration tests
docs/               FYP documents, architecture docs
```

## Architecture

### Observation Pipeline
- **Depth**: Forward camera (84x84), clipped at 20m, normalized [0,1], 4-frame stack
- **Velocity**: Body-frame [vx, vy, yaw_rate] from AirSim kinematics
- **Space**: `Dict{"image": Box(0,1, (84,84,4)), "velocity": Box(-inf,inf, (3,))}`

### Policy
- **Algorithm**: PPO (stable-baselines3) with MultiInputPolicy
- **Action**: Continuous [-1,1]^3 -> [vx, vy, yaw_rate] scaled to physical limits

### Reward
- Progress: `w_progress * vx_body` (incentivize forward motion)
- Collision: `w_collision` on terminal collision (default -100)
- Smoothness: `w_smoothness * ||action - prev_action||` (penalize jerk)

### Simulation
- **Lockstep**: `simContinueForTime(dt)` + `simPause(True)` for deterministic training
- **Altitude hold**: `moveByVelocityZBodyFrameAsync` (NOT `moveByVelocityBodyFrameAsync` which drifts)
- **Reset**: Retry loop with `simSetVehiclePose(ignore_collision=True)` to avoid spawn collisions

## Coding Conventions

- **Tensors**: All ML inputs `(N, C, H, W)` normalized to `[0, 1]`
- **Coordinates**: AirSim NED (Down is +Z). Altitude commands are negative.
- **Safety**: Never remove `client.enableApiControl(True)` checks. Always `try/finally` to land on error.
- **Config**: All hyperparameters in YAML files under `configs/`, never hardcoded.
- **Imports**: Use `src.environments.airsim_env`, `src.training.train`, etc.
