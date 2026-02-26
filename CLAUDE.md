# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

**GNSS-Denied Quadrotor Autonomy via Reinforcement Learning**.
A depth-only PPO agent learns obstacle avoidance in AirSim, generalizing across environments via domain randomization and multi-environment rotation. Simulation-first, designed for later sim-to-real transfer via ONNX export.

**Current Phase**: Infrastructure complete (Weeks 1–4). Next phase is running training campaigns and collecting ablation results for the FYP thesis.

**Status**: 58/58 tests passing, 0 lint errors, 14 YAML configs, 4 training profiles. Parallel multi-env training supported via SubprocVecEnv.

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
make train                                                  # base config (1M steps)
python -m src.training.train --config configs/train_ppo_fast.yaml        # ~3-4h session
python -m src.training.train --config configs/train_ppo_dr.yaml          # with domain randomization
python -m src.training.train --config configs/train_ppo_multienv.yaml    # multi-env rotation
python -m src.training.train --total_timesteps 4096                      # quick smoke test

# Resume training from checkpoint
python -m src.training.train --resume logs/ppo/checkpoints/rl_model_XXXXX_steps.zip

# Parallel training (N AirSim instances, ~N× FPS)
export AIRSIM_BIN=/path/to/AirSimNH.sh
bash scripts/launch_airsim_cluster.sh 4          # launch 4 train + 1 eval AirSim instances
python -m src.training.train --num_envs 4 --base_port 41451  # train with 4 parallel envs
python -m src.training.train --num_envs 4 --total_timesteps 4096  # smoke test
bash scripts/launch_airsim_cluster.sh 4 --stop   # kill all instances
bash scripts/launch_airsim_cluster.sh 4 --status # check running instances

# Evaluate trained policy (20-episode full eval)
python scripts/run_full_eval.py --model logs/ppo/best_model/best_model.zip
python -m src.evaluation.evaluate --model logs/ppo/best_model/best_model.zip  # single run

# Run ablation suite (7 experiments across 4 axes)
python scripts/run_ablations.py
python scripts/run_ablations.py --dry-run     # preview only
python scripts/run_ablations.py --only no_smoothness progress_only  # subset

# Run reward sweep (3 profiles: default, aggressive, cautious)
python scripts/run_reward_sweep.py

# Deploy (live inference with safety monitor)
python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip
python -m src.deployment.deploy --model logs/ppo/best_model/best_model.zip --no_safety

# Export policy to ONNX (for sim-to-real transfer)
python scripts/export_onnx.py --model logs/ppo/best_model/best_model.zip

# Benchmark training FPS
python scripts/benchmark_fps.py

# Visualize training
tensorboard --logdir logs/ppo/

# Lint + test
make lint
make test
```

## Source Layout

```
src/
  environments/     AirSimDroneEnv (Gymnasium wrapper, lockstep sim, domain randomization)
  training/         PPO training loop, callbacks, multi-env scheduler, resume support
  evaluation/       Standardized eval protocol, 5 metrics, plots, cross-experiment comparison
  deployment/       Live inference with safety monitor integration
  safety/           4-layer safety envelope (velocity clamp, proximity, altitude, e-stop)
  perception/       PLACEHOLDER — depth processing currently lives in AirSimDroneEnv
  control/          PID controller, AirSim interface wrappers
  utils/            Logging, camera helpers
scripts/            run_full_eval, run_ablations, run_reward_sweep, export_onnx, benchmark_fps
configs/
  train_ppo.yaml          Base config (1M steps, n_steps=2048)
  train_ppo_fast.yaml     Speed-optimized (500K steps, n_steps=1024)
  train_ppo_dr.yaml       Domain randomization enabled
  train_ppo_multienv.yaml Multi-env rotation (NH + Blocks every 50 episodes)
  safety.yaml             Safety monitor limits
  settings_training.json  AirSim optimized settings (84x84 depth-only, eliminates 58x pixel waste)
  rewards/                default, aggressive, cautious
  environments/           airsim_nh (outdoor), blocks (indoor)
  ablations/              no_smoothness, progress_only, frame_stack_1, no_dr
  curriculum/             PLACEHOLDER — planned but not yet implemented
tests/              58 unit tests (rewards, safety, metrics, PID, DR, comparison) — AirSim-free
docs/               FYP documents, architecture docs
```

## Architecture

### Observation Pipeline
- **Depth**: Forward camera (84x84), clipped at 20m, normalized [0,1], 4-frame stack via `VecFrameStack`
- **Velocity**: Body-frame [vx, vy, yaw_rate] rotated from AirSim global kinematics using yaw matrix
- **Space**: `Dict{"image": Box(0,1, (84,84,4)), "velocity": Box(-inf,inf, (3,))}`
- **Domain Randomization**: Optional Gaussian depth noise (5%) + spawn position/yaw jitter (5m radius)

### Policy
- **Algorithm**: PPO (stable-baselines3) with `MultiInputPolicy`
- **Feature extraction**: NatureCNN for image branch, MLP for velocity branch, late fusion
- **Action**: Continuous `[-1,1]^3` → `[vx, vy, yaw_rate]` scaled to `[3.0 m/s, 1.0 m/s, 45°/s]`

### Reward
- Progress:   `w_progress * vx_body`                       (default: +0.5)
- Collision:  `w_collision` on terminal collision           (default: -100)
- Smoothness: `w_smoothness * ||action - prev_action||`    (default: -0.1)
- All weights configurable via `configs/rewards/*.yaml`

### Safety Monitor (deployment only)
1. **Velocity clamp**: Hard-limits to physical bounds
2. **Proximity scaling**: Center 30% ROI depth < 1.5m → linearly scale vx down to 0.2x
3. **Altitude guard**: Flag if deviation > 1.0m from target
4. **Emergency stop**: Zero all commands on collision or comms timeout

### Simulation
- **Lockstep**: `simContinueForTime(dt)` + `simPause(True)` for deterministic training
- **Altitude hold**: `moveByVelocityZBodyFrameAsync` (NOT `moveByVelocityBodyFrameAsync` which drifts)
- **Reset**: Unpause sim first (blocking calls hang on paused sim), retry loop with `simSetVehiclePose(ignore_collision=True)`, then 0.5s physics settle delay
- **Collision filtering**: Track `_last_col_ts`; only count collision if `time_stamp` has changed (prevents stale AirSim flags from killing fresh episodes)
- **ViewMode**: Use `FlyWithMe` (NOT `NoDisplay` which blacks out viewport — speed gains come from 84x84 camera config, not viewport mode)
- **Parallel envs**: Use `--num_envs N` to run N AirSim instances on ports `base_port` through `base_port+N-1`. Uses `SubprocVecEnv(start_method="spawn")` for N>1, `DummyVecEnv` for N=1. Launch script patches `ApiServerPort` in each instance's `settings.json`.

## Coding Conventions

- **Tensors**: All ML inputs `(N, C, H, W)` normalized to `[0, 1]`
- **Coordinates**: AirSim NED (Down is +Z). Altitude commands are negative (e.g., `-3.0m`).
- **Safety**: Never remove `client.enableApiControl(True)` checks. Always `try/finally` to land on error.
- **Config**: All hyperparameters in YAML files under `configs/`, never hardcoded.
- **Imports**: Use `src.environments.airsim_env`, `src.training.train`, etc.
- **Tests**: All tests must run without AirSim. Mock the client at the boundary.

## Known Issues / Outstanding Work

1. **Ablation override mechanism broken**: `run_ablations.py` passes `--overrides` to `train.py`, but `train.py` does not accept that flag. Only reward ablations work via `--reward_config`. Frame stack and DR ablations will fail until `train.py` supports config key overrides.

2. **`src/perception/` is empty**: Depth processing (normalize, clip, noise injection) lives inside `AirSimDroneEnv._get_depth_image()`. Extracting it into a standalone perception module would improve modularity for sim-to-real.

3. **`configs/curriculum/` is empty**: True curriculum learning (progressively harder environments) is not yet implemented. Round-robin rotation via `EnvironmentScheduler` is the current substitute.

4. **`src/eval/` directory**: Appears to be an abandoned duplicate of `src/evaluation/`. Should be cleaned up.
