"""Run comprehensive evaluation protocol on a trained model.

Executes the full evaluation pipeline:
  1. N episodes with fixed seeds for reproducibility
  2. Per-episode trajectory CSV + metrics
  3. Aggregate summary JSON
  4. Worst-K episode failure analysis
  5. Trajectory plots for best, median, and worst episodes

Usage:
    python -m scripts.run_full_eval --model logs/ppo/best_model/best_model.zip
    python -m scripts.run_full_eval --model logs/ppo/best_model/best_model.zip --episodes 20
"""

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environments.airsim_env import AirSimDroneEnv
from src.evaluation.metrics import compute_episode_summary, goal_completion_rate


def make_env(cfg: dict):
    def _init():
        return Monitor(AirSimDroneEnv(cfg))
    return _init


def run_episode(model, vec_env, client, dt, max_steps, seed=None):
    """Run a single evaluation episode and return trajectory + summary."""
    obs = vec_env.reset()
    trajectory = []
    collided = False
    goals_reached = 0
    total_goals = 0
    mission_success = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vec_env.step(action)

        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        x, y, z = pos.x_val, pos.y_val, pos.z_val
        info = infos[0]

        trajectory.append({
            "step": step,
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "vx_body": round(info.get("vx_body", 0.0), 3),
            "reward": round(float(reward[0]), 4),
        })

        if done[0]:
            if info.get("r_collision", 0.0) < 0:
                collided = True
            # Collect waypoint metrics if goal navigation is active
            if "goals_reached" in info:
                goals_reached = info["goals_reached"]
                total_goals = info["total_goals"]
                mission_success = info.get("mission_success", False)
            break

    summary = compute_episode_summary(
        trajectory,
        dt=dt,
        collided=collided,
        goals_reached_count=goals_reached,
        total_goals_count=total_goals,
        mission_success_flag=mission_success,
    )
    return trajectory, summary


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model")
    parser.add_argument("--config", type=str, default="configs/train_ppo.yaml")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--worst_k", type=int, default=3, help="Number of worst episodes to flag")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dt = cfg.get("env", {}).get("dt", 0.1)
    max_steps = cfg.get("env", {}).get("max_steps", 1024)
    frame_stack = cfg.get("frame_stack", 4)

    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("logs", "eval", f"full_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Environment setup
    vec_env = DummyVecEnv([make_env(cfg)])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="last")
    model = PPO.load(args.model, env=vec_env)

    raw_env = vec_env.envs[0].env
    client = raw_env.client

    print(f"[eval] Model: {args.model}")
    print(f"[eval] Episodes: {args.episodes}")
    print(f"[eval] Output: {out_dir}")

    # Run episodes
    all_summaries = []
    all_trajectories = []

    for ep in range(args.episodes):
        print(f"\n[eval] Episode {ep + 1}/{args.episodes}")
        trajectory, summary = run_episode(model, vec_env, client, dt, max_steps, seed=ep)
        summary["episode"] = ep + 1
        all_summaries.append(summary)
        all_trajectories.append(trajectory)

        # Save per-episode trajectory
        csv_path = os.path.join(out_dir, f"trajectory_ep{ep + 1:03d}.csv")
        if trajectory:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                writer.writeheader()
                writer.writerows(trajectory)

        print(f"  DBC={summary['distance_before_collision_m']:.1f}m  "
              f"collided={summary['collided']}  speed={summary['average_speed_ms']:.2f} m/s")

    # Aggregate metrics
    avg_dbc = np.mean([s["distance_before_collision_m"] for s in all_summaries])
    col_rate = sum(1 for s in all_summaries if s["collided"]) / len(all_summaries)
    avg_speed = np.mean([s["average_speed_ms"] for s in all_summaries])
    avg_smoothness = np.mean([s["path_smoothness_jerk"] for s in all_summaries])

    # Waypoint metrics (only populated if goal_navigation is active)
    gcr = goal_completion_rate(all_summaries)
    waypoint_episodes = [s for s in all_summaries if "total_goals_count" in s]
    avg_goals_per_ep = (
        np.mean([s["goals_reached_count"] for s in waypoint_episodes])
        if waypoint_episodes else 0.0
    )
    mission_success_rate = (
        sum(1 for s in waypoint_episodes if s.get("mission_success", False))
        / len(waypoint_episodes)
        if waypoint_episodes else 0.0
    )

    # Identify worst episodes by DBC
    sorted_eps = sorted(all_summaries, key=lambda s: s["distance_before_collision_m"])
    worst_episodes = sorted_eps[:args.worst_k]

    # Failure taxonomy
    failure_analysis = []
    for ep_summary in worst_episodes:
        failure_analysis.append({
            "episode": ep_summary["episode"],
            "dbc_m": ep_summary["distance_before_collision_m"],
            "collided": ep_summary["collided"],
            "survival_time_s": ep_summary["survival_time_s"],
            "category": "collision" if ep_summary["collided"] else "truncated",
        })

    # Save aggregate summary
    aggregate = {
        "model": args.model,
        "config": args.config,
        "num_episodes": args.episodes,
        "avg_distance_before_collision_m": round(float(avg_dbc), 2),
        "collision_rate": round(float(col_rate), 3),
        "avg_speed_ms": round(float(avg_speed), 3),
        "avg_path_smoothness_jerk": round(float(avg_smoothness), 3),
        "worst_episodes": failure_analysis,
        "episode_summaries": all_summaries,
    }
    if waypoint_episodes:
        aggregate.update({
            "goal_completion_rate": round(float(gcr), 3),
            "avg_goals_per_episode": round(float(avg_goals_per_ep), 2),
            "mission_success_rate": round(float(mission_success_rate), 3),
        })

    json_path = os.path.join(out_dir, "eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Generate plots
    try:
        from src.evaluation.plots import plot_trajectory

        # Plot best, median, worst trajectories
        sorted_by_dbc = sorted(enumerate(all_trajectories), key=lambda x: all_summaries[x[0]]["distance_before_collision_m"])
        for label, idx in [("worst", 0), ("median", len(sorted_by_dbc) // 2), ("best", -1)]:
            ep_idx, traj = sorted_by_dbc[idx]
            csv_p = os.path.join(out_dir, f"trajectory_ep{ep_idx + 1:03d}.csv")
            if os.path.exists(csv_p):
                plot_trajectory(
                    csv_p,
                    output_path=os.path.join(out_dir, f"trajectory_{label}.png"),
                    title=f"{label.capitalize()} Episode (DBC={all_summaries[ep_idx]['distance_before_collision_m']:.1f}m)",
                )
    except Exception as e:
        print(f"[eval] Plot generation error: {e}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  EVALUATION SUMMARY ({args.episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Avg DBC:           {avg_dbc:.1f} m")
    print(f"  Collision rate:    {col_rate:.1%}")
    print(f"  Avg speed:         {avg_speed:.2f} m/s")
    print(f"  Avg smoothness:    {avg_smoothness:.3f}")
    if waypoint_episodes:
        print(f"  Goal completion:   {gcr:.1%}")
        print(f"  Avg goals/ep:      {avg_goals_per_ep:.2f}")
        print(f"  Mission success:   {mission_success_rate:.1%}")
    print(f"\n  Worst {args.worst_k} episodes:")
    for fa in failure_analysis:
        print(f"    Ep {fa['episode']}: DBC={fa['dbc_m']:.1f}m  ({fa['category']})")
    print(f"\n  Results saved to: {out_dir}")

    # Cleanup
    try:
        vec_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
