import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork
from agents.mappo_agent import MAPPOAgent


def _apply_overrides(overrides):
    for key, value in overrides.items():
        setattr(Cfg, key, value)


def _evaluate(agent, episodes=50, seed=0):
    env = VecOffloadingEnv(config=Cfg)
    obs_list, _ = env.reset(seed=seed)
    episode_metrics = []
    for ep in range(episodes):
        done = False
        while not done:
            action_dict = agent.select_action(obs_list, deterministic=True)
            actions = action_dict["actions"]
            obs_list, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        metrics = getattr(env, "_last_episode_metrics", {}) or {}
        episode_metrics.append(metrics)
        obs_list, _ = env.reset(seed=seed + ep + 1)
    return episode_metrics


def _summarize(metrics_list):
    if not metrics_list:
        return {
            "success_rate_mean": np.nan,
            "success_rate_p95": np.nan,
            "makespan_mean": np.nan,
            "makespan_p95": np.nan,
            "illegal_rate_mean": np.nan,
            "illegal_rate_p95": np.nan,
        }
    df = pd.DataFrame(metrics_list)
    success = df.get("task_success_rate", pd.Series([], dtype=float)).fillna(0.0)
    makespan = df.get("task_duration_mean", pd.Series([], dtype=float)).fillna(0.0)
    illegal = df.get("illegal_count", pd.Series([], dtype=float)).fillna(0.0)
    steps = df.get("episode_steps", pd.Series([], dtype=float)).replace(0.0, np.nan)
    vehs = df.get("episode_vehicle_count", pd.Series([], dtype=float)).replace(0.0, np.nan)
    illegal_rate = (illegal / (steps * vehs)).fillna(0.0)
    return {
        "success_rate_mean": float(success.mean()) if len(success) else 0.0,
        "success_rate_p95": float(np.percentile(success, 95)) if len(success) else 0.0,
        "makespan_mean": float(makespan.mean()) if len(makespan) else 0.0,
        "makespan_p95": float(np.percentile(makespan, 95)) if len(makespan) else 0.0,
        "illegal_rate_mean": float(illegal_rate.mean()) if len(illegal_rate) else 0.0,
        "illegal_rate_p95": float(np.percentile(illegal_rate, 95)) if len(illegal_rate) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Hold-out generalization evaluation.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="logs/generalization_eval.csv")
    args = parser.parse_args()

    policy = OffloadingPolicyNetwork()
    agent = MAPPOAgent(policy, device="cpu")
    agent.load(args.model)

    base_cfg = {
        "MIN_NODES": int(Cfg.MIN_NODES),
        "MAX_NODES": int(Cfg.MAX_NODES),
        "DAG_DENSITY": float(Cfg.DAG_DENSITY),
        "DAG_FAT": float(Cfg.DAG_FAT),
        "V2V_RANGE": float(Cfg.V2V_RANGE),
        "RSU_RANGE": float(Cfg.RSU_RANGE),
    }
    base_v2v = base_cfg["V2V_RANGE"]
    base_rsu = base_cfg["RSU_RANGE"]
    base_max_nodes = base_cfg["MAX_NODES"]

    scenarios = [
        ("dag_max", {"MIN_NODES": base_max_nodes, "MAX_NODES": base_max_nodes}),
        ("dag_dense", {"DAG_DENSITY": 0.5, "DAG_FAT": 0.8}),
        ("range_small", {"V2V_RANGE": base_v2v * 0.7, "RSU_RANGE": base_rsu * 0.8}),
        ("range_large", {"V2V_RANGE": base_v2v * 1.3, "RSU_RANGE": base_rsu * 1.2}),
    ]

    rows = []
    for name, overrides in scenarios:
        _apply_overrides(base_cfg)
        _apply_overrides(overrides)
        metrics = _evaluate(agent, episodes=args.episodes, seed=args.seed)
        summary = _summarize(metrics)
        summary["scenario"] = name
        summary["episodes"] = args.episodes
        rows.append(summary)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
