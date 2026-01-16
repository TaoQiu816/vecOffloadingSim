#!/usr/bin/env python3
"""
Standardized evaluation sweep.

Outputs:
  - metrics.csv: per-episode metrics
  - summary.csv: aggregate per seed and overall
"""

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from baselines import RandomPolicy
from agents.mappo_agent import MAPPOAgent
from models.offloading_policy import OffloadingPolicyNetwork


def _parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation sweep and export CSV metrics.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seed list")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "local", "eft", "mappo"],
                        help="Evaluation policy")
    parser.add_argument("--dag-source", type=str, default="synthetic_small",
                        choices=["synthetic_small", "synthetic_large", "workflow_json"])
    parser.add_argument("--large-nodes", type=str, default=None,
                        help="Comma-separated node counts for synthetic_large (e.g., 20,50,100)")
    parser.add_argument("--workflow-path", type=str, default=None, help="Local workflow JSON path")
    parser.add_argument("--max-nodes", type=int, default=None, help="Override MAX_NODES for evaluation")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for CSVs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic policy for checkpoint")
    return parser.parse_args()


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_seeds(seed_str: str) -> List[int]:
    return [int(s.strip()) for s in seed_str.split(",") if s.strip()]


def _configure_dag_source(args):
    Cfg.DAG_SOURCE = args.dag_source
    if args.dag_source == "synthetic_large":
        if args.large_nodes:
            values = [int(v.strip()) for v in args.large_nodes.split(",") if v.strip()]
            Cfg.DAG_LARGE_NODE_OPTIONS = values
        if not Cfg.DAG_LARGE_NODE_OPTIONS:
            raise ValueError("DAG_LARGE_NODE_OPTIONS is empty for synthetic_large")
    elif args.dag_source == "workflow_json":
        if not args.workflow_path:
            raise ValueError("--workflow-path is required for workflow_json")
        Cfg.WORKFLOW_JSON_PATH = args.workflow_path

    if args.max_nodes is not None:
        Cfg.MAX_NODES = int(args.max_nodes)


def _load_agent(checkpoint_path: str, device: str) -> MAPPOAgent:
    network = OffloadingPolicyNetwork(
        d_model=TC.EMBED_DIM,
        num_heads=TC.NUM_HEADS,
        num_layers=TC.NUM_LAYERS
    )
    agent = MAPPOAgent(network, device=device)
    agent.load(checkpoint_path)
    agent.network.eval()
    return agent


def _episode_metrics_from_env(env, info: Dict) -> Dict:
    if info and isinstance(info, dict):
        metrics = info.get("episode_metrics")
        if metrics:
            return metrics
    return getattr(env, "_last_episode_metrics", {}) or {}


def _summarize_records(records: List[Dict], seed: int = None) -> Dict:
    numeric_fields = [
        "success_rate",
        "deadline_miss_rate",
        "latency_mean",
        "latency_p95",
        "energy_norm_mean",
        "throughput",
    ]
    summary = {
        "seed": seed if seed is not None else "overall",
        "episodes": len(records),
        "terminated_rate": float(np.mean([r["terminated"] for r in records])) if records else 0.0,
        "truncated_rate": float(np.mean([r["truncated"] for r in records])) if records else 0.0,
    }
    for field in numeric_fields:
        values = [r[field] for r in records]
        summary[f"{field}_mean"] = float(np.mean(values)) if values else 0.0
        summary[f"{field}_std"] = float(np.std(values)) if values else 0.0
    return summary


def run_eval(args):
    _configure_dag_source(args)
    seeds = _parse_seeds(args.seeds)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_dir = os.path.join("eval_results", f"sweep_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    use_checkpoint = args.policy == "mappo"
    if use_checkpoint:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for policy=mappo")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        agent = _load_agent(args.checkpoint, device=args.device)
    else:
        agent = None

    all_records = []
    for seed in seeds:
        _set_global_seed(seed)
        env = VecOffloadingEnv(config=Cfg)
        if use_checkpoint:
            policy = agent
        elif args.policy == "local":
            from baselines import LocalOnlyPolicy
            policy = LocalOnlyPolicy()
        elif args.policy == "eft":
            from baselines import EFTPPolicy
            policy = EFTPPolicy(env)
        else:
            policy = RandomPolicy(seed=seed)

        for ep in range(args.episodes):
            ep_seed = seed + ep
            _set_global_seed(ep_seed)
            Cfg.SEED = ep_seed
            obs_list, _ = env.reset(seed=ep_seed)
            if hasattr(policy, "reset"):
                policy.reset()
            done = False
            while not done:
                if use_checkpoint:
                    action_out = policy.select_action(obs_list, deterministic=not args.stochastic)
                    actions = action_out["actions"]
                else:
                    actions = policy.select_action(obs_list)
                obs_list, rewards, terminated, truncated, info = env.step(actions)
                done = bool(terminated or truncated)

            metrics = _episode_metrics_from_env(env, info)
            episode_time = float(metrics.get("episode_time_seconds", env.time))
            completed_tasks = int(metrics.get("completed_tasks_count", 0))
            throughput = completed_tasks / max(episode_time, 1e-9)
            vehicle_counts = [v.task_dag.num_subtasks for v in env.vehicles]
            avg_nodes = float(np.mean(vehicle_counts)) if vehicle_counts else 0.0

            record = {
                "seed": seed,
                "episode": ep + 1,
                "policy": args.policy,
                "dag_source": args.dag_source,
                "avg_dag_nodes": avg_nodes,
                "terminated": bool(metrics.get("terminated", terminated)),
                "truncated": bool(metrics.get("truncated", truncated)),
                "success_rate": float(metrics.get("task_success_rate", 0.0)),
                "deadline_miss_rate": float(metrics.get("deadline_miss_rate", 0.0)),
                "latency_mean": float(metrics.get("task_duration_mean", 0.0)),
                "latency_p95": float(metrics.get("task_duration_p95", 0.0)),
                "energy_norm_mean": float(metrics.get("energy_norm_mean", 0.0)),
                "throughput": float(throughput),
                "episode_time_seconds": float(episode_time),
                "completed_tasks": int(completed_tasks),
            }
            all_records.append(record)

        env.close()

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    summary_rows = []
    for seed in seeds:
        seed_records = [r for r in all_records if r["seed"] == seed]
        summary_rows.append(_summarize_records(seed_records, seed=seed))
    summary_rows.append(_summarize_records(all_records, seed=None))

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[Eval] metrics.csv: {metrics_path}")
    print(f"[Eval] summary.csv: {summary_path}")


def main():
    args = _parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
