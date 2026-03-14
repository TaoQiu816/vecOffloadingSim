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
                        choices=["random", "local", "eft", "cp_eft", "greedy", "lb_greedy", "oracle_min", "static", "mappo"],
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
    parser.add_argument("--log-interval", type=int, default=1, help="Print progress every N eval episodes")
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
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
        "avg_rsu_queue",
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
    # 条件时延：仅对 completed_tasks>0 的 episode 求均值，避免零值污染
    cond_values = [r["latency_mean_cond_success"] for r in records
                   if not np.isnan(r.get("latency_mean_cond_success", float("nan")))]
    summary["latency_mean_cond_success"] = float(np.mean(cond_values)) if cond_values else float("nan")
    summary["latency_valid_n"] = len(cond_values)
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
        print(f"[Eval] loading checkpoint: {args.checkpoint}", flush=True)
        agent = _load_agent(args.checkpoint, device=args.device)
        print(f"[Eval] checkpoint loaded on {args.device}", flush=True)
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
            from baselines import EFTPolicy
            policy = EFTPolicy(env)
        elif args.policy == "cp_eft":
            from baselines.cp_first_eft_policy import CPFirstEFTPolicy
            policy = CPFirstEFTPolicy(env)
        elif args.policy == "greedy":
            from baselines import GreedyPolicy
            policy = GreedyPolicy(env)
        elif args.policy == "lb_greedy":
            from baselines import LBGreedyPolicy
            policy = LBGreedyPolicy(env)
        elif args.policy == "oracle_min":
            from baselines import OracleMinPolicy
            policy = OracleMinPolicy(env)
        elif args.policy == "static":
            from baselines import StaticPolicy
            policy = StaticPolicy()
        else:
            policy = RandomPolicy(seed=seed)

        for ep in range(args.episodes):
            ep_seed = seed + ep
            ep_start = time.time()
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

            # latency_mean 仅在有完成任务时有物理意义；completed_tasks=0 时存 NaN，避免混入跨策略均值
            latency_raw = float(metrics.get("task_duration_mean", 0.0))
            latency_cond = latency_raw if completed_tasks > 0 else float("nan")
            record = {
                "seed": seed,
                "episode": ep + 1,
                "policy": args.policy,
                "model_tag": "checkpoint" if use_checkpoint else "baseline",
                "policy_mode": "stochastic" if getattr(args, "stochastic", False) else "deterministic",
                "dag_source": args.dag_source,
                "avg_dag_nodes": avg_nodes,
                "terminated": bool(metrics.get("terminated", terminated)),
                "truncated": bool(metrics.get("truncated", truncated)),
                "success_rate": float(metrics.get("task_success_rate", 0.0)),
                "deadline_miss_rate": float(metrics.get("deadline_miss_rate", 0.0)),
                "latency_mean": latency_raw,
                "latency_mean_cond_success": latency_cond,
                "latency_p95": float(metrics.get("task_duration_p95", 0.0)),
                "energy_norm_mean": float(metrics.get("energy_norm_mean", 0.0)),
                "throughput": float(throughput),
                "episode_time_seconds": float(episode_time),
                "completed_tasks": int(completed_tasks),
                "decision_frac_local": float(metrics.get("decision_frac_local", 0.0)),
                "decision_frac_rsu": float(metrics.get("decision_frac_rsu", 0.0)),
                "decision_frac_v2v": float(metrics.get("decision_frac_v2v", 0.0)),
                "avg_rsu_queue": float(metrics.get("avg_rsu_queue", 0.0)),
            }
            all_records.append(record)
            log_interval = max(int(getattr(args, "log_interval", 1)), 1)
            if (ep + 1) % log_interval == 0 or ep == 0 or (ep + 1) == args.episodes:
                print(
                    f"[Eval] seed={seed} ep={ep + 1}/{args.episodes} "
                    f"sr={record['success_rate']:.3f} miss={record['deadline_miss_rate']:.3f} "
                    f"lat={record['latency_mean']:.3f}s done_in={time.time() - ep_start:.1f}s",
                    flush=True,
                )

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
