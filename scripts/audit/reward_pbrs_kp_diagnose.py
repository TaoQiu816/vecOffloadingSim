#!/usr/bin/env python3
"""
PBRS_KP audit script.
Runs episodes and outputs CSV diagnostics for reward consistency.
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


def _parse_args():
    parser = argparse.ArgumentParser(description="PBRS_KP audit diagnosis.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episodes-per-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--t-refs", type=str, default="0.5,0.7")
    parser.add_argument("--dag-source", type=str, default="synthetic_small",
                        choices=["synthetic_small", "synthetic_large", "workflow_json"])
    parser.add_argument("--large-nodes", type=str, default=None)
    parser.add_argument("--workflow-path", type=str, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_dag_source(args):
    Cfg.DAG_SOURCE = args.dag_source
    if args.dag_source == "synthetic_large":
        if args.large_nodes:
            Cfg.DAG_LARGE_NODE_OPTIONS = [int(v.strip()) for v in args.large_nodes.split(",") if v.strip()]
    elif args.dag_source == "workflow_json":
        if not args.workflow_path:
            raise ValueError("--workflow-path is required for workflow_json")
        Cfg.WORKFLOW_JSON_PATH = args.workflow_path
    if args.max_nodes is not None:
        Cfg.MAX_NODES = int(args.max_nodes)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
    }


def _parse_seeds(seed_str: str) -> List[int]:
    if not seed_str:
        return []
    seed_str = seed_str.strip()
    if ".." in seed_str:
        start, end = seed_str.split("..", 1)
        return list(range(int(start), int(end) + 1))
    if "-" in seed_str and "," not in seed_str:
        start, end = seed_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s.strip()) for s in seed_str.split(",") if s.strip()]


def _cohort_stats(records: List[Dict], prefix: str) -> Dict:
    delta_t_abs_norm = [abs(r["delta_t_norm"]) for r in records]
    delta_phi = [r["delta_phi"] for r in records]
    r_total = [r["r_total"] for r in records]
    base_clip_ratio = float(np.mean([r["base_clipped"] for r in records])) if records else 0.0
    shape_clip_ratio = float(np.mean([r["shape_clipped"] for r in records])) if records else 0.0
    total_clip_ratio = float(np.mean([r["total_clipped"] for r in records])) if records else 0.0

    nonzero = [r for r in records if abs(r["delta_t"]) > 1e-9]
    base_clipped_nonzero = float(np.mean([r["base_clipped"] for r in nonzero])) if nonzero else 0.0

    stats = {
        f"{prefix}steps": len(records),
        f"{prefix}r_base_clip_ratio": base_clip_ratio,
        f"{prefix}r_shape_clip_ratio": shape_clip_ratio,
        f"{prefix}r_total_clip_ratio": total_clip_ratio,
        f"{prefix}delta_t_abs_norm_p50": _summarize(delta_t_abs_norm)["p50"],
        f"{prefix}delta_t_abs_norm_p90": _summarize(delta_t_abs_norm)["p90"],
        f"{prefix}delta_t_abs_norm_p95": _summarize(delta_t_abs_norm)["p95"],
        f"{prefix}delta_phi_p50": _summarize(delta_phi)["p50"],
        f"{prefix}delta_phi_p90": _summarize(delta_phi)["p90"],
        f"{prefix}delta_phi_p95": _summarize(delta_phi)["p95"],
        f"{prefix}r_total_mean": _summarize(r_total)["mean"],
        f"{prefix}r_total_p50": _summarize(r_total)["p50"],
        f"{prefix}r_total_p90": _summarize(r_total)["p90"],
        f"{prefix}r_total_p95": _summarize(r_total)["p95"],
        f"{prefix}base_clipped_rate_nonzero": base_clipped_nonzero,
    }
    return stats


def main():
    args = _parse_args()
    _configure_dag_source(args)

    if Cfg.REWARD_SCHEME == "PBRS_KP":
        print(f"[PBRS] reward_gamma={Cfg.REWARD_GAMMA} train_gamma={TC.GAMMA}")
        if abs(Cfg.REWARD_GAMMA - TC.GAMMA) > 1e-9:
            print("[PBRS] Warning: reward_gamma != train_gamma, aligning reward_gamma to train_gamma.")
            Cfg.REWARD_GAMMA = float(TC.GAMMA)

    Cfg.DEBUG_REWARD_ASSERTS = True
    Cfg.DEBUG_PBRS_AUDIT = True
    Cfg.DEBUG_PHI_MONO_PROB = 0.0

    out_dir = args.out_dir or os.path.join("audit_results", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    t_refs = [float(v.strip()) for v in args.t_refs.split(",") if v.strip()]
    base_seeds = _parse_seeds(args.seeds) if args.seeds else []
    if not base_seeds:
        base_seeds = list(range(args.seed, args.seed + max(args.episodes, 1)))
    episodes_per_seed = args.episodes_per_seed
    if episodes_per_seed is None:
        episodes_per_seed = max(int(args.episodes / max(len(base_seeds), 1)), 1)

    for t_ref in t_refs:
        Cfg.T_REF = float(t_ref)
        env = VecOffloadingEnv(config=Cfg)

        episode_rows = []
        record_rows = []
        ep_idx = 0
        for base_seed in base_seeds:
            for rep in range(episodes_per_seed):
                seed = int(base_seed * 1000 + rep)
                _set_seed(seed)
                Cfg.SEED = seed
                obs_list, _ = env.reset(seed=seed)
                policy = RandomPolicy(seed=seed)
                policy.reset()

                done = False
                while not done:
                    actions = policy.select_action(obs_list)
                    obs_list, rewards, terminated, truncated, info = env.step(actions)
                    done = bool(terminated or truncated)

                metrics = info.get("episode_metrics", {})
                records = list(env._pbrs_debug_records)
                record_rows.extend(records)

                records_all = records
                records_decision = [r for r in records_all if r.get("is_decision_step")]
                records_nonzero = [r for r in records_all if abs(r.get("delta_t", 0.0)) > 1e-9]

                no_task_ratio = float(np.mean([r.get("is_no_task_step", False) for r in records_all])) if records_all else 0.0
                decision_ratio = float(np.mean([r.get("is_decision_step", False) for r in records_all])) if records_all else 0.0

                episode_rows.append({
                    "episode": ep_idx + 1,
                    "seed": seed,
                    "base_seed": int(base_seed),
                    "t_ref": float(t_ref),
                    "terminated": bool(metrics.get("terminated", False)),
                    "truncated": bool(metrics.get("truncated", False)),
                    "illegal_count": int(metrics.get("illegal_count", 0)),
                    "no_task_count": int(metrics.get("no_task_count", 0)),
                    "no_task_ratio": no_task_ratio,
                    "decision_step_ratio": decision_ratio,
                    **_cohort_stats(records_all, "all_"),
                    **_cohort_stats(records_decision, "decision_"),
                    **_cohort_stats(records_nonzero, "nonzero_"),
                })
                ep_idx += 1

        env.close()

        episode_csv = os.path.join(out_dir, f"pbrs_episode_stats_Tref{t_ref}.csv")
        with open(episode_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)

        record_csv = os.path.join(out_dir, f"pbrs_step_records_Tref{t_ref}.csv")
        with open(record_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record_rows[0].keys()))
            writer.writeheader()
            writer.writerows(record_rows)

        print(f"[Audit] episode stats: {episode_csv}")
        print(f"[Audit] step records: {record_csv}")


if __name__ == "__main__":
    main()
