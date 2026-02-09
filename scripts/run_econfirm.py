"""
E-confirm 中长跑实验
验证候选集机制(TOPK K=5 / TOPK K=8 / RANDOMK K=5)对策略的影响

用法:
    python scripts/run_econfirm.py --episodes 1000
    python scripts/run_econfirm.py --episodes 800 --seeds 0 1 2
"""
import argparse
import os
import sys
import json
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import SystemConfig as Cfg
import train


def _recompute_action_space():
    Cfg.MAX_NEIGHBORS = max(0, min(Cfg.NUM_VEHICLES - 1, Cfg.V2V_TOP_K))
    if getattr(Cfg, "ENABLE_RSU_SELECTION", False):
        Cfg.MAX_TARGETS = 1 + Cfg.NUM_RSU + Cfg.MAX_NEIGHBORS
    else:
        Cfg.MAX_TARGETS = 2 + Cfg.MAX_NEIGHBORS


# 实验矩阵: 仅3组条件
CONDITIONS = [
    {"name": "C1_TOPK_K5",   "mode": "TOPK",    "K": 5},
    {"name": "C2_TOPK_K8",   "mode": "TOPK",    "K": 8},
    {"name": "C3_RANDOMK_K5","mode": "RANDOMK",  "K": 5},
]


def _load_metrics(run_dir, tail_pct=0.2):
    """加载训练指标, 取最后tail_pct比例的episode做统计"""
    result = {}
    metrics_csv = os.path.join(run_dir, "metrics", "metrics.csv")
    training_csv = os.path.join(run_dir, "logs", "training_stats.csv")

    # 优先从 metrics.csv 读取（包含更多字段）
    if os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        n = max(1, int(len(df) * tail_pct))
        tail = df.tail(n)
        result["episodes"] = len(df)
        result["final_window_start"] = len(df) - n
        result["final_window_len"] = n
        result["success_rate_mean"] = float(tail["success_rate_end"].mean()) if "success_rate_end" in tail.columns else np.nan
        result["makespan_mean"] = float(tail["mean_cft"].mean()) if "mean_cft" in tail.columns else np.nan
        result["makespan_p95"] = float(tail["mean_cft"].quantile(0.95)) if "mean_cft" in tail.columns else np.nan
        result["reward_mean"] = float(tail["reward_mean"].mean()) if "reward_mean" in tail.columns else np.nan
        result["illegal_rate"] = float(tail["illegal_action_rate"].mean()) if "illegal_action_rate" in tail.columns else np.nan
    elif os.path.exists(training_csv):
        df = pd.read_csv(training_csv)
        n = max(1, int(len(df) * tail_pct))
        tail = df.tail(n)
        result["episodes"] = len(df)
        result["final_window_start"] = len(df) - n
        result["final_window_len"] = n
        result["success_rate_mean"] = float(tail["task_sr"].mean()) if "task_sr" in tail.columns else np.nan
        result["makespan_mean"] = float(tail["task_duration_mean"].mean()) if "task_duration_mean" in tail.columns else np.nan
        result["makespan_p95"] = float(tail["task_duration_p95"].mean()) if "task_duration_p95" in tail.columns else np.nan
        result["reward_mean"] = float(tail["reward_mean"].mean()) if "reward_mean" in tail.columns else np.nan
        result["illegal_rate"] = np.nan
    else:
        result["episodes"] = 0

    # fallback_rate from env_reward.jsonl
    reward_jsonl = os.path.join(run_dir, "logs", "env_reward.jsonl")
    result["fallback_rate"] = np.nan
    if os.path.exists(reward_jsonl):
        records = []
        with open(reward_jsonl) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj and "episode" in obj:
                        records.append(obj)
                except Exception:
                    pass
        if records:
            rdf = pd.DataFrame(records)
            n = max(1, int(len(rdf) * tail_pct))
            tail = rdf.tail(n)
            if "fallback_rate" in tail.columns:
                result["fallback_rate"] = float(tail["fallback_rate"].mean())
    return result


def main():
    parser = argparse.ArgumentParser(description="E-confirm candidate set experiment")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--logdir", type=str, default="logs/econfirm")
    args = parser.parse_args()

    os.environ["DISABLE_AUTO_PLOT"] = "1"

    # 确保 reward scheme 默认
    Cfg.REWARD_SCHEME = "PBRS_KP_V2"
    Cfg.PBRS_PHI_MODE = "STATE_ONLY"
    Cfg.CHAIN_ENABLED = False
    Cfg.CHAIN_MODE = "NONE"

    rows = []
    for cond in CONDITIONS:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"Running: {cond['name']} seed={seed} episodes={args.episodes}")
            print(f"{'='*60}")

            # 设置候选集参数
            Cfg.CANDIDATE_MODE = cond["mode"]
            Cfg.V2V_TOP_K = cond["K"]
            Cfg.TOPK_K = cond["K"]
            Cfg.RANDOMK_K = cond["K"]
            _recompute_action_space()

            run_dir = os.path.join(args.logdir, cond["name"], f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            argv = [
                "train.py",
                "--max-episodes", str(args.episodes),
                "--seed", str(seed),
                "--run-dir", run_dir,
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                train.main()
            except Exception as e:
                print(f"WARNING: {cond['name']} seed={seed} failed: {e}")
            finally:
                sys.argv = old_argv

            # 收集指标（找到实际run_dir，可能带时间戳后缀）
            actual_dirs = sorted([
                os.path.join(args.logdir, cond["name"], d)
                for d in os.listdir(os.path.join(args.logdir, cond["name"]))
                if d.startswith(f"seed_{seed}")
            ])
            actual_dir = actual_dirs[-1] if actual_dirs else run_dir

            metrics = _load_metrics(actual_dir)
            row = {
                "cond": cond["name"],
                "seed": seed,
                "episodes": metrics.get("episodes", 0),
                "success_rate_mean": metrics.get("success_rate_mean", np.nan),
                "makespan_mean": metrics.get("makespan_mean", np.nan),
                "makespan_p95": metrics.get("makespan_p95", np.nan),
                "reward_mean": metrics.get("reward_mean", np.nan),
                "illegal_rate": metrics.get("illegal_rate", np.nan),
                "fallback_rate": metrics.get("fallback_rate", np.nan),
                "final_window_start": metrics.get("final_window_start", 0),
                "final_window_len": metrics.get("final_window_len", 0),
            }
            rows.append(row)
            print(f"  -> {row}")

    # 保存summary
    df = pd.DataFrame(rows)
    os.makedirs(args.logdir, exist_ok=True)
    summary_path = os.path.join(args.logdir, "econfirm_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # 复制到 logs/ 根目录
    import shutil
    shutil.copy(summary_path, "logs/econfirm_summary.csv")
    print(f"Copied to: logs/econfirm_summary.csv")


if __name__ == "__main__":
    main()
