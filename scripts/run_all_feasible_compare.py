"""
ALL-feasible vs TopK(K=5) 对照实验。

两组条件：
  C1: CANDIDATE_MODE=TOPK, K=5    → MAX_TARGETS=9
  C2: CANDIDATE_MODE=ALL           → MAX_TARGETS=33

每组 seeds={0,1}, episodes=1500。
输出: logs/all_feasible_compare_summary.csv
"""

import os
import sys
import csv
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PYTHON = sys.executable
TRAIN_SCRIPT = os.path.join(ROOT, "train.py")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CONDITIONS = [
    {
        "name": "TOPK_K5",
        "env_overrides": {
            "CANDIDATE_MODE": "TOPK",
            "TOPK_K": "5",
            "V2V_TOP_K": "5",
        },
    },
    {
        "name": "ALL_FEASIBLE",
        "env_overrides": {
            "CANDIDATE_MODE": "ALL",
        },
    },
]

SEEDS = [0, 1]
MAX_EPISODES = 1500


def run_one(cond_name, seed, env_overrides):
    """运行单个训练条件，返回运行目录"""
    run_id = f"compare_{cond_name}_s{seed}"
    run_dir = os.path.join(ROOT, "runs", run_id)

    env = os.environ.copy()
    for k, v in env_overrides.items():
        env[k] = str(v)

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--max-episodes", str(MAX_EPISODES),
        "--seed", str(seed),
        "--device", "cpu",
        "--run-id", run_id,
        "--run-dir", run_dir,
        "--disable-baseline-eval",
        "--step-metrics",
    ]

    print(f"\n{'='*60}")
    print(f"[{cond_name}] seed={seed} episodes={MAX_EPISODES}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  ENV: {env_overrides}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] exit={result.returncode} ({elapsed:.0f}s)")
        # 打印最后20行stderr
        lines = result.stderr.strip().split("\n")
        for line in lines[-20:]:
            print(f"    {line}")
        return None, elapsed
    else:
        print(f"  [OK] ({elapsed:.0f}s)")
        return run_dir, elapsed


def collect_metrics(run_dir, cond_name, seed):
    """从训练输出目录中读取指标"""
    import numpy as np

    row = {
        "cond": cond_name,
        "seed": seed,
        "episodes": 0,
        "success_rate_mean": 0.0,
        "makespan_mean": 0.0,
        "makespan_p95": 0.0,
        "reward_mean": 0.0,
        "illegal_rate": 0.0,
        "fallback_rate": 0.0,
        "v2v_pick_rate": 0.0,
        "action_entropy": 0.0,
        "feasible_v2v_mean": 0.0,
        "padded_v2v_mean": 0.0,
    }
    if run_dir is None:
        row["episodes"] = -1
        return row

    # 尝试读取 metrics.csv 或 training_stats.csv
    for fname in ["metrics.csv", "training_stats.csv"]:
        metrics_path = os.path.join(run_dir, "logs", fname)
        if os.path.exists(metrics_path):
            break
    else:
        metrics_path = None

    if metrics_path is None:
        row["episodes"] = -2
        return row

    try:
        import pandas as pd
        df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"  [WARN] Failed to read {metrics_path}: {e}")
        row["episodes"] = -3
        return row

    if len(df) == 0:
        return row

    # 取最后20%窗口
    n = len(df)
    win_start = int(n * 0.8)
    win = df.iloc[win_start:]
    row["episodes"] = n

    def safe_mean(col):
        if col in win.columns:
            vals = win[col].dropna()
            return float(vals.mean()) if len(vals) > 0 else 0.0
        return 0.0

    def safe_p95(col):
        if col in win.columns:
            vals = win[col].dropna()
            return float(np.percentile(vals, 95)) if len(vals) > 0 else 0.0
        return 0.0

    # 常见列名映射
    row["success_rate_mean"] = safe_mean("success_rate") or safe_mean("ep_success_rate")
    row["makespan_mean"] = safe_mean("makespan") or safe_mean("ep_makespan")
    row["makespan_p95"] = safe_p95("makespan") or safe_p95("ep_makespan")
    row["reward_mean"] = safe_mean("ep_reward_mean") or safe_mean("reward_mean")
    row["illegal_rate"] = safe_mean("illegal_rate") or safe_mean("ep_illegal_rate")
    row["fallback_rate"] = safe_mean("fallback_rate") or safe_mean("ep_fallback_rate")
    row["v2v_pick_rate"] = safe_mean("v2v_pick_rate") or safe_mean("ep_v2v_pick_rate")
    row["action_entropy"] = safe_mean("target_entropy") or safe_mean("ep_target_entropy")
    row["feasible_v2v_mean"] = safe_mean("feasible_cnt_v2v_mean")
    row["padded_v2v_mean"] = safe_mean("padded_cnt_v2v_mean")

    return row


def main():
    results = []
    total_start = time.time()

    for cond in CONDITIONS:
        for seed in SEEDS:
            run_dir, elapsed = run_one(cond["name"], seed, cond["env_overrides"])
            row = collect_metrics(run_dir, cond["name"], seed)
            row["elapsed_s"] = round(elapsed, 0)
            results.append(row)

    # 写汇总CSV
    out_path = os.path.join(LOG_DIR, "all_feasible_compare_summary.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n汇总写入: {out_path}")

    total_elapsed = time.time() - total_start
    print(f"总耗时: {total_elapsed/60:.1f} min")

    # 打印汇总表
    print(f"\n{'='*80}")
    print("ALL-feasible vs TopK 对照汇总")
    print(f"{'='*80}")
    header = f"{'cond':<18} {'seed':>4} {'ep':>5} {'success':>8} {'makespan':>9} {'mk_p95':>8} {'reward':>8} {'illegal':>8} {'v2v_pick':>9} {'feasible':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['cond']:<18} {r['seed']:>4} {r['episodes']:>5} "
              f"{r['success_rate_mean']:>8.4f} {r['makespan_mean']:>9.3f} {r['makespan_p95']:>8.3f} "
              f"{r['reward_mean']:>8.3f} {r['illegal_rate']:>8.4f} {r['v2v_pick_rate']:>9.4f} "
              f"{r['feasible_v2v_mean']:>9.1f}")


if __name__ == "__main__":
    main()
