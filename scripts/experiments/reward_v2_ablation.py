"""
[对比实验] reward_v2_ablation.py
PBRS_KP vs PBRS_KP_V2 对比实验脚本（可复现）

使用方法:
  python scripts/experiments/reward_v2_ablation.py --episodes 300 --seeds 42,43,44 --device cuda
"""

import argparse
import csv
import os
import sys
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_args():
    parser = argparse.ArgumentParser(description="Reward V2 ablation runner")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--last-n", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def _run_train(scheme, seed, episodes, device, base_dir, run_tag, dry_run=False):
    run_dir = os.path.join(base_dir, run_tag)
    env = os.environ.copy()
    env.update({
        "RUN_DIR": run_dir,
        "RUN_ID": run_tag,
        "SEED": str(seed),
        "MAX_EPISODES": str(episodes),
        "REWARD_SCHEME": scheme,
        "DISABLE_BASELINE_EVAL": "1",
        "DISABLE_AUTO_PLOT": "1",
        "LOG_STEP_METRICS": "0",
        "LOG_STEP_LOGS": "0",
    })
    if device:
        env["DEVICE_NAME"] = device
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "train.py"),
        "--max-episodes", str(episodes),
        "--no-step-metrics",
        "--no-step-logs",
    ]
    print(f"[Run] {scheme} seed={seed} -> {run_dir}")
    if dry_run:
        return run_dir
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    return run_dir


def _load_metrics(run_dir):
    path = os.path.join(run_dir, "logs", "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"metrics.csv empty: {path}")
    return df


def _mean_safe(df, col):
    if col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce")
    return float(np.nanmean(vals))


def _summarize_run(df, last_n):
    df = df.sort_values("episode")
    tail = df.tail(last_n) if len(df) >= last_n else df
    summary = {
        "deadline_miss_rate": _mean_safe(tail, "deadline_miss_rate"),
        "task_success_rate": _mean_safe(tail, "task_success_rate"),
        "ratio_local": _mean_safe(tail, "decision_frac_local"),
        "ratio_rsu": _mean_safe(tail, "decision_frac_rsu"),
        "ratio_v2v": _mean_safe(tail, "decision_frac_v2v"),
        "avg_power": _mean_safe(tail, "avg_power"),
        "rsu_queue_p95": _mean_safe(tail, "rsu_queue_p95"),
        "v2v_beats_rsu_rate": _mean_safe(tail, "v2v_beats_rsu_rate"),
        "cost_gap_v2v_minus_rsu": _mean_safe(tail, "mean_cost_gap_v2v_minus_rsu"),
    }
    return summary


def _write_summary_csv(rows, output_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(grouped_runs, output_dir):
    metrics = [
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("task_success_rate", "Task Success Rate"),
        ("decision_frac_local", "Local Ratio"),
        ("v2v_beats_rsu_rate", "V2V Beats RSU Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    window = 20

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        for scheme, runs in grouped_runs.items():
            dfs = []
            for df in runs:
                if metric not in df.columns:
                    continue
                dfs.append(df[["episode", metric]].copy())
            if not dfs:
                continue
            all_df = pd.concat(dfs, ignore_index=True)
            mean_df = all_df.groupby("episode")[metric].mean().reset_index()
            series = mean_df[metric].rolling(window=window, min_periods=1).mean()
            ax.plot(mean_df["episode"], series, label=scheme)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "reward_v2_ablation_curves.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {out_path}")


def main():
    args = _parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    schemes = ["PBRS_KP", "PBRS_KP_V2"]

    if args.output_dir:
        base_dir = os.path.abspath(args.output_dir)
    else:
        base_dir = os.path.join(REPO_ROOT, "runs", f"reward_v2_ablation_{ts}")
    os.makedirs(base_dir, exist_ok=True)

    run_records = []
    grouped_dfs = {scheme: [] for scheme in schemes}

    for scheme in schemes:
        for seed in seeds:
            run_tag = f"{scheme}_seed{seed}_{ts}"
            run_dir = _run_train(scheme, seed, args.episodes, args.device, base_dir, run_tag, dry_run=args.dry_run)
            if args.dry_run:
                continue
            df = _load_metrics(run_dir)
            grouped_dfs[scheme].append(df)
            summary = _summarize_run(df, args.last_n)
            summary_row = {
                "scheme": scheme,
                "seed": seed,
                "run_dir": run_dir,
                "episodes": args.episodes,
                "last_n": args.last_n,
                **summary,
            }
            run_records.append(summary_row)

    if args.dry_run:
        print(f"[Dry-run] Output dir: {base_dir}")
        return

    summary_csv = os.path.join(base_dir, "summary_runs.csv")
    _write_summary_csv(run_records, summary_csv)
    print(f"✓ Summary saved: {summary_csv}")

    # scheme-level aggregation
    agg_rows = []
    if run_records:
        df_summary = pd.DataFrame(run_records)
        for scheme in schemes:
            sub = df_summary[df_summary["scheme"] == scheme]
            if sub.empty:
                continue
            agg = {
                "scheme": scheme,
                "seed_count": len(sub),
            }
            for col in [
                "deadline_miss_rate",
                "task_success_rate",
                "ratio_local",
                "ratio_rsu",
                "ratio_v2v",
                "avg_power",
                "rsu_queue_p95",
                "v2v_beats_rsu_rate",
                "cost_gap_v2v_minus_rsu",
            ]:
                agg[col] = float(np.nanmean(sub[col].values))
            agg_rows.append(agg)
        agg_csv = os.path.join(base_dir, "summary_schemes.csv")
        _write_summary_csv(agg_rows, agg_csv)
        print(f"✓ Aggregated summary saved: {agg_csv}")

    _plot_curves(grouped_dfs, base_dir)


if __name__ == "__main__":
    main()
