"""
[绘图脚本] plot_mappo_vs_baselines.py
绘制 MAPPO 与 Baselines 的综合对比图（基于 training/metrics + baseline_stats.csv）

使用方法:
  python scripts/plot_mappo_vs_baselines.py --run-dir runs/run_YYYYMMDD_HHMMSS
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _rolling(series, window=50):
    if series is None or series.empty:
        return series
    w = min(window, max(1, len(series) // 5))
    return series.rolling(window=w, min_periods=1).mean()


def _load_mappo(run_dir):
    metrics_path = os.path.join(run_dir, "logs", "metrics.csv")
    train_path = os.path.join(run_dir, "logs", "training_stats.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        return df, "metrics"
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        return df, "training"
    raise FileNotFoundError("metrics.csv 或 training_stats.csv 不存在")


def _load_baselines(run_dir):
    baseline_path = os.path.join(run_dir, "logs", "baseline_stats.csv")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError("baseline_stats.csv 不存在，请先运行 scripts/run_baselines.py")
    df = pd.read_csv(baseline_path)
    if df.empty:
        raise ValueError("baseline_stats.csv 为空")
    return df


def _expand_baselines(df_baseline, max_episode):
    expanded = []
    for policy in df_baseline["policy"].unique():
        policy_data = df_baseline[df_baseline["policy"] == policy].copy()
        policy_data = policy_data.set_index("episode")
        full_idx = pd.Index(range(1, max_episode + 1), name="episode")
        policy_expanded = policy_data.reindex(full_idx).ffill().bfill()
        policy_expanded["policy"] = policy
        policy_expanded = policy_expanded.reset_index()
        expanded.append(policy_expanded)
    return pd.concat(expanded, ignore_index=True)


def _plot_line(ax, x, y, label, color, linestyle="-", alpha=0.9):
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=2)


def main():
    parser = argparse.ArgumentParser(description="Plot MAPPO vs Baselines")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(run_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    df_mappo, source = _load_mappo(run_dir)
    df_baseline_raw = _load_baselines(run_dir)

    if "episode" not in df_mappo.columns:
        raise ValueError("MAPPO 数据缺少 episode 列")
    max_episode = int(df_mappo["episode"].max())
    df_baseline = _expand_baselines(df_baseline_raw, max_episode)

    # MAPPO 字段映射
    if source == "metrics":
        m_reward = df_mappo["reward_mean"]
        m_task_sr = df_mappo["task_success_rate"]
        m_subtask_sr = df_mappo["subtask_success_rate"]
        m_local = df_mappo["decision_frac_local"]
        m_rsu = df_mappo["decision_frac_rsu"]
        m_v2v = df_mappo["decision_frac_v2v"]
        m_power = df_mappo["avg_power"] if "avg_power" in df_mappo.columns else None
        m_rsu_queue = df_mappo["avg_rsu_queue"] if "avg_rsu_queue" in df_mappo.columns else None
    else:
        m_reward = df_mappo["reward_mean"]
        m_task_sr = df_mappo["task_sr"]
        m_subtask_sr = df_mappo["subtask_sr"]
        m_local = df_mappo["ratio_local"]
        m_rsu = df_mappo["ratio_rsu"]
        m_v2v = df_mappo["ratio_v2v"]
        m_power = df_mappo["energy_mean"] if "energy_mean" in df_mappo.columns else None
        m_rsu_queue = None

    # 颜色
    colors = {
        "MAPPO": "#2563eb",
        "Random": "#e74c3c",
        "Local-Only": "#95a5a6",
        "Greedy": "#f39c12",
        "EFT": "#16a34a",
        "Static": "#7c3aed",
    }

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()

    # 1. Reward
    ax = axes[0]
    _plot_line(ax, df_mappo["episode"], _rolling(m_reward, args.window), "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _rolling(df_baseline[df_baseline["policy"] == policy]["reward_mean"], args.window)
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Reward (per step)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Task SR
    ax = axes[1]
    _plot_line(ax, df_mappo["episode"], _rolling(m_task_sr, args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _rolling(df_baseline[df_baseline["policy"] == policy]["task_sr"], args.window) * 100
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Task Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Task SR (%)")
    ax.grid(True, alpha=0.3)

    # 3. Subtask SR
    ax = axes[2]
    _plot_line(ax, df_mappo["episode"], _rolling(m_subtask_sr, args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _rolling(df_baseline[df_baseline["policy"] == policy]["subtask_sr"], args.window) * 100
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Subtask Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Subtask SR (%)")
    ax.grid(True, alpha=0.3)

    # 4. Avg Power / Energy
    ax = axes[3]
    if m_power is not None:
        _plot_line(ax, df_mappo["episode"], _rolling(m_power, args.window), "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        if "avg_power" in df_baseline.columns:
            y = _rolling(df_baseline[df_baseline["policy"] == policy]["avg_power"], args.window)
            _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Avg Power (Offload)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)

    # 5. Local ratio
    ax = axes[4]
    _plot_line(ax, df_mappo["episode"], _rolling(m_local, args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _rolling(df_baseline[df_baseline["policy"] == policy]["ratio_local"], args.window) * 100
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Local Decision Ratio (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Local (%)")
    ax.grid(True, alpha=0.3)

    # 6. RSU ratio
    ax = axes[5]
    _plot_line(ax, df_mappo["episode"], _rolling(m_rsu, args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _rolling(df_baseline[df_baseline["policy"] == policy]["ratio_rsu"], args.window) * 100
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("RSU Decision Ratio (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("RSU (%)")
    ax.grid(True, alpha=0.3)

    # 7. V2V ratio
    ax = axes[6]
    _plot_line(ax, df_mappo["episode"], _rolling(m_v2v, args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _rolling(df_baseline[df_baseline["policy"] == policy]["ratio_v2v"], args.window) * 100
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("V2V Decision Ratio (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("V2V (%)")
    ax.grid(True, alpha=0.3)

    # 8. RSU queue
    ax = axes[7]
    if m_rsu_queue is not None:
        _plot_line(ax, df_mappo["episode"], _rolling(m_rsu_queue, args.window), "MAPPO", colors["MAPPO"])
    if "avg_rsu_queue" in df_baseline.columns:
        for policy in df_baseline["policy"].unique():
            y = _rolling(df_baseline[df_baseline["policy"] == policy]["avg_rsu_queue"], args.window)
            _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], y, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Avg RSU Queue Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Queue")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_mappo_vs_baselines.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
