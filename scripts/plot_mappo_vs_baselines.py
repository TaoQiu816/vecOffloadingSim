"""
[绘图脚本] plot_mappo_vs_baselines.py
绘制 MAPPO 与 Baselines 的综合对比图（基于训练指标 + baseline_stats.csv）

使用方法:
  python scripts/plot_mappo_vs_baselines.py --run-dir runs/run_YYYYMMDD_HHMMSS
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


BASELINE_POLICIES = ["Local-Only", "Greedy", "EFT", "CP-EFT"]
NON_NUMERIC_COLUMNS = {"policy", "termination_reason", "termination_reason_raw", "termination_reason_bucket", "abs_ratio_basis"}


def _rolling(series, window=50):
    if series is None or series.empty:
        return series
    w = min(window, max(1, len(series) // 5))
    return series.rolling(window=w, min_periods=1).mean()


def _coerce_numeric_columns(df):
    for col in df.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null = int(df[col].notna().sum())
        if non_null == 0:
            continue
        if int(converted.notna().sum()) >= max(1, int(non_null * 0.95)):
            df[col] = converted
    return df


def _load_csv_clean(path, kind):
    df = pd.read_csv(path, dtype=str)
    if "episode" not in df.columns:
        raise ValueError(f"{path} 缺少 episode 列")
    header_rows = df["episode"].astype(str).str.strip().eq("episode")
    if "policy" in df.columns:
        header_rows |= df["policy"].astype(str).str.strip().eq("policy")
    df = df.loc[~header_rows].copy()
    df = _coerce_numeric_columns(df)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df[df["episode"].notna()].copy()
    df["episode"] = df["episode"].astype(int)
    if kind == "baseline":
        if "policy" not in df.columns:
            raise ValueError(f"{path} 缺少 policy 列")
        df["policy"] = df["policy"].fillna("").astype(str)
        df = df[df["policy"].isin(BASELINE_POLICIES)].copy()
        df = df.sort_values(["policy", "episode"]).drop_duplicates(["policy", "episode"], keep="last")
    else:
        df = df.sort_values("episode").drop_duplicates(["episode"], keep="last")
    return df.reset_index(drop=True)


def _load_mappo(run_dir):
    candidates = [
        os.path.join(run_dir, "logs", "metrics.csv"),
        os.path.join(run_dir, "metrics", "train_metrics.csv"),
        os.path.join(run_dir, "logs", "training_stats.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("metrics.csv / train_metrics.csv / training_stats.csv 不存在")
    return _load_csv_clean(path, "mappo")


def _load_baselines(run_dir, baseline_csv=None):
    baseline_path = os.path.abspath(baseline_csv) if baseline_csv else os.path.join(run_dir, "logs", "baseline_stats.csv")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError("baseline_stats.csv 不存在，请先运行 scripts/run_baselines.py")
    df = _load_csv_clean(baseline_path, "baseline")
    if df.empty:
        raise ValueError("baseline_stats.csv 为空")
    return df


def _expand_baselines(df_baseline, max_episode):
    expanded = []
    for policy in df_baseline["policy"].unique():
        policy_data = df_baseline[df_baseline["policy"] == policy].copy().set_index("episode")
        full_idx = pd.Index(range(1, max_episode + 1), name="episode")
        policy_expanded = policy_data.reindex(full_idx).ffill().bfill().infer_objects(copy=False)
        policy_expanded["policy"] = policy
        expanded.append(policy_expanded.reset_index())
    return pd.concat(expanded, ignore_index=True)


def _plot_line(ax, x, y, label, color, linestyle="-", alpha=0.9):
    if y is None:
        return
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=2)


def _pick_col(df, *candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _policy_series(df, policy, col):
    if col is None or col not in df.columns:
        return None
    return df[df["policy"] == policy][col]


def main():
    parser = argparse.ArgumentParser(description="Plot MAPPO vs Baselines")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--baseline-csv", type=str, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(run_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    df_mappo = _load_mappo(run_dir)
    df_baseline_raw = _load_baselines(run_dir, args.baseline_csv)

    max_episode = int(df_mappo["episode"].max())
    df_baseline = _expand_baselines(df_baseline_raw, max_episode)

    reward_col = _pick_col(df_mappo, "reward_mean")
    task_sr_col = _pick_col(df_mappo, "task_success_rate", "task_sr")
    subtask_sr_col = _pick_col(df_mappo, "subtask_success_rate", "subtask_sr")
    local_col = _pick_col(df_mappo, "decision_frac_local", "ratio_local")
    rsu_col = _pick_col(df_mappo, "decision_frac_rsu", "ratio_rsu")
    v2v_col = _pick_col(df_mappo, "decision_frac_v2v", "ratio_v2v")
    power_col = _pick_col(df_mappo, "avg_power", "energy_mean")
    rsu_queue_col = _pick_col(df_mappo, "avg_rsu_queue")

    colors = {
        "MAPPO": "#2563eb",
        "Local-Only": "#95a5a6",
        "Greedy": "#f39c12",
        "EFT": "#16a34a",
        "CP-EFT": "#0ea5e9",
    }

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes = axes.flatten()

    ax = axes[0]
    _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[reward_col], args.window), "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "reward_mean")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window), policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Reward (per step)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[task_sr_col], args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "task_sr")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window) * 100, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Task Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Task SR (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[subtask_sr_col], args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "subtask_sr")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window) * 100, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Subtask Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Subtask SR (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    if power_col is not None:
        _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[power_col], args.window), "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "avg_power")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window), policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Avg Power / Energy")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[local_col], args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "ratio_local")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window) * 100, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("Local Decision Ratio (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Local (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[rsu_col], args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "ratio_rsu")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window) * 100, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("RSU Decision Ratio (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("RSU (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[6]
    _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[v2v_col], args.window) * 100, "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "ratio_v2v")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window) * 100, policy, colors.get(policy, "gray"), "--", 0.85)
    ax.set_title("V2V Decision Ratio (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("V2V (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[7]
    if rsu_queue_col is not None:
        _plot_line(ax, df_mappo["episode"], _rolling(df_mappo[rsu_queue_col], args.window), "MAPPO", colors["MAPPO"])
    for policy in df_baseline["policy"].unique():
        y = _policy_series(df_baseline, policy, "avg_rsu_queue")
        _plot_line(ax, df_baseline[df_baseline["policy"] == policy]["episode"], _rolling(y, args.window), policy, colors.get(policy, "gray"), "--", 0.85)
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
