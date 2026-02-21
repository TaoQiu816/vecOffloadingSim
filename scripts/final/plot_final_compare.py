"""
Final comparison plotting for a run directory.

Reads:
- <run_dir>/metrics/train_metrics_full.csv (preferred) or train_metrics.csv
- <run_dir>/logs/baseline_stats.csv (required for baselines)

Writes:
- <run_dir>/<out-name>/  (default: final_plots/)
  - figures (png)
  - summary_metrics.csv (episode-level RL + baseline means)
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_SR_ALIASES: dict = {
    # canonical name -> list of fallback column names to try
    "task_sr": ["task_sr", "task_success_rate", "success_rate"],
    "subtask_sr": ["subtask_sr", "subtask_success_rate"],
    "deadline_miss_rate": ["deadline_miss_rate"],
    "reward_mean": ["reward_mean"],
}


def _resolve_col(df: pd.DataFrame, canonical: str, warn_prefix: str = "") -> Optional[str]:
    """返回 df 中与 canonical 语义等价的第一个列名，找不到返回 None 并可选打印 warning。"""
    for alias in _SR_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            if alias != canonical and warn_prefix:
                print(f"[WARN] {warn_prefix}: using '{alias}' as '{canonical}'")
            return alias
    return None


def _col_or_none(df: pd.DataFrame, canonical: str, warn_prefix: str = "") -> Optional[str]:
    col = _resolve_col(df, canonical, warn_prefix)
    return col


def _is_effectively_zero(df: pd.DataFrame, col: str, tol: float = 1e-6) -> bool:
    """True if col exists but all values are ≈ 0 (disabled metric)."""
    if col not in df.columns:
        return False
    return bool(df[col].abs().max() < tol)


def _rolling(s: pd.Series, window: int) -> pd.Series:
    if s is None or s.empty:
        return s
    w = max(1, min(int(window), max(1, len(s) // 5)))
    return s.rolling(window=w, min_periods=1).mean()


def _load_rl(run_dir: str) -> pd.DataFrame:
    metrics_dir = os.path.join(run_dir, "metrics")
    cand = [
        os.path.join(metrics_dir, "train_metrics_full.csv"),
        os.path.join(metrics_dir, "train_metrics.csv"),
    ]
    path = next((p for p in cand if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("Missing RL metrics: metrics/train_metrics_full.csv or metrics/train_metrics.csv")
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        raise ValueError(f"RL metrics missing 'episode': {path}")
    df = df.sort_values("episode").reset_index(drop=True)
    df.attrs["_path"] = path
    return df


def _load_baselines(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "logs", "baseline_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing baselines: logs/baseline_stats.csv (run scripts/run_baselines.py first)")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("baseline_stats.csv empty")
    if "episode" not in df.columns or "policy" not in df.columns:
        raise ValueError("baseline_stats.csv missing required columns: episode, policy")
    df = df.sort_values(["policy", "episode"]).reset_index(drop=True)
    return df


def _policy_means(df_b: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df_b.columns if c not in ("episode", "policy")]
    out = df_b.groupby("policy")[numeric_cols].mean(numeric_only=True).reset_index()
    return out


def _save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


_BASELINE_COLORS = [
    "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#be185d", "#475569", "#ca8a04",
]


def _plot_with_baselines(
    df_rl: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: str,
    x_col: str,
    y_col_rl: str,
    y_col_b: str,
    title: str,
    ylabel: str,
    window: int = 50,
    percent: bool = False,
):
    # P1: resolve column aliases for RL data (train_metrics may use task_success_rate instead of task_sr)
    actual_rl_col = _col_or_none(df_rl, y_col_rl, warn_prefix=f"plot({title})")
    if actual_rl_col is None:
        actual_rl_col = y_col_rl  # will be skipped below if not in df

    # P4: skip if field is effectively zero in BOTH rl and baseline (disabled metric)
    rl_zero = _is_effectively_zero(df_rl, actual_rl_col)
    bl_zero = all(_is_effectively_zero(df_b[df_b["policy"] == p], y_col_b)
                  for p in df_b["policy"].unique() if y_col_b in df_b.columns)
    if rl_zero and bl_zero and actual_rl_col in df_rl.columns:
        print(f"[SKIP] '{y_col_rl}' is all-zero in RL and baselines (likely disabled). Skipping plot.")
        return

    plt.figure(figsize=(10.5, 4.5))
    x = df_rl[x_col]
    x_min, x_max = int(x.min()), int(x.max())

    if actual_rl_col in df_rl.columns:
        yy = _rolling(df_rl[actual_rl_col], window)
        if percent:
            yy = yy * 100.0
        plt.plot(x, yy, label="Ours (MAPPO)", linewidth=2.2, color="#2563eb", zorder=3)

    # P2: baseline 以全程水平线绘制（baseline固定策略无需显示episode轴演化）
    # 使用各策略在其评估期间的均值，画水平线覆盖全训练x轴，标注"N-ep mean"
    n_bl_ep = int(df_b[x_col].max()) if x_col in df_b.columns else "?"
    for i, policy in enumerate(sorted(df_b["policy"].unique())):
        d = df_b[df_b["policy"] == policy]
        if y_col_b not in d.columns:
            continue
        yb_vals = d[y_col_b].dropna()
        if yb_vals.empty:
            continue
        yb_mean = float(yb_vals.mean())
        if percent:
            yb_mean *= 100.0
        color = _BASELINE_COLORS[i % len(_BASELINE_COLORS)]
        plt.axhline(
            y=yb_mean,
            xmin=0, xmax=1,
            label=f"{policy} ({n_bl_ep}-ep mean)",
            linewidth=1.6,
            linestyle="--",
            color=color,
            alpha=0.85,
        )

    plt.title(title)
    plt.xlabel("Training Episode")
    plt.ylabel(ylabel)
    plt.xlim(x_min, x_max)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    _save_fig(os.path.join(out_dir, f"{y_col_rl}__vs_baselines.png"))


def _plot_rl_only(df_rl: pd.DataFrame, out_dir: str, x_col: str, y_col: str, title: str, ylabel: str, window: int = 50):
    if y_col not in df_rl.columns:
        return
    plt.figure(figsize=(10.5, 4.2))
    plt.plot(df_rl[x_col], _rolling(df_rl[y_col], window), linewidth=2.0, color="#111827")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    _save_fig(os.path.join(out_dir, f"{y_col}.png"))


def main():
    ap = argparse.ArgumentParser(description="Generate final comparison plots into a dedicated folder.")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out-name", type=str, default="final_plots")
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    df_rl = _load_rl(run_dir)
    df_b = _load_baselines(run_dir)

    # P0: 统一评估主表 — 含 eval_source / model_tag / episodes / policy_mode 字段
    # RL mean over all training episodes (last-N-ep mean would be more fair for final perf,
    # but we keep all-ep mean for consistency with existing summary_metrics.csv)
    rl_numeric = [c for c in df_rl.columns if c != "episode"]
    rl_mean = df_rl[rl_numeric].mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})
    rl_mean.insert(0, "source", "Ours (MAPPO)")
    b_mean = _policy_means(df_b)
    b_mean_long = b_mean.melt(id_vars=["policy"], var_name="metric", value_name="mean").rename(columns={"policy": "source"})
    summary = pd.concat([rl_mean, b_mean_long], ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    # P0: 输出规范化比较主表（wide-format，含协议字段，便于答辩/论文引用）
    rl_ep_count = int(df_rl["episode"].max())
    bl_ep_count = int(df_b["episode"].max()) if "episode" in df_b.columns else "?"

    def _metric_val(source_df, col, aliases=None):
        cols = [col] + (aliases or [])
        for c in cols:
            if c in source_df.columns:
                return float(source_df[c].mean())
        return float("nan")

    compare_rows = []
    # MAPPO row
    rl_task_sr_col = _col_or_none(df_rl, "task_sr") or ""
    compare_rows.append({
        "strategy": "Ours (MAPPO)",
        "eval_source": "training_log",
        "model_tag": "best_model",
        "policy_mode": "stochastic",
        "episodes": rl_ep_count,
        "task_success_rate": _metric_val(df_rl, "task_sr", ["task_success_rate", "success_rate"]),
        "subtask_success_rate": _metric_val(df_rl, "subtask_sr", ["subtask_success_rate"]),
        "deadline_miss_rate": _metric_val(df_rl, "deadline_miss_rate"),
        "mean_cft_est": _metric_val(df_rl, "mean_cft_est"),
        "reward_mean": _metric_val(df_rl, "reward_mean"),
    })
    # Baseline rows
    for policy in sorted(df_b["policy"].unique()):
        d = df_b[df_b["policy"] == policy]
        compare_rows.append({
            "strategy": policy,
            "eval_source": "baseline_stats_log",
            "model_tag": "N/A",
            "policy_mode": "deterministic",
            "episodes": bl_ep_count,
            "task_success_rate": _metric_val(d, "task_sr", ["task_success_rate"]),
            "subtask_success_rate": _metric_val(d, "subtask_sr", ["subtask_success_rate"]),
            "deadline_miss_rate": _metric_val(d, "deadline_miss_rate"),
            "mean_cft_est": _metric_val(d, "mean_cft_est"),
            "reward_mean": _metric_val(d, "reward_mean"),
        })
    pd.DataFrame(compare_rows).to_csv(os.path.join(out_dir, "unified_compare_table.csv"), index=False)
    print(f"✓ Unified compare table: {os.path.join(out_dir, 'unified_compare_table.csv')}")

    # Core comparisons (reward/success/latency/constraints)
    _plot_with_baselines(df_rl, df_b, out_dir, "episode", "reward_mean", "reward_mean", "Reward (per step)", "Reward", args.window, percent=False)
    _plot_with_baselines(df_rl, df_b, out_dir, "episode", "task_success_rate", "task_sr", "Task Success Rate", "Task SR (%)", args.window, percent=True)
    _plot_with_baselines(df_rl, df_b, out_dir, "episode", "subtask_success_rate", "subtask_sr", "Subtask Success Rate", "Subtask SR (%)", args.window, percent=True)
    _plot_with_baselines(df_rl, df_b, out_dir, "episode", "time_limit_rate", "time_limit_rate", "Time Limit Rate", "Time limit (%)", args.window, percent=True)
    _plot_with_baselines(df_rl, df_b, out_dir, "episode", "deadline_miss_rate", "deadline_miss_rate", "Deadline Miss Rate", "Deadline miss (%)", args.window, percent=True)

    # Latency-style metrics (if present)
    if "mean_cft_est" in df_rl.columns and "mean_cft_est" in df_b.columns:
        _plot_with_baselines(df_rl, df_b, out_dir, "episode", "mean_cft_est", "mean_cft_est", "Mean CFT (estimated)", "Seconds", args.window, percent=False)
    if "task_duration_p95" in df_rl.columns and "task_duration_p95" in df_b.columns:
        _plot_with_baselines(df_rl, df_b, out_dir, "episode", "task_duration_p95", "task_duration_p95", "Task Duration p95", "Seconds", args.window, percent=False)

    # Power and interference externality
    if "power_ratio_mean" in df_rl.columns and "power_ratio_mean" in df_b.columns:
        _plot_with_baselines(df_rl, df_b, out_dir, "episode", "power_ratio_mean", "power_ratio_mean", "Tx Power Ratio (remote-only)", "a_power", args.window, percent=False)
    if "I_total_p95" in df_rl.columns and "I_total_p95" in df_b.columns:
        _plot_with_baselines(df_rl, df_b, out_dir, "episode", "I_total_p95", "I_total_p95", "Interference Externality (p95)", "Watts (proxy)", args.window, percent=False)

    # Trust oracle and chain proxy (if present)
    if "rho_selected_p10" in df_rl.columns and "rho_selected_p10" in df_b.columns:
        _plot_with_baselines(df_rl, df_b, out_dir, "episode", "rho_selected_p10", "rho_selected_p10", "Reputation Oracle (p10 of selected remote)", "rho (0-1)", args.window, percent=False)
    if "chain_p95_mean" in df_rl.columns and "chain_p95_mean" in df_b.columns:
        _plot_with_baselines(df_rl, df_b, out_dir, "episode", "chain_p95_mean", "chain_p95_mean", "Chain Proxy p95 Confirm Delay", "Seconds", args.window, percent=False)
    if "trust_failure_rate" in df_rl.columns and "trust_failure_rate" in df_b.columns:
        _plot_with_baselines(df_rl, df_b, out_dir, "episode", "trust_failure_rate", "trust_failure_rate", "Trust Failure Rate", "Rate", args.window, percent=False)

    # Reward component dominance (RL only; baselines do not have UNIFIED components)
    for c in (
        "abs_ratio_r_energy",
        "abs_ratio_r_interf",
        "abs_ratio_r_risk",
        "abs_ratio_r_pbrs",
        "abs_ratio_r_term",
    ):
        _plot_rl_only(df_rl, out_dir, "episode", c, f"Component Dominance: {c}", "Abs share", args.window)

    # Small run metadata
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"rl_metrics_path={df_rl.attrs.get('_path','')}\n")
        f.write(f"episodes={int(df_rl['episode'].max())}\n")
        f.write(f"baseline_rows={len(df_b)}\n")

    print(f"✓ Final plots saved: {out_dir}")


if __name__ == "__main__":
    main()

