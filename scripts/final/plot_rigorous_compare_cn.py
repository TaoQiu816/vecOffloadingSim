#!/usr/bin/env python3
"""
更严谨的 MAPPO vs Baseline 对比（中文图示）

输入:
- <run_dir>/metrics/train_metrics_full.csv (优先) 或 train_metrics.csv
- <run_dir>/logs/baseline_stats.csv

输出:
- <run_dir>/<out_name>/
  - 中文标题图表（全程趋势、阶段对比、多窗口稳健性、配对胜率热图、尾段箱线图）
  - 结构化CSV（窗口统计、阶段统计、配对显著性）
  - 报告（Markdown）
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    rl_col: str
    b_col: str
    title_cn: str
    ylabel_cn: str
    direction: str  # "higher" | "lower"
    scale: float = 1.0
    core: bool = False


METRICS: List[MetricSpec] = [
    MetricSpec("reward_mean", "reward_mean", "reward_mean", "平均步奖励", "奖励", "higher", 1.0, True),
    MetricSpec("task_success_rate", "task_success_rate", "task_sr", "任务成功率", "成功率(%)", "higher", 100.0, True),
    MetricSpec("subtask_success_rate", "subtask_success_rate", "subtask_sr", "子任务成功率", "成功率(%)", "higher", 100.0, False),
    MetricSpec("deadline_miss_rate", "deadline_miss_rate", "deadline_miss_rate", "超时失败率", "比例(%)", "lower", 100.0, True),
    MetricSpec("time_limit_rate", "time_limit_rate", "time_limit_rate", "时间上限截断率", "比例(%)", "lower", 100.0, False),
    MetricSpec("mean_cft_est", "mean_cft_est", "mean_cft_est", "平均完工时间估计", "秒", "lower", 1.0, True),
    MetricSpec("power_ratio_mean", "power_ratio_mean", "power_ratio_mean", "远端功率比例", "a_power", "lower", 1.0, False),
    MetricSpec("rho_selected_p10", "rho_selected_p10", "rho_selected_p10", "远端信誉分位数(p10)", "rho", "higher", 1.0, False),
    MetricSpec("trust_failure_rate", "trust_failure_rate", "trust_failure_rate", "信誉失败率", "比例", "lower", 1.0, False),
]


def _set_cn_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#f8fafc"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["lines.linewidth"] = 2.0


def _rolling(s: pd.Series, window: int) -> pd.Series:
    if s is None or s.empty:
        return s
    w = max(1, min(int(window), max(1, len(s) // 5)))
    return s.rolling(window=w, min_periods=1).mean()


def _load_rl(run_dir: str) -> pd.DataFrame:
    metrics_dir = os.path.join(run_dir, "metrics")
    candidates = [
        os.path.join(metrics_dir, "train_metrics_full.csv"),
        os.path.join(metrics_dir, "train_metrics.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("未找到 RL 指标文件: metrics/train_metrics_full.csv 或 train_metrics.csv")
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        raise ValueError(f"RL 指标缺少 episode 列: {path}")
    df = df.sort_values("episode").reset_index(drop=True)
    df.attrs["_path"] = path
    return df


def _load_baselines(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "logs", "baseline_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("未找到 baseline: logs/baseline_stats.csv")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("baseline_stats.csv 为空")
    if "episode" not in df.columns or "policy" not in df.columns:
        raise ValueError("baseline_stats.csv 缺少必要列: episode, policy")
    return df.sort_values(["policy", "episode"]).reset_index(drop=True)


def _expand_baselines(df_b: pd.DataFrame, max_episode: int) -> pd.DataFrame:
    out: List[pd.DataFrame] = []
    full_idx = pd.Index(range(1, max_episode + 1), name="episode")
    for policy in sorted(df_b["policy"].unique()):
        d = df_b[df_b["policy"] == policy].copy().set_index("episode")
        e = d.reindex(full_idx).ffill().bfill()
        e["policy"] = policy
        out.append(e.reset_index())
    return pd.concat(out, ignore_index=True)


def _metric_available(df_rl: pd.DataFrame, df_b: pd.DataFrame, m: MetricSpec) -> bool:
    return m.rl_col in df_rl.columns and m.b_col in df_b.columns


def _bootstrap_ci_mean(arr: np.ndarray, rng: np.random.Generator, n_boot: int = 1500) -> Tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    samples = arr[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _log_binom_pmf_half(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)


def _binom_two_sided_pvalue(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    log_p_obs = _log_binom_pmf_half(n, k)
    probs = []
    for i in range(n + 1):
        lp = _log_binom_pmf_half(n, i)
        if lp <= log_p_obs + 1e-12:
            probs.append(math.exp(lp))
    p = float(min(1.0, sum(probs)))
    return p


def _phase_ranges(max_episode: int) -> List[Tuple[str, int, int]]:
    s1 = max_episode // 3
    s2 = (max_episode * 2) // 3
    return [
        ("前期", 1, s1),
        ("中期", s1 + 1, s2),
        ("后期", s2 + 1, max_episode),
    ]


def _window_defs(max_episode: int) -> List[Tuple[str, int, int]]:
    return [
        ("全程", 1, max_episode),
        (f"末{min(500, max_episode)}", max(1, max_episode - 500 + 1), max_episode),
        (f"末{min(300, max_episode)}", max(1, max_episode - 300 + 1), max_episode),
        (f"末{min(100, max_episode)}", max(1, max_episode - 100 + 1), max_episode),
    ]


def compute_window_summary(
    df_rl: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: List[MetricSpec],
    windows: List[Tuple[str, int, int]],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    policies = sorted(df_b["policy"].unique())

    for m in metrics:
        if not _metric_available(df_rl, df_b, m):
            continue
        for w_name, s, e in windows:
            r = df_rl[(df_rl["episode"] >= s) & (df_rl["episode"] <= e)]
            rv = (r[m.rl_col].to_numpy(dtype=float) * m.scale)
            lo, hi = _bootstrap_ci_mean(rv, rng)
            rows.append(
                {
                    "window": w_name,
                    "method": "当前算法(MAPPO)",
                    "policy": "MAPPO",
                    "metric_id": m.metric_id,
                    "metric_cn": m.title_cn,
                    "direction": m.direction,
                    "mean": float(np.mean(rv)),
                    "std": float(np.std(rv, ddof=1)) if rv.size > 1 else 0.0,
                    "p10": float(np.percentile(rv, 10)),
                    "p50": float(np.percentile(rv, 50)),
                    "p90": float(np.percentile(rv, 90)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": int(rv.size),
                }
            )
            for p in policies:
                b = df_b[(df_b["policy"] == p) & (df_b["episode"] >= s) & (df_b["episode"] <= e)]
                bv = (b[m.b_col].to_numpy(dtype=float) * m.scale)
                lo, hi = _bootstrap_ci_mean(bv, rng)
                rows.append(
                    {
                        "window": w_name,
                        "method": p,
                        "policy": p,
                        "metric_id": m.metric_id,
                        "metric_cn": m.title_cn,
                        "direction": m.direction,
                        "mean": float(np.mean(bv)),
                        "std": float(np.std(bv, ddof=1)) if bv.size > 1 else 0.0,
                        "p10": float(np.percentile(bv, 10)),
                        "p50": float(np.percentile(bv, 50)),
                        "p90": float(np.percentile(bv, 90)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n": int(bv.size),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["rank"] = np.nan
    for (w, mid), idx in out.groupby(["window", "metric_id"]).groups.items():
        asc = bool(out.loc[idx, "direction"].iloc[0] == "lower")
        out.loc[idx, "rank"] = out.loc[idx, "mean"].rank(method="dense", ascending=asc).to_numpy()
    return out.sort_values(["metric_id", "window", "rank", "method"]).reset_index(drop=True)


def compute_phase_summary(
    df_rl: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: List[MetricSpec],
    phases: List[Tuple[str, int, int]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    policies = sorted(df_b["policy"].unique())

    for m in metrics:
        if not _metric_available(df_rl, df_b, m):
            continue
        for ph_name, s, e in phases:
            r = df_rl[(df_rl["episode"] >= s) & (df_rl["episode"] <= e)]
            rv = r[m.rl_col].to_numpy(dtype=float) * m.scale
            rows.append(
                {
                    "phase": ph_name,
                    "method": "当前算法(MAPPO)",
                    "policy": "MAPPO",
                    "metric_id": m.metric_id,
                    "metric_cn": m.title_cn,
                    "direction": m.direction,
                    "mean": float(np.mean(rv)),
                    "std": float(np.std(rv, ddof=1)) if rv.size > 1 else 0.0,
                    "n": int(rv.size),
                }
            )
            for p in policies:
                b = df_b[(df_b["policy"] == p) & (df_b["episode"] >= s) & (df_b["episode"] <= e)]
                bv = b[m.b_col].to_numpy(dtype=float) * m.scale
                rows.append(
                    {
                        "phase": ph_name,
                        "method": p,
                        "policy": p,
                        "metric_id": m.metric_id,
                        "metric_cn": m.title_cn,
                        "direction": m.direction,
                        "mean": float(np.mean(bv)),
                        "std": float(np.std(bv, ddof=1)) if bv.size > 1 else 0.0,
                        "n": int(bv.size),
                    }
                )
    return pd.DataFrame(rows).sort_values(["metric_id", "phase", "method"]).reset_index(drop=True)


def compute_pairwise_significance(
    df_rl: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: List[MetricSpec],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    policies = sorted(df_b["policy"].unique())
    ep_rl = set(df_rl["episode"].tolist())

    for m in metrics:
        if not _metric_available(df_rl, df_b, m):
            continue
        r = df_rl[["episode", m.rl_col]].dropna()
        for p in policies:
            b = df_b[df_b["policy"] == p][["episode", m.b_col]].dropna()
            common_eps = sorted(ep_rl.intersection(set(b["episode"].tolist())))
            if not common_eps:
                continue
            rr = r[r["episode"].isin(common_eps)].sort_values("episode")[m.rl_col].to_numpy(dtype=float) * m.scale
            bb = b[b["episode"].isin(common_eps)].sort_values("episode")[m.b_col].to_numpy(dtype=float) * m.scale
            if rr.size != bb.size or rr.size == 0:
                continue

            if m.direction == "higher":
                better = rr > bb
                worse = rr < bb
                oriented_diff = rr - bb
            else:
                better = rr < bb
                worse = rr > bb
                oriented_diff = bb - rr

            wins = int(np.sum(better))
            losses = int(np.sum(worse))
            ties = int(rr.size - wins - losses)
            n_eff = wins + losses
            p_value = _binom_two_sided_pvalue(wins, n_eff) if n_eff > 0 else float("nan")
            rows.append(
                {
                    "baseline": p,
                    "metric_id": m.metric_id,
                    "metric_cn": m.title_cn,
                    "direction": m.direction,
                    "n_total": int(rr.size),
                    "n_effective": int(n_eff),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_rate": float(wins / n_eff) if n_eff > 0 else float("nan"),
                    "p_value_sign_test": p_value,
                    "mean_oriented_diff": float(np.mean(oriented_diff)),
                    "median_oriented_diff": float(np.median(oriented_diff)),
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["metric_id", "baseline"]).reset_index(drop=True)


def plot_trends_cn(
    df_rl: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: List[MetricSpec],
    out_dir: str,
    window: int,
) -> str:
    core = [m for m in metrics if m.core and _metric_available(df_rl, df_b, m)]
    if not core:
        return ""
    colors = {
        "MAPPO": "#2563eb",
        "Random": "#ef4444",
        "Local-Only": "#6b7280",
        "Greedy": "#f59e0b",
        "EFT": "#22c55e",
        "CP-EFT": "#0ea5e9",
        "Static": "#7c3aed",
    }
    policies = sorted(df_b["policy"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, m in zip(axes, core[:4]):
        y = _rolling(df_rl[m.rl_col], window) * m.scale
        ax.plot(df_rl["episode"], y, label="当前算法(MAPPO)", color=colors["MAPPO"], linewidth=2.4)
        for p in policies:
            d = df_b[df_b["policy"] == p]
            yb = _rolling(d[m.b_col], window) * m.scale
            ax.plot(d["episode"], yb, label=p, linestyle="--", alpha=0.92, color=colors.get(p, None), linewidth=1.5)
        ax.set_title(f"{m.title_cn}（全程趋势）")
        ax.set_xlabel("Episode")
        ax.set_ylabel(m.ylabel_cn)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"全程对比（滑动平均窗口={window}）", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    path = os.path.join(out_dir, "cn_fig_01_trends_full.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_window_robustness_cn(
    window_summary: pd.DataFrame,
    metrics: List[MetricSpec],
    out_dir: str,
) -> str:
    core = [m for m in metrics if m.core]
    core = [m for m in core if ((window_summary["metric_id"] == m.metric_id).any())][:4]
    if not core:
        return ""
    windows = list(dict.fromkeys(window_summary["window"].tolist()))
    methods = list(dict.fromkeys(window_summary["method"].tolist()))
    method_colors = {
        "当前算法(MAPPO)": "#2563eb",
        "Random": "#ef4444",
        "Local-Only": "#6b7280",
        "Greedy": "#f59e0b",
        "EFT": "#22c55e",
        "CP-EFT": "#0ea5e9",
        "Static": "#7c3aed",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, m in zip(axes, core):
        d = window_summary[window_summary["metric_id"] == m.metric_id]
        for method in methods:
            dm = d[d["method"] == method]
            ys = []
            for w in windows:
                ww = dm[dm["window"] == w]
                ys.append(float(ww["mean"].iloc[0]) if not ww.empty else np.nan)
            ax.plot(windows, ys, marker="o", label=method, color=method_colors.get(method, None), alpha=0.95)
        ax.set_title(f"{m.title_cn}（多窗口稳健性）")
        ax.set_xlabel("统计窗口")
        ax.set_ylabel(m.ylabel_cn)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("不同统计窗口下的结论稳健性", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    path = os.path.join(out_dir, "cn_fig_02_window_robustness.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_phase_cn(
    phase_summary: pd.DataFrame,
    metrics: List[MetricSpec],
    out_dir: str,
) -> str:
    core = [m for m in metrics if m.core]
    core = [m for m in core if ((phase_summary["metric_id"] == m.metric_id).any())][:4]
    if not core:
        return ""
    phases = list(dict.fromkeys(phase_summary["phase"].tolist()))
    methods = list(dict.fromkeys(phase_summary["method"].tolist()))
    method_colors = {
        "当前算法(MAPPO)": "#2563eb",
        "Random": "#ef4444",
        "Local-Only": "#6b7280",
        "Greedy": "#f59e0b",
        "EFT": "#22c55e",
        "CP-EFT": "#0ea5e9",
        "Static": "#7c3aed",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, m in zip(axes, core):
        d = phase_summary[phase_summary["metric_id"] == m.metric_id]
        for method in methods:
            dm = d[d["method"] == method]
            ys = []
            for p in phases:
                pp = dm[dm["phase"] == p]
                ys.append(float(pp["mean"].iloc[0]) if not pp.empty else np.nan)
            ax.plot(phases, ys, marker="o", label=method, color=method_colors.get(method, None), alpha=0.95)
        ax.set_title(f"{m.title_cn}（阶段演化）")
        ax.set_xlabel("训练阶段")
        ax.set_ylabel(m.ylabel_cn)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("分阶段均值对比（前期/中期/后期）", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    path = os.path.join(out_dir, "cn_fig_03_phase_evolution.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_pairwise_heatmap_cn(pair_df: pd.DataFrame, metrics: List[MetricSpec], out_dir: str) -> str:
    if pair_df.empty:
        return ""
    core = [m for m in metrics if m.core]
    core_ids = [m.metric_id for m in core if (pair_df["metric_id"] == m.metric_id).any()]
    if not core_ids:
        core_ids = sorted(pair_df["metric_id"].unique().tolist())
    baselines = sorted(pair_df["baseline"].unique().tolist())

    mat = np.full((len(baselines), len(core_ids)), np.nan, dtype=float)
    pmat = np.full_like(mat, np.nan)
    for i, b in enumerate(baselines):
        for j, mid in enumerate(core_ids):
            d = pair_df[(pair_df["baseline"] == b) & (pair_df["metric_id"] == mid)]
            if d.empty:
                continue
            mat[i, j] = float(d["win_rate"].iloc[0])
            pmat[i, j] = float(d["p_value_sign_test"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(core_ids)))
    cn_lookup = {m.metric_id: m.title_cn for m in metrics}
    ax.set_xticklabels([cn_lookup.get(m, m) for m in core_ids], rotation=15)
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(baselines)
    ax.set_title("逐 Episode 配对胜率热图（当前算法 vs 各 Baseline）")
    for i in range(len(baselines)):
        for j in range(len(core_ids)):
            if np.isnan(mat[i, j]):
                continue
            star = "*" if (not np.isnan(pmat[i, j]) and pmat[i, j] < 0.05) else ""
            ax.text(j, i, f"{mat[i, j]*100:.1f}%{star}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("胜率")
    plt.tight_layout()
    path = os.path.join(out_dir, "cn_fig_04_pairwise_winrate_heatmap.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_tail_box_cn(
    df_rl: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: List[MetricSpec],
    out_dir: str,
    tail_n: int = 300,
) -> str:
    core = [m for m in metrics if m.core and _metric_available(df_rl, df_b, m)][:4]
    if not core:
        return ""
    methods = ["当前算法(MAPPO)"] + sorted(df_b["policy"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for ax, m in zip(axes, core):
        data = []
        labels = []
        rr = (df_rl.tail(tail_n)[m.rl_col].to_numpy(dtype=float) * m.scale)
        data.append(rr)
        labels.append("当前算法")
        for p in sorted(df_b["policy"].unique()):
            bb = (df_b[df_b["policy"] == p].tail(tail_n)[m.b_col].to_numpy(dtype=float) * m.scale)
            data.append(bb)
            labels.append(p)
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"{m.title_cn}（末{tail_n}分布）")
        ax.set_ylabel(m.ylabel_cn)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"末{tail_n} Episode 分布对比（稳定性视角）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "cn_fig_05_tail_boxplots.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def write_report_md(
    out_dir: str,
    window_summary: pd.DataFrame,
    pair_df: pd.DataFrame,
    phases: List[Tuple[str, int, int]],
) -> str:
    report_path = os.path.join(out_dir, "rigorous_report_cn.md")
    lines: List[str] = []
    lines.append("# 严谨对比报告（中文）")
    lines.append("")
    lines.append("## 方法说明")
    lines.append("- 使用全程数据（非仅末段）进行趋势分析。")
    lines.append("- 同时报告多窗口统计（全程/末500/末300/末100）评估结论稳健性。")
    lines.append("- 采用逐 Episode 配对比较（Sign Test）给出胜率与显著性。")
    lines.append(f"- 训练阶段划分: {', '.join([f'{name}[{s}-{e}]' for name, s, e in phases])}")
    lines.append("")

    if not window_summary.empty:
        lines.append("## 末100关键指标排名（均值）")
        tail = window_summary[window_summary["window"].str.contains("末100", na=False)].copy()
        for mid in ["reward_mean", "task_success_rate", "deadline_miss_rate", "mean_cft_est"]:
            d = tail[tail["metric_id"] == mid].sort_values("rank")
            if d.empty:
                continue
            metric_cn = d["metric_cn"].iloc[0]
            lines.append(f"### {metric_cn}")
            for _, r in d.iterrows():
                lines.append(f"- {r['method']}: mean={r['mean']:.4f}, rank={int(r['rank'])}")
            lines.append("")

    if not pair_df.empty:
        lines.append("## 配对检验（当前算法 vs Baseline）")
        core = pair_df[pair_df["metric_id"].isin(["reward_mean", "task_success_rate", "deadline_miss_rate", "mean_cft_est"])].copy()
        core = core.sort_values(["metric_id", "baseline"])
        for mid, g in core.groupby("metric_id"):
            metric_cn = g["metric_cn"].iloc[0]
            lines.append(f"### {metric_cn}")
            for _, r in g.iterrows():
                ptxt = "显著" if (pd.notna(r["p_value_sign_test"]) and r["p_value_sign_test"] < 0.05) else "不显著"
                lines.append(
                    f"- vs {r['baseline']}: 胜率={r['win_rate']*100:.2f}%, p={r['p_value_sign_test']:.3e} ({ptxt})"
                )
            lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="更合理严谨的 MAPPO vs Baseline 对比（中文图示）")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out-name", type=str, default="final_rigorous_cn")
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()

    _set_cn_style()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    df_rl = _load_rl(run_dir)
    df_b_raw = _load_baselines(run_dir)
    max_ep = int(df_rl["episode"].max())
    df_b = _expand_baselines(df_b_raw, max_ep)

    metrics = [m for m in METRICS if _metric_available(df_rl, df_b, m)]
    windows = _window_defs(max_ep)
    phases = _phase_ranges(max_ep)

    window_summary = compute_window_summary(df_rl, df_b, metrics, windows)
    phase_summary = compute_phase_summary(df_rl, df_b, metrics, phases)
    pair_df = compute_pairwise_significance(df_rl, df_b, metrics)

    fig_paths = []
    p = plot_trends_cn(df_rl, df_b, metrics, out_dir, args.window)
    if p:
        fig_paths.append(("cn_fig_01_trends_full.png", "全程趋势对比（核心指标）"))
    p = plot_window_robustness_cn(window_summary, metrics, out_dir)
    if p:
        fig_paths.append(("cn_fig_02_window_robustness.png", "多窗口稳健性对比"))
    p = plot_phase_cn(phase_summary, metrics, out_dir)
    if p:
        fig_paths.append(("cn_fig_03_phase_evolution.png", "分阶段演化对比"))
    p = plot_pairwise_heatmap_cn(pair_df, metrics, out_dir)
    if p:
        fig_paths.append(("cn_fig_04_pairwise_winrate_heatmap.png", "逐Episode配对胜率热图"))
    p = plot_tail_box_cn(df_rl, df_b, metrics, out_dir, tail_n=min(300, max_ep))
    if p:
        fig_paths.append(("cn_fig_05_tail_boxplots.png", "尾段分布与稳定性对比"))

    window_summary.to_csv(os.path.join(out_dir, "rigorous_window_summary.csv"), index=False)
    phase_summary.to_csv(os.path.join(out_dir, "rigorous_phase_summary.csv"), index=False)
    pair_df.to_csv(os.path.join(out_dir, "rigorous_pairwise_significance.csv"), index=False)
    pd.DataFrame(fig_paths, columns=["file", "title_cn"]).to_csv(
        os.path.join(out_dir, "rigorous_figure_manifest_cn.csv"), index=False
    )
    report_path = write_report_md(out_dir, window_summary, pair_df, phases)

    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"rl_metrics_path={df_rl.attrs.get('_path', '')}\n")
        f.write(f"episodes={max_ep}\n")
        f.write(f"baseline_rows={len(df_b_raw)}\n")
        f.write(f"window={args.window}\n")
        f.write(f"report={report_path}\n")

    print(f"✓ 输出目录: {out_dir}")
    print(f"✓ 可用指标数: {len(metrics)}")
    print(f"✓ 图表数: {len(fig_paths)}")


if __name__ == "__main__":
    main()
