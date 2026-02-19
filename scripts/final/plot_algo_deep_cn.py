#!/usr/bin/env python3
"""
Deep plotting and diagnostics for a single algorithm run (Chinese labels).

Input:
- <run_dir>/logs/metrics.csv (preferred) or <run_dir>/metrics/metrics.csv

Output:
- <run_dir>/<out_name>/
  - 12+ figures
  - window/phase/stability CSV summaries
  - markdown report
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_cn_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#f8fafc"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.28


def rolling(s: pd.Series, w: int) -> pd.Series:
    if s is None or s.empty:
        return s
    ww = max(1, int(w))
    return s.rolling(window=ww, min_periods=1).mean()


def load_metrics(run_dir: str) -> pd.DataFrame:
    cands = [
        os.path.join(run_dir, "logs", "metrics.csv"),
        os.path.join(run_dir, "metrics", "metrics.csv"),
    ]
    path = next((p for p in cands if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("未找到 metrics.csv")
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        raise ValueError(f"metrics.csv 缺少 episode 列: {path}")
    return df.sort_values("episode").reset_index(drop=True)


def g(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def save_fig(fig: plt.Figure, out_dir: str, fname: str) -> str:
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(df: pd.DataFrame, out_dir: str, window: int) -> List[Tuple[str, str]]:
    x = df["episode"]
    files: List[Tuple[str, str]] = []

    # 00 单独收敛曲线
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs = axs.flatten()
    axs[0].plot(x, rolling(g(df, "task_success_rate") * 100, window), lw=2.2, label="任务成功率")
    axs[0].plot(x, rolling(g(df, "subtask_success_rate") * 100, window), lw=1.8, label="子任务成功率")
    axs[0].set_title("收敛曲线-成功率")
    axs[0].set_ylabel("%")
    axs[0].legend(fontsize=9)
    axs[1].plot(x, rolling(g(df, "deadline_miss_rate") * 100, window), lw=2.2, color="#d62728", label="超时失败率")
    axs[1].plot(x, rolling(g(df, "time_limit_rate") * 100, window), lw=1.8, color="#ff7f0e", label="回合时限率")
    axs[1].set_title("收敛曲线-失败/时限")
    axs[1].set_ylabel("%")
    axs[1].legend(fontsize=9)
    axs[2].plot(x, rolling(g(df, "reward_mean"), window), lw=2.2, color="#9467bd", label="reward_mean")
    if "reward_p50" in df.columns:
        axs[2].plot(x, rolling(g(df, "reward_p50"), window), lw=1.6, color="#8c564b", label="reward_p50")
    if "reward_p95" in df.columns:
        axs[2].plot(x, rolling(g(df, "reward_p95"), window), lw=1.6, color="#2ca02c", label="reward_p95")
    axs[2].set_title("收敛曲线-奖励")
    axs[2].set_xlabel("Episode")
    axs[2].legend(fontsize=9)
    axs[3].plot(x, rolling(g(df, "mean_cft_est"), window), lw=2.2, color="#1f77b4", label="mean_cft_est")
    axs[3].plot(x, rolling(g(df, "episode_time_seconds"), window), lw=1.8, color="#17becf", label="episode_time_seconds")
    axs[3].set_title("收敛曲线-时延")
    axs[3].set_xlabel("Episode")
    axs[3].set_ylabel("秒")
    axs[3].legend(fontsize=9)
    fig.suptitle("训练收敛曲线（单独图）", fontsize=14, fontweight="bold", y=1.02)
    files.append(("algo_deep_00_convergence_only.png", "训练收敛曲线（单独图）"))
    save_fig(fig, out_dir, files[-1][0])

    # 01 核心收敛
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs = axs.flatten()
    axs[0].plot(x, rolling(g(df, "task_success_rate") * 100, window), lw=2.2)
    axs[0].set_title("任务成功率（滑动均值）")
    axs[0].set_ylabel("%")
    axs[1].plot(x, rolling(g(df, "deadline_miss_rate") * 100, window), lw=2.2, color="#d62728")
    axs[1].set_title("超时失败率（滑动均值）")
    axs[1].set_ylabel("%")
    axs[2].plot(x, rolling(g(df, "mean_cft_est"), window), lw=2.2, color="#2ca02c")
    axs[2].set_title("平均完工时间估计 mean_cft_est")
    axs[2].set_ylabel("秒")
    axs[2].set_xlabel("Episode")
    axs[3].plot(x, rolling(g(df, "reward_mean"), window), lw=2.2, color="#9467bd")
    axs[3].set_title("平均步奖励 reward_mean")
    axs[3].set_xlabel("Episode")
    files.append(("algo_deep_01_core_convergence.png", "核心收敛指标"))
    save_fig(fig, out_dir, files[-1][0])

    # 02 决策占比
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, rolling(g(df, "decision_local_frac") * 100, window), label="本地", lw=2.0)
    ax.plot(x, rolling(g(df, "decision_rsu_frac") * 100, window), label="RSU", lw=2.0)
    ax.plot(x, rolling(g(df, "decision_v2v_frac") * 100, window), label="V2V", lw=2.0)
    ax.set_title("卸载决策占比演化")
    ax.set_xlabel("Episode")
    ax.set_ylabel("%")
    ax.legend()
    files.append(("algo_deep_02_policy_distribution.png", "卸载决策占比"))
    save_fig(fig, out_dir, files[-1][0])

    # 03 奖励分量占比
    fig, ax = plt.subplots(figsize=(13, 5))
    series = [
        ("abs_ratio_r_time", "|r_time|占比"),
        ("abs_ratio_r_energy", "|r_energy|占比"),
        ("abs_ratio_r_interf", "|r_interf|占比"),
        ("abs_ratio_r_risk", "|r_risk|占比"),
        ("abs_ratio_r_illegal", "|r_illegal|占比"),
        ("abs_ratio_r_pbrs", "|r_pbrs|占比"),
        ("abs_ratio_r_term", "|r_term|占比"),
    ]
    for c, lbl in series:
        if c in df.columns:
            ax.plot(x, rolling(g(df, c) * 100, window), lw=1.7, label=lbl)
    ax.set_title("奖励分量绝对占比演化")
    ax.set_xlabel("Episode")
    ax.set_ylabel("%")
    ax.legend(ncol=3, fontsize=9)
    files.append(("algo_deep_03_reward_ratio.png", "奖励分量占比"))
    save_fig(fig, out_dir, files[-1][0])

    # 04 奖励分量原值
    fig, ax = plt.subplots(figsize=(13, 5))
    for c, lbl in [
        ("r_time", "r_time"),
        ("r_energy", "r_energy"),
        ("r_interf", "r_interf"),
        ("r_risk", "r_risk"),
        ("r_illegal", "r_illegal"),
        ("r_pbrs", "r_pbrs"),
        ("r_term", "r_term"),
    ]:
        if c in df.columns:
            ax.plot(x, rolling(g(df, c), window), lw=1.5, label=lbl)
    ax.set_title("奖励分量原值演化")
    ax.set_xlabel("Episode")
    ax.set_ylabel("奖励值")
    ax.legend(ncol=4, fontsize=8)
    files.append(("algo_deep_04_reward_terms.png", "奖励分量原值"))
    save_fig(fig, out_dir, files[-1][0])

    # 05 时延与完成
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(x, rolling(g(df, "mean_cft_est"), window), lw=2.0)
    axs[0].set_title("mean_cft_est")
    axs[1].plot(x, rolling(g(df, "episode_time_seconds"), window), lw=2.0, color="#ff7f0e")
    axs[1].set_title("episode_time_seconds")
    axs[2].plot(x, rolling(g(df, "task_duration_mean"), window), lw=2.0, color="#2ca02c")
    axs[2].set_title("task_duration_mean")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_05_latency_completion.png", "时延与完成相关指标"))
    save_fig(fig, out_dir, files[-1][0])

    # 06 功率与能耗
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(x, rolling(g(df, "power_ratio_mean"), window), lw=2.0)
    axs[0].plot(x, rolling(g(df, "power_ratio_p95"), window), lw=1.8)
    axs[0].set_title("功率比例 mean/p95")
    axs[1].plot(x, rolling(g(df, "energy_norm_mean"), window), lw=2.0, color="#d62728")
    axs[1].plot(x, rolling(g(df, "energy_norm_p95"), window), lw=1.8, color="#8c564b")
    axs[1].set_title("能耗归一化 mean/p95")
    axs[2].plot(x, rolling(g(df, "avg_power"), window), lw=2.0, color="#9467bd")
    axs[2].set_title("avg_power")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_06_power_energy.png", "功率与能耗"))
    save_fig(fig, out_dir, files[-1][0])

    # 07 干扰与外部性
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(x, rolling(g(df, "I_total_mean"), window), lw=2.0)
    axs[0].plot(x, rolling(g(df, "I_total_p95"), window), lw=1.8)
    axs[0].set_title("I_total mean/p95")
    axs[1].plot(x, rolling(g(df, "I_caused_mean"), window), lw=2.0, color="#ff7f0e")
    axs[1].plot(x, rolling(g(df, "I_caused_p95"), window), lw=1.8, color="#d62728")
    axs[1].set_title("I_caused mean/p95")
    axs[2].plot(x, rolling(g(df, "abs_ratio_r_interf") * 100, window), lw=2.0, color="#2ca02c")
    axs[2].set_title("|r_interf|占比(%)")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_07_interference.png", "干扰与外部性"))
    save_fig(fig, out_dir, files[-1][0])

    # 08 信誉与不确定性
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(x, rolling(g(df, "trust_failure_rate") * 100, window), lw=2.0)
    axs[0].set_title("信誉失败率(%)")
    axs[1].plot(x, rolling(g(df, "rho_selected_p10"), window), lw=2.0, color="#1f77b4")
    axs[1].plot(x, rolling(g(df, "rho_selected_p50"), window), lw=1.8, color="#17becf")
    axs[1].set_title("rho_selected p10/p50")
    axs[2].plot(x, rolling(g(df, "uncertainty_selected_p90"), window), lw=2.0, color="#9467bd")
    axs[2].set_title("uncertainty_selected_p90")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_08_trust_uncertainty.png", "信誉与不确定性"))
    save_fig(fig, out_dir, files[-1][0])

    # 09 约束与口径
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(x, rolling(g(df, "illegal_action_rate") * 100, window), lw=2.0)
    axs[0].set_title("illegal_action_rate(%)")
    axs[1].plot(x, rolling(g(df, "no_task_rate") * 100, window), lw=2.0, color="#ff7f0e")
    axs[1].set_title("no_task_rate(%)")
    axs[2].plot(x, rolling(g(df, "unified_illegal_trigger_rate") * 100, window), lw=2.0, color="#d62728")
    axs[2].set_title("unified_illegal_trigger_rate(%)")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_09_constraints.png", "约束与口径"))
    save_fig(fig, out_dir, files[-1][0])

    # 10 候选与可行性诊断
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs = axs.flatten()
    axs[0].plot(x, rolling(g(df, "avail_V") * 100, window), lw=2.0)
    axs[0].set_title("avail_V(%)")
    axs[1].plot(x, rolling(g(df, "neighbor_count_mean"), window), lw=2.0, color="#2ca02c")
    axs[1].set_title("neighbor_count_mean")
    axs[2].plot(x, rolling(g(df, "v2v_beats_rsu_rate") * 100, window), lw=2.0, color="#d62728")
    axs[2].set_title("v2v_beats_rsu_rate(%)")
    axs[3].plot(x, rolling(g(df, "mean_cost_gap_v2v_minus_rsu"), window), lw=2.0, color="#9467bd")
    axs[3].axhline(0.0, color="#333", lw=1.0, alpha=0.7)
    axs[3].set_title("mean_cost_gap_v2v_minus_rsu")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_10_candidate_diagnostics.png", "候选与可行性诊断"))
    save_fig(fig, out_dir, files[-1][0])

    # 11 训练诊断
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()
    pairs = [
        ("entropy", "entropy"),
        ("approx_kl", "approx_kl"),
        ("clip_frac", "clip_frac"),
        ("grad_norm", "grad_norm"),
        ("active_ratio", "active_ratio"),
        ("value_clip_fraction", "value_clip_fraction"),
    ]
    for i, (c, t) in enumerate(pairs):
        axs[i].plot(x, rolling(g(df, c), window), lw=2.0)
        axs[i].set_title(t)
        axs[i].set_xlabel("Episode")
    files.append(("algo_deep_11_training_diagnostics.png", "训练稳定性诊断"))
    save_fig(fig, out_dir, files[-1][0])

    # 12 阶段箱线图
    n = len(df)
    s1 = n // 3
    s2 = (2 * n) // 3
    ph = [
        ("前期", df.iloc[:s1]),
        ("中期", df.iloc[s1:s2]),
        ("后期", df.iloc[s2:]),
    ]
    metrics = [
        ("task_success_rate", "任务成功率"),
        ("deadline_miss_rate", "超时失败率"),
        ("decision_rsu_frac", "RSU占比"),
        ("decision_v2v_frac", "V2V占比"),
        ("trust_failure_rate", "信誉失败率"),
        ("reward_mean", "平均步奖励"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()
    for i, (c, title) in enumerate(metrics):
        vals = [d[c].dropna().to_numpy(dtype=float) for _, d in ph if c in d.columns]
        axs[i].boxplot(vals, tick_labels=[n for n, _ in ph], showfliers=False)
        axs[i].set_title(title)
    files.append(("algo_deep_12_phase_boxplots.png", "分阶段箱线图"))
    save_fig(fig, out_dir, files[-1][0])

    # 13 相关性热图
    cols = [
        "task_success_rate",
        "deadline_miss_rate",
        "decision_rsu_frac",
        "decision_v2v_frac",
        "no_task_rate",
        "mean_cft_est",
        "reward_mean",
        "trust_failure_rate",
        "rho_selected_p10",
        "I_total_p95",
    ]
    use = [c for c in cols if c in df.columns]
    cmat = df[use].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    im = ax.imshow(cmat.values, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(use)))
    ax.set_yticks(range(len(use)))
    ax.set_xticklabels(use, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(use, fontsize=9)
    ax.set_title("关键指标相关性热图")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    files.append(("algo_deep_13_correlation_heatmap.png", "关键指标相关性热图"))
    save_fig(fig, out_dir, files[-1][0])

    # 14 尾段分布
    tail = df.tail(min(200, len(df)))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, (c, title) in enumerate(
        [
            ("task_success_rate", "任务成功率（末200）"),
            ("deadline_miss_rate", "超时失败率（末200）"),
            ("decision_rsu_frac", "RSU占比（末200）"),
            ("decision_v2v_frac", "V2V占比（末200）"),
        ]
    ):
        axs[i].hist(tail[c].dropna().to_numpy(dtype=float), bins=20, alpha=0.85, color="#4c78a8")
        axs[i].set_title(title)
    files.append(("algo_deep_14_tail_hist.png", "尾段分布直方图"))
    save_fig(fig, out_dir, files[-1][0])

    # 15 Deadline 与可行性
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    axs[0].plot(x, rolling(g(df, "deadline_seconds"), window), lw=2.0)
    axs[0].set_title("deadline_seconds")
    axs[1].plot(x, rolling(g(df, "deadline_gamma"), window), lw=2.0, color="#ff7f0e")
    axs[1].set_title("deadline_gamma")
    axs[2].plot(x, rolling(g(df, "time_limit_penalty_applied") * 100, window), lw=2.0, color="#d62728")
    axs[2].set_title("time_limit_penalty_applied(%)")
    axs[3].plot(x, rolling(g(df, "time_limit_penalty_value"), window), lw=2.0, color="#9467bd")
    axs[3].set_title("time_limit_penalty_value")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_15_deadline_feasibility.png", "Deadline与可行性"))
    save_fig(fig, out_dir, files[-1][0])

    # 16 任务负载与稀疏性
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs = axs.flatten()
    axs[0].plot(x, rolling(g(df, "episode_task_count"), window), lw=2.0)
    axs[0].set_title("episode_task_count")
    axs[1].plot(x, rolling(g(df, "total_subtasks"), window), lw=2.0, color="#2ca02c")
    axs[1].set_title("total_subtasks")
    axs[2].plot(x, rolling(g(df, "on_task_rate") * 100, window), lw=2.0, label="on_task_rate")
    axs[2].plot(x, rolling(g(df, "has_task_available_rate") * 100, window), lw=1.8, label="has_task_available_rate")
    axs[2].plot(x, rolling(g(df, "active_ratio") * 100, window), lw=1.8, label="active_ratio")
    axs[2].set_title("任务活跃度(%)")
    axs[2].legend(fontsize=8)
    axs[3].plot(x, rolling(g(df, "no_task_rate") * 100, window), lw=2.0, color="#d62728")
    axs[3].set_title("no_task_rate(%)")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_16_task_activity.png", "任务负载与活跃度"))
    save_fig(fig, out_dir, files[-1][0])

    # 17 队列与负载
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(x, rolling(g(df, "avg_rsu_queue"), window), lw=2.0)
    axs[0].set_title("avg_rsu_queue")
    axs[1].plot(x, rolling(g(df, "rsu_queue_p95"), window), lw=2.0, color="#ff7f0e")
    axs[1].set_title("rsu_queue_p95")
    axs[2].plot(x, rolling(g(df, "queue_lb"), window), lw=2.0, color="#2ca02c")
    axs[2].set_title("queue_lb")
    for ax in axs:
        ax.set_xlabel("Episode")
    files.append(("algo_deep_17_queue_load.png", "队列负载演化"))
    save_fig(fig, out_dir, files[-1][0])

    # 18 奖励分位稳定性
    fig, ax = plt.subplots(figsize=(13, 5))
    for c, lbl, col in [
        ("reward_min", "reward_min", "#d62728"),
        ("reward_p50", "reward_p50", "#1f77b4"),
        ("reward_p95", "reward_p95", "#2ca02c"),
        ("reward_max", "reward_max", "#9467bd"),
    ]:
        if c in df.columns:
            ax.plot(x, rolling(g(df, c), window), lw=1.8, label=lbl, color=col)
    ax.set_title("奖励分位与极值稳定性")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(ncol=4, fontsize=9)
    files.append(("algo_deep_18_reward_quantiles.png", "奖励分位稳定性"))
    save_fig(fig, out_dir, files[-1][0])

    return files


def summarize(df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    windows = [
        ("全程", df),
        ("末500", df.tail(min(500, len(df)))),
        ("末300", df.tail(min(300, len(df)))),
        ("末200", df.tail(min(200, len(df)))),
        ("末100", df.tail(min(100, len(df)))),
        ("末50", df.tail(min(50, len(df)))),
    ]
    cols = [
        "task_success_rate",
        "subtask_success_rate",
        "deadline_miss_rate",
        "time_limit_rate",
        "mean_cft_est",
        "reward_mean",
        "decision_local_frac",
        "decision_rsu_frac",
        "decision_v2v_frac",
        "illegal_action_rate",
        "no_task_rate",
        "unified_illegal_trigger_rate",
        "abs_ratio_r_time",
        "abs_ratio_r_energy",
        "abs_ratio_r_interf",
        "abs_ratio_r_risk",
        "abs_ratio_r_illegal",
        "trust_failure_rate",
        "rho_selected_p10",
        "uncertainty_selected_p90",
        "I_total_p95",
        "I_caused_p95",
        "avail_V",
        "neighbor_count_mean",
        "v2v_beats_rsu_rate",
        "mean_cost_gap_v2v_minus_rsu",
    ]
    rows = []
    for name, d in windows:
        row = {"window": name, "n": int(len(d))}
        for c in cols:
            if c in d.columns:
                row[c] = float(d[c].mean())
        rows.append(row)
    window_df = pd.DataFrame(rows)
    p_window = os.path.join(out_dir, "deep_window_summary.csv")
    window_df.to_csv(p_window, index=False)

    # phase summary
    n = len(df)
    s1 = n // 3
    s2 = (2 * n) // 3
    phases = [("前期", df.iloc[:s1]), ("中期", df.iloc[s1:s2]), ("后期", df.iloc[s2:])]
    rows = []
    for name, d in phases:
        row = {"phase": name, "n": int(len(d))}
        for c in cols:
            if c in d.columns:
                row[c] = float(d[c].mean())
        rows.append(row)
    phase_df = pd.DataFrame(rows)
    p_phase = os.path.join(out_dir, "deep_phase_summary.csv")
    phase_df.to_csv(p_phase, index=False)

    # stability last200
    tail = df.tail(min(200, len(df)))
    s_cols = [
        "task_success_rate",
        "deadline_miss_rate",
        "decision_rsu_frac",
        "decision_v2v_frac",
        "no_task_rate",
        "trust_failure_rate",
        "rho_selected_p10",
        "reward_mean",
        "entropy",
        "approx_kl",
        "clip_frac",
    ]
    stab = {}
    for c in s_cols:
        if c not in tail.columns:
            continue
        m = float(tail[c].mean())
        s = float(tail[c].std())
        stab[f"{c}_mean"] = m
        stab[f"{c}_std"] = s
        stab[f"{c}_cv"] = float(s / (abs(m) + 1e-9))
    p_stab = os.path.join(out_dir, "deep_last200_stability.csv")
    pd.DataFrame([stab]).to_csv(p_stab, index=False)

    # anomaly / sanity checks
    checks = {
        "episodes": int(len(df)),
        "episode_min": int(df["episode"].min()),
        "episode_max": int(df["episode"].max()),
    }
    for c in [
        "r_interf",
        "r_total",
        "reward_mean",
        "task_success_rate",
        "deadline_miss_rate",
        "decision_rsu_frac",
        "decision_v2v_frac",
        "no_task_rate",
        "I_total_p95",
        "I_caused_p95",
    ]:
        if c in df.columns:
            s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) > 0:
                checks[f"{c}_min"] = float(s.min())
                checks[f"{c}_p50"] = float(s.quantile(0.5))
                checks[f"{c}_p95"] = float(s.quantile(0.95))
                checks[f"{c}_max"] = float(s.max())
    p_check = os.path.join(out_dir, "deep_sanity_checks.csv")
    pd.DataFrame([checks]).to_csv(p_check, index=False)

    return {
        "window": p_window,
        "phase": p_phase,
        "stability": p_stab,
        "sanity": p_check,
    }


def write_report(df: pd.DataFrame, out_dir: str) -> str:
    tail = df.tail(min(100, len(df)))
    lines: List[str] = []
    lines.append("# 当前算法深度分析报告")
    lines.append("")
    lines.append("## 关键结论（末100ep均值）")
    for c in [
        "task_success_rate",
        "deadline_miss_rate",
        "time_limit_rate",
        "reward_mean",
        "decision_local_frac",
        "decision_rsu_frac",
        "decision_v2v_frac",
        "illegal_action_rate",
        "no_task_rate",
        "unified_illegal_trigger_rate",
        "mean_cft_est",
        "trust_failure_rate",
        "rho_selected_p10",
        "uncertainty_selected_p90",
        "I_total_p95",
        "I_caused_p95",
        "abs_ratio_r_time",
        "abs_ratio_r_energy",
        "abs_ratio_r_interf",
        "abs_ratio_r_risk",
        "abs_ratio_r_illegal",
    ]:
        if c in tail.columns:
            lines.append(f"- `{c}`: {float(tail[c].mean()):.6g}")
    lines.append("")
    lines.append("## 说明")
    lines.append("- 本报告仅针对当前算法，不包含 baseline 对比。")
    lines.append("- 完整统计见 `deep_window_summary.csv`、`deep_phase_summary.csv`、`deep_last200_stability.csv`。")
    path = os.path.join(out_dir, "deep_report_cn.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="当前算法深度绘图（中文）")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out-name", type=str, default="final_algo_deep_cn")
    ap.add_argument("--window", type=int, default=50, help="rolling window")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.join(run_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    set_cn_style()
    df = load_metrics(run_dir)
    fig_files = plot_all(df, out_dir, args.window)
    csvs = summarize(df, out_dir)
    report = write_report(df, out_dir)

    manifest_rows = [{"file": f, "title": t} for f, t in fig_files]
    for k, p in csvs.items():
        manifest_rows.append({"file": os.path.basename(p), "title": f"{k}统计"})
    manifest_rows.append({"file": os.path.basename(report), "title": "文字报告"})
    pd.DataFrame(manifest_rows).to_csv(os.path.join(out_dir, "deep_figure_manifest_cn.csv"), index=False)

    print(f"[OK] output_dir={out_dir}")
    print(f"[OK] figures={len(fig_files)}")
    print(f"[OK] report={report}")


if __name__ == "__main__":
    main()
