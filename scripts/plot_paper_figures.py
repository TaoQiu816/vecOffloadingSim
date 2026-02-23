"""
论文级可视化脚本 - VEC DAG任务卸载系统
生成6张符合 IEEE/ACM 顶刊规范的矢量图

用法:
    python scripts/plot_paper_figures.py
    python scripts/plot_paper_figures.py --run-dir runs/run_YYYYMMDD_HHMMSS
    python scripts/plot_paper_figures.py --run-dir runs/run_X --baseline-json runs/deadline_calib_baselines_10ep/baseline_comparison.json
    python scripts/plot_paper_figures.py --out-dir paper_figs/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ── 字体与样式 ─────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # TrueType embedding for IEEE PDF
    "ps.fonttype": 42,
})

# ── 配色方案（符合学术图规范） ──────────────────────────────────────────────────
COLORS = {
    "MAPPO":        "#D62728",   # 红 - 提出方法（突出）
    "Greedy":       "#1F77B4",   # 蓝
    "EFT":          "#FF7F0E",   # 橙
    "Local-Only":   "#2CA02C",   # 绿
    "Random":       "#7F7F7F",   # 灰
}
MARKERS = {"MAPPO": "s", "Greedy": "o", "EFT": "^", "Local-Only": "D", "Random": "x"}
HATCHES = {"MAPPO": "", "Greedy": "//", "EFT": "\\\\", "Local-Only": "xx", "Random": ".."}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_run(root: Path) -> Path:
    candidates = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under {root}")
    return candidates[0]


def smooth(series: np.ndarray, window: int = 20) -> np.ndarray:
    """指数移动平均平滑"""
    alpha = 2.0 / (window + 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def canonical_name(raw: str) -> str:
    """将 eval_baselines 输出的 policy_name 统一化"""
    mapping = {
        "random policy":    "Random",
        "local-only policy":"Local-Only",
        "local only policy":"Local-Only",
        "greedy policy":    "Greedy",
        "eft policy":       "EFT",
        "mappo":            "MAPPO",
    }
    return mapping.get(raw.lower().strip(), raw)


def load_training_csv(run_dir: Path) -> pd.DataFrame:
    for candidate in ["logs/training_stats.csv", "logs/metrics.csv", "episode_log.csv"]:
        p = run_dir / candidate
        if p.exists():
            df = pd.read_csv(p)
            return df
    raise FileNotFoundError(f"No training CSV found in {run_dir}")


def load_baselines(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    for d in data:
        d["_name"] = canonical_name(d.get("policy_name", ""))
    return data


def inject_mappo_row(baselines: list[dict], df: pd.DataFrame) -> list[dict]:
    """从训练末尾 10% episodes 估算 MAPPO 的评估指标并注入到基准列表"""
    tail = df.tail(max(1, len(df) // 10))
    mappo = {
        "_name": "MAPPO",
        "policy_name": "MAPPO (Ours)",
        "avg_reward": float(tail["reward_mean"].mean()),
        "std_reward": float(tail["reward_mean"].std()),
        "avg_vehicle_success_rate": float(tail["vehicle_sr"].mean()) if "vehicle_sr" in tail else 0.0,
        "avg_subtask_success_rate": float(tail["subtask_sr"].mean()) if "subtask_sr" in tail else 0.0,
        "deadline_meet_ratio": float(1.0 - tail["deadline_miss_rate"].mean()) if "deadline_miss_rate" in tail else 0.0,
        "avg_energy_consumption": float(tail["energy_mean"].mean()) if "energy_mean" in tail else 0.0,
        "std_energy_consumption": float(tail["energy_mean"].std()) if "energy_mean" in tail else 0.0,
        "decision_distribution": {
            "local": float(tail["ratio_local"].mean()) if "ratio_local" in tail else 0.33,
            "rsu":   float(tail["ratio_rsu"].mean())   if "ratio_rsu" in tail else 0.33,
            "v2v":   float(tail["ratio_v2v"].mean())   if "ratio_v2v" in tail else 0.33,
        },
    }
    existing_names = {d["_name"] for d in baselines}
    if "MAPPO" not in existing_names:
        baselines.insert(0, mappo)
    return baselines


# ══════════════════════════════════════════════════════════════════════════════
# 图 1: 训练收敛曲线
# ══════════════════════════════════════════════════════════════════════════════

def fig_convergence(df: pd.DataFrame, out_dir: Path, baselines: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    eps = df["episode"].values
    rw  = df["reward_mean"].values
    rw_s = smooth(rw, 30)

    # ── 左: 奖励曲线 ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps, rw,   color=COLORS["MAPPO"], alpha=0.18, linewidth=0.6)
    ax.plot(eps, rw_s, color=COLORS["MAPPO"], linewidth=1.6, label="MAPPO (ours)")

    # 绘制基准水平线（最后评估值）
    for b in baselines:
        name = b["_name"]
        if name == "MAPPO":
            continue
        val = b.get("avg_reward", None)
        if val is not None:
            c = COLORS.get(name, "#888888")
            ax.axhline(val, linestyle=":", linewidth=1.0, color=c, alpha=0.8, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("(a) Training Convergence")
    ax.legend(loc="lower right", framealpha=0.7)

    # ── 右: 任务成功率 ────────────────────────────────────────────────────────
    ax = axes[1]
    for col, label, ls in [
        ("vehicle_sr",  "Vehicle SR",  "-"),
        ("task_sr",     "Task SR",     "--"),
        ("subtask_sr",  "Subtask SR",  "-."),
    ]:
        if col not in df.columns:
            continue
        vals = df[col].values
        ax.plot(eps, smooth(vals, 30), linewidth=1.4, linestyle=ls, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("(b) Success Rate Progression")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower right", framealpha=0.7)

    plt.tight_layout(pad=0.8)
    out = out_dir / "fig1_convergence.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig1] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 图 2: 算法对比柱状图（4 指标）
# ══════════════════════════════════════════════════════════════════════════════

def fig_algorithm_comparison(baselines: list[dict], out_dir: Path):
    # 排序：MAPPO 最后（最右），突出
    ORDER = ["Random", "Local-Only", "EFT", "Greedy", "MAPPO"]
    bl = {b["_name"]: b for b in baselines}
    names = [n for n in ORDER if n in bl]
    if not names:
        print("[fig2] no data, skip")
        return

    metrics = [
        ("avg_reward",              "Avg. Reward",           False),
        ("avg_vehicle_success_rate","Vehicle Success Rate",  True),
        ("deadline_meet_ratio",     "Deadline Meet Ratio",   True),
        ("avg_energy_consumption",  "Avg. Energy (J)",       False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.6))

    for ax, (key, ylabel, is_rate) in zip(axes, metrics):
        vals  = [bl[n].get(key, 0.0) for n in names]
        stds_key = key.replace("avg_", "std_")
        errs  = [bl[n].get(stds_key, 0.0) for n in names]
        cols  = [COLORS.get(n, "#AAAAAA") for n in names]
        hatch = [HATCHES.get(n, "") for n in names]

        xs = np.arange(len(names))
        for i, (v, e, c, h) in enumerate(zip(vals, errs, cols, hatch)):
            bar = ax.bar(
                xs[i], v, yerr=e, capsize=3, width=0.6,
                color=c, edgecolor="black", linewidth=0.5,
                hatch=h, alpha=0.88,
                error_kw={"elinewidth": 0.8, "ecolor": "black"},
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        if is_rate:
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax.set_title(ylabel, fontsize=8, pad=3)

    # 全局图例
    patches = [mpatches.Patch(facecolor=COLORS.get(n, "#AAA"), edgecolor="black",
                               hatch=HATCHES.get(n, ""), label=n) for n in names]
    fig.legend(handles=patches, loc="lower center", ncol=len(names),
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8)
    fig.suptitle("Algorithm Comparison", fontsize=11, y=1.01)
    plt.tight_layout(pad=0.6)
    out = out_dir / "fig2_algorithm_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig2] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 图 3: 卸载决策演化（训练过程）
# ══════════════════════════════════════════════════════════════════════════════

def fig_offloading_evolution(df: pd.DataFrame, out_dir: Path):
    needed = ["ratio_local", "ratio_rsu", "ratio_v2v"]
    if not all(c in df.columns for c in needed):
        print("[fig3] missing offloading ratio columns, skip")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    eps = df["episode"].values
    loc = smooth(df["ratio_local"].values, 20)
    rsu = smooth(df["ratio_rsu"].values, 20)
    v2v = smooth(df["ratio_v2v"].values, 20)

    ax.stackplot(
        eps, loc, rsu, v2v,
        labels=["Local", "RSU", "V2V"],
        colors=["#AEC6E8", "#FFBE86", "#98D9A4"],
        alpha=0.85,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Decision Fraction")
    ax.set_title("(c) Offloading Decision Evolution")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    plt.tight_layout(pad=0.8)
    out = out_dir / "fig3_offloading_evolution.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig3] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 图 4: 雷达图 - 综合多维对比
# ══════════════════════════════════════════════════════════════════════════════

def fig_radar(baselines: list[dict], out_dir: Path):
    ORDER = ["Random", "Local-Only", "EFT", "Greedy", "MAPPO"]
    bl = {b["_name"]: b for b in baselines}
    names = [n for n in ORDER if n in bl]
    if len(names) < 2:
        print("[fig4] not enough policies for radar, skip")
        return

    # 5 维指标: 奖励/车辆SR/子任务SR/截止满足率/能效（反向能耗归一化）
    dim_keys  = ["avg_reward", "avg_vehicle_success_rate", "avg_subtask_success_rate",
                 "deadline_meet_ratio", "_energy_inv"]
    dim_labels = ["Reward", "Vehicle SR", "Subtask SR", "Deadline\nMeet", "Energy\nEfficiency"]
    N = len(dim_keys)

    # 计算原始值并归一化到 [0,1]
    raw = {}
    for n in names:
        b = bl[n]
        e_raw = b.get("avg_energy_consumption", 0.0)
        raw[n] = [
            b.get("avg_reward", 0.0),
            b.get("avg_vehicle_success_rate", 0.0),
            b.get("avg_subtask_success_rate", 0.0),
            b.get("deadline_meet_ratio", 0.0),
            -e_raw,   # 能耗越小越好，取负后越大越好
        ]

    all_vals = np.array(list(raw.values()))
    vmin = all_vals.min(axis=0)
    vmax = all_vals.max(axis=0)
    span = np.where(vmax - vmin < 1e-12, 1.0, vmax - vmin)
    norm = {n: (np.array(raw[n]) - vmin) / span for n in names}

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"polar": True})
    for n in names:
        vals = norm[n].tolist() + [norm[n][0]]
        c = COLORS.get(n, "#888888")
        ax.plot(angles, vals, linewidth=1.4, color=c, label=n,
                marker=MARKERS.get(n, "o"), markersize=4)
        ax.fill(angles, vals, color=c, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)
    ax.set_title("(d) Multi-Metric Radar", fontsize=10, pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    out = out_dir / "fig4_radar.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig4] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 图 5: 截止满足率 vs 任务负载（训练曲线）
# ══════════════════════════════════════════════════════════════════════════════

def fig_deadline_vs_reward(df: pd.DataFrame, out_dir: Path):
    if "deadline_miss_rate" not in df.columns:
        print("[fig5] deadline_miss_rate not found, skip")
        return

    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    eps   = df["episode"].values
    dmiss = smooth(df["deadline_miss_rate"].values, 30)
    dmeet = 1.0 - dmiss
    rw    = smooth(df["reward_mean"].values, 30)

    color_dmeet = "#1F77B4"
    color_rw    = COLORS["MAPPO"]

    ax1.fill_between(eps, dmeet, alpha=0.15, color=color_dmeet)
    ax1.plot(eps, dmeet, linewidth=1.4, color=color_dmeet, label="Deadline Meet Rate")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Deadline Meet Rate", color=color_dmeet)
    ax1.tick_params(axis="y", labelcolor=color_dmeet)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

    ax2 = ax1.twinx()
    ax2.plot(eps, rw, linewidth=1.4, color=color_rw, linestyle="--", label="Mean Reward")
    ax2.set_ylabel("Mean Reward", color=color_rw)
    ax2.tick_params(axis="y", labelcolor=color_rw)
    ax2.spines["right"].set_visible(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8, framealpha=0.7)
    ax1.set_title("(e) Deadline Satisfaction", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout(pad=0.8)
    out = out_dir / "fig5_deadline_satisfaction.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig5] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 图 6: 综合 Dashboard（2×3 拼图，论文 Fig. 综合展示）
# ══════════════════════════════════════════════════════════════════════════════

def fig_dashboard(df: pd.DataFrame, baselines: list[dict], out_dir: Path):
    ORDER = ["Random", "Local-Only", "EFT", "Greedy", "MAPPO"]
    bl = {b["_name"]: b for b in baselines}
    names = [n for n in ORDER if n in bl]

    fig = plt.figure(figsize=(7.16, 5.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)

    eps  = df["episode"].values
    rw   = smooth(df["reward_mean"].values, 30)
    rw_r = df["reward_mean"].values

    # ── A: 奖励收敛 ────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(eps, rw_r, alpha=0.15, color=COLORS["MAPPO"], linewidth=0.5)
    ax_a.plot(eps, rw,   color=COLORS["MAPPO"], linewidth=1.4, label="MAPPO")
    for n in [n for n in names if n != "MAPPO"]:
        val = bl[n].get("avg_reward")
        if val is not None:
            ax_a.axhline(val, linestyle=":", linewidth=0.8,
                         color=COLORS.get(n, "#888"), alpha=0.75, label=n)
    ax_a.set_title("(a) Reward Convergence", fontsize=9)
    ax_a.set_xlabel("Episode", fontsize=8)
    ax_a.set_ylabel("Mean Reward", fontsize=8)
    ax_a.legend(fontsize=6, loc="lower right")

    # ── B: 车辆成功率对比 ────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_bar_simple(ax_b, names, bl, "avg_vehicle_success_rate",
                     "Vehicle Success Rate", is_rate=True)
    ax_b.set_title("(b) Vehicle Success Rate", fontsize=9)

    # ── C: 截止满足率对比 ────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    _plot_bar_simple(ax_c, names, bl, "deadline_meet_ratio",
                     "Deadline Meet Ratio", is_rate=True)
    ax_c.set_title("(c) Deadline Satisfaction", fontsize=9)

    # ── D: 卸载决策分布（训练末尾）────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    if all(c in df.columns for c in ["ratio_local", "ratio_rsu", "ratio_v2v"]):
        tail_n = max(1, len(df) // 10)
        tail = df.tail(tail_n)
        dd_names, dd_loc, dd_rsu, dd_v2v = [], [], [], []
        for n in names:
            dd = bl[n].get("decision_distribution", {})
            dd_names.append(n)
            dd_loc.append(dd.get("local", 0))
            dd_rsu.append(dd.get("rsu", 0))
            dd_v2v.append(dd.get("v2v", 0))
        xs = np.arange(len(dd_names))
        ax_d.bar(xs, dd_loc, width=0.6, label="Local", color="#AEC6E8", edgecolor="black", lw=0.4)
        ax_d.bar(xs, dd_rsu, width=0.6, bottom=dd_loc, label="RSU",
                 color="#FFBE86", edgecolor="black", lw=0.4)
        bottom2 = [a + b for a, b in zip(dd_loc, dd_rsu)]
        ax_d.bar(xs, dd_v2v, width=0.6, bottom=bottom2, label="V2V",
                 color="#98D9A4", edgecolor="black", lw=0.4)
        ax_d.set_xticks(xs)
        ax_d.set_xticklabels(dd_names, rotation=25, ha="right", fontsize=7)
        ax_d.set_ylim(0, 1)
        ax_d.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax_d.set_ylabel("Fraction", fontsize=8)
        ax_d.legend(fontsize=6, loc="upper right")
    ax_d.set_title("(d) Decision Distribution", fontsize=9)

    # ── E: 能耗对比 ────────────────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    _plot_bar_simple(ax_e, names, bl, "avg_energy_consumption",
                     "Energy (J)", is_rate=False)
    ax_e.set_title("(e) Energy Consumption", fontsize=9)

    # ── F: 截止+奖励双轴曲线 ─────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    if "deadline_miss_rate" in df.columns:
        dmeet = smooth(1.0 - df["deadline_miss_rate"].values, 30)
        c1, c2 = "#1F77B4", COLORS["MAPPO"]
        ax_f.fill_between(eps, dmeet, alpha=0.12, color=c1)
        ax_f.plot(eps, dmeet, color=c1, linewidth=1.3, label="DL Meet")
        ax_f.set_ylim(0, 1.05)
        ax_f.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax_f.set_ylabel("Deadline Meet", fontsize=8, color=c1)
        ax_f.tick_params(axis="y", labelcolor=c1)
        ax_f2 = ax_f.twinx()
        ax_f2.plot(eps, rw, color=c2, linewidth=1.3, linestyle="--", label="Reward")
        ax_f2.set_ylabel("Reward", fontsize=8, color=c2)
        ax_f2.tick_params(axis="y", labelcolor=c2)
        ax_f2.spines["right"].set_visible(True)
        lines1, l1 = ax_f.get_legend_handles_labels()
        lines2, l2 = ax_f2.get_legend_handles_labels()
        ax_f.legend(lines1 + lines2, l1 + l2, fontsize=6, loc="lower right")
    ax_f.set_title("(f) Deadline vs Reward", fontsize=9)
    ax_f.set_xlabel("Episode", fontsize=8)

    fig.suptitle("VEC DAG Task Offloading — Paper Performance Summary", fontsize=11, y=1.01)
    out = out_dir / "fig6_dashboard.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"[fig6] {out}")


def _plot_bar_simple(ax, names, bl, key, ylabel, is_rate):
    vals = [bl[n].get(key, 0.0) for n in names]
    stds_key = key.replace("avg_", "std_")
    errs = [bl[n].get(stds_key, 0.0) for n in names]
    xs = np.arange(len(names))
    for i, (v, e, n) in enumerate(zip(vals, errs, names)):
        ax.bar(xs[i], v, yerr=e, capsize=3, width=0.6,
               color=COLORS.get(n, "#AAA"), edgecolor="black", linewidth=0.5,
               hatch=HATCHES.get(n, ""), alpha=0.88,
               error_kw={"elinewidth": 0.8})
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    if is_rate:
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir",      default=None,
                   help="Path to a specific run_* directory. Auto-detected if omitted.")
    p.add_argument("--baseline-json", default=None,
                   help="Path to baseline_comparison.json. Auto-searched in runs/ if omitted.")
    p.add_argument("--out-dir",      default=None,
                   help="Output directory. Defaults to <run-dir>/plots/paper/")
    return p.parse_args()


def find_baseline_json(root: Path) -> Path | None:
    for candidate in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not candidate.is_dir():
            continue
        for name in ["baseline_comparison.json", "eval_baselines.json"]:
            p = candidate / name
            if p.exists():
                return p
    return None


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    runs_root = repo_root / "runs"

    # ── 定位 run 目录 ──────────────────────────────────────────────────────────
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(runs_root)
    print(f"[info] run_dir  = {run_dir}")

    # ── 定位 baseline JSON ─────────────────────────────────────────────────────
    baseline_json = None
    if args.baseline_json:
        baseline_json = Path(args.baseline_json)
    else:
        # 先找 run_dir 内部，再全局搜索
        for name in ["baseline_comparison.json", "eval_baselines.json",
                     "metrics/eval_baselines.json"]:
            p = run_dir / name
            if p.exists():
                baseline_json = p
                break
        if baseline_json is None:
            baseline_json = find_baseline_json(runs_root)
    print(f"[info] baselines = {baseline_json}")

    # ── 输出目录 ───────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] out_dir  = {out_dir}")

    # ── 加载数据 ───────────────────────────────────────────────────────────────
    df = load_training_csv(run_dir)
    print(f"[info] training rows = {len(df)}")

    baselines: list[dict] = []
    if baseline_json and baseline_json.exists():
        baselines = load_baselines(baseline_json)
        print(f"[info] baselines loaded: {[b['_name'] for b in baselines]}")
    baselines = inject_mappo_row(baselines, df)

    # ── 绘图 ───────────────────────────────────────────────────────────────────
    fig_convergence(df, out_dir, baselines)
    fig_algorithm_comparison(baselines, out_dir)
    fig_offloading_evolution(df, out_dir)
    fig_radar(baselines, out_dir)
    fig_deadline_vs_reward(df, out_dir)
    fig_dashboard(df, baselines, out_dir)

    print(f"\n[done] All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
