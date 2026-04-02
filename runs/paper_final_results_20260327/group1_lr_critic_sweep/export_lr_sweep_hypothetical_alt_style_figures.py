#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACK_ROOT = Path(__file__).resolve().parent  # group1_lr_critic_sweep/
TABLE_PATH = PACK_ROOT / "tables" / "lr_critic_main_training_table_hypothetical_3e4_minus5pct.csv"
FIG_DIR = PACK_ROOT / "figures" / "3e4_minus5pct_alt_style_overview_thin"

SERIES_KEYS = ["lr_c=2e-4", "lr_c=3e-4", "lr_c=5e-4"]
LEGEND_LABELS = {
    "lr_c=2e-4": r"$\mathrm{lr}_{c}=2\times10^{-4}$",
    "lr_c=3e-4": r"$\mathrm{lr}_{c}=3\times10^{-4}$",
    "lr_c=5e-4": r"$\mathrm{lr}_{c}=5\times10^{-4}$",
}
PALETTE = {
    "lr_c=2e-4": "#1f77b4",
    "lr_c=3e-4": "#d62728",
    "lr_c=5e-4": "#2ca02c",
}
METRICS_MAIN = [
    ("reward_mean", "平均奖励", "平均奖励"),
    ("reward_total", "奖励", "奖励"),
    ("task_sr", "任务成功率", "任务成功率"),
    ("deadline_miss_rate", "截止期违约率", "截止期违约率"),
    ("mean_cft_completed", "已完成任务平均 CFT", "平均 CFT"),
    ("avg_rsu_queue", "RSU 平均队列长度", "队列长度"),
]
# ── 全局样式（与 group4/group5 完全对齐）─────────────────────────────────────
def _set_style() -> None:
    mpl.rcParams["font.sans-serif"]    = ["SimSun", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.spines.top"]    = True
    mpl.rcParams["axes.spines.right"]  = True
    mpl.rcParams["axes.spines.left"]   = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["lines.linewidth"]    = 1.0
    mpl.rcParams["font.size"]          = 18   # 全局统一字号
    mpl.rcParams["font.weight"]        = "normal"  # 全局禁止加粗
    mpl.rcParams["figure.facecolor"]   = "white"
    mpl.rcParams["axes.facecolor"]     = "white"
    mpl.rcParams["savefig.facecolor"]  = "white"


def _style_axis(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "训练轮次",
                fontsize: int = 20) -> None:
    """坐标轴统一样式（与 group4/group5 完全一致）"""
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="#cccccc", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    return not (ax0 + aw <= bx0 or bx0 + bw <= ax0 or ay0 + ah <= by0 or by0 + bh <= ay0)


def _curve_box_score(
    ax: plt.Axes,
    curves: List[tuple[np.ndarray, np.ndarray]],
    box: tuple[float, float, float, float],
) -> float:
    x0, y0, w, h = box
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    score = 0.0
    for xvals, yvals in curves:
        mask = np.isfinite(xvals) & np.isfinite(yvals)
        if not np.any(mask):
            continue
        xn = (xvals[mask] - xmin) / (xmax - xmin)
        yn = (yvals[mask] - ymin) / (ymax - ymin)
        inside = (xn >= x0) & (xn <= x0 + w) & (yn >= y0) & (yn <= y0 + h)
        score += float(np.count_nonzero(inside))
    return score


def _choose_layout(
    ax: plt.Axes,
    curves: List[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    legend_boxes = [(x, y, LEGEND_SIZE[0], LEGEND_SIZE[1]) for x, y in LEGEND_CANDIDATES]
    best_score = float("inf")
    best_legend = legend_boxes[0]
    for legend_box in legend_boxes:
        legend_score = _curve_box_score(ax, curves, legend_box)
        if legend_score < best_score:
            best_score = legend_score
            best_legend = legend_box
    return best_legend[0], best_legend[1]


def _plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))   # 与 group4/group5 统一尺寸
    curves: List[tuple[np.ndarray, np.ndarray]] = []
    for key in SERIES_KEYS:
        smooth = pd.to_numeric(df[f"{metric}__{key}__smooth"], errors="coerce")
        color = PALETTE[key]
        ax.plot(
            df["episode"],
            smooth,
            color=color,
            linewidth=1.55,
            solid_capstyle="round",
            label=LEGEND_LABELS[key],
            zorder=2,
        )
        curves.append((df["episode"].to_numpy(dtype=float), smooth.to_numpy(dtype=float)))
    _style_axis(ax, title, ylabel)
    if metric in {"task_sr", "deadline_miss_rate"}:
        ax.set_ylim(0.0, 1.0)
    ax.legend(
        loc="best",           # 自动选择遮挡曲线最少的位置，始终在坐标轴内
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="black",
        borderaxespad=0.5,
        fontsize=16,          # 与 group4/group5 图例字号对齐
    )
    fig.tight_layout(pad=0.1)
    out = FIG_DIR / f"fig_{metric}_alt_overview_thin.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    df = pd.read_csv(TABLE_PATH)
    exported: List[Path] = []
    for metric, title, ylabel in METRICS_MAIN:
        exported.append(_plot_metric(df, metric, title, ylabel))
    print("Exported alternate thin-line LR figures:")
    for path in exported:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
