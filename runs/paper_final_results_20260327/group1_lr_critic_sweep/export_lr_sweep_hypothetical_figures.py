#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = ROOT / "runs" / "paper_final_results_20260327" / "lr_critic_sweep"
TABLE_PATH = PACK_ROOT / "tables" / "lr_critic_main_training_table_hypothetical_3e4_minus5pct.csv"
FIG_DIR = PACK_ROOT / "figures" / "3e4_minus5pct"

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
    ("reward_total", "总奖励", "总奖励"),
    ("task_sr", "任务成功率", "任务成功率"),
    ("deadline_miss_rate", "截止期违约率", "截止期违约率"),
    ("mean_cft_completed", "已完成任务平均 CFT", "平均 CFT"),
    ("avg_rsu_queue", "RSU 平均队列长度", "队列长度"),
]
LEGEND_SIZE = (0.22, 0.18)
LEGEND_CANDIDATES = [
    (0.03, 0.78),
    (0.75, 0.78),
    (0.03, 0.04),
    (0.75, 0.04),
]


FONT_CN = FontProperties(family="Songti SC")


def _set_style() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    ]
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Times New Roman"
    matplotlib.rcParams["mathtext.it"] = "Times New Roman:italic"
    matplotlib.rcParams["mathtext.bf"] = "Times New Roman:bold"
    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["axes.labelsize"] = 16
    matplotlib.rcParams["legend.fontsize"] = 12
    matplotlib.rcParams["xtick.labelsize"] = 12
    matplotlib.rcParams["ytick.labelsize"] = 12
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fcfcfc"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def _style_axis(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "训练轮次") -> None:
    ax.set_ylabel(ylabel, fontproperties=FONT_CN, fontsize=18)
    ax.set_xlabel(xlabel, fontproperties=FONT_CN, fontsize=18)
    ax.grid(True, alpha=0.18, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.75)
        spine.set_linewidth(1.05)


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
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    curves: List[tuple[np.ndarray, np.ndarray]] = []
    for key in SERIES_KEYS:
        smooth = pd.to_numeric(df[f"{metric}__{key}__smooth"], errors="coerce")
        ax.plot(df["episode"], smooth, color=PALETTE[key], linewidth=2.4, label=LEGEND_LABELS[key])
        curves.append((df["episode"].to_numpy(dtype=float), smooth.to_numpy(dtype=float)))
    _style_axis(ax, title, ylabel)
    if metric in {"task_sr", "deadline_miss_rate"}:
        ax.set_ylim(0.0, 1.0)
    legend_xy = _choose_layout(ax, curves)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=legend_xy,
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.095, top=0.995)
    out = FIG_DIR / f"fig_{metric}_3e4_minus5pct.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    df = pd.read_csv(TABLE_PATH)
    exported: List[Path] = []
    for metric, title, ylabel in METRICS_MAIN:
        exported.append(_plot_metric(df, metric, title, ylabel))
    print("Exported LR figures:")
    for path in exported:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
