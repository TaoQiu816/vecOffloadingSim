#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = ROOT / "runs" / "paper_final_results_20260327" / "lr_critic_sweep"
FIG_DIR = PACK_ROOT / "figures" / "alt_style_overview_thin"
TAB_DIR = PACK_ROOT / "tables"

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
METRICS_DIAG = [
    ("approx_kl", "近似 KL 散度", "近似 KL"),
    ("entropy", "策略熵", "策略熵"),
    ("clip_frac", "裁剪比例", "裁剪比例"),
]
METRICS_DECISION = [
    ("ratio_local", "本地执行比例", "比例"),
    ("ratio_rsu", "RSU 卸载比例", "比例"),
    ("ratio_v2v", "V2V 卸载比例", "比例"),
]
LEGEND_SIZE = (0.22, 0.18)
LEGEND_CANDIDATES = [
    (0.03, 0.86),
    (0.75, 0.79),
    (0.03, 0.03),
    (0.75, 0.03),
]


def _set_style() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.size"] = 11.2
    matplotlib.rcParams["axes.titlesize"] = 12.6
    matplotlib.rcParams["axes.labelsize"] = 11.2
    matplotlib.rcParams["legend.fontsize"] = 9.3
    matplotlib.rcParams["xtick.labelsize"] = 9.8
    matplotlib.rcParams["ytick.labelsize"] = 9.8
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fbfbfb"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def _style_axis(ax: plt.Axes, title: str, ylabel: str, xlabel: str = "训练轮次") -> None:
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.75)
        spine.set_linewidth(1.05)


def _series_labels(df: pd.DataFrame, metric: str) -> List[str]:
    labels = []
    suffix = "__raw"
    prefix = f"{metric}__"
    for col in df.columns:
        if col.startswith(prefix) and col.endswith(suffix):
            labels.append(col[len(prefix):-len(suffix)])
    return [label for label in PALETTE if label in labels]


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


def _plot_metric(table_path: Path, metric: str, title: str, ylabel: str, stem: str) -> Path:
    df = pd.read_csv(table_path)
    series_labels = _series_labels(df, metric)
    fig, ax = plt.subplots(figsize=(7.35, 5.05))
    curves: List[tuple[np.ndarray, np.ndarray]] = []
    for label in series_labels:
        smooth = pd.to_numeric(df[f"{metric}__{label}__smooth"], errors="coerce")
        color = PALETTE[label]
        ax.plot(
            df["episode"],
            smooth,
            color=color,
            linewidth=1.55,
            solid_capstyle="round",
            label=label,
            zorder=2,
        )
        curves.append((df["episode"].to_numpy(dtype=float), smooth.to_numpy(dtype=float)))
    _style_axis(ax, title, ylabel)
    if metric in {"task_sr", "deadline_miss_rate", "ratio_local", "ratio_rsu", "ratio_v2v", "clip_frac"}:
        ax.set_ylim(0.0, 1.0)
    legend_xy = _choose_layout(ax, curves)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=legend_xy,
        bbox_transform=ax.transAxes,
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="#d0d0d0",
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.90)
    out = FIG_DIR / f"{stem}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _set_style()
    exported: List[Path] = []
    main_table = TAB_DIR / "lr_critic_main_training_table.csv"
    diag_table = TAB_DIR / "lr_critic_diagnostics_table.csv"
    decision_table = TAB_DIR / "lr_critic_decision_mix_table.csv"
    for metric, title, ylabel in METRICS_MAIN:
        exported.append(_plot_metric(main_table, metric, title, ylabel, f"fig_{metric}_alt_overview_thin"))
    for metric, title, ylabel in METRICS_DIAG:
        exported.append(_plot_metric(diag_table, metric, title, ylabel, f"fig_{metric}_alt_overview_thin"))
    for metric, title, ylabel in METRICS_DECISION:
        exported.append(_plot_metric(decision_table, metric, title, ylabel, f"fig_{metric}_alt_overview_thin"))
    print("Exported alternate thin-line LR figures:")
    for path in exported:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
