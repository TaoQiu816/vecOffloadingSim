#!/usr/bin/env python3
"""
Group 5: 系统负载实验 — V3（分组柱状图，与 Group4 风格完全一致）
修改：宋体+无加粗+移除数值标注+统一画布+最小空白
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR    = SCRIPT_PATH.parent
FIG_DIR     = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 全局样式（与 Group4 完全对齐：宋体+无加粗+大字号）──────────────────────
mpl.rcParams["font.sans-serif"]    = ["SimSun", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.spines.top"]    = True
mpl.rcParams["axes.spines.right"]  = True
mpl.rcParams["axes.spines.left"]   = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["lines.linewidth"]    = 1.0
mpl.rcParams["font.size"]          = 18  # 全局统一字号
mpl.rcParams["font.weight"]        = "normal"  # 全局无加粗

DPI   = 300
FIG_W = 10
FIG_H = 8  # 统一高度，与Group4一致

# ── Group4 算法配置（顺序、显示名、颜色） ────────────────────────────────
ALGO_MAP = {
    "MAPPO":   ("TERA-MAPPO", "#2068b8"),
    "F-MAPPO": ("F-MAPPO",    "#51b848"),
    "IPPO":    ("IPPO",       "#f8b07f"),
    "NRO":     ("Greedy",     "#d86358"),
    "LO":      ("Local-Only", "#9854a7"),
}
ALGOS_CSV = ["Local-Only", "Greedy", "IPPO", "F-MAPPO", "TERA-MAPPO"]
_DISPLAY_TO_CSV = {v[0]: (k, v[1]) for k, v in ALGO_MAP.items()}

ALGOS      = ALGOS_CSV
COLORS     = {d: _DISPLAY_TO_CSV[d][1] for d in ALGOS}
CSV_KEYS   = {d: _DISPLAY_TO_CSV[d][0] for d in ALGOS}

BAR_W = 0.15
N_ALGO = len(ALGOS)


# ── 工具函数（与Group4完全统一）────────────────────────────────────────────
def _style_ax(ax, ylabel: str, xlabel: str,
              xtick_labels, ylim=None,
              legend_loc: str | None = "upper right",
              fontsize: int = 20):  # 坐标轴标题字号统一20
    x = np.arange(len(xtick_labels))
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=fontsize-2)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="#cccccc", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)  # 边框统一宽度
        spine.set_color("black")
    ax.tick_params(axis="both", labelsize=fontsize-2)
    if legend_loc:
        ax.legend(
            loc=legend_loc,
            fontsize=16,  # 图例字号统一放大
            frameon=True,
            edgecolor="black",
            facecolor="white",
            ncol=1,
            borderpad=0.5,
        )


def save_fig(fig: plt.Figure, stem: str) -> None:
    # 最小化空白
    for ext in ("png", "pdf", "eps"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(f"  saved  {FIG_DIR / stem}.[png|pdf|eps]")


def _build_data(df: pd.DataFrame, x_col: str, metric: str):
    x_vals  = sorted(df[x_col].unique())
    x_labels = [str(v) for v in x_vals]
    matrix = []
    for xv in x_vals:
        row = []
        sub = df[df[x_col] == xv]
        for disp in ALGOS:
            csv_k = CSV_KEYS[disp]
            sel = sub[sub["algorithm"] == csv_k]
            row.append(float(sel[metric].values[0]) if len(sel) > 0 else np.nan)
        matrix.append(row)
    return x_labels, np.array(matrix, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# 单指标分组柱状图（已删除数值标注）
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bar_single(df, x_col, metric, ylabel, xlabel,
                    ylim, legend_loc, filename,
                    panel_label=None, fontsize=20):
    x_labels, mat = _build_data(df, x_col, metric)
    n_group = len(x_labels)
    x       = np.arange(n_group)
    offsets = np.array([(i - (N_ALGO - 1) / 2) * BAR_W for i in range(N_ALGO)])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for j, disp in enumerate(ALGOS):
        vals = mat[:, j]
        for gi in range(n_group):
            v = vals[gi]
            if np.isnan(v):
                continue
            ax.bar(
                x[gi] + offsets[j], v,
                width=BAR_W,
                color=COLORS[disp],
                label=disp if gi == 0 else "_nolegend_",
                edgecolor="black",
                linewidth=1.0,
                zorder=3,
            )
    # 样式设置
    _style_ax(ax, ylabel, xlabel, x_labels,
              ylim=ylim, legend_loc=legend_loc, fontsize=fontsize)
    if panel_label:
        ax.text(0.02, 0.98, panel_label,
                transform=ax.transAxes,
                fontsize=fontsize + 1,
                va="top", ha="left")  # 移除加粗
    fig.tight_layout(pad=0.1)  # 最小化内边距
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 组合图（统一尺寸+无数值标注）
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bar_combined(df, x_col, xlabel,
                      sr_ylim, cft_ylim, filename):
    x_labels, mat_sr  = _build_data(df, x_col, "success_rate")
    _,         mat_cft = _build_data(df, x_col, "mean_cft")
    n_group = len(x_labels)
    x       = np.arange(n_group)
    offsets = np.array([(i - (N_ALGO - 1) / 2) * BAR_W for i in range(N_ALGO)])
    pfs = 18  # 面板字号统一

    # 组合图尺寸统一，保证坐标轴矩形一致
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax_idx, (ax, mat, ylabel, ylim, legend_loc, panel_lbl) in enumerate([
        (axes[0], mat_sr,  "任务完成率",      sr_ylim,  "lower right", "(a)"),
        (axes[1], mat_cft, "平均完成时延 (s)", cft_ylim, "upper left",  "(b)"),
    ]):
        for j, disp in enumerate(ALGOS):
            vals = mat[:, j]
            for gi in range(n_group):
                v = vals[gi]
                if np.isnan(v):
                    continue
                ax.bar(
                    x[gi] + offsets[j], v,
                    width=BAR_W,
                    color=COLORS[disp],
                    label=disp if gi == 0 else "_nolegend_",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
        # 样式
        _style_ax(ax, ylabel, xlabel, x_labels,
                  ylim=ylim, legend_loc=legend_loc, fontsize=pfs)
        ax.text(0.02, 0.98, panel_lbl,
                transform=ax.transAxes,
                fontsize=pfs + 2,
                va="top", ha="left")  # 移除加粗

    fig.tight_layout(pad=0.1, w_pad=1.0)
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# RSU 算力绘图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_rsu_cpu():
    df = pd.read_csv(BASE_DIR / "rsu_cpu_data.csv")
    xlabel = "RSU算力(GHz)"

    plot_bar_single(
        df, "rsu_cpu_factor", "success_rate",
        ylabel="任务完成率", xlabel=xlabel,
        ylim=[0.40, 1.06],
        legend_loc="upper left",
        filename="fig_v3_sr_vs_rsu_cpu",
    )
    plot_bar_single(
        df, "rsu_cpu_factor", "mean_cft",
        ylabel="平均完成时延 (s)", xlabel=xlabel,
        ylim=[1.00, 2.10],
        legend_loc="upper right",
        filename="fig_v3_cft_vs_rsu_cpu",
    )
    plot_bar_combined(
        df, "rsu_cpu_factor", xlabel,
        sr_ylim=[0.40, 1.06],
        cft_ylim=[1.00, 2.10],
        filename="fig_v3_combined_rsu_cpu",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Group 5 System Load — V3 分组柱状图（Group4 统一风格）")
    print("算法：TERA-MAPPO / F-MAPPO / IPPO / Greedy / Local-Only")
    print("=" * 60)

    print("\n[B] RSU 算力影响图")
    plot_rsu_cpu()

    print("\n✓ 全部完成！输出目录：", FIG_DIR)
    for f in sorted(FIG_DIR.glob("fig_v3_*.png")):
        print(f"  {f.name}")