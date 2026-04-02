#!/usr/bin/env python3
"""
Group 5: V4 绘图脚本（仅车辆规模）
修改点：
1. fig_v4_cft_vs_vehicles.png 图例放在左上方
2. fig_v4_sr_vs_vehicles.png 图例放在右上方
3. 路径已恢复为原来的设置
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── 路径配置（已恢复为原来的设置）──────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR    = SCRIPT_PATH.parent
DATA_DIR    = BASE_DIR 
FIG_DIR     = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 全局样式（完全保留）────────────────────────────────────────────────────
mpl.rcParams["font.sans-serif"]    = ["SimHei", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.spines.top"]    = True
mpl.rcParams["axes.spines.right"]  = True
mpl.rcParams["axes.spines.left"]   = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["lines.linewidth"]    = 0.8
mpl.rcParams["font.size"]          = 12

DPI   = 300
FIG_W = 10
FIG_H = 7

# ── 算法配置（完全保留）────────────────────────────────────────────────────
ALGO_MAP = {
    'MAPPO':   ('TERA-MAPPO', '#2068b8'),
    'F-MAPPO': ('F-MAPPO',    '#51b848'),
    'IPPO':    ('IPPO',       '#f8b07f'),
    'NRO':     ('Greedy',     '#d86358'),
    'LO':      ('Local-Only', '#9854a7'),
}

DISPLAY_ORDER = ['TERA-MAPPO', 'F-MAPPO', 'IPPO', 'Greedy', 'Local-Only']

CSV_KEYS   = {v[0]: k for k, v in ALGO_MAP.items()}
COLORS     = {v[0]: v[1] for k, v in ALGO_MAP.items()}
ALGOS      = DISPLAY_ORDER

BAR_W = 0.15
N_ALGO = len(ALGOS)


# ── 工具函数（完全保留）────────────────────────────────────────────────────
def _style_ax(ax, ylabel: str, xlabel: str,
              xtick_labels, ylim=None,
              legend_loc: str | None = "upper right",
              fontsize: int = 14):
    x = np.arange(len(xtick_labels))
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="normal")
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=fontsize - 1)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="#cccccc", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    if legend_loc:
        ax.legend(
            loc=legend_loc,
            fontsize=12,
            frameon=True,
            edgecolor="black",
            facecolor="white",
            ncol=1,
            borderpad=0.5,
        )


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "eps"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight",
                    pad_inches=0.1, facecolor="white")
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
# 单指标分组柱状图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bar_single(df, x_col, metric, ylabel, xlabel,
                    ylim, legend_loc, filename,
                    decimal=3, fontsize=14):
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
            bar = ax.bar(
                x[gi] + offsets[j], v,
                width=BAR_W,
                color=COLORS[disp],
                label=disp if gi == 0 else "_nolegend_",
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            ax.text(
                x[gi] + offsets[j], v,
                f"{v:.{decimal}f}",
                ha="center", va="bottom",
                fontsize=9, zorder=4,
            )

    _style_ax(ax, ylabel, xlabel, x_labels,
              ylim=ylim, legend_loc=legend_loc, fontsize=fontsize)
    
    fig.tight_layout(pad=0.5)
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 组合图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bar_combined(df, x_col, xlabel,
                      sr_ylim, cft_ylim, filename):
    x_labels, mat_sr  = _build_data(df, x_col, "success_rate")
    _,         mat_cft = _build_data(df, x_col, "mean_cft")
    n_group = len(x_labels)
    x       = np.arange(n_group)
    offsets = np.array([(i - (N_ALGO - 1) / 2) * BAR_W for i in range(N_ALGO)])
    pfs = 12

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plot_configs = [
        (axes[0], mat_sr,  "任务完成率",      sr_ylim,  "lower right"),
        (axes[1], mat_cft, "平均完成时延 (s)", cft_ylim, "upper right"),
    ]

    for ax, mat, ylabel, ylim, legend_loc in plot_configs:
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
                    linewidth=0.7,
                    zorder=3,
                )
                ax.text(
                    x[gi] + offsets[j], v,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, zorder=4,
                )
        _style_ax(ax, ylabel, xlabel, x_labels,
                  ylim=ylim, legend_loc=legend_loc, fontsize=pfs)

    fig.tight_layout(pad=0.8, w_pad=1.5)
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 车辆规模绘图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_vehicle_scale():
    csv_path = DATA_DIR / "vehicle_scale_data_v4.csv"
    if not csv_path.exists():
        print(f"[错误] 找不到数据文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    xlabel = "车辆数量"

    print("\n正在生成 SR 单图（图例：右上方）...")
    plot_bar_single(
        df, "num_vehicles", "success_rate",
        ylabel="任务完成率", xlabel=xlabel,
        ylim=[0.50, 1.06],
        legend_loc="upper right",  # 图例：右上方
        filename="fig_v4_sr_vs_vehicles",
        decimal=3,
    )

    print("正在生成 CFT 单图（图例：左上方）...")
    plot_bar_single(
        df, "num_vehicles", "mean_cft",
        ylabel="平均完成时延 (s)", xlabel=xlabel,
        ylim=[0.90, 2.40],
        legend_loc="upper left",   # 图例：左上方
        filename="fig_v4_cft_vs_vehicles",
        decimal=3,
    )

    print("正在生成 SR+CFT 组合图...")
    plot_bar_combined(
        df, "num_vehicles", xlabel,
        sr_ylim=[0.50, 1.06],
        cft_ylim=[0.90, 2.40],
        filename="fig_v4_combined_vehicles",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Group 5 — V4 绘图（仅车辆规模）")
    print("=" * 60)

    plot_vehicle_scale()

    print("\n✓ 全部完成！输出目录：", FIG_DIR)