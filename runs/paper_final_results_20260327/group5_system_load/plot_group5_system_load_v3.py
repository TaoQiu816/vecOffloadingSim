#!/usr/bin/env python3
"""
Group 5: 系统负载实验 — V3（分组柱状图，与 Group4 风格完全一致）
只保留 Group4 评估的 5 个算法：TERA-MAPPO / F-MAPPO / IPPO / Greedy / Local-Only
算法名映射（CSV → 显示名）：
  MAPPO     → TERA-MAPPO
  F-MAPPO   → F-MAPPO
  IPPO      → IPPO
  NRO       → Greedy
  LO        → Local-Only

只保留：RSU 算力 (rsu_cpu_factor) — SR 单图 / CFT 单图 / SR+CFT 组合图
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

# ── 全局样式（与 Group4 完全对齐）──────────────────────────────────────────
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

# ── Group4 算法配置（顺序、显示名、颜色） ────────────────────────────────
# CSV列名 → (显示名, 十六进制颜色)
ALGO_MAP = {
    "MAPPO":   ("TERA-MAPPO", "#2068b8"),
    "F-MAPPO": ("F-MAPPO",    "#51b848"),
    "IPPO":    ("IPPO",       "#f8b07f"),
    "NRO":     ("Greedy",     "#d86358"),
    "LO":      ("Local-Only", "#9854a7"),
}
# 显示顺序（从弱到强，与 Group4 保持一致）
ALGOS_CSV = ["Local-Only", "Greedy", "IPPO", "F-MAPPO", "TERA-MAPPO"]
# 逆映射：显示名 → (csv_key, color)
_DISPLAY_TO_CSV = {v[0]: (k, v[1]) for k, v in ALGO_MAP.items()}

# 按显示名顺序整理
ALGOS      = ALGOS_CSV          # 显示名列表
COLORS     = {d: _DISPLAY_TO_CSV[d][1] for d in ALGOS}
CSV_KEYS   = {d: _DISPLAY_TO_CSV[d][0] for d in ALGOS}

BAR_W = 0.15
N_ALGO = len(ALGOS)


# ── 工具 ─────────────────────────────────────────────────────────────────────
def _style_ax(ax, ylabel: str, xlabel: str,
              xtick_labels, ylim=None,
              legend_loc: str | None = "upper right",
              fontsize: int = 14):
    """与 Group4 _style_ax 保持一致的轴样式"""
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
    """
    返回 (x_labels, data_matrix)
    data_matrix[i][j] = j-th 算法在 x_labels[i] 处的值，缺失为 NaN
    """
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
                    decimal=3, panel_label=None, fontsize=14):
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
    if panel_label:
        ax.text(0.02, 0.98, panel_label,
                transform=ax.transAxes,
                fontsize=fontsize + 1, fontweight="bold",
                va="top", ha="left")
    fig.tight_layout(pad=0.5)
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 组合图（1×2：左=SR, 右=CFT）
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bar_combined(df, x_col, xlabel,
                      sr_ylim, cft_ylim, filename):
    x_labels, mat_sr  = _build_data(df, x_col, "success_rate")
    _,         mat_cft = _build_data(df, x_col, "mean_cft")
    n_group = len(x_labels)
    x       = np.arange(n_group)
    offsets = np.array([(i - (N_ALGO - 1) / 2) * BAR_W for i in range(N_ALGO)])
    pfs = 12  # panel fontsize

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

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
        ax.text(0.02, 0.98, panel_lbl,
                transform=ax.transAxes,
                fontsize=pfs + 2, fontweight="bold",
                va="top", ha="left")

    fig.tight_layout(pad=0.8, w_pad=1.5)
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# [B] RSU 算力
# ═══════════════════════════════════════════════════════════════════════════════
def plot_rsu_cpu():
    df = pd.read_csv(BASE_DIR / "rsu_cpu_data.csv")
    xlabel = "RSU 算力因子"

    plot_bar_single(
        df, "rsu_cpu_factor", "success_rate",
        ylabel="任务完成率", xlabel=xlabel,
        ylim=[0.40, 1.06],
        legend_loc="upper left",
        filename="fig_v3_sr_vs_rsu_cpu",
        decimal=3,
    )
    plot_bar_single(
        df, "rsu_cpu_factor", "mean_cft",
        ylabel="平均完成时延 (s)", xlabel=xlabel,
        ylim=[1.00, 2.10],
        legend_loc="upper right",
        filename="fig_v3_cft_vs_rsu_cpu",
        decimal=3,
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
    print("Group 5 System Load — V3 分组柱状图（Group4 风格）")
    print("算法：TERA-MAPPO / F-MAPPO / IPPO / Greedy / Local-Only")
    print("=" * 60)

    print("\n[B] RSU 算力影响图")
    plot_rsu_cpu()

    print("\n✓ 全部完成！输出目录：", FIG_DIR)
    for f in sorted(FIG_DIR.glob("fig_v3_*.png")):
        print(f"  {f.name}")