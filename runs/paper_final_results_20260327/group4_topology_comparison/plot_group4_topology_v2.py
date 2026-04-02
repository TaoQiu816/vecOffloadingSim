#!/usr/bin/env python3
"""
Group 4: 拓扑鲁棒性对比 (V2 — SR / CFT / CWT(堆叠) / FoV)
修改说明：
- 大幅调大所有文字尺寸
- 去掉柱子上的数字标注
- 统一图片比例，减小空白
- 保留完整边框
- 中文标准宋体 + 全无加粗
"""
from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── 输出路径 ──────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
FIG_DIR = SCRIPT_PATH.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 字体与全局样式设置【核心修改：宋体 + 无加粗 + 更大字号】────────────────────
mpl.rcParams["font.sans-serif"] = ["SimSun", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["font.size"] = 18  # 全局字号再增大
mpl.rcParams["font.weight"] = "normal"  # 全局禁止加粗

# ── 加载 extra_metrics.json 中的分解数据 ──────────────────────────────────────
EXTRA_JSON = SCRIPT_PATH.parent / "extra_metrics.json"
with open(EXTRA_JSON, "r") as f:
    _extra = json.load(f)

# ── 原始数据 ──────────────────────────────────────────────────────────────────
TOPOS = ["Balanced", "Deep", "Parallel"]
TOPO_KEYS = ["balanced", "deep", "parallel"]
ALGOS = ["TERA-MAPPO", "F-MAPPO", "IPPO", "Greedy", "Local-Only"]

COLORS = {
    "TERA-MAPPO": "#2068b8",
    "F-MAPPO":    "#51b848",
    "IPPO":       "#f8b07f",
    "Greedy":     "#d86358",
    "Local-Only": "#9854a7",
}

# 1. Task Success Rate (SR)
SR = {
    "TERA-MAPPO": [0.9620, 0.9875, 0.9585],
    "F-MAPPO":    [0.9145, 0.9680, 0.9455],
    "IPPO":       [0.9065, 0.9065, 0.9065],
    "Greedy":     [0.8510, 0.9120, 0.8630],
    "Local-Only": [0.8530, 0.9510, 0.8530],
}

# 2. Mean CFT (All tasks)
CFT = {
    "TERA-MAPPO": [1.4865, 1.6691, 1.4645],
    "F-MAPPO":    [1.6181, 1.7442, 1.6339],
    "IPPO":       [1.7217, 1.7575, 1.6971],
    "Greedy":     [1.6971, 1.8395, 1.7475],
    "Local-Only": [1.8491, 1.9003, 1.8491],
}

# 3. CWT 分解数据
CWT_COMM = {
    "TERA-MAPPO": [0.0188, 0.0742, 0.0515],
    "F-MAPPO":    [0.1399, 0.1115, 0.1230],
    "IPPO":       [0.1422, 0.1403, 0.1271],
    "Greedy":     [0.1639, 0.1642, 0.1591],
    "Local-Only": [0.0000, 0.0000, 0.0000],
}
CWT_CPU_Q = {
    "TERA-MAPPO": [0.2166, 0.1231, 0.1926],
    "F-MAPPO":    [0.1658, 0.1202, 0.1530],
    "IPPO":       [0.1606, 0.0927, 0.1514],
    "Greedy":     [0.0974, 0.0897, 0.0931],
    "Local-Only": [0.2338, 0.1893, 0.2211],
}
CWT_TOTAL = {
    "TERA-MAPPO": [0.2354, 0.1973, 0.2441],
    "F-MAPPO":    [0.3057, 0.2318, 0.2760],
    "IPPO":       [0.3028, 0.2330, 0.2785],
    "Greedy":     [0.2614, 0.2539, 0.2522],
    "Local-Only": [0.2338, 0.1893, 0.2211],
}

# 4. FoV (Fairness of Vehicle, Jain's Index)
FOV = {
    "TERA-MAPPO": [0.9724, 0.9641, 0.9723],
    "F-MAPPO":    [0.9718, 0.9609, 0.9675],
    "IPPO":       [0.9627, 0.9592, 0.9627],
    "Greedy":     [0.9517, 0.9569, 0.9562],
    "Local-Only": [0.9499, 0.9561, 0.9564],
}

# ── 绘图参数 ──────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 8
DPI = 300
N_ALGO = len(ALGOS)
N_TOPO = len(TOPOS)
BAR_W = 0.15
offsets = np.array([(i - (N_ALGO - 1) / 2) * BAR_W for i in range(N_ALGO)])
x = np.arange(N_TOPO)


def _style_ax(ax, ylabel, xlabel="任务拓扑类型", ylim=None, legend_loc="upper right", fontsize=20):
    """修改：字号增大 + 无加粗"""
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(TOPOS, fontsize=fontsize-2)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="#cccccc", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")
    if legend_loc:
        ax.legend(
            loc=legend_loc,
            fontsize=16,  # 图例字号增大
            frameon=True,
            edgecolor="black",
            facecolor="white",
            ncol=1,
            borderpad=0.5
        )


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(
            out,
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white"
        )
    print(f"  saved  {FIG_DIR / stem}.[png|pdf]")


# ═══════════════════════════════════════════════════════════════════════════════
# 普通分组柱图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_metric(data_dict, ylabel, filename, ylim=None, legend_loc="upper right"):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, algo in enumerate(ALGOS):
        vals = np.array(data_dict[algo], dtype=float)
        ax.bar(
            x + offsets[i],
            vals,
            width=BAR_W,
            color=COLORS[algo],
            label=algo,
            edgecolor="black",
            linewidth=1.0,
            zorder=3
        )
    _style_ax(ax, ylabel, ylim=ylim, legend_loc=legend_loc)
    fig.tight_layout(pad=0.1)
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CWT 堆叠柱图
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cwt_stacked(filename="fig_group4_topology_cwt_v2"):
    from matplotlib.patches import Patch

    HATCH_COMM  = ""
    HATCH_CPU_Q = "////"

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, algo in enumerate(ALGOS):
        comm = np.array(CWT_COMM[algo], dtype=float)
        cpuq = np.array(CWT_CPU_Q[algo], dtype=float)
        color = COLORS[algo]
        ax.bar(
            x + offsets[i],
            comm,
            width=BAR_W,
            color=color,
            hatch=HATCH_COMM,
            edgecolor="black",
            linewidth=1.0,
            zorder=3
        )
        ax.bar(
            x + offsets[i],
            cpuq,
            width=BAR_W,
            bottom=comm,
            color=color,
            hatch=HATCH_CPU_Q,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.65,
            zorder=3
        )

    all_totals = [v for algo in ALGOS for v in CWT_TOTAL[algo]]
    _auto_max = max(all_totals) * 1.15
    _style_ax(ax, "子任务平均等待时间 (s)", ylim=[0.0, _auto_max], legend_loc=None)

    algo_handles = [
        Patch(facecolor=COLORS[algo], edgecolor="black", linewidth=1.0, label=algo)
        for algo in ALGOS
    ]
    type_handles = [
        Patch(facecolor="gray", hatch=HATCH_COMM,  edgecolor="black", linewidth=1.0, label="通信等待"),
        Patch(facecolor="gray", hatch=HATCH_CPU_Q, edgecolor="black", linewidth=1.0, alpha=0.65, label="计算排队等待"),
    ]

    leg1 = ax.legend(
        handles=algo_handles,
        loc="upper right",
        fontsize=14,
        frameon=True,
        edgecolor="black",
        ncol=1,
        title_fontsize=14
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=type_handles,
        loc="upper left",
        fontsize=14,
        frameon=True,
        edgecolor="black",
        title_fontsize=14
    )

    fig.tight_layout(pad=0.1)
    save_fig(fig, filename)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 2×2 综合面板【核心修改：删除所有 fontweight="bold"】
# ═══════════════════════════════════════════════════════════════════════════════
def plot_combined_panel():
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes_flat = axes.flatten()
    panel_fontsize = 18

    # ── (a) 任务成功率 SR ──
    ax = axes_flat[0]
    for i, algo in enumerate(ALGOS):
        vals = np.array(SR[algo], dtype=float)
        ax.bar(
            x + offsets[i],
            vals,
            width=BAR_W,
            color=COLORS[algo],
            label=algo,
            edgecolor="black",
            linewidth=0.8,
            zorder=3
        )
    _style_ax(
        ax,
        "任务成功率",
        ylim=[0.80, 1.03],
        fontsize=panel_fontsize,
        legend_loc=None
    )
    ax.text(
        0.02, 0.98, "(a)",
        transform=ax.transAxes,
        fontsize=panel_fontsize+2,
        va="top",
        ha="left"
    )
    ax.legend(
        loc="lower right",
        fontsize=12,
        frameon=True,
        edgecolor="black",
        ncol=1
    )

    # ── (b) 平均任务完成时间 CFT ──
    ax = axes_flat[1]
    for i, algo in enumerate(ALGOS):
        vals = np.array(CFT[algo], dtype=float)
        ax.bar(
            x + offsets[i],
            vals,
            width=BAR_W,
            color=COLORS[algo],
            label=algo,
            edgecolor="black",
            linewidth=0.8,
            zorder=3
        )
    _style_ax(
        ax,
        "任务平均完成时间 (s)",
        ylim=[1.0, 2.3],
        fontsize=panel_fontsize,
        legend_loc=None
    )
    ax.text(
        0.02, 0.98, "(b)",
        transform=ax.transAxes,
        fontsize=panel_fontsize+2,
        va="top",
        ha="left"
    )

    # ── (c) 子任务平均等待时间 CWT 堆叠柱图 ──
    ax = axes_flat[2]
    from matplotlib.patches import Patch
    HATCH_COMM  = ""
    HATCH_CPU_Q = "////"
    for i, algo in enumerate(ALGOS):
        comm = np.array(CWT_COMM[algo], dtype=float)
        cpuq = np.array(CWT_CPU_Q[algo], dtype=float)
        color = COLORS[algo]
        ax.bar(
            x + offsets[i],
            comm,
            width=BAR_W,
            color=color,
            hatch=HATCH_COMM,
            edgecolor="black",
            linewidth=0.6,
            zorder=3
        )
        ax.bar(
            x + offsets[i],
            cpuq,
            width=BAR_W,
            bottom=comm,
            color=color,
            hatch=HATCH_CPU_Q,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.65,
            zorder=3
        )

    _all_totals = [v for algo in ALGOS for v in CWT_TOTAL[algo]]
    _auto_max_c = max(_all_totals) * 1.15
    _style_ax(
        ax,
        "子任务平均等待时间 (s)",
        ylim=[0.0, _auto_max_c],
        fontsize=panel_fontsize,
        legend_loc=None
    )
    ax.text(
        0.02, 0.98, "(c)",
        transform=ax.transAxes,
        fontsize=panel_fontsize+2,
        va="top",
        ha="left"
    )

    algo_handles = [
        Patch(facecolor=COLORS[algo], edgecolor="black", linewidth=0.6, label=algo)
        for algo in ALGOS
    ]
    type_handles = [
        Patch(facecolor="gray", hatch=HATCH_COMM,  edgecolor="black", linewidth=0.6, label="通信等待"),
        Patch(facecolor="gray", hatch=HATCH_CPU_Q, edgecolor="black", linewidth=0.6, alpha=0.65, label="计算排队等待"),
    ]
    leg1 = ax.legend(
        handles=algo_handles,
        loc="upper right",
        fontsize=10,
        frameon=True,
        edgecolor="black",
        title="算法",
        title_fontsize=11,
        ncol=1
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=type_handles,
        loc="upper left",
        fontsize=10,
        frameon=True,
        edgecolor="black",
        title="等待类型",
        title_fontsize=11
    )

    # ── (d) 车辆公平性指数 FoV ──
    ax = axes_flat[3]
    for i, algo in enumerate(ALGOS):
        vals = np.array(FOV[algo], dtype=float)
        ax.bar(
            x + offsets[i],
            vals,
            width=BAR_W,
            color=COLORS[algo],
            label=algo,
            edgecolor="black",
            linewidth=0.8,
            zorder=3
        )
    _style_ax(
        ax,
        "车辆公平性指数 (Jain)",
        ylim=[0.90, 1.02],
        fontsize=panel_fontsize,
        legend_loc=None
    )
    ax.text(
        0.02, 0.98, "(d)",
        transform=ax.transAxes,
        fontsize=panel_fontsize+2,
        va="top",
        ha="left"
    )

    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.2)
    save_fig(fig, "fig_group4_topology_combined_v2")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 执行绘图
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Group 4 Topology Figures (宋体大字号版)...")

    plot_metric(
        SR,
        "任务成功率",
        "fig_group4_topology_sr_v2",
        ylim=[0.80, 1.03],
        legend_loc="upper right"
    )

    plot_metric(
        CFT,
        "平均任务完成时间 (s)",
        "fig_group4_topology_cft_v2",
        ylim=[1.0, 2.3],
        legend_loc="upper right"
    )

    plot_cwt_stacked("fig_group4_topology_cwt_v2")

    plot_metric(
        FOV,
        "车辆公平性指数 (Jain)",
        "fig_group4_topology_fov_v2",
        ylim=[0.90, 1.02],
        legend_loc="upper right"
    )

    plot_combined_panel()

    print("\n所有图片生成完成，文件列表：")
    for f in sorted(FIG_DIR.glob("fig_group4_topology_*_v2.*")):
        print(f"  {f.name}")