#!/usr/bin/env python3
"""
Group 2: 综合性能对比 - 柱状图+折线图组合
绘制 Mean CFT（柱状图）和 Success Rate（折线图）
格式与 group4/group5 完全对齐：宋体+无加粗+大字号+最小空白+保留完整边框
去掉 P95 CFT 柱
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── 全局样式（与 group4/group5 完全对齐）─────────────────────────────────────
mpl.rcParams["font.sans-serif"]    = ["SimSun", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.spines.top"]    = True
mpl.rcParams["axes.spines.right"]  = True
mpl.rcParams["axes.spines.left"]   = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["font.size"]          = 18   # 全局统一字号
mpl.rcParams["font.weight"]        = "normal"  # 全局禁止加粗
mpl.rcParams["figure.facecolor"]   = "white"
mpl.rcParams["axes.facecolor"]     = "white"
mpl.rcParams["savefig.facecolor"]  = "white"

DPI   = 300
FIG_W = 10
FIG_H = 8


def main():
    base_dir = Path("runs/paper_final_results_20260327")
    data_dir = base_dir / "group2_comprehensive_comparison"
    fig_dir  = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    df = pd.read_csv(data_dir / "comparison_summary.csv")

    # 方法顺序（按图片显示）
    method_order = ['LO', 'IPPO-H', 'NRO', 'TERA-MAPPO', 'EFT-H', 'F-MAPPO']
    df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
    df = df.sort_values('method')

    methods      = df['method'].tolist()
    mean_cft     = df['mean_cft'].tolist()
    success_rate = df['success_rate'].tolist()

    FONTSIZE = 20   # 坐标轴标题字号，与 group4/group5 统一

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))

    x     = np.arange(len(methods))
    width = 0.6   # 仅单列柱，加宽以充分利用空间

    # ── 仅保留 Mean CFT 柱（去掉 P95 CFT）─────────────────────────────────────
    bars1 = ax1.bar(x, mean_cft, width, label='Mean CFT',
                    color='#3498db', edgecolor='black', linewidth=1.0, zorder=3)

    # ── 左轴样式（与 group4/group5 _style_ax 完全对齐）──────────────────────
    ax1.set_xlabel('方法', fontsize=FONTSIZE)
    ax1.set_ylabel('完成时延 (s)', fontsize=FONTSIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=FONTSIZE - 2)
    ax1.tick_params(axis='y', labelsize=FONTSIZE - 2)
    valid_cft = [v for v in mean_cft if v == v]
    max_bar   = max(valid_cft) if valid_cft else 8.0
    ax1.set_ylim(0, max_bar * 1.30)
    ax1.grid(axis='y', linestyle='--', alpha=0.7, color='#cccccc', zorder=0)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # ── 右轴 SR 折线────────────────────────────────────────────────────────────
    ax2 = ax1.twinx()
    line_obj, = ax2.plot(x, success_rate, 'o-', color='#e74c3c', linewidth=2.5,
                         markersize=8, label='任务成功率', zorder=10)
    ax2.set_ylabel('任务成功率', fontsize=FONTSIZE)
    ax2.set_ylim(0.5, 1.05)
    ax2.tick_params(axis='y', labelsize=FONTSIZE - 2)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # ── 图例（合并两轴，与 group4/group5 字号对齐）──────────────────────────
    ax1.legend(
        [bars1, line_obj],
        ['Mean CFT', '任务成功率'],
        loc='upper left',
        fontsize=16,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor='black',
    )

    fig.tight_layout(pad=0.1)

    # 保存
    for ext in ['png', 'pdf', 'eps']:
        output_path = fig_dir / f"fig_comprehensive_comparison_bars_line.{ext}"
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight',
                    pad_inches=0.02, facecolor='white')
        print(f"已保存: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
