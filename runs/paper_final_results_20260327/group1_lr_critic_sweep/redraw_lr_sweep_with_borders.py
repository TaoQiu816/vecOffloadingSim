#!/usr/bin/env python3
"""
为 lr_critic_sweep 图表添加完整的坐标轴边框
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = ROOT / "runs" / "paper_final_results_20260327" / "lr_critic_sweep"
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
METRICS = [
    ("reward_mean", "平均奖励"),
    ("reward_total", "总奖励"),
    ("task_sr", "任务成功率"),
    ("deadline_miss_rate", "截止期违约率"),
    ("mean_cft_completed", "平均 CFT"),
    ("avg_rsu_queue", "队列长度"),
]

FONT_CN = FontProperties(family="Songti SC")

def _set_style():
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times"]
    matplotlib.rcParams["font.size"] = 14
    matplotlib.rcParams["axes.labelsize"] = 16
    matplotlib.rcParams["legend.fontsize"] = 12
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "#fbfbfb"
    matplotlib.rcParams["savefig.facecolor"] = "white"

def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key in SERIES_KEYS:
        col_smooth = f"{metric}__{key}__smooth"
        if col_smooth in df.columns:
            xs = df["episode"].values
            ys = df[col_smooth].values
            ax.plot(xs, ys, color=PALETTE[key], linewidth=2.5, label=LEGEND_LABELS[key])
    
    ax.set_ylabel(ylabel, fontproperties=FONT_CN, fontsize=18)
    ax.set_xlabel("训练轮次", fontproperties=FONT_CN, fontsize=18)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.set_xlim(left=0)
    
    # 显示所有边框，保持原有格式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(1.0)
    
    legend = ax.legend(loc='best', frameon=True, fancybox=True)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    plt.tight_layout()
    output_path = FIG_DIR / f"fig_{metric}_alt_overview_thin.png"
    plt.savefig(output_path, dpi=320, bbox_inches='tight')
    plt.close()
    print(f"✓ {output_path.name}")

def main():
    _set_style()
    
    if not TABLE_PATH.exists():
        print(f"错误: 数据文件不存在 {TABLE_PATH}")
        return 1
    
    df = pd.read_csv(TABLE_PATH)
    print(f"开始重新绘制图表（带完整坐标轴边框）...")
    
    for metric, ylabel in METRICS:
        _plot_metric(df, metric, ylabel)
    
    print(f"\n所有图表已保存到: {FIG_DIR}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
