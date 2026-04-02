#!/usr/bin/env python3
"""
Group 3: 消融实验柱状图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from scripts.paper.plot_utils_paper import COLORS


def main():
    mpl.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
    
    base_dir = Path("runs/paper_final_results_20260327")
    data_dir = base_dir / "group3_ablation"
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_dir / "tables/ablation_table.csv")
    
    # 性能对比柱状图
    metrics = ["success_rate", "mean_cft", "p95_cft"]
    metric_names = ["成功率", "平均完成时间 (s)", "95%完成时间 (s)"]
    
    for metric, name in zip(metrics, metric_names):
        if metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        methods = df["variant"].tolist()
        values = df[metric].tolist()
        colors = ["#1abc9c", "#e67e22", "#3498db"]
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=13)
        ax.set_ylabel(name, fontsize=14)
        ax.set_title(f"消融实验: {name}", fontsize=16, pad=15)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"fig_group3_{metric}.png", dpi=320, bbox_inches='tight')
        plt.close()
        print(f"已保存: fig_group3_{metric}.png")
    
    print(f"\n所有图表已保存到: {fig_dir}")


if __name__ == "__main__":
    main()
