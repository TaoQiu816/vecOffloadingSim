#!/usr/bin/env python3
"""
Group 2: 综合性能对比柱状图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams

from scripts.paper.plot_utils_paper import (
    COLORS, set_paper_style, plot_bar_chart
)


def main():
    # 设置中文字体
    mpl.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
    
    base_dir = Path("runs/paper_final_results_20260327")
    data_dir = base_dir / "group2_comprehensive_comparison"
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    df = pd.read_csv(data_dir / "comparison_summary.csv")
    
    # 指标映射
    metric_map = {
        "success_rate": ("成功率", "Success Rate (%)"),
        "mean_cft": ("平均完成时间", "Mean CFT (s)"),
        "p95_cft": ("95%完成时间", "P95 CFT (s)")
    }
    
    # 为每个指标创建柱状图
    for metric, (cn_name, ylabel) in metric_map.items():
        if metric not in df.columns:
            continue
        
        # 准备数据
        methods = df["method"].tolist()
        values = df[metric].tolist()
        
        # 颜色
        colors = [COLORS.get(m, "#7f8c8d") for m in methods]
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # 设置标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(cn_name, fontsize=16, pad=15)
        
        # 网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # 在柱子上添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=11)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        output_path = fig_dir / f"fig_group2_{metric}.png"
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        plt.close()
        
        print(f"已保存: {output_path}")
    
    print(f"\n所有图表已保存到: {fig_dir}")


if __name__ == "__main__":
    import numpy as np
    main()
