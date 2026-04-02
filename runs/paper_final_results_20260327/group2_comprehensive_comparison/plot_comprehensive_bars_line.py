#!/usr/bin/env python3
"""
Group 2: 综合性能对比 - 柱状图+折线图组合
绘制Mean CFT、P95 CFT（柱状图）和Success Rate（折线图）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False


def main():
    base_dir = Path("runs/paper_final_results_20260327")
    data_dir = base_dir / "group2_comprehensive_comparison"
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    df = pd.read_csv(data_dir / "comparison_summary.csv")
    
    # 方法顺序（按图片显示）
    method_order = ['LO', 'IPPO-H', 'NRO', 'TERA-MAPPO', 'EFT-H', 'F-MAPPO']
    df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
    df = df.sort_values('method')
    
    methods = df['method'].tolist()
    mean_cft = df['mean_cft'].tolist()
    p95_cft = df['p95_cft'].tolist()
    success_rate = df['success_rate'].tolist()
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax1.bar(x - width/2, mean_cft, width, label='Mean CFT', 
                     color='#3498db', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, p95_cft, width, label='P95 CFT',
                     color='#e67e22', edgecolor='black', linewidth=1)
    
    # 设置左侧Y轴（时延）
    ax1.set_xlabel('方法', fontsize=14)
    ax1.set_ylabel('完成时延 (s)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_ylim(0, 8)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)
    
    # 在柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 创建右侧Y轴（成功率）
    ax2 = ax1.twinx()
    line = ax2.plot(x, success_rate, 'o-', color='#e74c3c', linewidth=2.5,
                    markersize=8, label='SR', zorder=10)
    
    ax2.set_ylabel('成功率', fontsize=14)
    ax2.set_ylim(0.5, 1.0)
    ax2.tick_params(axis='y', labelsize=12)
    
    # 在折线点上添加数值标签
    for i, (xi, yi) in enumerate(zip(x, success_rate)):
        ax2.text(xi, yi + 0.02, f'{yi:.2f}', ha='center', va='bottom',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', linewidth=0.8))
    
    # 合并图例
    bars_legend = [bars1, bars2]
    labels1 = ['Mean CFT', 'P95 CFT']
    labels2 = ['SR']
    
    ax1.legend(bars_legend + line, labels1 + labels2, 
              loc='upper left', fontsize=11, frameon=True)
    
    plt.tight_layout()
    
    # 保存多种格式
    for ext in ['png', 'pdf', 'eps']:
        output_path = fig_dir / f"fig_comprehensive_comparison_bars_line.{ext}"
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        print(f"已保存: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
