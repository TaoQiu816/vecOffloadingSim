#!/usr/bin/env python3
"""
Group 6: 机制分析图
包括决策分布、延迟分解、资源利用率
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def main():
    mpl.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
    
    base_dir = Path("runs/paper_final_results_20260327")
    data_dir = base_dir / "group6_mechanism_analysis"
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 决策分布饼图
    df_decision = pd.read_csv(data_dir / "decision_distribution.csv")
    fig, ax = plt.subplots(figsize=(7, 7))
    
    labels = ["本地执行", "RSU卸载", "V2V协作"]
    sizes = [df_decision["local"].values[0], df_decision["rsu"].values[0], df_decision["v2v"].values[0]]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', shadow=True, startangle=90,
                                       textprops={'fontsize': 13})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax.set_title("决策类型分布", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_group6_decision_dist.png", dpi=320, bbox_inches='tight')
    plt.close()
    print("已保存: fig_group6_decision_dist.png")
    
    # 2. 延迟分解堆叠柱状图
    df_delay = pd.read_csv(data_dir / "delay_decomposition.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    components = df_delay["component"].tolist()
    means = df_delay["mean"].tolist()
    colors_delay = ["#3498db", "#e74c3c", "#f39c12"]
    
    x_pos = np.arange(len(components))
    bars = ax.bar(x_pos, means, color=colors_delay, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    component_names = ["计算延迟", "传输延迟", "队列延迟"]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(component_names, fontsize=13)
    ax.set_ylabel("平均延迟 (s)", fontsize=14)
    ax.set_title("延迟分解分析", fontsize=16, pad=15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    for bar, val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_group6_delay_decomp.png", dpi=320, bbox_inches='tight')
    plt.close()
    print("已保存: fig_group6_delay_decomp.png")
    
    # 3. 资源利用率对比
    df_util = pd.read_csv(data_dir / "resource_utilization.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    
    resources = ["RSU计算资源", "车辆计算资源"]
    means = df_util["mean"].tolist()
    stds = df_util["std"].tolist()
    
    x_pos = np.arange(len(resources))
    bars = ax.bar(x_pos, means, yerr=stds, color=["#9b59b6", "#1abc9c"],
                  alpha=0.8, capsize=10, edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(resources, fontsize=13)
    ax.set_ylabel("平均利用率", fontsize=14)
    ax.set_title("计算资源利用率", fontsize=16, pad=15)
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    for bar, val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[list(bars).index(bar)],
               f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_group6_resource_util.png", dpi=320, bbox_inches='tight')
    plt.close()
    print("已保存: fig_group6_resource_util.png")
    
    print(f"\n所有图表已保存到: {fig_dir}")


if __name__ == "__main__":
    main()
