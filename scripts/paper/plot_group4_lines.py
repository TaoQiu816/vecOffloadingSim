#!/usr/bin/env python3
"""
Group 4: 任务复杂度和截止期敏感性折线图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def main():
    mpl.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
    
    base_dir = Path("runs/paper_final_results_20260327")
    data_dir = base_dir / "group4_complexity_sensitivity"
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # DAG规模影响
    df_dag = pd.read_csv(data_dir / "dag_size_results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("success_rate", "成功率"), ("mean_cft", "平均完成时间 (s)"), ("p95_cft", "95%完成时间 (s)")]
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(df_dag["dag_size"], df_dag[metric], marker='o', linewidth=2, markersize=8, color='#1abc9c')
        ax.set_xlabel("DAG规模", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_group4_dag_size.png", dpi=320, bbox_inches='tight')
    plt.close()
    print("已保存: fig_group4_dag_size.png")
    
    # 截止期因子影响
    df_deadline = pd.read_csv(data_dir / "deadline_factor_results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(df_deadline["deadline_factor"], df_deadline[metric], marker='s', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel("截止期因子", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_group4_deadline_factor.png", dpi=320, bbox_inches='tight')
    plt.close()
    print("已保存: fig_group4_deadline_factor.png")
    
    print(f"\n所有图表已保存到: {fig_dir}")


if __name__ == "__main__":
    main()
