#!/usr/bin/env python3
"""
Group 5: 系统负载和资源竞争折线图
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
    data_dir = base_dir / "group5_system_load"
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 车辆数量影响
    df_veh = pd.read_csv(data_dir / "vehicle_count_results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("success_rate", "成功率"), ("mean_cft", "平均完成时间 (s)"), ("energy", "能耗 (J)")]
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(df_veh["num_vehicles"], df_veh[metric], marker='o', linewidth=2, markersize=8, color='#9b59b6')
        ax.set_xlabel("车辆数量", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_group5_vehicle_count.png", dpi=320, bbox_inches='tight')
    plt.close()
    print("已保存: fig_group5_vehicle_count.png")
    
    # RSU计算能力影响
    df_rsu = pd.read_csv(data_dir / "rsu_compute_results.csv")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(df_rsu["rsu_compute_factor"], df_rsu[metric], marker='s', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel("RSU计算能力因子", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_group5_rsu_compute.png", dpi=320, bbox_inches='tight')
    plt.close()
    print("已保存: fig_group5_rsu_compute.png")
    
    print(f"\n所有图表已保存到: {fig_dir}")


if __name__ == "__main__":
    main()
