#!/usr/bin/env python3
"""
Group 5: 系统负载实验 - RSU算力影响
绘制 SR 和 Mean CFT vs RSU CPU Factor
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


def plot_rsu_cpu_separate():
    """分别绘制SR和CFT vs RSU CPU Factor"""
    base_dir = Path("runs/paper_final_results_20260327/group5_system_load")
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    df = pd.read_csv(base_dir / "rsu_cpu_data.csv")

    # 算法顺序和颜色
    algo_order = ['LO', 'IPPO', 'NRO', 'MAPPO', 'CP-EFT', 'F-MAPPO']
    algo_colors = {
        'LO': '#95a5a6',
        'IPPO': '#9b59b6',
        'NRO': '#e67e22',
        'MAPPO': '#3498db',
        'CP-EFT': '#f39c12',
        'F-MAPPO': '#e74c3c'
    }
    algo_markers = {
        'LO': 's',
        'IPPO': '^',
        'NRO': 'D',
        'MAPPO': 'o',
        'CP-EFT': 'v',
        'F-MAPPO': '*'
    }

    rsu_factors = sorted(df['rsu_cpu_factor'].unique())

    # ========== 图1: Success Rate vs RSU CPU Factor ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algo_order:
        df_algo = df[df['algorithm'] == algo].sort_values('rsu_cpu_factor')
        if len(df_algo) == 0:
            continue

        ax.plot(df_algo['rsu_cpu_factor'], df_algo['success_rate'],
                marker=algo_markers.get(algo, 'o'),
                color=algo_colors.get(algo, 'gray'),
                linewidth=2.5,
                markersize=10,
                label=algo,
                zorder=10)

        # 添加数值标签
        for _, row in df_algo.iterrows():
            ax.text(row['rsu_cpu_factor'], row['success_rate'] + 0.015,
                   f"{row['success_rate']:.2f}",
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', linewidth=0.5, alpha=0.8))

    ax.set_xlabel('RSU 算力因子', fontsize=14, fontweight='bold')
    ax.set_ylabel('成功率 (SR)', fontsize=14, fontweight='bold')
    ax.set_xticks(rsu_factors)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=11, frameon=True, ncol=2)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    for ext in ['png', 'pdf', 'eps']:
        output_path = fig_dir / f"fig_sr_vs_rsu_cpu.{ext}"
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")

    plt.close()

    # ========== 图2: Mean CFT vs RSU CPU Factor ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algo_order:
        df_algo = df[df['algorithm'] == algo].sort_values('rsu_cpu_factor')
        if len(df_algo) == 0:
            continue

        ax.plot(df_algo['rsu_cpu_factor'], df_algo['mean_cft'],
                marker=algo_markers.get(algo, 'o'),
                color=algo_colors.get(algo, 'gray'),
                linewidth=2.5,
                markersize=10,
                label=algo,
                zorder=10)

        # 添加数值标签
        for _, row in df_algo.iterrows():
            ax.text(row['rsu_cpu_factor'], row['mean_cft'] + 0.05,
                   f"{row['mean_cft']:.2f}",
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', linewidth=0.5, alpha=0.8))

    ax.set_xlabel('RSU 算力因子', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均完成时延 (s)', fontsize=14, fontweight='bold')
    ax.set_xticks(rsu_factors)
    ax.set_ylim(1.0, 2.1)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=11, frameon=True, ncol=2)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    for ext in ['png', 'pdf', 'eps']:
        output_path = fig_dir / f"fig_cft_vs_rsu_cpu.{ext}"
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")

    plt.close()


def plot_rsu_cpu_combined():
    """组合绘制SR和CFT vs RSU CPU Factor（双Y轴）"""
    base_dir = Path("runs/paper_final_results_20260327/group5_system_load")
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    df = pd.read_csv(base_dir / "rsu_cpu_data.csv")

    # 算法顺序和颜色
    algo_order = ['LO', 'IPPO', 'NRO', 'MAPPO', 'CP-EFT', 'F-MAPPO']
    algo_colors = {
        'LO': '#95a5a6',
        'IPPO': '#9b59b6',
        'NRO': '#e67e22',
        'MAPPO': '#3498db',
        'CP-EFT': '#f39c12',
        'F-MAPPO': '#e74c3c'
    }
    algo_markers = {
        'LO': 's',
        'IPPO': '^',
        'NRO': 'D',
        'MAPPO': 'o',
        'CP-EFT': 'v',
        'F-MAPPO': '*'
    }

    rsu_factors = sorted(df['rsu_cpu_factor'].unique())

    # 创建双Y轴图
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 左Y轴：Success Rate
    for algo in algo_order:
        df_algo = df[df['algorithm'] == algo].sort_values('rsu_cpu_factor')
        if len(df_algo) == 0:
            continue

        ax1.plot(df_algo['rsu_cpu_factor'], df_algo['success_rate'],
                marker=algo_markers.get(algo, 'o'),
                color=algo_colors.get(algo, 'gray'),
                linewidth=2.5,
                markersize=10,
                label=f"{algo} (SR)",
                linestyle='-',
                zorder=10)

    ax1.set_xlabel('RSU 算力因子', fontsize=14, fontweight='bold')
    ax1.set_ylabel('成功率 (SR)', fontsize=14, fontweight='bold', color='black')
    ax1.set_xticks(rsu_factors)
    ax1.set_ylim(0.4, 1.05)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # 右Y轴：Mean CFT
    ax2 = ax1.twinx()

    for algo in algo_order:
        df_algo = df[df['algorithm'] == algo].sort_values('rsu_cpu_factor')
        if len(df_algo) == 0:
            continue

        ax2.plot(df_algo['rsu_cpu_factor'], df_algo['mean_cft'],
                marker=algo_markers.get(algo, 'o'),
                color=algo_colors.get(algo, 'gray'),
                linewidth=2.0,
                markersize=8,
                label=f"{algo} (CFT)",
                linestyle='--',
                alpha=0.7,
                zorder=5)

    ax2.set_ylabel('平均完成时延 (s)', fontsize=14, fontweight='bold', color='black')
    ax2.set_ylim(1.0, 2.1)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              loc='center left', fontsize=10, frameon=True, ncol=2)

    plt.tight_layout()

    for ext in ['png', 'pdf', 'eps']:
        output_path = fig_dir / f"fig_sr_cft_vs_rsu_cpu_combined.{ext}"
        plt.savefig(output_path, dpi=320, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")

    plt.close()


def main():
    print("========== 绘制RSU算力影响图 ==========")
    plot_rsu_cpu_separate()
    print("\n========== 绘制RSU算力组合图 ==========")
    plot_rsu_cpu_combined()
    print("\n✓ 所有RSU算力图表已生成")


if __name__ == "__main__":
    main()
