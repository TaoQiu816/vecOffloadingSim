"""
[绘图脚本] plot_results.py
Training Results Visualization Script

作用 (Purpose):
读取训练过程的CSV文件，生成论文级别的可视化图表，包括收敛曲线、策略演化、物理指标和训练诊断。
Reads training CSV files and generates publication-quality visualizations including convergence curves,
policy evolution, physical metrics, and training diagnostics.

使用方法 (Usage):
    python plot_results.py --log-file logs/run_YYYYMMDD_HHMMSS/training_stats.csv --output-dir plots/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def rolling_mean(data, window=50):
    """计算滚动平均"""
    if len(data) < window:
        window = max(1, len(data) // 10)
    return data.rolling(window=window, min_periods=1).mean()


def plot_convergence(df, output_dir):
    """
    绘制收敛曲线 (Reward, Task Success Rate, Subtask Success Rate)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Reward
    ax = axes[0]
    ax.plot(df['episode'], df['reward'], alpha=0.3, color='steelblue', linewidth=0.8, label='Raw')
    ax.plot(df['episode'], rolling_mean(df['reward'], 50), color='darkblue', linewidth=2, label='Smoothed (50-ep)')
    ax.set_ylabel('Episode Reward', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title('Training Convergence', fontweight='bold', fontsize=14)
    
    # 2. Task Success Rate
    ax = axes[1]
    ax.plot(df['episode'], df['task_sr'] * 100, alpha=0.3, color='forestgreen', linewidth=0.8, label='Raw')
    ax.plot(df['episode'], rolling_mean(df['task_sr'], 50) * 100, color='darkgreen', linewidth=2, label='Smoothed (50-ep)')
    ax.set_ylabel('Task Success Rate (%)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (80%)')
    
    # 3. Subtask Success Rate
    ax = axes[2]
    ax.plot(df['episode'], df['subtask_sr'] * 100, alpha=0.3, color='orange', linewidth=0.8, label='Raw')
    ax.plot(df['episode'], rolling_mean(df['subtask_sr'], 50) * 100, color='darkorange', linewidth=2, label='Smoothed (50-ep)')
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Subtask Success Rate (%)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_convergence.png")


def plot_policy_evolution(df, output_dir):
    """
    绘制策略演化 (Stacked Area Chart: Local/RSU/V2V)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 滑动平均
    local_smooth = rolling_mean(df['ratio_local'], 50)
    rsu_smooth = rolling_mean(df['ratio_rsu'], 50)
    v2v_smooth = rolling_mean(df['ratio_v2v'], 50)
    
    # 堆叠面积图
    ax.fill_between(df['episode'], 0, local_smooth, label='Local', alpha=0.7, color='#1f77b4')
    ax.fill_between(df['episode'], local_smooth, local_smooth + rsu_smooth, label='RSU', alpha=0.7, color='#ff7f0e')
    ax.fill_between(df['episode'], local_smooth + rsu_smooth, 1.0, label='V2V', alpha=0.7, color='#2ca02c')
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Action Ratio', fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.set_title('Policy Evolution (Action Distribution Over Training)', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_policy_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_policy_evolution.png")


def plot_physics(df, output_dir):
    """
    绘制物理指标 (Latency and Energy)
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 检查列是否存在
    if 'latency' not in df.columns or 'energy' not in df.columns:
        print("⚠ Warning: 'latency' or 'energy' columns not found, skipping physics plot.")
        return
    
    # 过滤掉无效值
    valid_mask = (df['latency'].notna()) & (df['energy'].notna())
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print("⚠ Warning: No valid latency/energy data, skipping physics plot.")
        return
    
    # Latency (左轴)
    color1 = 'tab:blue'
    ax1.set_xlabel('Episode', fontweight='bold')
    ax1.set_ylabel('Avg Latency (s)', color=color1, fontweight='bold')
    ax1.plot(df_valid['episode'], rolling_mean(df_valid['latency'], 50), color=color1, linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Energy (右轴)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Avg Energy (J)', color=color2, fontweight='bold')
    ax2.plot(df_valid['episode'], rolling_mean(df_valid['energy'], 50), color=color2, linewidth=2, label='Energy', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('Physical Metrics (Latency & Energy)', fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_physics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_physics.png")


def plot_training_diagnostics(df, output_dir):
    """
    绘制训练诊断 (Actor Loss, Critic Loss, Entropy)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Actor Loss
    ax = axes[0]
    if 'actor_loss' in df.columns:
        valid_mask = df['actor_loss'].notna()
        df_valid = df[valid_mask]
        ax.plot(df_valid['episode'], df_valid['actor_loss'], alpha=0.3, color='crimson', linewidth=0.8, label='Raw')
        ax.plot(df_valid['episode'], rolling_mean(df_valid['actor_loss'], 50), color='darkred', linewidth=2, label='Smoothed')
        ax.set_ylabel('Actor Loss', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Actor Loss not available', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Training Diagnostics', fontweight='bold', fontsize=14)
    
    # 2. Critic Loss
    ax = axes[1]
    if 'critic_loss' in df.columns:
        valid_mask = df['critic_loss'].notna()
        df_valid = df[valid_mask]
        ax.plot(df_valid['episode'], df_valid['critic_loss'], alpha=0.3, color='purple', linewidth=0.8, label='Raw')
        ax.plot(df_valid['episode'], rolling_mean(df_valid['critic_loss'], 50), color='indigo', linewidth=2, label='Smoothed')
        ax.set_ylabel('Critic Loss', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Critic Loss not available', ha='center', va='center', transform=ax.transAxes)
    
    # 3. Entropy
    ax = axes[2]
    if 'entropy' in df.columns:
        valid_mask = df['entropy'].notna()
        df_valid = df[valid_mask]
        ax.plot(df_valid['episode'], df_valid['entropy'], alpha=0.3, color='teal', linewidth=0.8, label='Raw')
        ax.plot(df_valid['episode'], rolling_mean(df_valid['entropy'], 50), color='darkslategray', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Entropy', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Entropy not available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_training.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_training.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training results from CSV")
    parser.add_argument('--log-file', type=str, required=True, help='Path to training_stats.csv')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots (default: same as log-file)')
    args = parser.parse_args()
    
    # 读取CSV
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return
    
    df = pd.read_csv(args.log_file)
    print(f"✓ Loaded {len(df)} episodes from {args.log_file}")
    
    # 输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.log_file), '..', 'plots')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ Output directory: {args.output_dir}")
    
    # 检查必要列
    required_cols = ['episode', 'reward', 'task_sr', 'subtask_sr', 'ratio_local', 'ratio_rsu', 'ratio_v2v']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # 生成图表
    print("\n[Generating Plots]")
    plot_convergence(df, args.output_dir)
    plot_policy_evolution(df, args.output_dir)
    plot_physics(df, args.output_dir)
    plot_training_diagnostics(df, args.output_dir)
    
    print(f"\n✓ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

