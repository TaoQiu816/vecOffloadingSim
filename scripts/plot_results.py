"""
[绘图脚本] plot_results.py
Training Results Visualization Script

作用 (Purpose):
读取训练过程的CSV文件，生成论文级别的可视化图表，包括收敛曲线、策略演化、物理指标和训练诊断。
支持baseline对比曲线。

使用方法 (Usage):
    python scripts/plot_results.py --log-file logs/run_YYYYMMDD_HHMMSS/training_stats.csv --output-dir plots/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# 设置绘图风格
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# 配色方案
COLORS = {
    'primary': '#2563eb',      # 蓝色 - 主曲线
    'secondary': '#16a34a',    # 绿色 - 次要曲线
    'accent': '#f59e0b',       # 橙色 - 强调
    'danger': '#dc2626',       # 红色 - 警告
    'muted': '#6b7280',        # 灰色 - 辅助
    'Random': '#e74c3c',       # Baseline: Random
    'Local-Only': '#95a5a6',   # Baseline: Local
    'Greedy': '#f39c12',       # Baseline: Greedy
}


def rolling_mean(data, window=50):
    """计算滚动平均"""
    if len(data) < window:
        window = max(1, len(data) // 10)
    return data.rolling(window=window, min_periods=1).mean()


def load_baseline_data(training_csv, max_episode=None):
    """
    加载baseline数据（从同目录的baseline_stats.csv）并扩展为完整曲线
    使用forward fill插值，确保baseline在图中显示为完整曲线而非散点
    """
    baseline_path = os.path.join(os.path.dirname(training_csv), 'baseline_stats.csv')
    if not os.path.exists(baseline_path):
        return None
    
    df_baseline_raw = pd.read_csv(baseline_path)
    if df_baseline_raw.empty:
        return None
    
    # 如果没有指定max_episode，从训练数据推断
    if max_episode is None:
        training_df = pd.read_csv(training_csv)
        max_episode = training_df['episode'].max() if not training_df.empty else 100
    
    # 扩展每个policy的数据为完整episode范围
    expanded_rows = []
    for policy in df_baseline_raw['policy'].unique():
        policy_data = df_baseline_raw[df_baseline_raw['policy'] == policy].copy()
        policy_data = policy_data.set_index('episode')
        # 创建完整episode范围的索引
        full_idx = pd.Index(range(1, max_episode + 1), name='episode')
        # 重新索引并forward fill
        policy_expanded = policy_data.reindex(full_idx).ffill().bfill()
        policy_expanded['policy'] = policy
        policy_expanded = policy_expanded.reset_index()
        expanded_rows.append(policy_expanded)
    
    return pd.concat(expanded_rows, ignore_index=True)


def plot_convergence_with_baseline(df, df_baseline, output_dir):
    """
    绘制收敛曲线 (Reward + Success Rate) 包含Baseline对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Reward with Baseline
    ax = axes[0, 0]
    ax.plot(df['episode'], df['reward_mean'], alpha=0.15, color=COLORS['primary'], linewidth=0.8)
    ax.plot(df['episode'], rolling_mean(df['reward_mean'], 50), 
            color=COLORS['primary'], linewidth=2.5, label='MAPPO')
    
    if df_baseline is not None:
        for policy in ['Random', 'Local-Only', 'Greedy']:
            policy_data = df_baseline[df_baseline['policy'] == policy].sort_values('episode')
            if not policy_data.empty:
                # 使用平滑曲线绘制baseline，与MAPPO风格一致
                y_smooth = rolling_mean(policy_data['reward_mean'], 50)
                ax.plot(policy_data['episode'], y_smooth,
                       color=COLORS.get(policy, 'gray'), linestyle='--',
                       linewidth=2, label=f'{policy}', alpha=0.85)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (per step)')
    ax.set_title('Reward Convergence', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 2. Task Success Rate with Baseline
    ax = axes[0, 1]
    ax.plot(df['episode'], df['task_sr'] * 100, alpha=0.15, color=COLORS['secondary'], linewidth=0.8)
    ax.plot(df['episode'], rolling_mean(df['task_sr'], 50) * 100, 
            color=COLORS['secondary'], linewidth=2.5, label='MAPPO')
    
    if df_baseline is not None:
        for policy in ['Random', 'Local-Only', 'Greedy']:
            policy_data = df_baseline[df_baseline['policy'] == policy].sort_values('episode')
            if not policy_data.empty and 'task_sr' in policy_data.columns:
                # 使用平滑曲线绘制baseline，与MAPPO风格一致
                y_smooth = rolling_mean(policy_data['task_sr'], 50) * 100
                ax.plot(policy_data['episode'], y_smooth,
                       color=COLORS.get(policy, 'gray'), linestyle='--',
                       linewidth=2, label=f'{policy}', alpha=0.85)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Task Success Rate (%)')
    ax.set_title('Task Success Rate', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(y=80, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Target 80%')
    ax.set_ylim([0, 105])
    
    # 3. Vehicle Success Rate (DAG Level)
    ax = axes[1, 0]
    ax.plot(df['episode'], df['vehicle_sr'] * 100, alpha=0.15, color=COLORS['primary'], linewidth=0.8)
    ax.plot(df['episode'], rolling_mean(df['vehicle_sr'], 50) * 100, 
            color=COLORS['primary'], linewidth=2.5, label='V_SR (MAPPO)')
    ax.plot(df['episode'], rolling_mean(df['subtask_sr'], 50) * 100, 
            color=COLORS['secondary'], linewidth=2, linestyle='-.', label='S_SR (Subtask)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Vehicle & Subtask Success Rate', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim([0, 105])
    
    # 4. Deadline Misses
    ax = axes[1, 1]
    if 'deadline_misses' in df.columns:
        ax.bar(df['episode'], df['deadline_misses'], alpha=0.5, color=COLORS['danger'], label='Deadline Misses')
        ax.plot(df['episode'], rolling_mean(df['deadline_misses'], 50), 
                color=COLORS['danger'], linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Deadline Misses (count)')
    ax.set_title('Deadline Miss Count', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_convergence_baseline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_convergence_baseline.png")


def plot_policy_evolution(df, output_dir):
    """
    绘制策略演化 (Stacked Area Chart: Local/RSU/V2V)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 滑动平均
    local_smooth = rolling_mean(df['ratio_local'], 50)
    rsu_smooth = rolling_mean(df['ratio_rsu'], 50)
    v2v_smooth = rolling_mean(df['ratio_v2v'], 50)
    
    # 堆叠面积图
    ax.fill_between(df['episode'], 0, local_smooth * 100, 
                    label='Local', alpha=0.7, color='#3b82f6')
    ax.fill_between(df['episode'], local_smooth * 100, (local_smooth + rsu_smooth) * 100, 
                    label='RSU', alpha=0.7, color='#f59e0b')
    ax.fill_between(df['episode'], (local_smooth + rsu_smooth) * 100, 100, 
                    label='V2V', alpha=0.7, color='#10b981')
    
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Action Ratio (%)', fontweight='bold')
    ax.set_ylim([0, 100])
    ax.set_title('Policy Evolution (Offloading Decision Distribution)', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', frameon=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_policy_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_policy_evolution.png")


def plot_training_diagnostics(df, output_dir):
    """
    绘制训练诊断 (Actor Loss, Critic Loss, Entropy)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Actor Loss
    ax = axes[0, 0]
    if 'actor_loss' in df.columns:
        valid_mask = df['actor_loss'].notna()
        df_valid = df[valid_mask]
        if len(df_valid) > 0:
            ax.plot(df_valid['episode'], df_valid['actor_loss'], 
                   alpha=0.2, color=COLORS['danger'], linewidth=0.8)
            ax.plot(df_valid['episode'], rolling_mean(df_valid['actor_loss'], 50), 
                   color=COLORS['danger'], linewidth=2.5, label='Actor Loss')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Actor (Policy) Loss', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    # 2. Critic Loss
    ax = axes[0, 1]
    if 'critic_loss' in df.columns:
        valid_mask = df['critic_loss'].notna()
        df_valid = df[valid_mask]
        if len(df_valid) > 0:
            ax.plot(df_valid['episode'], df_valid['critic_loss'], 
                   alpha=0.2, color=COLORS['primary'], linewidth=0.8)
            ax.plot(df_valid['episode'], rolling_mean(df_valid['critic_loss'], 50), 
                   color=COLORS['primary'], linewidth=2.5, label='Critic Loss')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Critic (Value) Loss', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    # 3. Entropy
    ax = axes[1, 0]
    if 'entropy' in df.columns:
        valid_mask = df['entropy'].notna()
        df_valid = df[valid_mask]
        if len(df_valid) > 0:
            ax.plot(df_valid['episode'], df_valid['entropy'], 
                   alpha=0.2, color=COLORS['secondary'], linewidth=0.8)
            ax.plot(df_valid['episode'], rolling_mean(df_valid['entropy'], 50), 
                   color=COLORS['secondary'], linewidth=2.5, label='Policy Entropy')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy (Exploration)', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    # 4. KL Divergence & Clip Fraction
    ax = axes[1, 1]
    ax2 = ax.twinx()
    if 'approx_kl' in df.columns:
        valid_mask = df['approx_kl'].notna()
        df_valid = df[valid_mask]
        if len(df_valid) > 0:
            ax.plot(df_valid['episode'], rolling_mean(df_valid['approx_kl'], 50), 
                   color=COLORS['accent'], linewidth=2.5, label='Approx KL')
    if 'clip_frac' in df.columns:
        valid_mask = df['clip_frac'].notna()
        df_valid = df[valid_mask]
        if len(df_valid) > 0:
            ax2.plot(df_valid['episode'], rolling_mean(df_valid['clip_frac'], 50), 
                    color=COLORS['muted'], linewidth=2, linestyle='--', label='Clip Fraction')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Approx KL', color=COLORS['accent'])
    ax2.set_ylabel('Clip Fraction', color=COLORS['muted'])
    ax.set_title('PPO Diagnostics', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_training_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_training_diagnostics.png")


def plot_physical_metrics(df, output_dir):
    """
    绘制物理性能指标 (Task Duration, Service Rate, etc.)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Task Duration
    ax = axes[0, 0]
    if 'task_duration_mean' in df.columns:
        valid = df['task_duration_mean'].notna()
        ax.plot(df.loc[valid, 'episode'], df.loc[valid, 'task_duration_mean'], 
               alpha=0.2, color=COLORS['primary'])
        ax.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'task_duration_mean'], 50), 
               linewidth=2.5, color=COLORS['primary'], label='Mean Duration')
    if 'task_duration_p95' in df.columns:
        valid = df['task_duration_p95'].notna()
        ax.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'task_duration_p95'], 50), 
               linewidth=2, color=COLORS['accent'], linestyle='--', label='P95 Duration')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Task Duration (s)')
    ax.set_title('Task Completion Time', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    # 2. Service Rate & Idle Fraction
    ax = axes[0, 1]
    ax2 = ax.twinx()
    if 'service_rate_ghz' in df.columns:
        valid = df['service_rate_ghz'].notna()
        ax.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'service_rate_ghz'], 50), 
               linewidth=2.5, color=COLORS['secondary'], label='Service Rate (GHz)')
    if 'idle_fraction' in df.columns:
        valid = df['idle_fraction'].notna()
        ax2.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'idle_fraction'], 50) * 100, 
                linewidth=2, color=COLORS['accent'], linestyle='--', label='Idle Fraction (%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Service Rate (GHz)', color=COLORS['secondary'])
    ax2.set_ylabel('Idle Fraction (%)', color=COLORS['accent'])
    ax.set_title('Resource Utilization', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)
    
    # 3. TX Created & NoTX
    ax = axes[1, 0]
    if 'tx_created' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['tx_created'], 50), 
               linewidth=2.5, color=COLORS['primary'], label='TX Created')
    if 'same_node_no_tx' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['same_node_no_tx'], 50), 
               linewidth=2, color=COLORS['muted'], linestyle='--', label='Same Node (NoTX)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count per Episode')
    ax.set_title('Transmission Statistics', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    # 4. Completed Tasks
    ax = axes[1, 1]
    if 'completed_tasks' in df.columns:
        ax.plot(df['episode'], df['completed_tasks'], alpha=0.2, color=COLORS['secondary'])
        ax.plot(df['episode'], rolling_mean(df['completed_tasks'], 50), 
               linewidth=2.5, color=COLORS['secondary'], label='Completed Tasks')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    ax.set_title('Completed Tasks per Episode', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_physical_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_physical_metrics.png")


def plot_summary_dashboard(df, df_baseline, output_dir):
    """
    绘制综合仪表板（单张图包含关键指标）
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 使用GridSpec进行更灵活的布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Reward (大图)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(df['episode'], df['reward_mean'], alpha=0.15, color=COLORS['primary'])
    ax.plot(df['episode'], rolling_mean(df['reward_mean'], 50), 
            color=COLORS['primary'], linewidth=3, label='MAPPO')
    if df_baseline is not None:
        for policy in ['Random', 'Local-Only', 'Greedy']:
            policy_data = df_baseline[df_baseline['policy'] == policy]
            if not policy_data.empty:
                ax.plot(policy_data['episode'], policy_data['reward_mean'],
                       color=COLORS.get(policy, 'gray'), linestyle='--',
                       linewidth=2, marker='s', markersize=5, label=policy, alpha=0.8)
    ax.set_title('Reward Convergence', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward/Step')
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 2. Success Rate (大图)
    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(df['episode'], rolling_mean(df['task_sr'], 50) * 100, 
            color=COLORS['secondary'], linewidth=3, label='MAPPO T_SR')
    if df_baseline is not None:
        for policy in ['Random', 'Local-Only', 'Greedy']:
            policy_data = df_baseline[df_baseline['policy'] == policy]
            if not policy_data.empty and 'task_sr' in policy_data.columns:
                ax.plot(policy_data['episode'], policy_data['task_sr'] * 100,
                       color=COLORS.get(policy, 'gray'), linestyle='--',
                       linewidth=2, marker='s', markersize=5, label=policy, alpha=0.8)
    ax.axhline(y=80, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    ax.set_title('Task Success Rate', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim([0, 105])
    ax.legend(loc='best', framealpha=0.9)
    
    # 3. Policy Evolution
    ax = fig.add_subplot(gs[1, :2])
    local_smooth = rolling_mean(df['ratio_local'], 50)
    rsu_smooth = rolling_mean(df['ratio_rsu'], 50)
    ax.fill_between(df['episode'], 0, local_smooth * 100, label='Local', alpha=0.7, color='#3b82f6')
    ax.fill_between(df['episode'], local_smooth * 100, (local_smooth + rsu_smooth) * 100, 
                    label='RSU', alpha=0.7, color='#f59e0b')
    ax.fill_between(df['episode'], (local_smooth + rsu_smooth) * 100, 100, 
                    label='V2V', alpha=0.7, color='#10b981')
    ax.set_title('Policy Evolution', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Ratio (%)')
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right', framealpha=0.9)
    
    # 4. Training Loss
    ax = fig.add_subplot(gs[1, 2])
    if 'actor_loss' in df.columns:
        valid = df['actor_loss'].notna()
        ax.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'actor_loss'], 50), 
               color=COLORS['danger'], linewidth=2, label='Actor')
    if 'critic_loss' in df.columns:
        valid = df['critic_loss'].notna()
        ax.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'critic_loss'], 50), 
               color=COLORS['primary'], linewidth=2, label='Critic')
    ax.set_title('Training Loss', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.legend(loc='best', framealpha=0.9)
    
    # 5. Entropy
    ax = fig.add_subplot(gs[1, 3])
    if 'entropy' in df.columns:
        valid = df['entropy'].notna()
        ax.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'entropy'], 50), 
               color=COLORS['secondary'], linewidth=2.5)
    ax.set_title('Policy Entropy', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    
    # 6. Deadline Misses
    ax = fig.add_subplot(gs[2, 0])
    if 'deadline_misses' in df.columns:
        ax.bar(df['episode'], df['deadline_misses'], alpha=0.4, color=COLORS['danger'])
        ax.plot(df['episode'], rolling_mean(df['deadline_misses'], 50), 
               color=COLORS['danger'], linewidth=2)
    ax.set_title('Deadline Misses', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    
    # 7. Service Rate
    ax = fig.add_subplot(gs[2, 1])
    if 'service_rate_ghz' in df.columns:
        valid = df['service_rate_ghz'].notna()
        ax.plot(df.loc[valid, 'episode'], rolling_mean(df.loc[valid, 'service_rate_ghz'], 50), 
               color=COLORS['secondary'], linewidth=2.5)
    ax.set_title('Service Rate', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('GHz')
    
    # 8. Final Statistics (文字统计)
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    # 计算最终统计
    last_n = min(50, len(df))
    final_reward = df['reward_mean'].iloc[-last_n:].mean()
    final_task_sr = df['task_sr'].iloc[-last_n:].mean() * 100
    final_vehicle_sr = df['vehicle_sr'].iloc[-last_n:].mean() * 100
    final_local = df['ratio_local'].iloc[-last_n:].mean() * 100
    final_rsu = df['ratio_rsu'].iloc[-last_n:].mean() * 100
    final_v2v = df['ratio_v2v'].iloc[-last_n:].mean() * 100
    
    stats_text = f"""
╔══════════════════════════════════════════════════════════════╗
║               Final Statistics (Last {last_n} Episodes)               ║
╠══════════════════════════════════════════════════════════════╣
║  Reward (per step):     {final_reward:>8.3f}                           ║
║  Task Success Rate:     {final_task_sr:>8.1f}%                          ║
║  Vehicle Success Rate:  {final_vehicle_sr:>8.1f}%                          ║
╠══════════════════════════════════════════════════════════════╣
║  Offloading Distribution:                                    ║
║    • Local:  {final_local:>5.1f}%                                       ║
║    • RSU:    {final_rsu:>5.1f}%                                       ║
║    • V2V:    {final_v2v:>5.1f}%                                       ║
╚══════════════════════════════════════════════════════════════╝
"""
    ax.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=11, 
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.savefig(os.path.join(output_dir, 'fig_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_summary_dashboard.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training results from CSV")
    parser.add_argument('--log-file', type=str, required=True, help='Path to training_stats.csv')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    # 读取CSV
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return
    
    df = pd.read_csv(args.log_file)
    print(f"✓ Loaded {len(df)} episodes from {args.log_file}")
    
    # 加载baseline数据
    df_baseline = load_baseline_data(args.log_file)
    if df_baseline is not None:
        print(f"✓ Loaded baseline data with {len(df_baseline)} entries")
    else:
        print("⚠ No baseline data found")
    
    # 输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.log_file), '..', 'plots')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ Output directory: {args.output_dir}")
    
    # 检查必要列
    required_cols = ['episode', 'reward_mean', 'task_sr', 'subtask_sr', 'vehicle_sr',
                     'ratio_local', 'ratio_rsu', 'ratio_v2v']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # 生成图表
    print("\n[Generating Plots]")
    plot_convergence_with_baseline(df, df_baseline, args.output_dir)
    plot_policy_evolution(df, args.output_dir)
    plot_training_diagnostics(df, args.output_dir)
    plot_physical_metrics(df, args.output_dir)
    plot_summary_dashboard(df, df_baseline, args.output_dir)
    
    print(f"\n✓ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
