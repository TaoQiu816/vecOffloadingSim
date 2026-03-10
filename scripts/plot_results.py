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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

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
    'EFT': '#16a34a',          # Baseline: EFT
    'CP-EFT': '#0ea5e9',       # Baseline: CP-EFT
    'Static': '#7c3aed',       # Baseline: Static
}


def rolling_mean(data, window=50):
    """计算滚动平均"""
    if len(data) < window:
        window = max(1, len(data) // 10)
    return data.rolling(window=window, min_periods=1).mean()


def rolling_quantile(data, window=50, q=0.5):
    """计算滚动分位数，用于高噪声曲线的稳健趋势可视化"""
    if len(data) < window:
        window = max(1, len(data) // 10)
    return data.rolling(window=window, min_periods=1).quantile(q)


def maybe_add_legend(ax, **kwargs):
    """仅在当前坐标轴存在有效图例项时添加图例，避免空legend警告"""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _plot_baseline_hlines(ax, df_baseline, col, scale=1.0, n_ep_label=None):
    """
    P2 统一口径：baseline 策略以水平线绘制，避免 10ep 压缩在 3000ep x 轴左侧造成误导。
    scale: 乘以 100 用于百分比指标。
    n_ep_label: 用于图例后缀，如 "10-ep mean"。
    """
    if df_baseline is None or col not in df_baseline.columns:
        return
    n_ep = n_ep_label or f"{int(df_baseline['episode'].max())}-ep mean"
    _BL_COLORS = [
        '#dc2626', '#16a34a', '#d97706', '#7c3aed',
        '#0891b2', '#be185d', '#475569', '#ca8a04',
    ]
    for i, policy in enumerate(sorted(df_baseline['policy'].unique())):
        pdata = df_baseline[df_baseline['policy'] == policy]
        if pdata.empty or col not in pdata.columns:
            continue
        val = float(pdata[col].dropna().mean()) * scale
        ax.axhline(
            y=val,
            linestyle='--',
            linewidth=1.8,
            color=_BL_COLORS[i % len(_BL_COLORS)],
            alpha=0.85,
            label=f'{policy} ({n_ep})',
        )


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
    reward_q25 = rolling_quantile(df['reward_mean'], 50, 0.25)
    reward_q75 = rolling_quantile(df['reward_mean'], 50, 0.75)
    reward_med = rolling_quantile(df['reward_mean'], 20, 0.50)
    reward_ma50 = rolling_mean(df['reward_mean'], 50)
    ax.plot(df['episode'], df['reward_mean'], alpha=0.08, color=COLORS['primary'], linewidth=0.8, label='Raw Reward')
    ax.fill_between(df['episode'], reward_q25, reward_q75, color=COLORS['primary'], alpha=0.14, label='IQR (50-ep)')
    ax.plot(df['episode'], reward_med, color=COLORS['accent'], linewidth=1.8, label='Median (20-ep)')
    ax.plot(df['episode'], reward_ma50, color=COLORS['primary'], linewidth=2.5, label='Mean (50-ep)')
    
    _plot_baseline_hlines(ax, df_baseline, 'reward_mean')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (per step)')
    ax.set_title('Reward Convergence (Raw + Robust Trend)', fontweight='bold')
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 2. Task Success Rate with Baseline
    ax = axes[0, 1]
    ax.plot(df['episode'], df['task_sr'] * 100, alpha=0.15, color=COLORS['secondary'], linewidth=0.8)
    ax.plot(df['episode'], rolling_mean(df['task_sr'], 50) * 100, 
            color=COLORS['secondary'], linewidth=2.5, label='MAPPO')
    
    _plot_baseline_hlines(ax, df_baseline, 'task_sr', scale=100.0)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Task Success Rate (%)')
    ax.set_title('Task Success Rate', fontweight='bold')
    maybe_add_legend(ax, loc='best', framealpha=0.9)
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
    # 4. Completed Tasks
    ax = axes[1, 1]
    if 'completed_tasks' in df.columns:
        ax.plot(df['episode'], df['completed_tasks'], alpha=0.2, color=COLORS['secondary'])
        ax.plot(df['episode'], rolling_mean(df['completed_tasks'], 50), 
               linewidth=2.5, color=COLORS['secondary'], label='Completed Tasks')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    ax.set_title('Completed Tasks per Episode', fontweight='bold')
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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
    reward_q25 = rolling_quantile(df['reward_mean'], 50, 0.25)
    reward_q75 = rolling_quantile(df['reward_mean'], 50, 0.75)
    reward_med = rolling_quantile(df['reward_mean'], 20, 0.50)
    reward_ma50 = rolling_mean(df['reward_mean'], 50)
    ax.plot(df['episode'], df['reward_mean'], alpha=0.08, color=COLORS['primary'], label='Raw Reward')
    ax.fill_between(df['episode'], reward_q25, reward_q75, color=COLORS['primary'], alpha=0.14, label='IQR (50-ep)')
    ax.plot(df['episode'], reward_med, color=COLORS['accent'], linewidth=1.8, label='Median (20-ep)')
    ax.plot(df['episode'], reward_ma50, color=COLORS['primary'], linewidth=3, label='Mean (50-ep)')
    _plot_baseline_hlines(ax, df_baseline, 'reward_mean')
    ax.set_title('Reward Convergence (Raw + Robust Trend)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward/Step')
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 2. Success Rate (大图)
    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(df['episode'], rolling_mean(df['task_sr'], 50) * 100, 
            color=COLORS['secondary'], linewidth=3, label='MAPPO T_SR')
    _plot_baseline_hlines(ax, df_baseline, 'task_sr', scale=100.0)
    ax.axhline(y=80, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    ax.set_title('Task Success Rate', fontweight='bold', fontsize=14)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim([0, 105])
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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
    maybe_add_legend(ax, loc='best', framealpha=0.9)
    
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


def plot_latency_with_baseline(df, df_baseline, output_dir):
    """
    绘制时延相关指标对比（MAPPO曲线 + Baseline水平线/曲线）
    使用 mean_cft_est / episode_time_seconds / task_duration_mean 等（若存在）。
    """
    # Pick a primary latency metric for the top plot.
    latency_cols = [
        ("mean_cft_est", "Mean CFT Estimate (s)"),
        ("episode_time_seconds", "Episode Time (s)"),
        ("task_duration_mean", "Task Duration Mean (s)"),
    ]
    primary = next(((c, title) for c, title in latency_cols if c in df.columns), None)
    if primary is None:
        return
    col, title = primary

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Primary latency
    ax = axes[0, 0]
    ax.plot(df["episode"], rolling_mean(df[col], 50), color=COLORS["primary"], linewidth=2.5, label="MAPPO")
    _plot_baseline_hlines(ax, df_baseline, col)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Seconds")
    ax.legend(loc="best", framealpha=0.9)

    # 2) Deadline miss rate
    ax = axes[0, 1]
    if "deadline_miss_rate" in df.columns:
        ax.plot(df["episode"], rolling_mean(df["deadline_miss_rate"], 50) * 100, color=COLORS["danger"], linewidth=2.5, label="MAPPO")
        _plot_baseline_hlines(ax, df_baseline, "deadline_miss_rate", scale=100.0)
        ax.set_title("Deadline Miss Rate (%)", fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("%")
        ax.legend(loc="best", framealpha=0.9)

    # 3) Time limit rate
    ax = axes[1, 0]
    if "time_limit_rate" in df.columns:
        ax.plot(df["episode"], rolling_mean(df["time_limit_rate"], 50) * 100, color=COLORS["accent"], linewidth=2.5, label="MAPPO")
        _plot_baseline_hlines(ax, df_baseline, "time_limit_rate", scale=100.0)
        ax.set_title("Time Limit Rate (%)", fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("%")
        ax.legend(loc="best", framealpha=0.9)

    # 4) Power ratio (action) mean
    ax = axes[1, 1]
    power_col = None
    for c in ("power_ratio_mean", "avg_power"):
        if c in df.columns:
            power_col = c
            break
    if power_col is not None:
        ax.plot(df["episode"], rolling_mean(df[power_col], 50), color=COLORS["muted"], linewidth=2.5, label="MAPPO")
        base_power_col = "power_ratio_mean" if df_baseline is not None and "power_ratio_mean" in df_baseline.columns else "avg_power"
        _plot_baseline_hlines(ax, df_baseline, base_power_col)
        ax.set_title("Power Ratio (mean)", fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("a_power in [0,1]")
        ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_latency_with_baselines.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved: fig_latency_with_baselines.png")


def plot_constraints_and_health(df, output_dir):
    """
    绘制约束与训练健康指标
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1) 约束命中率
    ax = axes[0, 0]
    if 'deadline_miss_rate' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['deadline_miss_rate'], 50) * 100,
                color=COLORS['danger'], linewidth=2.5, label='Deadline Miss Rate')
    if 'time_limit_rate' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['time_limit_rate'], 50) * 100,
                color=COLORS['accent'], linewidth=2, linestyle='--', label='Time Limit Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Constraint Hit Rate', fontweight='bold')
    maybe_add_legend(ax, loc='best', framealpha=0.9)

    # 2) 非法与硬触发
    ax = axes[0, 1]
    if 'illegal_action_rate' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['illegal_action_rate'], 50) * 100,
                color='#ef4444', linewidth=2.5, label='Illegal Action Rate')
    if 'hard_trigger_rate' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['hard_trigger_rate'], 50) * 100,
                color='#8b5cf6', linewidth=2, linestyle='--', label='Hard Trigger Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Safety Trigger Rate', fontweight='bold')
    maybe_add_legend(ax, loc='best', framealpha=0.9)

    # 3) 资源压力
    ax = axes[1, 0]
    if 'avg_rsu_queue' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['avg_rsu_queue'], 50),
                color=COLORS['primary'], linewidth=2.5, label='Avg RSU Queue')
    if 'power_ratio_mean' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df['episode'], rolling_mean(df['power_ratio_mean'], 50),
                 color=COLORS['secondary'], linewidth=2, linestyle='--', label='Power Ratio Mean')
        ax2.set_ylabel('Power Ratio', color=COLORS['secondary'])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)
    else:
        maybe_add_legend(ax, loc='best', framealpha=0.9)
    ax.set_xlabel('Episode')
    ax.set_ylabel('RSU Queue Length')
    ax.set_title('Resource Pressure', fontweight='bold')

    # 4) PPO健康
    ax = axes[1, 1]
    if 'grad_norm' in df.columns:
        ax.plot(df['episode'], rolling_mean(df['grad_norm'], 50),
                color=COLORS['muted'], linewidth=2, label='Grad Norm')
    if 'active_ratio' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df['episode'], rolling_mean(df['active_ratio'], 50) * 100,
                 color=COLORS['secondary'], linewidth=2.5, label='Active Ratio')
        ax2.set_ylabel('Active Ratio (%)', color=COLORS['secondary'])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)
    else:
        maybe_add_legend(ax, loc='best', framealpha=0.9)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Grad Norm')
    ax.set_title('PPO Update Health', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_constraint_health.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig_constraint_health.png")


def plot_convergence_recheck(df, output_dir):
    """
    面向收敛性复核的补充图：
    把 reward、成功率、约束、终止原因、策略分布、PPO健康拆开看，
    避免单看 reward 均值曲线造成误判。
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    episodes = df['episode']

    # 1) Success rate
    ax = axes[0, 0]
    task_sr_raw = df['task_sr'] * 100.0
    task_sr_ma20 = rolling_mean(df['task_sr'], 20) * 100.0
    task_sr_ma50 = rolling_mean(df['task_sr'], 50) * 100.0
    subtask_sr_ma50 = rolling_mean(df['subtask_sr'], 50) * 100.0
    ax.plot(episodes, task_sr_raw, alpha=0.10, color=COLORS['secondary'], linewidth=0.8, label='Task SR Raw')
    ax.plot(episodes, task_sr_ma20, color=COLORS['accent'], linewidth=1.8, label='Task SR Mean (20-ep)')
    ax.plot(episodes, task_sr_ma50, color=COLORS['secondary'], linewidth=2.6, label='Task SR Mean (50-ep)')
    ax.plot(episodes, subtask_sr_ma50, color=COLORS['primary'], linewidth=2.0, linestyle='--', label='Subtask SR Mean (50-ep)')
    best_idx = int(task_sr_ma50.idxmax())
    ax.scatter([episodes.iloc[best_idx]], [task_sr_ma50.iloc[best_idx]], color=COLORS['danger'], s=35, zorder=5)
    ax.axvline(episodes.iloc[best_idx], color=COLORS['danger'], linestyle=':', linewidth=1.2, alpha=0.8)
    ax.set_title('Success Convergence Recheck', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim([0, 105])
    maybe_add_legend(ax, loc='best', framealpha=0.9)

    # 2) Reward recheck
    ax = axes[0, 1]
    reward_q25 = rolling_quantile(df['reward_mean'], 50, 0.25)
    reward_q75 = rolling_quantile(df['reward_mean'], 50, 0.75)
    reward_med = rolling_quantile(df['reward_mean'], 20, 0.50)
    reward_ma50 = rolling_mean(df['reward_mean'], 50)
    ax.plot(episodes, df['reward_mean'], alpha=0.08, color=COLORS['primary'], linewidth=0.8, label='Raw Reward')
    ax.fill_between(episodes, reward_q25, reward_q75, color=COLORS['primary'], alpha=0.14, label='IQR (50-ep)')
    ax.plot(episodes, reward_med, color=COLORS['accent'], linewidth=1.8, label='Median (20-ep)')
    ax.plot(episodes, reward_ma50, color=COLORS['primary'], linewidth=2.6, label='Mean (50-ep)')
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
    ax.set_title('Reward Recheck (High-Noise Friendly)', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward / Step')
    maybe_add_legend(ax, loc='best', framealpha=0.9)

    # 3) Constraint rates
    ax = axes[1, 0]
    if 'deadline_miss_rate' in df.columns:
        ax.plot(episodes, rolling_mean(df['deadline_miss_rate'], 20) * 100,
                color=COLORS['danger'], linewidth=1.8, alpha=0.9, label='Deadline Miss (20-ep)')
        ax.plot(episodes, rolling_mean(df['deadline_miss_rate'], 50) * 100,
                color=COLORS['danger'], linewidth=2.6, label='Deadline Miss (50-ep)')
    if 'time_limit_rate' in df.columns:
        ax.plot(episodes, rolling_mean(df['time_limit_rate'], 20) * 100,
                color=COLORS['accent'], linewidth=1.8, linestyle='--', alpha=0.9, label='Time Limit (20-ep)')
        ax.plot(episodes, rolling_mean(df['time_limit_rate'], 50) * 100,
                color=COLORS['accent'], linewidth=2.6, linestyle='-.', label='Time Limit (50-ep)')
    ax.set_title('Constraint Pressure', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate (%)')
    maybe_add_legend(ax, loc='best', framealpha=0.9)

    # 4) Termination mix
    ax = axes[1, 1]
    term_col = None
    for candidate in ('termination_reason', 'termination_reason_bucket', 'termination_reason_raw'):
        if candidate in df.columns:
            term_col = candidate
            break
    if term_col is not None:
        if 'time_limit' in set(df[term_col].astype(str)):
            tlimit = rolling_mean((df[term_col].astype(str) == 'time_limit').astype(float), 50) * 100
            ax.plot(episodes, tlimit, color=COLORS['accent'], linewidth=2.5, label='Time Limit (50-ep)')
        if 'idle' in set(df[term_col].astype(str)):
            idle = rolling_mean((df[term_col].astype(str) == 'idle').astype(float), 50) * 100
            ax.plot(episodes, idle, color=COLORS['muted'], linewidth=2.2, label='Idle (50-ep)')
        if 'success_all_done' in set(df[term_col].astype(str)):
            all_done = rolling_mean((df[term_col].astype(str) == 'success_all_done').astype(float), 50) * 100
            ax.plot(episodes, all_done, color=COLORS['secondary'], linewidth=2.5, label='All Done (50-ep)')
        if 'terminated' in set(df[term_col].astype(str)):
            terminated = rolling_mean((df[term_col].astype(str) == 'terminated').astype(float), 50) * 100
            ax.plot(episodes, terminated, color=COLORS['secondary'], linewidth=2.5, label='Terminated (50-ep)')
    ax.set_title('Termination Mix', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rolling Share (%)')
    ax.set_ylim([0, 105])
    maybe_add_legend(ax, loc='best', framealpha=0.9)

    # 5) Policy evolution as lines
    ax = axes[2, 0]
    ax.plot(episodes, rolling_mean(df['ratio_local'], 20) * 100,
            color='#3b82f6', linewidth=1.6, alpha=0.9, label='Local (20-ep)')
    ax.plot(episodes, rolling_mean(df['ratio_local'], 50) * 100,
            color='#1d4ed8', linewidth=2.4, label='Local (50-ep)')
    ax.plot(episodes, rolling_mean(df['ratio_rsu'], 20) * 100,
            color='#f59e0b', linewidth=1.6, alpha=0.9, label='RSU (20-ep)')
    ax.plot(episodes, rolling_mean(df['ratio_rsu'], 50) * 100,
            color='#b45309', linewidth=2.4, label='RSU (50-ep)')
    ax.plot(episodes, rolling_mean(df['ratio_v2v'], 20) * 100,
            color='#10b981', linewidth=1.6, alpha=0.9, label='V2V (20-ep)')
    ax.plot(episodes, rolling_mean(df['ratio_v2v'], 50) * 100,
            color='#047857', linewidth=2.4, label='V2V (50-ep)')
    ax.set_title('Policy Narrowing', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Decision Ratio (%)')
    ax.set_ylim([0, 105])
    ax.legend(loc='best', ncol=2, framealpha=0.9)

    # 6) PPO health
    ax = axes[2, 1]
    health_handles = []
    health_labels = []
    if 'entropy' in df.columns:
        line = ax.plot(episodes, rolling_mean(df['entropy'], 50),
                       color=COLORS['secondary'], linewidth=2.4, label='Entropy (50-ep)')
        health_handles.extend(line)
        health_labels.extend(['Entropy (50-ep)'])
    if 'critic_loss' in df.columns:
        line = ax.plot(episodes, rolling_mean(df['critic_loss'], 50),
                       color=COLORS['primary'], linewidth=2.2, label='Critic Loss (50-ep)')
        health_handles.extend(line)
        health_labels.extend(['Critic Loss (50-ep)'])
    if 'grad_norm' in df.columns:
        ax2 = ax.twinx()
        line = ax2.plot(episodes, rolling_mean(df['grad_norm'], 50),
                        color=COLORS['danger'], linewidth=2.0, linestyle='--', label='Grad Norm (50-ep)')
        health_handles.extend(line)
        health_labels.extend(['Grad Norm (50-ep)'])
        ax2.set_ylabel('Grad Norm', color=COLORS['danger'])
    ax.set_title('PPO Update Health', fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy / Critic Loss')
    if health_handles:
        ax.legend(health_handles, health_labels, loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_convergence_recheck.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_convergence_recheck.png")


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

    # Accept both "training_stats.csv" schema and the newer "metrics.csv" schema by normalizing key columns.
    # This keeps the plotting code stable across runs copied between machines / versions.
    if "task_sr" not in df.columns and "task_success_rate" in df.columns:
        df["task_sr"] = df["task_success_rate"]
    if "subtask_sr" not in df.columns and "subtask_success_rate" in df.columns:
        df["subtask_sr"] = df["subtask_success_rate"]
    if "vehicle_sr" not in df.columns:
        if "success_rate_end" in df.columns:
            df["vehicle_sr"] = df["success_rate_end"]
        elif "task_success_rate" in df.columns:
            df["vehicle_sr"] = df["task_success_rate"]
    if "ratio_local" not in df.columns:
        if "decision_frac_local" in df.columns:
            df["ratio_local"] = df["decision_frac_local"]
            df["ratio_rsu"] = df.get("decision_frac_rsu", 0.0)
            df["ratio_v2v"] = df.get("decision_frac_v2v", 0.0)
        elif "decision_local_frac" in df.columns:
            df["ratio_local"] = df["decision_local_frac"]
            df["ratio_rsu"] = df.get("decision_rsu_frac", 0.0)
            df["ratio_v2v"] = df.get("decision_v2v_frac", 0.0)

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
    plot_convergence_recheck(df, args.output_dir)
    plot_convergence_with_baseline(df, df_baseline, args.output_dir)
    plot_policy_evolution(df, args.output_dir)
    plot_training_diagnostics(df, args.output_dir)
    plot_physical_metrics(df, args.output_dir)
    plot_constraints_and_health(df, args.output_dir)
    plot_latency_with_baseline(df, df_baseline, args.output_dir)
    plot_summary_dashboard(df, df_baseline, args.output_dir)
    
    print(f"\n✓ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
