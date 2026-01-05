"""
生成完整训练分析图表

从episode_log.csv生成所有可用的指标图和分析图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def smooth(data, window=20):
    """滑动平均"""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

def plot_training_overview(df, output_dir):
    """训练总览（4x2子图）"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Training Overview', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    
    # 1. Total Reward
    ax = axes[0, 0]
    ax.plot(episodes, df['total_reward'], alpha=0.3, color='blue', linewidth=0.5)
    ax.plot(episodes, smooth(df['total_reward'], 50), color='darkblue', linewidth=2)
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Reward')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Task Success Rate
    ax = axes[0, 1]
    ax.plot(episodes, df['task_success_rate'] * 100, alpha=0.3, color='green', linewidth=0.5)
    ax.plot(episodes, smooth(df['task_success_rate'] * 100, 50), color='darkgreen', linewidth=2)
    ax.set_ylabel('Task Success Rate (%)')
    ax.set_title('Task Success Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Subtask Success Rate
    ax = axes[1, 0]
    ax.plot(episodes, df['subtask_success_rate'] * 100, alpha=0.3, color='orange', linewidth=0.5)
    ax.plot(episodes, smooth(df['subtask_success_rate'] * 100, 50), color='darkorange', linewidth=2)
    ax.set_ylabel('Subtask Success Rate (%)')
    ax.set_title('Subtask Success Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 4. V2V Subtask Success Rate
    ax = axes[1, 1]
    ax.plot(episodes, df['v2v_subtask_success_rate'] * 100, alpha=0.3, color='purple', linewidth=0.5)
    ax.plot(episodes, smooth(df['v2v_subtask_success_rate'] * 100, 50), color='darkviolet', linewidth=2)
    ax.set_ylabel('V2V Success Rate (%)')
    ax.set_title('V2V Subtask Success Rate')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 5. Decision Distribution - Local
    ax = axes[2, 0]
    ax.plot(episodes, df['decision_frac_local'] * 100, alpha=0.3, color='cyan', linewidth=0.5)
    ax.plot(episodes, smooth(df['decision_frac_local'] * 100, 50), color='darkcyan', linewidth=2)
    ax.set_ylabel('Local Decision (%)')
    ax.set_title('Local Execution Ratio')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 6. Decision Distribution - RSU
    ax = axes[2, 1]
    ax.plot(episodes, df['decision_frac_rsu'] * 100, alpha=0.3, color='red', linewidth=0.5)
    ax.plot(episodes, smooth(df['decision_frac_rsu'] * 100, 50), color='darkred', linewidth=2)
    ax.set_ylabel('RSU Decision (%)')
    ax.set_title('RSU Offloading Ratio')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 7. Decision Distribution - V2V
    ax = axes[3, 0]
    ax.plot(episodes, df['decision_frac_v2v'] * 100, alpha=0.3, color='magenta', linewidth=0.5)
    ax.plot(episodes, smooth(df['decision_frac_v2v'] * 100, 50), color='darkmagenta', linewidth=2)
    ax.set_ylabel('V2V Decision (%)')
    ax.set_title('V2V Offloading Ratio')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 8. Loss
    ax = axes[3, 1]
    valid_loss = df[df['loss'] > 0]
    if len(valid_loss) > 0:
        ax.plot(valid_loss['episode'], valid_loss['loss'], alpha=0.3, color='brown', linewidth=0.5)
        ax.plot(valid_loss['episode'], smooth(valid_loss['loss'], 20), color='saddlebrown', linewidth=2)
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_training_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_training_overview.png")

def plot_success_rate_detailed(df, output_dir):
    """成功率详细分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Success Rate Analysis', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    
    # 1. Task vs Subtask Success Rate
    ax = axes[0, 0]
    ax.plot(episodes, smooth(df['task_success_rate'] * 100, 50), 
            color='green', linewidth=2, label='Task SR')
    ax.plot(episodes, smooth(df['subtask_success_rate'] * 100, 50), 
            color='orange', linewidth=2, label='Subtask SR')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Task vs Subtask Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 2. Vehicle Success Rate
    ax = axes[0, 1]
    ax.plot(episodes, smooth(df['vehicle_success_rate'] * 100, 50), 
            color='blue', linewidth=2, label='Vehicle SR')
    ax.plot(episodes, smooth(df['veh_success_rate'] * 100, 50), 
            color='cyan', linewidth=2, label='Veh SR (alt)', alpha=0.7)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Vehicle-level Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Success Rate Distribution
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 21)
    ax.hist(df['task_success_rate'], bins=bins, alpha=0.5, color='green', label='Task SR')
    ax.hist(df['subtask_success_rate'], bins=bins, alpha=0.5, color='orange', label='Subtask SR')
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Frequency')
    ax.set_title('Success Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Success Rate vs Episode (bins)
    ax = axes[1, 1]
    n_bins = 10
    episode_bins = np.linspace(episodes.min(), episodes.max(), n_bins+1)
    bin_centers = []
    task_sr_bins = []
    subtask_sr_bins = []
    
    for i in range(n_bins):
        mask = (episodes >= episode_bins[i]) & (episodes < episode_bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((episode_bins[i] + episode_bins[i+1]) / 2)
            task_sr_bins.append(df[mask]['task_success_rate'].mean() * 100)
            subtask_sr_bins.append(df[mask]['subtask_success_rate'].mean() * 100)
    
    ax.bar(bin_centers, task_sr_bins, width=(episode_bins[1]-episode_bins[0])*0.8, 
           alpha=0.7, color='green', label='Task SR')
    ax.plot(bin_centers, subtask_sr_bins, 'o-', color='orange', 
            linewidth=2, markersize=8, label='Subtask SR')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Success Rate (%)')
    ax.set_title('Success Rate by Training Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_success_rate_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_success_rate_analysis.png")

def plot_decision_distribution(df, output_dir):
    """决策分布分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Decision Distribution Analysis', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    
    # 1. Stacked Area Chart
    ax = axes[0, 0]
    ax.fill_between(episodes, 0, smooth(df['decision_frac_local'] * 100, 50),
                     color='cyan', alpha=0.6, label='Local')
    ax.fill_between(episodes, smooth(df['decision_frac_local'] * 100, 50),
                     smooth((df['decision_frac_local'] + df['decision_frac_rsu']) * 100, 50),
                     color='red', alpha=0.6, label='RSU')
    ax.fill_between(episodes, smooth((df['decision_frac_local'] + df['decision_frac_rsu']) * 100, 50),
                     100, color='magenta', alpha=0.6, label='V2V')
    ax.set_ylabel('Decision Distribution (%)')
    ax.set_title('Decision Distribution Over Training')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # 2. Individual Lines
    ax = axes[0, 1]
    ax.plot(episodes, smooth(df['decision_frac_local'] * 100, 50), 
            color='cyan', linewidth=2.5, label='Local')
    ax.plot(episodes, smooth(df['decision_frac_rsu'] * 100, 50), 
            color='red', linewidth=2.5, label='RSU')
    ax.plot(episodes, smooth(df['decision_frac_v2v'] * 100, 50), 
            color='magenta', linewidth=2.5, label='V2V')
    ax.set_ylabel('Decision Fraction (%)')
    ax.set_title('Decision Trends')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Box Plot by Phase
    ax = axes[1, 0]
    n_phases = 5
    phase_size = len(df) // n_phases
    data_local = []
    data_rsu = []
    data_v2v = []
    labels = []
    
    for i in range(n_phases):
        start = i * phase_size
        end = (i+1) * phase_size if i < n_phases-1 else len(df)
        phase_df = df.iloc[start:end]
        data_local.append(phase_df['decision_frac_local'].values * 100)
        data_rsu.append(phase_df['decision_frac_rsu'].values * 100)
        data_v2v.append(phase_df['decision_frac_v2v'].values * 100)
        labels.append(f'Phase {i+1}')
    
    positions_local = np.arange(n_phases) * 3 - 0.8
    positions_rsu = np.arange(n_phases) * 3
    positions_v2v = np.arange(n_phases) * 3 + 0.8
    
    bp1 = ax.boxplot(data_local, positions=positions_local, widths=0.6, 
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_rsu, positions=positions_rsu, widths=0.6, 
                     patch_artist=True, showfliers=False)
    bp3 = ax.boxplot(data_v2v, positions=positions_v2v, widths=0.6, 
                     patch_artist=True, showfliers=False)
    
    for patch in bp1['boxes']:
        patch.set_facecolor('cyan')
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.6)
    for patch in bp3['boxes']:
        patch.set_facecolor('magenta')
        patch.set_alpha(0.6)
    
    ax.set_xticks(np.arange(n_phases) * 3)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Decision Fraction (%)')
    ax.set_title('Decision Distribution by Training Phase')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0], bp3['boxes'][0]], 
             ['Local', 'RSU', 'V2V'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Final Distribution (last 50 episodes)
    ax = axes[1, 1]
    last_50 = df.tail(50)
    means = [
        last_50['decision_frac_local'].mean() * 100,
        last_50['decision_frac_rsu'].mean() * 100,
        last_50['decision_frac_v2v'].mean() * 100
    ]
    stds = [
        last_50['decision_frac_local'].std() * 100,
        last_50['decision_frac_rsu'].std() * 100,
        last_50['decision_frac_v2v'].std() * 100
    ]
    colors = ['cyan', 'red', 'magenta']
    labels = ['Local', 'RSU', 'V2V']
    
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.7, 
                  capsize=10, error_kw={'linewidth': 2})
    ax.set_ylabel('Decision Fraction (%)')
    ax.set_title(f'Final Decision Distribution (Last 50 Episodes)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_decision_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_decision_distribution.png")

def plot_reward_analysis(df, output_dir):
    """奖励分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reward Analysis', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    
    # 1. Total Reward with std band
    ax = axes[0, 0]
    window = 50
    reward_smooth = smooth(df['total_reward'], window)
    reward_std = df['total_reward'].rolling(window=window, min_periods=1).std()
    
    ax.plot(episodes, df['total_reward'], alpha=0.2, color='blue', linewidth=0.5)
    ax.plot(episodes, reward_smooth, color='darkblue', linewidth=2.5, label='Mean')
    ax.fill_between(episodes, reward_smooth - reward_std, reward_smooth + reward_std,
                     color='blue', alpha=0.2, label='±1 std')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Reward with Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Avg Step Reward
    ax = axes[0, 1]
    ax.plot(episodes, df['avg_step_reward'], alpha=0.3, color='green', linewidth=0.5)
    ax.plot(episodes, smooth(df['avg_step_reward'], window), color='darkgreen', linewidth=2.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Avg Step Reward')
    ax.set_title('Average Step Reward')
    ax.grid(True, alpha=0.3)
    
    # 3. Reward Distribution
    ax = axes[1, 0]
    ax.hist(df['total_reward'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=df['total_reward'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["total_reward"].mean():.2f}')
    ax.axvline(x=df['total_reward'].median(), color='orange', linestyle='--', 
               linewidth=2, label=f'Median: {df["total_reward"].median():.2f}')
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Reward vs Success Rate
    ax = axes[1, 1]
    scatter = ax.scatter(df['task_success_rate'] * 100, df['total_reward'], 
                        c=df['episode'], cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Task Success Rate (%)')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward vs Success Rate')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Episode')
    
    # Add correlation
    corr = df['task_success_rate'].corr(df['total_reward'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_reward_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_reward_analysis.png")

def plot_multi_agent_metrics(df, output_dir):
    """多智能体指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Agent Metrics', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    
    # 1. Fairness
    ax = axes[0, 0]
    ax.plot(episodes, df['ma_fairness'], alpha=0.3, color='purple', linewidth=0.5)
    ax.plot(episodes, smooth(df['ma_fairness'], 50), color='darkviolet', linewidth=2.5)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Fairness')
    ax.set_ylabel('Jain\'s Fairness Index')
    ax.set_title('Multi-Agent Fairness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 2. Reward Gap
    ax = axes[0, 1]
    ax.plot(episodes, df['ma_reward_gap'], alpha=0.3, color='orange', linewidth=0.5)
    ax.plot(episodes, smooth(df['ma_reward_gap'], 50), color='darkorange', linewidth=2.5)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='No Gap')
    ax.set_ylabel('Reward Gap (Max - Min)')
    ax.set_title('Multi-Agent Reward Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Collaboration
    ax = axes[1, 0]
    ax.plot(episodes, df['ma_collaboration'], alpha=0.3, color='blue', linewidth=0.5)
    ax.plot(episodes, smooth(df['ma_collaboration'], 50), color='darkblue', linewidth=2.5)
    ax.set_ylabel('Collaboration Score')
    ax.set_title('Multi-Agent Collaboration')
    ax.grid(True, alpha=0.3)
    
    # 4. Agent Reward Range
    ax = axes[1, 1]
    ax.plot(episodes, smooth(df['max_agent_reward'], 50), 
            color='green', linewidth=2, label='Max Agent Reward')
    ax.plot(episodes, smooth(df['min_agent_reward'], 50), 
            color='red', linewidth=2, label='Min Agent Reward')
    ax.fill_between(episodes, 
                     smooth(df['min_agent_reward'], 50),
                     smooth(df['max_agent_reward'], 50),
                     color='gray', alpha=0.2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('Agent Reward')
    ax.set_title('Agent Reward Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_multi_agent_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_multi_agent_metrics.png")

def plot_system_metrics(df, output_dir):
    """系统指标"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('System Metrics', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    
    # 1. Average Power
    ax = axes[0, 0]
    ax.plot(episodes, df['avg_power'], alpha=0.3, color='red', linewidth=0.5)
    ax.plot(episodes, smooth(df['avg_power'], 50), color='darkred', linewidth=2.5)
    ax.set_ylabel('Average Power (W)')
    ax.set_title('Average Power Consumption')
    ax.grid(True, alpha=0.3)
    
    # 2. Average Queue Length
    ax = axes[0, 1]
    ax.plot(episodes, df['avg_queue_len'], alpha=0.3, color='brown', linewidth=0.5)
    ax.plot(episodes, smooth(df['avg_queue_len'], 50), color='saddlebrown', linewidth=2.5)
    ax.set_ylabel('Average Queue Length')
    ax.set_title('Average Queue Length')
    ax.grid(True, alpha=0.3)
    
    # 3. Episode Duration
    ax = axes[1, 0]
    durations = df['duration'].astype(float)
    ax.plot(episodes, durations, alpha=0.3, color='blue', linewidth=0.5)
    ax.plot(episodes, smooth(durations, 50), color='darkblue', linewidth=2.5)
    ax.set_ylabel('Duration (s)')
    ax.set_title('Episode Duration')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)
    
    # 4. Task & Vehicle Count
    ax = axes[1, 1]
    ax.plot(episodes, df['episode_vehicle_count'], 'o-', alpha=0.6, 
            color='green', markersize=3, label='Vehicles')
    ax.plot(episodes, df['episode_task_count'], 'o-', alpha=0.6, 
            color='orange', markersize=3, label='Tasks')
    ax.set_ylabel('Count')
    ax.set_title('Episode Vehicle & Task Count')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_system_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_system_metrics.png")

def plot_convergence_analysis(df, output_dir):
    """收敛性分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
    
    episodes = df['episode']
    
    # 1. Rolling Mean & Std of Reward
    ax = axes[0, 0]
    window = 50
    reward_mean = df['total_reward'].rolling(window=window, min_periods=1).mean()
    reward_std = df['total_reward'].rolling(window=window, min_periods=1).std()
    
    ax.plot(episodes, reward_mean, color='blue', linewidth=2.5, label='Mean')
    ax.fill_between(episodes, reward_mean - reward_std, reward_mean + reward_std,
                     color='blue', alpha=0.2)
    ax.set_ylabel('Reward (50-ep window)')
    ax.set_title('Reward Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Variance Trend
    ax = axes[0, 1]
    ax.plot(episodes, reward_std, color='red', linewidth=2.5)
    ax.set_ylabel('Reward Std Dev (50-ep window)')
    ax.set_title('Reward Variance Over Training')
    ax.grid(True, alpha=0.3)
    
    # 3. Success Rate Smoothness
    ax = axes[1, 0]
    sr_smooth = smooth(df['task_success_rate'] * 100, 50)
    sr_change = np.abs(np.diff(sr_smooth, prepend=sr_smooth[0]))
    ax.plot(episodes, sr_change, color='green', linewidth=1.5)
    ax.set_ylabel('|ΔSuccess Rate| (%)')
    ax.set_title('Success Rate Volatility')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.3)
    
    # 4. Policy Entropy (Decision Diversity)
    ax = axes[1, 1]
    # Calculate entropy from decision distribution
    eps = 1e-10
    p_local = df['decision_frac_local'] + eps
    p_rsu = df['decision_frac_rsu'] + eps
    p_v2v = df['decision_frac_v2v'] + eps
    entropy = -(p_local * np.log(p_local) + p_rsu * np.log(p_rsu) + p_v2v * np.log(p_v2v))
    max_entropy = np.log(3)  # Maximum entropy for 3 choices
    normalized_entropy = entropy / max_entropy
    
    ax.plot(episodes, normalized_entropy, alpha=0.3, color='purple', linewidth=0.5)
    ax.plot(episodes, smooth(normalized_entropy, 50), color='darkviolet', linewidth=2.5)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max Diversity')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='No Diversity')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Policy Diversity (Decision Entropy)')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_convergence_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_convergence_analysis.png")

def plot_baseline_comparison(df, output_dir):
    """与Baseline对比"""
    # 分离MAPPO和baseline数据
    mappo_df = df[~df['duration'].isin(['Random', 'Local-Only', 'Greedy'])].copy()
    baseline_df = df[df['duration'].isin(['Random', 'Local-Only', 'Greedy'])].copy()
    
    if len(baseline_df) == 0:
        print("⚠ No baseline data found, skipping baseline comparison plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline Comparison', fontsize=16, fontweight='bold')
    
    # 1. Task Success Rate
    ax = axes[0, 0]
    ax.plot(mappo_df['episode'], smooth(mappo_df['task_success_rate'] * 100, 50),
            color='blue', linewidth=2.5, label='MAPPO')
    
    for policy in ['Random', 'Local-Only', 'Greedy']:
        policy_df = baseline_df[baseline_df['duration'] == policy]
        if len(policy_df) > 0:
            ax.plot(policy_df['episode'], policy_df['task_success_rate'] * 100,
                   'o-', alpha=0.7, markersize=6, label=policy)
    
    ax.set_ylabel('Task Success Rate (%)')
    ax.set_title('Task Success Rate Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total Reward
    ax = axes[0, 1]
    ax.plot(mappo_df['episode'], smooth(mappo_df['total_reward'], 50),
            color='blue', linewidth=2.5, label='MAPPO')
    
    for policy in ['Random', 'Local-Only', 'Greedy']:
        policy_df = baseline_df[baseline_df['duration'] == policy]
        if len(policy_df) > 0:
            ax.plot(policy_df['episode'], policy_df['total_reward'],
                   'o-', alpha=0.7, markersize=6, label=policy)
    
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    
    # 3. Subtask Success Rate
    ax = axes[1, 0]
    ax.plot(mappo_df['episode'], smooth(mappo_df['subtask_success_rate'] * 100, 50),
            color='blue', linewidth=2.5, label='MAPPO')
    
    for policy in ['Random', 'Local-Only', 'Greedy']:
        policy_df = baseline_df[baseline_df['duration'] == policy]
        if len(policy_df) > 0:
            ax.plot(policy_df['episode'], policy_df['subtask_success_rate'] * 100,
                   'o-', alpha=0.7, markersize=6, label=policy)
    
    ax.set_ylabel('Subtask Success Rate (%)')
    ax.set_title('Subtask Success Rate Comparison')
    ax.set_xlabel('Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Bar Chart (Final Performance)
    ax = axes[1, 1]
    policies = ['MAPPO'] + ['Random', 'Local-Only', 'Greedy']
    task_sr_vals = []
    subtask_sr_vals = []
    
    # MAPPO last 50 episodes
    task_sr_vals.append(mappo_df.tail(50)['task_success_rate'].mean() * 100)
    subtask_sr_vals.append(mappo_df.tail(50)['subtask_success_rate'].mean() * 100)
    
    for policy in ['Random', 'Local-Only', 'Greedy']:
        policy_df = baseline_df[baseline_df['duration'] == policy]
        if len(policy_df) > 0:
            task_sr_vals.append(policy_df['task_success_rate'].mean() * 100)
            subtask_sr_vals.append(policy_df['subtask_success_rate'].mean() * 100)
        else:
            task_sr_vals.append(0)
            subtask_sr_vals.append(0)
    
    x = np.arange(len(policies))
    width = 0.35
    ax.bar(x - width/2, task_sr_vals, width, label='Task SR', alpha=0.8)
    ax.bar(x + width/2, subtask_sr_vals, width, label='Subtask SR', alpha=0.8)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_baseline_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig_baseline_comparison.png")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate all training plots')
    parser.add_argument('--run-dir', type=str, required=True, 
                       help='Run directory (e.g., runs/run_20260105_112351)')
    args = parser.parse_args()
    
    # 读取数据
    episode_log = os.path.join(args.run_dir, 'episode_log.csv')
    if not os.path.exists(episode_log):
        print(f"Error: {episode_log} not found")
        return
    
    df = pd.read_csv(episode_log)
    print(f"✓ Loaded {len(df)} episodes from {episode_log}")
    
    # 过滤MAPPO数据（用于部分图表）
    mappo_df = df[~df['duration'].isin(['Random', 'Local-Only', 'Greedy'])].copy()
    print(f"✓ MAPPO episodes: {len(mappo_df)}")
    
    # 创建输出目录
    output_dir = os.path.join(args.run_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}\n")
    
    # 生成所有图表
    print("[Generating Comprehensive Plots]")
    plot_training_overview(mappo_df, output_dir)
    plot_success_rate_detailed(mappo_df, output_dir)
    plot_decision_distribution(mappo_df, output_dir)
    plot_reward_analysis(mappo_df, output_dir)
    plot_multi_agent_metrics(mappo_df, output_dir)
    plot_system_metrics(mappo_df, output_dir)
    plot_convergence_analysis(mappo_df, output_dir)
    plot_baseline_comparison(df, output_dir)
    
    print(f"\n✓ All plots saved to: {output_dir}")
    print(f"  Total plots: 8")

if __name__ == "__main__":
    main()

