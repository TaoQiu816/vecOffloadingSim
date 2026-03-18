#!/usr/bin/env python3
"""分析run_20260318_145633训练结果"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
run_dir = Path('runs/run_20260318_145633')
df = pd.read_csv(run_dir / 'logs/training_stats.csv')

# 创建输出目录
plot_dir = run_dir / 'analysis_plots'
plot_dir.mkdir(exist_ok=True)

print(f"训练集数: {len(df)}")
print(f"\n=== 动作分布统计 ===")
print(df[['ratio_local', 'ratio_rsu', 'ratio_v2v']].describe())
print(f"\n最后100集动作分布:")
print(df.tail(100)[['ratio_local', 'ratio_rsu', 'ratio_v2v']].mean())

# 1. 动作分布演化图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 动作比例堆叠图
ax = axes[0, 0]
window = 50
df_smooth = df[['ratio_local', 'ratio_rsu', 'ratio_v2v']].rolling(window, min_periods=1).mean()
ax.fill_between(df.index, 0, df_smooth['ratio_local'], label='Local', alpha=0.7)
ax.fill_between(df.index, df_smooth['ratio_local'],
                df_smooth['ratio_local'] + df_smooth['ratio_rsu'],
                label='RSU', alpha=0.7)
ax.fill_between(df.index, df_smooth['ratio_local'] + df_smooth['ratio_rsu'], 1,
                label='V2V', alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('Action Ratio')
ax.set_title('动作分布演化 (50-ep滑动平均)')
ax.legend()
ax.grid(alpha=0.3)

# 1.2 RSU占比单独展示
ax = axes[0, 1]
ax.plot(df.index, df['ratio_rsu'].rolling(50, min_periods=1).mean(),
        label='RSU Ratio (50-ep MA)', linewidth=2)
ax.axhline(y=0.95, color='r', linestyle='--', label='95%阈值', alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('RSU Ratio')
ax.set_title('RSU卸载占比')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

# 1.3 奖励演化
ax = axes[1, 0]
ax.plot(df.index, df['episode_reward'].rolling(50, min_periods=1).mean(),
        label='Episode Reward (50-ep MA)', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Episode Reward')
ax.set_title('累积奖励演化')
ax.legend()
ax.grid(alpha=0.3)

# 1.4 任务成功率
ax = axes[1, 1]
ax.plot(df.index, df['task_sr'].rolling(50, min_periods=1).mean(),
        label='Task Success Rate (50-ep MA)', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Success Rate')
ax.set_title('任务成功率')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(plot_dir / '1_action_reward_overview.png', dpi=150, bbox_inches='tight')
print(f"已保存: {plot_dir / '1_action_reward_overview.png'}")

# 2. 训练指标图
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 2.1 Actor Loss
ax = axes[0, 0]
ax.plot(df.index, df['actor_loss'].rolling(20, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Actor Loss')
ax.set_title('Actor Loss')
ax.grid(alpha=0.3)

# 2.2 Critic Loss
ax = axes[0, 1]
ax.plot(df.index, df['critic_loss'].rolling(20, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Critic Loss')
ax.set_title('Critic Loss')
ax.grid(alpha=0.3)

# 2.3 Entropy
ax = axes[0, 2]
ax.plot(df.index, df['entropy'].rolling(20, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Entropy')
ax.set_title('策略熵')
ax.grid(alpha=0.3)

# 2.4 KL Divergence
ax = axes[1, 0]
ax.plot(df.index, df['approx_kl'].rolling(20, min_periods=1).mean())
ax.axhline(y=0.05, color='r', linestyle='--', label='Target KL=0.05', alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('Approx KL')
ax.set_title('KL散度')
ax.legend()
ax.grid(alpha=0.3)

# 2.5 Clip Fraction
ax = axes[1, 1]
ax.plot(df.index, df['clip_frac'].rolling(20, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Clip Fraction')
ax.set_title('Clip比例')
ax.grid(alpha=0.3)

# 2.6 Learning Rate
ax = axes[1, 2]
ax.plot(df.index, df['lr'])
ax.set_xlabel('Episode')
ax.set_ylabel('Learning Rate')
ax.set_title('学习率衰减')
ax.grid(alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(plot_dir / '2_training_metrics.png', dpi=150, bbox_inches='tight')
print(f"已保存: {plot_dir / '2_training_metrics.png'}")

# 3. 奖励分解图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 Progress Reward
ax = axes[0, 0]
if 'mean_r_prog' in df.columns:
    ax.plot(df.index, df['mean_r_prog'].rolling(50, min_periods=1).mean())
else:
    ax.plot(df.index, df['reward_mean'].rolling(50, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Mean Reward')
ax.set_title('平均奖励')
ax.grid(alpha=0.3)

# 3.2 Terminal Reward
ax = axes[0, 1]
if 'mean_r_term' in df.columns:
    ax.plot(df.index, df['mean_r_term'].rolling(50, min_periods=1).mean())
else:
    ax.plot(df.index, df['reward_abs_mean'].rolling(50, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Mean R_term')
ax.set_title('终止奖励 (r_term)')
ax.grid(alpha=0.3)

# 3.3 Energy Cost
ax = axes[1, 0]
if 'mean_cost_power' in df.columns:
    ax.plot(df.index, df['mean_cost_power'].rolling(50, min_periods=1).mean())
else:
    ax.plot(df.index, df['energy_mean'].rolling(50, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Energy')
ax.set_title('能耗')
ax.grid(alpha=0.3)

# 3.4 奖励组成比例
ax = axes[1, 1]
window = 50
r_time_abs = df['abs_ratio_r_time'].rolling(window, min_periods=1).mean()
r_energy_abs = df['abs_ratio_r_energy'].rolling(window, min_periods=1).mean()
r_interf_abs = df['abs_ratio_r_interf'].rolling(window, min_periods=1).mean()
r_term_abs = df['abs_ratio_r_term'].rolling(window, min_periods=1).mean()

ax.plot(df.index, r_time_abs, label='Time', linewidth=2)
ax.plot(df.index, r_energy_abs, label='Energy', linewidth=2)
ax.plot(df.index, r_interf_abs, label='Interference', linewidth=2)
ax.plot(df.index, r_term_abs, label='Terminal', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Absolute Ratio')
ax.set_title('奖励组成比例 (绝对值)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / '3_reward_decomposition.png', dpi=150, bbox_inches='tight')
print(f"已保存: {plot_dir / '3_reward_decomposition.png'}")

# 4. 性能指标图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4.1 任务完成时间
ax = axes[0, 0]
ax.plot(df.index, df['task_duration_mean'].rolling(50, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Task Duration (s)')
ax.set_title('平均任务完成时间')
ax.grid(alpha=0.3)

# 4.2 能耗
ax = axes[0, 1]
ax.plot(df.index, df['energy_mean'].rolling(50, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Energy (J)')
ax.set_title('平均能耗')
ax.grid(alpha=0.3)

# 4.3 RSU队列长度
ax = axes[1, 0]
ax.plot(df.index, df['avg_rsu_queue'].rolling(50, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('RSU Queue (cycles)')
ax.set_title('RSU平均队列长度')
ax.grid(alpha=0.3)

# 4.4 发射功率
ax = axes[1, 1]
ax.plot(df.index, df['avg_power'].rolling(50, min_periods=1).mean())
ax.set_xlabel('Episode')
ax.set_ylabel('Power (W)')
ax.set_title('平均发射功率')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / '4_performance_metrics.png', dpi=150, bbox_inches='tight')
print(f"已保存: {plot_dir / '4_performance_metrics.png'}")

# 5. 统计分析
print("\n=== 阶段性统计 ===")
stages = [
    (0, 200, "初期 (0-200)"),
    (200, 500, "中期 (200-500)"),
    (500, 800, "后期 (500-800)"),
    (800, 1154, "末期 (800-1154)")
]

for start, end, label in stages:
    stage_df = df.iloc[start:end]
    print(f"\n{label}:")
    print(f"  RSU占比: {stage_df['ratio_rsu'].mean():.3f} ± {stage_df['ratio_rsu'].std():.3f}")
    print(f"  Local占比: {stage_df['ratio_local'].mean():.3f} ± {stage_df['ratio_local'].std():.3f}")
    print(f"  V2V占比: {stage_df['ratio_v2v'].mean():.3f} ± {stage_df['ratio_v2v'].std():.3f}")
    print(f"  Episode Reward: {stage_df['episode_reward'].mean():.3f} ± {stage_df['episode_reward'].std():.3f}")
    print(f"  Task Success Rate: {stage_df['task_sr'].mean():.3f}")
    print(f"  Task Duration: {stage_df['task_duration_mean'].mean():.3f}s")
    print(f"  Entropy: {stage_df['entropy'].mean():.4f}")

print("\n分析完成！")
