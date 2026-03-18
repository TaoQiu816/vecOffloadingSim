#!/usr/bin/env python3
"""对比分析两次训练运行"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取两次运行的数据
run1_dir = Path('runs/run_20260318_003251')
run2_dir = Path('runs/run_20260318_145633')

df1 = pd.read_csv(run1_dir / 'logs/training_stats.csv')
df2 = pd.read_csv(run2_dir / 'logs/training_stats.csv')

with open(run1_dir / 'logs/config_snapshot.json') as f:
    config1 = json.load(f)
with open(run2_dir / 'logs/config_snapshot.json') as f:
    config2 = json.load(f)

print("=" * 80)
print("两次训练运行对比分析")
print("=" * 80)

print(f"\nRun 1: {run1_dir.name}")
print(f"  训练集数: {len(df1)}")
print(f"  Git Commit: {config1['env']['GIT_COMMIT'][:8]}")

print(f"\nRun 2: {run2_dir.name}")
print(f"  训练集数: {len(df2)}")
print(f"  Git Commit: {config2['env']['GIT_COMMIT'][:8]}")

# ============================================================================
# 1. 关键参数对比
# ============================================================================
print("\n" + "=" * 80)
print("1. 关键参数对比")
print("=" * 80)

def compare_param(name, path, config1, config2):
    """递归获取嵌套配置参数"""
    keys = path.split('.')
    val1 = config1
    val2 = config2
    for key in keys:
        val1 = val1.get(key, 'N/A')
        val2 = val2.get(key, 'N/A')

    if val1 != val2:
        print(f"  {name:30s}: {val1:>15} → {val2:>15}")
        return True
    return False

print("\n【训练超参数】")
changed = False
changed |= compare_param("Critic学习率", "train_config.LR_CRITIC", config1, config2)
changed |= compare_param("Actor学习率", "train_config.LR_ACTOR", config1, config2)
changed |= compare_param("熵系数", "train_config.ENTROPY_COEF", config1, config2)
changed |= compare_param("最大训练集数", "train_config.MAX_EPISODES", config1, config2)
changed |= compare_param("Gamma", "train_config.GAMMA", config1, config2)
changed |= compare_param("GAE Lambda", "train_config.GAE_LAMBDA", config1, config2)
if not changed:
    print("  (无变化)")

print("\n【Logit Bias】")
changed = False
changed |= compare_param("RSU Bias", "train_config.LOGIT_BIAS_RSU", config1, config2)
changed |= compare_param("Local Bias", "train_config.LOGIT_BIAS_LOCAL", config1, config2)
changed |= compare_param("V2V Bias Init", "train_config.LOGIT_BIAS_V2V_INIT", config1, config2)
if not changed:
    print("  (无变化)")

print("\n【奖励函数】")
changed = False
changed |= compare_param("时间权重", "system_config.W_TIME", config1, config2)
changed |= compare_param("能耗权重", "system_config.W_ENERGY", config1, config2)
changed |= compare_param("干扰权重", "system_config.W_INTERF", config1, config2)
changed |= compare_param("奖励方案", "system_config.REWARD_SCHEME", config1, config2)
if not changed:
    print("  (无变化)")

print("\n【环境配置】")
changed = False
changed |= compare_param("车辆数", "system_config.NUM_VEHICLES", config1, config2)
changed |= compare_param("RSU数", "system_config.NUM_RSU", config1, config2)
changed |= compare_param("RSU频率", "system_config.F_RSU", config1, config2)
changed |= compare_param("RSU核心数", "system_config.RSU_NUM_PROCESSORS", config1, config2)
changed |= compare_param("车辆CPU最小", "system_config.MIN_VEHICLE_CPU_FREQ", config1, config2)
changed |= compare_param("车辆CPU最大", "system_config.MAX_VEHICLE_CPU_FREQ", config1, config2)
changed |= compare_param("V2V范围", "system_config.V2V_RANGE", config1, config2)
changed |= compare_param("RSU范围", "system_config.RSU_RANGE", config1, config2)
if not changed:
    print("  (无变化)")

print("\n【信任/延迟相关】")
changed = False
changed |= compare_param("信任启用", "system_config.TRUST_ENABLED", config1, config2)
changed |= compare_param("信任延迟步数", "system_config.TRUST_DELAY_STEPS", config1, config2)
changed |= compare_param("V2I切换延迟", "system_config.V2I_HANDOVER_DELAY_STEPS", config1, config2)
changed |= compare_param("HO冻结步数", "system_config.HO_FREEZE_STEPS", config1, config2)
if not changed:
    print("  (无变化)")

# ============================================================================
# 2. 训练结果对比
# ============================================================================
print("\n" + "=" * 80)
print("2. 训练结果对比")
print("=" * 80)

def print_stage_stats(df, label):
    """打印阶段性统计"""
    stages = [
        (0, 200, "初期"),
        (200, 500, "中期"),
        (500, min(800, len(df)), "后期"),
    ]

    print(f"\n{label}:")
    for start, end, stage_name in stages:
        if end > len(df):
            continue
        stage_df = df.iloc[start:end]
        print(f"  {stage_name} ({start}-{end}):")
        print(f"    RSU占比:    {stage_df['ratio_rsu'].mean():.3f} ± {stage_df['ratio_rsu'].std():.3f}")
        print(f"    Local占比:  {stage_df['ratio_local'].mean():.3f} ± {stage_df['ratio_local'].std():.3f}")
        print(f"    V2V占比:    {stage_df['ratio_v2v'].mean():.3f} ± {stage_df['ratio_v2v'].std():.3f}")
        print(f"    Episode奖励: {stage_df['episode_reward'].mean():.3f} ± {stage_df['episode_reward'].std():.3f}")
        print(f"    任务成功率: {stage_df['task_sr'].mean():.3f}")
        print(f"    策略熵:     {stage_df['entropy'].mean():.4f}")

print_stage_stats(df1, "Run 1 (003251)")
print_stage_stats(df2, "Run 2 (145633)")

# ============================================================================
# 3. 可视化对比
# ============================================================================
print("\n" + "=" * 80)
print("3. 生成对比图表")
print("=" * 80)

output_dir = Path('runs/comparison_20260318')
output_dir.mkdir(exist_ok=True)

# 3.1 动作分布对比
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

window = 50

# Run 1
ax = axes[0, 0]
ax.plot(df1.index, df1['ratio_rsu'].rolling(window, min_periods=1).mean(),
        label='RSU', linewidth=2)
ax.plot(df1.index, df1['ratio_local'].rolling(window, min_periods=1).mean(),
        label='Local', linewidth=2)
ax.plot(df1.index, df1['ratio_v2v'].rolling(window, min_periods=1).mean(),
        label='V2V', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Action Ratio')
ax.set_title(f'Run 1 动作分布\n(LR_critic=5e-4, entropy=0.012)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

# Run 2
ax = axes[0, 1]
ax.plot(df2.index, df2['ratio_rsu'].rolling(window, min_periods=1).mean(),
        label='RSU', linewidth=2)
ax.plot(df2.index, df2['ratio_local'].rolling(window, min_periods=1).mean(),
        label='Local', linewidth=2)
ax.plot(df2.index, df2['ratio_v2v'].rolling(window, min_periods=1).mean(),
        label='V2V', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('Action Ratio')
ax.set_title(f'Run 2 动作分布\n(LR_critic=2e-4, entropy=0.02)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

# RSU占比对比
ax = axes[0, 2]
ax.plot(df1.index, df1['ratio_rsu'].rolling(window, min_periods=1).mean(),
        label='Run 1', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['ratio_rsu'].rolling(window, min_periods=1).mean(),
        label='Run 2', linewidth=2, alpha=0.8)
ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%阈值')
ax.set_xlabel('Episode')
ax.set_ylabel('RSU Ratio')
ax.set_title('RSU占比对比')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

# 奖励对比
ax = axes[1, 0]
ax.plot(df1.index, df1['episode_reward'].rolling(window, min_periods=1).mean(),
        label='Run 1', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['episode_reward'].rolling(window, min_periods=1).mean(),
        label='Run 2', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Episode Reward')
ax.set_title('累积奖励对比')
ax.legend()
ax.grid(alpha=0.3)

# 熵对比
ax = axes[1, 1]
ax.plot(df1.index, df1['entropy'].rolling(window, min_periods=1).mean(),
        label='Run 1', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['entropy'].rolling(window, min_periods=1).mean(),
        label='Run 2', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Entropy')
ax.set_title('策略熵对比')
ax.legend()
ax.grid(alpha=0.3)

# 任务成功率对比
ax = axes[1, 2]
ax.plot(df1.index, df1['task_sr'].rolling(window, min_periods=1).mean(),
        label='Run 1', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['task_sr'].rolling(window, min_periods=1).mean(),
        label='Run 2', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Task Success Rate')
ax.set_title('任务成功率对比')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(output_dir / 'comparison_overview.png', dpi=150, bbox_inches='tight')
print(f"已保存: {output_dir / 'comparison_overview.png'}")

# 3.2 训练指标对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Critic Loss
ax = axes[0, 0]
ax.plot(df1.index, df1['critic_loss'].rolling(20, min_periods=1).mean(),
        label='Run 1 (LR=5e-4)', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['critic_loss'].rolling(20, min_periods=1).mean(),
        label='Run 2 (LR=2e-4)', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Critic Loss')
ax.set_title('Critic Loss对比')
ax.legend()
ax.grid(alpha=0.3)

# Actor Loss
ax = axes[0, 1]
ax.plot(df1.index, df1['actor_loss'].rolling(20, min_periods=1).mean(),
        label='Run 1', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['actor_loss'].rolling(20, min_periods=1).mean(),
        label='Run 2', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Actor Loss')
ax.set_title('Actor Loss对比')
ax.legend()
ax.grid(alpha=0.3)

# KL Divergence
ax = axes[1, 0]
ax.plot(df1.index, df1['approx_kl'].rolling(20, min_periods=1).mean(),
        label='Run 1', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['approx_kl'].rolling(20, min_periods=1).mean(),
        label='Run 2', linewidth=2, alpha=0.8)
ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Target KL')
ax.set_xlabel('Episode')
ax.set_ylabel('Approx KL')
ax.set_title('KL散度对比')
ax.legend()
ax.grid(alpha=0.3)

# Clip Fraction
ax = axes[1, 1]
ax.plot(df1.index, df1['clip_frac'].rolling(20, min_periods=1).mean(),
        label='Run 1', linewidth=2, alpha=0.8)
ax.plot(df2.index, df2['clip_frac'].rolling(20, min_periods=1).mean(),
        label='Run 2', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Clip Fraction')
ax.set_title('Clip比例对比')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'comparison_training_metrics.png', dpi=150, bbox_inches='tight')
print(f"已保存: {output_dir / 'comparison_training_metrics.png'}")

# ============================================================================
# 4. 数值对比表
# ============================================================================
print("\n" + "=" * 80)
print("4. 最终性能对比 (最后100集)")
print("=" * 80)

def get_final_stats(df, n=100):
    """获取最后n集的统计"""
    final_df = df.tail(n)
    return {
        'rsu_ratio': final_df['ratio_rsu'].mean(),
        'local_ratio': final_df['ratio_local'].mean(),
        'v2v_ratio': final_df['ratio_v2v'].mean(),
        'episode_reward': final_df['episode_reward'].mean(),
        'task_sr': final_df['task_sr'].mean(),
        'task_duration': final_df['task_duration_mean'].mean(),
        'entropy': final_df['entropy'].mean(),
    }

stats1 = get_final_stats(df1)
stats2 = get_final_stats(df2)

print(f"\n{'指标':<20s} {'Run 1':>15s} {'Run 2':>15s} {'变化':>15s}")
print("-" * 70)
for key in stats1.keys():
    val1 = stats1[key]
    val2 = stats2[key]
    delta = val2 - val1
    delta_pct = (delta / val1 * 100) if val1 != 0 else 0

    if 'ratio' in key:
        print(f"{key:<20s} {val1:>14.2%} {val2:>14.2%} {delta:>+14.2%}")
    elif 'entropy' in key:
        print(f"{key:<20s} {val1:>14.4f} {val2:>14.4f} {delta:>+14.4f}")
    else:
        print(f"{key:<20s} {val1:>14.3f} {val2:>14.3f} {delta:>+14.3f}")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
