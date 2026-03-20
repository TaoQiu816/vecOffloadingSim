#!/usr/bin/env python
"""
运行baseline策略评估并生成对比分析
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入环境和baseline策略
from envs.vec_offloading_env import VecOffloadingEnv
from baselines.local_only_policy import LocalOnlyPolicy
from baselines.random_policy import RandomPolicy
from baselines.greedy_policy import GreedyPolicy
from baselines.eft_policy import EFTPolicy

def load_config(run_dir):
    """加载训练配置"""
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    project_root = run_dir.parent.parent
    sys.path.insert(0, str(project_root))
    
    from configs.config import SystemConfig
    
    # 加载JSON配置
    with open(run_dir / 'config.json', 'r') as f:
        config_dict = json.load(f)
    
    # 创建SystemConfig对象
    config = SystemConfig()
    
    # 将字典中的值设置到SystemConfig对象
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    return config

def evaluate_policy(policy, env, num_episodes=100, desc="Evaluating"):
    """评估单个策略"""
    results = []
    
    for ep in tqdm(range(num_episodes), desc=desc):
        obs, info = env.reset()
        done = False
        truncated = False
        
        ep_reward = 0
        ep_steps = 0
        
        while not (done or truncated):
            # 获取动作
            if hasattr(policy, 'get_action'):
                action = policy.get_action(obs, info)
            else:
                # 对于简单策略，使用环境信息
                action = policy.select_action(env)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
        
        # 收集episode结果
        results.append({
            'episode': ep,
            'reward': ep_reward,
            'steps': ep_steps,
            'task_success_rate': info.get('task_success_rate', 0),
            'subtask_success_rate': info.get('subtask_success_rate', 0),
            'mean_cft': info.get('mean_cft_est', 0),
            'energy_norm_mean': info.get('energy_norm_mean', 0),
            'terminated': done,
            'truncated': truncated,
        })
    
    return pd.DataFrame(results)

def main():
    run_dir = Path(__file__).parent
    print(f"运行目录: {run_dir}")
    
    # 加载配置
    config = load_config(run_dir)
    print(f"✓ 加载配置")
    
    # 创建环境
    print("\n创建评估环境...")
    env = VecOffloadingEnv(config)
    print(f"✓ 环境创建成功")
    
    # 定义baseline策略
    baselines = {
        'Local-Only': LocalOnlyPolicy(),
        'Random': RandomPolicy(),
        'Greedy': GreedyPolicy(env),
        'EFT': EFTPolicy(env),
    }
    
    # 评估每个baseline
    all_results = {}
    num_eval_episodes = 100
    
    print(f"\n开始评估 {len(baselines)} 个baseline策略 (每个 {num_eval_episodes} episodes)...")
    print("=" * 80)
    
    for name, policy in baselines.items():
        print(f"\n评估 {name} 策略...")
        try:
            results_df = evaluate_policy(
                policy, env, 
                num_episodes=num_eval_episodes,
                desc=f"{name}"
            )
            all_results[name] = results_df
            
            # 打印统计
            print(f"\n{name} 结果:")
            print(f"  平均奖励: {results_df['reward'].mean():.4f} ± {results_df['reward'].std():.4f}")
            print(f"  任务成功率: {results_df['task_success_rate'].mean():.4f}")
            print(f"  子任务成功率: {results_df['subtask_success_rate'].mean():.4f}")
            
        except Exception as e:
            print(f"✗ {name} 评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    print("\n" + "=" * 80)
    print("保存结果...")
    
    output_dir = run_dir / 'baseline_results'
    output_dir.mkdir(exist_ok=True)
    
    # 保存每个baseline的详细结果
    for name, df in all_results.items():
        filename = output_dir / f"{name.lower().replace('-', '_')}_results.csv"
        df.to_csv(filename, index=False)
        print(f"✓ 保存 {name}: {filename}")
    
    # 创建汇总表
    summary_data = []
    for name, df in all_results.items():
        summary_data.append({
            'Policy': name,
            'Reward_Mean': df['reward'].mean(),
            'Reward_Std': df['reward'].std(),
            'Task_Success_Rate': df['task_success_rate'].mean(),
            'Subtask_Success_Rate': df['subtask_success_rate'].mean(),
            'Mean_CFT': df['mean_cft'].mean(),
            'Energy_Norm': df['energy_norm_mean'].mean(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / 'baseline_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ 保存汇总: {summary_file}")
    
    print("\n" + "=" * 80)
    print("Baseline评估完成!")
    print(f"结果保存在: {output_dir}")
    
    return all_results, summary_df

def plot_comparison(mappo_data, baseline_results, output_dir):
    """绘制MAPPO vs Baselines对比图"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # 准备数据
    policies = ['MAPPO'] + list(baseline_results.keys())
    
    # 从MAPPO数据中提取最后100 episodes的平均值
    mappo_reward = mappo_data['r_total'].tail(100).mean()
    mappo_task_success = mappo_data['task_success_rate'].tail(100).mean()
    mappo_subtask_success = mappo_data['subtask_success_rate'].tail(100).mean()
    mappo_energy = mappo_data['energy_norm_mean'].tail(100).mean()
    
    rewards = [mappo_reward]
    task_success_rates = [mappo_task_success]
    subtask_success_rates = [mappo_subtask_success]
    energy_norms = [mappo_energy]
    
    for name, df in baseline_results.items():
        rewards.append(df['reward'].mean())
        task_success_rates.append(df['task_success_rate'].mean())
        subtask_success_rates.append(df['subtask_success_rate'].mean())
        energy_norms.append(df['energy_norm_mean'].mean())
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAPPO vs Baselines 性能对比', fontsize=16, fontweight='bold')
    
    # 1. 奖励对比
    colors = ['#2ecc71'] + ['#95a5a6'] * (len(policies) - 1)
    axes[0, 0].bar(policies, rewards, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].set_title('平均奖励对比')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 任务成功率对比
    axes[0, 1].bar(policies, task_success_rates, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('任务成功率')
    axes[0, 1].set_title('任务成功率对比')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 子任务成功率对比
    axes[1, 0].bar(policies, subtask_success_rates, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('子任务成功率')
    axes[1, 0].set_title('子任务成功率对比')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 归一化能耗对比
    axes[1, 1].bar(policies, energy_norms, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('归一化能耗')
    axes[1, 1].set_title('归一化能耗对比 (越低越好)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    comparison_file = output_dir / 'mappo_vs_baselines_comparison.png'
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存对比图: {comparison_file}")
    plt.close()
    
    # 创建详细对比表
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i, policy in enumerate(policies):
        table_data.append([
            policy,
            f'{rewards[i]:.4f}',
            f'{task_success_rates[i]:.2%}',
            f'{subtask_success_rates[i]:.2%}',
            f'{energy_norms[i]:.4f}'
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['策略', '平均奖励', '任务成功率', '子任务成功率', '归一化能耗'],
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 高亮MAPPO行
    for i in range(5):
        table[(1, i)].set_facecolor('#d5f4e6')
    
    plt.title('MAPPO vs Baselines 详细对比表', fontsize=14, fontweight='bold', pad=20)
    table_file = output_dir / 'mappo_vs_baselines_table.png'
    plt.savefig(table_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存对比表: {table_file}")
    plt.close()
    
    # 生成性能提升百分比
    print("\n" + "=" * 80)
    print("MAPPO相对于Baselines的性能提升")
    print("=" * 80)
    for i, name in enumerate(list(baseline_results.keys())):
        reward_improve = ((mappo_reward - rewards[i+1]) / abs(rewards[i+1])) * 100 if rewards[i+1] != 0 else 0
        task_improve = ((mappo_task_success - task_success_rates[i+1]) / task_success_rates[i+1]) * 100 if task_success_rates[i+1] != 0 else 0
        print(f"\nvs {name}:")
        print(f"  奖励提升: {reward_improve:+.1f}%")
        print(f"  任务成功率提升: {task_improve:+.1f}%")

if __name__ == '__main__':
    results, summary = main()
    
    # 加载MAPPO训练数据
    print("\n" + "=" * 80)
    print("加载MAPPO训练数据进行对比...")
    print("=" * 80)
    
    run_dir = Path(__file__).parent
    mappo_data = pd.read_csv(run_dir / 'logs' / 'metrics.csv')
    
    # 绘制对比图
    output_dir = run_dir / 'baseline_results'
    plot_comparison(mappo_data, results, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ Baseline评估和对比分析完成!")
    print("=" * 80)
