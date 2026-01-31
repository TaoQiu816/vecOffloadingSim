#!/usr/bin/env python3
"""
Baseline方法评估脚本
评估Random、Greedy、Local Only、EFT等baseline在各场景下的性能
用于与MAPPO方法进行公平对比
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg
from baselines.random_policy import RandomPolicy
from baselines.greedy_policy import GreedyPolicy
from baselines.local_only_policy import LocalOnlyPolicy
from baselines.eft_policy import EFTPPolicy

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_env_with_overrides(overrides):
    """根据参数覆盖创建环境"""
    # 保存原始配置
    original_values = {}
    for key, value in overrides.items():
        if hasattr(Cfg, key):
            original_values[key] = getattr(Cfg, key)
            setattr(Cfg, key, value)
    
    # 更新派生参数
    if 'NUM_VEHICLES' in overrides or 'V2V_TOP_K' in overrides:
        Cfg.MAX_NEIGHBORS = max(0, min(Cfg.NUM_VEHICLES - 1, Cfg.V2V_TOP_K))
        Cfg.MAX_TARGETS = (1 + Cfg.NUM_RSU + Cfg.MAX_NEIGHBORS) if Cfg.ENABLE_RSU_SELECTION else (2 + Cfg.MAX_NEIGHBORS)
    
    env = VecOffloadingEnv(Cfg)
    
    # 保存恢复函数到env
    env._original_cfg_values = original_values
    
    return env

def evaluate_policy(env, policy, num_episodes=30, seed=42):
    """评估单个策略"""
    np.random.seed(seed)
    
    results = {
        'success_rates': [],
        'mean_rewards': [],
        'deadline_miss_rates': [],
        'rsu_queue_loads': [],
        'action_distributions': {'local': [], 'rsu': [], 'v2v': []},
    }
    
    for ep in range(num_episodes):
        obs_list, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0
        
        while not done:
            # 获取动作（使用统一接口select_action）
            actions = policy.select_action(obs_list)
            
            # 执行动作
            next_obs_list, rewards, terminated, truncated, info = env.step(actions)
            
            episode_reward += sum(rewards) / len(rewards) if rewards else 0
            done = terminated or truncated
            obs_list = next_obs_list
        
        # 收集episode统计（数据直接在info根层）
        if 'subtask_success_rate' in info:
            results['success_rates'].append(info.get('subtask_success_rate', 0) * 100)  # 转换为百分比
            results['deadline_miss_rates'].append(info.get('deadline_miss_rate', 0) * 100)
            results['rsu_queue_loads'].append(info.get('episode_metrics', {}).get('rsu_queue_mean', 0) if 'episode_metrics' in info else 0)
            
            # 动作分布（从decision_frac_*字段获取）
            results['action_distributions']['local'].append(info.get('decision_frac_local', 0) * 100)
            results['action_distributions']['rsu'].append(info.get('decision_frac_rsu', 0) * 100)
            results['action_distributions']['v2v'].append(info.get('decision_frac_v2v', 0) * 100)
        
        results['mean_rewards'].append(episode_reward)
    
    # 计算统计量
    summary = {
        'success_rate_mean': np.mean(results['success_rates']),
        'success_rate_std': np.std(results['success_rates']),
        'reward_mean': np.mean(results['mean_rewards']),
        'reward_std': np.std(results['mean_rewards']),
        'deadline_miss_mean': np.mean(results['deadline_miss_rates']),
        'rsu_queue_mean': np.mean(results['rsu_queue_loads']),
        'action_local': np.mean(results['action_distributions']['local']),
        'action_rsu': np.mean(results['action_distributions']['rsu']),
        'action_v2v': np.mean(results['action_distributions']['v2v']),
    }
    
    return summary, results

def run_baseline_evaluation(scene_config, output_dir, num_episodes=30, seed=42):
    """在指定场景下评估所有baseline"""
    
    # 创建环境
    env = create_env_with_overrides(scene_config)
    
    # 定义baseline策略（需要env参数的策略在创建env后初始化）
    policies = {
        'Random': RandomPolicy(seed=seed),
        'Greedy': GreedyPolicy(env),  # GreedyPolicy需要env参数
        'Local Only': LocalOnlyPolicy(),
        'EFT': EFTPPolicy(env),  # EFTPPolicy需要env参数
    }
    
    results = {}
    
    for name, policy in policies.items():
        print(f"  Evaluating {name}...")
        try:
            summary, raw = evaluate_policy(env, policy, num_episodes, seed)
            results[name] = summary
            print(f"    Success Rate: {summary['success_rate_mean']:.1f}%")
        except Exception as e:
            print(f"    Error: {e}")
            results[name] = {'error': str(e)}
    
    # 恢复原始配置
    if hasattr(env, '_original_cfg_values'):
        for key, value in env._original_cfg_values.items():
            setattr(Cfg, key, value)
    
    env.close()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='eval_results/baseline_comparison')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 定义场景（参数经过验证确保能拉开差距）
    scenes = {
        'Easy (Baseline)': {},  # 默认配置
        'Strict-Challenge': {
            # 严格场景：Random ~70%, 目标MAPPO 85%+
            'MIN_NODES': 25,
            'MAX_NODES': 35,
            'DEADLINE_TIGHTENING_MIN': 0.5,  # 关键：收紧Deadline
            'DEADLINE_TIGHTENING_MAX': 0.65,
            'NUM_VEHICLES': 50,
            'MAX_COMP': 4.0e9,
            'RSU_QUEUE_CYCLES_LIMIT': 100.0e9,
        },
        'Large-DAG': {
            # 大规模DAG场景：体现Transformer优势
            'MIN_NODES': 35,
            'MAX_NODES': 50,
            'DEADLINE_TIGHTENING_MIN': 0.55,
            'DEADLINE_TIGHTENING_MAX': 0.7,
            'NUM_VEHICLES': 30,
            'MAX_COMP': 3.0e9,
        },
        'Resource-Contention': {
            # 资源竞争场景：体现Physics Bias优势
            'MIN_NODES': 20,
            'MAX_NODES': 28,
            'DEADLINE_TIGHTENING_MIN': 0.5,
            'DEADLINE_TIGHTENING_MAX': 0.65,
            'NUM_VEHICLES': 60,  # 更多车辆竞争
            'RSU_QUEUE_CYCLES_LIMIT': 70.0e9,  # 更小队列
            'V2V_TOP_K': 7,
        },
    }
    
    all_results = {}
    
    for scene_name, scene_config in scenes.items():
        print(f"\n=== Evaluating Scene: {scene_name} ===")
        results = run_baseline_evaluation(scene_config, args.output, args.episodes, args.seed)
        all_results[scene_name] = results
    
    # 保存结果
    output_file = os.path.join(args.output, 'baseline_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # 生成对比表格
    rows = []
    for scene_name, scene_results in all_results.items():
        for policy_name, metrics in scene_results.items():
            if 'error' not in metrics:
                rows.append({
                    'Scene': scene_name,
                    'Policy': policy_name,
                    'Success Rate (%)': f"{metrics['success_rate_mean']:.1f}±{metrics['success_rate_std']:.1f}",
                    'Mean Reward': f"{metrics['reward_mean']:.3f}",
                    'Deadline Miss (%)': f"{metrics['deadline_miss_mean']:.1f}",
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output, 'baseline_comparison.csv'), index=False)
    
    # 绘制对比图
    fig, axes = plt.subplots(1, len(scenes), figsize=(5 * len(scenes), 5))
    if len(scenes) == 1:
        axes = [axes]
    
    for ax, (scene_name, scene_results) in zip(axes, all_results.items()):
        policies = []
        success_rates = []
        errors = []
        
        for policy_name, metrics in scene_results.items():
            if 'error' not in metrics:
                policies.append(policy_name)
                success_rates.append(metrics['success_rate_mean'])
                errors.append(metrics['success_rate_std'])
        
        x = np.arange(len(policies))
        bars = ax.bar(x, success_rates, yerr=errors, capsize=5, color='steelblue', alpha=0.8)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(scene_name)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'baseline_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Comparison plot saved to: {os.path.join(args.output, 'baseline_comparison.png')}")

if __name__ == '__main__':
    main()
