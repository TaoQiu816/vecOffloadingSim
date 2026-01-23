#!/usr/bin/env python3
"""
最小训练 Sanity Check

验证目标：
- 不出现"RSU塌缩 + 队列爆炸 + 成功率下降"的旧现象
- 动作分布保持多样性
- RSU队列负载可控

用法:
    python scripts/audit/training_sanity_check.py --seed 0 --episodes 100 --out out/sanity_seed0
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def run_sanity_check(seed, episodes, output_prefix):
    """运行sanity check"""
    from configs.config import SystemConfig as Cfg
    from envs.vec_offloading_env import VecOffloadingEnv
    
    np.random.seed(seed)
    env = VecOffloadingEnv()
    
    # 数据收集
    episode_stats = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step_count = 0
        
        ep_actions = {'local': 0, 'rsu': 0, 'v2v': 0}
        ep_rsu_loads = {0: [], 1: [], 2: []}
        ep_rewards = []
        
        while not done and step_count < 200:
            step_count += 1
            
            # 随机动作
            action = env.action_space.sample()
            
            # 统计动作
            for v in env.vehicles:
                if isinstance(action, (tuple, list)) and v.id < len(action):
                    act = action[v.id]
                    if isinstance(act, dict):
                        target_idx = int(act.get("target", 0))
                    else:
                        target_idx = int(act[0]) if len(act) > 0 else 0
                else:
                    target_idx = 0
                
                if target_idx == 0:
                    ep_actions['local'] += 1
                elif target_idx == 1:
                    ep_actions['rsu'] += 1
                else:
                    ep_actions['v2v'] += 1
            
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rewards.append(float(np.mean(rewards)))
            
            # RSU队列负载
            for rsu_id in range(len(env.rsus)):
                rsu_q = env.rsu_cpu_q.get(rsu_id, {})
                total_cycles = sum(sum(j.rem_cycles for j in q) for q in rsu_q.values())
                ep_rsu_loads[rsu_id].append(total_cycles)
        
        # Episode统计
        total_actions = sum(ep_actions.values())
        finished = sum(1 for v in env.vehicles if v.task_dag.is_finished)
        failed = sum(1 for v in env.vehicles if v.task_dag.is_failed)
        
        episode_stats.append({
            'episode': ep,
            'steps': step_count,
            'local_frac': ep_actions['local'] / total_actions if total_actions > 0 else 0,
            'rsu_frac': ep_actions['rsu'] / total_actions if total_actions > 0 else 0,
            'v2v_frac': ep_actions['v2v'] / total_actions if total_actions > 0 else 0,
            'rsu0_mean_load': np.mean(ep_rsu_loads[0]),
            'rsu0_max_load': np.max(ep_rsu_loads[0]),
            'rsu1_mean_load': np.mean(ep_rsu_loads[1]),
            'rsu1_max_load': np.max(ep_rsu_loads[1]),
            'rsu2_mean_load': np.mean(ep_rsu_loads[2]),
            'reward_mean': np.mean(ep_rewards),
            'reward_std': np.std(ep_rewards),
            'success_count': finished,
            'fail_count': failed,
            'success_rate': finished / len(env.vehicles),
        })
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    df = pd.DataFrame(episode_stats)
    df.to_csv(f"{output_prefix}_episodes.csv", index=False)
    
    # 汇总
    summary = {
        'seed': seed,
        'episodes': episodes,
        'action_distribution': {
            'local_mean': float(df['local_frac'].mean()),
            'rsu_mean': float(df['rsu_frac'].mean()),
            'v2v_mean': float(df['v2v_frac'].mean()),
            'local_std': float(df['local_frac'].std()),
            'rsu_std': float(df['rsu_frac'].std()),
            'v2v_std': float(df['v2v_frac'].std()),
        },
        'rsu_queue_load': {
            'rsu0_mean': float(df['rsu0_mean_load'].mean()),
            'rsu0_p95': float(df['rsu0_max_load'].quantile(0.95)),
            'rsu1_mean': float(df['rsu1_mean_load'].mean()),
            'rsu1_p95': float(df['rsu1_max_load'].quantile(0.95)),
            'rsu2_mean': float(df['rsu2_mean_load'].mean()),
        },
        'task_metrics': {
            'success_rate_mean': float(df['success_rate'].mean()),
            'success_rate_std': float(df['success_rate'].std()),
        },
        'reward': {
            'mean': float(df['reward_mean'].mean()),
            'std': float(df['reward_std'].mean()),
        },
    }
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SANITY CHECK SUMMARY (seed={seed}, episodes={episodes})")
    print(f"{'='*60}")
    print(f"\n[动作分布]")
    print(f"  Local: {summary['action_distribution']['local_mean']:.1%} ± {summary['action_distribution']['local_std']:.1%}")
    print(f"  RSU:   {summary['action_distribution']['rsu_mean']:.1%} ± {summary['action_distribution']['rsu_std']:.1%}")
    print(f"  V2V:   {summary['action_distribution']['v2v_mean']:.1%} ± {summary['action_distribution']['v2v_std']:.1%}")
    print(f"\n[RSU队列负载]")
    print(f"  RSU_0: mean={summary['rsu_queue_load']['rsu0_mean']:.2e}, p95_max={summary['rsu_queue_load']['rsu0_p95']:.2e}")
    print(f"  RSU_1: mean={summary['rsu_queue_load']['rsu1_mean']:.2e}, p95_max={summary['rsu_queue_load']['rsu1_p95']:.2e}")
    print(f"  RSU_2: mean={summary['rsu_queue_load']['rsu2_mean']:.2e}")
    print(f"\n[任务指标]")
    print(f"  Success rate: {summary['task_metrics']['success_rate_mean']:.2%} ± {summary['task_metrics']['success_rate_std']:.2%}")
    print(f"\n[奖励]")
    print(f"  Mean: {summary['reward']['mean']:.4f}")
    print(f"{'='*60}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Training Sanity Check')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--out', type=str, default='out/sanity')
    
    args = parser.parse_args()
    run_sanity_check(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
