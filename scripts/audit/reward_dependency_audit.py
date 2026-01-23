#!/usr/bin/env python3
"""
Reward Dependency Audit - 奖励依赖审计

对每个reward term列出其依赖的变量来源（估计/真实/快照），
并输出分项幅度与相关性。

用法:
    python scripts/audit/reward_dependency_audit.py --seed 0 --episodes 20 --out out/reward_audit
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from utils.reward_stats import RewardStats


class RewardCollector:
    """奖励收集器"""
    
    def __init__(self, env):
        self.env = env
        self.records = []
        self.step_count = 0
        
    def collect_from_reward_stats(self, step, info):
        """从RewardStats收集数据"""
        self.step_count = step
        
        # 尝试从env的_reward_stats获取
        if hasattr(self.env, '_reward_stats') and self.env._reward_stats is not None:
            stats = self.env._reward_stats
            metrics = getattr(stats, 'metrics', {})
            
            # 记录聚合指标
            record = {
                'step': step,
                'time': self.env.time,
                'r_lat_mean': metrics.get('r_lat_mean', 0),
                'r_shape_mean': metrics.get('r_shape_mean', 0),
                'r_energy_mean': metrics.get('r_energy_mean', 0),
                'r_power_mean': metrics.get('r_power_mean', 0),
                'r_timeout_mean': metrics.get('r_timeout_mean', 0),
                'r_term_mean': metrics.get('r_term_mean', 0),
                'r_illegal_mean': metrics.get('r_illegal_mean', 0),
                'r_total_mean': metrics.get('r_total_mean', 0),
                # 动作分布
                'decision_local': info.get('decision_frac_local', 0),
                'decision_rsu': info.get('decision_frac_rsu', 0),
                'decision_v2v': info.get('decision_frac_v2v', 0),
            }
            self.records.append(record)
            
    def to_dataframe(self):
        return pd.DataFrame(self.records)


def run_audit(seed, episodes, output_prefix):
    """运行奖励审计"""
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    all_records = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        collector = RewardCollector(env)
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 随机动作
            action = env.action_space.sample()
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 收集奖励统计
            collector.collect_from_reward_stats(step, info)
            
        # 合并记录
        df = collector.to_dataframe()
        df['episode'] = ep
        all_records.append(df)
        
    # 合并所有episode
    final_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    csv_path = f"{output_prefix}_raw.csv"
    if not final_df.empty:
        final_df.to_csv(csv_path, index=False)
        print(f"[Reward Audit] 保存原始数据到 {csv_path}")
    
    # 生成摘要
    summary = {
        'seed': seed,
        'episodes': episodes,
        'total_records': len(final_df),
        'reward_dependency_map': {
            'r_lat': {
                'depends_on': ['t_actual', 't_alt', 'queue_wait', 'rate'],
                'source': '步前估计值（_estimate_t_actual使用冻结的queue和rate快照）',
                'timing': 'snapshot_time (步开始)',
            },
            'r_shape': {
                'depends_on': ['phi_prev', 'phi_next', 'CFT估计'],
                'source': 'phi_prev=步前状态, phi_next=步后状态',
                'timing': 'phi_prev在步前计算, phi_next在步后计算',
            },
            'r_energy': {
                'depends_on': ['power_ratio', 't_tx'],
                'source': 't_tx来自估计（din/rate），power_ratio是动作直接映射',
                'timing': 'snapshot_time',
            },
            'r_power': {
                'depends_on': ['power_ratio'],
                'source': '动作直接映射',
                'timing': '真实',
            },
            'r_timeout': {
                'depends_on': ['elapsed', 'deadline', 'is_failed'],
                'source': 'self.time - dag.start_time（步后真实时间）',
                'timing': '步后真实',
            },
            'r_term': {
                'depends_on': ['is_finished', 'is_failed'],
                'source': 'DAG状态（步后检查）',
                'timing': '步后真实',
            },
            'r_illegal': {
                'depends_on': ['illegal_reason'],
                'source': '动作解析结果',
                'timing': '真实',
            },
        },
    }
    
    # 统计各奖励项幅度
    if not final_df.empty:
        reward_cols = [c for c in final_df.columns if c.startswith('r_') and c.endswith('_mean')]
        summary['magnitude_stats'] = {}
        
        for col in reward_cols:
            if col in final_df.columns:
                data = final_df[col].dropna()
                if len(data) > 0:
                    summary['magnitude_stats'][col] = {
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'p50': float(data.quantile(0.5)),
                        'p95': float(data.quantile(0.95)),
                        'abs_mean': float(data.abs().mean()),
                    }
                    
        # 动作分布统计
        summary['action_distribution'] = {
            'mean_local': float(final_df['decision_local'].mean()) if 'decision_local' in final_df else 0,
            'mean_rsu': float(final_df['decision_rsu'].mean()) if 'decision_rsu' in final_df else 0,
            'mean_v2v': float(final_df['decision_v2v'].mean()) if 'decision_v2v' in final_df else 0,
        }
        
    json_path = f"{output_prefix}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Reward Audit] 保存摘要到 {json_path}")
    
    # 控制台摘要
    print("\n=== Reward Dependency Audit Summary ===")
    print(f"Episodes: {episodes}")
    print(f"Total records: {len(final_df)}")
    
    print("\n--- Reward Term Dependencies ---")
    for term, dep in summary['reward_dependency_map'].items():
        print(f"\n{term}:")
        print(f"  Depends on: {', '.join(dep['depends_on'])}")
        print(f"  Source: {dep['source']}")
        print(f"  Timing: {dep['timing']}")
        
    if 'magnitude_stats' in summary:
        print("\n--- Magnitude Statistics ---")
        for col, stats in summary['magnitude_stats'].items():
            print(f"{col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"abs_mean={stats['abs_mean']:.4f}, p95={stats['p95']:.4f}")
                  
    if 'action_distribution' in summary:
        print("\n--- Action Distribution ---")
        print(f"  Local: {summary['action_distribution']['mean_local']:.2%}")
        print(f"  RSU: {summary['action_distribution']['mean_rsu']:.2%}")
        print(f"  V2V: {summary['action_distribution']['mean_v2v']:.2%}")
        
    return final_df, summary


def main():
    parser = argparse.ArgumentParser(description='Reward Dependency Audit')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/reward_audit', help='Output prefix')
    
    args = parser.parse_args()
    run_audit(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
