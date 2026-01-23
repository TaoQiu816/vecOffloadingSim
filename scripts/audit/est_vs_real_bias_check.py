#!/usr/bin/env python3
"""
Estimation vs Real Bias Check - 估计值与真实值偏差检查

利用现有的t_est/t_real记录，统计误差分布，并检查其与动作类型、同一步n_rsu的相关性。

用法:
    python scripts/audit/est_vs_real_bias_check.py --seed 0 --episodes 20 --out out/est_vs_real
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


def run_bias_check(seed, episodes, output_prefix):
    """运行偏差检查"""
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    all_records = []
    step_rsu_counts = []  # 记录每步RSU选择数
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        # 清空审计记录
        env._audit_subtask_est = {}
        env._audit_t_est_records = []
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 随机动作
            action = env.action_space.sample()
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录本步RSU选择数
            n_rsu = int(info.get('decision_frac_rsu', 0) * info.get('active_agents', 1))
            step_rsu_counts.append({
                'episode': ep,
                'step': step,
                'n_rsu': n_rsu,
                'n_local': int(info.get('decision_frac_local', 0) * info.get('active_agents', 1)),
                'n_v2v': int(info.get('decision_frac_v2v', 0) * info.get('active_agents', 1)),
            })
            
        # 收集该episode的t_est记录
        for record in getattr(env, '_audit_t_est_records', []):
            record['episode'] = ep
            all_records.append(record)
            
    # 转换为DataFrame
    df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    step_df = pd.DataFrame(step_rsu_counts)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    if not df.empty:
        csv_path = f"{output_prefix}_raw.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Est vs Real] 保存原始数据到 {csv_path}")
    else:
        print("[Est vs Real] 警告: 没有收集到t_est/t_real记录")
        print("  这可能是因为任务未在episode内完成")
        
    step_df.to_csv(f"{output_prefix}_step_counts.csv", index=False)
    
    # 生成摘要
    summary = {
        'seed': seed,
        'episodes': episodes,
        'total_records': len(df),
    }
    
    if not df.empty and 'est_error' in df.columns:
        # 总体误差统计
        summary['overall'] = {
            'mean_error': float(df['est_error'].mean()),
            'std_error': float(df['est_error'].std()),
            'p50_error': float(df['est_error'].quantile(0.5)),
            'p95_error': float(df['est_error'].quantile(0.95)),
            'min_error': float(df['est_error'].min()),
            'max_error': float(df['est_error'].max()),
        }
        
        # 按动作类型统计
        if 'action_type' in df.columns:
            summary['by_action_type'] = {}
            for action_type in df['action_type'].unique():
                type_df = df[df['action_type'] == action_type]
                if len(type_df) > 0:
                    summary['by_action_type'][action_type] = {
                        'count': len(type_df),
                        'mean_error': float(type_df['est_error'].mean()),
                        'std_error': float(type_df['est_error'].std()),
                        'p50_error': float(type_df['est_error'].quantile(0.5)),
                        'p95_error': float(type_df['est_error'].quantile(0.95)),
                        'mean_t_est': float(type_df['t_actual_est'].mean()),
                        'mean_t_real': float(type_df['t_actual_real'].mean()),
                    }
                    
    # 步级RSU统计
    if not step_df.empty:
        summary['step_rsu_stats'] = {
            'mean_n_rsu': float(step_df['n_rsu'].mean()),
            'max_n_rsu': int(step_df['n_rsu'].max()),
            'mean_n_local': float(step_df['n_local'].mean()),
            'mean_n_v2v': float(step_df['n_v2v'].mean()),
        }
        
    json_path = f"{output_prefix}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Est vs Real] 保存摘要到 {json_path}")
    
    # 控制台摘要
    print("\n=== Est vs Real Bias Summary ===")
    print(f"Episodes: {episodes}")
    print(f"Total t_est/t_real records: {len(df)}")
    
    if 'overall' in summary:
        print(f"\nOverall error (t_real - t_est):")
        print(f"  Mean: {summary['overall']['mean_error']:.4f}s")
        print(f"  Std: {summary['overall']['std_error']:.4f}s")
        print(f"  P50: {summary['overall']['p50_error']:.4f}s")
        print(f"  P95: {summary['overall']['p95_error']:.4f}s")
        
    if 'by_action_type' in summary:
        print(f"\nBy action type:")
        for action_type, stats in summary['by_action_type'].items():
            print(f"  {action_type}: count={stats['count']}, "
                  f"mean_error={stats['mean_error']:.4f}s, "
                  f"t_est={stats['mean_t_est']:.4f}s, "
                  f"t_real={stats['mean_t_real']:.4f}s")
                  
    if 'step_rsu_stats' in summary:
        print(f"\nStep-level RSU selection:")
        print(f"  Mean n_rsu per step: {summary['step_rsu_stats']['mean_n_rsu']:.2f}")
        print(f"  Max n_rsu: {summary['step_rsu_stats']['max_n_rsu']}")
        
    return df, summary


def main():
    parser = argparse.ArgumentParser(description='Est vs Real Bias Check')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/est_vs_real', help='Output prefix')
    
    args = parser.parse_args()
    run_bias_check(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
