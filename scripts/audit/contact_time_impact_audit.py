#!/usr/bin/env python3
"""
Contact Time Impact Audit - 检查contact time约束的影响

核心目标：
1. 统计被mask掉的V2V候选中，有多少仅因comp_time而被排除
2. 验证当前约束是否过于严格

用法:
    python scripts/audit/contact_time_impact_audit.py --seed 0 --episodes 10 --out out/contact_audit
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


def run_audit(seed, episodes, output_prefix):
    """运行contact time影响审计"""
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    records = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 在step之前，模拟候选筛选过程并记录
            for v in env.vehicles:
                dag = v.task_dag
                selected_subtask = dag.get_top_priority_task()
                if selected_subtask is None:
                    continue
                    
                task_data = dag.total_data[selected_subtask]
                task_comp = dag.total_comp[selected_subtask]
                
                # 检查每个邻居
                for other in env.vehicles:
                    if v.id == other.id:
                        continue
                        
                    dist = np.linalg.norm(v.pos - other.pos)
                    if dist > Cfg.V2V_RANGE:
                        continue
                        
                    # 估计传输时间
                    est_rate = env.channel.compute_one_rate(v, other.pos, 'V2V', env.time)
                    est_rate = max(est_rate, 1e-6)
                    trans_time = task_data / est_rate
                    
                    # 估计队列等待
                    queue_wait = env._get_node_delay(other)
                    
                    # 估计计算时间
                    comp_time = task_comp / max(other.cpu_freq, 1e-9)
                    
                    # 计算contact time
                    rel_vel = other.vel - v.vel
                    pos_diff = other.pos - v.pos
                    pos_diff_norm = np.linalg.norm(pos_diff)
                    
                    if pos_diff_norm < 1e-6:
                        time_to_break = 100.0  # 很大
                    else:
                        rel_vel_proj = np.dot(rel_vel, pos_diff) / pos_diff_norm
                        if rel_vel_proj > 0.1:
                            time_to_break = (Cfg.V2V_RANGE - dist) / rel_vel_proj
                        else:
                            time_to_break = 100.0
                            
                    # 当前约束（包含comp_time）
                    comm_wait = 0.0  # 简化，假设无通信队列
                    t_finish_est = comm_wait + trans_time + queue_wait + comp_time
                    
                    # 建议约束（仅通信）
                    t_comm_only = comm_wait + trans_time
                    
                    # 判断是否被mask
                    current_masked = t_finish_est > time_to_break
                    should_be_masked = t_comm_only > time_to_break
                    
                    # 只记录因comp_time而被多mask的情况
                    if current_masked and not should_be_masked:
                        records.append({
                            'episode': ep,
                            'step': step,
                            'vehicle_id': v.id,
                            'neighbor_id': other.id,
                            'dist': dist,
                            'trans_time': trans_time,
                            'queue_wait': queue_wait,
                            'comp_time': comp_time,
                            't_finish_est': t_finish_est,
                            't_comm_only': t_comm_only,
                            'time_to_break': time_to_break,
                            'excess_time': t_finish_est - time_to_break,
                            'reason': 'comp_time_caused',
                        })
                        
            # 执行step
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
    # 转换为DataFrame
    df = pd.DataFrame(records)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    if not df.empty:
        df.to_csv(f"{output_prefix}_raw.csv", index=False)
        
    # 生成摘要
    summary = {
        'seed': seed,
        'episodes': episodes,
        'total_over_masked': len(df),
        'note': 'Candidates masked only due to comp_time inclusion',
    }
    
    if not df.empty:
        summary['statistics'] = {
            'mean_excess_time': float(df['excess_time'].mean()),
            'mean_comp_time': float(df['comp_time'].mean()),
            'mean_trans_time': float(df['trans_time'].mean()),
            'mean_time_to_break': float(df['time_to_break'].mean()),
        }
        
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n=== Contact Time Impact Audit ===")
    print(f"Episodes: {episodes}")
    print(f"Over-masked candidates (due to comp_time): {len(df)}")
    
    if not df.empty:
        print(f"\nStatistics:")
        print(f"  Mean excess time: {summary['statistics']['mean_excess_time']:.4f}s")
        print(f"  Mean comp_time: {summary['statistics']['mean_comp_time']:.4f}s")
        print(f"  Mean trans_time: {summary['statistics']['mean_trans_time']:.4f}s")
        print(f"  Mean time_to_break: {summary['statistics']['mean_time_to_break']:.4f}s")
        
    return df, summary


def main():
    parser = argparse.ArgumentParser(description='Contact Time Impact Audit')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/contact_audit', help='Output prefix')
    
    args = parser.parse_args()
    run_audit(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
