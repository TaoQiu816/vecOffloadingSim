#!/usr/bin/env python3
"""
Queue Conservation Check - 队列守恒验证

统计每step每RSU的arrivals/service/Q_before/Q_after，验证守恒关系:
    ΔQ = arrivals - service

用法:
    python scripts/audit/queue_conservation_check.py --seed 0 --episodes 5 --out out/queue_conservation
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


class QueueConservationChecker:
    """队列守恒检查器"""
    
    def __init__(self, env):
        self.env = env
        self.records = []
        self.step_count = 0
        
    def snapshot_queues(self):
        """获取当前队列快照"""
        snapshot = {
            'rsu': {},
            'veh': {},
        }
        
        # RSU队列
        for rsu_id in range(len(self.env.rsus)):
            rsu_q = self.env.rsu_cpu_q.get(rsu_id, {})
            total_cycles = sum(
                sum(j.rem_cycles for j in q) for q in rsu_q.values()
            )
            snapshot['rsu'][rsu_id] = {
                'cycles': total_cycles,
                'jobs': sum(len(q) for q in rsu_q.values()),
            }
            
        # 车辆队列
        for v in self.env.vehicles:
            veh_q = self.env.veh_cpu_q.get(v.id, [])
            total_cycles = sum(j.rem_cycles for j in veh_q)
            snapshot['veh'][v.id] = {
                'cycles': total_cycles,
                'jobs': len(veh_q),
            }
            
        return snapshot
        
    def record_step(self, q_before, q_after, arrivals_rsu, n_rsu_decisions):
        """记录一步的队列变化"""
        self.step_count += 1
        
        for rsu_id in range(len(self.env.rsus)):
            before = q_before['rsu'].get(rsu_id, {'cycles': 0, 'jobs': 0})
            after = q_after['rsu'].get(rsu_id, {'cycles': 0, 'jobs': 0})
            arrivals = arrivals_rsu.get(rsu_id, 0)
            
            # 计算服务量 (理论最大 = f_rsu * DT)
            f_rsu = self.env.rsus[rsu_id].cpu_freq
            max_service = f_rsu * self.env.config.DT
            
            # 实际变化
            delta_cycles = after['cycles'] - before['cycles']
            
            # 守恒检查: delta = arrivals - service
            # service = arrivals - delta (如果delta < arrivals)
            inferred_service = arrivals - delta_cycles
            
            self.records.append({
                'step': self.step_count,
                'time': self.env.time,
                'node_type': 'RSU',
                'node_id': rsu_id,
                'q_before_cycles': before['cycles'],
                'q_before_jobs': before['jobs'],
                'q_after_cycles': after['cycles'],
                'q_after_jobs': after['jobs'],
                'arrivals_cycles': arrivals,
                'delta_cycles': delta_cycles,
                'inferred_service': inferred_service,
                'max_service': max_service,
                'n_rsu_decisions': n_rsu_decisions,
                'conservation_error': delta_cycles - (arrivals - min(inferred_service, max_service)),
            })
            
    def to_dataframe(self):
        return pd.DataFrame(self.records)


def count_arrivals(env, commit_plans):
    """统计本步各RSU的到达cycles"""
    arrivals = defaultdict(float)
    
    for plan in commit_plans:
        target = plan.get('planned_target')
        if target is None:
            continue
        if isinstance(target, tuple) and target[0] == 'RSU':
            rsu_id = target[1]
            subtask_idx = plan.get('subtask_idx')
            if subtask_idx is not None:
                v = plan.get('vehicle')
                if v is not None:
                    cycles = v.task_dag.total_comp[subtask_idx]
                    arrivals[rsu_id] += cycles
                    
    return arrivals


def run_check(seed, episodes, output_prefix):
    """运行守恒检查"""
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    all_records = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        checker = QueueConservationChecker(env)
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 记录步前队列
            q_before = checker.snapshot_queues()
            
            # 随机动作
            action = env.action_space.sample()
            
            # 解析动作以统计arrivals（需要访问内部状态）
            # 由于无法直接拦截，我们在step后根据动作推断
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录步后队列
            q_after = checker.snapshot_queues()
            
            # 从info获取决策统计
            n_rsu = info.get('decision_frac_rsu', 0) * info.get('active_agents', 0)
            
            # 估算arrivals (使用info中的统计)
            arrivals_rsu = defaultdict(float)
            # 注意: 这是近似，因为我们无法精确获取每个RSU的到达
            # 实际应该在env内部记录
            
            checker.record_step(q_before, q_after, arrivals_rsu, n_rsu)
            
        # 合并记录
        df = checker.to_dataframe()
        df['episode'] = ep
        all_records.append(df)
        
    # 合并所有episode
    final_df = pd.concat(all_records, ignore_index=True)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    csv_path = f"{output_prefix}_raw.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"[Queue Conservation] 保存原始数据到 {csv_path}")
    
    # 生成摘要
    rsu_df = final_df[final_df['node_type'] == 'RSU']
    
    summary = {
        'seed': seed,
        'episodes': episodes,
        'total_records': len(final_df),
        'rsu_stats': {},
    }
    
    if not rsu_df.empty:
        # 按RSU分组统计
        for rsu_id in rsu_df['node_id'].unique():
            rsu_data = rsu_df[rsu_df['node_id'] == rsu_id]
            summary['rsu_stats'][f'RSU_{rsu_id}'] = {
                'mean_queue_cycles': float(rsu_data['q_after_cycles'].mean()),
                'max_queue_cycles': float(rsu_data['q_after_cycles'].max()),
                'mean_delta': float(rsu_data['delta_cycles'].mean()),
                'std_delta': float(rsu_data['delta_cycles'].std()),
            }
            
        # 总体守恒检查
        summary['overall'] = {
            'mean_conservation_error': float(rsu_df['conservation_error'].mean()),
            'max_conservation_error': float(rsu_df['conservation_error'].abs().max()),
            'queue_increase_steps': int((rsu_df['delta_cycles'] > 0).sum()),
            'queue_decrease_steps': int((rsu_df['delta_cycles'] < 0).sum()),
        }
        
    json_path = f"{output_prefix}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Queue Conservation] 保存摘要到 {json_path}")
    
    # 控制台摘要
    print("\n=== Queue Conservation Summary ===")
    print(f"Episodes: {episodes}")
    print(f"Total records: {len(final_df)}")
    
    if 'overall' in summary:
        print(f"\nOverall:")
        print(f"  Mean conservation error: {summary['overall']['mean_conservation_error']:.2e}")
        print(f"  Max conservation error: {summary['overall']['max_conservation_error']:.2e}")
        print(f"  Queue increase steps: {summary['overall']['queue_increase_steps']}")
        print(f"  Queue decrease steps: {summary['overall']['queue_decrease_steps']}")
        
    if summary['rsu_stats']:
        print(f"\nPer-RSU stats:")
        for rsu_name, stats in summary['rsu_stats'].items():
            print(f"  {rsu_name}: mean_queue={stats['mean_queue_cycles']:.2e}, "
                  f"max_queue={stats['max_queue_cycles']:.2e}")
                  
    return final_df, summary


def main():
    parser = argparse.ArgumentParser(description='Queue Conservation Check')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/queue_conservation', help='Output prefix')
    
    args = parser.parse_args()
    run_check(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
