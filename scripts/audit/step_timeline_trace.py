#!/usr/bin/env python3
"""
Step Timeline Trace - 逐步关键事件追踪

对单个episode的少量车辆，输出逐step的关键事件追踪：
- DAG状态变化: ready→running→done
- 队列变化: enqueue/dequeue事件
- 传输/计算完成事件

用法:
    python scripts/audit/step_timeline_trace.py --seed 0 --episodes 1 --vehicles 3 --out out/timeline_trace
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


class TimelineTracer:
    """时间线追踪器"""
    
    def __init__(self, env, vehicle_ids):
        self.env = env
        self.vehicle_ids = set(vehicle_ids)
        self.events = []
        self.step_count = 0
        
    def record_step_start(self):
        """记录step开始时的状态"""
        self.step_count += 1
        for v in self.env.vehicles:
            if v.id not in self.vehicle_ids:
                continue
            dag = v.task_dag
            # 记录DAG状态
            for i in range(dag.num_subtasks):
                status_names = {0: 'PENDING', 1: 'READY', 2: 'RUNNING', 3: 'COMPLETED'}
                self.events.append({
                    'step': self.step_count,
                    'time': self.env.time,
                    'phase': 'START',
                    'vehicle_id': v.id,
                    'event_type': 'DAG_STATUS',
                    'subtask_id': i,
                    'status': status_names.get(dag.status[i], 'UNKNOWN'),
                    'exec_loc': str(dag.exec_locations[i]),
                    'rem_comp': float(dag.rem_comp[i]),
                    'rem_data': float(dag.rem_data[i]),
                })
                
    def record_decision(self, plans):
        """记录决策"""
        for plan in plans:
            if plan['vehicle_id'] not in self.vehicle_ids:
                continue
            self.events.append({
                'step': self.step_count,
                'time': self.env.time,
                'phase': 'DECISION',
                'vehicle_id': plan['vehicle_id'],
                'event_type': 'ACTION',
                'subtask_id': plan.get('subtask_idx'),
                'target': str(plan.get('planned_target')),
                'kind': plan.get('planned_kind'),
                'illegal': plan.get('illegal_reason'),
            })
            
    def record_queue_state(self, phase_name):
        """记录队列状态"""
        # RSU队列
        for rsu_id in range(len(self.env.rsus)):
            rsu_q = self.env.rsu_cpu_q.get(rsu_id, {})
            total_jobs = sum(len(q) for q in rsu_q.values())
            total_cycles = sum(
                sum(j.rem_cycles for j in q) for q in rsu_q.values()
            )
            self.events.append({
                'step': self.step_count,
                'time': self.env.time,
                'phase': phase_name,
                'vehicle_id': -1,
                'event_type': 'RSU_QUEUE',
                'rsu_id': rsu_id,
                'queue_jobs': total_jobs,
                'queue_cycles': float(total_cycles),
            })
            
        # 追踪车辆的队列
        for vid in self.vehicle_ids:
            veh_q = self.env.veh_cpu_q.get(vid, [])
            total_jobs = len(veh_q)
            total_cycles = sum(j.rem_cycles for j in veh_q)
            self.events.append({
                'step': self.step_count,
                'time': self.env.time,
                'phase': phase_name,
                'vehicle_id': vid,
                'event_type': 'VEH_QUEUE',
                'queue_jobs': total_jobs,
                'queue_cycles': float(total_cycles),
            })
            
    def record_step_end(self, rewards, info):
        """记录step结束时的状态"""
        for v in self.env.vehicles:
            if v.id not in self.vehicle_ids:
                continue
            dag = v.task_dag
            self.events.append({
                'step': self.step_count,
                'time': self.env.time,
                'phase': 'END',
                'vehicle_id': v.id,
                'event_type': 'DAG_SUMMARY',
                'is_finished': dag.is_finished,
                'is_failed': dag.is_failed,
                'reward': float(rewards[v.id]) if v.id < len(rewards) else 0.0,
            })
            
    def to_dataframe(self):
        return pd.DataFrame(self.events)


def run_trace(seed, episodes, vehicle_count, output_prefix):
    """运行追踪"""
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    all_events = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        # 选择要追踪的车辆
        vehicle_ids = list(range(min(vehicle_count, len(env.vehicles))))
        tracer = TimelineTracer(env, vehicle_ids)
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 记录step开始状态
            tracer.record_step_start()
            tracer.record_queue_state('BEFORE_PHASE')
            
            # 随机动作
            action = env.action_space.sample()
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录step结束状态
            tracer.record_queue_state('AFTER_PHASE')
            tracer.record_step_end(rewards, info)
            
        # 合并事件
        df = tracer.to_dataframe()
        df['episode'] = ep
        all_events.append(df)
        
    # 合并所有episode
    final_df = pd.concat(all_events, ignore_index=True)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    csv_path = f"{output_prefix}_raw.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"[Timeline Trace] 保存原始数据到 {csv_path}")
    
    # 生成摘要
    summary = {
        'seed': seed,
        'episodes': episodes,
        'vehicle_count': vehicle_count,
        'total_events': len(final_df),
        'event_types': final_df['event_type'].value_counts().to_dict(),
    }
    
    # 统计状态转换
    dag_events = final_df[final_df['event_type'] == 'DAG_STATUS']
    if not dag_events.empty:
        summary['status_distribution'] = dag_events['status'].value_counts().to_dict()
        
    # 统计队列
    rsu_events = final_df[final_df['event_type'] == 'RSU_QUEUE']
    if not rsu_events.empty:
        summary['rsu_queue_stats'] = {
            'mean_jobs': float(rsu_events['queue_jobs'].mean()),
            'max_jobs': int(rsu_events['queue_jobs'].max()),
            'mean_cycles': float(rsu_events['queue_cycles'].mean()),
        }
        
    json_path = f"{output_prefix}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[Timeline Trace] 保存摘要到 {json_path}")
    
    # 控制台摘要
    print("\n=== Timeline Trace Summary ===")
    print(f"Episodes: {episodes}, Vehicles traced: {vehicle_count}")
    print(f"Total events: {len(final_df)}")
    if 'status_distribution' in summary:
        print(f"DAG status distribution: {summary['status_distribution']}")
    if 'rsu_queue_stats' in summary:
        print(f"RSU queue: mean_jobs={summary['rsu_queue_stats']['mean_jobs']:.2f}, "
              f"max_jobs={summary['rsu_queue_stats']['max_jobs']}")
              
    return final_df, summary


def main():
    parser = argparse.ArgumentParser(description='Step Timeline Trace')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--vehicles', type=int, default=3, help='Number of vehicles to trace')
    parser.add_argument('--out', type=str, default='out/timeline_trace', help='Output prefix')
    
    args = parser.parse_args()
    run_trace(args.seed, args.episodes, args.vehicles, args.out)


if __name__ == '__main__':
    main()
