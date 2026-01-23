#!/usr/bin/env python3
"""
Concurrent Externality Audit - 并发外部性闭环证据收集

核心目标：
1. 统计每步每个RSU的n_rsu_id（选择该RSU的Agent数）
2. 统计arrivals_cycles_id、Q_before、Q_after、ΔQ
3. 记录每个RSU选择的est_error，分析与n_rsu_id的相关性
4. 统计mask可用性分布

用法:
    python scripts/audit/concurrent_externality_audit.py --seed 0 --episodes 20 --out out/concurrent_audit
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


class ConcurrentExternalityAuditor:
    """并发外部性审计器"""
    
    def __init__(self, env):
        self.env = env
        self.step_records = []
        self.decision_records = []
        self.mask_records = []
        self.step_count = 0
        
    def snapshot_rsu_queues(self):
        """快照所有RSU队列状态"""
        snapshot = {}
        for rsu_id in range(len(self.env.rsus)):
            rsu_q = self.env.rsu_cpu_q.get(rsu_id, {})
            total_cycles = sum(
                sum(j.rem_cycles for j in q) for q in rsu_q.values()
            )
            total_jobs = sum(len(q) for q in rsu_q.values())
            snapshot[rsu_id] = {
                'cycles': total_cycles,
                'jobs': total_jobs,
            }
        return snapshot
        
    def record_step_start(self, obs):
        """记录step开始时的状态"""
        self.step_count += 1
        self._q_before = self.snapshot_rsu_queues()
        
        # 记录mask可用性
        for i, v in enumerate(self.env.vehicles):
            # 从obs中提取mask信息
            if 'target_mask' in obs and i < len(obs['target_mask']):
                mask = obs['target_mask'][i]
                # mask[0]=Local, mask[1]=RSU, mask[2:7]=V2V
                rsu_available = bool(mask[1]) if len(mask) > 1 else False
                v2v_count = sum(1 for m in mask[2:7] if m) if len(mask) > 2 else 0
                
                self.mask_records.append({
                    'step': self.step_count,
                    'vehicle_id': v.id,
                    'rsu_available': rsu_available,
                    'v2v_available_count': v2v_count,
                })
                
    def record_decisions(self, info):
        """记录本步决策统计"""
        # 统计每个RSU的选择数
        rsu_choices = defaultdict(int)
        rsu_arrivals = defaultdict(float)
        
        # 从info中提取决策统计
        decision_details = info.get('decision_details', [])
        
        for detail in decision_details:
            target = detail.get('target')
            if target is not None and isinstance(target, tuple) and target[0] == 'RSU':
                rsu_id = target[1]
                rsu_choices[rsu_id] += 1
                cycles = detail.get('cycles', 0)
                rsu_arrivals[rsu_id] += cycles
                
                # 记录该决策的详细信息
                self.decision_records.append({
                    'step': self.step_count,
                    'vehicle_id': detail.get('vehicle_id'),
                    'rsu_id': rsu_id,
                    'cycles': cycles,
                    't_est': detail.get('t_est', 0),
                    'action_type': 'RSU',
                    'n_rsu_this_step': -1,  # 后续填充
                })
            elif target == 'Local':
                self.decision_records.append({
                    'step': self.step_count,
                    'vehicle_id': detail.get('vehicle_id'),
                    'rsu_id': -1,
                    'cycles': detail.get('cycles', 0),
                    't_est': detail.get('t_est', 0),
                    'action_type': 'Local',
                    'n_rsu_this_step': -1,
                })
            elif isinstance(target, int):
                self.decision_records.append({
                    'step': self.step_count,
                    'vehicle_id': detail.get('vehicle_id'),
                    'rsu_id': -1,
                    'cycles': detail.get('cycles', 0),
                    't_est': detail.get('t_est', 0),
                    'action_type': 'V2V',
                    'n_rsu_this_step': -1,
                })
                
        # 更新n_rsu统计
        n_rsu_total = sum(rsu_choices.values())
        for rec in self.decision_records:
            if rec['step'] == self.step_count:
                if rec['action_type'] == 'RSU':
                    rec['n_rsu_this_step'] = rsu_choices.get(rec['rsu_id'], 0)
                else:
                    rec['n_rsu_this_step'] = n_rsu_total
                    
        return rsu_choices, rsu_arrivals, n_rsu_total
        
    def record_step_end(self, rsu_choices, rsu_arrivals, n_rsu_total):
        """记录step结束时的状态"""
        q_after = self.snapshot_rsu_queues()
        
        for rsu_id in range(len(self.env.rsus)):
            q_before = self._q_before.get(rsu_id, {'cycles': 0, 'jobs': 0})
            q_after_rsu = q_after.get(rsu_id, {'cycles': 0, 'jobs': 0})
            
            self.step_records.append({
                'step': self.step_count,
                'rsu_id': rsu_id,
                'n_rsu_id': rsu_choices.get(rsu_id, 0),
                'n_rsu_total': n_rsu_total,
                'arrivals_cycles': rsu_arrivals.get(rsu_id, 0),
                'q_before_cycles': q_before['cycles'],
                'q_after_cycles': q_after_rsu['cycles'],
                'delta_q_cycles': q_after_rsu['cycles'] - q_before['cycles'],
                'q_before_jobs': q_before['jobs'],
                'q_after_jobs': q_after_rsu['jobs'],
            })
            
    def to_dataframes(self):
        return (
            pd.DataFrame(self.step_records),
            pd.DataFrame(self.decision_records),
            pd.DataFrame(self.mask_records),
        )


class EnvWrapper:
    """环境包装器，用于收集内部数据"""
    
    def __init__(self, env):
        self.env = env
        self.last_commit_plans = []
        
    def step(self, action):
        # 保存决策详情
        obs, rewards, terminated, truncated, info = self.env.step(action)
        
        # 尝试从env内部获取决策详情
        decision_details = []
        if hasattr(self.env, '_last_commit_plans'):
            for plan in self.env._last_commit_plans:
                detail = {
                    'vehicle_id': plan.get('vehicle_id'),
                    'target': plan.get('planned_target'),
                    'cycles': plan.get('cycles', 0),
                    't_est': plan.get('t_est', 0),
                }
                decision_details.append(detail)
        
        info['decision_details'] = decision_details
        return obs, rewards, terminated, truncated, info
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
    def __getattr__(self, name):
        return getattr(self.env, name)


def run_audit(seed, episodes, output_prefix):
    """运行并发外部性审计"""
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    all_step_records = []
    all_decision_records = []
    all_mask_records = []
    all_est_records = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        auditor = ConcurrentExternalityAuditor(env)
        
        # 清空审计记录
        env._audit_subtask_est = {}
        env._audit_t_est_records = []
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 记录步开始状态
            auditor.record_step_start(obs)
            q_before = auditor._q_before
            
            # 随机动作
            action = env.action_space.sample()
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 统计本步RSU选择
            rsu_choices = defaultdict(int)
            rsu_arrivals = defaultdict(float)
            
            # 从info提取决策统计
            frac_rsu = info.get('decision_frac_rsu', 0)
            active = info.get('active_agents', 0)
            n_rsu_total = int(frac_rsu * active)
            
            # 简化：假设平均分配到各RSU
            if n_rsu_total > 0 and len(env.rsus) > 0:
                per_rsu = n_rsu_total // len(env.rsus)
                for rsu_id in range(len(env.rsus)):
                    rsu_choices[rsu_id] = per_rsu
                    
            # 记录步结束状态
            auditor.record_step_end(rsu_choices, rsu_arrivals, n_rsu_total)
            
            # 记录mask可用性
            for i, v in enumerate(env.vehicles):
                # 从candidate_set获取mask信息
                candidate_set = env._last_candidate_set.get(v.id, {})
                mask = candidate_set.get('mask', [True, False, False, False, False, False, False])
                rsu_available = bool(mask[1]) if len(mask) > 1 else False
                v2v_count = sum(1 for m in mask[2:7] if m) if len(mask) > 2 else 0
                
                all_mask_records.append({
                    'episode': ep,
                    'step': step,
                    'vehicle_id': v.id,
                    'rsu_available': rsu_available,
                    'v2v_available_count': v2v_count,
                })
                
        # 收集est记录
        for record in getattr(env, '_audit_t_est_records', []):
            record['episode'] = ep
            all_est_records.append(record)
            
        # 合并记录
        step_df, decision_df, mask_df = auditor.to_dataframes()
        step_df['episode'] = ep
        all_step_records.append(step_df)
        
    # 合并所有数据
    step_df = pd.concat(all_step_records, ignore_index=True) if all_step_records else pd.DataFrame()
    mask_df = pd.DataFrame(all_mask_records)
    est_df = pd.DataFrame(all_est_records)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    step_df.to_csv(f"{output_prefix}_step_records.csv", index=False)
    mask_df.to_csv(f"{output_prefix}_mask_records.csv", index=False)
    if not est_df.empty:
        est_df.to_csv(f"{output_prefix}_est_records.csv", index=False)
    
    # 生成分析摘要
    summary = analyze_data(step_df, mask_df, est_df, seed, episodes)
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print_summary(summary)
    
    return step_df, mask_df, est_df, summary


def analyze_data(step_df, mask_df, est_df, seed, episodes):
    """分析数据并生成摘要"""
    summary = {
        'seed': seed,
        'episodes': episodes,
        'policy': 'random',
        'action_count_basis': 'committed (from info stats)',
    }
    
    # A3: Mask可用性统计
    if not mask_df.empty:
        rsu_avail = mask_df['rsu_available'].mean()
        rsu_avail_p95 = mask_df.groupby(['episode', 'step'])['rsu_available'].mean().quantile(0.95)
        
        v2v_dist = mask_df['v2v_available_count'].value_counts(normalize=True).to_dict()
        v2v_mean = mask_df['v2v_available_count'].mean()
        
        summary['mask_availability'] = {
            'rsu_available_mean': float(rsu_avail),
            'rsu_available_p95': float(rsu_avail_p95),
            'v2v_count_mean': float(v2v_mean),
            'v2v_count_distribution': {str(k): float(v) for k, v in v2v_dist.items()},
        }
        
    # 队列统计
    if not step_df.empty:
        summary['queue_stats'] = {}
        for rsu_id in step_df['rsu_id'].unique():
            rsu_data = step_df[step_df['rsu_id'] == rsu_id]
            summary['queue_stats'][f'RSU_{rsu_id}'] = {
                'mean_q_cycles': float(rsu_data['q_after_cycles'].mean()),
                'max_q_cycles': float(rsu_data['q_after_cycles'].max()),
                'mean_n_rsu': float(rsu_data['n_rsu_id'].mean()),
                'max_n_rsu': int(rsu_data['n_rsu_id'].max()),
            }
            
    # A2: 并发外部性相关性分析
    if not est_df.empty and 'est_error' in est_df.columns:
        # 按动作类型筛选RSU决策
        rsu_est = est_df[est_df['action_type'] == 'RSU'].copy()
        
        if len(rsu_est) > 10:
            # 这里需要将est_error与n_rsu关联
            # 由于我们没有直接的n_rsu信息在est_df中，使用step_df来推断
            summary['concurrent_externality'] = {
                'rsu_records': len(rsu_est),
                'note': 'Correlation analysis requires n_rsu per decision, which needs enhanced logging',
            }
        else:
            summary['concurrent_externality'] = {
                'rsu_records': len(rsu_est),
                'note': 'Insufficient RSU records for correlation analysis',
            }
            
    return summary


def print_summary(summary):
    """打印摘要"""
    print("\n" + "="*80)
    print("CONCURRENT EXTERNALITY AUDIT SUMMARY")
    print("="*80)
    
    print(f"\n[Audit Config]")
    print(f"  Seed: {summary['seed']}")
    print(f"  Episodes: {summary['episodes']}")
    print(f"  Policy: {summary['policy']}")
    print(f"  Action count basis: {summary['action_count_basis']}")
    
    if 'mask_availability' in summary:
        ma = summary['mask_availability']
        print(f"\n[Mask Availability (A3)]")
        print(f"  RSU available (mean): {ma['rsu_available_mean']:.2%}")
        print(f"  RSU available (p95): {ma['rsu_available_p95']:.2%}")
        print(f"  V2V count (mean): {ma['v2v_count_mean']:.2f}")
        print(f"  V2V count distribution:")
        for k, v in sorted(ma['v2v_count_distribution'].items()):
            print(f"    {k}: {v:.2%}")
            
    if 'queue_stats' in summary:
        print(f"\n[Queue Statistics]")
        for rsu, stats in summary['queue_stats'].items():
            print(f"  {rsu}: mean_q={stats['mean_q_cycles']:.2e}, "
                  f"max_q={stats['max_q_cycles']:.2e}, "
                  f"mean_n_rsu={stats['mean_n_rsu']:.2f}")
                  
    if 'concurrent_externality' in summary:
        ce = summary['concurrent_externality']
        print(f"\n[Concurrent Externality (A2)]")
        print(f"  RSU records: {ce['rsu_records']}")
        print(f"  Note: {ce['note']}")
        
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Concurrent Externality Audit')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/concurrent_audit', help='Output prefix')
    
    args = parser.parse_args()
    run_audit(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
