#!/usr/bin/env python3
"""
Enhanced Concurrent Externality Audit - 增强版并发外部性审计

核心目标：
1. 记录每个RSU决策的n_rsu_id（同一步选择同一RSU的Agent数）
2. 记录arrivals_cycles/service_cycles/ΔQ（cycles）
3. 计算corr(est_error, n_rsu_id)
4. 输出分桶曲线（n_rsu=1..k时error均值/p95）

用法:
    python scripts/audit/concurrent_externality_enhanced.py --seed 0 --episodes 20 --out out/concurrent_enhanced
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


def snapshot_rsu_queues(env):
    """快照所有RSU队列状态"""
    snapshot = {}
    for rsu_id in range(len(env.rsus)):
        rsu_q = env.rsu_cpu_q.get(rsu_id, {})
        total_cycles = sum(
            sum(j.rem_cycles for j in q) for q in rsu_q.values()
        )
        snapshot[rsu_id] = total_cycles
    return snapshot


def parse_action_target(action, vid):
    """解析动作的target_idx
    
    action格式：tuple of dicts, 每个dict包含 'target' 和 'power'
    例如：({'power': 0.5, 'target': 1}, {'power': 0.8, 'target': 0}, ...)
    """
    if isinstance(action, (tuple, list)):
        if vid < len(action):
            act = action[vid]
            if isinstance(act, dict):
                return int(act.get("target", 0))
            # 数组格式 [target, power]
            act_array = np.asarray(act).flatten()
            return int(act_array[0]) if len(act_array) > 0 else 0
    elif isinstance(action, dict):
        act = action.get(vid, {})
        if isinstance(act, dict):
            return int(act.get("target", 0))
    return 0


def run_audit(seed, episodes, output_prefix):
    """运行增强版并发外部性审计"""
    np.random.seed(seed)
    
    env = VecOffloadingEnv()
    
    step_records = []
    decision_records = []
    est_records_all = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        # 清空审计记录
        env._audit_subtask_est = {}
        env._audit_t_est_records = []
        
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            step_count += 1
            
            # 记录步前队列
            q_before = snapshot_rsu_queues(env)
            
            # 记录每个车辆的serving_rsu_id（在step前获取）
            serving_rsu_by_vid = {}
            for v in env.vehicles:
                # serving_rsu_id在_get_obs中被更新
                rsu_id = getattr(v, 'serving_rsu_id', None)
                if rsu_id is None:
                    # 手动计算最近RSU
                    rsu_id = env._update_serving_rsu(v)
                serving_rsu_by_vid[v.id] = rsu_id
            
            # 随机动作
            action = env.action_space.sample()
            
            # 解析动作，统计RSU选择
            rsu_choices = defaultdict(list)  # rsu_id -> [vehicle_ids]
            for v in env.vehicles:
                target_idx = parse_action_target(action, v.id)
                if target_idx == 1:  # RSU选择
                    rsu_id = serving_rsu_by_vid.get(v.id)
                    if rsu_id is not None:
                        rsu_choices[rsu_id].append(v.id)
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录步后队列
            q_after = snapshot_rsu_queues(env)
            
            # 计算n_rsu_id
            n_rsu_by_id = {rsu_id: len(vids) for rsu_id, vids in rsu_choices.items()}
            n_rsu_total = sum(n_rsu_by_id.values())
            
            # 从队列变化推算arrivals（arrivals = delta_q + service_actual）
            # service_actual = min(q_before, service_capacity)
            for rsu_id in range(len(env.rsus)):
                service_capacity = env.rsus[rsu_id].cpu_freq * env.config.DT
                service_actual = min(q_before.get(rsu_id, 0), service_capacity)
                delta_q = q_after.get(rsu_id, 0) - q_before.get(rsu_id, 0)
                arrivals_inferred = delta_q + service_actual
                
                step_records.append({
                    'episode': ep,
                    'step': step_count,
                    'rsu_id': rsu_id,
                    'n_rsu_id': n_rsu_by_id.get(rsu_id, 0),
                    'n_rsu_total': n_rsu_total,
                    'arrivals_cycles': max(0, arrivals_inferred),
                    'q_before': q_before.get(rsu_id, 0),
                    'q_after': q_after.get(rsu_id, 0),
                    'delta_q': delta_q,
                    'service_capacity': service_capacity,
                    'service_actual': service_actual,
                })
                
            # 记录每个RSU决策
            for rsu_id, vids in rsu_choices.items():
                for vid in vids:
                    decision_records.append({
                        'episode': ep,
                        'step': step_count,
                        'vehicle_id': vid,
                        'rsu_id': rsu_id,
                        'n_rsu_id': n_rsu_by_id.get(rsu_id, 0),
                        'n_rsu_total': n_rsu_total,
                        'q_before': q_before.get(rsu_id, 0),
                    })
        
        # 收集本episode的est_records
        for rec in env._audit_t_est_records:
            rec['episode'] = ep  # 确保episode正确
            est_records_all.append(rec)
                    
    # 构建DataFrame
    est_df = pd.DataFrame(est_records_all)
    decision_df = pd.DataFrame(decision_records)
    step_df = pd.DataFrame(step_records)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    step_df.to_csv(f"{output_prefix}_step_records.csv", index=False)
    decision_df.to_csv(f"{output_prefix}_decision_records.csv", index=False)
    if not est_df.empty:
        est_df.to_csv(f"{output_prefix}_est_records.csv", index=False)
        
    # 分析
    summary = analyze_concurrent_externality(step_df, decision_df, est_df, seed, episodes)
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print_analysis(summary)
    
    return step_df, decision_df, est_df, summary


def analyze_concurrent_externality(step_df, decision_df, est_df, seed, episodes):
    """分析并发外部性"""
    summary = {
        'seed': seed,
        'episodes': episodes,
        'total_step_records': len(step_df),
        'total_decision_records': len(decision_df),
    }
    
    # 分析n_rsu分布
    if not decision_df.empty:
        n_rsu_dist = decision_df['n_rsu_id'].value_counts().sort_index().to_dict()
        summary['n_rsu_id_distribution'] = {int(k): int(v) for k, v in n_rsu_dist.items()}
        
        # n_rsu统计
        summary['n_rsu_stats'] = {
            'mean': float(decision_df['n_rsu_id'].mean()),
            'max': int(decision_df['n_rsu_id'].max()),
            'p50': float(decision_df['n_rsu_id'].quantile(0.5)),
            'p95': float(decision_df['n_rsu_id'].quantile(0.95)),
        }
        
    # 尝试关联est_error和n_rsu_id
    if not est_df.empty and 'est_error' in est_df.columns and not decision_df.empty:
        # 合并数据 - est_df没有step，使用decision_time转换
        # decision_time对应仿真时间，step = int(decision_time / DT) + 1
        DT = 0.1  # 从config
        rsu_est = est_df[est_df['action_type'] == 'RSU'].copy()
        
        if len(rsu_est) > 10 and 'decision_time' in rsu_est.columns:
            # 从decision_time推算step
            rsu_est['step'] = (rsu_est['decision_time'] / DT).astype(int) + 1
            
            # 按step聚合n_rsu_total（使用step_df，它有完整的step数据）
            step_n_rsu = step_df.groupby(['episode', 'step'])['n_rsu_total'].first().reset_index()
            
            merged = rsu_est.merge(
                step_n_rsu, 
                on=['episode', 'step'],
                how='left'
            )
            
            if 'n_rsu_total' in merged.columns and merged['n_rsu_total'].notna().sum() > 10:
                valid = merged.dropna(subset=['est_error', 'n_rsu_total'])
                valid = valid[valid['n_rsu_total'] > 0]  # 只看有RSU选择的步
                
                if len(valid) > 10:
                    corr, p_value = stats.pearsonr(valid['est_error'], valid['n_rsu_total'])
                    summary['correlation_est_error_n_rsu'] = {
                        'pearson_r': float(corr),
                        'p_value': float(p_value),
                        'sample_size': int(len(valid)),
                        'significant': bool(p_value < 0.05),
                    }
                    
                    # 分桶分析
                    bucket_stats = []
                    for n in range(1, int(valid['n_rsu_total'].max()) + 1):
                        bucket = valid[valid['n_rsu_total'] == n]['est_error']
                        if len(bucket) > 0:
                            bucket_stats.append({
                                'n_rsu': n,
                                'count': len(bucket),
                                'mean_error': float(bucket.mean()),
                                'p95_error': float(bucket.quantile(0.95)) if len(bucket) >= 20 else None,
                            })
                    summary['error_by_n_rsu_bucket'] = bucket_stats
                    
    # 队列守恒分析
    if not step_df.empty:
        summary['queue_conservation'] = {}
        for rsu_id in step_df['rsu_id'].unique():
            rsu_data = step_df[step_df['rsu_id'] == rsu_id].copy()
            
            # 守恒误差 = delta_q - (arrivals - service_actual)
            rsu_data['expected_delta'] = rsu_data['arrivals_cycles'] - rsu_data['service_actual']
            rsu_data['conservation_error'] = rsu_data['delta_q'] - rsu_data['expected_delta']
            
            summary['queue_conservation'][f'RSU_{rsu_id}'] = {
                'mean_arrivals': float(rsu_data['arrivals_cycles'].mean()),
                'mean_service_actual': float(rsu_data['service_actual'].mean()),
                'mean_service_capacity': float(rsu_data['service_capacity'].mean()),
                'mean_delta_q': float(rsu_data['delta_q'].mean()),
                'mean_conservation_error': float(rsu_data['conservation_error'].mean()),
            }
            
    return summary


def print_analysis(summary):
    """打印分析结果"""
    print("\n" + "="*80)
    print("ENHANCED CONCURRENT EXTERNALITY AUDIT")
    print("="*80)
    
    print(f"\n[Data Size]")
    print(f"  Step records: {summary['total_step_records']}")
    print(f"  Decision records: {summary['total_decision_records']}")
    
    if 'n_rsu_stats' in summary:
        ns = summary['n_rsu_stats']
        print(f"\n[n_rsu_id Statistics (RSU decisions)]")
        print(f"  Mean: {ns['mean']:.2f}")
        print(f"  Max: {ns['max']}")
        print(f"  P50: {ns['p50']:.2f}")
        print(f"  P95: {ns['p95']:.2f}")
        
    if 'n_rsu_id_distribution' in summary:
        print(f"\n[n_rsu_id Distribution]")
        for n, count in sorted(summary['n_rsu_id_distribution'].items()):
            print(f"  n={n}: {count}")
            
    if 'correlation_est_error_n_rsu' in summary:
        corr = summary['correlation_est_error_n_rsu']
        print(f"\n[Correlation: est_error vs n_rsu_id]")
        print(f"  Pearson r: {corr['pearson_r']:.4f}")
        print(f"  p-value: {corr['p_value']:.4f}")
        print(f"  Sample size: {corr['sample_size']}")
        print(f"  Significant: {corr['significant']}")
        
    if 'error_by_n_rsu_bucket' in summary:
        print(f"\n[Est Error by n_rsu Bucket]")
        for bucket in summary['error_by_n_rsu_bucket']:
            p95_str = f", p95={bucket['p95_error']:.4f}" if bucket['p95_error'] else ""
            print(f"  n={bucket['n_rsu']}: count={bucket['count']}, "
                  f"mean_error={bucket['mean_error']:.4f}{p95_str}")
                  
    if 'queue_conservation' in summary:
        print(f"\n[Queue Conservation]")
        for rsu, stats in summary['queue_conservation'].items():
            print(f"  {rsu}: arrivals={stats['mean_arrivals']:.2e}, "
                  f"service_actual={stats['mean_service_actual']:.2e}, "
                  f"conservation_error={stats['mean_conservation_error']:.2e}")
                  
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Concurrent Externality Audit')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/concurrent_enhanced', help='Output prefix')
    
    args = parser.parse_args()
    run_audit(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
