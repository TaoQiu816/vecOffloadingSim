#!/usr/bin/env python3
"""
论文级领域一致性回归验证脚本 v3

验证目标：
1. per-decision奖励分桶闭环验证
2. 能耗标定真实分位数
3. RSU_2未使用的可验证证据
4. 并发外部性→奖励敏感性

用法:
    python scripts/audit/post_fix_validation_v3.py --seed 0 --episodes 50 --out out/val_v3
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

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def snapshot_rsu_queues(env):
    """快照所有RSU队列状态(cycles)"""
    snapshot = {}
    for rsu_id in range(len(env.rsus)):
        rsu_q = env.rsu_cpu_q.get(rsu_id, {})
        total_cycles = sum(
            sum(j.rem_cycles for j in q) for q in rsu_q.values()
        )
        snapshot[rsu_id] = total_cycles
    return snapshot


def get_rsu_positions(env):
    """获取RSU位置"""
    positions = {}
    for i, rsu in enumerate(env.rsus):
        positions[i] = {
            'x': float(rsu.position[0]),
            'y': float(rsu.position[1]),
            'coverage_radius': float(getattr(env.config, 'RSU_COVERAGE_RADIUS', 350)),
        }
    return positions


def run_validation(seed, episodes, output_prefix):
    """运行完整验证"""
    import random
    from configs.config import SystemConfig as Cfg
    from envs.vec_offloading_env import VecOffloadingEnv
    
    # [P0-2修复] 统一设置所有随机源
    np.random.seed(seed)
    random.seed(seed)
    
    # 启用per-decision审计
    Cfg.AUDIT_PER_DECISION_REWARD = True
    
    env = VecOffloadingEnv()
    
    # 数据收集
    all_per_decision = []
    nearest_rsu_stats = defaultdict(int)
    vehicle_positions = []
    rsu_queue_records = []
    action_records = []
    task_records = []
    
    # 获取RSU位置（一次性）
    rsu_positions = get_rsu_positions(env)
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            step_count += 1
            
            # 记录每个车辆的nearest RSU
            for v in env.vehicles:
                rsu_id = getattr(v, 'serving_rsu_id', None)
                if rsu_id is None:
                    rsu_id = env._update_serving_rsu(v)
                nearest_rsu_stats[rsu_id] += 1
                
                # 记录车辆位置（每10步采样一次）
                if step_count % 10 == 1:
                    vehicle_positions.append({
                        'episode': ep,
                        'step': step_count,
                        'veh_id': v.id,
                        'x': float(v.pos[0]),
                        'y': float(v.pos[1]),
                        'nearest_rsu': rsu_id,
                    })
            
            # 随机动作
            action = env.action_space.sample()
            
            # 统计动作分布
            action_counts = {'local': 0, 'rsu': 0, 'v2v': 0}
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
                    action_counts['local'] += 1
                elif target_idx == 1:
                    action_counts['rsu'] += 1
                else:
                    action_counts['v2v'] += 1
            
            action_records.append({
                'episode': ep,
                'step': step_count,
                **action_counts,
            })
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录RSU队列负载
            for rsu_id in range(len(env.rsus)):
                q_cycles = snapshot_rsu_queues(env).get(rsu_id, 0)
                rsu_queue_records.append({
                    'episode': ep,
                    'step': step_count,
                    'rsu_id': rsu_id,
                    'q_cycles': q_cycles,
                })
        
        # 收集per-decision奖励记录
        if hasattr(env, '_audit_per_decision_rewards'):
            all_per_decision.extend(env._audit_per_decision_rewards)
        
        # 收集任务指标
        for v in env.vehicles:
            dag = v.task_dag
            if dag is not None:
                task_records.append({
                    'episode': ep,
                    'vehicle_id': v.id,
                    'is_finished': dag.is_finished,
                    'is_failed': dag.is_failed,
                    'fail_reason': dag.fail_reason if dag.is_failed else None,
                    'deadline': dag.deadline,
                })
    
    # 转换为DataFrame
    per_decision_df = pd.DataFrame(all_per_decision) if all_per_decision else pd.DataFrame()
    vehicle_pos_df = pd.DataFrame(vehicle_positions)
    rsu_queue_df = pd.DataFrame(rsu_queue_records)
    action_df = pd.DataFrame(action_records)
    task_df = pd.DataFrame(task_records)
    
    # 保存原始数据
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    if not per_decision_df.empty:
        per_decision_df.to_csv(f"{output_prefix}_per_decision.csv", index=False)
    vehicle_pos_df.to_csv(f"{output_prefix}_veh_positions.csv", index=False)
    rsu_queue_df.to_csv(f"{output_prefix}_rsu_queue.csv", index=False)
    action_df.to_csv(f"{output_prefix}_actions.csv", index=False)
    task_df.to_csv(f"{output_prefix}_tasks.csv", index=False)
    
    # 分析
    summary = analyze_validation(
        per_decision_df, vehicle_pos_df, rsu_queue_df, action_df, task_df,
        rsu_positions, nearest_rsu_stats, env, seed, episodes
    )
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print_summary(summary)
    
    return summary


def analyze_validation(per_decision_df, vehicle_pos_df, rsu_queue_df, action_df, task_df,
                       rsu_positions, nearest_rsu_stats, env, seed, episodes):
    """分析验证结果"""
    summary = {
        'seed': seed,
        'episodes': episodes,
        'rsu_positions': rsu_positions,
    }
    
    # ========================================
    # 1. RSU_2 未使用证据
    # ========================================
    total_nearest = sum(nearest_rsu_stats.values())
    summary['nearest_rsu_distribution'] = {
        f'RSU_{k}': {
            'count': int(v),
            'fraction': float(v / total_nearest) if total_nearest > 0 else 0,
        }
        for k, v in sorted(nearest_rsu_stats.items())
    }
    
    # 计算每个RSU到道路中心的距离
    road_y = 0  # 假设道路在y=0
    for rsu_id, pos in rsu_positions.items():
        dist_to_road = abs(pos['y'] - road_y)
        rsu_key = f'RSU_{rsu_id}'
        if rsu_key not in summary['nearest_rsu_distribution']:
            summary['nearest_rsu_distribution'][rsu_key] = {
                'count': 0,
                'fraction': 0.0,
            }
        summary['nearest_rsu_distribution'][rsu_key]['dist_to_road'] = dist_to_road
    
    # ========================================
    # 2. Per-decision奖励分桶分析
    # ========================================
    if not per_decision_df.empty:
        # 仅RSU动作
        rsu_df = per_decision_df[per_decision_df['action_type'] == 'RSU'].copy()
        
        if len(rsu_df) > 20:
            # 2.1 按est_error分桶
            # 注意：我们需要从_audit_t_est_records获取est_error，这里暂用t_est做近似分析
            # 实际上需要合并数据
            
            # 按n_rsu_total分桶
            bucket_by_n_rsu = []
            for n in sorted(rsu_df['n_rsu_total'].unique()):
                bucket = rsu_df[rsu_df['n_rsu_total'] == n]
                if len(bucket) >= 5:
                    bucket_by_n_rsu.append({
                        'n_rsu_total': int(n),
                        'count': int(len(bucket)),
                        'r_lat_mean': float(bucket['r_lat'].mean()),
                        'r_queue_mean': float(bucket['r_queue'].mean()),
                        'r_total_mean': float(bucket['r_total'].mean()),
                        'r_lat_std': float(bucket['r_lat'].std()),
                    })
            
            summary['bucket_by_n_rsu'] = bucket_by_n_rsu
            
            # 按t_est分位数分桶
            if 't_est' in rsu_df.columns and rsu_df['t_est'].std() > 1e-6:
                p10 = rsu_df['t_est'].quantile(0.1)
                p50 = rsu_df['t_est'].quantile(0.5)
                p90 = rsu_df['t_est'].quantile(0.9)
                
                low_t = rsu_df[rsu_df['t_est'] <= p10]
                mid_t = rsu_df[(rsu_df['t_est'] > p10) & (rsu_df['t_est'] <= p50)]
                high_t = rsu_df[rsu_df['t_est'] > p90]
                
                summary['bucket_by_t_est'] = {
                    'p10_threshold': float(p10),
                    'p50_threshold': float(p50),
                    'p90_threshold': float(p90),
                    'low_t_est': {
                        'count': int(len(low_t)),
                        'r_lat_mean': float(low_t['r_lat'].mean()) if len(low_t) > 0 else None,
                        'r_total_mean': float(low_t['r_total'].mean()) if len(low_t) > 0 else None,
                    },
                    'mid_t_est': {
                        'count': int(len(mid_t)),
                        'r_lat_mean': float(mid_t['r_lat'].mean()) if len(mid_t) > 0 else None,
                        'r_total_mean': float(mid_t['r_total'].mean()) if len(mid_t) > 0 else None,
                    },
                    'high_t_est': {
                        'count': int(len(high_t)),
                        'r_lat_mean': float(high_t['r_lat'].mean()) if len(high_t) > 0 else None,
                        'r_total_mean': float(high_t['r_total'].mean()) if len(high_t) > 0 else None,
                    },
                }
        
        # 2.2 能耗标定真实分位数
        if 'r_energy' in per_decision_df.columns and 'r_lat' in per_decision_df.columns:
            r_energy_abs = per_decision_df['r_energy'].abs()
            r_lat_abs = per_decision_df['r_lat'].abs()
            
            summary['energy_calibration'] = {
                'r_energy': {
                    'p50': float(r_energy_abs.quantile(0.5)),
                    'p90': float(r_energy_abs.quantile(0.9)),
                    'p95': float(r_energy_abs.quantile(0.95)),
                    'max': float(r_energy_abs.max()),
                },
                'r_lat': {
                    'p50': float(r_lat_abs.quantile(0.5)),
                    'p90': float(r_lat_abs.quantile(0.9)),
                    'p95': float(r_lat_abs.quantile(0.95)),
                    'max': float(r_lat_abs.max()),
                },
            }
            
            # 计算比值
            ratios = {}
            for pct in ['p50', 'p90', 'p95']:
                r_lat_val = summary['energy_calibration']['r_lat'][pct]
                r_energy_val = summary['energy_calibration']['r_energy'][pct]
                if r_lat_val > 1e-9:
                    ratios[pct] = float(r_energy_val / r_lat_val)
                else:
                    ratios[pct] = None
            summary['energy_calibration']['ratios'] = ratios
        
        # 2.3 e_tx分布（用于E_REF标定）
        if 'e_tx' in per_decision_df.columns:
            e_tx = per_decision_df['e_tx']
            e_tx_nonzero = e_tx[e_tx > 1e-12]
            if len(e_tx_nonzero) > 0:
                summary['e_tx_distribution'] = {
                    'p50': float(e_tx_nonzero.quantile(0.5)),
                    'p80': float(e_tx_nonzero.quantile(0.8)),
                    'p90': float(e_tx_nonzero.quantile(0.9)),
                    'p95': float(e_tx_nonzero.quantile(0.95)),
                    'max': float(e_tx_nonzero.max()),
                    'recommended_E_REF': float(e_tx_nonzero.quantile(0.85)),
                }
    
    # ========================================
    # 3. 任务级指标
    # ========================================
    if not task_df.empty:
        total_tasks = len(task_df)
        finished = task_df['is_finished'].sum()
        failed = task_df['is_failed'].sum()
        deadline_miss = task_df[task_df['fail_reason'] == 'deadline'].shape[0]
        
        summary['task_metrics'] = {
            'total_tasks': int(total_tasks),
            'success_rate': float(finished / total_tasks) if total_tasks > 0 else 0,
            'fail_rate': float(failed / total_tasks) if total_tasks > 0 else 0,
            'deadline_miss_rate': float(deadline_miss / total_tasks) if total_tasks > 0 else 0,
        }
    
    # ========================================
    # 4. RSU队列负载
    # ========================================
    if not rsu_queue_df.empty:
        rsu_load_stats = {}
        for rsu_id in rsu_queue_df['rsu_id'].unique():
            rsu_data = rsu_queue_df[rsu_queue_df['rsu_id'] == rsu_id]['q_cycles']
            rsu_load_stats[f'RSU_{rsu_id}'] = {
                'mean_cycles': float(rsu_data.mean()),
                'max_cycles': float(rsu_data.max()),
                'p95_cycles': float(rsu_data.quantile(0.95)),
            }
        summary['rsu_queue_load'] = rsu_load_stats
    
    # ========================================
    # 5. 动作分布
    # ========================================
    if not action_df.empty:
        total_actions = action_df[['local', 'rsu', 'v2v']].sum().sum()
        summary['action_distribution'] = {
            'local_frac': float(action_df['local'].sum() / total_actions) if total_actions > 0 else 0,
            'rsu_frac': float(action_df['rsu'].sum() / total_actions) if total_actions > 0 else 0,
            'v2v_frac': float(action_df['v2v'].sum() / total_actions) if total_actions > 0 else 0,
        }
    
    return summary


def print_summary(summary):
    """打印摘要"""
    print("\n" + "="*80)
    print("POST-FIX VALIDATION SUMMARY V3")
    print("="*80)
    
    print(f"\n[RSU Positions]")
    for rsu_id, pos in summary.get('rsu_positions', {}).items():
        print(f"  RSU_{rsu_id}: x={pos['x']:.1f}, y={pos['y']:.1f}, coverage={pos['coverage_radius']:.1f}m")
    
    print(f"\n[Nearest RSU Distribution (证明RSU_2未使用)]")
    for rsu, stats in summary.get('nearest_rsu_distribution', {}).items():
        print(f"  {rsu}: {stats['fraction']*100:.1f}% ({stats['count']} samples)")
    
    if 'bucket_by_n_rsu' in summary:
        print(f"\n[RSU动作按n_rsu_total分桶 (验证并发→奖励敏感性)]")
        print(f"  {'n_rsu':>6} {'count':>6} {'r_lat_mean':>12} {'r_queue_mean':>12} {'r_total_mean':>12}")
        for b in summary['bucket_by_n_rsu'][:8]:
            print(f"  {b['n_rsu_total']:>6} {b['count']:>6} {b['r_lat_mean']:>12.4f} {b['r_queue_mean']:>12.4f} {b['r_total_mean']:>12.4f}")
    
    if 'energy_calibration' in summary:
        ec = summary['energy_calibration']
        print(f"\n[能耗标定真实分位数]")
        print(f"  |r_energy|: p50={ec['r_energy']['p50']:.4f}, p90={ec['r_energy']['p90']:.4f}, p95={ec['r_energy']['p95']:.4f}")
        print(f"  |r_lat|:    p50={ec['r_lat']['p50']:.4f}, p90={ec['r_lat']['p90']:.4f}, p95={ec['r_lat']['p95']:.4f}")
        print(f"  Ratios: p50={ec['ratios'].get('p50', 'N/A')}, p90={ec['ratios'].get('p90', 'N/A')}, p95={ec['ratios'].get('p95', 'N/A')}")
        print(f"  Target range: [0.2, 0.5]")
    
    if 'e_tx_distribution' in summary:
        etx = summary['e_tx_distribution']
        print(f"\n[E_tx分布 (用于E_REF标定)]")
        print(f"  p50={etx['p50']:.6f}J, p80={etx['p80']:.6f}J, p90={etx['p90']:.6f}J")
        print(f"  推荐 E_REF = {etx['recommended_E_REF']:.6f}J")
    
    if 'task_metrics' in summary:
        tm = summary['task_metrics']
        print(f"\n[任务指标]")
        print(f"  Success rate: {tm['success_rate']:.2%}")
        print(f"  Deadline miss rate: {tm['deadline_miss_rate']:.2%}")
    
    if 'action_distribution' in summary:
        ad = summary['action_distribution']
        print(f"\n[动作分布]")
        print(f"  Local: {ad['local_frac']:.1%}, RSU: {ad['rsu_frac']:.1%}, V2V: {ad['v2v_frac']:.1%}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Post-Fix Validation V3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--out', type=str, default='out/val_v3')
    
    args = parser.parse_args()
    run_validation(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
