#!/usr/bin/env python3
"""
修复后领域一致性回归验证脚本

验证目标：
1. 仿真语义正确（contact time, commit shuffle, RSU选择）
2. 并发外部性可学习（est_error vs n_rsu相关性下降）
3. 奖励与真实目标一致（能耗标定、误差→奖励分桶）
4. 未引入新偏置

用法:
    python scripts/audit/post_fix_validation.py --seed 0 --episodes 50 --out out/post_fix_val
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

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def parse_action_target(action, vid):
    """解析动作的target_idx"""
    if isinstance(action, (tuple, list)):
        if vid < len(action):
            act = action[vid]
            if isinstance(act, dict):
                return int(act.get("target", 0))
            act_array = np.asarray(act).flatten()
            return int(act_array[0]) if len(act_array) > 0 else 0
    return 0


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


def run_validation(seed, episodes, output_prefix):
    """运行完整验证"""
    np.random.seed(seed)
    env = VecOffloadingEnv()
    DT = env.config.DT
    
    # 数据收集
    contact_mask_records = []  # Contact time mask验证
    illegal_records = []  # Commit顺序验证
    rsu_selection_records = []  # RSU选择验证
    concurrent_records = []  # 并发外部性验证
    reward_records = []  # 奖励项验证
    task_records = []  # 任务级指标
    step_rsu_records = []  # RSU队列负载
    action_records = []  # 动作分布
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        env._audit_subtask_est = {}
        env._audit_t_est_records = []
        
        done = False
        step_count = 0
        step_n_rsu_map = {}
        
        while not done and step_count < 200:
            step_count += 1
            sim_time = env.time
            
            # 步前队列快照
            q_before = snapshot_rsu_queues(env)
            
            # 获取每个车辆的serving_rsu
            serving_rsu_by_vid = {}
            for v in env.vehicles:
                rsu_id = getattr(v, 'serving_rsu_id', None)
                if rsu_id is None:
                    rsu_id = env._update_serving_rsu(v)
                serving_rsu_by_vid[v.id] = rsu_id
            
            # 随机动作
            action = env.action_space.sample()
            
            # 解析动作，统计
            action_counts = {'local': 0, 'rsu': 0, 'v2v': 0}
            rsu_choices = defaultdict(list)
            
            for v in env.vehicles:
                target_idx = parse_action_target(action, v.id)
                if target_idx == 0:
                    action_counts['local'] += 1
                elif target_idx == 1:
                    action_counts['rsu'] += 1
                    rsu_id = serving_rsu_by_vid.get(v.id)
                    if rsu_id is not None:
                        rsu_choices[rsu_id].append(v.id)
                else:
                    action_counts['v2v'] += 1
            
            n_rsu_by_id = {rsu_id: len(vids) for rsu_id, vids in rsu_choices.items()}
            n_rsu_total = sum(n_rsu_by_id.values())
            
            step_n_rsu_map[round(sim_time, 4)] = {
                'by_id': n_rsu_by_id.copy(),
                'total': n_rsu_total,
                'q_before': q_before.copy(),
            }
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 步后队列
            q_after = snapshot_rsu_queues(env)
            
            # 收集illegal统计
            for v in env.vehicles:
                if getattr(v, 'illegal_action', False):
                    illegal_records.append({
                        'episode': ep,
                        'step': step_count,
                        'vehicle_id': v.id,
                        'illegal_reason': getattr(v, 'illegal_reason', 'unknown'),
                    })
            
            # 收集RSU队列负载
            for rsu_id in range(len(env.rsus)):
                step_rsu_records.append({
                    'episode': ep,
                    'step': step_count,
                    'rsu_id': rsu_id,
                    'q_cycles': q_after.get(rsu_id, 0),
                    'n_rsu_id': n_rsu_by_id.get(rsu_id, 0),
                })
            
            # 收集动作分布
            action_records.append({
                'episode': ep,
                'step': step_count,
                'local': action_counts['local'],
                'rsu': action_counts['rsu'],
                'v2v': action_counts['v2v'],
            })
            
            # 收集奖励
            reward_records.append({
                'episode': ep,
                'step': step_count,
                'r_total_mean': float(np.mean(rewards)),
                'n_rsu_total': n_rsu_total,
            })
        
        # Episode结束后收集RSU决策的est_error
        for rec in env._audit_t_est_records:
            if rec.get('action_type') != 'RSU':
                continue
            
            decision_time = rec.get('decision_time', 0)
            vid = rec.get('vehicle_id')
            
            matched_key = None
            for t in step_n_rsu_map.keys():
                if abs(t - decision_time) < 0.05:
                    matched_key = t
                    break
            
            if matched_key is not None:
                n_rsu_info = step_n_rsu_map[matched_key]
                v = env.vehicles[vid] if vid < len(env.vehicles) else None
                rsu_id = getattr(v, 'serving_rsu_id', -1) if v else -1
                n_rsu_id = n_rsu_info['by_id'].get(rsu_id, 0)
                n_rsu_total = n_rsu_info['total']
            else:
                n_rsu_id = 0
                n_rsu_total = 0
            
            concurrent_records.append({
                'episode': ep,
                'vehicle_id': vid,
                'decision_time': decision_time,
                't_est': rec.get('t_actual_est', 0),
                't_real': rec.get('t_actual_real', 0),
                'est_error': rec.get('est_error', 0),
                'n_rsu_id': n_rsu_id,
                'n_rsu_total': n_rsu_total,
            })
        
        # 收集任务级指标
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
                    'start_time': dag.start_time,
                    'completion_time': env.time - dag.start_time if dag.is_finished else None,
                })
    
    # 转换为DataFrame
    illegal_df = pd.DataFrame(illegal_records)
    concurrent_df = pd.DataFrame(concurrent_records)
    reward_df = pd.DataFrame(reward_records)
    task_df = pd.DataFrame(task_records)
    step_rsu_df = pd.DataFrame(step_rsu_records)
    action_df = pd.DataFrame(action_records)
    
    # 保存原始数据
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    concurrent_df.to_csv(f"{output_prefix}_concurrent.csv", index=False)
    task_df.to_csv(f"{output_prefix}_tasks.csv", index=False)
    step_rsu_df.to_csv(f"{output_prefix}_rsu_load.csv", index=False)
    action_df.to_csv(f"{output_prefix}_actions.csv", index=False)
    illegal_df.to_csv(f"{output_prefix}_illegal.csv", index=False)
    
    # 分析
    summary = analyze_validation(
        concurrent_df, task_df, step_rsu_df, action_df, illegal_df, reward_df,
        env, seed, episodes
    )
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print_summary(summary)
    
    return summary


def analyze_validation(concurrent_df, task_df, step_rsu_df, action_df, illegal_df, reward_df, env, seed, episodes):
    """分析验证结果"""
    summary = {
        'seed': seed,
        'episodes': episodes,
        'config': {
            'USE_POST_COMMIT_SNAPSHOT': getattr(env.config, 'USE_POST_COMMIT_SNAPSHOT', True),
            'USE_RELATIVE_ADVANTAGE': getattr(env.config, 'USE_RELATIVE_ADVANTAGE', False),
            'RSU_LOAD_WEIGHT': getattr(env.config, 'RSU_LOAD_WEIGHT', 0.3),
            'SLACK_S0': getattr(env.config, 'SLACK_S0', 3.0),
            'E_REF_CALIBRATED': getattr(env.config, 'E_REF_CALIBRATED', 0.01),
        },
    }
    
    # ========================================
    # 1.2 Commit shuffle 公平性验证
    # ========================================
    if not illegal_df.empty:
        illegal_by_vid = illegal_df.groupby('vehicle_id').size()
        low_ids = illegal_by_vid[illegal_by_vid.index < 15]
        high_ids = illegal_by_vid[illegal_by_vid.index >= 15]
        
        if len(low_ids) > 0 and len(high_ids) > 0:
            t_stat, p_value = stats.ttest_ind(
                low_ids.values, high_ids.values, equal_var=False
            )
            summary['commit_shuffle'] = {
                'low_id_mean': float(low_ids.mean()),
                'high_id_mean': float(high_ids.mean()),
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'is_fair': bool(p_value > 0.1),  # p>0.1 说明不显著
            }
        else:
            summary['commit_shuffle'] = {'note': 'insufficient data'}
    
    # ========================================
    # 2.1 est_error vs 并发相关性
    # ========================================
    if not concurrent_df.empty:
        valid_conc = concurrent_df[concurrent_df['n_rsu_total'] > 0].copy()
        if len(valid_conc) > 20:
            corr, p_val = stats.pearsonr(valid_conc['est_error'], valid_conc['n_rsu_total'])
            
            # 分桶曲线
            bucket_stats = []
            for n in range(1, int(valid_conc['n_rsu_total'].max()) + 1):
                bucket = valid_conc[valid_conc['n_rsu_total'] == n]['est_error']
                if len(bucket) > 0:
                    bucket_stats.append({
                        'n_rsu': n,
                        'count': int(len(bucket)),
                        'mean_error': float(bucket.mean()),
                        'std_error': float(bucket.std()),
                    })
            
            summary['concurrent_externality'] = {
                'corr_est_error_n_rsu': float(corr),
                'p_value': float(p_val),
                'sample_size': int(len(valid_conc)),
                'bucket_stats': bucket_stats,
            }
    
    # ========================================
    # 2.2 误差→奖励 分桶分析
    # ========================================
    if not concurrent_df.empty and not reward_df.empty:
        # 合并数据
        merged = concurrent_df.copy()
        merged['sim_time_approx'] = (merged['decision_time'] / 0.1).astype(int) * 0.1
        
        # 按est_error分位数分桶
        if len(merged) > 20:
            p10 = merged['est_error'].quantile(0.1)
            p90 = merged['est_error'].quantile(0.9)
            
            low_error = merged[merged['est_error'] <= p10]
            high_error = merged[merged['est_error'] >= p90]
            
            # 注意：这里我们没有每条记录的r_time，只能用步级r_total近似
            # 真正的验证需要在env内部记录每条决策的r_time
            summary['error_reward_bucket'] = {
                'p10_threshold': float(p10),
                'p90_threshold': float(p90),
                'low_error_count': int(len(low_error)),
                'high_error_count': int(len(high_error)),
                'low_error_mean': float(low_error['est_error'].mean()),
                'high_error_mean': float(high_error['est_error'].mean()),
                'note': '需要env内部记录per-decision r_time以完成完整验证',
            }
    
    # ========================================
    # 3. 任务级指标
    # ========================================
    if not task_df.empty:
        total_tasks = len(task_df)
        finished = task_df['is_finished'].sum()
        failed = task_df['is_failed'].sum()
        deadline_miss = task_df[task_df['fail_reason'] == 'deadline'].shape[0]
        
        completion_times = task_df[task_df['completion_time'].notna()]['completion_time']
        
        summary['task_metrics'] = {
            'total_tasks': int(total_tasks),
            'success_rate': float(finished / total_tasks) if total_tasks > 0 else 0,
            'fail_rate': float(failed / total_tasks) if total_tasks > 0 else 0,
            'deadline_miss_rate': float(deadline_miss / total_tasks) if total_tasks > 0 else 0,
            'completion_time_mean': float(completion_times.mean()) if len(completion_times) > 0 else None,
            'completion_time_p95': float(completion_times.quantile(0.95)) if len(completion_times) >= 20 else None,
        }
    
    # ========================================
    # RSU队列负载
    # ========================================
    if not step_rsu_df.empty:
        rsu_load_stats = {}
        for rsu_id in step_rsu_df['rsu_id'].unique():
            rsu_data = step_rsu_df[step_rsu_df['rsu_id'] == rsu_id]['q_cycles']
            rsu_load_stats[f'RSU_{rsu_id}'] = {
                'mean_cycles': float(rsu_data.mean()),
                'max_cycles': float(rsu_data.max()),
                'p95_cycles': float(rsu_data.quantile(0.95)),
            }
        summary['rsu_queue_load'] = rsu_load_stats
    
    # ========================================
    # 动作分布
    # ========================================
    if not action_df.empty:
        total_actions = action_df[['local', 'rsu', 'v2v']].sum().sum()
        summary['action_distribution'] = {
            'local_frac': float(action_df['local'].sum() / total_actions) if total_actions > 0 else 0,
            'rsu_frac': float(action_df['rsu'].sum() / total_actions) if total_actions > 0 else 0,
            'v2v_frac': float(action_df['v2v'].sum() / total_actions) if total_actions > 0 else 0,
        }
    
    # ========================================
    # 奖励分布
    # ========================================
    if not reward_df.empty:
        summary['reward_stats'] = {
            'r_total_mean': float(reward_df['r_total_mean'].mean()),
            'r_total_std': float(reward_df['r_total_mean'].std()),
            'r_total_p95': float(reward_df['r_total_mean'].quantile(0.95)),
        }
    
    return summary


def print_summary(summary):
    """打印摘要"""
    print("\n" + "="*80)
    print("POST-FIX VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n[Config]")
    for k, v in summary.get('config', {}).items():
        print(f"  {k}: {v}")
    
    if 'commit_shuffle' in summary:
        cs = summary['commit_shuffle']
        print(f"\n[1.2 Commit Shuffle Fairness]")
        if 'p_value' in cs:
            print(f"  Low ID (0-14) illegal mean: {cs['low_id_mean']:.2f}")
            print(f"  High ID (15-29) illegal mean: {cs['high_id_mean']:.2f}")
            print(f"  t-test p-value: {cs['p_value']:.4f}")
            print(f"  Is Fair (p>0.1): {cs['is_fair']}")
    
    if 'concurrent_externality' in summary:
        ce = summary['concurrent_externality']
        print(f"\n[2.1 Concurrent Externality]")
        print(f"  corr(est_error, n_rsu_total): {ce['corr_est_error_n_rsu']:.4f}")
        print(f"  p-value: {ce['p_value']:.4f}")
        print(f"  Sample size: {ce['sample_size']}")
        
        if 'bucket_stats' in ce:
            print(f"\n  [n_rsu Bucket Stats]")
            for bs in ce['bucket_stats'][:8]:
                print(f"    n={bs['n_rsu']}: count={bs['count']}, mean_error={bs['mean_error']:.4f}")
    
    if 'task_metrics' in summary:
        tm = summary['task_metrics']
        print(f"\n[3. Task Metrics]")
        print(f"  Total tasks: {tm['total_tasks']}")
        print(f"  Success rate: {tm['success_rate']:.2%}")
        print(f"  Deadline miss rate: {tm['deadline_miss_rate']:.2%}")
        if tm['completion_time_mean']:
            print(f"  Completion time mean: {tm['completion_time_mean']:.2f}s")
    
    if 'rsu_queue_load' in summary:
        print(f"\n[RSU Queue Load]")
        for rsu, stats in summary['rsu_queue_load'].items():
            print(f"  {rsu}: mean={stats['mean_cycles']:.2e}, p95={stats['p95_cycles']:.2e}")
    
    if 'action_distribution' in summary:
        ad = summary['action_distribution']
        print(f"\n[Action Distribution]")
        print(f"  Local: {ad['local_frac']:.1%}, RSU: {ad['rsu_frac']:.1%}, V2V: {ad['v2v_frac']:.1%}")
    
    if 'reward_stats' in summary:
        rs = summary['reward_stats']
        print(f"\n[Reward Stats]")
        print(f"  r_total mean: {rs['r_total_mean']:.4f}")
        print(f"  r_total std: {rs['r_total_std']:.4f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Post-Fix Validation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--out', type=str, default='out/post_fix_val')
    
    args = parser.parse_args()
    run_validation(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
