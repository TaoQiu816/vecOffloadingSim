#!/usr/bin/env python3
"""
回归证据审计脚本 - 控制混杂变量分析并发外部性

目标:
1. 收集决策级数据: est_error, n_rsu_id, n_rsu_total, Q_before, cycles, data, rsu_id
2. 控制混杂回归: est_error ~ n_rsu + Q_before + cycles + data + episode_fixed_effect
3. 误差→奖励误导闭环: corr(r_lat, est_error)
4. Per-RSU粒度相关性: corr(est_error, n_rsu_id) 分桶曲线

用法:
    python scripts/audit/regression_evidence_audit.py --seed 0 --episodes 50 --out out/regression_audit
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


def run_audit(seed, episodes, output_prefix):
    """运行回归证据审计"""
    np.random.seed(seed)
    env = VecOffloadingEnv()
    DT = env.config.DT
    
    # 决策级记录
    decision_records = []
    # 步级记录(用于per-RSU分析)
    step_rsu_records = []
    # 奖励记录
    reward_records = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        env._audit_subtask_est = {}
        env._audit_t_est_records = []
        
        done = False
        step_count = 0
        
        # 每步的决策缓存，用于与完成时配对
        # key = (vid, subtask_id) -> 决策信息(包含n_rsu_id等)
        episode_decision_cache = {}
        
        # 步级n_rsu映射: decision_time -> {rsu_id: n_rsu_id, 'total': n_rsu_total}
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
            
            # 解析动作，统计RSU选择
            rsu_choices = defaultdict(list)  # rsu_id -> [vid]
            for v in env.vehicles:
                target_idx = parse_action_target(action, v.id)
                if target_idx == 1:  # RSU选择
                    rsu_id = serving_rsu_by_vid.get(v.id)
                    if rsu_id is not None:
                        rsu_choices[rsu_id].append(v.id)
            
            # 计算n_rsu统计
            n_rsu_by_id = {rsu_id: len(vids) for rsu_id, vids in rsu_choices.items()}
            n_rsu_total = sum(n_rsu_by_id.values())
            
            # 存储本步的n_rsu信息（用sim_time作为key）
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
            
            # 记录步级RSU数据
            for rsu_id in range(len(env.rsus)):
                step_rsu_records.append({
                    'episode': ep,
                    'step': step_count,
                    'sim_time': sim_time,
                    'rsu_id': rsu_id,
                    'n_rsu_id': n_rsu_by_id.get(rsu_id, 0),
                    'n_rsu_total': n_rsu_total,
                    'q_before': q_before.get(rsu_id, 0),
                    'q_after': q_after.get(rsu_id, 0),
                    'delta_q': q_after.get(rsu_id, 0) - q_before.get(rsu_id, 0),
                })
            
            # 记录本步的奖励信息
            reward_records.append({
                'episode': ep,
                'step': step_count,
                'sim_time': sim_time,
                'n_rsu_total': n_rsu_total,
                'r_total_mean': sum(rewards) / len(rewards) if len(rewards) > 0 else 0,
            })
        
        # Episode结束后，从env收集est_error数据
        for rec in env._audit_t_est_records:
            if rec.get('action_type') != 'RSU':
                continue
            
            decision_time = rec.get('decision_time', 0)
            vid = rec.get('vehicle_id')
            
            # 根据decision_time找到对应的n_rsu信息
            # 允许小误差匹配
            matched_key = None
            for t in step_n_rsu_map.keys():
                if abs(t - decision_time) < 0.05:  # 允许0.05s误差
                    matched_key = t
                    break
            
            if matched_key is not None:
                n_rsu_info = step_n_rsu_map[matched_key]
                # 获取该车辆的serving_rsu
                v = env.vehicles[vid]
                rsu_id = getattr(v, 'serving_rsu_id', -1)
                
                n_rsu_id = n_rsu_info['by_id'].get(rsu_id, 0)
                n_rsu_total = n_rsu_info['total']
                q_before = n_rsu_info['q_before'].get(rsu_id, 0)
            else:
                rsu_id = -1
                n_rsu_id = 0
                n_rsu_total = 0
                q_before = 0
            
            # 获取任务信息
            dag = env.vehicles[vid].task_dag if vid < len(env.vehicles) else None
            cycles = 0
            data_bits = 0
            if dag is not None and hasattr(dag, 'profiles'):
                subtask_id = rec.get('subtask_id', 0)
                if subtask_id < len(dag.profiles):
                    cycles = dag.profiles[subtask_id].get('cycles', 0)
                    data_bits = dag.profiles[subtask_id].get('data', 0)
            
            decision_records.append({
                'episode': ep,
                'vehicle_id': vid,
                'subtask_id': rec.get('subtask_id'),
                'decision_time': decision_time,
                'finish_time': rec.get('finish_time', 0),
                't_est': rec.get('t_actual_est', 0),
                't_real': rec.get('t_actual_real', 0),
                'est_error': rec.get('est_error', 0),
                'rsu_id': rsu_id,
                'n_rsu_id': n_rsu_id,
                'n_rsu_total': n_rsu_total,
                'q_before': q_before,
                'cycles': cycles,
                'data_bits': data_bits,
            })
    
    # 转换为DataFrame
    decision_df = pd.DataFrame(decision_records)
    step_rsu_df = pd.DataFrame(step_rsu_records)
    reward_df = pd.DataFrame(reward_records)
    
    # 保存原始数据
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    decision_df.to_csv(f"{output_prefix}_decision_records.csv", index=False)
    step_rsu_df.to_csv(f"{output_prefix}_step_rsu_records.csv", index=False)
    reward_df.to_csv(f"{output_prefix}_reward_records.csv", index=False)
    
    # 分析
    summary = analyze_regression(decision_df, step_rsu_df, reward_df, seed, episodes)
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print_analysis(summary)
    
    return decision_df, step_rsu_df, reward_df, summary


def analyze_regression(decision_df, step_rsu_df, reward_df, seed, episodes):
    """执行回归分析"""
    summary = {
        'seed': seed,
        'episodes': episodes,
        'total_decision_records': len(decision_df),
    }
    
    if decision_df.empty:
        return summary
    
    # 过滤有效数据（放宽条件，只要有n_rsu信息即可）
    valid_df = decision_df[
        (decision_df['n_rsu_total'] > 0) & 
        (decision_df['est_error'].notna())
    ].copy()
    
    summary['valid_records'] = len(valid_df)
    
    if len(valid_df) < 50:
        return summary
    
    # ==========================================
    # 1.1 控制混杂回归
    # ==========================================
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        # 归一化特征用于回归
        valid_df['q_before_norm'] = valid_df['q_before'] / 1e9  # 归一化到GCycles
        valid_df['cycles_norm'] = valid_df['cycles'] / 1e6  # 归一化到MCycles
        valid_df['data_norm'] = valid_df['data_bits'] / 1e6  # 归一化到Mbits
        
        # 简单线性回归: est_error ~ n_rsu_total
        simple_model = ols('est_error ~ n_rsu_total', data=valid_df).fit()
        
        # 控制混杂变量的回归
        # est_error ~ n_rsu_total + q_before_norm + cycles_norm + data_norm
        controlled_model = ols(
            'est_error ~ n_rsu_total + q_before_norm + cycles_norm + data_norm',
            data=valid_df
        ).fit()
        
        # 带episode固定效应的回归
        valid_df['episode_factor'] = valid_df['episode'].astype('category')
        try:
            fe_model = ols(
                'est_error ~ n_rsu_total + q_before_norm + cycles_norm + data_norm + C(episode)',
                data=valid_df
            ).fit()
            fe_n_rsu_coef = fe_model.params.get('n_rsu_total', np.nan)
            fe_n_rsu_pval = fe_model.pvalues.get('n_rsu_total', np.nan)
        except Exception:
            fe_n_rsu_coef = np.nan
            fe_n_rsu_pval = np.nan
        
        summary['regression'] = {
            'simple': {
                'n_rsu_coef': float(simple_model.params.get('n_rsu_total', np.nan)),
                'n_rsu_pvalue': float(simple_model.pvalues.get('n_rsu_total', np.nan)),
                'r_squared': float(simple_model.rsquared),
            },
            'controlled': {
                'n_rsu_coef': float(controlled_model.params.get('n_rsu_total', np.nan)),
                'n_rsu_pvalue': float(controlled_model.pvalues.get('n_rsu_total', np.nan)),
                'q_before_coef': float(controlled_model.params.get('q_before_norm', np.nan)),
                'cycles_coef': float(controlled_model.params.get('cycles_norm', np.nan)),
                'data_coef': float(controlled_model.params.get('data_norm', np.nan)),
                'r_squared': float(controlled_model.rsquared),
            },
            'fixed_effect': {
                'n_rsu_coef': float(fe_n_rsu_coef),
                'n_rsu_pvalue': float(fe_n_rsu_pval),
            },
            'interpretation': (
                "n_rsu_total系数仍显著(p<0.05)说明并发本身导致误差上升，"
                "而非混杂变量(队列初值/任务规模)的作用"
            ) if controlled_model.pvalues.get('n_rsu_total', 1) < 0.05 else (
                "n_rsu_total系数不显著，误差可能主要来自队列初值或任务规模"
            )
        }
        
    except ImportError:
        summary['regression'] = {'error': 'statsmodels not installed'}
    except Exception as e:
        summary['regression'] = {'error': str(e)}
    
    # ==========================================
    # 1.2 误差→奖励误导闭环
    # ==========================================
    # 合并reward数据到decision_df (通过episode和decision_time匹配sim_time)
    DT = 0.1
    if not reward_df.empty and len(valid_df) > 0:
        # 将decision_time转换为sim_time进行匹配
        valid_df['sim_time_approx'] = (valid_df['decision_time'] / DT).astype(int) * DT
        
        # 合并
        merged = valid_df.merge(
            reward_df[['episode', 'sim_time', 'n_rsu_total', 'r_total_mean']],
            left_on=['episode', 'sim_time_approx'],
            right_on=['episode', 'sim_time'],
            how='left',
            suffixes=('', '_reward')
        )
        
        if 'r_total_mean' in merged.columns:
            valid_merged = merged.dropna(subset=['est_error', 'r_total_mean'])
            
            if len(valid_merged) > 20:
                # 相关性
                corr, pval = stats.pearsonr(valid_merged['est_error'], valid_merged['r_total_mean'])
                
                # 分位数比较
                q25 = valid_merged['est_error'].quantile(0.25)
                q75 = valid_merged['est_error'].quantile(0.75)
                
                low_error = valid_merged[valid_merged['est_error'] <= q25]['r_total_mean']
                high_error = valid_merged[valid_merged['est_error'] >= q75]['r_total_mean']
                
                summary['error_reward_link'] = {
                    'corr_est_error_r_total': float(corr),
                    'p_value': float(pval),
                    'mean_r_total_low_error': float(low_error.mean()) if len(low_error) > 0 else np.nan,
                    'mean_r_total_high_error': float(high_error.mean()) if len(high_error) > 0 else np.nan,
                    'reward_bias': (
                        "高误差样本获得更高/相当奖励 → 奖励在并发高峰发错钱"
                        if float(high_error.mean()) >= float(low_error.mean()) - 0.1 else
                        "高误差样本获得更低奖励 → 奖励信号正确惩罚了误差"
                    ),
                }
    
    # ==========================================
    # 1.3 Per-RSU粒度分析
    # ==========================================
    if 'rsu_id' in valid_df.columns and 'n_rsu_id' in valid_df.columns:
        # 只看有有效rsu_id的记录
        per_rsu = valid_df[valid_df['rsu_id'] >= 0].copy()
        
        if len(per_rsu) > 20:
            # 相关性
            corr_per_rsu, pval_per_rsu = stats.pearsonr(per_rsu['est_error'], per_rsu['n_rsu_id'])
            
            # 分桶曲线
            bucket_stats = []
            for n in range(1, int(per_rsu['n_rsu_id'].max()) + 1):
                bucket = per_rsu[per_rsu['n_rsu_id'] == n]['est_error']
                if len(bucket) > 0:
                    bucket_stats.append({
                        'n_rsu_id': n,
                        'count': int(len(bucket)),
                        'mean_error': float(bucket.mean()),
                        'std_error': float(bucket.std()),
                        'p95_error': float(bucket.quantile(0.95)) if len(bucket) >= 20 else None,
                    })
            
            # 与n_rsu_total对比
            corr_total, pval_total = stats.pearsonr(valid_df['est_error'], valid_df['n_rsu_total'])
            
            summary['per_rsu_analysis'] = {
                'corr_est_error_n_rsu_id': float(corr_per_rsu),
                'p_value_n_rsu_id': float(pval_per_rsu),
                'corr_est_error_n_rsu_total': float(corr_total),
                'p_value_n_rsu_total': float(pval_total),
                'bucket_stats': bucket_stats,
                'comparison': (
                    f"per-rsu相关性(r={corr_per_rsu:.3f})与total相关性(r={corr_total:.3f})比较"
                ),
            }
    
    # ==========================================
    # 基础统计
    # ==========================================
    summary['basic_stats'] = {
        'est_error_mean': float(valid_df['est_error'].mean()),
        'est_error_std': float(valid_df['est_error'].std()),
        'est_error_p95': float(valid_df['est_error'].quantile(0.95)),
        'n_rsu_total_mean': float(valid_df['n_rsu_total'].mean()),
        'n_rsu_total_max': int(valid_df['n_rsu_total'].max()),
        'q_before_mean': float(valid_df['q_before'].mean()),
        'cycles_mean': float(valid_df['cycles'].mean()),
    }
    
    return summary


def print_analysis(summary):
    """打印分析结果"""
    print("\n" + "="*80)
    print("REGRESSION EVIDENCE AUDIT")
    print("="*80)
    
    print(f"\n[Data Size]")
    print(f"  Total decision records: {summary.get('total_decision_records', 0)}")
    print(f"  Valid records: {summary.get('valid_records', 0)}")
    
    if 'basic_stats' in summary:
        bs = summary['basic_stats']
        print(f"\n[Basic Statistics]")
        print(f"  est_error: mean={bs['est_error_mean']:.4f}, std={bs['est_error_std']:.4f}, p95={bs['est_error_p95']:.4f}")
        print(f"  n_rsu_total: mean={bs['n_rsu_total_mean']:.2f}, max={bs['n_rsu_total_max']}")
    
    if 'regression' in summary and 'error' not in summary['regression']:
        reg = summary['regression']
        print(f"\n[1.1 Regression Analysis]")
        
        if 'simple' in reg:
            s = reg['simple']
            print(f"  Simple: n_rsu_coef={s['n_rsu_coef']:.4f}, p={s['n_rsu_pvalue']:.4f}, R²={s['r_squared']:.4f}")
        
        if 'controlled' in reg:
            c = reg['controlled']
            print(f"  Controlled: n_rsu_coef={c['n_rsu_coef']:.4f}, p={c['n_rsu_pvalue']:.4f}, R²={c['r_squared']:.4f}")
            print(f"    q_before_coef={c['q_before_coef']:.4f}, cycles_coef={c['cycles_coef']:.4f}")
        
        if 'fixed_effect' in reg:
            fe = reg['fixed_effect']
            print(f"  Fixed Effect: n_rsu_coef={fe['n_rsu_coef']:.4f}, p={fe['n_rsu_pvalue']:.4f}")
        
        if 'interpretation' in reg:
            print(f"\n  >>> {reg['interpretation']}")
    
    if 'error_reward_link' in summary:
        erl = summary['error_reward_link']
        print(f"\n[1.2 Error→Reward Link]")
        print(f"  corr(est_error, r_total): {erl['corr_est_error_r_total']:.4f}, p={erl['p_value']:.4f}")
        print(f"  mean(r_total | low_error): {erl['mean_r_total_low_error']:.4f}")
        print(f"  mean(r_total | high_error): {erl['mean_r_total_high_error']:.4f}")
        print(f"\n  >>> {erl['reward_bias']}")
    
    if 'per_rsu_analysis' in summary:
        pra = summary['per_rsu_analysis']
        print(f"\n[1.3 Per-RSU Analysis]")
        print(f"  corr(est_error, n_rsu_id): {pra['corr_est_error_n_rsu_id']:.4f}, p={pra['p_value_n_rsu_id']:.4f}")
        print(f"  corr(est_error, n_rsu_total): {pra['corr_est_error_n_rsu_total']:.4f}, p={pra['p_value_n_rsu_total']:.4f}")
        
        if 'bucket_stats' in pra:
            print(f"\n  [n_rsu_id Bucket Stats]")
            for bs in pra['bucket_stats'][:10]:
                p95_str = f", p95={bs['p95_error']:.4f}" if bs['p95_error'] else ""
                print(f"    n={bs['n_rsu_id']}: count={bs['count']}, mean={bs['mean_error']:.4f}{p95_str}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Regression Evidence Audit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--out', type=str, default='out/regression_audit')
    
    args = parser.parse_args()
    run_audit(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
