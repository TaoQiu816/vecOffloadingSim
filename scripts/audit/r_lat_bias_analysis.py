#!/usr/bin/env python3
"""
r_lat Bias Analysis - 时延奖励偏置机制证明

核心目标：
1. 证明t_alt的来源通常是RSU（t_alt==t_R的占比）
2. 按动作类型统计r_lat的mean/p50/p95
3. 证明相对优势设计导致RSU系统性正奖励，Local/V2V系统性负奖励

用法:
    python scripts/audit/r_lat_bias_analysis.py --seed 0 --episodes 20 --out out/r_lat_analysis
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


def run_analysis(seed, episodes, output_prefix):
    """运行r_lat偏置分析"""
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    lat_records = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 随机动作
            action = env.action_space.sample()
            
            # 执行step
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 从_reward_stats获取详细数据
            if hasattr(env, '_reward_stats') and env._reward_stats is not None:
                metrics = getattr(env._reward_stats, 'metrics', {})
                
                # 获取分类型统计
                for action_type in ['Local', 'RSU', 'V2V']:
                    key_prefix = f'{action_type.lower()}_'
                    r_lat = metrics.get(f'{key_prefix}r_lat_mean', None)
                    t_L = metrics.get(f'{key_prefix}t_L_mean', None)
                    t_R = metrics.get(f'{key_prefix}t_R_mean', None)
                    t_V = metrics.get(f'{key_prefix}t_V_mean', None)
                    t_a = metrics.get(f'{key_prefix}t_a_mean', None)
                    t_alt = metrics.get(f'{key_prefix}t_alt_mean', None)
                    A_t = metrics.get(f'{key_prefix}A_t_mean', None)
                    count = metrics.get(f'{key_prefix}count', 0)
                    
                    if count > 0:
                        lat_records.append({
                            'episode': ep,
                            'step': step,
                            'action_type': action_type,
                            'count': count,
                            'r_lat': r_lat,
                            't_L': t_L,
                            't_R': t_R,
                            't_V': t_V,
                            't_a': t_a,
                            't_alt': t_alt,
                            'A_t': A_t,
                        })
                        
    # 转换为DataFrame
    df = pd.DataFrame(lat_records)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    df.to_csv(f"{output_prefix}_raw.csv", index=False)
    
    # 分析
    summary = analyze_r_lat(df, seed, episodes)
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print_analysis(summary)
    
    return df, summary


def analyze_r_lat(df, seed, episodes):
    """分析r_lat偏置"""
    summary = {
        'seed': seed,
        'episodes': episodes,
        'policy': 'random',
        'config': {
            'T_REF': Cfg.T_REF,
            'LAT_ALPHA': Cfg.LAT_ALPHA,
            'F_RSU': Cfg.F_RSU,
            'MIN_VEHICLE_CPU_FREQ': Cfg.MIN_VEHICLE_CPU_FREQ,
            'MAX_VEHICLE_CPU_FREQ': Cfg.MAX_VEHICLE_CPU_FREQ,
        },
    }
    
    if df.empty:
        summary['error'] = 'No data collected'
        return summary
        
    # 按动作类型统计
    summary['by_action_type'] = {}
    
    for action_type in ['Local', 'RSU', 'V2V']:
        type_df = df[df['action_type'] == action_type]
        
        if len(type_df) == 0:
            continue
            
        # 统计r_lat
        r_lat_vals = type_df['r_lat'].dropna()
        if len(r_lat_vals) > 0:
            summary['by_action_type'][action_type] = {
                'count': len(type_df),
                'r_lat_mean': float(r_lat_vals.mean()),
                'r_lat_std': float(r_lat_vals.std()),
                'r_lat_p50': float(r_lat_vals.quantile(0.5)),
                'r_lat_p95': float(r_lat_vals.quantile(0.95)),
                'r_lat_min': float(r_lat_vals.min()),
                'r_lat_max': float(r_lat_vals.max()),
            }
            
            # 时延估计
            t_L_vals = type_df['t_L'].dropna()
            t_R_vals = type_df['t_R'].dropna()
            t_V_vals = type_df['t_V'].dropna()
            t_a_vals = type_df['t_a'].dropna()
            t_alt_vals = type_df['t_alt'].dropna()
            A_t_vals = type_df['A_t'].dropna()
            
            if len(t_L_vals) > 0:
                summary['by_action_type'][action_type]['t_L_mean'] = float(t_L_vals.mean())
            if len(t_R_vals) > 0:
                summary['by_action_type'][action_type]['t_R_mean'] = float(t_R_vals.mean())
            if len(t_V_vals) > 0:
                summary['by_action_type'][action_type]['t_V_mean'] = float(t_V_vals.mean())
            if len(t_a_vals) > 0:
                summary['by_action_type'][action_type]['t_a_mean'] = float(t_a_vals.mean())
            if len(t_alt_vals) > 0:
                summary['by_action_type'][action_type]['t_alt_mean'] = float(t_alt_vals.mean())
            if len(A_t_vals) > 0:
                summary['by_action_type'][action_type]['A_t_mean'] = float(A_t_vals.mean())
                
    # 计算t_alt来源统计
    # t_alt = min(t_L, t_R, t_V) 中哪个最常是最小值
    t_alt_source = {'t_R': 0, 't_L': 0, 't_V': 0, 'unknown': 0}
    
    for _, row in df.iterrows():
        t_L = row.get('t_L')
        t_R = row.get('t_R')
        t_V = row.get('t_V')
        t_alt = row.get('t_alt')
        
        if pd.isna(t_alt):
            continue
            
        # 判断t_alt最接近哪个
        candidates = []
        if not pd.isna(t_L):
            candidates.append(('t_L', t_L))
        if not pd.isna(t_R):
            candidates.append(('t_R', t_R))
        if not pd.isna(t_V):
            candidates.append(('t_V', t_V))
            
        if candidates:
            # 找最接近t_alt的
            closest = min(candidates, key=lambda x: abs(x[1] - t_alt))
            if abs(closest[1] - t_alt) < 0.001:
                t_alt_source[closest[0]] += 1
            else:
                t_alt_source['unknown'] += 1
                
    total = sum(t_alt_source.values())
    if total > 0:
        summary['t_alt_source_distribution'] = {
            k: {'count': v, 'ratio': v / total}
            for k, v in t_alt_source.items()
        }
        
    # 证明偏置机制
    summary['bias_proof'] = {}
    
    # 检查：是否t_R << t_L
    all_t_R = df['t_R'].dropna()
    all_t_L = df['t_L'].dropna()
    
    if len(all_t_R) > 0 and len(all_t_L) > 0:
        ratio = all_t_R.mean() / all_t_L.mean() if all_t_L.mean() > 0 else float('inf')
        summary['bias_proof']['t_R_over_t_L_ratio'] = float(ratio)
        summary['bias_proof']['t_R_mean'] = float(all_t_R.mean())
        summary['bias_proof']['t_L_mean'] = float(all_t_L.mean())
        
    # 检查：Local选择时A_t是否系统性为负
    local_df = df[df['action_type'] == 'Local']
    if len(local_df) > 0:
        A_t_local = local_df['A_t'].dropna()
        if len(A_t_local) > 0:
            neg_ratio = (A_t_local < 0).mean()
            summary['bias_proof']['A_t_negative_ratio_when_Local'] = float(neg_ratio)
            summary['bias_proof']['A_t_mean_when_Local'] = float(A_t_local.mean())
            
    # 检查：RSU选择时A_t是否系统性为正或接近0
    rsu_df = df[df['action_type'] == 'RSU']
    if len(rsu_df) > 0:
        A_t_rsu = rsu_df['A_t'].dropna()
        if len(A_t_rsu) > 0:
            pos_ratio = (A_t_rsu > 0).mean()
            summary['bias_proof']['A_t_positive_ratio_when_RSU'] = float(pos_ratio)
            summary['bias_proof']['A_t_mean_when_RSU'] = float(A_t_rsu.mean())
            
    return summary


def print_analysis(summary):
    """打印分析结果"""
    print("\n" + "="*80)
    print("r_lat BIAS ANALYSIS")
    print("="*80)
    
    print(f"\n[Config]")
    if 'config' in summary:
        for k, v in summary['config'].items():
            print(f"  {k}: {v}")
            
    print(f"\n[By Action Type]")
    if 'by_action_type' in summary:
        for action_type, stats in summary['by_action_type'].items():
            print(f"\n  {action_type} (n={stats['count']}):")
            print(f"    r_lat: mean={stats['r_lat_mean']:.4f}, "
                  f"p50={stats['r_lat_p50']:.4f}, p95={stats['r_lat_p95']:.4f}")
            if 't_a_mean' in stats and 't_alt_mean' in stats:
                print(f"    t_a={stats.get('t_a_mean', 0):.4f}s, "
                      f"t_alt={stats.get('t_alt_mean', 0):.4f}s")
            if 'A_t_mean' in stats:
                print(f"    A_t={stats['A_t_mean']:.4f}")
                
    print(f"\n[t_alt Source Distribution]")
    if 't_alt_source_distribution' in summary:
        for source, data in summary['t_alt_source_distribution'].items():
            print(f"  {source}: {data['ratio']:.2%} (n={data['count']})")
            
    print(f"\n[Bias Proof]")
    if 'bias_proof' in summary:
        bp = summary['bias_proof']
        if 't_R_over_t_L_ratio' in bp:
            print(f"  t_R/t_L ratio: {bp['t_R_over_t_L_ratio']:.2f}")
            print(f"    (t_R={bp['t_R_mean']:.4f}s, t_L={bp['t_L_mean']:.4f}s)")
        if 'A_t_negative_ratio_when_Local' in bp:
            print(f"  When Local: A_t<0 in {bp['A_t_negative_ratio_when_Local']:.1%} cases")
            print(f"    A_t mean = {bp['A_t_mean_when_Local']:.4f}")
        if 'A_t_positive_ratio_when_RSU' in bp:
            print(f"  When RSU: A_t>0 in {bp['A_t_positive_ratio_when_RSU']:.1%} cases")
            print(f"    A_t mean = {bp['A_t_mean_when_RSU']:.4f}")
            
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='r_lat Bias Analysis')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/r_lat_analysis', help='Output prefix')
    
    args = parser.parse_args()
    run_analysis(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
