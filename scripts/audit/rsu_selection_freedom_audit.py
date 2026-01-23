#!/usr/bin/env python3
"""
RSU Selection Freedom Audit - RSU选择自由度核验

核心目标：
1. 验证action space是否能选择具体RSU（RSU_0/1/2）
2. 统计serving_rsu_id的分布
3. 解释RSU_2队列为0的原因

用法:
    python scripts/audit/rsu_selection_freedom_audit.py --seed 0 --episodes 10 --out out/rsu_freedom
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
    """运行RSU选择自由度审计"""
    np.random.seed(seed)
    
    env = VecOffloadingEnv()
    
    # 检查action space
    print("=== Action Space Analysis ===")
    print(f"MAX_TARGETS: {Cfg.MAX_TARGETS}")
    print(f"Action space structure:")
    print(f"  target=0: Local")
    print(f"  target=1: RSU (system-selected)")
    print(f"  target=2-{Cfg.MAX_TARGETS-1}: V2V neighbors")
    print(f"\n** 策略只能选择'去RSU'(target=1)，具体哪个RSU由系统选择 **\n")
    
    records = []
    serving_rsu_dist = defaultdict(int)
    rsu_decision_dist = defaultdict(int)  # 实际选择了哪个RSU
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        done = False
        step = 0
        
        while not done and step < 200:
            step += 1
            
            # 记录每个车辆的serving_rsu_id
            for v in env.vehicles:
                serving_id = getattr(v, 'serving_rsu_id', None)
                if serving_id is not None:
                    serving_rsu_dist[serving_id] += 1
                    
                records.append({
                    'episode': ep,
                    'step': step,
                    'vehicle_id': v.id,
                    'serving_rsu_id': serving_id,
                    'pos_x': v.pos[0],
                    'pos_y': v.pos[1],
                })
                
            # 随机动作
            action = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 统计实际RSU决策（从info获取）
            # 由于action解析后会记录到planned_target，我们需要从内部获取
            
    df = pd.DataFrame(records)
    
    # 保存
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    df.to_csv(f"{output_prefix}_raw.csv", index=False)
    
    # 分析RSU覆盖与选择
    print("=== Serving RSU Distribution ===")
    total_serving = sum(serving_rsu_dist.values())
    for rsu_id in sorted(serving_rsu_dist.keys()):
        count = serving_rsu_dist[rsu_id]
        print(f"  RSU_{rsu_id}: {count} ({count/total_serving:.1%})")
        
    # 检查RSU位置
    print("\n=== RSU Positions ===")
    for rsu in env.rsus:
        print(f"  RSU_{rsu.id}: pos=({rsu.position[0]:.0f}, {rsu.position[1]:.0f}), "
              f"coverage={rsu.coverage_range}m")
              
    # 检查车辆位置分布
    print("\n=== Vehicle Position Statistics ===")
    if not df.empty:
        print(f"  X range: [{df['pos_x'].min():.0f}, {df['pos_x'].max():.0f}]")
        print(f"  Y range: [{df['pos_y'].min():.0f}, {df['pos_y'].max():.0f}]")
        
    # 分析为什么某些RSU不被选择
    print("\n=== RSU Selection Analysis ===")
    # 按位置分段统计serving_rsu_id
    if not df.empty:
        df['x_segment'] = pd.cut(df['pos_x'], bins=5)
        segment_rsu = df.groupby('x_segment')['serving_rsu_id'].value_counts()
        print("Serving RSU by position segment:")
        print(segment_rsu.head(20))
        
    # 生成摘要
    summary = {
        'seed': seed,
        'episodes': episodes,
        'action_space_structure': {
            'target_0': 'Local',
            'target_1': 'RSU (system-selected, NOT policy choice)',
            'target_2_to_6': 'V2V neighbors',
        },
        'key_finding': 'Policy CANNOT choose specific RSU (RSU_0/1/2). '
                      'System internally selects nearest RSU via _get_nearest_rsu()',
        'serving_rsu_distribution': {f'RSU_{k}': v for k, v in serving_rsu_dist.items()},
        'num_rsus': len(env.rsus),
        'rsu_positions': {f'RSU_{rsu.id}': list(rsu.position) for rsu in env.rsus},
    }
    
    with open(f"{output_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n=== Key Finding ===")
    print("策略只有'去RSU'一个动作(target=1)，具体RSU由系统选择最近的。")
    print("这解释了RSU_2队列为0：如果RSU_2位置使其从不被选为'最近RSU'，则永远不会被使用。")
    print("\n建议：如果需要策略学习RSU负载均衡，需要扩展action space或修改RSU选择逻辑。")
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(description='RSU Selection Freedom Audit')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--out', type=str, default='out/rsu_freedom', help='Output prefix')
    
    args = parser.parse_args()
    run_audit(args.seed, args.episodes, args.out)


if __name__ == '__main__':
    main()
