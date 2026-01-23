#!/usr/bin/env python3
"""
P0-1 训练A/B对照实验

A组: TIME_QUEUE_PENALTY_WEIGHT=0 (关闭并发拥塞惩罚)
B组: TIME_QUEUE_PENALTY_WEIGHT=1.5 (V6默认)

每组运行 seed=0,42; episodes=500
"""

import argparse
import subprocess
import os
import sys
from datetime import datetime

def run_experiment(weight: float, seed: int, episodes: int, run_name: str):
    """运行单次实验"""
    
    # 设置环境变量覆盖config
    env = os.environ.copy()
    env['TIME_QUEUE_PENALTY_WEIGHT'] = str(weight)
    
    cmd = [
        sys.executable, 'train.py',
        '--seed', str(seed),
        '--max-episodes', str(episodes),
        '--run-name', run_name,
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: {run_name}")
    print(f"  TIME_QUEUE_PENALTY_WEIGHT = {weight}")
    print(f"  seed = {seed}")
    print(f"  episodes = {episodes}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--seeds', type=str, default='0,42')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    experiments = [
        # A组: 关闭拥塞惩罚
        {'weight': 0.0, 'group': 'A_no_penalty'},
        # B组: 默认惩罚
        {'weight': 1.5, 'group': 'B_default'},
    ]
    
    print("="*60)
    print("P0-1 训练A/B对照实验")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {seeds}")
    print(f"Experiments: {len(experiments) * len(seeds)}")
    print()
    
    if args.dry_run:
        print("[Dry run] 以下是将要运行的实验:")
        for exp in experiments:
            for seed in seeds:
                run_name = f"ablation_{exp['group']}_seed{seed}_{timestamp}"
                print(f"  - {run_name}: weight={exp['weight']}, seed={seed}")
        return
    
    # 实际运行
    results = []
    for exp in experiments:
        for seed in seeds:
            run_name = f"ablation_{exp['group']}_seed{seed}_{timestamp}"
            ret = run_experiment(exp['weight'], seed, args.episodes, run_name)
            results.append({
                'run_name': run_name,
                'weight': exp['weight'],
                'seed': seed,
                'return_code': ret
            })
    
    # 总结
    print("\n" + "="*60)
    print("实验完成汇总")
    print("="*60)
    for r in results:
        status = '✓' if r['return_code'] == 0 else '✗'
        print(f"{status} {r['run_name']}: return_code={r['return_code']}")
    
    print(f"\n结果保存在 runs/ 目录下")
    print(f"使用 tensorboard --logdir runs/ 查看训练曲线")

if __name__ == '__main__':
    main()
