#!/usr/bin/env python
"""
Deadline扫描实验（分位数驱动标定）

目的：
1. 扫描不同的DEADLINE_TIGHTENING_MIN/MAX
2. 对每个设置运行Local-only策略
3. 输出SR、平均makespan、deadline miss率
4. 找到让Local-only SR≈60-80%的最优点

用法：
    python scripts/sweep_deadline.py --episodes-per-config 15
"""

import sys
import os
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def local_only_policy(env):
    """Local-only策略"""
    return [{"target": 0, "power": 1.0} for _ in env.vehicles]


def test_deadline_config(gamma_min, gamma_max, num_episodes=15, seed=42):
    """测试特定deadline配置"""
    # 临时修改config（仅在此进程有效）
    original_min = Cfg.DEADLINE_TIGHTENING_MIN
    original_max = Cfg.DEADLINE_TIGHTENING_MAX
    
    Cfg.DEADLINE_TIGHTENING_MIN = gamma_min
    Cfg.DEADLINE_TIGHTENING_MAX = gamma_max
    
    np.random.seed(seed)
    env = VecOffloadingEnv()
    
    results = []
    makespans = []  # 实际完工时间（墙上时钟时间）
    deadlines = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            actions = local_only_policy(env)
            obs, rewards, done, truncated, info = env.step(actions)
        
        # 提取指标
        success_rate = info.get('task_success_rate', 0.0)
        total_tasks = info.get('episode_task_count', 1)
        miss_deadline = info.get('miss_reason_deadline', 0)
        deadline_miss_rate = miss_deadline / max(total_tasks, 1)
        
        task_duration_mean = info.get('task_duration_mean', 0.0)
        deadline_mean = info.get('deadline_seconds_mean', 0.0)
        episode_time = info.get('episode_time_seconds', 10.0)  # 实际episode时长
        
        # makespan：如果有完成任务用task_duration，否则用episode_time作为上界
        makespan_estimate = task_duration_mean if task_duration_mean > 0 else episode_time
        
        results.append({
            'success_rate': success_rate,
            'deadline_miss_rate': deadline_miss_rate,
            'makespan': makespan_estimate,
            'deadline': deadline_mean,
            'total_tasks': total_tasks,
            'miss_deadline': miss_deadline
        })
        
        if makespan_estimate > 0:
            makespans.append(makespan_estimate)
        if deadline_mean > 0:
            deadlines.append(deadline_mean)
    
    # 恢复原始config
    Cfg.DEADLINE_TIGHTENING_MIN = original_min
    Cfg.DEADLINE_TIGHTENING_MAX = original_max
    
    # 统计
    avg_sr = np.mean([r['success_rate'] for r in results])
    avg_miss_rate = np.mean([r['deadline_miss_rate'] for r in results])
    
    makespan_p50 = np.percentile(makespans, 50) if makespans else 0
    makespan_p90 = np.percentile(makespans, 90) if makespans else 0
    makespan_mean = np.mean(makespans) if makespans else 0
    
    deadline_mean = np.mean(deadlines) if deadlines else 0
    
    return {
        'gamma_min': gamma_min,
        'gamma_max': gamma_max,
        'success_rate_mean': avg_sr,
        'deadline_miss_rate_mean': avg_miss_rate,
        'makespan_p50': makespan_p50,
        'makespan_p90': makespan_p90,
        'makespan_mean': makespan_mean,
        'deadline_mean': deadline_mean,
        'num_episodes': num_episodes
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-per-config', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("="*80)
    print("Deadline扫描实验（分位数驱动标定）")
    print("="*80)
    print(f"Episodes per config: {args.episodes_per_config}")
    print(f"Seed: {args.seed}")
    print()
    
    # 扫描配置：(gamma_min, gamma_max)
    configs = [
        (1.0, 1.2),   # 当前设置（baseline）
        (1.2, 1.4),
        (1.4, 1.6),
        (1.6, 1.8),
        (1.8, 2.0),
        (2.0, 2.2),
    ]
    
    sweep_results = []
    
    for gamma_min, gamma_max in configs:
        print(f"Testing γ∈[{gamma_min:.1f}, {gamma_max:.1f}]...", end='', flush=True)
        
        result = test_deadline_config(
            gamma_min, gamma_max,
            num_episodes=args.episodes_per_config,
            seed=args.seed
        )
        sweep_results.append(result)
        
        sr = result['success_rate_mean']
        miss = result['deadline_miss_rate_mean']
        makespan_p90 = result['makespan_p90']
        
        status = "✅" if 0.6 <= sr <= 0.8 else ("⚠️" if sr > 0 else "❌")
        print(f" {status} SR={sr:.1%}, Miss={miss:.1%}, P90_makespan={makespan_p90:.3f}s")
    
    # 输出结果表格
    print("\n" + "="*80)
    print("Deadline扫描结果汇总")
    print("="*80)
    print()
    print(f"{'γ_min':>6} {'γ_max':>6} {'SR':>8} {'Miss':>8} {'P50':>8} {'P90':>8} {'Mean':>8} {'DL':>8}")
    print("-"*80)
    
    for r in sweep_results:
        print(f"{r['gamma_min']:6.1f} {r['gamma_max']:6.1f} "
              f"{r['success_rate_mean']:7.1%} {r['deadline_miss_rate_mean']:7.1%} "
              f"{r['makespan_p50']:7.3f}s {r['makespan_p90']:7.3f}s "
              f"{r['makespan_mean']:7.3f}s {r['deadline_mean']:7.3f}s")
    
    # 推荐配置
    print("\n" + "="*80)
    print("推荐配置")
    print("="*80)
    
    # 找到SR在60-80%的配置
    target_configs = [r for r in sweep_results if 0.6 <= r['success_rate_mean'] <= 0.8]
    
    if target_configs:
        # 选择SR最接近70%的
        best = min(target_configs, key=lambda r: abs(r['success_rate_mean'] - 0.7))
        print(f"\n✅ 推荐warmup配置：")
        print(f"   DEADLINE_TIGHTENING_MIN = {best['gamma_min']}")
        print(f"   DEADLINE_TIGHTENING_MAX = {best['gamma_max']}")
        print(f"   预期Local-only SR: {best['success_rate_mean']:.1%}")
        print(f"   P90 makespan: {best['makespan_p90']:.3f}s")
    else:
        # 如果没有在目标范围，选SR最高的
        best = max(sweep_results, key=lambda r: r['success_rate_mean'])
        if best['success_rate_mean'] < 0.6:
            print(f"\n⚠️  所有配置SR均<60%，推荐进一步放宽：")
            print(f"   DEADLINE_TIGHTENING_MIN = {best['gamma_max']}")
            print(f"   DEADLINE_TIGHTENING_MAX = {best['gamma_max'] + 0.4}")
        else:
            print(f"\n✅ 推荐配置（SR略高）：")
            print(f"   DEADLINE_TIGHTENING_MIN = {best['gamma_min']}")
            print(f"   DEADLINE_TIGHTENING_MAX = {best['gamma_max']}")
            print(f"   预期Local-only SR: {best['success_rate_mean']:.1%}")
    
    # 保存结果
    output_file = 'logs/deadline_sweep_results.json'
    with open(output_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()

