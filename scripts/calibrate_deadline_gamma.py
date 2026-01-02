#!/usr/bin/env python
"""
Deadline Gamma标定脚本（基于新基准：max(CP, W)/f）

目标：找到让Local-only SR≈60-80%的γ区间

扫描配置：γ ∈ [1.2, 1.4, 1.6, 1.8]
每个配置运行50 episodes
输出：成功率、平均makespan、deadline miss率
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def local_only_policy(env):
    """Local-only策略"""
    return [{"target": 0, "power": 1.0} for _ in env.vehicles]


def test_gamma_config(gamma_min, gamma_max, num_episodes=50, seed=42):
    """测试特定gamma配置"""
    original_min = Cfg.DEADLINE_TIGHTENING_MIN
    original_max = Cfg.DEADLINE_TIGHTENING_MAX
    
    Cfg.DEADLINE_TIGHTENING_MIN = gamma_min
    Cfg.DEADLINE_TIGHTENING_MAX = gamma_max
    
    np.random.seed(seed)
    env = VecOffloadingEnv()
    
    results = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            actions = local_only_policy(env)
            obs, rewards, done, truncated, info = env.step(actions)
        
        # 提取关键指标
        success_rate = info.get('task_success_rate', 0.0)
        total_tasks = info.get('episode_task_count', 1)
        miss_deadline = info.get('miss_reason_deadline', 0)
        miss_truncated = info.get('miss_reason_truncated', 0)
        task_duration_mean = info.get('task_duration_mean', 0.0)
        deadline_mean = info.get('deadline_seconds_mean', 0.0)
        
        results.append({
            'success_rate': success_rate,
            'deadline_miss_rate': miss_deadline / max(total_tasks, 1),
            'truncated_rate': miss_truncated / max(total_tasks, 1),
            'makespan': task_duration_mean,
            'deadline': deadline_mean,
            'total_tasks': total_tasks,
            'miss_deadline': miss_deadline,
            'miss_truncated': miss_truncated
        })
    
    # 恢复原始config
    Cfg.DEADLINE_TIGHTENING_MIN = original_min
    Cfg.DEADLINE_TIGHTENING_MAX = original_max
    
    # 统计
    avg_sr = np.mean([r['success_rate'] for r in results])
    avg_miss_dl = np.mean([r['deadline_miss_rate'] for r in results])
    avg_miss_trunc = np.mean([r['truncated_rate'] for r in results])
    
    makespans = [r['makespan'] for r in results if r['makespan'] > 0]
    deadlines = [r['deadline'] for r in results if r['deadline'] > 0]
    
    makespan_p50 = np.percentile(makespans, 50) if makespans else 0
    makespan_p90 = np.percentile(makespans, 90) if makespans else 0
    makespan_mean = np.mean(makespans) if makespans else 0
    deadline_mean = np.mean(deadlines) if deadlines else 0
    
    return {
        'gamma_min': gamma_min,
        'gamma_max': gamma_max,
        'success_rate_mean': avg_sr,
        'deadline_miss_rate_mean': avg_miss_dl,
        'truncated_rate_mean': avg_miss_trunc,
        'makespan_p50': makespan_p50,
        'makespan_p90': makespan_p90,
        'makespan_mean': makespan_mean,
        'deadline_mean': deadline_mean,
        'num_episodes': num_episodes
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-per-config', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("="*80)
    print("Deadline Gamma标定（新基准：max(CP, W)/f）")
    print("="*80)
    print(f"Episodes per config: {args.episodes_per_config}")
    print(f"Seed: {args.seed}")
    print()
    
    # 扫描配置
    configs = [
        (1.2, 1.2),  # 固定值测试
        (1.4, 1.4),
        (1.6, 1.6),
        (1.8, 1.8),
    ]
    
    sweep_results = []
    
    for gamma_min, gamma_max in configs:
        gamma_val = gamma_min  # 固定值
        print(f"Testing γ={gamma_val:.1f}...", end='', flush=True)
        
        result = test_gamma_config(
            gamma_min, gamma_max,
            num_episodes=args.episodes_per_config,
            seed=args.seed
        )
        sweep_results.append(result)
        
        sr = result['success_rate_mean']
        miss_dl = result['deadline_miss_rate_mean']
        miss_trunc = result['truncated_rate_mean']
        makespan_p90 = result['makespan_p90']
        
        status = "✅" if 0.6 <= sr <= 0.8 else ("⚠️" if sr > 0.2 else "❌")
        print(f" {status} SR={sr:.1%}, Miss_DL={miss_dl:.1%}, Miss_Trunc={miss_trunc:.1%}, P90={makespan_p90:.3f}s")
    
    # 输出结果表格
    print("\n" + "="*80)
    print("Gamma标定结果汇总")
    print("="*80)
    print()
    print(f"{'γ':>6} {'SR':>8} {'Miss_DL':>10} {'Miss_Trunc':>12} {'P50':>8} {'P90':>8} {'Mean':>8} {'DL':>8}")
    print("-"*80)
    
    for r in sweep_results:
        gamma_val = r['gamma_min']
        print(f"{gamma_val:6.1f} "
              f"{r['success_rate_mean']:7.1%} "
              f"{r['deadline_miss_rate_mean']:9.1%} "
              f"{r['truncated_rate_mean']:11.1%} "
              f"{r['makespan_p50']:7.3f}s "
              f"{r['makespan_p90']:7.3f}s "
              f"{r['makespan_mean']:7.3f}s "
              f"{r['deadline_mean']:7.3f}s")
    
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
            print(f"\n⚠️  所有配置SR均<60%，建议进一步放宽：")
            print(f"   DEADLINE_TIGHTENING_MIN = {best['gamma_min'] + 0.2}")
            print(f"   DEADLINE_TIGHTENING_MAX = {best['gamma_max'] + 0.2}")
        elif best['success_rate_mean'] > 0.8:
            print(f"\n✅ 推荐配置（SR略高但可用）：")
            print(f"   DEADLINE_TIGHTENING_MIN = {best['gamma_min']}")
            print(f"   DEADLINE_TIGHTENING_MAX = {best['gamma_max']}")
            print(f"   预期Local-only SR: {best['success_rate_mean']:.1%}")
    
    # 保存结果
    output_file = 'logs/gamma_calibration_results.json'
    with open(output_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    main()

