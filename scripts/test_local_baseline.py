#!/usr/bin/env python
"""
Local-only基线测试 (Deadline可达性验证)

目的：
1. 强制所有任务在本地执行（不卸载）
2. 统计在当前deadline设置下，本地执行的成功率
3. 如果Local-only也是0%，则证明deadline过紧
4. 建立deadline调整的baseline

用法：
    python scripts/test_local_baseline.py --episodes 20 --seed 42
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
    """
    Local-only策略：所有任务都在本地执行
    
    Returns:
        list: actions for all vehicles (all target=0)
    """
    actions = []
    for v in env.vehicles:
        actions.append({"target": 0, "power": 1.0})  # target=0表示Local
    return actions


def run_local_baseline(num_episodes=20, seed=42):
    """运行Local-only基线测试"""
    print("="*80)
    print("Local-only基线测试 (Deadline可达性验证)")
    print("="*80)
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print(f"Deadline设置: γ_min={Cfg.DEADLINE_TIGHTENING_MIN}, γ_max={Cfg.DEADLINE_TIGHTENING_MAX}")
    print(f"策略: 强制Local（无卸载）")
    print()
    
    np.random.seed(seed)
    env = VecOffloadingEnv()
    
    episode_results = []
    all_task_durations = []
    all_deadlines = []
    all_gammas = []
    
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}...", end='', flush=True)
        
        obs, info = env.reset()
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated):
            actions = local_only_policy(env)
            obs, rewards, done, truncated, info = env.step(actions)
            step_count += 1
        
        # 提取关键指标
        task_success_rate = info.get('task_success_rate', 0.0)
        deadline_miss_rate = info.get('deadline_miss_rate', 0.0)
        completed_tasks = info.get('completed_tasks_count', 0)
        total_tasks = info.get('episode_task_count', 0)
        
        # Miss Reason分解
        miss_deadline = info.get('miss_reason_deadline', 0)
        miss_overflow = info.get('miss_reason_overflow', 0)
        miss_illegal = info.get('miss_reason_illegal', 0)
        miss_unfinished = info.get('miss_reason_unfinished', 0)
        miss_truncated = info.get('miss_reason_truncated', 0)
        
        # 任务时延
        task_duration_mean = info.get('task_duration_mean', 0.0)
        deadline_mean = info.get('deadline_seconds_mean', 0.0)
        gamma_mean = info.get('deadline_gamma_mean', 0.0)
        
        if task_duration_mean > 0:
            all_task_durations.append(task_duration_mean)
        if deadline_mean > 0:
            all_deadlines.append(deadline_mean)
        if gamma_mean > 0:
            all_gammas.append(gamma_mean)
        
        result = {
            'episode': ep + 1,
            'success_rate': task_success_rate,
            'deadline_miss_rate': deadline_miss_rate,
            'completed': completed_tasks,
            'total': total_tasks,
            'miss_reasons': {
                'deadline': miss_deadline,
                'overflow': miss_overflow,
                'illegal': miss_illegal,
                'unfinished': miss_unfinished,
                'truncated': miss_truncated
            },
            'task_duration_mean': task_duration_mean,
            'deadline_mean': deadline_mean,
            'gamma_mean': gamma_mean
        }
        episode_results.append(result)
        
        status = "✅" if task_success_rate > 0 else "❌"
        print(f" {status} SR={task_success_rate:.1%}, DL_miss={deadline_miss_rate:.1%}, "
              f"Miss=[DL:{miss_deadline} OF:{miss_overflow} IL:{miss_illegal} UF:{miss_unfinished} TR:{miss_truncated}]")
    
    # 生成报告
    print("\n" + "="*80)
    print("Local-only基线测试报告")
    print("="*80)
    
    # 统计成功率
    success_rates = [r['success_rate'] for r in episode_results]
    deadline_miss_rates = [r['deadline_miss_rate'] for r in episode_results]
    
    print(f"\n【任务完成情况】")
    print(f"  平均成功率: {np.mean(success_rates):.1%}")
    print(f"  最高成功率: {np.max(success_rates):.1%}")
    print(f"  成功episode数: {sum(1 for r in success_rates if r > 0)}/{num_episodes}")
    print(f"  平均deadline miss率: {np.mean(deadline_miss_rates):.1%}")
    
    # Miss Reason汇总
    total_miss_reasons = defaultdict(int)
    for r in episode_results:
        for reason, count in r['miss_reasons'].items():
            total_miss_reasons[reason] += count
    
    print(f"\n【失败原因分解】")
    total_miss = sum(total_miss_reasons.values())
    if total_miss > 0:
        for reason in ['deadline', 'overflow', 'illegal', 'unfinished', 'truncated']:
            count = total_miss_reasons[reason]
            ratio = count / total_miss
            print(f"  {reason:12s}: {count:4d} ({ratio:6.1%})")
    else:
        print(f"  无失败任务（所有任务成功）")
    
    # 时延分析
    if all_task_durations:
        print(f"\n【时延分析】")
        print(f"  平均任务完成时间: {np.mean(all_task_durations):.3f}s")
        print(f"  P50任务完成时间:  {np.percentile(all_task_durations, 50):.3f}s")
        print(f"  P95任务完成时间:  {np.percentile(all_task_durations, 95):.3f}s")
        print(f"  平均deadline:     {np.mean(all_deadlines):.3f}s")
        print(f"  平均γ系数:        {np.mean(all_gammas):.3f}")
    
    # Deadline可达性判定
    print(f"\n{'='*80}")
    print("【Deadline可达性判定】")
    print(f"{'='*80}")
    
    avg_success_rate = np.mean(success_rates)
    
    if avg_success_rate < 0.05:  # <5%
        print(f"\n❌ **Deadline过紧** (Local-only成功率<5%)")
        print(f"\n  当前设置:")
        print(f"    γ_min = {Cfg.DEADLINE_TIGHTENING_MIN}")
        print(f"    γ_max = {Cfg.DEADLINE_TIGHTENING_MAX}")
        print(f"\n  推荐调整（Curriculum Learning）:")
        if avg_success_rate == 0:
            print(f"    γ_min = 1.5  # 从1.5×本地时间开始（放宽50%）")
            print(f"    γ_max = 2.0  # 最宽松2×本地时间")
        else:
            print(f"    γ_min = 1.2  # 从1.2×本地时间开始（放宽20%）")
            print(f"    γ_max = 1.5  # 最宽松1.5×本地时间")
        print(f"\n  修复后验证方法:")
        print(f"    python scripts/test_local_baseline.py --episodes 10")
        print(f"    # 目标: 成功率>60%")
        return False
    
    elif avg_success_rate < 0.30:  # 5-30%
        print(f"\n⚠️  **Deadline偏紧** (Local-only成功率5-30%)")
        print(f"\n  当前成功率足以开始训练，但可能导致：")
        print(f"    - 训练初期reward极低")
        print(f"    - 策略难以学到有效卸载")
        print(f"\n  建议:")
        print(f"    - 如训练目标是学会卸载：保持当前deadline")
        print(f"    - 如训练目标是稳定收敛：放宽至γ_min=1.2")
        return True
    
    else:  # >30%
        print(f"\n✅ **Deadline合理** (Local-only成功率>30%)")
        print(f"\n  当前deadline设置适合训练:")
        print(f"    - Local-only基线: {avg_success_rate:.1%}")
        print(f"    - 卸载有明确价值")
        print(f"    - 可以开始正式训练")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20, help="测试episode数（推荐20-50）")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    success = run_local_baseline(args.episodes, args.seed)
    
    if not success:
        print("\n⚠️  必须先修复deadline设置，再进行训练！")
        sys.exit(1)
    else:
        print("\n✅ Local-only基线测试完成！可以开始训练。")
        sys.exit(0)

