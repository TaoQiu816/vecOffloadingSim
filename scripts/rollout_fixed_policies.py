#!/usr/bin/env python3
"""
固定策略对比测试脚本

对比 AlwaysLocal / AlwaysRSU / AlwaysV2V 三种固定策略，
用于判断"Local是否真的更优"还是"reward/机制偏差"。

输出指标（按episode统计）：
- success_rate: 任务成功率（完成且未超时）
- failed_rate: 任务失败率（deadline miss）
- mean_cft: 平均完成时间
- p95_cft: P95完成时间
- mean_return: 平均episode回报

Usage:
    python scripts/rollout_fixed_policies.py --seeds 42,43,44 --episodes 10
"""

import argparse
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg


class FixedPolicy:
    """固定策略基类"""

    def __init__(self, env):
        self.env = env
        self.num_vehicles = env.config.NUM_VEHICLES
        self.max_targets = env.config.MAX_TARGETS

    def get_action(self, obs):
        """返回动作列表，每个车辆一个动作"""
        raise NotImplementedError


class AlwaysLocalPolicy(FixedPolicy):
    """总是选择本地执行"""

    def get_action(self, obs):
        return [{'target': 0, 'power': 1.0} for _ in range(self.num_vehicles)]


class AlwaysRSUPolicy(FixedPolicy):
    """总是选择RSU卸载（如果可用）"""

    def get_action(self, obs):
        return [{'target': 1, 'power': 1.0} for _ in range(self.num_vehicles)]


class AlwaysV2VPolicy(FixedPolicy):
    """总是选择V2V卸载（选择第一个可用邻居）"""

    def get_action(self, obs):
        # target=2 表示第一个V2V邻居
        return [{'target': 2, 'power': 1.0} for _ in range(self.num_vehicles)]


def run_episode(env, policy, seed):
    """运行单个episode，返回统计信息"""
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    step_count = 0

    while True:
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # reward是列表，取平均
        if isinstance(reward, (list, np.ndarray)):
            total_reward += np.mean(reward)
        else:
            total_reward += reward

        step_count += 1

        if terminated or truncated:
            break

    # 收集episode统计
    metrics = info.get('episode_metrics', {})

    # 计算CFT（完成时间）
    cfts = []
    for v in env.vehicles:
        dag = v.task_dag
        if dag.completion_time is not None:
            cfts.append(dag.completion_time)
        elif dag.is_finished:
            # 如果没有记录completion_time，使用elapsed作为近似
            cfts.append(env.time - dag.start_time)

    return {
        'success_rate': metrics.get('task_success_rate', 0.0),
        'failed_rate': metrics.get('deadline_miss_rate', 0.0),
        'mean_cft': np.mean(cfts) if cfts else np.nan,
        'p95_cft': np.percentile(cfts, 95) if len(cfts) >= 2 else (cfts[0] if cfts else np.nan),
        'total_reward': total_reward,
        'steps': step_count,
        'p_target_raw': metrics.get('p_target_raw_local_frac', 0.0),
        'p_target_eff': metrics.get('p_target_eff_local_frac', 0.0),
        'fallback_rate': metrics.get('fallback_rate', 0.0),
    }


def run_policy_evaluation(policy_class, policy_name, seeds, episodes_per_seed):
    """评估单个策略"""
    results = []

    for seed in seeds:
        env = VecOffloadingEnv()
        policy = policy_class(env)

        for ep in range(episodes_per_seed):
            ep_seed = seed * 1000 + ep
            result = run_episode(env, policy, ep_seed)
            result['seed'] = seed
            result['episode'] = ep
            results.append(result)

    # 汇总统计
    success_rates = [r['success_rate'] for r in results]
    failed_rates = [r['failed_rate'] for r in results]
    cfts = [r['mean_cft'] for r in results if np.isfinite(r['mean_cft'])]
    rewards = [r['total_reward'] for r in results]
    fallback_rates = [r['fallback_rate'] for r in results]

    summary = {
        'policy': policy_name,
        'num_episodes': len(results),
        'success_rate_mean': np.mean(success_rates),
        'success_rate_std': np.std(success_rates),
        'failed_rate_mean': np.mean(failed_rates),
        'failed_rate_std': np.std(failed_rates),
        'cft_mean': np.mean(cfts) if cfts else np.nan,
        'cft_p95': np.percentile(cfts, 95) if len(cfts) >= 2 else np.nan,
        'return_mean': np.mean(rewards),
        'return_std': np.std(rewards),
        'fallback_rate_mean': np.mean(fallback_rates),
    }

    return summary, results


def main():
    parser = argparse.ArgumentParser(description='固定策略对比测试')
    parser.add_argument('--seeds', type=str, default='42,43,44',
                        help='随机种子列表，逗号分隔')
    parser.add_argument('--episodes', type=int, default=5,
                        help='每个种子运行的episode数')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    episodes_per_seed = args.episodes

    print("=" * 70)
    print("固定策略对比测试")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {episodes_per_seed}")
    print(f"Total episodes per policy: {len(seeds) * episodes_per_seed}")
    print()

    policies = [
        (AlwaysLocalPolicy, 'AlwaysLocal'),
        (AlwaysRSUPolicy, 'AlwaysRSU'),
        (AlwaysV2VPolicy, 'AlwaysV2V'),
    ]

    all_summaries = []
    all_results = []

    for policy_class, policy_name in policies:
        print(f"评估策略: {policy_name}...")
        summary, results = run_policy_evaluation(
            policy_class, policy_name, seeds, episodes_per_seed
        )
        all_summaries.append(summary)
        for r in results:
            r['policy'] = policy_name
        all_results.extend(results)
        print(f"  完成 {len(results)} episodes")

    # 打印汇总表格
    print()
    print("=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(f"{'Policy':<15} {'SuccessRate':<12} {'FailedRate':<12} {'CFT_mean':<10} {'CFT_p95':<10} {'Return':<12} {'Fallback':<10}")
    print("-" * 70)

    for s in all_summaries:
        print(f"{s['policy']:<15} "
              f"{s['success_rate_mean']:.2%}±{s['success_rate_std']:.2%} "
              f"{s['failed_rate_mean']:.2%}±{s['failed_rate_std']:.2%} "
              f"{s['cft_mean']:.3f}s    "
              f"{s['cft_p95']:.3f}s    "
              f"{s['return_mean']:.1f}±{s['return_std']:.1f} "
              f"{s['fallback_rate_mean']:.2%}")

    # 保存CSV
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n详细结果已保存到: {args.output}")

    # 分析结论
    print()
    print("=" * 70)
    print("分析结论")
    print("=" * 70)

    best_success = max(all_summaries, key=lambda x: x['success_rate_mean'])
    best_return = max(all_summaries, key=lambda x: x['return_mean'])

    print(f"最高成功率: {best_success['policy']} ({best_success['success_rate_mean']:.2%})")
    print(f"最高回报: {best_return['policy']} ({best_return['return_mean']:.1f})")

    # 判断是否存在偏差
    local_summary = next(s for s in all_summaries if s['policy'] == 'AlwaysLocal')
    rsu_summary = next(s for s in all_summaries if s['policy'] == 'AlwaysRSU')

    if local_summary['success_rate_mean'] < rsu_summary['success_rate_mean']:
        print("\n[发现] RSU策略成功率更高，但如果训练后Local占比高，可能存在reward/机制偏差")
    elif local_summary['success_rate_mean'] > rsu_summary['success_rate_mean']:
        print("\n[发现] Local策略确实成功率更高，偏向Local可能是合理的")
    else:
        print("\n[发现] Local和RSU成功率接近，需要进一步分析")


if __name__ == '__main__':
    main()
