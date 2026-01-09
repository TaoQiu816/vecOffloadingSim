#!/usr/bin/env python3
"""
Phase 3: 固定策略对比

对比 AlwaysLocal, GreedyMinTime, AlwaysRSU 的性能差距。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.vec_offloading_env import VecOffloadingEnv


def policy_always_local(env):
    """策略: 所有任务本地执行"""
    return [{'target': 0, 'power': 1.0} for _ in range(env.config.NUM_VEHICLES)]


def policy_always_rsu(env):
    """策略: 所有任务卸载RSU"""
    return [{'target': 1, 'power': 1.0} for _ in range(env.config.NUM_VEHICLES)]


def policy_greedy_mintime(env):
    """策略: 每步选择预计最小执行时间的target"""
    actions = []
    for v in env.vehicles:
        if not v.task_dag:
            actions.append({'target': 0, 'power': 1.0})
            continue

        subtask_idx = v.task_dag.get_top_priority_task()
        if subtask_idx is None:
            actions.append({'target': 0, 'power': 1.0})
            continue

        cycles = v.task_dag.comp[subtask_idx]

        # 枚举所有target，选择t_actual最小的
        t_estimates = []
        for target in range(env.config.MAX_TARGETS):
            try:
                t_actual, _ = env._estimate_t_actual(v, subtask_idx, target, cycles, 1.0)
                t_estimates.append((target, t_actual))
            except:
                # 某些target可能无效 (如无邻居)
                pass

        if t_estimates:
            best_target = min(t_estimates, key=lambda x: x[1])[0]
            actions.append({'target': best_target, 'power': 1.0})
        else:
            actions.append({'target': 0, 'power': 1.0})

    return actions


def run_policy(policy_name, policy_fn, num_episodes=100):
    """运行单个策略并收集统计"""
    print(f"\n运行策略: {policy_name}")
    env = VecOffloadingEnv()
    ep_results = []

    for ep in tqdm(range(num_episodes), desc=f"{policy_name}"):
        env.reset(seed=ep)
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            actions = policy_fn(env)
            obs, rewards, done, truncated, info = env.step(actions)
            step_count += 1

            if step_count > 500:  # 防止死循环
                break

        # 提取episode统计
        ep_results.append({
            'V_SR': info.get('vehicle_success_rate', 0.0),
            'S_SR': info.get('subtask_success_rate', 0.0),
            'mean_CFT': info.get('mean_completion_time', np.nan),
            'steps': step_count,
        })

    return pd.DataFrame(ep_results)


def main():
    print("=" * 70)
    print("Phase 3: 固定策略对比")
    print("=" * 70)

    num_episodes = 100

    print(f"\n配置: 每个策略运行 {num_episodes} episodes")

    # 运行各策略
    results = {}

    results['AlwaysLocal'] = run_policy(
        'AlwaysLocal',
        policy_always_local,
        num_episodes
    )

    results['GreedyMinTime'] = run_policy(
        'GreedyMinTime',
        policy_greedy_mintime,
        num_episodes
    )

    results['AlwaysRSU'] = run_policy(
        'AlwaysRSU',
        policy_always_rsu,
        num_episodes
    )

    # 统计汇总
    print("\n" + "=" * 70)
    print("性能对比表")
    print("=" * 70)

    summary = []
    for policy_name, df in results.items():
        summary.append({
            'Policy': policy_name,
            'V_SR (%)': df['V_SR'].mean() * 100,
            'V_SR_std': df['V_SR'].std() * 100,
            'S_SR (%)': df['S_SR'].mean() * 100,
            'mean_CFT (s)': df['mean_CFT'].mean(),
            'CFT_std': df['mean_CFT'].std(),
        })

    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))

    # 相对AlwaysLocal的改善
    print("\n" + "=" * 70)
    print("相对 AlwaysLocal 的改善")
    print("=" * 70)

    baseline_vsr = summary_df[summary_df['Policy'] == 'AlwaysLocal']['V_SR (%)'].values[0]
    baseline_cft = summary_df[summary_df['Policy'] == 'AlwaysLocal']['mean_CFT (s)'].values[0]

    print("\n【GreedyMinTime vs AlwaysLocal】")
    greedy_vsr = summary_df[summary_df['Policy'] == 'GreedyMinTime']['V_SR (%)'].values[0]
    greedy_cft = summary_df[summary_df['Policy'] == 'GreedyMinTime']['mean_CFT (s)'].values[0]

    delta_vsr = greedy_vsr - baseline_vsr
    delta_cft_pct = (greedy_cft - baseline_cft) / baseline_cft * 100

    print(f"  ΔV_SR:  {delta_vsr:+.1f}%")
    print(f"  ΔCFT:   {delta_cft_pct:+.1f}%")

    print("\n【AlwaysRSU vs AlwaysLocal】")
    rsu_vsr = summary_df[summary_df['Policy'] == 'AlwaysRSU']['V_SR (%)'].values[0]
    rsu_cft = summary_df[summary_df['Policy'] == 'AlwaysRSU']['mean_CFT (s)'].values[0]

    delta_rsu_vsr = rsu_vsr - baseline_vsr
    delta_rsu_cft_pct = (rsu_cft - baseline_cft) / baseline_cft * 100

    print(f"  ΔV_SR:  {delta_rsu_vsr:+.1f}%")
    print(f"  ΔCFT:   {delta_rsu_cft_pct:+.1f}%")

    # 判定结果
    print("\n" + "=" * 70)
    print("判定结果")
    print("=" * 70)

    print("\n标准: ΔV_SR >= 10% OR ΔCFT <= -15%")

    criterion_vsr = delta_vsr >= 10.0
    criterion_cft = delta_cft_pct <= -15.0

    print(f"  ΔV_SR = {delta_vsr:+.1f}%  {'✓' if criterion_vsr else '✗'} (>= 10%)")
    print(f"  ΔCFT  = {delta_cft_pct:+.1f}%  {'✓' if criterion_cft else '✗'} (<= -15%)")

    overall_pass = criterion_vsr or criterion_cft

    print(f"\n{'='*70}")
    print(f"Phase 3 最终判定: {'✓ 通过' if overall_pass else '✗ 不通过'}")
    print(f"{'='*70}")

    # 保存详细数据
    output_dir = Path(__file__).parent.parent / "docs"
    for policy_name, df in results.items():
        output_file = output_dir / f"phase3_{policy_name.lower()}_episodes.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{policy_name} 详细数据已保存至: {output_file}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
