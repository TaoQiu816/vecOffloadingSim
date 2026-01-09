#!/usr/bin/env python3
"""
Phase 1: 通信速率验证

验证V2I/V2V实际速率是否符合带宽分时共享模型预期。
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.vec_offloading_env import VecOffloadingEnv


def random_action():
    """生成随机动作"""
    return {
        'target': np.random.randint(0, 13),  # 0-12 (Local, RSU, V2V×11)
        'power': 1.0
    }


def main():
    print("=" * 70)
    print("Phase 1: 通信速率验证")
    print("=" * 70)

    env = VecOffloadingEnv()
    rate_v2i_samples = []
    rate_v2v_samples = []
    v2i_user_counts = []

    num_episodes = 10
    steps_per_episode = 100

    print(f"\n采样配置: {num_episodes} episodes × {steps_per_episode} steps")

    for ep in range(num_episodes):
        env.reset(seed=42 + ep)

        for step in range(steps_per_episode):
            # 触发rate snapshot计算
            env._capture_rate_snapshot()

            # 采样V2I速率
            v2i_users = 0
            for v in env.vehicles:
                for rsu in env.rsus:
                    rate = env._get_rate_from_snapshot(v.id, rsu.id, 'v2i')
                    if rate > 0:
                        rate_v2i_samples.append(rate / 1e6)  # Mbps
                        v2i_users += 1

            if v2i_users > 0:
                v2i_user_counts.append(v2i_users)

            # 采样V2V速率
            for v in env.vehicles:
                neighbors = v.get_neighbors_within_range(env.vehicles)
                for n in neighbors[:3]:  # 前3个邻居
                    rate = env._get_rate_from_snapshot(v.id, n.id, 'v2v')
                    if rate > 0:
                        rate_v2v_samples.append(rate / 1e6)  # Mbps

            # 推进环境
            actions = [random_action() for _ in range(20)]
            env.step(actions)

        print(f"  Episode {ep+1}/{num_episodes} 完成")

    # 统计分析
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)

    df_v2i = pd.DataFrame({
        'rate_mbps': rate_v2i_samples
    })

    df_v2v = pd.DataFrame({
        'rate_mbps': rate_v2v_samples
    })

    print("\n【V2I速率统计】")
    print(f"  样本数:     {len(rate_v2i_samples)}")
    print(f"  均值:       {df_v2i['rate_mbps'].mean():.2f} Mbps")
    print(f"  中位数(p50): {df_v2i['rate_mbps'].quantile(0.50):.2f} Mbps")
    print(f"  p25:        {df_v2i['rate_mbps'].quantile(0.25):.2f} Mbps")
    print(f"  p75:        {df_v2i['rate_mbps'].quantile(0.75):.2f} Mbps")
    print(f"  p95:        {df_v2i['rate_mbps'].quantile(0.95):.2f} Mbps")
    print(f"  最小值:     {df_v2i['rate_mbps'].min():.2f} Mbps")
    print(f"  最大值:     {df_v2i['rate_mbps'].max():.2f} Mbps")

    print("\n【V2V速率统计】")
    print(f"  样本数:     {len(rate_v2v_samples)}")
    print(f"  均值:       {df_v2v['rate_mbps'].mean():.2f} Mbps")
    print(f"  中位数(p50): {df_v2v['rate_mbps'].quantile(0.50):.2f} Mbps")
    print(f"  p25:        {df_v2v['rate_mbps'].quantile(0.25):.2f} Mbps")
    print(f"  p75:        {df_v2v['rate_mbps'].quantile(0.75):.2f} Mbps")
    print(f"  p95:        {df_v2v['rate_mbps'].quantile(0.95):.2f} Mbps")
    print(f"  最小值:     {df_v2v['rate_mbps'].min():.2f} Mbps")
    print(f"  最大值:     {df_v2v['rate_mbps'].max():.2f} Mbps")

    print("\n【V2I并发用户数统计】")
    print(f"  平均用户数: {np.mean(v2i_user_counts):.1f}")
    print(f"  p50:        {np.percentile(v2i_user_counts, 50):.1f}")
    print(f"  p95:        {np.percentile(v2i_user_counts, 95):.1f}")
    print(f"  最大值:     {np.max(v2i_user_counts):.0f}")

    # 判定结果
    print("\n" + "=" * 70)
    print("判定结果")
    print("=" * 70)

    v2i_p50 = df_v2i['rate_mbps'].quantile(0.50)
    v2v_p50 = df_v2v['rate_mbps'].quantile(0.50)

    print("\n文档预期:")
    print("  V2I p50: 1.0-3.0 Mbps (20MHz分时共享)")
    print("  V2V p50: ≈10 Mbps (10MHz独占，干扰降级)")

    print("\n实际测量:")
    print(f"  V2I p50: {v2i_p50:.2f} Mbps")
    print(f"  V2V p50: {v2v_p50:.2f} Mbps")

    v2i_pass = 1.0 <= v2i_p50 <= 3.0
    v2v_pass = 8.0 <= v2v_p50 <= 15.0

    print("\n判定:")
    print(f"  V2I: {'✓ 通过' if v2i_pass else '✗ 不通过'}")
    print(f"  V2V: {'✓ 通过' if v2v_pass else '✗ 通过'}")

    overall_pass = v2i_pass and v2v_pass
    print(f"\n{'='*70}")
    print(f"Phase 1 最终判定: {'✓ 通过' if overall_pass else '✗ 不通过'}")
    print(f"{'='*70}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
