#!/usr/bin/env python3
"""
Phase 2: 时延分解验证

验证 t_local, t_rsu, t_v2v 的分解 (tx/queue/comp) 与卸载优势概率。
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
        'target': np.random.randint(0, 13),
        'power': 1.0
    }


def main():
    print("=" * 70)
    print("Phase 2: 时延分解验证")
    print("=" * 70)

    env = VecOffloadingEnv()
    time_breakdown = []

    num_episodes = 10
    steps_per_episode = 20

    print(f"\n采样配置: {num_episodes} episodes × {steps_per_episode} steps")

    for ep in range(num_episodes):
        env.reset(seed=42 + ep)

        for step in range(steps_per_episode):
            # 触发rate snapshot
            env._capture_rate_snapshot()

            for v in env.vehicles:
                if not v.task_dag:
                    continue

                subtask_idx = v.task_dag.get_top_priority_task()
                if subtask_idx is None:
                    continue

                cycles = v.task_dag.comp[subtask_idx]
                data = v.task_dag.data[subtask_idx]

                # Local执行时间
                t_local, _ = env._estimate_t_actual(v, subtask_idx, 0, cycles, 1.0)
                queue_local = env._get_vehicle_queue_wait_time(v.id)
                comp_local = cycles / v.cpu_freq

                # RSU执行时间
                rsu_id = 0
                t_rsu, t_tx_rsu = env._estimate_t_actual(v, subtask_idx, 1, cycles, 1.0)
                wait_rsu = env._get_rsu_queue_wait_time(rsu_id)
                comp_rsu = cycles / env.rsus[rsu_id].cpu_freq

                # V2V执行时间 (选最佳邻居)
                neighbors = v.get_neighbors_within_range(env.vehicles)
                if neighbors:
                    best_n = max(neighbors, key=lambda n: n.cpu_freq)
                    # 找到best_n在neighbors中的索引
                    n_idx = neighbors.index(best_n)
                    target_v2v = 2 + n_idx  # Local(0) + RSU(1) + V2V(2+...)

                    t_v2v, t_tx_v2v = env._estimate_t_actual(v, subtask_idx,
                                                              target_v2v,
                                                              cycles, 1.0)
                    wait_v2v = env._get_vehicle_queue_wait_time(best_n.id)
                    comp_v2v = cycles / best_n.cpu_freq
                else:
                    t_v2v = t_tx_v2v = wait_v2v = comp_v2v = None

                time_breakdown.append({
                    'cycles_gcycles': cycles / 1e9,
                    'data_mb': data / 8 / 1e6,
                    'cpu_self_ghz': v.cpu_freq / 1e9,
                    'cpu_best_neighbor_ghz': best_n.cpu_freq / 1e9 if neighbors else None,
                    't_local': t_local,
                    'queue_local': queue_local,
                    'comp_local': comp_local,
                    't_rsu': t_rsu,
                    't_tx_rsu': t_tx_rsu,
                    'wait_rsu': wait_rsu,
                    'comp_rsu': comp_rsu,
                    't_v2v': t_v2v,
                    't_tx_v2v': t_tx_v2v,
                    'wait_v2v': wait_v2v,
                    'comp_v2v': comp_v2v,
                    'rsu_wins': t_rsu < t_local if t_rsu else False,
                    'v2v_wins': t_v2v < t_local if t_v2v else False,
                    'rsu_advantage': t_local - t_rsu if t_rsu else None,
                    'v2v_advantage': t_local - t_v2v if t_v2v else None,
                })

            # 推进环境
            actions = [random_action() for _ in range(20)]
            env.step(actions)

        print(f"  Episode {ep+1}/{num_episodes} 完成")

    # 转换为DataFrame
    df = pd.DataFrame(time_breakdown)

    print("\n" + "=" * 70)
    print("时延统计分析")
    print("=" * 70)

    print("\n【本地执行时间 (t_local)】")
    print(f"  均值:  {df['t_local'].mean():.3f}s")
    print(f"  p50:   {df['t_local'].quantile(0.50):.3f}s")
    print(f"  p95:   {df['t_local'].quantile(0.95):.3f}s")

    print("\n【RSU卸载时间 (t_rsu)】")
    print(f"  均值:  {df['t_rsu'].mean():.3f}s")
    print(f"  p50:   {df['t_rsu'].quantile(0.50):.3f}s")
    print(f"  p95:   {df['t_rsu'].quantile(0.95):.3f}s")
    print(f"\n  分解:")
    print(f"    传输(t_tx):  {df['t_tx_rsu'].mean():.3f}s ({df['t_tx_rsu'].mean()/df['t_rsu'].mean()*100:.1f}%)")
    print(f"    等待(wait):  {df['wait_rsu'].mean():.3f}s ({df['wait_rsu'].mean()/df['t_rsu'].mean()*100:.1f}%)")
    print(f"    计算(comp):  {df['comp_rsu'].mean():.3f}s ({df['comp_rsu'].mean()/df['t_rsu'].mean()*100:.1f}%)")

    print("\n【V2V卸载时间 (t_v2v)】")
    print(f"  均值:  {df['t_v2v'].mean():.3f}s")
    print(f"  p50:   {df['t_v2v'].quantile(0.50):.3f}s")
    print(f"  p95:   {df['t_v2v'].quantile(0.95):.3f}s")
    print(f"\n  分解:")
    print(f"    传输(t_tx):  {df['t_tx_v2v'].mean():.3f}s ({df['t_tx_v2v'].mean()/df['t_v2v'].mean()*100:.1f}%)")
    print(f"    等待(wait):  {df['wait_v2v'].mean():.3f}s ({df['wait_v2v'].mean()/df['t_v2v'].mean()*100:.1f}%)")
    print(f"    计算(comp):  {df['comp_v2v'].mean():.3f}s ({df['comp_v2v'].mean()/df['t_v2v'].mean()*100:.1f}%)")

    # 卸载优势分析
    print("\n" + "=" * 70)
    print("卸载优势分析")
    print("=" * 70)

    p_rsu_wins = df['rsu_wins'].mean() * 100
    p_v2v_wins = df['v2v_wins'].mean() * 100

    print(f"\n【RSU优势】")
    print(f"  P(t_rsu < t_local):  {p_rsu_wins:.1f}%")
    if p_rsu_wins > 0:
        rsu_wins = df[df['rsu_wins'] == True]
        print(f"  median(t_local - t_rsu | rsu_wins): {rsu_wins['rsu_advantage'].median():.3f}s")
        print(f"  mean(t_local - t_rsu | rsu_wins):   {rsu_wins['rsu_advantage'].mean():.3f}s")

    print(f"\n【V2V优势】")
    print(f"  P(t_v2v < t_local):  {p_v2v_wins:.1f}%")
    if p_v2v_wins > 0:
        v2v_wins = df[df['v2v_wins'] == True]
        print(f"  median(t_local - t_v2v | v2v_wins): {v2v_wins['v2v_advantage'].median():.3f}s")
        print(f"  mean(t_local - t_v2v | v2v_wins):   {v2v_wins['v2v_advantage'].mean():.3f}s")

    # CCR分析
    print("\n" + "=" * 70)
    print("CCR (通信计算比) 分析")
    print("=" * 70)

    ccr_rsu = df['t_tx_rsu'].mean() / df['comp_local'].mean()
    ccr_v2v = df['t_tx_v2v'].mean() / df['comp_local'].mean()

    print(f"\n  CCR_RSU = t_tx_rsu / t_comp_local = {ccr_rsu:.2f}")
    print(f"  CCR_V2V = t_tx_v2v / t_comp_local = {ccr_v2v:.2f}")
    print(f"\n  CCR > 1: 通信主导")
    print(f"  CCR < 1: 计算主导")

    # 判定结果
    print("\n" + "=" * 70)
    print("判定结果")
    print("=" * 70)

    print("\n标准1: P(t_offload < t_local) >= 10%")
    rsu_criterion = p_rsu_wins >= 10.0
    v2v_criterion = p_v2v_wins >= 10.0
    print(f"  P(t_rsu < t_local) = {p_rsu_wins:.1f}%  {'✓' if rsu_criterion else '✗'}")
    print(f"  P(t_v2v < t_local) = {p_v2v_wins:.1f}%  {'✓' if v2v_criterion else '✗'}")
    criterion1_pass = rsu_criterion or v2v_criterion

    print("\n标准2: median(优势幅度) >= 0.05s (当有卸载优势时)")
    criterion2_pass = True
    if p_rsu_wins > 0:
        rsu_median_adv = df[df['rsu_wins'] == True]['rsu_advantage'].median()
        rsu_adv_ok = rsu_median_adv >= 0.05
        print(f"  RSU median优势: {rsu_median_adv:.3f}s  {'✓' if rsu_adv_ok else '✗'}")
        criterion2_pass = criterion2_pass and rsu_adv_ok

    if p_v2v_wins > 0:
        v2v_median_adv = df[df['v2v_wins'] == True]['v2v_advantage'].median()
        v2v_adv_ok = v2v_median_adv >= 0.05
        print(f"  V2V median优势: {v2v_median_adv:.3f}s  {'✓' if v2v_adv_ok else '✗'}")
        criterion2_pass = criterion2_pass and v2v_adv_ok

    print("\n标准3: 瓶颈诊断")
    t_tx_rsu_ratio = df['t_tx_rsu'].mean() / df['t_rsu'].mean()
    print(f"  t_tx_rsu / t_rsu = {t_tx_rsu_ratio:.2f}")
    if t_tx_rsu_ratio > 0.6:
        print(f"  ⚠ V2I传输主导 (>60%)，建议增加带宽或减少数据量")

    overall_pass = criterion1_pass and criterion2_pass

    print(f"\n{'='*70}")
    print(f"Phase 2 最终判定: {'✓ 通过' if overall_pass else '✗ 不通过'}")
    print(f"{'='*70}")

    # 保存详细数据
    output_file = Path(__file__).parent.parent / "docs" / "phase2_time_breakdown.csv"
    df.to_csv(output_file, index=False)
    print(f"\n详细数据已保存至: {output_file}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
