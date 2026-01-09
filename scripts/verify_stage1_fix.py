"""
验证 Stage 1 修复效果的简单脚本
"""

import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv

print("=" * 70)
print("Stage 1 修复效果验证")
print("=" * 70)

env = VecOffloadingEnv()
env.config.REWARD_SCHEME = "PBRS_KP"
env.reset(seed=42)

print(f"\n配置验证:")
print(f"  NO_TASK_PENALTY_DAG_DONE:  {env.config.NO_TASK_PENALTY_DAG_DONE}")
print(f"  NO_TASK_PENALTY_BLOCKED:   {env.config.NO_TASK_PENALTY_BLOCKED}")
print(f"  ILLEGAL_PENALTY:           {env.config.ILLEGAL_PENALTY}")

# 运行一个 episode
print(f"\n运行 1 个 episode...")
for step in range(200):
    actions = [{"target": 0, "power": 1.0} for _ in range(env.config.NUM_VEHICLES)]
    obs, rewards, done, trunc, info = env.step(actions)
    if trunc:
        break

print(f"Episode 完成，共 {step+1} 步")

# 提取统计
if hasattr(env, '_reward_stats'):
    counters = env._reward_stats.counters
    metrics = env._reward_stats.metrics

    # 汇总 illegal 分布
    illegal_reasons = {k: v for k, v in counters.items() if k.startswith("illegal_")}

    if illegal_reasons:
        print(f"\n{'='*70}")
        print("no_task 细分分布（关键验证）:")
        print(f"{'='*70}")
        total = sum(illegal_reasons.values())
        for reason, count in sorted(illegal_reasons.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(total, 1)
            print(f"  {reason:40s}: {count:5d} ({pct:5.1f}%)")

        print(f"\n总决策数: {total}")

        # 关键指标
        dag_done_count = counters.get("illegal_task_done", 0) + counters.get("illegal_no_task_dag_done", 0)
        blocked_count = counters.get("illegal_no_task_blocked", 0)

        print(f"\n关键指标:")
        print(f"  DAG完成后等待 (task_done + no_task_dag_done): {dag_done_count} ({100*dag_done_count/total:.1f}%)")
        print(f"  依赖阻塞 (no_task_blocked):                    {blocked_count} ({100*blocked_count/total:.1f}%)")

    # 奖励组成
    if hasattr(metrics, 'get') or isinstance(metrics, dict):
        r_illegal_metric = metrics.get("r_illegal") if isinstance(metrics, dict) else metrics.get("r_illegal")
        if r_illegal_metric and hasattr(r_illegal_metric, 'values'):
            r_illegal_vals = r_illegal_metric.values
            print(f"\n{'='*70}")
            print("r_illegal 统计（关键修复验证）:")
            print(f"{'='*70}")
            print(f"  均值:     {np.mean(r_illegal_vals):.4f}  (修复前: -1.796)")
            print(f"  标准差:   {np.std(r_illegal_vals):.4f}")
            print(f"  最小值:   {np.min(r_illegal_vals):.4f}")
            print(f"  最大值:   {np.max(r_illegal_vals):.4f}")

            # 统计 r_illegal = 0 的比例
            zero_count = np.sum(np.isclose(r_illegal_vals, 0.0, atol=1e-9))
            total_count = len(r_illegal_vals)
            print(f"  零惩罚比例: {100*zero_count/total_count:.1f}% ({zero_count}/{total_count})")

            # 预期：修复后大部分 r_illegal 应为 0
            if zero_count / total_count > 0.8:
                print(f"\n✓ 修复成功！零惩罚比例 {100*zero_count/total_count:.1f}% > 80%")
            else:
                print(f"\n⚠ 零惩罚比例 {100*zero_count/total_count:.1f}% < 80%，可能需要进一步优化")

print(f"\n{'='*70}")
print("验证完成")
print(f"{'='*70}")
