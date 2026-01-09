#!/usr/bin/env python3
"""
Logit Bias消融实验脚本

通过改变RSU的logit bias值，运行rollouts对比：
- effective action分布
- success_rate
- 可用目标分布

Usage:
    python scripts/ablate_logit_bias.py --bias-values 0,2.0,3.5,5.0 --max-episodes 10
"""

import argparse
import numpy as np
import sys
import os
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


def run_rollouts(bias_rsu, max_episodes, seed=42):
    """
    使用指定的RSU bias运行rollouts (随机策略)

    Args:
        bias_rsu: RSU的logit bias值
        max_episodes: 最大episode数
        seed: 随机种子

    Returns:
        dict: 统计结果
    """
    # 临时修改TrainConfig的bias值
    original_bias_rsu = TC.LOGIT_BIAS_RSU
    TC.LOGIT_BIAS_RSU = bias_rsu

    try:
        np.random.seed(seed)

        # 创建环境
        env = VecOffloadingEnv()

        # 统计
        episode_stats = []
        all_rewards = []
        target_counts = {'local': 0, 'rsu': 0, 'v2v': 0}
        mask_avail = {'local': 0, 'rsu': 0, 'v2v': 0, 'total': 0}

        for ep in range(max_episodes):
            obs, _ = env.reset(seed=seed + ep)
            episode_reward = 0.0
            step_count = 0

            while True:
                # 随机策略：在可用动作中随机选择
                actions = []
                for i, v in enumerate(env.vehicles):
                    # 获取target_mask
                    if i < len(obs) and isinstance(obs[i], dict):
                        target_mask = obs[i].get('target_mask', np.ones(Cfg.MAX_TARGETS, dtype=bool))
                    else:
                        target_mask = np.ones(Cfg.MAX_TARGETS, dtype=bool)

                    # 统计mask可用性
                    mask_avail['total'] += 1
                    mask_avail['local'] += int(target_mask[0]) if len(target_mask) > 0 else 0
                    mask_avail['rsu'] += int(target_mask[1]) if len(target_mask) > 1 else 0
                    mask_avail['v2v'] += int(np.any(target_mask[2:])) if len(target_mask) > 2 else 0

                    # 应用logit bias后采样
                    valid_indices = np.where(target_mask)[0]
                    if len(valid_indices) == 0:
                        valid_indices = [0]  # fallback to local

                    # 创建带bias的概率
                    logits = np.zeros(Cfg.MAX_TARGETS)
                    logits[0] = TC.LOGIT_BIAS_LOCAL
                    logits[1] = TC.LOGIT_BIAS_RSU
                    # V2V targets保持0

                    # 只对valid actions计算softmax
                    valid_logits = logits[valid_indices]
                    probs = np.exp(valid_logits - np.max(valid_logits))
                    probs = probs / np.sum(probs)

                    chosen_idx = np.random.choice(len(valid_indices), p=probs)
                    target = valid_indices[chosen_idx]

                    # 统计选择
                    if target == 0:
                        target_counts['local'] += 1
                    elif target == 1:
                        target_counts['rsu'] += 1
                    else:
                        target_counts['v2v'] += 1

                    actions.append({
                        'target': int(target),
                        'power': np.random.uniform(0.5, 1.0)
                    })

                obs, rewards, terminated, truncated, info = env.step(actions)

                if isinstance(rewards, (list, np.ndarray)):
                    episode_reward += np.mean(rewards)
                else:
                    episode_reward += rewards

                step_count += 1

                if terminated or truncated:
                    break

            # 记录统计
            metrics = info.get('episode_metrics', {})
            episode_stats.append({
                'episode': ep,
                'reward': episode_reward,
                'steps': step_count,
                'success_rate': metrics.get('task_success_rate', 0.0),
                'failed_rate': metrics.get('deadline_miss_rate', 0.0),
                'p_target_eff_local': metrics.get('p_target_eff_local_frac', 0.0),
                'p_target_eff_rsu': metrics.get('p_target_eff_rsu_frac', 0.0),
                'p_target_eff_v2v': metrics.get('p_target_eff_v2v_frac', 0.0),
                'fallback_rate': metrics.get('fallback_rate', 0.0),
            })
            all_rewards.append(episode_reward)

            # 进度显示
            if (ep + 1) % 5 == 0:
                recent_reward = np.mean(all_rewards[-5:])
                recent_success = np.mean([s['success_rate'] for s in episode_stats[-5:]])
                print(f"    Episode {ep+1}/{max_episodes}: "
                      f"reward={recent_reward:.2f}, success_rate={recent_success:.2%}")

        # 汇总统计
        total_targets = sum(target_counts.values())
        summary = {
            'bias_rsu': bias_rsu,
            'num_episodes': max_episodes,
            'final_reward_mean': np.mean([s['reward'] for s in episode_stats]),
            'final_reward_std': np.std([s['reward'] for s in episode_stats]),
            'final_success_rate': np.mean([s['success_rate'] for s in episode_stats]),
            'final_failed_rate': np.mean([s['failed_rate'] for s in episode_stats]),
            'target_frac_local': target_counts['local'] / max(total_targets, 1),
            'target_frac_rsu': target_counts['rsu'] / max(total_targets, 1),
            'target_frac_v2v': target_counts['v2v'] / max(total_targets, 1),
            'mask_avail_local': mask_avail['local'] / max(mask_avail['total'], 1),
            'mask_avail_rsu': mask_avail['rsu'] / max(mask_avail['total'], 1),
            'mask_avail_v2v': mask_avail['v2v'] / max(mask_avail['total'], 1),
            'final_fallback_rate': np.mean([s['fallback_rate'] for s in episode_stats]),
        }

        return summary, episode_stats

    finally:
        # 恢复原始bias值
        TC.LOGIT_BIAS_RSU = original_bias_rsu


def main():
    parser = argparse.ArgumentParser(description='Logit Bias消融实验')
    parser.add_argument('--bias-values', type=str, default='0,2.0,3.5,5.0',
                        help='RSU bias值列表，逗号分隔')
    parser.add_argument('--max-episodes', type=int, default=10,
                        help='每个bias值运行的episode数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--output', type=str, default=None,
                        help='输出JSON文件路径')
    args = parser.parse_args()

    bias_values = [float(b.strip()) for b in args.bias_values.split(',')]

    print("=" * 70)
    print("Logit Bias消融实验")
    print("=" * 70)
    print(f"Bias values: {bias_values}")
    print(f"Episodes per bias: {args.max_episodes}")
    print(f"Seed: {args.seed}")
    print()

    all_summaries = []
    all_details = {}

    for bias in bias_values:
        print(f"\n--- Testing LOGIT_BIAS_RSU = {bias} ---")
        summary, details = run_rollouts(bias, args.max_episodes, args.seed)
        all_summaries.append(summary)
        all_details[f'bias_{bias}'] = details

    # 打印汇总表格
    print()
    print("=" * 90)
    print("消融实验结果汇总")
    print("=" * 90)
    print(f"{'Bias_RSU':<10} {'SuccessRate':<12} {'TargetLocal':<12} {'TargetRSU':<12} {'TargetV2V':<12} "
          f"{'MaskRSU':<10} {'Return':<12}")
    print("-" * 90)

    for s in all_summaries:
        print(f"{s['bias_rsu']:<10.1f} "
              f"{s['final_success_rate']:.2%}      "
              f"{s['target_frac_local']:.2%}       "
              f"{s['target_frac_rsu']:.2%}       "
              f"{s['target_frac_v2v']:.2%}       "
              f"{s['mask_avail_rsu']:.2%}     "
              f"{s['final_reward_mean']:.1f}±{s['final_reward_std']:.1f}")

    # 分析结论
    print()
    print("=" * 70)
    print("分析结论")
    print("=" * 70)

    # 找到最佳bias
    best_success = max(all_summaries, key=lambda x: x['final_success_rate'])
    best_return = max(all_summaries, key=lambda x: x['final_reward_mean'])

    print(f"最高成功率: bias={best_success['bias_rsu']} ({best_success['final_success_rate']:.2%})")
    print(f"最高回报: bias={best_return['bias_rsu']} ({best_return['final_reward_mean']:.1f})")

    # 检查RSU占比变化
    print("\nRSU选择率变化趋势:")
    for s in all_summaries:
        rsu_pct = s['target_frac_rsu']
        indicator = "↑" if rsu_pct > 0.2 else "→" if rsu_pct > 0.1 else "↓"
        print(f"  bias={s['bias_rsu']:.1f}: RSU选择={rsu_pct:.1%}, RSU可用={s['mask_avail_rsu']:.1%} {indicator}")

    # 保存JSON
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'bias_values': bias_values,
                'max_episodes': args.max_episodes,
                'seed': args.seed,
            },
            'summaries': all_summaries,
            'details': all_details,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=float)
        print(f"\n详细结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
