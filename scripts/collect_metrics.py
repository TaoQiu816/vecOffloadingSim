#!/usr/bin/env python3
"""
短训练几十 episode，输出成功率与功率曲线汇总。
用法: python scripts/collect_metrics.py [--episodes 30] [--seed 42]
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from configs.config import SystemConfig as Cfg

Cfg.REWARD_SCHEME = "UNIFIED"
Cfg.TRUST_ENABLED = True

from envs.vec_offloading_env import VecOffloadingEnv


def run_episodes(n_episodes, seed):
    env = VecOffloadingEnv()
    records = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        ep_rewards = []
        for step in range(Cfg.MAX_STEPS):
            actions = {}
            for v in env.vehicles:
                target_idx = np.random.randint(0, Cfg.MAX_TARGETS)
                power = np.random.random()
                actions[v.id] = np.array([target_idx, power], dtype=np.float32)
            obs, rewards, terminated, truncated, info = env.step(actions)
            ep_rewards.extend(rewards if isinstance(rewards, list) else [rewards])
            if terminated or truncated:
                break
        rec = {
            'episode': ep,
            'reward_mean': float(np.mean(ep_rewards)),
            'reward_sum': float(np.sum(ep_rewards)),
        }
        for k in ['ep_success_rate', 'ep_makespan', 'ep_T_finish_mean',
                   'ep_delta_T_p50', 'ep_delta_T_p95',
                   'ep_sinr_p50', 'ep_sinr_p05', 'ep_sinr_p95',
                   'ep_rb_concurrency_mean', 'ep_i_caused_mean',
                   'ep_E_tx_total', 'ep_E_tx_mean', 'ep_power_mean_W',
                   'ep_jain_fairness', 'ep_worst10_mean',
                   'ep_T_tx_svc_mean', 'ep_T_tx_wait_mean',
                   'ep_T_cpu_svc_mean', 'ep_T_cpu_wait_mean',
                   'trust_attempts', 'trust_failures', 'trust_failure_rate',
                   'trust_retry_count', 'ho_event_count']:
            rec[k] = info.get(k, 0.0)
        records.append(rec)
        print(f"  ep={ep:3d} succ={rec['ep_success_rate']:.2f} "
              f"make={rec['ep_makespan']:.2f} R={rec['reward_mean']:+.3f} "
              f"SINR_p50={rec['ep_sinr_p50']:.1f} trust_fail={rec.get('trust_failure_rate',0):.2f}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='logs/metrics_summary.jsonl')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    print(f"Running {args.episodes} episodes with random policy (seed={args.seed})...")
    records = run_episodes(args.episodes, args.seed)

    with open(args.out, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"\nSaved {len(records)} records to {args.out}")

    # 汇总
    keys = ['ep_success_rate', 'ep_makespan', 'reward_mean',
            'ep_sinr_p50', 'ep_i_caused_mean', 'ep_E_tx_total',
            'ep_jain_fairness', 'ep_power_mean_W',
            'trust_failure_rate', 'trust_retry_count']
    print("\n=== Aggregate ===")
    for k in keys:
        vals = [r.get(k, 0.0) for r in records]
        print(f"  {k:30s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
