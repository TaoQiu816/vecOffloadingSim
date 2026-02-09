#!/usr/bin/env python3
"""
多 seed 汇总脚本：对比 Random / EFT / CP-First+EFT 三种策略。

输出：
  - 成功率 (mean ± std)
  - 条件完成时延 T_finish (mean/p50/p95)
  - mean/p95 功率 (W)
  - I_total (mean)
  - Jain index (成功率公平性)
  - worst 10% 完成时延

用法:
  python scripts/aggregate_multiseed.py --seeds 5 --episodes 10 --vehicles 10
  python scripts/aggregate_multiseed.py --seeds 3 --episodes 5 --vehicles 30 --out logs/multiseed_30v.csv
"""

import sys, os, argparse, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from configs.config import SystemConfig as Cfg

Cfg.REWARD_SCHEME = "UNIFIED"
Cfg.TRUST_ENABLED = True

from envs.vec_offloading_env import VecOffloadingEnv


def _run_policy(env, policy, obs_list, rng):
    """执行一个 episode，返回 info。"""
    done = False
    while not done:
        if policy == 'random':
            n = len(obs_list)
            targets = []
            for i in range(n):
                m = obs_list[i]['action_mask']
                valid = np.where(m > 0.5)[0]
                t = rng.choice(valid) if len(valid) > 0 else 0
                targets.append(t)
            powers = rng.random(size=n)
            actions = np.column_stack([targets, powers])
        else:
            # EFT or CP-First+EFT
            act_list = policy.select_action(obs_list)
            n = len(act_list)
            actions = np.zeros((n, 2))
            for j, a in enumerate(act_list):
                actions[j, 0] = a['target']
                actions[j, 1] = a['power']

        obs_list, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
    return info


def _extract_metrics(info, env):
    """从 info 字典提取标准指标。"""
    m = {}
    m['success_rate'] = info.get('ep_success_rate', 0.0)
    m['makespan'] = info.get('ep_makespan', 0.0)
    m['T_finish_mean'] = info.get('ep_T_finish_mean', 0.0)
    m['T_finish_p50'] = info.get('ep_T_finish_p50', 0.0)
    m['delta_T_p50'] = info.get('ep_delta_T_p50', 0.0)
    m['delta_T_p95'] = info.get('ep_delta_T_p95', 0.0)
    m['power_mean_W'] = info.get('ep_power_mean_W', 0.0)
    m['E_tx_total'] = info.get('ep_E_tx_total', 0.0)
    m['sinr_p50'] = info.get('ep_sinr_p50', 0.0)
    m['i_caused_mean'] = info.get('ep_i_caused_mean', 0.0)
    m['i_total_mean'] = info.get('ep_i_total_mean', m['i_caused_mean'])
    m['rb_concurrency'] = info.get('ep_rb_concurrency_mean', 0.0)
    m['T_tx_svc'] = info.get('ep_T_tx_svc_mean', 0.0)
    m['T_tx_wait'] = info.get('ep_T_tx_wait_mean', 0.0)
    m['T_cpu_svc'] = info.get('ep_T_cpu_svc_mean', 0.0)
    m['T_cpu_wait'] = info.get('ep_T_cpu_wait_mean', 0.0)
    m['trust_failures'] = info.get('trust_failures', 0)
    m['trust_attempts'] = info.get('trust_attempts', 0)
    m['ho_events'] = info.get('ho_event_count', 0)

    # Jain index (per-vehicle success)
    m['jain_fairness'] = info.get('ep_jain_fairness', 0.0)
    m['worst10_mean'] = info.get('ep_worst10_mean', 0.0)

    return m


def _jain_index(values):
    """计算 Jain's fairness index。"""
    n = len(values)
    if n == 0:
        return 0.0
    s = np.sum(values)
    ss = np.sum(np.square(values))
    if ss < 1e-12:
        return 1.0
    return (s ** 2) / (n * ss)


def run_experiment(n_seeds, n_episodes, n_vehicles):
    """Run all policies across seeds and episodes."""
    Cfg.NUM_VEHICLES = n_vehicles

    from baselines.eft_policy import EFTPPolicy
    from baselines.cp_first_eft_policy import CPFirstEFTPolicy

    policy_names = ['Random', 'EFT', 'CP-First+EFT']
    all_results = {name: [] for name in policy_names}

    for seed in range(n_seeds):
        env = VecOffloadingEnv(config=Cfg)

        for ep in range(n_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)

            for pname in policy_names:
                obs_list, _ = env.reset(seed=ep_seed)

                if pname == 'Random':
                    info = _run_policy(env, 'random', obs_list, rng)
                elif pname == 'EFT':
                    pol = EFTPPolicy(env)
                    info = _run_policy(env, pol, obs_list, rng)
                else:
                    pol = CPFirstEFTPolicy(env)
                    info = _run_policy(env, pol, obs_list, rng)

                metrics = _extract_metrics(info, env)
                metrics['seed'] = seed
                metrics['episode'] = ep
                metrics['policy'] = pname
                all_results[pname].append(metrics)

            if ep % 5 == 0:
                print(f"  seed={seed} ep={ep}/{n_episodes}")

    return all_results


def aggregate_and_print(all_results, out_csv=None):
    """汇总并打印/保存结果。"""
    keys = ['success_rate', 'T_finish_mean', 'T_finish_p50', 'delta_T_p50', 'delta_T_p95',
            'power_mean_W', 'E_tx_total', 'sinr_p50', 'i_caused_mean',
            'jain_fairness', 'worst10_mean',
            'T_tx_svc', 'T_tx_wait', 'T_cpu_svc', 'T_cpu_wait',
            'trust_failures', 'ho_events']

    print("\n" + "=" * 100)
    print(f"{'Metric':<25}", end='')
    for pname in all_results:
        print(f"  {pname:>20}", end='')
    print()
    print("=" * 100)

    summary_rows = []
    for k in keys:
        row = {'metric': k}
        print(f"{k:<25}", end='')
        for pname, records in all_results.items():
            vals = [r[k] for r in records]
            mu = np.mean(vals)
            sd = np.std(vals)
            print(f"  {mu:>9.4f}±{sd:>7.4f}", end='')
            row[f'{pname}_mean'] = mu
            row[f'{pname}_std'] = sd
        print()
        summary_rows.append(row)

    print("=" * 100)

    # Jain index 补充：基于 per-seed 的成功率
    print("\nJain index (跨episode成功率):")
    for pname, records in all_results.items():
        seed_success = {}
        for r in records:
            s = r['seed']
            if s not in seed_success:
                seed_success[s] = []
            seed_success[s].append(r['success_rate'])
        per_seed_means = [np.mean(v) for v in seed_success.values()]
        jain = _jain_index(per_seed_means)
        print(f"  {pname}: Jain={jain:.4f}")

    # worst 10%
    print("\nWorst 10% 完成时延 (条件: 成功 episode):")
    for pname, records in all_results.items():
        finish_times = [r['T_finish_mean'] for r in records if r['success_rate'] > 0]
        if len(finish_times) > 0:
            finish_times.sort(reverse=True)
            n10 = max(1, len(finish_times) // 10)
            worst10 = np.mean(finish_times[:n10])
            print(f"  {pname}: worst10%_T_finish={worst10:.4f}s (n={len(finish_times)})")
        else:
            print(f"  {pname}: no successful episodes")

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else '.', exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nCSV saved to {out_csv}")

        # 也保存详细记录为 JSONL
        jsonl_path = out_csv.replace('.csv', '_detail.jsonl')
        with open(jsonl_path, 'w') as f:
            for pname, records in all_results.items():
                for r in records:
                    f.write(json.dumps(r) + '\n')
        print(f"Detail JSONL saved to {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description='多 seed 策略对比汇总')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--vehicles', type=int, default=10)
    parser.add_argument('--out', type=str, default='logs/multiseed_summary.csv')
    args = parser.parse_args()

    print(f"配置: seeds={args.seeds}, episodes={args.episodes}, vehicles={args.vehicles}")
    results = run_experiment(args.seeds, args.episodes, args.vehicles)
    aggregate_and_print(results, args.out)


if __name__ == "__main__":
    main()
