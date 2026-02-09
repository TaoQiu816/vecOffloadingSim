#!/usr/bin/env python3
"""
Reward 三分量比例审计（训练前必跑）。

跑 N 个 episode（随机策略），统计每 episode 累计的:
  sum_r_step, sum_r_term, sum_r_pbrs
输出 mean/p50/p95 + 告警阈值检查，保存 CSV。

用法:
  python scripts/reward_scale_report.py --episodes 20 --seeds 3
  python scripts/reward_scale_report.py --episodes 50 --out logs/reward_scale.csv
"""
import sys, os, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from configs.config import SystemConfig as Cfg

Cfg.REWARD_SCHEME = "UNIFIED"
Cfg.TRUST_ENABLED = True

from envs.vec_offloading_env import VecOffloadingEnv


def _run_one_episode(env, seed):
    """跑一个 episode（随机策略），返回三分量的 per-agent 累计。"""
    obs_list, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    n = len(obs_list)
    acc = {"r_step": np.zeros(n), "r_term": np.zeros(n), "r_pbrs": np.zeros(n)}

    for _ in range(Cfg.MAX_STEPS):
        actions = []
        for obs in obs_list:
            mask = np.asarray(obs["action_mask"]).astype(bool)
            valid = np.where(mask)[0]
            t = int(rng.choice(valid)) if len(valid) > 0 else 0
            actions.append({"target": t, "power": float(rng.random())})
        obs_list, rewards, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    # 从 env._reward_stats 收集
    if hasattr(env, "_reward_stats"):
        s = env._reward_stats.summary().get("metrics", {})
        for key in ("r_step", "r_term", "r_pbrs"):
            m = s.get(key, {})
            # mean * count 近似 sum；但这里用 agent 维度的 sum 更精确
            # 简化：用 mean * count 作为 episode 总量
            acc[key] = np.array([m.get("mean", 0.0) * m.get("count", 0)])

    return {k: float(v.sum()) for k, v in acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--vehicles", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    Cfg.NUM_VEHICLES = args.vehicles
    Cfg.MAX_STEPS = args.max_steps

    env = VecOffloadingEnv()
    rows = []

    for seed_base in range(args.seeds):
        for ep in range(args.episodes):
            seed = seed_base * 10000 + ep
            result = _run_one_episode(env, seed)
            result["seed"] = seed_base
            result["episode"] = ep
            rows.append(result)
            if (ep + 1) % 10 == 0:
                print(f"  seed={seed_base} ep={ep+1}/{args.episodes}")

    env.close()

    # 聚合
    arr_step = np.array([r["r_step"] for r in rows])
    arr_term = np.array([r["r_term"] for r in rows])
    arr_pbrs = np.array([r["r_pbrs"] for r in rows])
    arr_total = arr_step + arr_term + arr_pbrs

    def _stats(name, arr):
        return {
            "component": name,
            "mean": float(np.mean(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    stats = [
        _stats("sum_r_step", arr_step),
        _stats("sum_r_term", arr_term),
        _stats("sum_r_pbrs", arr_pbrs),
        _stats("sum_total", arr_total),
    ]

    print("\n=== Reward Scale Report ===")
    print(f"  episodes={len(rows)}, vehicles={args.vehicles}, max_steps={args.max_steps}")
    for s in stats:
        print(f"  {s['component']:>14s}: mean={s['mean']:+10.2f}  abs_mean={s['abs_mean']:10.2f}  "
              f"p50={s['p50']:+10.2f}  p95={s['p95']:+10.2f}  "
              f"[{s['min']:+.2f}, {s['max']:+.2f}]")

    # 告警检查
    warnings = []
    abs_pbrs = np.mean(np.abs(arr_pbrs))
    abs_term = np.mean(np.abs(arr_term))
    if abs_term > 1e-6 and abs_pbrs > 2 * abs_term:
        warnings.append(f"|sum_r_pbrs|={abs_pbrs:.2f} > 2*|sum_r_term|={abs_term:.2f}: PBRS 可能主导")
    if np.mean(np.abs(arr_step)) < 1e-6:
        warnings.append("sum_r_step ≈ 0: step reward 未生效")
    if not np.all(np.isfinite(arr_total)):
        warnings.append("存在 NaN/Inf")

    if warnings:
        print("\n  ⚠ WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  ✓ 所有检查通过")

    # 保存 CSV
    out_path = args.out
    if out_path is None:
        os.makedirs("logs", exist_ok=True)
        out_path = "logs/reward_scale_report.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "episode", "r_step", "r_term", "r_pbrs"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "seed": r["seed"], "episode": r["episode"],
                "r_step": f"{r['r_step']:.4f}",
                "r_term": f"{r['r_term']:.4f}",
                "r_pbrs": f"{r['r_pbrs']:.4f}",
            })
    print(f"\n  CSV 已保存: {out_path}")


if __name__ == "__main__":
    main()
