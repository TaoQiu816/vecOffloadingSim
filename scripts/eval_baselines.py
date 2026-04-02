#!/usr/bin/env python3
"""评估baseline方法（LO/NRO/EFT-H）"""
import os
import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.vec_offloading_env import VecOffloadingEnv
from baselines.local_only_policy import LocalOnlyPolicy
from baselines.nearest_rsu_policy import NearestRSUPolicy
from baselines.eft_policy import EFTPolicy


def make_env():
    return VecOffloadingEnv()


def run_episode(env, seed, policy):
    obs, _ = env.reset(seed=seed)
    done = False
    ep_reward = 0.0
    MAX_STEPS = 500
    
    while not done:
        actions = policy.select_action(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)
        ep_reward += sum(rewards)
        done = terminated or truncated
    
    return {
        "seed": seed,
        "episode_reward": ep_reward,
        "task_success_rate": info.get("task_success_rate", 0.0),
        "mean_cft": info.get("mean_cft", 0.0),
        "mean_energy": info.get("mean_energy", 0.0),
        "mean_interference": info.get("mean_interference", 0.0),
    }


def evaluate(name, env, policy, seeds=range(9000, 9010)):
    print(f"\n评估 {name}...")
    rows = []
    for seed in seeds:
        result = run_episode(env, seed, policy)
        rows.append(result)
        print(f"  Seed {seed}: SR={result['task_success_rate']:.3f}")
    
    # 计算统计
    sr_vals = [r["task_success_rate"] for r in rows]
    cft_vals = [r["mean_cft"] for r in rows]
    energy_vals = [r["mean_energy"] for r in rows]
    
    summary = {
        "policy": name,
        "task_success_rate_mean": sum(sr_vals) / len(sr_vals),
        "task_success_rate_std": (sum((x - sum(sr_vals)/len(sr_vals))**2 for x in sr_vals) / len(sr_vals))**0.5,
        "mean_cft_mean": sum(cft_vals) / len(cft_vals),
        "mean_energy_mean": sum(energy_vals) / len(energy_vals),
    }
    
    return rows, summary


def main():
    out_dir = Path("runs/paper_final_results/group2_comprehensive_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    env = make_env()
    
    policies = [
        ("LO", LocalOnlyPolicy()),
        ("NRO", NearestRSUPolicy(env)),
        ("EFT-H", EFTPolicy(env)),
    ]
    
    all_summaries = []
    
    for name, policy in policies:
        rows, summary = evaluate(name, env, policy)
        all_summaries.append(summary)
        
        # 保存详细结果
        csv_path = out_dir / f"tables/{name}_episodes.csv"
        csv_path.parent.mkdir(exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    # 保存汇总
    summary_path = out_dir / "tables/baseline_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    
    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
