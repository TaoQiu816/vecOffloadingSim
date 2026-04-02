#!/usr/bin/env python3
"""评估所有RL模型（F-MAPPO, TERA-MAPPO, IPPO-H, 消融变体）"""
import os
import sys
import json
import csv
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from envs.vec_offloading_env import VecOffloadingEnv
from agents.agent_factory import create_agent
from configs import train_config as TC


def make_env():
    return VecOffloadingEnv()


def load_agent(model_path, algo_mode="mappo"):
    """加载训练好的agent"""
    device = torch.device("cpu")
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"模型不存在: {ckpt_path}")
    
    # 保存并设置algo_mode
    old_algo = getattr(TC, "ALGO_MODE", "mappo")
    TC.ALGO_MODE = algo_mode
    
    env = make_env()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(state_dict)
    except Exception as e:
        # 尝试加载checkpoint目录中的best_model.pth
        if ckpt_path.is_dir():
            best_model = ckpt_path / "models" / "best_model.pth"
            if best_model.exists():
                state_dict = torch.load(best_model, map_location=device)
                agent.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"找不到模型文件: {best_model}")
        else:
            raise e
    
    # 恢复algo_mode
    TC.ALGO_MODE = old_algo
    
    agent.eval()
    return agent, env


@torch.no_grad()
def run_episode(env, seed, agent):
    obs, _ = env.reset(seed=seed)
    agent.reset()
    done = False
    ep_reward = 0.0
    MAX_STEPS = 500
    step_count = 0
    
    while not done and step_count < MAX_STEPS:
        result = agent.select_action(obs, deterministic=True)
        actions = result["actions"]
        obs, rewards, terminated, truncated, info = env.step(actions)
        ep_reward += sum(rewards)
        done = terminated or truncated
        step_count += 1
    
    return {
        "seed": seed,
        "episode_reward": ep_reward,
        "task_success_rate": info.get("task_success_rate", 0.0),
        "mean_cft": info.get("mean_cft", 0.0),
        "mean_energy": info.get("mean_energy", 0.0),
        "mean_interference": info.get("mean_interference", 0.0),
        "deadline_miss_rate": info.get("deadline_miss_rate", 0.0),
    }


def evaluate(name, model_path, algo_mode="mappo", seeds=range(9000, 9010)):
    print(f"\n评估 {name}...")
    print(f"  模型路径: {model_path}")
    print(f"  算法模式: {algo_mode}")
    
    agent, env = load_agent(model_path, algo_mode)
    rows = []
    
    for seed in seeds:
        result = run_episode(env, seed, agent)
        rows.append(result)
        print(f"  Seed {seed}: SR={result['task_success_rate']:.3f}, CFT={result['mean_cft']:.3f}")
    
    # 计算统计
    sr_vals = [r["task_success_rate"] for r in rows]
    cft_vals = [r["mean_cft"] for r in rows]
    energy_vals = [r["mean_energy"] for r in rows]
    dm_vals = [r["deadline_miss_rate"] for r in rows]
    
    summary = {
        "policy": name,
        "task_success_rate_mean": sum(sr_vals) / len(sr_vals),
        "task_success_rate_std": (sum((x - sum(sr_vals)/len(sr_vals))**2 for x in sr_vals) / len(sr_vals))**0.5,
        "mean_cft_mean": sum(cft_vals) / len(cft_vals),
        "mean_cft_std": (sum((x - sum(cft_vals)/len(cft_vals))**2 for x in cft_vals) / len(cft_vals))**0.5,
        "mean_energy_mean": sum(energy_vals) / len(energy_vals),
        "mean_energy_std": (sum((x - sum(energy_vals)/len(energy_vals))**2 for x in energy_vals) / len(energy_vals))**0.5,
        "deadline_miss_rate_mean": sum(dm_vals) / len(dm_vals),
    }
    
    return rows, summary


def main():
    out_dir = Path("runs/paper_final_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义所有要评估的模型
    models_to_eval = [
        # Group 2: 综合对比
        {
            "name": "F-MAPPO",
            "path": "runs/rc1_default_fmappo_20260328_224844/fmappo_flat",
            "algo": "mappo",
            "group": "group2_comprehensive_comparison",
        },
        {
            "name": "TERA-MAPPO",
            "path": "runs/lr_critic_1500ep_20260327_163712/lr_c5e4",
            "algo": "mappo",
            "group": "group2_comprehensive_comparison",
        },
        {
            "name": "IPPO-H",
            "path": "runs/rc1_batch1_part1_topology_20260323_182712/ippo_main",
            "algo": "ippo",
            "group": "group2_comprehensive_comparison",
        },
        # Group 3: 消融实验
        {
            "name": "TERA-MAPPO",
            "path": "runs/rc1_ablation_1500ep_20260322_180707/full",
            "algo": "mappo",
            "group": "group3_ablation",
        },
        {
            "name": "w/o-TDE",
            "path": "runs/rc1_ablation_1500ep_20260322_180707/wo_dag",
            "algo": "mappo",
            "group": "group3_ablation",
        },
        {
            "name": "w/o-CARE",
            "path": "runs/rc1_ablation_1500ep_20260322_180707/wo_resource",
            "algo": "mappo",
            "group": "group3_ablation",
        },
    ]
    
    all_results = {}
    
    for spec in models_to_eval:
        group_dir = out_dir / spec["group"] / "tables"
        group_dir.mkdir(parents=True, exist_ok=True)
        
        rows, summary = evaluate(
            spec["name"],
            spec["path"],
            spec["algo"]
        )
        
        # 保存详细结果
        csv_path = group_dir / f"{spec['name']}_episodes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        all_results[spec["name"]] = summary
        
        print(f"  结果已保存到: {csv_path}")
    
    # 保存各组汇总
    for group in ["group2_comprehensive_comparison", "group3_ablation"]:
        group_results = [r for n, r in all_results.items() 
                        if any(m["group"] == group and m["name"] == n for m in models_to_eval)]
        
        if group_results:
            summary_path = out_dir / group / "tables" / "summary.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=group_results[0].keys())
                writer.writeheader()
                writer.writerows(group_results)
            print(f"\n{group} 汇总已保存到: {summary_path}")
    
    print(f"\n所有评估完成！结果保存在 {out_dir}")


if __name__ == "__main__":
    main()
