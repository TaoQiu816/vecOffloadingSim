#!/usr/bin/env python3
"""
Reward Term Magnitude Audit Script
===================================
【只读审计】收集每步每个 reward term 的值，输出 overall + 分动作类型统计。

运行示例:
    python scripts/audit/reward_term_magnitude_audit.py --seed 0 --episodes 20 --out out/audit_reward

输出:
    - {out}_raw.csv: 每步明细
    - {out}_summary.json: 汇总统计
    - 控制台摘要
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Reward Term Magnitude Audit")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--out", type=str, default="out/audit_reward", help="Output file prefix")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode (None=use config)")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "local_only", "rsu_only"], 
                        help="Policy to use for action selection")
    return parser.parse_args()


def classify_action_type(target):
    """根据 target 分类动作类型"""
    if target is None or target == "Local" or target == 0:
        return "Local"
    if isinstance(target, tuple) and len(target) > 0 and target[0] == "RSU":
        return "RSU"
    if isinstance(target, int) and target >= 2:
        # target_idx >= 2 通常是 V2V
        return "V2V"
    if target == 1:
        # target_idx == 1 是 RSU
        return "RSU"
    return "Other"


def run_audit(args):
    """执行审计，收集所有 reward term 数据
    
    数据来源：
    1. env._reward_stats: 环境内部的 RewardStats 统计器
    2. info['audit_step_info']: 每步审计信息
    3. 直接从 rewards 列表和 action 记录推断
    
    注意：DEBUG_PBRS_AUDIT=True 会导致代码 bug（active_agent_mask 被覆盖），故禁用。
    """
    np.random.seed(args.seed)
    
    # 确保 DEBUG_PBRS_AUDIT=False 以避免现有代码 bug
    Cfg.DEBUG_PBRS_AUDIT = False
    
    # 创建环境
    env = VecOffloadingEnv()
    
    # 收集的数据列表（每步每个 agent 一条记录）
    records = []
    
    # 统计变量
    total_steps = 0
    total_decisions = 0
    
    print(f"[Audit] Starting reward term magnitude audit")
    print(f"[Audit] Seed={args.seed}, Episodes={args.episodes}, Policy={args.policy}")
    print(f"[Audit] Config: NUM_VEHICLES={Cfg.NUM_VEHICLES}, NUM_RSU={Cfg.NUM_RSU}, DT={Cfg.DT}")
    print(f"[Audit] Reward Scheme: {getattr(Cfg, 'REWARD_SCHEME', 'N/A')}")
    print("-" * 60)
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        step = 0
        ep_rewards = []
        
        while not done:
            # 根据策略选择动作
            actions = []
            action_types = []
            action_masks = []  # 保存每个 agent 的 action mask
            for i, o in enumerate(obs):
                action_mask = o.get("action_mask", np.ones(Cfg.MAX_TARGETS))
                action_masks.append(action_mask.copy())
                valid_targets = np.where(action_mask > 0)[0]
                
                if args.policy == "random":
                    if len(valid_targets) > 0:
                        target = np.random.choice(valid_targets)
                    else:
                        target = 0
                elif args.policy == "local_only":
                    target = 0
                elif args.policy == "rsu_only":
                    if 1 in valid_targets:
                        target = 1
                    else:
                        target = 0
                else:
                    target = 0
                
                if target == 0:
                    action_type = "Local"
                elif target == 1:
                    action_type = "RSU"
                else:
                    action_type = "V2V"
                action_types.append(action_type)
                
                power = np.random.uniform(0.3, 0.7)
                actions.append(np.array([target, power], dtype=np.float32))
            
            # 执行一步
            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1
            total_steps += 1
            
            # 从 info 中提取审计信息
            audit_info = info.get("audit_step_info", {})
            decision_step_mask = info.get("decision_step_mask", [])
            active_agent_mask = info.get("active_agent_mask", [])
            
            # 记录每个 agent 的数据
            for i in range(len(rewards)):
                target_idx = int(actions[i][0])
                power_ratio = float(actions[i][1])
                action_type = action_types[i]
                
                # 判断是否是决策步骤
                is_decision = False
                if i < len(decision_step_mask):
                    is_decision = bool(decision_step_mask[i])
                elif i < len(active_agent_mask):
                    is_decision = bool(active_agent_mask[i])
                
                # mask 信息
                mask = action_masks[i] if i < len(action_masks) else np.ones(Cfg.MAX_TARGETS)
                mask_rsu = bool(mask[1]) if len(mask) > 1 else False
                mask_v2v_count = int(np.sum(mask[2:])) if len(mask) > 2 else 0
                
                record = {
                    "episode": ep,
                    "step": step,
                    "agent_id": i,
                    "action_type": action_type,
                    "target_idx": target_idx,
                    "power_ratio": power_ratio,
                    "total_reward": float(rewards[i]),
                    "is_decision_step": is_decision,
                    "mask_rsu": mask_rsu,
                    "mask_v2v_count": mask_v2v_count,
                }
                records.append(record)
                
                if is_decision:
                    total_decisions += 1
            
            ep_rewards.extend([float(r) for r in rewards])
            
            done = terminated or truncated
            if args.max_steps and step >= args.max_steps:
                break
        
        # Episode 结束后，从 _reward_stats 提取汇总统计
        reward_stats = getattr(env, '_reward_stats', None)
        
        mean_r = np.mean(ep_rewards) if ep_rewards else 0.0
        decision_count = sum(1 for r in records if r["episode"] == ep and r.get("is_decision_step", False))
        print(f"[Ep {ep+1:3d}/{args.episodes}] Steps={step:3d}, Decisions={decision_count:4d}, MeanReward={mean_r:.4f}")
    
    # Episode 循环结束后，从 _reward_stats 提取聚合统计
    reward_stats = getattr(env, '_reward_stats', None)
    agg_stats = {}
    if reward_stats is not None:
        for name, bucket in reward_stats.metrics.items():
            if bucket.count > 0:
                agg_stats[name] = {
                    "count": bucket.count,
                    "mean": bucket.sum / bucket.count,
                    "min": bucket.min,
                    "max": bucket.max,
                }
    
    env.close()
    
    print("-" * 60)
    print(f"[Audit] Total steps: {total_steps}, Total decisions: {total_decisions}")
    
    # 将聚合统计附加到返回值
    return records, agg_stats


def compute_statistics(records):
    """计算统计数据"""
    if not records:
        return {}
    
    df = pd.DataFrame(records)
    
    # 只保留 is_decision_step=True 的记录（真正做了决策的）
    decision_df = df[df["is_decision_step"] == True] if "is_decision_step" in df.columns else df
    
    # 定义要统计的 reward terms
    reward_terms = ["r_lat", "r_shape", "r_timeout", "r_energy", "r_power", "r_term", "r_illegal", "total_reward"]
    
    def compute_term_stats(data, prefix=""):
        """计算单个组的统计"""
        stats = {}
        for term in reward_terms:
            if term not in data.columns:
                continue
            values = data[term].dropna()
            if len(values) == 0:
                continue
            stats[f"{prefix}{term}_count"] = int(len(values))
            stats[f"{prefix}{term}_mean"] = float(values.mean())
            stats[f"{prefix}{term}_std"] = float(values.std())
            stats[f"{prefix}{term}_p50"] = float(values.median())
            stats[f"{prefix}{term}_p95"] = float(values.quantile(0.95))
            stats[f"{prefix}{term}_min"] = float(values.min())
            stats[f"{prefix}{term}_max"] = float(values.max())
            stats[f"{prefix}{term}_nonzero_count"] = int((values.abs() > 1e-9).sum())
            stats[f"{prefix}{term}_nonzero_frac"] = float((values.abs() > 1e-9).mean())
            stats[f"{prefix}{term}_abs_mean"] = float(values.abs().mean())
        return stats
    
    result = {
        "config": {
            "NUM_VEHICLES": Cfg.NUM_VEHICLES,
            "NUM_RSU": Cfg.NUM_RSU,
            "DT": Cfg.DT,
            "MAX_TARGETS": Cfg.MAX_TARGETS,
            "REWARD_SCHEME": getattr(Cfg, "REWARD_SCHEME", "N/A"),
            "T_REF": getattr(Cfg, "T_REF", "N/A"),
            "LAT_ALPHA": getattr(Cfg, "LAT_ALPHA", "N/A"),
            "ENERGY_LAMBDA": getattr(Cfg, "ENERGY_LAMBDA", "N/A"),
            "POWER_LAMBDA": getattr(Cfg, "POWER_LAMBDA", "N/A"),
        },
        "total_records": len(df),
        "decision_records": len(decision_df),
        "overall": compute_term_stats(decision_df, prefix="overall_"),
        "by_action_type": {},
    }
    
    # 按动作类型分组统计（只统计 decision 记录）
    for action_type in ["Local", "RSU", "V2V", "Other"]:
        subset = decision_df[decision_df["action_type"] == action_type]
        if len(subset) == 0:
            continue
        result["by_action_type"][action_type] = compute_term_stats(subset, prefix=f"{action_type}_")
        result["by_action_type"][action_type]["count"] = int(len(subset))
        result["by_action_type"][action_type]["fraction"] = float(len(subset) / len(decision_df)) if len(decision_df) > 0 else 0.0
    
    # delta_phi 分布统计
    if "delta_phi" in decision_df.columns:
        valid_delta_phi = decision_df["delta_phi"].dropna()
        if len(valid_delta_phi) > 0:
            result["delta_phi_overall"] = {
                "mean": float(valid_delta_phi.mean()),
                "std": float(valid_delta_phi.std()),
                "p5": float(valid_delta_phi.quantile(0.05)),
                "p50": float(valid_delta_phi.median()),
                "p95": float(valid_delta_phi.quantile(0.95)),
                "min": float(valid_delta_phi.min()),
                "max": float(valid_delta_phi.max()),
            }
        
        # 按动作类型的 delta_phi 分布
        result["delta_phi_by_action"] = {}
        for action_type in ["Local", "RSU", "V2V"]:
            subset = decision_df[decision_df["action_type"] == action_type]["delta_phi"].dropna()
            if len(subset) > 0:
                result["delta_phi_by_action"][action_type] = {
                    "count": int(len(subset)),
                    "mean": float(subset.mean()),
                    "std": float(subset.std()),
                    "p50": float(subset.median()),
                    "p5": float(subset.quantile(0.05)),
                    "p95": float(subset.quantile(0.95)),
                }
    
    # f_max 分布统计（用于验证潜势计算）
    if "f_max" in decision_df.columns:
        f_max_vals = decision_df["f_max"].dropna()
        if len(f_max_vals) > 0:
            result["f_max_stats"] = {
                "mean": float(f_max_vals.mean()),
                "std": float(f_max_vals.std()),
                "min": float(f_max_vals.min()),
                "max": float(f_max_vals.max()),
            }
    
    # is_no_task_step 统计
    if "is_no_task_step" in df.columns:
        no_task_count = int(df["is_no_task_step"].sum())
        result["no_task_stats"] = {
            "no_task_count": no_task_count,
            "no_task_frac": float(no_task_count / len(df)) if len(df) > 0 else 0.0,
        }
    
    return result


def print_summary(stats):
    """打印控制台摘要"""
    print("\n" + "=" * 80)
    print("REWARD TERM MAGNITUDE AUDIT SUMMARY")
    print("=" * 80)
    
    print("\n[Config]")
    for k, v in stats.get("config", {}).items():
        print(f"  {k}: {v}")
    
    print(f"\n[Record Counts]")
    print(f"  Total records: {stats.get('total_records', 0)}")
    print(f"  Decision records: {stats.get('decision_records', 0)}")
    
    # 从 reward_stats_aggregate 打印 reward term 统计
    agg = stats.get("reward_stats_aggregate", {})
    if agg:
        print("\n[Reward Term Statistics from _reward_stats (aggregated)]")
        terms = ["r_lat", "r_shape", "r_timeout", "r_energy", "r_power", "r_term", "r_illegal", "r_total"]
        header = f"{'Term':<15} {'Count':>10} {'Mean':>12} {'Min':>12} {'Max':>12}"
        print(header)
        print("-" * len(header))
        for term in terms:
            term_stats = agg.get(term, {})
            if term_stats:
                cnt = term_stats.get("count", 0)
                mean = term_stats.get("mean", 0.0)
                min_v = term_stats.get("min", 0.0)
                max_v = term_stats.get("max", 0.0)
                print(f"{term:<15} {cnt:>10} {mean:>12.4f} {min_v:>12.4f} {max_v:>12.4f}")
        
        # 额外打印 latency 相关
        lat_terms = ["t_L", "t_R", "t_V", "t_a", "t_alt", "A_t"]
        print("\n[Latency Estimation Statistics]")
        header2 = f"{'Term':<15} {'Count':>10} {'Mean':>12} {'Min':>12} {'Max':>12}"
        print(header2)
        print("-" * len(header2))
        for term in lat_terms:
            term_stats = agg.get(term, {})
            if term_stats:
                cnt = term_stats.get("count", 0)
                mean = term_stats.get("mean", 0.0)
                min_v = term_stats.get("min", 0.0)
                max_v = term_stats.get("max", 0.0)
                print(f"{term:<15} {cnt:>10} {mean:>12.4f} {min_v:>12.4f} {max_v:>12.4f}")
    
    print("\n[By Action Type - from raw records]")
    by_type = stats.get("by_action_type", {})
    for action_type in ["Local", "RSU", "V2V"]:
        type_stats = by_type.get(action_type, {})
        if not type_stats:
            continue
        count = type_stats.get("count", 0)
        frac = type_stats.get("fraction", 0.0)
        print(f"\n  {action_type}: {count} decisions ({frac*100:.1f}%)")
        mean_r = type_stats.get(f"{action_type}_total_reward_mean", 0.0)
        print(f"    total_reward mean: {mean_r:.4f}")
    
    print("\n[delta_phi (PBRS) Distribution]")
    delta_phi = stats.get("delta_phi_overall", {})
    if delta_phi:
        print(f"  Overall: mean={delta_phi.get('mean', 0):.4f}, p50={delta_phi.get('p50', 0):.4f}, p5={delta_phi.get('p5', 0):.4f}, p95={delta_phi.get('p95', 0):.4f}")
    
    delta_phi_by_action = stats.get("delta_phi_by_action", {})
    for action_type in ["Local", "RSU", "V2V"]:
        dp = delta_phi_by_action.get(action_type, {})
        if dp:
            print(f"  {action_type}: mean={dp.get('mean', 0):.4f}, p50={dp.get('p50', 0):.4f}, count={dp.get('count', 0)}")
    
    print("\n[f_max Statistics]")
    f_max = stats.get("f_max_stats", {})
    if f_max:
        print(f"  Mean: {f_max.get('mean', 0):.2e} Hz")
        print(f"  Min/Max: {f_max.get('min', 0):.2e} / {f_max.get('max', 0):.2e} Hz")
    
    print("\n[No Task Step Statistics]")
    no_task = stats.get("no_task_stats", {})
    if no_task:
        print(f"  No task count: {no_task.get('no_task_count', 0)}")
        print(f"  No task fraction: {no_task.get('no_task_frac', 0)*100:.1f}%")
    
    print("\n" + "=" * 80)


def main():
    args = parse_args()
    
    # 确保输出目录存在
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行审计
    records, agg_stats = run_audit(args)
    
    if not records:
        print("[Warning] No records collected!")
        return
    
    # 保存原始数据
    df = pd.DataFrame(records)
    raw_path = f"{args.out}_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"[Output] Raw data saved to: {raw_path}")
    
    # 计算统计
    stats = compute_statistics(records)
    stats["audit_params"] = {
        "seed": args.seed,
        "episodes": args.episodes,
        "policy": args.policy,
        "max_steps": args.max_steps,
    }
    
    # 添加从 _reward_stats 提取的聚合统计
    stats["reward_stats_aggregate"] = agg_stats
    
    # 保存汇总
    summary_path = f"{args.out}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"[Output] Summary saved to: {summary_path}")
    
    # 打印控制台摘要
    print_summary(stats)


if __name__ == "__main__":
    main()
