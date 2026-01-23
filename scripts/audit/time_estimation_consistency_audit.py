#!/usr/bin/env python3
"""
Time Estimation Consistency Audit Script
==========================================
【只读审计】记录每次子任务调度的 t_est (估计完成时间) 与 t_real (真实完成耗时)，
输出误差分布及按动作类型统计。

运行示例:
    python scripts/audit/time_estimation_consistency_audit.py --seed 0 --episodes 20 --out out/audit_time

输出:
    - {out}_raw.csv: 每次调度明细
    - {out}_summary.json: 汇总统计
    - 控制台摘要

核心检验点:
    1. t_est (奖励/潜势使用的估计完成时间) vs t_real (真实完成耗时) 的偏差
    2. 按动作类型 (Local/RSU/V2V) 分桶分析
    3. 检测系统性偏差 (如 RSU 偏乐观)
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
    parser = argparse.ArgumentParser(description="Time Estimation Consistency Audit")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--out", type=str, default="out/audit_time", help="Output file prefix")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode (None=use config)")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "local_only", "rsu_only", "mixed"], 
                        help="Policy to use for action selection")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    return parser.parse_args()


def classify_action_type(target):
    """根据 target 分类动作类型"""
    if target is None or target == "Local" or target == 0:
        return "Local"
    if isinstance(target, tuple) and len(target) > 0 and target[0] == "RSU":
        return "RSU"
    if isinstance(target, int) and target >= 2:
        return "V2V"
    if target == 1:
        return "RSU"
    return "Other"


class TimeEstimationTracker:
    """追踪子任务调度与完成时间"""
    
    def __init__(self):
        # 存储进行中的调度: (veh_id, subtask_idx) -> record
        self.pending = {}
        # 完成的记录
        self.completed = []
        # 统计
        self.schedule_count = 0
        self.complete_count = 0
    
    def on_schedule(self, veh_id, subtask_idx, action_type, target, t_est, sim_time, 
                    cycles=None, t_tx=None, queue_wait=None, rate=None, power_ratio=None):
        """记录子任务调度事件"""
        key = (veh_id, subtask_idx)
        self.pending[key] = {
            "veh_id": veh_id,
            "subtask_idx": subtask_idx,
            "action_type": action_type,
            "target": str(target),
            "t_est": t_est,  # 估计的执行时间 (不包含调度时刻)
            "schedule_time": sim_time,  # 调度时刻
            "t_est_finish": sim_time + t_est if t_est is not None else None,  # 估计完成时刻
            "cycles": cycles,
            "t_tx": t_tx,
            "queue_wait": queue_wait,
            "rate": rate,
            "power_ratio": power_ratio,
        }
        self.schedule_count += 1
    
    def on_complete(self, veh_id, subtask_idx, sim_time, success=True):
        """记录子任务完成事件"""
        key = (veh_id, subtask_idx)
        if key not in self.pending:
            return None
        
        record = self.pending.pop(key)
        record["complete_time"] = sim_time
        record["success"] = success
        record["t_real"] = sim_time - record["schedule_time"]  # 真实执行时间
        
        # 计算误差
        if record["t_est"] is not None and record["t_real"] is not None:
            record["error"] = record["t_est"] - record["t_real"]  # 正值=高估，负值=低估
            record["error_ratio"] = record["error"] / max(record["t_real"], 1e-9)
        else:
            record["error"] = None
            record["error_ratio"] = None
        
        self.completed.append(record)
        self.complete_count += 1
        return record
    
    def get_pending_count(self):
        return len(self.pending)
    
    def get_completed_records(self):
        return self.completed


def run_audit(args):
    """执行审计，收集时间估计与真实值"""
    np.random.seed(args.seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    # 时间追踪器
    tracker = TimeEstimationTracker()
    
    # 收集的详细步骤记录（用于队列/mask分析）
    step_records = []
    
    print(f"[Audit] Starting time estimation consistency audit")
    print(f"[Audit] Seed={args.seed}, Episodes={args.episodes}, Policy={args.policy}")
    print(f"[Audit] Config: NUM_VEHICLES={Cfg.NUM_VEHICLES}, NUM_RSU={Cfg.NUM_RSU}, DT={Cfg.DT}")
    print("-" * 60)
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        step = 0
        
        while not done:
            # 根据策略选择动作
            actions = []
            for i, o in enumerate(obs):
                action_mask = o.get("action_mask", np.ones(Cfg.MAX_TARGETS))
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
                elif args.policy == "mixed":
                    # 均匀分布在 Local/RSU/V2V 之间
                    local_targets = [t for t in valid_targets if t == 0]
                    rsu_targets = [t for t in valid_targets if t == 1]
                    v2v_targets = [t for t in valid_targets if t >= 2]
                    
                    choices = []
                    if local_targets:
                        choices.append(("local", local_targets))
                    if rsu_targets:
                        choices.append(("rsu", rsu_targets))
                    if v2v_targets:
                        choices.append(("v2v", v2v_targets))
                    
                    if choices:
                        _, targets = choices[np.random.randint(len(choices))]
                        target = np.random.choice(targets)
                    else:
                        target = 0
                else:
                    target = 0
                
                power = np.random.uniform(0.3, 0.7)
                actions.append(np.array([target, power], dtype=np.float32))
            
            # 执行一步前，记录当前仿真时间
            sim_time_before = env.time
            
            # 执行一步
            obs, rewards, terminated, truncated, infos = env.step(actions)
            step += 1
            
            sim_time_after = env.time
            
            # 从 info 中提取调度信息
            for i, info in enumerate(infos):
                # 检测是否有新的子任务被调度
                subtask_idx = info.get("subtask_idx")
                if subtask_idx is None:
                    continue
                
                # 获取动作类型和目标
                target = info.get("target")
                action_type = classify_action_type(target)
                
                # 获取时间估计相关字段
                t_actual = info.get("t_actual")  # 这是 _estimate_t_actual 返回的估计
                t_tx = info.get("t_tx")
                cycles = info.get("cycles")
                power_ratio = float(actions[i][1])
                
                # 尝试获取队列等待和速率信息（如果 info 中有）
                queue_wait = info.get("queue_wait")
                rate = info.get("rate")
                
                # 只有当有有效的子任务和时间估计时才记录
                if subtask_idx is not None and t_actual is not None:
                    tracker.on_schedule(
                        veh_id=i,
                        subtask_idx=subtask_idx,
                        action_type=action_type,
                        target=target,
                        t_est=t_actual,
                        sim_time=sim_time_before,
                        cycles=cycles,
                        t_tx=t_tx,
                        queue_wait=queue_wait,
                        rate=rate,
                        power_ratio=power_ratio,
                    )
                
                # 检测子任务完成事件
                # 通过检查 DAG 状态变化或 info 中的完成标记
                completed_subtasks = info.get("completed_subtasks", [])
                for completed_idx in completed_subtasks:
                    tracker.on_complete(i, completed_idx, sim_time_after, success=True)
                
                # 记录步骤详情
                step_records.append({
                    "episode": ep,
                    "step": step,
                    "agent_id": i,
                    "sim_time": sim_time_after,
                    "action_type": action_type,
                    "target": str(target),
                    "subtask_idx": subtask_idx,
                    "t_actual": t_actual,
                    "t_tx": t_tx,
                    "cycles": cycles,
                    "power_ratio": power_ratio,
                    "illegal": info.get("illegal", False),
                })
            
            done = terminated or truncated
            if args.max_steps and step >= args.max_steps:
                break
        
        # Episode 结束时，清理未完成的任务（标记为 timeout/unknown）
        pending_keys = list(tracker.pending.keys())
        for key in pending_keys:
            veh_id, subtask_idx = key
            tracker.on_complete(veh_id, subtask_idx, env.time, success=False)
        
        completed = len(tracker.completed)
        pending = tracker.get_pending_count()
        print(f"[Ep {ep+1:3d}/{args.episodes}] Steps={step:3d}, Scheduled={tracker.schedule_count}, Completed={completed}, Pending={pending}")
    
    env.close()
    
    print("-" * 60)
    print(f"[Audit] Total scheduled: {tracker.schedule_count}")
    print(f"[Audit] Total completed: {tracker.complete_count}")
    
    return tracker.get_completed_records(), step_records


def compute_statistics(records, step_records):
    """计算统计数据"""
    if not records:
        return {}
    
    df = pd.DataFrame(records)
    
    result = {
        "config": {
            "NUM_VEHICLES": Cfg.NUM_VEHICLES,
            "NUM_RSU": Cfg.NUM_RSU,
            "DT": Cfg.DT,
            "MAX_TARGETS": Cfg.MAX_TARGETS,
            "T_REF": getattr(Cfg, "T_REF", "N/A"),
        },
        "total_records": len(df),
    }
    
    # 整体误差统计
    valid_error = df["error"].dropna()
    if len(valid_error) > 0:
        result["overall_error"] = {
            "count": int(len(valid_error)),
            "mean": float(valid_error.mean()),
            "std": float(valid_error.std()),
            "p5": float(valid_error.quantile(0.05)),
            "p25": float(valid_error.quantile(0.25)),
            "p50": float(valid_error.median()),
            "p75": float(valid_error.quantile(0.75)),
            "p95": float(valid_error.quantile(0.95)),
            "min": float(valid_error.min()),
            "max": float(valid_error.max()),
            "abs_mean": float(valid_error.abs().mean()),
            "abs_p95": float(valid_error.abs().quantile(0.95)),
        }
    
    # 整体误差比例统计
    valid_error_ratio = df["error_ratio"].dropna()
    if len(valid_error_ratio) > 0:
        result["overall_error_ratio"] = {
            "count": int(len(valid_error_ratio)),
            "mean": float(valid_error_ratio.mean()),
            "std": float(valid_error_ratio.std()),
            "p5": float(valid_error_ratio.quantile(0.05)),
            "p50": float(valid_error_ratio.median()),
            "p95": float(valid_error_ratio.quantile(0.95)),
            "abs_mean": float(valid_error_ratio.abs().mean()),
        }
    
    # 按动作类型分组
    result["by_action_type"] = {}
    for action_type in ["Local", "RSU", "V2V", "Other"]:
        subset = df[df["action_type"] == action_type]
        if len(subset) == 0:
            continue
        
        type_stats = {
            "count": int(len(subset)),
            "fraction": float(len(subset) / len(df)),
        }
        
        # t_est 分布
        t_est = subset["t_est"].dropna()
        if len(t_est) > 0:
            type_stats["t_est_mean"] = float(t_est.mean())
            type_stats["t_est_std"] = float(t_est.std())
            type_stats["t_est_p50"] = float(t_est.median())
            type_stats["t_est_p95"] = float(t_est.quantile(0.95))
        
        # t_real 分布
        t_real = subset["t_real"].dropna()
        if len(t_real) > 0:
            type_stats["t_real_mean"] = float(t_real.mean())
            type_stats["t_real_std"] = float(t_real.std())
            type_stats["t_real_p50"] = float(t_real.median())
            type_stats["t_real_p95"] = float(t_real.quantile(0.95))
        
        # error 分布
        error = subset["error"].dropna()
        if len(error) > 0:
            type_stats["error_mean"] = float(error.mean())
            type_stats["error_std"] = float(error.std())
            type_stats["error_p5"] = float(error.quantile(0.05))
            type_stats["error_p50"] = float(error.median())
            type_stats["error_p95"] = float(error.quantile(0.95))
            type_stats["error_abs_mean"] = float(error.abs().mean())
            type_stats["error_abs_p95"] = float(error.abs().quantile(0.95))
            
            # 偏差方向分析
            type_stats["overestimate_frac"] = float((error > 0).mean())  # 高估比例
            type_stats["underestimate_frac"] = float((error < 0).mean())  # 低估比例
        
        # error_ratio 分布
        error_ratio = subset["error_ratio"].dropna()
        if len(error_ratio) > 0:
            type_stats["error_ratio_mean"] = float(error_ratio.mean())
            type_stats["error_ratio_p50"] = float(error_ratio.median())
            type_stats["error_ratio_p95"] = float(error_ratio.quantile(0.95))
        
        result["by_action_type"][action_type] = type_stats
    
    # 成功率统计
    success_rate = df["success"].mean()
    result["success_rate"] = float(success_rate)
    
    # 按动作类型的成功率
    result["success_by_type"] = {}
    for action_type in ["Local", "RSU", "V2V"]:
        subset = df[df["action_type"] == action_type]
        if len(subset) > 0:
            result["success_by_type"][action_type] = float(subset["success"].mean())
    
    # 从 step_records 计算额外统计
    if step_records:
        step_df = pd.DataFrame(step_records)
        
        # t_actual 分布（按类型）
        result["t_actual_by_type"] = {}
        for action_type in ["Local", "RSU", "V2V"]:
            subset = step_df[(step_df["action_type"] == action_type) & (step_df["t_actual"].notna())]
            if len(subset) > 0:
                t_actual = subset["t_actual"]
                result["t_actual_by_type"][action_type] = {
                    "count": int(len(t_actual)),
                    "mean": float(t_actual.mean()),
                    "std": float(t_actual.std()),
                    "p50": float(t_actual.median()),
                    "p95": float(t_actual.quantile(0.95)),
                }
    
    return result


def print_summary(stats):
    """打印控制台摘要"""
    print("\n" + "=" * 80)
    print("TIME ESTIMATION CONSISTENCY AUDIT SUMMARY")
    print("=" * 80)
    
    print("\n[Config]")
    for k, v in stats.get("config", {}).items():
        print(f"  {k}: {v}")
    
    print(f"\n[Total Records]: {stats.get('total_records', 0)}")
    print(f"[Success Rate]: {stats.get('success_rate', 0)*100:.1f}%")
    
    print("\n[Overall Error Statistics (t_est - t_real)]")
    overall = stats.get("overall_error", {})
    if overall:
        print(f"  Count: {overall.get('count', 0)}")
        print(f"  Mean: {overall.get('mean', 0):.4f}s (正=高估, 负=低估)")
        print(f"  Std: {overall.get('std', 0):.4f}s")
        print(f"  P5/P50/P95: {overall.get('p5', 0):.4f} / {overall.get('p50', 0):.4f} / {overall.get('p95', 0):.4f}s")
        print(f"  Abs Mean: {overall.get('abs_mean', 0):.4f}s")
        print(f"  Abs P95: {overall.get('abs_p95', 0):.4f}s")
    
    print("\n[Error Statistics by Action Type]")
    by_type = stats.get("by_action_type", {})
    header = f"{'Type':<8} {'Count':>8} {'Frac%':>7} {'ErrMean':>10} {'ErrP50':>10} {'ErrP95':>10} {'Over%':>8}"
    print(header)
    print("-" * len(header))
    for action_type in ["Local", "RSU", "V2V", "Other"]:
        ts = by_type.get(action_type, {})
        if not ts:
            continue
        count = ts.get("count", 0)
        frac = ts.get("fraction", 0) * 100
        err_mean = ts.get("error_mean", 0)
        err_p50 = ts.get("error_p50", 0)
        err_p95 = ts.get("error_p95", 0)
        over_frac = ts.get("overestimate_frac", 0) * 100
        print(f"{action_type:<8} {count:>8} {frac:>6.1f}% {err_mean:>10.4f} {err_p50:>10.4f} {err_p95:>10.4f} {over_frac:>7.1f}%")
    
    print("\n[t_est vs t_real Comparison by Type]")
    header2 = f"{'Type':<8} {'t_est_mean':>12} {'t_real_mean':>12} {'Bias':>10}"
    print(header2)
    print("-" * len(header2))
    for action_type in ["Local", "RSU", "V2V"]:
        ts = by_type.get(action_type, {})
        if not ts:
            continue
        t_est_mean = ts.get("t_est_mean", 0)
        t_real_mean = ts.get("t_real_mean", 0)
        bias = t_est_mean - t_real_mean
        print(f"{action_type:<8} {t_est_mean:>12.4f} {t_real_mean:>12.4f} {bias:>10.4f}")
    
    print("\n[Success Rate by Type]")
    success_by = stats.get("success_by_type", {})
    for action_type in ["Local", "RSU", "V2V"]:
        rate = success_by.get(action_type, 0)
        print(f"  {action_type}: {rate*100:.1f}%")
    
    # 关键发现
    print("\n[Key Findings]")
    findings = []
    
    # 检查系统性偏差
    for action_type in ["Local", "RSU", "V2V"]:
        ts = by_type.get(action_type, {})
        if not ts:
            continue
        err_mean = ts.get("error_mean", 0)
        if abs(err_mean) > 0.1:  # 超过100ms的系统性偏差
            direction = "高估" if err_mean > 0 else "低估"
            findings.append(f"⚠️  {action_type} 存在系统性{direction}: mean_error={err_mean:.4f}s")
    
    # 检查高估/低估比例严重失衡
    for action_type in ["Local", "RSU", "V2V"]:
        ts = by_type.get(action_type, {})
        over_frac = ts.get("overestimate_frac", 0.5)
        if over_frac > 0.8:
            findings.append(f"⚠️  {action_type} 大部分被高估 ({over_frac*100:.1f}%)")
        elif over_frac < 0.2:
            findings.append(f"⚠️  {action_type} 大部分被低估 ({(1-over_frac)*100:.1f}%)")
    
    if findings:
        for f in findings:
            print(f"  {f}")
    else:
        print("  ✓ 未发现明显的系统性偏差")
    
    print("\n" + "=" * 80)


def main():
    args = parse_args()
    
    # 确保输出目录存在
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行审计
    records, step_records = run_audit(args)
    
    if not records:
        print("[Warning] No completed records collected!")
        print("[Note] This may be because subtask completion events are not tracked in info dict.")
        print("[Note] Consider using step_records for partial analysis.")
    
    # 保存原始数据
    if records:
        df = pd.DataFrame(records)
        raw_path = f"{args.out}_raw.csv"
        df.to_csv(raw_path, index=False)
        print(f"[Output] Raw data saved to: {raw_path}")
    
    if step_records:
        step_df = pd.DataFrame(step_records)
        step_path = f"{args.out}_steps.csv"
        step_df.to_csv(step_path, index=False)
        print(f"[Output] Step data saved to: {step_path}")
    
    # 计算统计
    stats = compute_statistics(records, step_records)
    stats["audit_params"] = {
        "seed": args.seed,
        "episodes": args.episodes,
        "policy": args.policy,
        "max_steps": args.max_steps,
    }
    
    # 保存汇总
    summary_path = f"{args.out}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"[Output] Summary saved to: {summary_path}")
    
    # 打印控制台摘要
    print_summary(stats)


if __name__ == "__main__":
    main()
