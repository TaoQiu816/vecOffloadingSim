#!/usr/bin/env python3
"""
验证 DT 与单次计算/传输事件时长量级匹配（不修改 DT）。
原则：T_exec = C/f、T_tx = D/R 应在约 0.5*DT ~ 2*DT 范围内。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from configs.exp_dynamic_config import apply_exp_dynamic_config

apply_exp_dynamic_config(Cfg, TC)
DT = float(getattr(Cfg, "DT", 0.1))

# 理论单子任务执行时间 T_exec = C / f
f_local_min = float(Cfg.MIN_VEHICLE_CPU_FREQ)
f_local_max = float(Cfg.MAX_VEHICLE_CPU_FREQ)
f_rsu = float(Cfg.F_RSU)
min_comp = float(Cfg.MIN_COMP)
max_comp = float(Cfg.MAX_COMP)

t_local_min = min_comp / f_local_max   # 最小计算量 @ 最高本地频率
t_local_max = max_comp / f_local_min   # 最大计算量 @ 最低本地频率
t_rsu_min = min_comp / f_rsu
t_rsu_max = max_comp / f_rsu

# 典型传输时间 T_tx = D/R，取典型速率约 3–5 Mbps（10MHz/5RB、中等 SNR）
R_nominal_bps = 4.0e6
min_data = float(Cfg.MIN_DATA)
max_data = float(Cfg.MAX_DATA)
t_tx_min = min_data / R_nominal_bps
t_tx_max = max_data / R_nominal_bps

print("=== DT 与事件时长匹配验证 (DT 未修改) ===\n")
print(f"  DT = {DT} s")
print(f"  目标: 单次事件时长 ≈ 0.5*DT ~ 2*DT 即 {0.5*DT:.2f}s ~ {2*DT:.2f}s\n")

print("  计算 (T_exec = C/f):")
print(f"    本地: T_min={t_local_min:.3f}s, T_max={t_local_max:.3f}s  [MIN_COMP={min_comp:.0e}, MAX_COMP={max_comp:.0e}, f_veh {f_local_min/1e9:.1f}~{f_local_max/1e9:.1f} GHz]")
print(f"    RSU:  T_min={t_rsu_min:.3f}s, T_max={t_rsu_max:.3f}s  [f_rsu={f_rsu/1e9:.0f} GHz]")
print("  传输 (T_tx = D/R, R≈4 Mbps):")
print(f"    T_min={t_tx_min:.3f}s, T_max={t_tx_max:.3f}s  [data {min_data:.0e}~{max_data:.0e} bits]")

ok = True
if t_local_min < 0.5 * DT or t_local_max > 2.5 * DT:
    print(f"\n  [注意] 本地 T_exec 部分超出 0.5*DT~2*DT 范围，可接受为 1~3 个时隙内完成")
if t_tx_min < 0.5 * DT or t_tx_max > 2.5 * DT:
    print(f"\n  [注意] 传输 T_tx 部分超出 0.5*DT~2*DT 范围，可接受为 1~3 个时隙内完成")

# 短 episode 采样
from envs.vec_offloading_env import VecOffloadingEnv
import numpy as np

env = VecOffloadingEnv()
obs, _ = env.reset(seed=42)
np.random.seed(42)
steps = 0
max_steps = 150
while steps < max_steps:
    actions = []
    for v in env.vehicles:
        dag = v.task_dag
        mask = dag.get_action_mask()
        sched = np.where(mask)[0]
        s_idx = int(np.random.choice(sched)) if len(sched) > 0 else 0
        t_idx = int(np.random.randint(0, Cfg.MAX_TARGETS))
        actions.append({"target": t_idx, "subtask": s_idx, "power": np.float32(0.5)})
    obs, _, term, trunc, info = env.step(actions)
    steps += 1
    if term or trunc:
        break

durations = getattr(env, "_episode_task_durations", [])
on_task = info.get("on_task_rate") or info.get("has_task_available_rate")
print(f"\n  短 episode ({steps} 步) 采样:")
print(f"    本集成功完成 DAG 数: {len(durations)}")
if durations:
    print(f"    单 DAG 完成时间 (elapsed): mean={np.mean(durations):.2f}s, p50={np.percentile(durations, 50):.2f}s")
print(f"    on_task_rate: {on_task}")
print("\n=== 验证结束 ===")
