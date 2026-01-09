#!/usr/bin/env python3
"""
参数验证脚本 - 基于最终审计方案
验证A/B/C/D四项指标是否通过阈值

使用方法:
    python scripts/audit/param_validation.py --episodes 50 --steps 50
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg


def collect_samples(env, n_episodes, n_steps):
    """采样收集时延分解数据"""
    samples = []
    power_samples = []
    deadline_samples = []
    
    for ep in range(n_episodes):
        env.reset()
        
        for step in range(n_steps):
            for veh in env.vehicles:
                dag = veh.task_dag
                if dag.is_finished or dag.is_failed:
                    continue
                
                # 找READY任务
                ready_tasks = [i for i in range(dag.num_subtasks) if dag.status[i] == 1]
                
                for task_id in ready_tasks:
                    rem_comp = dag.rem_comp[task_id]
                    rem_data = dag.rem_data[task_id]
                    
                    # t_local
                    t_comp_local = rem_comp / veh.cpu_freq
                    w_local = max(0, veh.fat_processor - env.time)
                    t_local = w_local + t_comp_local
                    
                    # t_rsu (使用rate_snapshot)
                    rsu = env.rsus[0]
                    rate_v2i = env.channel.compute_one_rate(
                        veh, rsu.position, 'V2I', env.time,
                        v2i_user_count=max(len(env.vehicles)//3, 1)
                    )
                    t_tx_rsu = rem_data / max(rate_v2i, 1e-6)
                    w_rsu = max(0, rsu.get_min_processor_fat() - env.time)
                    t_comp_rsu = rem_comp / rsu.cpu_freq
                    t_rsu = t_tx_rsu + w_rsu + t_comp_rsu
                    
                    # t_v2v_best
                    neighbors = [v for v in env.vehicles if v.id != veh.id]
                    t_v2v_list = []
                    for neighbor in neighbors:
                        dist = np.linalg.norm(veh.pos - neighbor.pos)
                        if dist > Cfg.V2V_RANGE:
                            continue
                        rate_v2v = env.channel.compute_one_rate(
                            veh, neighbor.pos, 'V2V', env.time
                        )
                        t_tx_v2v = rem_data / max(rate_v2v, 1e-6)
                        w_v2v = max(0, neighbor.fat_processor - env.time)
                        t_comp_v2v = rem_comp / neighbor.cpu_freq
                        t_v2v_list.append(t_tx_v2v + w_v2v + t_comp_v2v)
                    
                    t_v2v_best = min(t_v2v_list) if t_v2v_list else float('inf')
                    
                    # 时间分解
                    t_actual = min(t_rsu, t_v2v_best) if t_v2v_best < float('inf') else t_rsu
                    tx_frac = t_tx_rsu / max(t_actual, 1e-9) if t_actual == t_rsu else (t_v2v_list[0] if t_v2v_list else 0) / max(t_actual, 1e-9)
                    
                    samples.append({
                        't_local': t_local,
                        't_rsu': t_rsu,
                        't_v2v_best': t_v2v_best,
                        't_tx_rsu': t_tx_rsu,
                        'w_rsu': w_rsu,
                        't_comp_rsu': t_comp_rsu,
                        'tx_frac': tx_frac,
                    })
                    
                    # 功率有效性采样 (同state，P_min vs P_max)
                    rate_P_max = env.channel.compute_one_rate(
                        veh, rsu.position, 'V2I', env.time,
                        v2i_user_count=max(len(env.vehicles)//3, 1),
                        power_dbm_override=Cfg.TX_POWER_MAX_DBM
                    )
                    rate_P_min = env.channel.compute_one_rate(
                        veh, rsu.position, 'V2I', env.time,
                        v2i_user_count=max(len(env.vehicles)//3, 1),
                        power_dbm_override=Cfg.TX_POWER_MIN_DBM
                    )
                    if rate_P_max > 1e-6 and rate_P_min > 1e-6 and rem_data > 1e-6:
                        t_tx_ratio_v2i = (rem_data / rate_P_min) / (rem_data / rate_P_max)
                        power_samples.append({
                            't_tx_ratio_v2i': t_tx_ratio_v2i,
                            'rate_P_max': rate_P_max,
                            'rate_P_min': rate_P_min,
                        })
            
            # 随机动作推进
            actions = []
            for veh in env.vehicles:
                action = {
                    'target': np.random.randint(0, Cfg.MAX_TARGETS),
                    'power': np.float32(np.random.uniform(0, 1)),
                }
                actions.append(action)
            env.step(actions)
        
        # Deadline可行性采样
        for veh in env.vehicles:
            dag = veh.task_dag
            total_comp = np.sum(dag.total_comp)
            deadline = dag.deadline
            f_max = max(Cfg.MAX_VEHICLE_CPU_FREQ, Cfg.F_RSU)
            cp_min_time = total_comp / f_max
            deadline_samples.append({
                'deadline': deadline,
                'cp_min_time': cp_min_time,
                'ratio': deadline / max(cp_min_time, 1e-9),
            })
    
    return samples, power_samples, deadline_samples


def validate_A(samples):
    """验证A: 优势动作存在性"""
    print("\n" + "="*60)
    print("验证A: 优势动作存在性")
    print("="*60)
    
    t_locals = np.array([s['t_local'] for s in samples])
    t_rsus = np.array([s['t_rsu'] for s in samples])
    t_v2vs = np.array([s['t_v2v_best'] for s in samples])
    tx_fracs = np.array([s['tx_frac'] for s in samples])
    
    # 统计
    p_rsu_better = np.mean(t_rsus < t_locals)
    p_v2v_better = np.mean(t_v2vs < t_locals)
    median_rsu_adv = np.median(t_locals - t_rsus)
    median_v2v_adv = np.median(t_locals[t_v2vs < float('inf')] - t_v2vs[t_v2vs < float('inf')]) if np.any(t_v2vs < float('inf')) else 0
    mean_tx_frac = np.mean(tx_fracs)
    
    print(f"  P(t_rsu < t_local):           {p_rsu_better*100:.1f}% (阈值≥25%)")
    print(f"  P(t_v2v_best < t_local):      {p_v2v_better*100:.1f}% (阈值≥10%)")
    print(f"  median(t_local - t_rsu):      {median_rsu_adv:.3f}s (阈值≥0.15s)")
    print(f"  median(t_local - t_v2v_best): {median_v2v_adv:.3f}s (阈值≥0.05s)")
    print(f"  mean(tx_frac):                {mean_tx_frac*100:.1f}% (阈值<50%)")
    
    pass_A = (p_rsu_better >= 0.25 and p_v2v_better >= 0.10 and 
              median_rsu_adv >= 0.15 and mean_tx_frac < 0.50)
    print(f"\n  验证A结果: {'✓ 通过' if pass_A else '✗ 未通过'}")
    return pass_A


def validate_B(power_samples):
    """验证B: 功率有效性"""
    print("\n" + "="*60)
    print("验证B: 功率有效性")
    print("="*60)
    
    if not power_samples:
        print("  无功率样本")
        return False
    
    t_tx_ratios = np.array([s['t_tx_ratio_v2i'] for s in power_samples])
    rate_ratios = np.array([s['rate_P_max'] / s['rate_P_min'] for s in power_samples])
    
    median_t_tx_ratio = np.median(t_tx_ratios)
    median_rate_ratio = np.median(rate_ratios)
    
    print(f"  median(t_tx(P_min)/t_tx(P_max)) V2I: {median_t_tx_ratio:.2f} (阈值≥1.25)")
    print(f"  median(rate(P_max)/rate(P_min)):     {median_rate_ratio:.2f}")
    
    pass_B = median_t_tx_ratio >= 1.25
    print(f"\n  验证B结果: {'✓ 通过' if pass_B else '✗ 未通过'}")
    return pass_B


def validate_C(deadline_samples):
    """验证C: Deadline可行性"""
    print("\n" + "="*60)
    print("验证C: Deadline可行性")
    print("="*60)
    
    ratios = np.array([s['ratio'] for s in deadline_samples])
    infeasible_frac = np.mean(ratios < 1.0)
    p10_ratio = np.percentile(ratios, 10)
    
    print(f"  infeasible_frac (deadline < CP/f_max): {infeasible_frac*100:.1f}% (阈值≤15%)")
    print(f"  p10(deadline / (CP/f_max)):            {p10_ratio:.2f} (阈值≥1.1)")
    
    pass_C = infeasible_frac <= 0.15 and p10_ratio >= 1.1
    print(f"\n  验证C结果: {'✓ 通过' if pass_C else '✗ 未通过'}")
    return pass_C


def main():
    parser = argparse.ArgumentParser(description='参数验证脚本')
    parser.add_argument('--episodes', type=int, default=50, help='采样episode数')
    parser.add_argument('--steps', type=int, default=50, help='每episode采样步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("="*60)
    print("参数验证 - 基于最终审计方案")
    print("="*60)
    print(f"\n当前关键参数:")
    print(f"  MIN_COMP = {Cfg.MIN_COMP:.2e}, MAX_COMP = {Cfg.MAX_COMP:.2e}")
    print(f"  MIN_DATA = {Cfg.MIN_DATA:.2e}, MAX_DATA = {Cfg.MAX_DATA:.2e}")
    print(f"  MIN_VEH_FREQ = {Cfg.MIN_VEHICLE_CPU_FREQ:.2e}, MAX_VEH_FREQ = {Cfg.MAX_VEHICLE_CPU_FREQ:.2e}")
    print(f"  F_RSU = {Cfg.F_RSU:.2e}")
    print(f"  BW_V2I = {Cfg.BW_V2I:.2e}, BW_V2V = {Cfg.BW_V2V:.2e}")
    print(f"  TX_POWER: {Cfg.TX_POWER_MIN_DBM} ~ {Cfg.TX_POWER_MAX_DBM} dBm")
    print(f"  SNR_MAX_DB = {Cfg.SNR_MAX_DB}")
    print(f"  T_REF = {Cfg.T_REF}, REWARD_BETA = {Cfg.REWARD_BETA}")
    print(f"  DT = {Cfg.DT}, MAX_STEPS = {Cfg.MAX_STEPS}")
    
    print(f"\n采样配置: {args.episodes} episodes × {args.steps} steps")
    
    # 创建环境
    env = VecOffloadingEnv()
    
    # 采样
    print("\n正在采样...")
    samples, power_samples, deadline_samples = collect_samples(env, args.episodes, args.steps)
    print(f"  采集样本数: {len(samples)} (优势), {len(power_samples)} (功率), {len(deadline_samples)} (deadline)")
    
    # 验证
    pass_A = validate_A(samples)
    pass_B = validate_B(power_samples)
    pass_C = validate_C(deadline_samples)
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    all_pass = pass_A and pass_B and pass_C
    print(f"  验证A (优势存在性): {'✓' if pass_A else '✗'}")
    print(f"  验证B (功率有效性): {'✓' if pass_B else '✗'}")
    print(f"  验证C (可行性):     {'✓' if pass_C else '✗'}")
    print(f"\n  总体结果: {'✓ 全部通过' if all_pass else '✗ 存在未通过项'}")
    
    if not all_pass:
        print("\n  调整建议:")
        if not pass_A:
            print("    - 若P(t_rsu<t_local)低: 增大MIN_COMP/MAX_COMP或降低F_RSU")
            print("    - 若P(t_v2v<t_local)低: 扩大MAX_VEHICLE_CPU_FREQ或降低MAX_DATA")
            print("    - 若tx_frac过高: 降低MAX_DATA或增大BW_V2I")
        if not pass_B:
            print("    - 若功率无梯度: 增大SNR_MAX_DB或扩大功率范围")
        if not pass_C:
            print("    - 若infeasible过高: 增大DEADLINE_TIGHTENING_MAX")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
