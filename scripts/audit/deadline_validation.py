#!/usr/bin/env python3
"""
Deadline验证脚本 - 验证deadline参数是否满足可学习条件

验证指标:
- infeasible_frac: 先天不可行比例 (deadline < LB0)
- tightness_ratio: deadline / LB0 分布
- AlwaysLocal/AlwaysRSU成功率估计

使用方法:
    python scripts/audit/deadline_validation.py --episodes 100
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg
from utils.dag_generator import DAGGenerator


def compute_critical_path(adj, comp_arr):
    """计算DAG关键路径长度（最长路径计算量和）"""
    n = len(comp_arr)
    if n == 0:
        return 0.0
    indeg = np.sum(adj, axis=0).astype(int)
    order = []
    queue = [i for i in range(n) if indeg[i] == 0]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in np.where(adj[u] == 1)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(int(v))
    if len(order) != n:
        return float(np.sum(comp_arr))
    dp = np.zeros(n, dtype=float)
    for u in order:
        preds = np.where(adj[:, u] == 1)[0]
        if len(preds) == 0:
            dp[u] = comp_arr[u]
        else:
            dp[u] = comp_arr[u] + np.max(dp[preds])
    return float(np.max(dp))


def collect_deadline_samples(n_episodes):
    """采样DAG并收集deadline统计"""
    env = VecOffloadingEnv()
    samples = []
    
    f_median = (Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ) / 2.0
    f_max = max(Cfg.MAX_VEHICLE_CPU_FREQ, Cfg.F_RSU)
    
    for ep in range(n_episodes):
        env.reset()
        
        for veh in env.vehicles:
            dag = veh.task_dag
            comp_arr = dag.total_comp
            adj = dag.adj
            
            # 计算关键路径
            cp_total = compute_critical_path(adj, comp_arr)
            total_comp = np.sum(comp_arr)
            
            # 物理下界
            LB0 = cp_total / f_max
            
            # deadline相关
            deadline = dag.deadline
            slack = deadline - LB0
            tightness = deadline / max(LB0, 1e-9)
            infeasible = deadline < LB0
            
            # 估算本地串行时间
            t_local_serial = total_comp / veh.cpu_freq
            
            samples.append({
                'num_nodes': dag.num_subtasks,
                'total_comp': total_comp,
                'cp_total': cp_total,
                'cp_ratio': cp_total / max(total_comp, 1e-9),  # 关键路径占比
                'f_veh': veh.cpu_freq,
                'f_median': f_median,
                'f_max': f_max,
                'LB0': LB0,
                'deadline': deadline,
                'slack': slack,
                'tightness': tightness,
                'infeasible': infeasible,
                't_local_serial': t_local_serial,
                'local_feasible': t_local_serial <= deadline,
            })
    
    return samples


def analyze_samples(samples):
    """分析采样结果"""
    n = len(samples)
    
    # 基础统计
    infeasible_frac = np.mean([s['infeasible'] for s in samples])
    local_success_rate = np.mean([s['local_feasible'] for s in samples])
    
    # tightness分布
    tightness = np.array([s['tightness'] for s in samples])
    slack = np.array([s['slack'] for s in samples])
    cp_ratio = np.array([s['cp_ratio'] for s in samples])
    
    print("="*60)
    print("Deadline验证报告")
    print("="*60)
    
    print(f"\n当前参数:")
    print(f"  DEADLINE_MODE = {Cfg.DEADLINE_MODE}")
    print(f"  DEADLINE_TIGHTENING_MIN = {Cfg.DEADLINE_TIGHTENING_MIN}")
    print(f"  DEADLINE_TIGHTENING_MAX = {Cfg.DEADLINE_TIGHTENING_MAX}")
    print(f"  DEADLINE_LB_EPS = {getattr(Cfg, 'DEADLINE_LB_EPS', 0.05)}")
    print(f"  f_median = {(Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ)/2:.2e}")
    print(f"  f_max = {max(Cfg.MAX_VEHICLE_CPU_FREQ, Cfg.F_RSU):.2e}")
    
    print(f"\n采样数量: {n}")
    
    print(f"\n" + "="*60)
    print("验证指标")
    print("="*60)
    
    print(f"\n1. 可行性检查:")
    print(f"   infeasible_frac (deadline < LB0): {infeasible_frac*100:.1f}% (阈值≤10%)")
    pass_infeasible = infeasible_frac <= 0.10
    print(f"   状态: {'✓ 通过' if pass_infeasible else '✗ 未通过'}")
    
    print(f"\n2. 紧缩比分布 (tightness = deadline / LB0):")
    print(f"   p10:  {np.percentile(tightness, 10):.2f} (阈值≥1.1)")
    print(f"   p50:  {np.percentile(tightness, 50):.2f}")
    print(f"   p90:  {np.percentile(tightness, 90):.2f}")
    print(f"   mean: {np.mean(tightness):.2f}")
    pass_tightness = np.percentile(tightness, 10) >= 1.1
    print(f"   状态: {'✓ 通过' if pass_tightness else '✗ 未通过'}")
    
    print(f"\n3. 松弛时间分布 (slack = deadline - LB0):")
    print(f"   p10:  {np.percentile(slack, 10):.3f}s")
    print(f"   p50:  {np.percentile(slack, 50):.3f}s")
    print(f"   p90:  {np.percentile(slack, 90):.3f}s")
    
    print(f"\n4. 关键路径占比 (CP_total / total_comp):")
    print(f"   p10:  {np.percentile(cp_ratio, 10)*100:.1f}%")
    print(f"   p50:  {np.percentile(cp_ratio, 50)*100:.1f}%")
    print(f"   p90:  {np.percentile(cp_ratio, 90)*100:.1f}%")
    
    print(f"\n5. AlwaysLocal成功率估计:")
    print(f"   success_rate: {local_success_rate*100:.1f}% (目标区间20%~70%)")
    pass_local = 0.20 <= local_success_rate <= 0.70
    print(f"   状态: {'✓ 在可学习区间' if pass_local else '✗ 超出可学习区间'}")
    if local_success_rate > 0.70:
        print(f"   建议: deadline过松，降低DEADLINE_TIGHTENING_MAX")
    elif local_success_rate < 0.20:
        print(f"   建议: deadline过紧，提高DEADLINE_TIGHTENING_MIN")
    
    print(f"\n" + "="*60)
    print("总结")
    print("="*60)
    all_pass = pass_infeasible and pass_tightness and pass_local
    print(f"  可行性:       {'✓' if pass_infeasible else '✗'}")
    print(f"  紧缩比:       {'✓' if pass_tightness else '✗'}")
    print(f"  可学习区间:   {'✓' if pass_local else '✗'}")
    print(f"\n  总体结果: {'✓ 全部通过' if all_pass else '✗ 存在问题'}")
    
    if not all_pass:
        print(f"\n调整建议:")
        if not pass_infeasible:
            print(f"  - infeasible过高: 增大DEADLINE_LB_EPS或提高gamma下限")
        if not pass_tightness:
            print(f"  - p10(tightness)过低: 提高DEADLINE_TIGHTENING_MIN")
        if not pass_local:
            if local_success_rate > 0.70:
                print(f"  - 过松: 降低DEADLINE_TIGHTENING_MAX到{Cfg.DEADLINE_TIGHTENING_MAX*0.8:.1f}")
            else:
                print(f"  - 过紧: 提高DEADLINE_TIGHTENING_MIN到{Cfg.DEADLINE_TIGHTENING_MIN*1.2:.1f}")
    
    return all_pass


def print_sample_table(samples, n_show=10):
    """打印样本表格"""
    print(f"\n" + "="*60)
    print(f"采样明细 (前{n_show}个)")
    print("="*60)
    print(f"{'nodes':>5} {'CP_total':>10} {'total':>10} {'LB0':>8} {'deadline':>8} {'slack':>8} {'tight':>6} {'infeas':>6}")
    print("-"*70)
    
    for s in samples[:n_show]:
        print(f"{s['num_nodes']:>5} {s['cp_total']:.2e} {s['total_comp']:.2e} "
              f"{s['LB0']:>8.3f} {s['deadline']:>8.3f} {s['slack']:>8.3f} "
              f"{s['tightness']:>6.2f} {'Y' if s['infeasible'] else 'N':>6}")


def main():
    parser = argparse.ArgumentParser(description='Deadline验证脚本')
    parser.add_argument('--episodes', type=int, default=100, help='采样episode数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--show-table', action='store_true', help='显示采样明细表')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("正在采样...")
    samples = collect_deadline_samples(args.episodes)
    
    if args.show_table:
        print_sample_table(samples)
    
    passed = analyze_samples(samples)
    
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
