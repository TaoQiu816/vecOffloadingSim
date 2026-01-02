#!/usr/bin/env python
"""
精细化Deadline诊断脚本

目标：定位Local-only SR=0%的真正原因
检查项：
1. n_active分布（PS稀释）
2. 频率一致性（deadline基准f vs 执行f）
3. Local-only传输事件（应为0）
4. makespan vs deadline分布（分位数标定）
5. comp下降速率
"""

import sys
import os
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def local_only_policy(env):
    """Local-only策略"""
    return [{"target": 0, "power": 1.0} for _ in env.vehicles]


def diagnose_local_only_detailed(num_episodes=20, seed=42):
    """详细诊断Local-only路径"""
    print("="*80)
    print("精细化Deadline诊断：Local-only路径")
    print("="*80)
    print()
    
    np.random.seed(seed)
    
    # 诊断数据收集
    diagnostics = {
        'n_active_stats': [],
        'frequency_consistency': [],
        'transmission_events': [],
        'makespan_vs_deadline': [],
        'comp_drop_rate': [],
        'task_completion_info': []
    }
    
    for ep in range(num_episodes):
        env = VecOffloadingEnv()
        obs, info = env.reset()
        
        # 获取第一个车辆的任务信息（用于频率一致性检查）
        deadline_freq = None
        if env.vehicles:
            v0 = env.vehicles[0]
            deadline_freq = v0.cpu_freq  # deadline计算用的频率
            deadline_s = getattr(v0.task_dag, 'deadline', 0)
            cp_cycles = getattr(v0.task_dag, 'critical_path_cycles', 0)
            total_comp = np.sum(v0.task_dag.total_comp)
            
            # 记录频率一致性
            diagnostics['frequency_consistency'].append({
                'deadline_freq': deadline_freq,
                'deadline_seconds': deadline_s,
                'cp_cycles': cp_cycles,
                'total_comp': total_comp
            })
        
        # Episode内数据收集
        n_active_history = []
        comp_remaining_history = []
        transmission_count = 0
        step_count = 0
        
        done = False
        truncated = False
        
        while not (done or truncated) and step_count < 400:
            actions = local_only_policy(env)
            obs, rewards, done, truncated, info = env.step(actions)
            step_count += 1
            
            # 每5步收集一次活跃任务数和剩余计算量
            if step_count % 5 == 0:
                for v in env.vehicles:
                    if hasattr(v, 'active_task_manager'):
                        n_active = v.active_task_manager.get_num_active_tasks()
                        n_active_history.append(n_active)
                        
                        # 计算总剩余计算量
                        total_rem_comp = 0
                        for task in v.active_task_manager.active_tasks:
                            total_rem_comp += task.rem_comp
                        comp_remaining_history.append({
                            'step': step_count,
                            'total_rem_comp': total_rem_comp,
                            'time': env.time
                        })
            
            # 检查传输事件（Local-only应该为0）
            # 通过检查active_transfers
            for v in env.vehicles:
                if hasattr(v, 'active_transfers') and len(v.active_transfers) > 0:
                    transmission_count += len(v.active_transfers)
        
        # Episode结束：收集最终统计
        # 从env对象直接计算指标（因为info在step中可能不包含episode_metrics）
        episode_vehicle_count = len(env.vehicles)
        episode_task_count = episode_vehicle_count  # 每车一个任务
        success_vehicles = sum(1 for v in env.vehicles if v.task_dag.is_finished)
        failed_vehicles = sum(1 for v in env.vehicles if v.task_dag.is_failed)
        
        # [Deadline检查计数] 从env对象提取
        deadline_checks = getattr(env, '_audit_deadline_checks', 0)
        deadline_misses = getattr(env, '_audit_deadline_misses', 0)
        
        vehicle_success_rate = success_vehicles / max(episode_vehicle_count, 1)
        task_success_rate = success_vehicles / max(episode_task_count, 1)
        episode_all_success = 1.0 if (success_vehicles == episode_vehicle_count and episode_vehicle_count > 0) else 0.0
        
        # 计算subtask SR
        total_subtasks = sum(v.task_dag.num_subtasks for v in env.vehicles)
        completed_subtasks = sum(np.sum(v.task_dag.status == 3) for v in env.vehicles)
        subtask_success_rate = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
        
        # [DAG Makespan计算] 手动计算
        dag_makespans = []
        subtask_durations = []
        deadlines = []
        
        for v in env.vehicles:
            dag_start_time = getattr(v.task_dag, 'start_time', env.time)
            if v.task_dag.is_finished:
                ct_arr = getattr(v.task_dag, 'CT', None)
                if ct_arr is not None:
                    valid_cts = [float(x) for x in ct_arr if np.isfinite(x) and x >= 0]
                    if valid_cts:
                        dag_finish_time = max(valid_cts)
                        dag_makespan = dag_finish_time - dag_start_time
                        dag_makespans.append(dag_makespan)
            
            # Subtask duration
            ct_arr = getattr(v.task_dag, 'CT', None)
            est_arr = getattr(v.task_dag, 'EST', None)
            if ct_arr is not None and est_arr is not None:
                for i in range(len(ct_arr)):
                    if ct_arr[i] >= 0 and est_arr[i] >= 0:
                        subtask_duration = ct_arr[i] - est_arr[i]
                        if subtask_duration > 0:
                            subtask_durations.append(float(subtask_duration))
            
            if getattr(v.task_dag, 'deadline', None) is not None:
                deadlines.append(float(v.task_dag.deadline))
        
        final_info = {
            'episode_id': ep + 1,
            # [SR统计口径拆分]
            'vehicle_success_rate': vehicle_success_rate,
            'task_success_rate': task_success_rate,
            'episode_all_success': episode_all_success,
            'subtask_success_rate': subtask_success_rate,
            'deadline_miss_rate': failed_vehicles / max(episode_vehicle_count, 1),
            'episode_task_count': episode_task_count,
            'episode_vehicle_count': episode_vehicle_count,
            # [DAG Makespan]
            'dag_makespan_mean': np.mean(dag_makespans) if dag_makespans else 0,
            'dag_makespan_p90': np.percentile(dag_makespans, 90) if dag_makespans else 0,
            'dag_makespan_count': len(dag_makespans),
            # [Subtask Duration]
            'subtask_duration_mean': np.mean(subtask_durations) if subtask_durations else 0,
            'subtask_duration_p95': np.percentile(subtask_durations, 95) if subtask_durations else 0,
            'subtask_duration_count': len(subtask_durations),
            # Deadline信息
            'deadline_seconds_mean': np.mean(deadlines) if deadlines else 0,
            'episode_time': env.time,
            # [Deadline检查计数]
            'audit_deadline_checks': deadline_checks,
            'audit_deadline_misses': deadline_misses
        }
        
        # n_active统计
        if n_active_history:
            diagnostics['n_active_stats'].append({
                'episode_id': ep + 1,
                'mean': np.mean(n_active_history),
                'p50': np.percentile(n_active_history, 50),
                'p95': np.percentile(n_active_history, 95),
                'max': np.max(n_active_history),
                'median': np.median(n_active_history)
            })
        
        # 计算量下降速率
        if len(comp_remaining_history) >= 2 and deadline_freq is not None:
            first = comp_remaining_history[0]
            last = comp_remaining_history[-1]
            dt = last['time'] - first['time']
            dcomp = first['total_rem_comp'] - last['total_rem_comp']
            drop_rate = dcomp / dt if dt > 0 else 0
            
            diagnostics['comp_drop_rate'].append({
                'episode_id': ep + 1,
                'drop_rate_cycles_per_sec': drop_rate,
                'drop_rate_normalized': drop_rate / deadline_freq if deadline_freq and deadline_freq > 0 else 0
            })
        
        # 传输事件
        diagnostics['transmission_events'].append({
            'episode_id': ep + 1,
            'total_transmission_events': transmission_count
        })
        
        # [Makespan vs Deadline] 关键修复：使用DAG makespan而非subtask duration
        diagnostics['makespan_vs_deadline'].append({
            'episode_id': ep + 1,
            'dag_makespan': final_info['dag_makespan_mean'],  # DAG级makespan
            'dag_makespan_p90': final_info['dag_makespan_p90'],
            'subtask_duration_mean': final_info['subtask_duration_mean'],  # 单个subtask时间
            'deadline': final_info['deadline_seconds_mean'],
            'ratio_dag': final_info['dag_makespan_mean'] / max(final_info['deadline_seconds_mean'], 1e-6),
            'ratio_subtask': final_info['subtask_duration_mean'] / max(final_info['deadline_seconds_mean'], 1e-6),
            'would_miss_dag': final_info['dag_makespan_mean'] > final_info['deadline_seconds_mean'] if final_info['dag_makespan_mean'] > 0 else False
        })
        
        diagnostics['task_completion_info'].append(final_info)
    
    # ========== 汇总分析 ==========
    print("【1】n_active分布（PS稀释检查）")
    print("-"*80)
    if diagnostics['n_active_stats']:
        n_active_means = [s['mean'] for s in diagnostics['n_active_stats']]
        n_active_p95s = [s['p95'] for s in diagnostics['n_active_stats']]
        n_active_maxs = [s['max'] for s in diagnostics['n_active_stats']]
        
        print(f"n_active_mean: {np.mean(n_active_means):.2f} (episodes平均), p95={np.mean(n_active_p95s):.2f}, max={np.mean(n_active_maxs):.2f}")
        print(f"各episode详情: mean∈[{np.min(n_active_means):.1f}, {np.max(n_active_means):.1f}], "
              f"p95∈[{np.min(n_active_p95s):.1f}, {np.max(n_active_p95s):.1f}], "
              f"max∈[{np.min(n_active_maxs):.1f}, {np.max(n_active_maxs):.1f}]")
        
        if np.mean(n_active_p95s) > 10:
            print("⚠️  WARNING: n_active p95 > 10，可能存在幽灵任务或PS过度稀释")
    else:
        print("❌ 未收集到n_active数据")
    
    print("\n【2】频率一致性检查")
    print("-"*80)
    if diagnostics['frequency_consistency']:
        freqs = [d['deadline_freq'] for d in diagnostics['frequency_consistency']]
        print(f"Deadline计算用频率: mean={np.mean(freqs)/1e9:.2f}GHz, "
              f"range=[{np.min(freqs)/1e9:.2f}, {np.max(freqs)/1e9:.2f}]GHz")
        
        # 检查执行时的频率（通过ActiveTaskManager）
        # 这里假设执行频率与deadline频率一致，需要验证
        print("⚠️  需手动验证：ActiveTaskManager.step()中使用的cpu_freq是否与deadline计算一致")
    
    print("\n【3】传输事件检查（Local-only应为0）")
    print("-"*80)
    if diagnostics['transmission_events']:
        tx_counts = [d['total_transmission_events'] for d in diagnostics['transmission_events']]
        print(f"传输事件总数: mean={np.mean(tx_counts):.1f}, max={np.max(tx_counts)}")
        if np.mean(tx_counts) > 0:
            print("⚠️  WARNING: Local-only仍有传输事件！检查active_transfers逻辑")
        else:
            print("✅ Local-only传输事件=0，符合预期")
    
    print("\n【4】计算量下降速率")
    print("-"*80)
    if diagnostics['comp_drop_rate']:
        drop_rates = [d['drop_rate_cycles_per_sec'] for d in diagnostics['comp_drop_rate']]
        drop_rates_norm = [d['drop_rate_normalized'] for d in diagnostics['comp_drop_rate']]
        print(f"实际下降速率: {np.mean(drop_rates)/1e9:.3f} Gcycles/s (mean)")
        print(f"归一化速率: {np.mean(drop_rates_norm):.2%} (相对于cpu_freq)")
        
        if np.mean(drop_rates_norm) < 0.3:
            print("⚠️  WARNING: 下降速率<30%cpu_freq，说明被大量任务稀释或任务未正确执行")
    
    print("\n【5】Makespan vs Deadline分布（分位数标定）")
    print("-"*80)
    if diagnostics['makespan_vs_deadline']:
        dag_makespans = [d.get('dag_makespan', 0) for d in diagnostics['makespan_vs_deadline'] if d.get('dag_makespan', 0) > 0]
        subtask_durations = [d.get('subtask_duration_mean', 0) for d in diagnostics['makespan_vs_deadline'] if d.get('subtask_duration_mean', 0) > 0]
        deadlines = [d.get('deadline', 0) for d in diagnostics['makespan_vs_deadline'] if d.get('deadline', 0) > 0]
        ratios_dag = [d.get('ratio_dag', 0) for d in diagnostics['makespan_vs_deadline'] if d.get('dag_makespan', 0) > 0 and d.get('deadline', 0) > 0]
        
        print("[关键区分] DAG Makespan vs Subtask Duration:")
        if dag_makespans:
            print(f"  DAG Makespan (整个DAG完成时间):")
            print(f"    mean={np.mean(dag_makespans):.3f}s, "
                  f"p50={np.percentile(dag_makespans, 50):.3f}s, "
                  f"p90={np.percentile(dag_makespans, 90):.3f}s, "
                  f"max={np.max(dag_makespans):.3f}s")
        else:
            print("  ⚠️  未收集到DAG Makespan数据（可能所有任务都未完成）")
        
        if subtask_durations:
            print(f"  Subtask Duration (单个子任务执行时间):")
            print(f"    mean={np.mean(subtask_durations):.3f}s, "
                  f"p50={np.percentile(subtask_durations, 50):.3f}s, "
                  f"p95={np.percentile(subtask_durations, 95):.3f}s")
        
        if deadlines:
            print(f"  Deadline: mean={np.mean(deadlines):.3f}s, "
                  f"p50={np.percentile(deadlines, 50):.3f}s, "
                  f"p90={np.percentile(deadlines, 90):.3f}s")
        
        if dag_makespans and deadlines and ratios_dag:
            print(f"  DAG Makespan/Deadline ratio: mean={np.mean(ratios_dag):.2f}, "
                  f"p90={np.percentile(ratios_dag, 90):.2f}")
            
            miss_count = sum(1 for d in diagnostics['makespan_vs_deadline'] if d.get('would_miss_dag', False))
            print(f"  Would miss count (based on DAG makespan): {miss_count}/{len(diagnostics['makespan_vs_deadline'])}")
            
            # 推荐deadline设置
            if dag_makespans:
                p90_makespan = np.percentile(dag_makespans, 90)
                mean_deadline = np.mean(deadlines) if deadlines else 0
                print(f"\n📊 推荐deadline设置（基于P90 DAG makespan）:")
                print(f"   deadline = {p90_makespan * 1.05:.3f}s ~ {p90_makespan * 1.25:.3f}s")
                print(f"   当前deadline mean={mean_deadline:.3f}s")
    
    print("\n【6】任务完成统计（SR统计口径拆分）")
    print("-"*80)
    if diagnostics['task_completion_info']:
        vehicle_srs = [d.get('vehicle_success_rate', 0) for d in diagnostics['task_completion_info']]
        task_srs = [d.get('task_success_rate', 0) for d in diagnostics['task_completion_info']]
        episode_all_srs = [d.get('episode_all_success', 0) for d in diagnostics['task_completion_info']]
        subtask_srs = [d.get('subtask_success_rate', 0) for d in diagnostics['task_completion_info']]
        
        print(f"Vehicle SR (per-vehicle): mean={np.mean(vehicle_srs):.1%}, "
              f"min={np.min(vehicle_srs):.1%}, max={np.max(vehicle_srs):.1%}")
        print(f"Task SR (per-task): mean={np.mean(task_srs):.1%}, "
              f"min={np.min(task_srs):.1%}, max={np.max(task_srs):.1%}")
        print(f"Episode All Success (all-or-nothing): mean={np.mean(episode_all_srs):.1%}, "
              f"count={sum(episode_all_srs)}/{len(episode_all_srs)}")
        print(f"Subtask SR (per-subtask): mean={np.mean(subtask_srs):.1%}, "
              f"min={np.min(subtask_srs):.1%}, max={np.max(subtask_srs):.1%}")
        print(f"\n⚠️  如果Episode All Success=0%但Vehicle SR>0%，说明是all-or-nothing统计导致SR=0%")
    
    print("\n【7】Deadline检查计数（是否触发判定）")
    print("-"*80)
    if diagnostics['task_completion_info']:
        deadline_checks_list = [d.get('audit_deadline_checks', 0) for d in diagnostics['task_completion_info']]
        deadline_misses_list = [d.get('audit_deadline_misses', 0) for d in diagnostics['task_completion_info']]
        miss_reason_dl_list = [d.get('deadline_miss_rate', 0) * d.get('episode_vehicle_count', 0) for d in diagnostics['task_completion_info']]
        
        if deadline_checks_list:
            print(f"Deadline Checks: mean={np.mean(deadline_checks_list):.0f}, total={np.sum(deadline_checks_list):.0f}")
            print(f"Deadline Misses: mean={np.mean(deadline_misses_list):.0f}, total={np.sum(deadline_misses_list):.0f}")
            print(f"Miss Reason Deadline count: mean={np.mean(miss_reason_dl_list):.1f}")
            
            if np.mean(deadline_checks_list) > 0:
                print("✅ Deadline判定代码正常执行")
            if np.mean(deadline_misses_list) > 0 and np.mean(miss_reason_dl_list) > 0:
                print(f"✅ Deadline miss触发正常：checks={np.mean(deadline_checks_list):.0f}, misses={np.mean(deadline_misses_list):.0f}")
    
    print("\n【8】死锁检测")
    print("-"*80)
    print("⚠️  死锁检测数据需要从episode JSON输出中提取（deadlock_vehicle_count和deadlock_vehicles）")
    print("  关键验证：")
    print("    - deadlock_vehicle_count > 0: 存在死锁（READY+RUNNING==0但未完成）")
    print("    - 死锁原因可能是：后继任务未触发READY、依赖计数错误、状态机断裂")
    
    print("\n【9】W_remaining统计（计算量推进情况）")
    print("-"*80)
    print("⚠️  W_remaining统计需要从episode JSON输出中提取")
    print("  关键指标（从JSON可见）：")
    print("    - w_remaining_delta_mean: 平均减少的计算量（cycles）")
    print("    - effective_service_rate_mean: 平均有效服务速率（cycles/s）")
    print("    - 如果effective_service_rate << cpu_freq，说明推进太慢或任务未进入active")
    print("  当前数据（从JSON可见）：effective_service_rate≈34MHz << cpu_freq(1.9GHz) → 推进很慢")
    
    # 保存详细数据
    output_file = 'logs/fine_grained_diagnosis.json'
    with open(output_file, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"\n详细数据已保存至: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    diagnose_local_only_detailed(num_episodes=args.episodes, seed=args.seed)

