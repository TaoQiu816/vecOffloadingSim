#!/usr/bin/env python3
"""
反向场景回归测试：确保护栏不误伤Local↔RSU与V2V传输

测试目标：
- 场景A：RSU-only单车，验证Local→RSU传输正常创建
- 场景B：RSU-only多车，验证多车同时Offload RSU正常
- 场景C：强制V2V，验证V2V生命周期和传输创建

运行：python scripts/regression_tx_sanity_suite.py --seed 42
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg
import numpy as np


def run_scenario_rsu_only_single(episodes=10, seed=42, verbose=True):
    """场景A：RSU-only单车"""
    original_min = Cfg.DEADLINE_TIGHTENING_MIN
    original_max = Cfg.DEADLINE_TIGHTENING_MAX
    Cfg.DEADLINE_TIGHTENING_MIN = 3.0  # 宽松deadline
    Cfg.DEADLINE_TIGHTENING_MAX = 5.0
    
    try:
        env = VecOffloadingEnv()
        results = []
        
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            
            # 强制所有可卸载任务走RSU（target_id=1）
            while not (done or truncated):
                actions = []
                for v in env.vehicles:
                    dag = v.task_dag
                    ready_tasks = np.where(dag.status == 1)[0]
                    if len(ready_tasks) > 0:
                        subtask_idx = int(ready_tasks[0])
                    else:
                        subtask_idx = 0
                    
                    # 强制RSU（如果RSU可用）
                    actions.append({
                        'target_id': 1,  # RSU
                        'power_ratio': 1.0,
                        'subtask_index': subtask_idx
                    })
                
                obs, rewards, done, truncated, info = env.step(actions)
            
            # 收集结果
            completed_vehicles = sum(1 for v in env.vehicles if v.task_dag.is_finished)
            total_vehicles = len(env.vehicles)
            vehicle_sr = completed_vehicles / max(total_vehicles, 1)
            
            total_tx_created = sum(getattr(v.task_dag, '_tx_tasks_created_count', 0) for v in env.vehicles)
            total_same_node_no_tx = sum(getattr(v.task_dag, '_same_node_no_tx_count', 0) for v in env.vehicles)
            
            deadlock_count = 0
            for v in env.vehicles:
                dag = v.task_dag
                count_completed = int(np.sum(dag.status == 3))
                count_ready = int(np.sum(dag.status == 1))
                count_running = int(np.sum(dag.status == 2))
                total_dag_subtasks = dag.num_subtasks
                is_deadlocked = (count_ready == 0 and count_running == 0 and 
                                count_completed < total_dag_subtasks)
                if is_deadlocked:
                    deadlock_count += 1
            
            results.append({
                'vehicle_sr': vehicle_sr,
                'tx_created': total_tx_created,
                'same_node_no_tx': total_same_node_no_tx,
                'deadlock_count': deadlock_count,
                'deadline_misses': env._audit_deadline_misses
            })
            
            if verbose and ep < 3:
                print(f"  Episode {ep+1}: SR={vehicle_sr:.2%}, tx={total_tx_created}, deadlock={deadlock_count}")
        
        avg_sr = np.mean([r['vehicle_sr'] for r in results])
        avg_tx = np.mean([r['tx_created'] for r in results])
        total_same_node = sum([r['same_node_no_tx'] for r in results])
        total_deadlock = sum([r['deadlock_count'] for r in results])
        
        return {
            'name': '场景A: RSU-only单车',
            'passed': True,
            'warnings': [],
            'vehicle_sr': avg_sr,
            'tx_created_mean': avg_tx,
            'same_node_no_tx_total': total_same_node,
            'deadlock_count': total_deadlock,
            'deadline_misses_mean': np.mean([r['deadline_misses'] for r in results])
        }
    finally:
        Cfg.DEADLINE_TIGHTENING_MIN = original_min
        Cfg.DEADLINE_TIGHTENING_MAX = original_max


def run_scenario_rsu_only_multi(episodes=10, seed=42, verbose=True):
    """场景B：RSU-only多车"""
    original_min = Cfg.DEADLINE_TIGHTENING_MIN
    original_max = Cfg.DEADLINE_TIGHTENING_MAX
    Cfg.DEADLINE_TIGHTENING_MIN = 3.0
    Cfg.DEADLINE_TIGHTENING_MAX = 5.0
    
    try:
        env = VecOffloadingEnv()
        results = []
        
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            
            while not (done or truncated):
                actions = []
                for v in env.vehicles:
                    dag = v.task_dag
                    ready_tasks = np.where(dag.status == 1)[0]
                    if len(ready_tasks) > 0:
                        subtask_idx = int(ready_tasks[0])
                    else:
                        subtask_idx = 0
                    
                    actions.append({
                        'target_id': 1,  # RSU
                        'power_ratio': 1.0,
                        'subtask_index': subtask_idx
                    })
                
                obs, rewards, done, truncated, info = env.step(actions)
            
            # 收集结果（同场景A）
            completed_vehicles = sum(1 for v in env.vehicles if v.task_dag.is_finished)
            total_vehicles = len(env.vehicles)
            vehicle_sr = completed_vehicles / max(total_vehicles, 1)
            
            total_tx_created = sum(getattr(v.task_dag, '_tx_tasks_created_count', 0) for v in env.vehicles)
            
            deadlock_count = 0
            for v in env.vehicles:
                dag = v.task_dag
                count_completed = int(np.sum(dag.status == 3))
                count_ready = int(np.sum(dag.status == 1))
                count_running = int(np.sum(dag.status == 2))
                total_dag_subtasks = dag.num_subtasks
                is_deadlocked = (count_ready == 0 and count_running == 0 and 
                                count_completed < total_dag_subtasks)
                if is_deadlocked:
                    deadlock_count += 1
            
            results.append({
                'vehicle_sr': vehicle_sr,
                'tx_created': total_tx_created,
                'deadlock_count': deadlock_count
            })
            
            if verbose and ep < 3:
                print(f"  Episode {ep+1}: SR={vehicle_sr:.2%}, tx={total_tx_created}, deadlock={deadlock_count}")
        
        avg_sr = np.mean([r['vehicle_sr'] for r in results])
        avg_tx = np.mean([r['tx_created'] for r in results])
        total_deadlock = sum([r['deadlock_count'] for r in results])
        
        return {
            'name': '场景B: RSU-only多车',
            'passed': True,
            'warnings': [],
            'vehicle_sr': avg_sr,
            'tx_created_mean': avg_tx,
            'deadlock_count': total_deadlock
        }
    finally:
        Cfg.DEADLINE_TIGHTENING_MIN = original_min
        Cfg.DEADLINE_TIGHTENING_MAX = original_max


def run_scenario_forced_v2v(episodes=10, seed=42, verbose=True):
    """场景C：强制V2V（车辆0 -> 车辆1）"""
    original_min = Cfg.DEADLINE_TIGHTENING_MIN
    original_max = Cfg.DEADLINE_TIGHTENING_MAX
    Cfg.DEADLINE_TIGHTENING_MIN = 3.0
    Cfg.DEADLINE_TIGHTENING_MAX = 5.0
    
    try:
        env = VecOffloadingEnv()
        results = []
        
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            
            while not (done or truncated):
                actions = []
                for v_idx, v in enumerate(env.vehicles):
                    dag = v.task_dag
                    ready_tasks = np.where(dag.status == 1)[0]
                    if len(ready_tasks) > 0:
                        subtask_idx = int(ready_tasks[0])
                    else:
                        subtask_idx = 0
                    
                    # 强制V2V：车辆0发送给车辆1，其他车辆Local
                    if v_idx == 0 and len(env.vehicles) > 1:
                        target_id = env.vehicles[1].id  # 车辆1的ID
                    else:
                        target_id = 0  # Local
                    
                    actions.append({
                        'target_id': target_id,
                        'power_ratio': 1.0,
                        'subtask_index': subtask_idx
                    })
                
                obs, rewards, done, truncated, info = env.step(actions)
            
            # 收集结果
            v2v_tx_started = info.get('audit_v2v_tx_started', 0)
            v2v_added_to_active = info.get('audit_v2v_added_to_active', 0)
            v2v_cpu_finished = info.get('audit_v2v_cpu_finished', 0)
            v2v_dag_completed = info.get('audit_v2v_dag_completed', 0)
            
            total_tx_created = sum(getattr(v.task_dag, '_tx_tasks_created_count', 0) for v in env.vehicles)
            
            # V2V生命周期守恒检查
            lifecycle_breach = False
            if v2v_tx_started > 0:
                max_count = max(v2v_tx_started, v2v_added_to_active, v2v_cpu_finished, v2v_dag_completed)
                min_count = min(v2v_tx_started, v2v_added_to_active, v2v_cpu_finished, v2v_dag_completed)
                if max_count - min_count > max(1, int(0.01 * max_count)):
                    lifecycle_breach = True
            
            results.append({
                'v2v_tx_started': v2v_tx_started,
                'v2v_added_to_active': v2v_added_to_active,
                'v2v_cpu_finished': v2v_cpu_finished,
                'v2v_dag_completed': v2v_dag_completed,
                'tx_created': total_tx_created,
                'lifecycle_breach': lifecycle_breach
            })
            
            if verbose and ep < 3:
                print(f"  Episode {ep+1}: tx_started={v2v_tx_started}, added={v2v_added_to_active}, "
                     f"finished={v2v_cpu_finished}, completed={v2v_dag_completed}, tx={total_tx_created}")
        
        # 汇总
        total_tx_started = sum([r['v2v_tx_started'] for r in results])
        total_added = sum([r['v2v_added_to_active'] for r in results])
        total_finished = sum([r['v2v_cpu_finished'] for r in results])
        total_completed = sum([r['v2v_dag_completed'] for r in results])
        avg_tx = np.mean([r['tx_created'] for r in results])
        breach_count = sum([r['lifecycle_breach'] for r in results])
        
        return {
            'name': '场景C: 强制V2V',
            'passed': True,
            'warnings': [],
            'v2v_tx_started': total_tx_started,
            'v2v_added_to_active': total_added,
            'v2v_cpu_finished': total_finished,
            'v2v_dag_completed': total_completed,
            'tx_created_mean': avg_tx,
            'lifecycle_breach_count': breach_count
        }
    finally:
        Cfg.DEADLINE_TIGHTENING_MIN = original_min
        Cfg.DEADLINE_TIGHTENING_MAX = original_max


def run_all_scenarios(seed=42, verbose=True):
    """运行所有场景"""
    print("="*80)
    print("反向场景回归测试：护栏不误伤验证")
    print("="*80)
    print(f"Seed: {seed}\n")
    
    all_results = []
    
    # 场景A
    print("【场景A】RSU-only单车")
    print("-"*80)
    result_a = run_scenario_rsu_only_single(episodes=10, seed=seed, verbose=verbose)
    all_results.append(result_a)
    
    # 断言检查
    warnings_a = []
    if result_a['vehicle_sr'] < 0.95:
        warnings_a.append(f"vehicle_success_rate={result_a['vehicle_sr']:.2%} < 95%")
    if result_a['tx_created_mean'] <= 0:
        warnings_a.append(f"tx_tasks_created_count={result_a['tx_created_mean']} <= 0（需要创建传输）")
    if result_a['deadlock_count'] > 0:
        warnings_a.append(f"deadlock_count={result_a['deadlock_count']} > 0")
    
    result_a['warnings'] = warnings_a
    result_a['passed'] = len(warnings_a) == 0
    
    print(f"结果: SR={result_a['vehicle_sr']:.2%}, tx_created={result_a['tx_created_mean']:.1f}, "
         f"same_node_no_tx={result_a['same_node_no_tx_total']}, deadlock={result_a['deadlock_count']}")
    if warnings_a:
        print(f"⚠️  警告: {', '.join(warnings_a)}")
    print()
    
    # 场景B
    print("【场景B】RSU-only多车")
    print("-"*80)
    result_b = run_scenario_rsu_only_multi(episodes=10, seed=seed, verbose=verbose)
    
    warnings_b = []
    if result_b['vehicle_sr'] < 0.85:
        warnings_b.append(f"vehicle_success_rate={result_b['vehicle_sr']:.2%} < 85%")
    if result_b['tx_created_mean'] <= 0:
        warnings_b.append(f"tx_tasks_created_count={result_b['tx_created_mean']} <= 0")
    if result_b['deadlock_count'] > 0:
        warnings_b.append(f"deadlock_count={result_b['deadlock_count']} > 0")
    
    result_b['warnings'] = warnings_b
    result_b['passed'] = len(warnings_b) == 0
    all_results.append(result_b)
    
    print(f"结果: SR={result_b['vehicle_sr']:.2%}, tx_created={result_b['tx_created_mean']:.1f}, "
         f"deadlock={result_b['deadlock_count']}")
    if warnings_b:
        print(f"⚠️  警告: {', '.join(warnings_b)}")
    print()
    
    # 场景C
    print("【场景C】强制V2V")
    print("-"*80)
    result_c = run_scenario_forced_v2v(episodes=10, seed=seed, verbose=verbose)
    
    warnings_c = []
    if result_c['v2v_tx_started'] > 0:
        max_count = max(result_c['v2v_tx_started'], result_c['v2v_added_to_active'], 
                       result_c['v2v_cpu_finished'], result_c['v2v_dag_completed'])
        min_count = min(result_c['v2v_tx_started'], result_c['v2v_added_to_active'], 
                       result_c['v2v_cpu_finished'], result_c['v2v_dag_completed'])
        if max_count - min_count > max(1, int(0.01 * max_count)):
            warnings_c.append(f"V2V生命周期不守恒（差值={max_count-min_count}）")
    if result_c['tx_created_mean'] <= 0 and result_c['v2v_tx_started'] > 0:
        warnings_c.append(f"tx_tasks_created_count={result_c['tx_created_mean']} <= 0但V2V有传输")
    
    result_c['warnings'] = warnings_c
    result_c['passed'] = len(warnings_c) == 0
    all_results.append(result_c)
    
    print(f"结果: tx_started={result_c['v2v_tx_started']}, added={result_c['v2v_added_to_active']}, "
         f"finished={result_c['v2v_cpu_finished']}, completed={result_c['v2v_dag_completed']}, "
         f"tx_created={result_c['tx_created_mean']:.1f}")
    if warnings_c:
        print(f"⚠️  警告: {', '.join(warnings_c)}")
    print()
    
    # 总结
    print("="*80)
    print("总结")
    print("="*80)
    all_passed = all(r['passed'] for r in all_results)
    for result in all_results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status}: {result['name']}")
        if result['warnings']:
            for warn in result['warnings']:
                print(f"  - {warn}")
    
    if all_passed:
        print(f"\n✅ 所有场景通过：护栏未误伤跨节点传输")
        return 0
    else:
        print(f"\n❌ 部分场景失败：请检查护栏逻辑")
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="反向场景回归测试")
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    exit_code = run_all_scenarios(seed=args.seed, verbose=True)
    sys.exit(exit_code)

