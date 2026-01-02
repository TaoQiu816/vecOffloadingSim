#!/usr/bin/env python3
"""
回归测试：Local-only同节点依赖误创建传输任务导致死锁

测试目标：
- 验证Local-only场景下不会创建传输任务
- 验证不会出现死锁
- 验证任务能正常完成

运行：python scripts/regression_local_no_tx_deadlock.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg
import numpy as np


def run_regression_test(episodes=3, seed=42, verbose=True):
    """运行Local-only回归测试"""
    
    # 临时放宽deadline以避免deadline干扰（测试重点是死锁）
    original_min = Cfg.DEADLINE_TIGHTENING_MIN
    original_max = Cfg.DEADLINE_TIGHTENING_MAX
    Cfg.DEADLINE_TIGHTENING_MIN = 3.0  # 非常宽松
    Cfg.DEADLINE_TIGHTENING_MAX = 5.0
    
    try:
        env = VecOffloadingEnv()
        
        all_results = []
        
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            step_count = 0
            
            # 强制Local-only策略：所有动作都是Local (target_id=0)
            while not (done or truncated):
                # 构建动作：所有车辆都选择Local
                actions = []
                for v in env.vehicles:
                    # Local动作：target_id=0, power_ratio=1.0, subtask_index=第一个READY任务
                    dag = v.task_dag
                    ready_tasks = np.where(dag.status == 1)[0]  # READY状态
                    if len(ready_tasks) > 0:
                        subtask_idx = int(ready_tasks[0])
                    else:
                        subtask_idx = 0
                    
                    actions.append({
                        'target_id': 0,  # Local
                        'power_ratio': 1.0,
                        'subtask_index': subtask_idx
                    })
                
                obs, rewards, done, truncated, info = env.step(actions)
                step_count += 1
                
                # 防止无限循环
                if step_count > 10000:
                    if verbose:
                        print(f"⚠️  Episode {ep+1} exceeded 10000 steps")
                    break
            
            # 等待episode结束，确保指标被正确记录
            # 通常在done或truncated时，_log_episode_stats会被调用
            # 但为了确保，我们手动触发一次统计（如果需要）
            
            # 收集episode结果：从env对象直接计算（因为_log_episode_stats只写JSON不写info）
            completed_subtasks = sum(int(np.sum(v.task_dag.status == 3)) for v in env.vehicles)
            total_subtasks = sum(int(v.task_dag.num_subtasks) for v in env.vehicles)
            completed_vehicles = sum(1 for v in env.vehicles if v.task_dag.is_finished)
            total_vehicles = len(env.vehicles)
            
            # 死锁检查
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
            
            # 传输任务统计
            total_tx_created = sum(getattr(v.task_dag, '_tx_tasks_created_count', 0) for v in env.vehicles)
            total_same_node_no_tx = sum(getattr(v.task_dag, '_same_node_no_tx_count', 0) for v in env.vehicles)
            
            vehicle_sr = completed_vehicles / max(total_vehicles, 1)
            subtask_sr = completed_subtasks / max(total_subtasks, 1)
            episode_all_success = 1.0 if all(v.task_dag.is_finished for v in env.vehicles) else 0.0
            
            # [P2性能统计] 从env对象获取
            total_time = env._p2_active_time + env._p2_idle_time
            service_rate_when_active = env._p2_deltaW_active / max(env._p2_active_time, 1e-9) if env._p2_active_time > 0 else 0.0
            idle_fraction = env._p2_idle_time / max(total_time, 1e-9)
            
            results = {
                'episode': ep + 1,
                'deadlock_count': deadlock_count,
                'deadline_misses': env._audit_deadline_misses,
                'tx_created': total_tx_created,
                'same_node_no_tx': total_same_node_no_tx,
                'vehicle_sr': vehicle_sr,
                'subtask_sr': subtask_sr,
                'episode_all_success': episode_all_success,
                'steps': step_count,
                'service_rate_when_active': service_rate_when_active,
                'idle_fraction': idle_fraction
            }
            all_results.append(results)
            
            if verbose:
                print(f"Episode {ep+1}: "
                      f"deadlock={results['deadlock_count']}, "
                      f"tx_created={results['tx_created']}, "
                      f"vehicle_SR={results['vehicle_sr']:.2%}, "
                      f"subtask_SR={results['subtask_sr']:.2%}")
        
        # 汇总结果
        avg_deadlock = np.mean([r['deadlock_count'] for r in all_results])
        avg_deadline_misses = np.mean([r['deadline_misses'] for r in all_results])
        avg_tx_created = np.mean([r['tx_created'] for r in all_results])
        avg_vehicle_sr = np.mean([r['vehicle_sr'] for r in all_results])
        avg_subtask_sr = np.mean([r['subtask_sr'] for r in all_results])
        avg_episode_all_success = np.mean([r['episode_all_success'] for r in all_results])
        total_same_node_no_tx = sum([r['same_node_no_tx'] for r in all_results])
        avg_service_rate_active = np.mean([r['service_rate_when_active'] for r in all_results])
        avg_idle_fraction = np.mean([r['idle_fraction'] for r in all_results])
        
        # 断言检查
        passed = True
        failure_reasons = []
        
        if avg_deadlock > 0:
            passed = False
            failure_reasons.append(f"deadlock_count={avg_deadlock} > 0")
        
        if avg_tx_created > 0:
            passed = False
            failure_reasons.append(f"tx_tasks_created_count={avg_tx_created} > 0 (Local-only不应创建传输)")
        
        if avg_vehicle_sr < 0.90:
            passed = False
            failure_reasons.append(f"vehicle_success_rate={avg_vehicle_sr:.2%} < 90%")
        
        if avg_subtask_sr < 0.95:
            passed = False
            failure_reasons.append(f"subtask_success_rate={avg_subtask_sr:.2%} < 95%")
        
        # 输出结果
        print("\n" + "="*80)
        print("Local-only同节点传输死锁回归测试")
        print("="*80)
        print(f"Episodes: {episodes}")
        print(f"Seed: {seed}")
        print(f"\n关键指标（平均）：")
        print(f"  deadlock_vehicle_count: {avg_deadlock:.1f}")
        print(f"  audit_deadline_misses: {avg_deadline_misses:.1f}")
        print(f"  tx_tasks_created_count: {avg_tx_created:.1f}")
        print(f"  same_node_no_tx_count: {total_same_node_no_tx}")
        print(f"  vehicle_success_rate: {avg_vehicle_sr:.2%}")
        print(f"  subtask_success_rate: {avg_subtask_sr:.2%}")
        print(f"  episode_all_success: {avg_episode_all_success:.2%}")
        print(f"\n[P2性能统计]（平均）：")
        print(f"  service_rate_when_active: {avg_service_rate_active:.2e} cycles/s ({avg_service_rate_active/1e9:.3f} Gcycles/s)")
        print(f"  idle_fraction: {avg_idle_fraction:.2%}")
        
        # [P2软检查] 不作为硬断言，避免因环境差异误报
        # 优化：仅在任务未全部完成时检查idle_fraction（任务完成后系统继续idle属正常）
        p2_warnings = []
        if avg_service_rate_active <= 0:
            p2_warnings.append(f"⚠️  service_rate_when_active={avg_service_rate_active:.2e} <= 0（必须>0）")
        
        # 条件性检查idle_fraction：仅当任务未全部完成时提示
        if avg_episode_all_success < 1.0:
            # 有未完成的episode，检查idle_fraction是否异常
            if avg_idle_fraction >= 0.8:
                p2_warnings.append(
                    f"⚠️  idle_fraction={avg_idle_fraction:.2%} >= 80% 且存在未完成episode"
                    f"（可能任务推进缓慢或死锁）"
                )
        
        if p2_warnings:
            print(f"\n[P2性能统计警告]（非阻塞）：")
            for warn in p2_warnings:
                print(f"  {warn}")
        
        if passed:
            print(f"\n✅ PASS: 所有断言通过")
            return 0
        else:
            print(f"\n❌ FAIL: 以下断言失败:")
            for reason in failure_reasons:
                print(f"  - {reason}")
            return 1
            
    finally:
        # 恢复原始deadline配置
        Cfg.DEADLINE_TIGHTENING_MIN = original_min
        Cfg.DEADLINE_TIGHTENING_MAX = original_max


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Local-only死锁回归测试")
    parser.add_argument('--episodes', type=int, default=3, help='测试episode数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    exit_code = run_regression_test(episodes=args.episodes, seed=args.seed, verbose=True)
    sys.exit(exit_code)

