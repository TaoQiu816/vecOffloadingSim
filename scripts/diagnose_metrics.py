#!/usr/bin/env python3
"""
诊断脚本：检查时延和能耗指标统计问题

检查项：
1. _episode_task_durations 是否正确收集
2. _episode_energy_norm_values 是否正确收集
3. DAG completion_time 计算是否正确
4. episode_metrics 输出是否正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def diagnose_metrics():
    """诊断指标收集问题"""
    print("=" * 80)
    print("时延能耗指标诊断")
    print("=" * 80)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    # 运行一个episode
    print("\n[1] 运行测试episode...")
    obs, info = env.reset()
    
    step_count = 0
    done = False
    
    while not done and step_count < 100:
        # 随机动作
        actions = []
        for v in env.vehicles:
            action = {
                'target': np.random.randint(0, 3),
                'subtask': 0,
                'power': 0.5
            }
            actions.append(action)
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        step_count += 1
    
    print(f"   Episode完成: {step_count} steps, terminated={terminated}, truncated={truncated}")
    
    # 检查内部统计列表
    print("\n[2] 检查内部统计列表...")
    print(f"   _episode_task_durations: {len(env._episode_task_durations)} 条记录")
    if env._episode_task_durations:
        print(f"      值: {env._episode_task_durations[:5]}...")
        print(f"      均值: {np.mean(env._episode_task_durations):.4f}")
    else:
        print(f"      ⚠️  列表为空！")
    
    print(f"   _episode_energy_norm_values: {len(env._episode_energy_norm_values)} 条记录")
    if env._episode_energy_norm_values:
        print(f"      值: {env._episode_energy_norm_values[:5]}...")
        print(f"      均值: {np.mean(env._episode_energy_norm_values):.4f}")
    else:
        print(f"      ⚠️  列表为空！")
    
    # 检查车辆DAG状态
    print("\n[3] 检查车辆DAG状态...")
    completed_count = 0
    failed_count = 0
    running_count = 0
    
    for i, v in enumerate(env.vehicles):
        dag = v.task_dag
        if dag.is_finished:
            if dag.is_failed:
                failed_count += 1
            else:
                completed_count += 1
                if i < 3:  # 只打印前3个
                    print(f"   Vehicle {v.id}: 完成")
                    print(f"      completion_time: {dag.completion_time}")
                    print(f"      _completion_logged: {getattr(dag, '_completion_logged', False)}")
        else:
            running_count += 1
    
    print(f"   总计: 完成={completed_count}, 失败={failed_count}, 运行中={running_count}")
    
    # 检查episode_metrics输出
    print("\n[4] 检查episode_metrics输出...")
    if hasattr(env, '_last_episode_metrics'):
        metrics = env._last_episode_metrics
        print(f"   task_duration_mean: {metrics.get('task_duration_mean', 'N/A')}")
        print(f"   task_duration_p95: {metrics.get('task_duration_p95', 'N/A')}")
        print(f"   completed_tasks_count: {metrics.get('completed_tasks_count', 'N/A')}")
        print(f"   energy_norm_mean: {metrics.get('energy_norm_mean', 'N/A')}")
        print(f"   energy_norm_p95: {metrics.get('energy_norm_p95', 'N/A')}")
    else:
        print(f"   ⚠️  _last_episode_metrics 不存在")
    
    # 诊断结论
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    
    issues = []
    
    if len(env._episode_task_durations) == 0 and completed_count > 0:
        issues.append("❌ 问题1: 有完成的任务但 _episode_task_durations 为空")
        issues.append("   可能原因: completion_time 记录逻辑有误或 _completion_logged 标志问题")
    
    if len(env._episode_energy_norm_values) == 0:
        issues.append("❌ 问题2: _episode_energy_norm_values 为空")
        issues.append("   可能原因: cost_power 计算条件不满足或值异常")
    
    if len(env._episode_task_durations) > 0 and np.mean(env._episode_task_durations) < 0.001:
        issues.append("❌ 问题3: task_duration 值过小（< 0.001）")
        issues.append("   可能原因: completion_time 计算错误")
    
    if len(env._episode_energy_norm_values) > 0 and np.mean(env._episode_energy_norm_values) < 0.001:
        issues.append("❌ 问题4: energy_norm 值过小（< 0.001）")
        issues.append("   可能原因: cost_power 归一化错误")
    
    if not issues:
        print("✅ 未发现明显问题")
    else:
        for issue in issues:
            print(issue)
    
    print("\n建议修复方案:")
    print("1. 检查 DAG.completion_time 的计算逻辑（第4708-4715行）")
    print("2. 检查 cost_power 的计算和收集条件（第4898行）")
    print("3. 添加调试日志跟踪指标收集过程")
    print("4. 验证 episode_metrics 的输出路径")


if __name__ == "__main__":
    diagnose_metrics()
