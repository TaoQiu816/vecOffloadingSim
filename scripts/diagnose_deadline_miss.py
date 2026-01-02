#!/usr/bin/env python
"""
Deadline Miss根因诊断脚本

输出关键信息：
1. Deadline起点/单位/比较逻辑
2. n_active/effective_speed分布
3. tx_time分布
4. 三刀流实验结果
"""

import sys
import os
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def diagnose_episode(env, policy_name="local_only", max_steps=400):
    """诊断单个episode的关键指标"""
    obs, info = env.reset()
    
    # 记录关键信息
    diagnostics = {
        'policy': policy_name,
        'deadline_info': {},
        'active_tasks_history': [],
        'tx_times': [],
        'effective_speeds': [],
        'task_completion_times': []
    }
    
    # 获取第一个任务的关键信息
    if env.vehicles:
        v = env.vehicles[0]
        deadline_s = getattr(v.task_dag, 'deadline', 0)
        cp_cycles = getattr(v.task_dag, 'critical_path_cycles', 0)
        base_time = getattr(v.task_dag, 'deadline_base_time', 0)
        gamma = getattr(v.task_dag, 'deadline_gamma', 0)
        start_time = getattr(v.task_dag, 'start_time', env.time)
        
        diagnostics['deadline_info'] = {
            'deadline_seconds': deadline_s,
            'critical_path_cycles': cp_cycles,
            'base_time': base_time,
            'gamma': gamma,
            'cpu_freq': v.cpu_freq,
            'estimated_cp_time_local': cp_cycles / v.cpu_freq if v.cpu_freq > 0 else 0,
            'estimated_cp_time_rsu': cp_cycles / Cfg.F_RSU if Cfg.F_RSU > 0 else 0,
            'release_time': start_time,  # 任务开始时间
            'deadline_absolute': start_time + deadline_s
        }
    
    step_count = 0
    done = False
    truncated = False
    
    while not (done or truncated) and step_count < max_steps:
        # 选择策略
        if policy_name == "local_only":
            actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        elif policy_name == "rsu_only":
            actions = [{"target": 1, "power": 1.0} for _ in env.vehicles]
        else:
            actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        
        obs, rewards, done, truncated, info = env.step(actions)
        step_count += 1
        
        # 每10步记录一次active任务数和有效速度
        if step_count % 10 == 0:
            for v in env.vehicles:
                if hasattr(v, 'active_task_manager'):
                    n_active = v.active_task_manager.get_num_active_tasks()
                    if n_active > 0:
                        # 估算有效速度
                        effective_speed = v.cpu_freq / max(1, n_active)
                        diagnostics['active_tasks_history'].append({
                            'step': step_count,
                            'vehicle_id': v.id,
                            'n_active': n_active,
                            'cpu_freq': v.cpu_freq,
                            'effective_speed': effective_speed
                        })
                        diagnostics['effective_speeds'].append(effective_speed)
            
            # RSU active任务
            for rsu in env.rsus:
                if hasattr(rsu, 'active_task_manager'):
                    n_active = rsu.active_task_manager.get_num_active_tasks()
                    if n_active > 0:
                        effective_speed = (rsu.cpu_freq * rsu.num_processors) / max(1, n_active)
                        diagnostics['active_tasks_history'].append({
                            'step': step_count,
                            'rsu_id': rsu.id,
                            'n_active': n_active,
                            'cpu_freq': rsu.cpu_freq,
                            'num_processors': rsu.num_processors,
                            'effective_speed': effective_speed
                        })
                        diagnostics['effective_speeds'].append(effective_speed)
    
    # 提取最终信息
    diagnostics['final_info'] = {
        'episode_time': env.time,
        'steps': step_count,
        'task_success_rate': info.get('task_success_rate', 0),
        'deadline_miss_rate': info.get('deadline_miss_rate', 0),
        'miss_reason_deadline': info.get('miss_reason_deadline', 0),
        'task_duration_mean': info.get('task_duration_mean', 0),
        'episode_task_count': info.get('episode_task_count', 0)
    }
    
    # 检查deadline比较逻辑
    if diagnostics['deadline_info'].get('deadline_seconds', 0) > 0:
        deadline_start = diagnostics['deadline_info']['release_time']
        deadline_s = diagnostics['deadline_info']['deadline_seconds']
        finish_time = diagnostics['final_info']['episode_time']
        
        diagnostics['deadline_comparison'] = {
            'deadline_start_time': deadline_start,
            'deadline_seconds': deadline_s,
            'deadline_absolute': deadline_start + deadline_s,
            'finish_time': finish_time,
            'would_miss': finish_time > (deadline_start + deadline_s),
            'time_to_deadline': (deadline_start + deadline_s) - finish_time if finish_time <= (deadline_start + deadline_s) else finish_time - (deadline_start + deadline_s)
        }
    
    return diagnostics


def run_three_knife_experiment(num_episodes=5, seed=42):
    """三刀流实验"""
    print("="*80)
    print("三刀流实验：Deadline Miss根因诊断")
    print("="*80)
    print()
    
    results = {
        'exp1_local_only': [],
        'exp2_rsu_only': [],
        'exp3_multivehicle_rsu': []
    }
    
    # ========== 实验1：单车、无通信（强制全Local） ==========
    print("【实验1】单车、无通信（强制全Local）")
    print("-"*80)
    
    np.random.seed(seed)
    # 临时修改为单车辆
    original_num_vehicles = Cfg.NUM_VEHICLES
    Cfg.NUM_VEHICLES = 1
    
    for ep in range(num_episodes):
        env = VecOffloadingEnv()
        diag = diagnose_episode(env, policy_name="local_only")
        results['exp1_local_only'].append(diag)
        
        dl_info = diag['deadline_info']
        dl_comp = diag.get('deadline_comparison', {})
        final = diag['final_info']
        
        print(f"Ep{ep+1}: DL={dl_info.get('deadline_seconds', 0):.2f}s, "
              f"CP={dl_info.get('critical_path_cycles', 0)/1e9:.2f}G, "
              f"Finish={final['episode_time']:.2f}s, "
              f"SR={final['task_success_rate']:.1%}, "
              f"Miss={final['miss_reason_deadline']}/{final['episode_task_count']}")
        
        if dl_comp:
            print(f"      DL_start={dl_comp.get('deadline_start_time', 0):.2f}s, "
                  f"DL_abs={dl_comp.get('deadline_absolute', 0):.2f}s, "
                  f"Would_miss={dl_comp.get('would_miss', False)}")
        
        # 输出active任务统计
        if diag['active_tasks_history']:
            n_active_max = max(h['n_active'] for h in diag['active_tasks_history'])
            n_active_mean = np.mean([h['n_active'] for h in diag['active_tasks_history']])
            print(f"      n_active: max={n_active_max:.1f}, mean={n_active_mean:.1f}")
    
    Cfg.NUM_VEHICLES = original_num_vehicles
    
    # ========== 实验2：单车、强制全RSU ==========
    print("\n【实验2】单车、强制全RSU")
    print("-"*80)
    
    np.random.seed(seed)
    Cfg.NUM_VEHICLES = 1
    
    for ep in range(num_episodes):
        env = VecOffloadingEnv()
        diag = diagnose_episode(env, policy_name="rsu_only")
        results['exp2_rsu_only'].append(diag)
        
        final = diag['final_info']
        print(f"Ep{ep+1}: SR={final['task_success_rate']:.1%}, "
              f"Miss={final['miss_reason_deadline']}/{final['episode_task_count']}")
    
    Cfg.NUM_VEHICLES = original_num_vehicles
    
    # ========== 实验3：多车、强制全RSU ==========
    print("\n【实验3】多车、强制全RSU")
    print("-"*80)
    
    np.random.seed(seed)
    
    for ep in range(num_episodes):
        env = VecOffloadingEnv()
        diag = diagnose_episode(env, policy_name="rsu_only")
        results['exp3_multivehicle_rsu'].append(diag)
        
        final = diag['final_info']
        print(f"Ep{ep+1}: SR={final['task_success_rate']:.1%}, "
              f"Miss={final['miss_reason_deadline']}/{final['episode_task_count']}")
        
        # 输出RSU active任务统计
        if diag['active_tasks_history']:
            rsu_history = [h for h in diag['active_tasks_history'] if 'rsu_id' in h]
            if rsu_history:
                n_active_max = max(h['n_active'] for h in rsu_history)
                n_active_mean = np.mean([h['n_active'] for h in rsu_history])
                eff_speed_min = min(h['effective_speed'] for h in rsu_history)
                print(f"      RSU n_active: max={n_active_max:.1f}, mean={n_active_mean:.1f}, "
                      f"eff_speed_min={eff_speed_min/1e9:.2f}GHz")
    
    # ========== 汇总分析 ==========
    print("\n" + "="*80)
    print("汇总分析")
    print("="*80)
    
    for exp_name, exp_results in results.items():
        if not exp_results:
            continue
        
        avg_sr = np.mean([r['final_info']['task_success_rate'] for r in exp_results])
        avg_miss_rate = np.mean([r['final_info']['miss_reason_deadline'] / max(1, r['final_info']['episode_task_count']) 
                                 for r in exp_results])
        
        print(f"\n{exp_name}:")
        print(f"  Avg SR: {avg_sr:.1%}")
        print(f"  Avg Miss Rate: {avg_miss_rate:.1%}")
    
    # 保存详细结果
    output_file = 'logs/deadline_diagnosis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_three_knife_experiment(num_episodes=args.episodes, seed=args.seed)

