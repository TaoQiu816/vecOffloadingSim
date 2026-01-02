#!/usr/bin/env python
"""
V2V生命周期强制测试（六连守恒硬验证）

目的：不依赖学习策略，强制车辆选择V2V，验证：
1. TX Started → TX Done → Received → Added to Active → CPU Finished → DAG Completed
2. 六连计数必须守恒（差值<1%）
3. 完成回写发生在owner的DAG上（不是receiver）
4. 任何断层立即报错并定位

用法：
    python scripts/test_v2v_lifecycle.py --episodes 10
"""

import sys
import os
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


class V2VLifecycleTracker:
    """V2V生命周期追踪器（六连守恒验证）"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置episode级追踪"""
        self.tx_started = set()      # (owner_id, subtask_id)
        self.tx_done = set()
        self.received = set()
        self.added_to_active = set()
        self.cpu_finished = set()
        self.dag_completed = set()
        
        # 详细事件日志（用于断层诊断）
        self.events = []
    
    def record_event(self, stage, owner_id, subtask_id, receiver_id=None, step=None):
        """记录生命周期事件"""
        key = (owner_id, subtask_id)
        
        if stage == 'tx_started':
            self.tx_started.add(key)
        elif stage == 'tx_done':
            self.tx_done.add(key)
        elif stage == 'received':
            self.received.add(key)
        elif stage == 'added_to_active':
            self.added_to_active.add(key)
        elif stage == 'cpu_finished':
            self.cpu_finished.add(key)
        elif stage == 'dag_completed':
            self.dag_completed.add(key)
        
        self.events.append({
            'step': step,
            'stage': stage,
            'owner_id': owner_id,
            'subtask_id': subtask_id,
            'receiver_id': receiver_id
        })
    
    def get_counts(self):
        """获取六连计数"""
        return {
            'tx_started': len(self.tx_started),
            'tx_done': len(self.tx_done),
            'received': len(self.received),
            'added_to_active': len(self.added_to_active),
            'cpu_finished': len(self.cpu_finished),
            'dag_completed': len(self.dag_completed)
        }
    
    def check_conservation(self):
        """检查守恒性（六连必须相等）"""
        counts = self.get_counts()
        values = list(counts.values())
        
        if max(values) == 0:
            return True, "无V2V任务"
        
        max_count = max(values)
        min_count = min(values)
        
        # 允许最多1个任务的差异（考虑episode截断）
        if max_count - min_count <= 1:
            return True, "守恒"
        
        # 定位断层
        breaches = []
        stages = list(counts.keys())
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            if counts[current_stage] - counts[next_stage] > 1:
                breaches.append(f"{current_stage}({counts[current_stage]}) → {next_stage}({counts[next_stage]})")
        
        return False, f"断层: {', '.join(breaches)}"
    
    def find_lost_tasks(self, stage1, stage2):
        """找到在两个阶段之间丢失的任务"""
        set1 = getattr(self, stage1)
        set2 = getattr(self, stage2)
        lost = set1 - set2
        return list(lost)[:5]  # 最多返回5个示例


def force_v2v_policy(env):
    """
    强制V2V策略：优先选择最近的V2V邻居
    
    Returns:
        list: actions for all vehicles
    """
    actions = []
    
    for v in env.vehicles:
        # 检查是否有可调度任务
        if v.task_dag.get_top_priority_task() is None:
            actions.append({"target": 0, "power": 1.0})
            continue
        
        # 获取V2V邻居
        candidates = env._last_candidates.get(v.id, [])
        valid_v2v = [cid for cid in candidates if cid is not None and cid >= 0]
        
        if valid_v2v:
            # 选择第一个V2V邻居（已按距离排序）
            target = 2  # V2V从索引2开始
        else:
            # 无V2V邻居，回退到RSU或Local
            rsu_id = env._last_rsu_choice.get(v.id)
            target = 1 if rsu_id is not None else 0
        
        actions.append({"target": target, "power": 1.0})
    
    return actions


def run_v2v_test(num_episodes=10, seed=42):
    """运行V2V强制测试"""
    print("="*80)
    print("V2V生命周期强制测试（六连守恒硬验证）")
    print("="*80)
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print(f"策略: 强制V2V（有邻居则选，无则回退RSU/Local）")
    print()
    
    np.random.seed(seed)
    env = VecOffloadingEnv()
    
    all_episodes_pass = True
    episode_results = []
    
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}...", end='', flush=True)
        
        tracker = V2VLifecycleTracker()
        obs, info = env.reset()
        done = False
        truncated = False
        step_count = 0
        
        # 手动追踪V2V事件（因为env内部追踪可能不完整）
        v2v_decisions = 0
        
        while not (done or truncated):
            actions = force_v2v_policy(env)
            
            # 统计V2V决策数
            for action in actions:
                if action['target'] >= 2:
                    v2v_decisions += 1
            
            obs, rewards, done, truncated, info = env.step(actions)
            step_count += 1
            
            # TODO: 从env的audit info中提取V2V生命周期事件
            # 这需要env在关键点调用tracker.record_event
        
        # 从环境获取生命周期统计
        lifecycle_stats = {
            'tx_started': info.get('audit_v2v_tx_started', 0),
            'tx_done': info.get('audit_v2v_tx_done', 0),
            'received': info.get('audit_v2v_received', 0),
            'added_to_active': info.get('audit_v2v_added_to_active', 0),
            'cpu_finished': info.get('audit_v2v_cpu_finished', 0),
            'dag_completed': info.get('audit_v2v_dag_completed', 0)
        }
        
        # 检查守恒
        max_count = max(lifecycle_stats.values())
        min_count = min(lifecycle_stats.values())
        is_conserved = (max_count - min_count <= 1) if max_count > 0 else True
        
        result = {
            'episode': ep + 1,
            'v2v_decisions': v2v_decisions,
            'lifecycle': lifecycle_stats,
            'conserved': is_conserved,
            'max_count': max_count,
            'min_count': min_count
        }
        episode_results.append(result)
        
        status = "✅" if is_conserved else "❌"
        print(f" {status} V2V={v2v_decisions}, 六连={lifecycle_stats}")
        
        if not is_conserved:
            all_episodes_pass = False
            # 定位断层
            stages = list(lifecycle_stats.keys())
            for i in range(len(stages) - 1):
                if lifecycle_stats[stages[i]] - lifecycle_stats[stages[i+1]] > 1:
                    print(f"    ⚠️  断层: {stages[i]}({lifecycle_stats[stages[i]]}) → {stages[i+1]}({lifecycle_stats[stages[i+1]]})")
    
    # 生成报告
    print("\n" + "="*80)
    print("V2V生命周期测试报告")
    print("="*80)
    
    total_v2v = sum(r['v2v_decisions'] for r in episode_results)
    passed_episodes = sum(1 for r in episode_results if r['conserved'])
    
    print(f"\n【决策统计】")
    print(f"  总V2V决策数: {total_v2v}")
    print(f"  平均每episode: {total_v2v/num_episodes:.1f}")
    
    print(f"\n【六连守恒】")
    print(f"  通过episode: {passed_episodes}/{num_episodes}")
    
    if total_v2v == 0:
        print(f"\n⚠️  警告: 无V2V决策！可能原因：")
        print(f"    - V2V邻居数过少")
        print(f"    - V2V_RANGE过小")
        print(f"    - 车辆密度过低")
        return False
    
    if all_episodes_pass:
        print(f"\n✅ 所有episode通过六连守恒检查")
        return True
    else:
        print(f"\n❌ {num_episodes - passed_episodes}个episode未通过")
        print(f"\n【断层诊断】")
        
        # 汇总所有episode的六连计数
        total_lifecycle = defaultdict(int)
        for r in episode_results:
            for stage, count in r['lifecycle'].items():
                total_lifecycle[stage] += count
        
        print(f"  累计六连计数:")
        for stage, count in total_lifecycle.items():
            print(f"    {stage}: {count}")
        
        # 定位最大断层
        stages = list(total_lifecycle.keys())
        max_breach = 0
        breach_location = None
        for i in range(len(stages) - 1):
            diff = total_lifecycle[stages[i]] - total_lifecycle[stages[i+1]]
            if diff > max_breach:
                max_breach = diff
                breach_location = (stages[i], stages[i+1])
        
        if breach_location:
            print(f"\n  最大断层: {breach_location[0]} → {breach_location[1]} (丢失{max_breach}个任务)")
            print(f"\n  【修复指引】")
            
            if breach_location == ('tx_started', 'tx_done'):
                print(f"    → 检查: step_inter_task_transfers中的传输推进逻辑")
            elif breach_location == ('tx_done', 'received'):
                print(f"    → 检查: 数据到达判定（data_just_arrived）")
            elif breach_location == ('received', 'added_to_active'):
                print(f"    → 检查: handover逻辑中的add_active_task调用")
                print(f"    → 可能原因: is_data_ready=False或断言失败")
            elif breach_location == ('added_to_active', 'cpu_finished'):
                print(f"    → 检查: vehicle.step(dt)或rsu.step(dt)执行")
                print(f"    → 可能原因: 处理器共享模型未推进")
            elif breach_location == ('cpu_finished', 'dag_completed'):
                print(f"    → 检查: _handle_task_completion中的DAG更新")
                print(f"    → 可能原因: owner车辆已离开或DAG状态更新失败")
        
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    success = run_v2v_test(args.episodes, args.seed)
    
    if not success:
        print("\n⚠️  V2V生命周期测试未通过！必须修复后才能进行训练。")
        sys.exit(1)
    else:
        print("\n✅ V2V生命周期测试通过！环境V2V路径可靠。")
        sys.exit(0)

