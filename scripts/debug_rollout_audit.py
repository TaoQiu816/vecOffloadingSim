#!/usr/bin/env python
"""
严格审计脚本：定位 VEC/MAPPO 策略崩溃根因

运行命令：
    cd /path/to/vecOffloadingSim
    python scripts/debug_rollout_audit.py --episodes 20 --seed 42

依赖：
    - configs/config.py 中关键参数已正确设置
    - envs/vec_offloading_env.py 已插入 instrumentation
    
输出：
    - logs/audit_summary.jsonl  (每episode一行)
    - logs/audit_events.jsonl   (仅异常事件)
    - logs/audit_report.txt     (最终报告)
"""

import sys
import os
import json
import numpy as np
import argparse
from collections import defaultdict
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


class AuditCollector:
    """审计数据收集器"""
    
    def __init__(self):
        self.reset_episode()
        self.episode_summaries = []
        self.events = []
        
    def reset_episode(self):
        """重置episode级别统计"""
        self.ep_data = {
            # (1) RSU mask统计
            'rsu_mask_true_count': 0,
            'rsu_mask_total_count': 0,
            
            # (2) V2V可选数统计
            'valid_v2v_counts': [],
            
            # (3) Illegal动作统计
            'illegal_count_local': 0,
            'illegal_count_rsu': 0,
            'illegal_count_v2v': 0,
            'illegal_reasons': [],
            
            # (4) RSU队列统计
            'rsu_queue_lens': [],
            'rsu_overflow_count': 0,
            
            # (5) V2V生命周期（六连）
            'v2v_tx_started': 0,
            'v2v_tx_done': 0,
            'v2v_received': 0,
            'v2v_added_to_active': 0,
            'v2v_cpu_finished': 0,
            'v2v_dag_completed': 0,
            
            # (6) Active准入断言失败
            'active_assert_fail_local': 0,
            'active_assert_fail_v2v': 0,
            'active_assert_fail_v2i': 0,
            
            # (7) Done reason
            'done_reason': None,  # 'SUCCESS'/'FAIL'/'TRUNCATED'
            
            # (8) Reward分解
            'reward_by_action_type': {'local': [], 'rsu': [], 'v2v': []},
            'reward_components': [],
            
            # (9) Terminated触发
            'terminated_trigger': None,
            
            # (10) Latency
            'wall_clock_latencies': [],
            'dT_eff_values': [],
            
            # (11) 任务状态冲突
            'task_state_conflicts': 0,
            
            # (12) 双重推进检测
            'double_progress_detected': 0,
            
            # 辅助统计
            'steps': 0,
            'total_reward': 0.0,
            'task_success_rate': 0.0,
        }
    
    def record_step(self, step_info):
        """记录单步数据"""
        self.ep_data['steps'] += 1
        
        # (1) RSU mask
        if 'rsu_mask_true' in step_info:
            self.ep_data['rsu_mask_true_count'] += step_info['rsu_mask_true']
            self.ep_data['rsu_mask_total_count'] += 1
        
        # (2) V2V可选数
        if 'valid_v2v_count' in step_info:
            self.ep_data['valid_v2v_counts'].append(step_info['valid_v2v_count'])
        
        # (3) Illegal
        if 'illegal_action' in step_info and step_info['illegal_action']:
            action_type = step_info.get('action_type', 'unknown')
            reason = step_info.get('illegal_reason', 'unknown')
            
            if action_type == 'local':
                self.ep_data['illegal_count_local'] += 1
            elif action_type == 'rsu':
                self.ep_data['illegal_count_rsu'] += 1
            elif action_type == 'v2v':
                self.ep_data['illegal_count_v2v'] += 1
            
            self.ep_data['illegal_reasons'].append({
                'step': self.ep_data['steps'],
                'type': action_type,
                'reason': reason
            })
            
            # 记录异常事件
            self.events.append({
                'type': 'illegal_action',
                'step': self.ep_data['steps'],
                'action_type': action_type,
                'reason': reason,
                'mask_was_true': step_info.get('mask_was_true', False)  # 关键：mask与illegal是否一致
            })
        
        # (4) RSU队列
        if 'rsu_queue_len' in step_info:
            self.ep_data['rsu_queue_lens'].append(step_info['rsu_queue_len'])
        if step_info.get('rsu_overflow', False):
            self.ep_data['rsu_overflow_count'] += 1
        
        # (8) Reward分解
        if 'reward_component' in step_info:
            self.ep_data['reward_components'].append(step_info['reward_component'])
            action_type = step_info.get('action_type', 'local')
            reward_val = step_info.get('reward', 0.0)
            if action_type in self.ep_data['reward_by_action_type']:
                self.ep_data['reward_by_action_type'][action_type].append(reward_val)
        
        # (10) Latency
        if 'wall_clock_latency' in step_info:
            lat = step_info['wall_clock_latency']
            self.ep_data['wall_clock_latencies'].append(lat)
            if lat < 0:
                self.events.append({
                    'type': 'CRITICAL_negative_latency',
                    'step': self.ep_data['steps'],
                    'latency': lat
                })
        
        if 'dT_eff' in step_info:
            self.ep_data['dT_eff_values'].append(step_info['dT_eff'])
    
    def record_v2v_lifecycle(self, event_type, task_key=None):
        """记录V2V生命周期事件
        
        Args:
            event_type: 事件类型
            task_key: (owner_id, subtask_id) 或 None
        """
        if event_type == 'tx_started':
            self.ep_data['v2v_tx_started'] += 1
        elif event_type == 'tx_done':
            self.ep_data['v2v_tx_done'] += 1
        elif event_type == 'received':
            self.ep_data['v2v_received'] += 1
        elif event_type == 'added_to_active':
            self.ep_data['v2v_added_to_active'] += 1
        elif event_type == 'cpu_finished':
            self.ep_data['v2v_cpu_finished'] += 1
        elif event_type == 'dag_completed':
            self.ep_data['v2v_dag_completed'] += 1
    
    def finalize_episode(self, env, terminated, truncated, info):
        """Episode结束，生成摘要"""
        # (7) Done reason
        if terminated and not truncated:
            self.ep_data['done_reason'] = 'SUCCESS'
        elif truncated:
            self.ep_data['done_reason'] = 'TRUNCATED'
        else:
            self.ep_data['done_reason'] = 'FAIL'
        
        # (9) Terminated触发源
        self.ep_data['terminated_trigger'] = info.get('terminated_trigger', 'unknown')
        
        # 计算统计
        summary = {
            'episode': len(self.episode_summaries) + 1,
            
            # (1) RSU mask
            'rsu_mask_true_rate': (self.ep_data['rsu_mask_true_count'] / max(1, self.ep_data['rsu_mask_total_count'])),
            
            # (2) V2V可选数
            'valid_v2v_count_mean': np.mean(self.ep_data['valid_v2v_counts']) if self.ep_data['valid_v2v_counts'] else 0,
            'valid_v2v_count_p95': np.percentile(self.ep_data['valid_v2v_counts'], 95) if len(self.ep_data['valid_v2v_counts']) > 1 else 0,
            
            # (3) Illegal
            'illegal_count_local': self.ep_data['illegal_count_local'],
            'illegal_count_rsu': self.ep_data['illegal_count_rsu'],
            'illegal_count_v2v': self.ep_data['illegal_count_v2v'],
            'illegal_total': self.ep_data['illegal_count_local'] + self.ep_data['illegal_count_rsu'] + self.ep_data['illegal_count_v2v'],
            'illegal_rate': (self.ep_data['illegal_count_local'] + self.ep_data['illegal_count_rsu'] + self.ep_data['illegal_count_v2v']) / max(1, self.ep_data['steps']),
            
            # (4) RSU队列
            'rsu_queue_len_mean': np.mean(self.ep_data['rsu_queue_lens']) if self.ep_data['rsu_queue_lens'] else 0,
            'rsu_queue_len_max': np.max(self.ep_data['rsu_queue_lens']) if self.ep_data['rsu_queue_lens'] else 0,
            'rsu_overflow_count': self.ep_data['rsu_overflow_count'],
            
            # (5) V2V生命周期守恒检查
            'v2v_tx_started': self.ep_data['v2v_tx_started'],
            'v2v_tx_done': self.ep_data['v2v_tx_done'],
            'v2v_received': self.ep_data['v2v_received'],
            'v2v_added_to_active': self.ep_data['v2v_added_to_active'],
            'v2v_cpu_finished': self.ep_data['v2v_cpu_finished'],
            'v2v_dag_completed': self.ep_data['v2v_dag_completed'],
            'v2v_lifecycle_breach': self._check_v2v_lifecycle_breach(),
            
            # (6) Active准入失败
            'active_assert_fail_local': self.ep_data['active_assert_fail_local'],
            'active_assert_fail_v2v': self.ep_data['active_assert_fail_v2v'],
            'active_assert_fail_v2i': self.ep_data['active_assert_fail_v2i'],
            
            # (7) Done reason
            'done_reason': self.ep_data['done_reason'],
            
            # (10) Latency
            'wall_clock_latency_mean': np.mean(self.ep_data['wall_clock_latencies']) if self.ep_data['wall_clock_latencies'] else 0,
            'wall_clock_latency_has_negative': any(l < 0 for l in self.ep_data['wall_clock_latencies']),
            'dT_eff_mean': np.mean(self.ep_data['dT_eff_values']) if self.ep_data['dT_eff_values'] else 0,
            
            # (11)(12) 冲突检测
            'task_state_conflicts': self.ep_data['task_state_conflicts'],
            'double_progress_detected': self.ep_data['double_progress_detected'],
            
            # 其他
            'steps': self.ep_data['steps'],
            'task_success_rate': info.get('task_success_rate', 0.0),
            
            # [P2/P0新增字段] 关键健康指标
            'deadlock_vehicle_count': info.get('deadlock_vehicle_count', 0),
            'audit_deadline_checks': info.get('audit_deadline_checks', 0),
            'audit_deadline_misses': info.get('audit_deadline_misses', 0),
            'miss_reason_deadline': info.get('miss_reason_deadline', 0),
            'miss_reason_overflow': info.get('miss_reason_overflow', 0),
            'miss_reason_illegal': info.get('miss_reason_illegal', 0),
            'miss_reason_unfinished': info.get('miss_reason_unfinished', 0),
            'miss_reason_truncated': info.get('miss_reason_truncated', 0),
            'tx_tasks_created_count': info.get('tx_tasks_created_count', 0),
            'same_node_no_tx_count': info.get('same_node_no_tx_count', 0),
            'service_rate_when_active': info.get('service_rate_when_active', 0.0),
            'idle_fraction': info.get('idle_fraction', 0.0),
            'vehicle_success_rate': info.get('vehicle_success_rate', 0.0),
            'episode_all_success': info.get('episode_all_success', 0.0),
            'subtask_success_rate': info.get('subtask_success_rate', 0.0),
        }
        
        self.episode_summaries.append(summary)
        return summary
    
    def _check_v2v_lifecycle_breach(self):
        """检查V2V生命周期是否违反守恒"""
        counts = [
            self.ep_data['v2v_tx_started'],
            self.ep_data['v2v_tx_done'],
            self.ep_data['v2v_received'],
            self.ep_data['v2v_added_to_active'],
            self.ep_data['v2v_cpu_finished'],
            self.ep_data['v2v_dag_completed']
        ]
        
        if max(counts) == 0:
            return False  # 没有V2V任务，不算breach
        
        # 允许最多1%差异（考虑episode截断）
        max_diff = max(counts) - min(counts)
        threshold = max(1, int(0.01 * max(counts)))
        
        return max_diff > threshold


def run_audit(num_episodes=20, seed=42):
    """运行审计"""
    print("="*80)
    print("VEC/MAPPO 系统审计")
    print("="*80)
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print()
    
    # 设置seed
    np.random.seed(seed)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    # 创建审计收集器
    collector = AuditCollector()
    
    # 创建输出目录
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    summary_file = f'logs/audit_summary_{timestamp}.jsonl'
    events_file = f'logs/audit_events_{timestamp}.jsonl'
    report_file = f'logs/audit_report_{timestamp}.txt'
    
    # 运行episodes
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes}...", end='', flush=True)
        
        collector.reset_episode()
        obs = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Greedy策略：优先RSU，其次Local
            actions = []
            for v_idx, v in enumerate(env.vehicles):
                # 检查是否有可调度任务
                if v.task_dag.get_top_priority_task() is None:
                    actions.append({"target": 0, "power": 1.0})
                    continue
                
                # 优先选择RSU（如果可用）
                rsu_id = env._last_rsu_choice.get(v.id)
                if rsu_id is not None:
                    target = 1  # RSU
                else:
                    target = 0  # Local
                
                power = 1.0  # 最大功率
                actions.append({"target": target, "power": power})
            
            # Step
            obs, rewards, done, truncated, info = env.step(actions)
            
            # 收集审计数据（从env的debug信息中提取）
            step_info = info.get('audit_step_info', {})
            collector.record_step(step_info)
            
            # 记录V2V生命周期事件（如果info中有）
            if 'v2v_lifecycle_events' in info:
                for event in info['v2v_lifecycle_events']:
                    collector.record_v2v_lifecycle(
                        event['type'], 
                        (event['owner_id'], event['subtask_id'])
                    )
        
        # Episode结束
        summary = collector.finalize_episode(env, done, truncated, info)
        print(f" Done. SR={summary['task_success_rate']:.1%}, Steps={summary['steps']}")
        
        # 写入summary
        with open(summary_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
    
    # 写入events
    with open(events_file, 'w') as f:
        for event in collector.events:
            f.write(json.dumps(event) + '\n')
    
    # 生成报告
    generate_report(collector.episode_summaries, collector.events, report_file)
    
    print()
    print(f"审计完成！")
    print(f"摘要: {summary_file}")
    print(f"事件: {events_file}")
    print(f"报告: {report_file}")


def generate_report(summaries, events, output_file):
    """生成审计报告"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VEC/MAPPO 系统审计报告\n")
        f.write("="*80 + "\n\n")
        
        # (1) RSU mask统计
        f.write("【1. RSU Mask可用性】\n")
        rsu_mask_rates = [s['rsu_mask_true_rate'] for s in summaries]
        f.write(f"  均值: {np.mean(rsu_mask_rates):.2%}\n")
        f.write(f"  中位数: {np.median(rsu_mask_rates):.2%}\n")
        zero_mask_eps = sum(1 for r in rsu_mask_rates if r < 0.01)
        f.write(f"  ⚠️  RSU mask≈0的episode: {zero_mask_eps}/{len(summaries)}\n")
        if zero_mask_eps > len(summaries) * 0.1:
            f.write(f"  ❌ FAIL: RSU被长期mask，检查mask逻辑/队列释放\n")
        f.write("\n")
        
        # (2) V2V可选数
        f.write("【2. V2V可选邻居数】\n")
        v2v_means = [s['valid_v2v_count_mean'] for s in summaries]
        f.write(f"  均值: {np.mean(v2v_means):.1f}\n")
        f.write(f"  最大: {np.max(v2v_means):.1f}\n")
        if np.mean(v2v_means) > 8:
            f.write(f"  ⚠️  V2V邻居数过多，可能导致类别不平衡\n")
        f.write("\n")
        
        # (3) Illegal统计
        f.write("【3. Illegal动作】\n")
        total_illegal = sum(s['illegal_total'] for s in summaries)
        total_steps = sum(s['steps'] for s in summaries)
        f.write(f"  总Illegal: {total_illegal}\n")
        f.write(f"  Illegal率: {total_illegal/max(1, total_steps):.2%}\n")
        f.write(f"  Local: {sum(s['illegal_count_local'] for s in summaries)}\n")
        f.write(f"  RSU: {sum(s['illegal_count_rsu'] for s in summaries)}\n")
        f.write(f"  V2V: {sum(s['illegal_count_v2v'] for s in summaries)}\n")
        
        # 检查mask与illegal一致性
        mask_illegal_conflicts = [e for e in events if e['type'] == 'illegal_action' and e.get('mask_was_true', False)]
        if mask_illegal_conflicts:
            f.write(f"  ❌ CRITICAL: {len(mask_illegal_conflicts)} 次mask=True但illegal（mask与env不一致）\n")
        f.write("\n")
        
        # (5) V2V生命周期
        f.write("【5. V2V生命周期守恒】\n")
        total_v2v_lifecycle = {
            'tx_started': sum(s['v2v_tx_started'] for s in summaries),
            'tx_done': sum(s['v2v_tx_done'] for s in summaries),
            'received': sum(s['v2v_received'] for s in summaries),
            'added_to_active': sum(s['v2v_added_to_active'] for s in summaries),
            'cpu_finished': sum(s['v2v_cpu_finished'] for s in summaries),
            'dag_completed': sum(s['v2v_dag_completed'] for s in summaries),
        }
        f.write(f"  TX Started:      {total_v2v_lifecycle['tx_started']}\n")
        f.write(f"  TX Done:         {total_v2v_lifecycle['tx_done']}\n")
        f.write(f"  Received:        {total_v2v_lifecycle['received']}\n")
        f.write(f"  Added to Active: {total_v2v_lifecycle['added_to_active']}\n")
        f.write(f"  CPU Finished:    {total_v2v_lifecycle['cpu_finished']}\n")
        f.write(f"  DAG Completed:   {total_v2v_lifecycle['dag_completed']}\n")
        
        breach_count = sum(1 for s in summaries if s['v2v_lifecycle_breach'])
        if breach_count > 0:
            f.write(f"  ❌ {breach_count}/{len(summaries)} episodes违反生命周期守恒\n")
            # 找到断层位置
            max_val = max(total_v2v_lifecycle.values())
            min_val = min(total_v2v_lifecycle.values())
            if max_val - min_val > max(1, int(0.01 * max_val)):
                for k, v in total_v2v_lifecycle.items():
                    if v < max_val * 0.95:
                        f.write(f"  ⚠️  断点: {k} (仅{v}/{max_val})\n")
        f.write("\n")
        
        # (10) Latency
        f.write("【10. Latency指标】\n")
        has_negative = sum(1 for s in summaries if s['wall_clock_latency_has_negative'])
        if has_negative > 0:
            f.write(f"  ❌ CRITICAL: {has_negative}/{len(summaries)} episodes出现负latency\n")
        
        avg_wall_latency = np.mean([s['wall_clock_latency_mean'] for s in summaries if s['wall_clock_latency_mean'] > 0])
        f.write(f"  Wall Clock Latency均值: {avg_wall_latency:.3f}s\n")
        f.write("\n")
        
        # [P2/P0新增] 关键健康指标
        f.write("【13. 死锁与Deadline检查】\n")
        deadlock_counts = [s['deadlock_vehicle_count'] for s in summaries]
        deadline_checks = [s['audit_deadline_checks'] for s in summaries]
        deadline_misses = [s['audit_deadline_misses'] for s in summaries]
        f.write(f"  Deadlock车辆数: mean={np.mean(deadlock_counts):.1f}, max={np.max(deadlock_counts)}\n")
        f.write(f"  Deadline检查次数: mean={np.mean(deadline_checks):.0f}, total={np.sum(deadline_checks)}\n")
        f.write(f"  Deadline Miss次数: mean={np.mean(deadline_misses):.1f}, total={np.sum(deadline_misses)}\n")
        if np.sum(deadlock_counts) > 0:
            f.write(f"  ❌ {np.sum(deadlock_counts)} 辆车出现死锁\n")
        f.write("\n")
        
        f.write("【14. 失败原因分解】\n")
        miss_reason_deadline = [s['miss_reason_deadline'] for s in summaries]
        miss_reason_overflow = [s['miss_reason_overflow'] for s in summaries]
        miss_reason_illegal = [s['miss_reason_illegal'] for s in summaries]
        miss_reason_unfinished = [s['miss_reason_unfinished'] for s in summaries]
        miss_reason_truncated = [s['miss_reason_truncated'] for s in summaries]
        f.write(f"  Deadline: mean={np.mean(miss_reason_deadline):.1f}, total={np.sum(miss_reason_deadline)}\n")
        f.write(f"  Overflow: mean={np.mean(miss_reason_overflow):.1f}, total={np.sum(miss_reason_overflow)}\n")
        f.write(f"  Illegal: mean={np.mean(miss_reason_illegal):.1f}, total={np.sum(miss_reason_illegal)}\n")
        f.write(f"  Unfinished: mean={np.mean(miss_reason_unfinished):.1f}, total={np.sum(miss_reason_unfinished)}\n")
        f.write(f"  Truncated: mean={np.mean(miss_reason_truncated):.1f}, total={np.sum(miss_reason_truncated)}\n")
        f.write("\n")
        
        f.write("【15. 传输任务统计（P0护栏）】\n")
        tx_created = [s['tx_tasks_created_count'] for s in summaries]
        same_node_no_tx = [s['same_node_no_tx_count'] for s in summaries]
        f.write(f"  TX创建次数: mean={np.mean(tx_created):.1f}, total={np.sum(tx_created)}\n")
        f.write(f"  同节点不传输次数: mean={np.mean(same_node_no_tx):.1f}, total={np.sum(same_node_no_tx)}\n")
        f.write("\n")
        
        f.write("【16. 成功率统计口径拆分】\n")
        vehicle_srs = [s['vehicle_success_rate'] for s in summaries]
        task_srs = [s['task_success_rate'] for s in summaries]
        episode_all_srs = [s['episode_all_success'] for s in summaries]
        subtask_srs = [s['subtask_success_rate'] for s in summaries]
        f.write(f"  Vehicle SR: mean={np.mean(vehicle_srs):.2%}, p50={np.percentile(vehicle_srs, 50):.2%}, p90={np.percentile(vehicle_srs, 90):.2%}\n")
        f.write(f"  Task SR: mean={np.mean(task_srs):.2%}, p50={np.percentile(task_srs, 50):.2%}, p90={np.percentile(task_srs, 90):.2%}\n")
        f.write(f"  Episode All Success: mean={np.mean(episode_all_srs):.2%}, count={np.sum(episode_all_srs)}/{len(summaries)}\n")
        f.write(f"  Subtask SR: mean={np.mean(subtask_srs):.2%}, p50={np.percentile(subtask_srs, 50):.2%}, p90={np.percentile(subtask_srs, 90):.2%}\n")
        f.write("\n")
        
        f.write("【17. P2性能统计】\n")
        service_rates = [s['service_rate_when_active'] for s in summaries if s['service_rate_when_active'] > 0]
        idle_fractions = [s['idle_fraction'] for s in summaries]
        if service_rates:
            f.write(f"  Service Rate (when active): mean={np.mean(service_rates)/1e9:.3f} Gcycles/s, "
                   f"p50={np.percentile(service_rates, 50)/1e9:.3f} Gcycles/s, "
                   f"p90={np.percentile(service_rates, 90)/1e9:.3f} Gcycles/s\n")
        else:
            f.write(f"  Service Rate (when active): 无数据（所有episode active_time=0）\n")
        f.write(f"  Idle Fraction: mean={np.mean(idle_fractions):.2%}, p50={np.percentile(idle_fractions, 50):.2%}, "
               f"p90={np.percentile(idle_fractions, 90):.2%}\n")
        f.write("\n")
        
        # 总结
        f.write("="*80 + "\n")
        f.write("【审计总结】\n")
        f.write("="*80 + "\n")
        
        failures = []
        if zero_mask_eps > len(summaries) * 0.1:
            failures.append("RSU长期被mask")
        if mask_illegal_conflicts:
            failures.append("Mask与illegal不一致（致命）")
        if breach_count > 0:
            failures.append("V2V生命周期断裂")
        if has_negative > 0:
            failures.append("负Latency（metric bug）")
        if np.sum(deadlock_counts) > 0:
            failures.append(f"死锁（{np.sum(deadlock_counts)}辆车）")
        if np.mean(episode_all_srs) < 0.5 and len(summaries) >= 10:
            failures.append(f"Episode All Success率过低（{np.mean(episode_all_srs):.1%}）")
        
        if failures:
            f.write("❌ 发现以下问题：\n")
            for fail in failures:
                f.write(f"  - {fail}\n")
        else:
            f.write("✅ 所有核心审计项通过\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_audit(args.episodes, args.seed)

