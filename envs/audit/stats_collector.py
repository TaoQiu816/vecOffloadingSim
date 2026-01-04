"""
统计收集器：聚合episode级别的统计信息

职责：
- 收集episode级别的统计（SR, makespan, 死锁等）
- 聚合多episode统计（均值、方差等）
- 支持开关控制（默认开启，用于训练监控）

设计原则：
- 观察者模式（不修改环境状态）
- 轻量级（不影响性能）
- 可选输出格式（dict, tensorboard等）
"""
from typing import Dict, List, Any
import numpy as np


class StatsCollector:
    """
    统计收集器：聚合episode级别的统计信息
    
    【重要】默认开启，用于训练监控，但不影响环境行为。
    """
    
    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: 是否启用统计收集（默认True）
        """
        self.enabled = enabled
        self.reset_episode_stats()
        self.reset_global_stats()
    
    def reset_episode_stats(self):
        """重置episode级别统计"""
        self.episode_stats = {
            "steps": 0,
            "decisions": 0,
            "decision_local": 0,
            "decision_rsu": 0,
            "decision_v2v": 0,
            "tasks_completed": 0,
            "tasks_total": 0,
            "dags_completed": 0,
            "dags_total": 0,
            "dags_deadline_miss": 0,
            "energy_tx_input": 0.0,
            "energy_tx_edge": 0.0,
            "energy_cpu_local": 0.0,
            "makespan_sum": 0.0,
            "makespan_count": 0,
        }
    
    def reset_global_stats(self):
        """重置全局统计"""
        self.global_stats = {
            "episodes": 0,
            "sr_sum": 0.0,
            "makespan_sum": 0.0,
            "makespan_count": 0,
            "energy_sum": 0.0,
        }
    
    def on_decision(self, vehicle_id: int, target: Any):
        """
        决策事件
        
        Args:
            vehicle_id: 车辆ID
            target: 卸载目标
        """
        if not self.enabled:
            return
        
        self.episode_stats["decisions"] += 1
        
        if target == 'Local':
            self.episode_stats["decision_local"] += 1
        elif isinstance(target, tuple) and target[0] == 'RSU':
            self.episode_stats["decision_rsu"] += 1
        elif isinstance(target, int):
            self.episode_stats["decision_v2v"] += 1
    
    def on_task_complete(self, vehicle_id: int, subtask_id: int, makespan: float):
        """
        任务完成事件
        
        Args:
            vehicle_id: 车辆ID
            subtask_id: 子任务ID
            makespan: 任务完成时间
        """
        if not self.enabled:
            return
        
        self.episode_stats["tasks_completed"] += 1
    
    def on_dag_complete(self, vehicle_id: int, makespan: float, deadline: float,
                       deadline_miss: bool):
        """
        DAG完成事件
        
        Args:
            vehicle_id: 车辆ID
            makespan: DAG完成时间
            deadline: 截止时间
            deadline_miss: 是否超时
        """
        if not self.enabled:
            return
        
        self.episode_stats["dags_completed"] += 1
        if deadline_miss:
            self.episode_stats["dags_deadline_miss"] += 1
        
        self.episode_stats["makespan_sum"] += makespan
        self.episode_stats["makespan_count"] += 1
    
    def on_energy_update(self, vehicle_id: int, tx_input: float, tx_edge: float,
                        cpu_local: float):
        """
        能耗更新事件
        
        Args:
            vehicle_id: 车辆ID
            tx_input: INPUT传输能耗
            tx_edge: EDGE传输能耗
            cpu_local: 本地计算能耗
        """
        if not self.enabled:
            return
        
        self.episode_stats["energy_tx_input"] += tx_input
        self.episode_stats["energy_tx_edge"] += tx_edge
        self.episode_stats["energy_cpu_local"] += cpu_local
    
    def on_episode_end(self, success_rate: float):
        """
        Episode结束事件
        
        Args:
            success_rate: 成功率
        """
        if not self.enabled:
            return
        
        self.global_stats["episodes"] += 1
        self.global_stats["sr_sum"] += success_rate
        
        if self.episode_stats["makespan_count"] > 0:
            avg_makespan = self.episode_stats["makespan_sum"] / self.episode_stats["makespan_count"]
            self.global_stats["makespan_sum"] += avg_makespan
            self.global_stats["makespan_count"] += 1
        
        total_energy = (self.episode_stats["energy_tx_input"] +
                       self.episode_stats["energy_tx_edge"] +
                       self.episode_stats["energy_cpu_local"])
        self.global_stats["energy_sum"] += total_energy
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """获取episode统计"""
        stats = self.episode_stats.copy()
        
        # 计算派生统计
        if stats["dags_total"] > 0:
            stats["sr"] = stats["dags_completed"] / stats["dags_total"]
        else:
            stats["sr"] = 0.0
        
        if stats["makespan_count"] > 0:
            stats["avg_makespan"] = stats["makespan_sum"] / stats["makespan_count"]
        else:
            stats["avg_makespan"] = 0.0
        
        return stats
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计"""
        stats = self.global_stats.copy()
        
        if stats["episodes"] > 0:
            stats["avg_sr"] = stats["sr_sum"] / stats["episodes"]
        else:
            stats["avg_sr"] = 0.0
        
        if stats["makespan_count"] > 0:
            stats["avg_makespan"] = stats["makespan_sum"] / stats["makespan_count"]
        else:
            stats["avg_makespan"] = 0.0
        
        if stats["episodes"] > 0:
            stats["avg_energy"] = stats["energy_sum"] / stats["episodes"]
        else:
            stats["avg_energy"] = 0.0
        
        return stats

