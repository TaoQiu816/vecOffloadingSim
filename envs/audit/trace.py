"""
Trace收集器：订阅环境事件并记录详细trace

职责：
- 订阅step事件（TX, CPU, TASK等）
- 记录详细的数值trace（用于调试和审计）
- 支持开关控制（默认关闭，不影响性能）

设计原则：
- 观察者模式（不修改环境状态）
- 默认关闭（不影响determinism）
- 可选输出格式（dict, json, csv等）
"""
from typing import Dict, List, Any
import numpy as np


class TraceCollector:
    """
    Trace收集器：订阅环境事件并记录详细trace
    
    【重要】默认关闭，不影响环境行为和性能。
    """
    
    def __init__(self, enabled: bool = False):
        """
        Args:
            enabled: 是否启用trace收集（默认False）
        """
        self.enabled = enabled
        self.traces = []
        self.step_idx = 0
    
    def reset(self):
        """重置trace记录"""
        self.traces = []
        self.step_idx = 0
    
    def on_step_start(self, step_idx: int, time_now: float, dt: float):
        """
        步骤开始事件
        
        Args:
            step_idx: 步骤索引
            time_now: 当前时间
            dt: 时间步长
        """
        if not self.enabled:
            return
        
        self.step_idx = step_idx
        self.traces.append({
            "event": "STEP_START",
            "step_idx": step_idx,
            "time": time_now,
            "dt": dt
        })
    
    def on_transfer_progress(self, job: Any, bytes_sent: float, time_used: float,
                            rate: float, energy: float, power_dbm: float):
        """
        传输推进事件
        
        Args:
            job: TransferJob对象
            bytes_sent: 本步发送字节数
            time_used: 本步使用时间
            rate: 传输速率
            energy: 能耗
            power_dbm: 发射功率
        """
        if not self.enabled:
            return
        
        self.traces.append({
            "event": "TX",
            "step_idx": self.step_idx,
            "kind": job.kind,
            "owner_vehicle_id": job.owner_vehicle_id,
            "subtask_id": job.subtask_id,
            "rem_bytes_before": job.rem_bytes + bytes_sent,
            "rem_bytes_after": job.rem_bytes,
            "bytes_sent": bytes_sent,
            "time_used_s": time_used,
            "rate_bps": rate,
            "energy_joule": energy,
            "power_dbm": power_dbm,
            "link_type": job.link_type
        })
    
    def on_compute_progress(self, job: Any, cycles_done: float, time_used: float,
                           speed_hz: float, energy: float):
        """
        计算推进事件
        
        Args:
            job: ComputeJob对象
            cycles_done: 本步完成周期数
            time_used: 本步使用时间
            speed_hz: CPU频率
            energy: 能耗
        """
        if not self.enabled:
            return
        
        self.traces.append({
            "event": "CPU",
            "step_idx": self.step_idx,
            "owner_vehicle_id": job.owner_vehicle_id,
            "subtask_id": job.subtask_id,
            "rem_cycles_before": job.rem_cycles + cycles_done,
            "rem_cycles_after": job.rem_cycles,
            "cycles_done": cycles_done,
            "time_used_s": time_used,
            "speed_hz": speed_hz,
            "energy_joule": energy,
            "exec_node": job.exec_node
        })
    
    def on_task_state_change(self, vehicle_id: int, subtask_id: int,
                            old_status: int, new_status: int,
                            exec_location: Any, input_ready: bool, edge_ready: bool):
        """
        任务状态变化事件
        
        Args:
            vehicle_id: 车辆ID
            subtask_id: 子任务ID
            old_status: 旧状态
            new_status: 新状态
            exec_location: 执行位置
            input_ready: INPUT数据是否就绪
            edge_ready: EDGE数据是否就绪
        """
        if not self.enabled:
            return
        
        self.traces.append({
            "event": "TASK",
            "step_idx": self.step_idx,
            "vehicle_id": vehicle_id,
            "subtask_id": subtask_id,
            "old_status": old_status,
            "new_status": new_status,
            "exec_location": str(exec_location),
            "input_ready": input_ready,
            "edge_ready": edge_ready
        })
    
    def get_traces(self) -> List[Dict[str, Any]]:
        """获取所有trace记录"""
        return self.traces
    
    def export_json(self, filepath: str):
        """导出trace为JSON文件"""
        if not self.enabled:
            return
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.traces, f, indent=2, default=str)

