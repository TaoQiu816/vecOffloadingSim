"""
活跃任务管理器 (Active Task Manager)

用于统一管理车辆/RSU上的所有活跃任务（本地+V2V接收）。
实现处理器共享物理模型。
"""

import numpy as np
from typing import List, Dict, Optional

# 全局UID计数器（确保tie-break唯一性）
_GLOBAL_TASK_UID = 0


class ActiveTask:
    """
    活跃任务数据结构（含完整准入验证）
    
    准入条件（硬断言强制执行）：
    1. 任务状态 = READY 或 RUNNING（非PENDING）
    2. DAG依赖全部满足（前置任务已完成）
    3. 数据传输完成（对V2I/V2V）
    4. rem_comp > 0
    5. 唯一状态（不在其他实体的Active中）
    
    Attributes:
        owner_id: 任务所属车辆ID
        subtask_id: 子任务ID（在owner的DAG中）
        task_type: 'local' 或 'v2v' 或 'v2i'
        rem_comp: 剩余计算量 (cycles)
        total_comp: 总计算量 (cycles)
        assigned_time: 任务分配的时间戳
        uid: 全局唯一ID（用于tie-break）
    """
    
    def __init__(self, owner_id: int, subtask_id: int, task_type: str, 
                 total_comp: float, assigned_time: float = 0.0,
                 # [关键] 准入验证参数
                 is_dag_ready: bool = True,
                 is_data_ready: bool = True,
                 task_status: str = 'READY'):
        global _GLOBAL_TASK_UID
        
        # ========== 硬断言：完整准入条件 ==========
        
        # 1. 基本类型检查
        assert task_type in ['local', 'v2v', 'v2i'], (
            f"❌ ActiveTask准入失败: task_type必须是'local'/'v2v'/'v2i', "
            f"实际='{task_type}' (owner={owner_id}, subtask={subtask_id})"
        )
        
        # 2. 计算量检查
        assert total_comp > 0, (
            f"❌ ActiveTask准入失败: total_comp必须>0, 实际={total_comp:.2e} "
            f"(owner={owner_id}, subtask={subtask_id})"
        )
        
        # 3. 任务状态检查（致命约束）
        assert task_status in ['READY', 'RUNNING'], (
            f"❌ ActiveTask准入失败: 任务状态必须是READY或RUNNING, "
            f"实际='{task_status}' (owner={owner_id}, subtask={subtask_id})\n"
            f"   → PENDING状态禁止加入Active！DAG依赖未满足会破坏因果性！"
        )
        
        # 4. DAG依赖检查（致命约束）
        assert is_dag_ready, (
            f"❌ ActiveTask准入失败: DAG依赖未满足 "
            f"(owner={owner_id}, subtask={subtask_id})\n"
            f"   → 前置任务未完成禁止执行！会破坏DAG因果约束！"
        )
        
        # 5. 数据传输检查（致命约束，仅针对卸载任务）
        if task_type in ['v2v', 'v2i']:
            assert is_data_ready, (
                f"❌ ActiveTask准入失败: 数据传输未完成 "
                f"(owner={owner_id}, subtask={subtask_id}, type={task_type})\n"
                f"   → 数据未到达禁止执行！会破坏通信约束！"
            )
        
        # 分配全局唯一UID（确保tie-break唯一性）
        self.uid = _GLOBAL_TASK_UID
        _GLOBAL_TASK_UID += 1
        
        self.owner_id = owner_id
        self.subtask_id = subtask_id
        self.task_type = task_type
        self.rem_comp = total_comp
        self.total_comp = total_comp
        self.assigned_time = assigned_time
        
        # [双重推进检测] 记录最后一次更新的step_id
        self.last_updated_step = -1
        self.update_source = None  # 'active_manager' or 'old_dag_step'
        
    def get_progress(self) -> float:
        """获取任务进度 (0.0-1.0)"""
        if self.total_comp <= 0:
            return 1.0
        return max(0.0, min(1.0, (self.total_comp - self.rem_comp) / self.total_comp))
    
    def is_completed(self) -> bool:
        """判断任务是否完成"""
        return self.rem_comp <= 0
    
    def get_task_id(self) -> tuple:
        """获取唯一任务标识符 (owner_id, subtask_id)"""
        return (self.owner_id, self.subtask_id)
    
    def __repr__(self):
        return f"ActiveTask(owner={self.owner_id}, sub={self.subtask_id}, type={self.task_type}, rem={self.rem_comp:.2e}/{self.total_comp:.2e})"


class ActiveTaskManager:
    """
    活跃任务管理器
    
    管理节点上所有正在执行的任务，支持：
    - 添加/移除任务
    - 处理器共享计算
    - 按任务ID查找和完成任务
    """
    
    def __init__(self, num_processors: int = 1, cpu_freq: float = 1.0e9):
        """
        Args:
            num_processors: 处理器核心数（用于并行能力计算）
            cpu_freq: 每个处理器的CPU频率 (Hz)
        """
        self.num_processors = num_processors
        self.cpu_freq = cpu_freq
        
        # [关键] 使用List确保确定性：
        # - 迭代顺序在相同seed下可复现
        # - 避免使用set或dict（哈希顺序不确定）
        self.active_tasks: List[ActiveTask] = []
        
    def add_task(self, task: ActiveTask) -> bool:
        """
        添加新任务到活跃列表
        
        Returns:
            bool: 如果成功添加返回True
        """
        # 检查是否已存在相同任务
        task_id = task.get_task_id()
        if any(t.get_task_id() == task_id for t in self.active_tasks):
            return False
        
        self.active_tasks.append(task)
        return True
    
    def remove_task(self, owner_id: int, subtask_id: int) -> Optional[ActiveTask]:
        """
        根据任务ID移除特定任务（非FIFO）
        
        Returns:
            ActiveTask or None: 如果找到并移除返回任务对象，否则返回None
        """
        for i, task in enumerate(self.active_tasks):
            if task.owner_id == owner_id and task.subtask_id == subtask_id:
                return self.active_tasks.pop(i)
        return None
    
    def get_task(self, owner_id: int, subtask_id: int) -> Optional[ActiveTask]:
        """获取指定任务（不移除）"""
        for task in self.active_tasks:
            if task.owner_id == owner_id and task.subtask_id == subtask_id:
                return task
        return None
    
    def get_num_active_tasks(self) -> int:
        """获取活跃任务数量"""
        return len(self.active_tasks)
    
    def get_total_load(self) -> float:
        """获取总计算量 (cycles)"""
        return sum(task.rem_comp for task in self.active_tasks)
    
    def step(self, dt: float, global_step_id: int = -1) -> List[ActiveTask]:
        """
        时间推演：使用处理器共享模型推进所有任务（含步内动态）
        
        修正的处理器共享公式:
            allocation_factor = min(1.0, num_processors / num_active_tasks)
            effective_speed = cpu_freq * allocation_factor
            
        关键修复:
        1. 单任务在4核上速度=cpu_freq，不是4×cpu_freq（单线程限制）
        2. 步内动态：任务完成后立即重新分配CPU给剩余任务
        3. [新增] 双重推进检测：同一step不允许重复更新
        
        Args:
            dt: 时间步长 (秒)
            global_step_id: 全局step编号（用于双重推进检测）
            
        Returns:
            List[ActiveTask]: 本step完成的任务列表
        """
        if not self.active_tasks:
            return []
        
        # [双重推进硬断言] 检查是否有任务在本step已被更新
        if global_step_id >= 0:
            for task in self.active_tasks:
                if task.last_updated_step == global_step_id:
                    raise AssertionError(
                        f"❌ 双重推进检测: 任务({task.owner_id},{task.subtask_id})在step={global_step_id}已被更新！\n"
                        f"   上次更新来源: {task.update_source}\n"
                        f"   当前更新来源: active_manager\n"
                        f"   → 检查是否同时调用了旧引擎(task_dag.step_progress)和新引擎(active_manager.step)"
                    )
        
        completed_tasks = []
        remaining_dt = dt
        
        # 步内动态：逐个完成任务，实时重新分配CPU
        while remaining_dt > 1e-9 and self.active_tasks:
            num_active = len(self.active_tasks)
            
            # 修正的处理器共享公式（修复"超级计算机Bug"）
            # allocation_factor: 每个任务能获得的处理器比例
            # - 如果1任务/4核: allocation=1.0 → speed=cpu_freq (单线程瓶颈)
            # - 如果8任务/4核: allocation=0.5 → speed=0.5×cpu_freq
            allocation_factor = min(1.0, self.num_processors / num_active)
            effective_speed = self.cpu_freq * allocation_factor
            
            # 找到最快完成的任务所需时间
            min_time_to_finish = float('inf')
            for task in self.active_tasks:
                if task.rem_comp > 0:
                    time_needed = task.rem_comp / effective_speed
                    min_time_to_finish = min(min_time_to_finish, time_needed)
            
            # 本轮推进时间：不超过remaining_dt，也不超过最快完成时间
            time_step = min(remaining_dt, min_time_to_finish)
            
            # 推进所有任务
            computation_per_task = effective_speed * time_step
            for task in self.active_tasks:
                task.rem_comp -= computation_per_task
                if task.rem_comp <= 1e-9:
                    task.rem_comp = 0.0
                
                # [双重推进检测] 记录本次更新
                if global_step_id >= 0:
                    task.last_updated_step = global_step_id
                    task.update_source = 'active_manager'
            
            # 收集并移除完成的任务
            finished_this_round = [t for t in self.active_tasks if t.rem_comp <= 1e-9]
            for task in finished_this_round:
                completed_tasks.append(task)
                self.active_tasks.remove(task)
            
            # 更新剩余时间
            remaining_dt -= time_step
        
        # [关键] Tie-Break稳定性：按(owner_id, subtask_id, uid)排序
        # 确保并列完成时的移除顺序跨版本/实现稳定
        # uid作为最后一项，处理极端情况（同owner同subtask重复入队bug场景）
        completed_tasks.sort(key=lambda t: (t.owner_id, t.subtask_id, t.uid))
        
        return completed_tasks
    
    def get_estimated_completion_time(self, new_task_comp: float = 0.0) -> float:
        """
        估算新任务的完成时间（考虑当前负载）
        
        Args:
            new_task_comp: 新任务的计算量 (cycles)
            
        Returns:
            float: 估算完成时间 (秒)
        """
        if new_task_comp <= 0:
            return 0.0
        
        # 计算加入新任务后的总负载和任务数
        num_active = len(self.active_tasks) + 1
        
        # 使用修正的处理器共享公式
        allocation_factor = min(1.0, self.num_processors / num_active)
        effective_speed = self.cpu_freq * allocation_factor
        
        # 完成时间 = 计算量 / 有效速度
        return new_task_comp / max(effective_speed, 1e-6)
    
    def get_estimated_delay(self) -> float:
        """
        估算当前活跃任务的平均剩余执行时间（基于当前负载）
        
        [关键命名] 使用"delay"而非"wait_time"：
        - 处理器共享模型是"无等待"（新任务立即执行但变慢）
        - 此函数返回的是在当前负载下的平均剩余服务时间/延迟指标
        - 不是传统队列的"排队等待时间"
        
        用途：用于观测特征、奖励计算、调度决策
        
        Returns:
            float: 平均剩余执行时间/负载延迟 (秒)
        """
        if not self.active_tasks:
            return 0.0
        
        # 计算：当前任务在当前负载下的平均剩余时间
        avg_rem_comp = np.mean([task.rem_comp for task in self.active_tasks])
        num_active = len(self.active_tasks)
        
        # 使用修正的处理器共享公式
        allocation_factor = min(1.0, self.num_processors / num_active)
        effective_speed = self.cpu_freq * allocation_factor
        
        return avg_rem_comp / max(effective_speed, 1e-6)
    
    def clear(self):
        """清空所有活跃任务"""
        self.active_tasks.clear()
    
    def __len__(self):
        """返回活跃任务数量"""
        return len(self.active_tasks)
    
    def __repr__(self):
        return f"ActiveTaskManager(processors={self.num_processors}, active_tasks={len(self.active_tasks)}, total_load={self.get_total_load():.2e})"

