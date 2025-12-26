"""
FIFO队列系统模块

实现严格的FIFO队列，用于车辆和RSU的任务管理。
核心功能：
- 严格FIFO逻辑
- 精确的排队时间计算
- 队列溢出检查
- 时间推演（基于CPU频率和计算量）
"""

import numpy as np
from collections import deque


class FIFOQueue:
    """
    FIFO队列系统
    
    职责：
    - 管理任务的严格FIFO顺序
    - 跟踪队列中每个任务的计算量
    - 计算排队时间
    - 处理队列溢出
    
    注意：保持最简单的FIFO逻辑，不实现插队等复杂功能
    """
    
    def __init__(self, max_buffer_size=20):
        """
        初始化FIFO队列
        
        Args:
            max_buffer_size: 队列最大容量（任务数量）
        """
        self.max_buffer_size = max_buffer_size
        # 队列：存储每个任务的计算量（cycles）
        self._queue = deque()
        # 当前正在执行的任务的剩余计算量
        self._current_task_remaining = 0.0
    
    def enqueue(self, comp_cycles):
        """
        入队操作
        
        Args:
            comp_cycles: 任务的计算量（cycles）
        
        Returns:
            bool: 如果成功入队返回True，如果队列已满返回False
        """
        # 入队检查：如果队列已满，拒绝新任务
        if len(self._queue) >= self.max_buffer_size:
            return False
        
        # 将任务加入队列
        self._queue.append(comp_cycles)
        return True
    
    def step(self, dt, cpu_freq):
        """
        时间推演：推进队列中任务的计算进度
        
        严格FIFO：只有当前任务完成后，才处理下一个任务
        
        Args:
            dt: 时间步长（秒）
            cpu_freq: CPU频率（Hz，cycles/s）
        
        Returns:
            int: 完成的任务数量（通常为0或1）
        """
        if len(self._queue) == 0 and self._current_task_remaining <= 0:
            return 0  # 队列为空，无事可做
        
        # 如果当前没有正在执行的任务，从队列中取出下一个
        if self._current_task_remaining <= 0:
            if len(self._queue) > 0:
                self._current_task_remaining = self._queue.popleft()
            else:
                return 0
        
        # 推进当前任务的计算进度
        # 计算量 = 时间 * CPU频率
        computed = dt * cpu_freq
        self._current_task_remaining -= computed
        
        # 如果当前任务完成，返回1表示完成了一个任务
        if self._current_task_remaining <= 0:
            self._current_task_remaining = 0.0
            return 1
        
        return 0
    
    def get_estimated_wait_time(self, cpu_freq):
        """
        获取估计的等待时间
        
        公式：wait_time = (当前任务剩余 + 队列中所有任务) / cpu_freq
        
        Args:
            cpu_freq: CPU频率（Hz）
        
        Returns:
            float: 估计等待时间（秒）
        """
        if cpu_freq <= 0:
            return 0.0
        
        # 计算当前任务剩余时间
        current_wait = max(0.0, self._current_task_remaining) / cpu_freq
        
        # 计算队列中所有任务的等待时间
        queue_wait = sum(self._queue) / cpu_freq
        
        return current_wait + queue_wait
    
    def get_current_size(self):
        """
        获取当前队列大小（包括正在执行的任务）
        
        Returns:
            int: 队列中的任务数量（包括正在执行的任务）
        """
        size = len(self._queue)
        if self._current_task_remaining > 0:
            size += 1
        return size
    
    def get_queue_length(self):
        """
        获取队列长度（不包括正在执行的任务）
        
        Returns:
            int: 等待中的任务数量
        """
        return len(self._queue)
    
    def dequeue_one(self):
        """
        从队列中移除一个任务（FIFO顺序）
        
        注意：这个方法用于任务完成时手动移除队列中的任务
        由于队列只存储计算量，我们假设按照FIFO顺序移除
        
        Returns:
            bool: 如果成功移除返回True，队列为空返回False
        """
        if len(self._queue) > 0:
            self._queue.popleft()
            return True
        return False
    
    def is_full(self):
        """
        检查队列是否已满
        
        Returns:
            bool: 如果队列已满返回True
        """
        return len(self._queue) >= self.max_buffer_size
    
    def clear(self):
        """清空队列"""
        self._queue.clear()
        self._current_task_remaining = 0.0
    
    def __len__(self):
        """返回队列长度（等待中的任务数）"""
        return len(self._queue)
    
    def __repr__(self):
        return f"FIFOQueue(size={len(self._queue)}/{self.max_buffer_size}, current_remaining={self._current_task_remaining:.2e})"

