"""
RSU（路侧单元）实体类

实现RSU作为独立的实体，支持：
- 多个RSU实例
- 多处理器队列管理
- 位置和覆盖范围
- 计算资源管理
"""

import numpy as np
from configs.config import SystemConfig as Cfg
from envs.modules.rsu_queue_manager import RSUQueueManager


class RSU:
    """
    RSU（Road Side Unit）路侧单元实体
    
    功能：
    - 管理位置和覆盖范围
    - 多处理器队列管理
    - 计算资源（CPU频率）
    - 任务分配和等待时间计算
    """
    
    def __init__(self, rsu_id, position, cpu_freq=None, num_processors=None, queue_limit=None, coverage_range=None):
        """
        初始化RSU实体
        
        Args:
            rsu_id: RSU唯一ID
            position: RSU位置坐标 [x, y] (m)
            cpu_freq: CPU频率 (Hz)，如果为None则使用配置值
            num_processors: 处理器数量，如果为None则使用配置值
            queue_limit: 队列上限，如果为None则使用配置值
            coverage_range: 覆盖半径 (m)，如果为None则使用配置值
        """
        self.id = rsu_id
        self.position = np.array(position, dtype=float)
        
        # 计算资源
        self.cpu_freq = cpu_freq if cpu_freq is not None else Cfg.F_RSU
        self.num_processors = num_processors if num_processors is not None else getattr(Cfg, 'RSU_NUM_PROCESSORS', 4)
        
        # 覆盖范围
        self.coverage_range = coverage_range if coverage_range is not None else Cfg.RSU_RANGE
        
        # 队列管理器（多处理器队列）
        queue_limit_val = queue_limit if queue_limit is not None else Cfg.RSU_QUEUE_LIMIT
        self.queue_manager = RSUQueueManager(
            num_processors=self.num_processors,
            queue_limit_per_processor=queue_limit_val
        )
        
        # 向后兼容：队列长度属性
        self.queue_length = 0
    
    def is_in_coverage(self, position):
        """
        检查位置是否在RSU覆盖范围内
        
        Args:
            position: 位置坐标 [x, y] (m)
        
        Returns:
            bool: 如果在覆盖范围内返回True
        """
        dist = np.linalg.norm(np.array(position) - self.position)
        return dist <= self.coverage_range
    
    def get_distance(self, position):
        """
        获取到指定位置的距离
        
        Args:
            position: 位置坐标 [x, y] (m)
        
        Returns:
            float: 距离 (m)
        """
        return np.linalg.norm(np.array(position) - self.position)
    
    def enqueue_task(self, comp_cycles):
        """
        将任务加入队列（选择负载最低的处理器）
        
        Args:
            comp_cycles: 任务的计算量（cycles）
        
        Returns:
            int or None: 成功时返回分配的处理器ID，失败返回None
        """
        processor_id = self.queue_manager.enqueue(comp_cycles)
        self.queue_length = self.queue_manager.get_queue_length()
        return processor_id
    
    def dequeue_task(self):
        """
        从队列中移除一个任务（FIFO顺序）
        
        Returns:
            bool: 如果成功移除返回True
        """
        success = self.queue_manager.dequeue_one()
        self.queue_length = self.queue_manager.get_queue_length()
        return success
    
    def get_estimated_wait_time(self):
        """
        获取估计的等待时间（负载最低处理器的等待时间）
        
        Returns:
            float: 等待时间（秒）
        """
        return self.queue_manager.get_min_wait_time(self.cpu_freq)
    
    def is_queue_full(self):
        """
        检查队列是否已满
        
        Returns:
            bool: 如果所有处理器队列都满返回True
        """
        return self.queue_manager.is_full()
    
    def clear_queue(self):
        """清空队列"""
        self.queue_manager.clear()
        self.queue_length = 0
    
    def get_load_normalized(self):
        """
        获取归一化的负载（0-1之间）
        
        Returns:
            float: 归一化负载
        """
        wait_time = self.get_estimated_wait_time()
        max_wait = Cfg.DYNAMIC_MAX_WAIT_TIME if hasattr(Cfg, 'DYNAMIC_MAX_WAIT_TIME') else 10.0
        return np.clip(wait_time / max_wait, 0.0, 1.0)
    
    def update_queue_sync(self):
        """同步队列长度（用于向后兼容）"""
        self.queue_length = self.queue_manager.get_queue_length()
    
    def __repr__(self):
        return f"RSU(id={self.id}, pos={self.position}, queue_len={self.queue_length}/{self.queue_manager.num_processors * self.queue_manager.queue_limit_per_processor})"

