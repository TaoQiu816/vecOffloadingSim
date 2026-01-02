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
from envs.modules.active_task_manager import ActiveTaskManager, ActiveTask


class RSU:
    """
    RSU（Road Side Unit）路侧单元实体
    
    功能：
    - 管理位置和覆盖范围
    - 多处理器队列管理
    - 计算资源（CPU频率）
    - 任务分配和等待时间计算
    - [新增] 统一处理器共享物理模型
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
        
        # [新增] 活跃任务管理器：统一管理所有接收的任务
        self.active_task_manager = ActiveTaskManager(
            num_processors=self.num_processors, 
            cpu_freq=self.cpu_freq
        )
        
        # [保留] 旧的队列管理器（用于向后兼容和容量检查）
        queue_limit_val = queue_limit if queue_limit is not None else Cfg.RSU_QUEUE_LIMIT
        self.queue_manager = RSUQueueManager(
            num_processors=self.num_processors,
            queue_limit_per_processor=queue_limit_val
        )
        
        # 向后兼容：队列长度属性
        self.queue_length = 0
        
        # FAT管理：每个处理器独立维护FAT
        # 格式: {processor_id: FAT_value}
        self.fat_processors = {i: 0.0 for i in range(self.num_processors)}
    
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
            tuple: (success: bool, processor_id: int or None)
                success: 如果成功移除返回True
                processor_id: 被移除任务所在的处理器ID，如果失败则为None
        """
        success, processor_id = self.queue_manager.dequeue_one()
        self.queue_length = self.queue_manager.get_queue_length()
        return (success, processor_id)
    
    def get_estimated_delay(self):
        """
        获取基于处理器共享模型的估计延迟（新版）
        
        使用ActiveTaskManager计算当前负载下的平均剩余时间。
        
        Returns:
            float: 估计延迟（秒）
        """
        return self.active_task_manager.get_estimated_delay()
    
    def get_estimated_wait_time(self):
        """
        [已弃用] 获取估计的等待时间（基于旧队列模型）
        
        ⚠️  警告：此方法使用旧的FIFO队列模型，不反映处理器共享物理。
        推荐使用 get_estimated_delay() 替代。
        
        保留此方法仅用于向后兼容和容量检查。
        
        Returns:
            float: 等待时间（秒）
        """
        return self.queue_manager.get_min_wait_time(self.cpu_freq)
    
    def is_queue_full(self, new_task_cycles=0):
        """
        检查队列是否已满（基于计算量）
        
        Args:
            new_task_cycles: 要添加的新任务计算量（用于检查加入后是否溢出），默认为0表示仅检查当前状态
        
        Returns:
            bool: 如果所有处理器队列都满返回True
        """
        return self.queue_manager.is_full(new_task_cycles=new_task_cycles)
    
    def clear_queue(self):
        """清空队列"""
        self.queue_manager.clear()
        self.queue_length = 0
    
    def reset_fat(self):
        """重置所有处理器的FAT为初始值（在reset()时调用）"""
        self.fat_processors = {i: 0.0 for i in range(self.num_processors)}
    
    def get_processor_fat(self, processor_id):
        """获取指定处理器的FAT"""
        return self.fat_processors.get(processor_id, 0.0)
    
    def update_processor_fat(self, processor_id, new_fat):
        """更新指定处理器的FAT"""
        if processor_id in self.fat_processors:
            self.fat_processors[processor_id] = new_fat
    
    def get_min_processor_fat(self):
        """获取所有处理器中的最小FAT（用于负载均衡）"""
        if not self.fat_processors:
            return 0.0
        return min(self.fat_processors.values())
    
    def get_load_normalized(self):
        """
        获取归一化的负载（0-1之间）
        
        Returns:
            float: 归一化负载
        """
        wait_time = self.get_estimated_wait_time()
        max_wait = Cfg.DYNAMIC_MAX_WAIT_TIME if hasattr(Cfg, 'DYNAMIC_MAX_WAIT_TIME') else 10.0
        return np.clip(wait_time / max_wait, 0.0, 1.0)
    
    def step(self, dt: float, global_step_id: int = -1) -> list:
        """
        [统一处理器共享物理模型]
        推进RSU上所有活跃任务的执行
        
        处理器共享逻辑：
        - 多个处理器提供并行能力
        - total_capacity = num_processors * cpu_freq
        - effective_speed = total_capacity / max(1, num_active_tasks)
        - 修复"无限并行"bug：任务数增加时速度下降
        
        Args:
            dt: 时间步长 (秒)
            global_step_id: 全局step编号（用于双重推进检测）
            
        Returns:
            list: 本step完成的ActiveTask列表
        """
        return self.active_task_manager.step(dt, global_step_id=global_step_id)
    
    def add_active_task(self, owner_id: int, subtask_id: int, task_type: str, 
                       total_comp: float, current_time: float = 0.0,
                       # [关键] 准入验证参数（必须由调用方提供）
                       is_dag_ready: bool = True,
                       is_data_ready: bool = True,
                       task_status: str = 'READY') -> bool:
        """
        添加活跃任务到RSU处理器（含完整准入验证）
        
        [硬断言] 准入条件（由ActiveTask强制执行）：
        1. task_status in ['READY', 'RUNNING']
        2. is_dag_ready=True（DAG依赖已满足）
        3. is_data_ready=True（数据传输已完成）
        4. total_comp > 0
        
        Args:
            owner_id: 任务所属车辆ID
            subtask_id: 子任务ID
            task_type: 通常为 'v2i'
            total_comp: 总计算量 (cycles)
            current_time: 当前时间戳
            is_dag_ready: DAG依赖是否已满足
            is_data_ready: 数据是否已传输完成
            task_status: 任务状态（'READY'或'RUNNING'）
            
        Returns:
            bool: 是否成功添加
            
        Raises:
            AssertionError: 准入条件不满足时触发
        """
        task = ActiveTask(owner_id, subtask_id, task_type, total_comp, current_time,
                         is_dag_ready=is_dag_ready,
                         is_data_ready=is_data_ready,
                         task_status=task_status)
        return self.active_task_manager.add_task(task)
    
    def remove_active_task(self, owner_id: int, subtask_id: int):
        """移除指定的活跃任务"""
        return self.active_task_manager.remove_task(owner_id, subtask_id)
    
    def get_active_task(self, owner_id: int, subtask_id: int):
        """获取指定的活跃任务（不移除）"""
        return self.active_task_manager.get_task(owner_id, subtask_id)
    
    def get_num_active_tasks(self) -> int:
        """获取活跃任务数量"""
        return self.active_task_manager.get_num_active_tasks()
    
    def update_queue_sync(self):
        """同步队列长度（用于向后兼容）"""
        self.queue_length = self.queue_manager.get_queue_length()
    
    def __repr__(self):
        return f"RSU(id={self.id}, pos={self.position}, active_tasks={self.get_num_active_tasks()}, queue_len={self.queue_length}/{self.queue_manager.num_processors * self.queue_manager.queue_limit_per_processor})"

