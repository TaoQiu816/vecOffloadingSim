"""
RSU（路侧单元）实体类

[P36修复] 队列架构简化：
- 移除 active_task_manager (处理器共享模型)
- 保留 RSUQueueManager 仅用于容量检查
- 实际任务执行由环境的 CpuQueueService 处理

功能：
- 多个RSU实例
- 位置和覆盖范围
- 容量检查和延迟估计
"""

import numpy as np
from configs.config import SystemConfig as Cfg
from envs.modules.rsu_queue_manager import RSUQueueManager


class RSU:
    """
    RSU（Road Side Unit）路侧单元实体

    功能：
    - 管理位置和覆盖范围
    - 容量检查（基于计算量）
    - 延迟估计
    - [P36] 实际任务执行由环境的CpuQueueService处理
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

        # [P36] 容量跟踪器：仅用于容量检查和延迟估计
        # 实际任务执行由环境的rsu_cpu_q处理
        queue_limit_val = queue_limit if queue_limit is not None else 100
        self.capacity_tracker = RSUQueueManager(
            num_processors=self.num_processors,
            queue_limit_per_processor=queue_limit_val
        )

        # 向后兼容：队列长度属性
        self.queue_length = 0

        # FAT管理：每个处理器独立维护FAT
        # 格式: {processor_id: FAT_value}
        self.fat_processors = {i: 0.0 for i in range(self.num_processors)}

    def _compute_processor_fats(self, current_time=0.0):
        """
        计算每个处理器的FAT（基于当前剩余计算量/频率）

        定义：FAT[p] = current_time + (sum(rem_cycles_in_queue) + current_running_rem) / cpu_freq
        """
        fats = []
        for queue in self.capacity_tracker.processor_queues:
            load = queue.get_total_load_with_current() if hasattr(queue, "get_total_load_with_current") else queue.get_total_load()
            fat = current_time + load / max(self.cpu_freq, 1e-9)
            fats.append(fat)
        return fats

    def get_earliest_available_processor(self, current_time=None):
        """
        返回FAT最小的处理器ID（平局选编号最小）

        Args:
            current_time: 当前时间，用于FAT基准；默认为0.0
        """
        if current_time is None:
            current_time = 0.0
        fats = self._compute_processor_fats(current_time=current_time)
        if not fats:
            return 0
        min_fat = min(fats)
        # tie-break by processor id
        for pid, fat in enumerate(fats):
            if abs(fat - min_fat) <= 1e-12:
                return pid
        return 0

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

    def sync_capacity_from_queues(self, processor_queues: dict):
        """
        [P36] 从环境的CPU队列同步容量跟踪器

        Args:
            processor_queues: {processor_id: deque[ComputeJob]} - 环境维护的实际FIFO队列
        """
        self.capacity_tracker.clear()
        for proc_id, queue in processor_queues.items():
            for job in queue:
                self.capacity_tracker.processor_queues[proc_id].enqueue(job.rem_cycles)
        self.queue_length = self.capacity_tracker.get_queue_length()

    def get_estimated_delay(self):
        """
        获取基于FIFO队列的估计延迟

        计算公式: 最小处理器等待时间

        Returns:
            float: 估计延迟（秒）
        """
        return self.capacity_tracker.get_min_wait_time(self.cpu_freq)

    def get_estimated_wait_time(self):
        """
        获取估计的等待时间（基于队列模型）

        Returns:
            float: 等待时间（秒）
        """
        return self.capacity_tracker.get_min_wait_time(self.cpu_freq)

    def is_queue_full(self, new_task_cycles=0):
        """
        检查队列是否已满（基于计算量）

        Args:
            new_task_cycles: 要添加的新任务计算量（用于检查加入后是否溢出），默认为0表示仅检查当前状态

        Returns:
            bool: 如果所有处理器队列都满返回True
        """
        return self.capacity_tracker.is_full(new_task_cycles=new_task_cycles)

    def clear_queue(self):
        """清空队列"""
        self.capacity_tracker.clear()
        self.queue_length = 0

    def reset_fat(self):
        """重置所有处理器的FAT为初始值（在reset()时调用）"""
        self.fat_processors = {i: 0.0 for i in range(self.num_processors)}

    def reset_capacity_tracker(self):
        """重置容量跟踪器（在reset()时调用）"""
        self.capacity_tracker.clear()
        self.queue_length = 0

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

    def get_queue_load(self):
        """
        获取当前队列总负载（总剩余计算量）

        Returns:
            float: 剩余计算量 (cycles)
        """
        return self.capacity_tracker.get_total_load()

    def update_queue_sync(self):
        """同步队列长度（用于向后兼容）"""
        self.queue_length = self.capacity_tracker.get_queue_length()

    def __repr__(self):
        return f"RSU(id={self.id}, pos={self.position}, queue_len={self.queue_length}/{self.num_processors * self.capacity_tracker.queue_limit_per_processor})"
