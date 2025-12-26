"""
RSU多处理器队列管理器

实现RSU的多处理器队列系统，每个处理器有独立的FIFO队列。
任务分配时选择负载最低的处理器队列。
"""

import numpy as np
from envs.modules.queue_system import FIFOQueue
from configs.config import SystemConfig as Cfg


class RSUQueueManager:
    """
    RSU多处理器队列管理器
    
    管理RSU的多个处理器，每个处理器有独立的FIFO队列。
    
    核心功能：
    - 任务分配时选择负载最低的处理器
    - 计算平均等待时间（所有处理器的平均等待时间）
    - 队列溢出检查（所有处理器队列都已满）
    - 任务完成时从对应处理器队列移除
    """
    
    def __init__(self, num_processors=None, queue_limit_per_processor=None):
        """
        初始化RSU队列管理器
        
        Args:
            num_processors: 处理器数量，如果为None则使用配置值
            queue_limit_per_processor: 每个处理器的队列上限，如果为None则使用配置值
        """
        self.num_processors = num_processors if num_processors is not None else getattr(Cfg, 'RSU_NUM_PROCESSORS', 4)
        queue_limit = queue_limit_per_processor if queue_limit_per_processor is not None else Cfg.RSU_QUEUE_LIMIT
        # 每个处理器的队列上限：总限制除以处理器数量
        self.queue_limit_per_processor = queue_limit // self.num_processors
        
        # 为每个处理器创建独立的FIFO队列
        self.processor_queues = [FIFOQueue(max_buffer_size=self.queue_limit_per_processor) 
                                 for _ in range(self.num_processors)]
        
        # 记录每个任务分配的处理器ID（用于任务完成时正确移除）
        # 结构: {task_id: processor_id}，但这里我们无法直接获取task_id
        # 所以使用FIFO顺序处理，即任务完成时从最早分配的处理器队列移除
        self._task_to_processor = []  # 记录任务分配顺序和处理器ID
    
    def enqueue(self, comp_cycles):
        """
        入队操作：选择负载最低的处理器队列
        
        Args:
            comp_cycles: 任务的计算量（cycles）
        
        Returns:
            int or None: 成功时返回分配的处理器ID，失败返回None
        """
        # 找到负载最低的处理器（队列长度最小，如果有多个相同则选择第一个未满的）
        best_processor = None
        min_load = float('inf')
        
        for proc_id, queue in enumerate(self.processor_queues):
            # 计算负载：只考虑队列中的等待任务数量
            load = queue.get_queue_length()
            
            # 如果队列未满且负载更小，则选择它
            if not queue.is_full() and load < min_load:
                min_load = load
                best_processor = proc_id
        
        # 尝试加入负载最低的处理器队列
        if best_processor is not None:
            success = self.processor_queues[best_processor].enqueue(comp_cycles)
            if success:
                self._task_to_processor.append(best_processor)
                return best_processor
            else:
                # 该处理器队列已满，检查是否所有队列都满
                if all(q.is_full() for q in self.processor_queues):
                    return None
                # 如果还有未满的队列，应该重新查找（理论上不应该发生，因为best_processor已经是负载最低的）
                # 但为了安全，这里返回None
                return None
        
        return None
    
    def dequeue_one(self):
        """
        从队列中移除一个任务（FIFO顺序，从最早分配的处理器队列移除）
        
        Returns:
            bool: 如果成功移除返回True，否则返回False
        """
        if len(self._task_to_processor) > 0:
            processor_id = self._task_to_processor.pop(0)
            if 0 <= processor_id < len(self.processor_queues):
                return self.processor_queues[processor_id].dequeue_one()
        return False
    
    def get_estimated_wait_time(self, cpu_freq_per_processor):
        """
        获取估计的等待时间（所有处理器的平均等待时间）
        
        Args:
            cpu_freq_per_processor: 每个处理器的CPU频率（Hz）
        
        Returns:
            float: 平均等待时间（秒）
        """
        wait_times = []
        for queue in self.processor_queues:
            wait_time = queue.get_estimated_wait_time(cpu_freq_per_processor)
            wait_times.append(wait_time)
        
        # 返回平均等待时间（更准确的是返回最小等待时间，因为任务会选择负载最低的处理器）
        # 但为了简化，这里返回平均等待时间
        return np.mean(wait_times) if wait_times else 0.0
    
    def get_min_wait_time(self, cpu_freq_per_processor):
        """
        获取最小等待时间（负载最低的处理器的等待时间）
        
        这更适合用于任务分配决策，因为新任务会选择负载最低的处理器
        
        Args:
            cpu_freq_per_processor: 每个处理器的CPU频率（Hz）
        
        Returns:
            float: 最小等待时间（秒）
        """
        if len(self.processor_queues) == 0:
            return 0.0
        
        wait_times = [queue.get_estimated_wait_time(cpu_freq_per_processor) 
                     for queue in self.processor_queues]
        return min(wait_times) if wait_times else 0.0
    
    def is_full(self):
        """
        检查所有处理器队列是否都已满
        
        Returns:
            bool: 如果所有队列都满返回True
        """
        return all(queue.is_full() for queue in self.processor_queues)
    
    def get_queue_length(self):
        """
        获取总队列长度（所有处理器的队列长度之和）
        
        Returns:
            int: 总队列长度
        """
        return sum(queue.get_queue_length() for queue in self.processor_queues)
    
    def clear(self):
        """清空所有队列"""
        for queue in self.processor_queues:
            queue.clear()
        self._task_to_processor.clear()
    
    def __len__(self):
        """返回总队列长度"""
        return self.get_queue_length()
    
    def __repr__(self):
        queue_lengths = [q.get_queue_length() for q in self.processor_queues]
        return f"RSUQueueManager(processors={self.num_processors}, queue_lengths={queue_lengths})"

