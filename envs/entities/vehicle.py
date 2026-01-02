import numpy as np
from configs.config import SystemConfig as Cfg
from envs.modules.queue_system import FIFOQueue
from envs.modules.active_task_manager import ActiveTaskManager, ActiveTask


class Vehicle:
    def __init__(self, v_id, pos):
        """
        [车辆实体类]
        维护车辆的物理状态 (位置、速度) 和资源状态 (队列)。

        Args:
            v_id (int): 车辆唯一 ID
            pos (np.array): 初始坐标 [x, y]
        """
        self.id = v_id
        self.pos = np.array(pos, dtype=float)

        self.curr_assigned_target = None  # 任务分配时确定的目标

        # --- 移动性初始化 (Mobility Model) ---
        # 道路模型：截断正态分布速度，恒定方向（沿X轴正方向）
        if hasattr(Cfg, 'VEL_MEAN') and hasattr(Cfg, 'VEL_STD'):
            # 截断正态分布
            speed = np.random.normal(Cfg.VEL_MEAN, Cfg.VEL_STD)
            speed = np.clip(speed, Cfg.VEL_MIN, Cfg.VEL_MAX)
        else:
            # 向后兼容：均匀分布
            speed = np.random.uniform(Cfg.VEL_MIN, Cfg.VEL_MAX)
        
        # 道路模型：速度方向固定为X轴正方向（沿道路方向）
        # vel = [v_x, v_y]，其中v_x = speed, v_y = 0
        self.vel = np.array([speed, 0.0])

        # --- CPU 频率初始化 ---
        # 异构算力：从配置范围中随机采样
        self.cpu_freq = np.random.uniform(Cfg.MIN_VEHICLE_CPU_FREQ, Cfg.MAX_VEHICLE_CPU_FREQ)
        
        # --- 资源约束 (Unified Processor Sharing) ---
        # [新增] 活跃任务管理器：统一管理本地和V2V接收的任务
        self.active_task_manager = ActiveTaskManager(num_processors=1, cpu_freq=self.cpu_freq)
        
        # [保留] 旧的FIFO队列系统（用于向后兼容和容量检查）
        self.task_queue = FIFOQueue(
            max_buffer_size=Cfg.VEHICLE_QUEUE_LIMIT,  # 向后兼容
            max_load_cycles=Cfg.VEHICLE_QUEUE_CYCLES_LIMIT  # 基于计算量的限制
        )
        # 保持向后兼容的属性
        self.task_queue_len = 0
        self.max_queue_size = Cfg.VEHICLE_QUEUE_LIMIT

        # --- 任务状态 ---
        self.task_dag = None  # 车辆的 DAG 任务
        self.curr_subtask = None  # 当前正在处理的子任务 ID

        # [关键修正] 初始化为 'Local'，防止 Env.reset() 计算 CFT 时报错
        self.curr_target = 'Local'
        self.last_scheduled_subtask = -1
        self.last_action_step = -1
        self.last_action_target = 'Local'

        # --- 通信状态 ---
        self.tx_power_dbm = Cfg.TX_POWER_MIN_DBM  # 发射功率 (受 RL 动作控制)
        self.last_target = None  # 记录上一次动作 (用于日志或调试)

        # [新增] FAT (Earliest Available Time) 管理
        # 处理器FAT：本地CPU的最早可用时间
        self.fat_processor = 0.0
        # 上传信道FAT：车辆上传信道的最早可用时间（V2I和V2V共享）
        self.fat_uplink = 0.0

        # [新增] 活跃数据传输跟踪
        # 格式: [{'child_id': int, 'parent_id': int, 'rem_data': float, 'speed': float}, ...]
        self.active_transfers = []

    def is_queue_full(self, new_task_cycles=0):
        """
        判断队列是否已满（基于计算量）
        用于Env中的Action Masking
        
        Args:
            new_task_cycles: 要添加的新任务计算量（用于检查加入后是否溢出）
        """
        return self.task_queue.is_full(new_task_cycles=new_task_cycles)
    
    def update_queue_sync(self):
        """同步队列长度属性（用于向后兼容）"""
        self.task_queue_len = self.task_queue.get_queue_length()
    
    def reset_fat(self):
        """重置FAT为初始值（在reset()时调用）"""
        self.fat_processor = 0.0
        self.fat_uplink = 0.0

    def step(self, dt: float) -> list:
        """
        [统一处理器共享物理模型]
        推进车辆CPU上所有活跃任务的执行（本地+V2V接收）
        
        处理器共享逻辑：
        - 所有活跃任务平等共享CPU资源
        - effective_speed = cpu_freq / max(1, num_active_tasks)
        - 每个任务推进: effective_speed * dt 的计算量
        
        Args:
            dt: 时间步长 (秒)
            
        Returns:
            list: 本step完成的ActiveTask列表
        """
        return self.active_task_manager.step(dt)
    
    def add_active_task(self, owner_id: int, subtask_id: int, task_type: str, 
                       total_comp: float, current_time: float = 0.0,
                       # [关键] 准入验证参数（必须由调用方提供）
                       is_dag_ready: bool = True,
                       is_data_ready: bool = True,
                       task_status: str = 'READY') -> bool:
        """
        添加活跃任务到处理器（含完整准入验证）
        
        [硬断言] 准入条件（由ActiveTask强制执行）：
        1. task_status in ['READY', 'RUNNING']
        2. is_dag_ready=True（DAG依赖已满足）
        3. is_data_ready=True（数据传输已完成，针对v2v/v2i）
        4. total_comp > 0
        
        Args:
            owner_id: 任务所属车辆ID
            subtask_id: 子任务ID
            task_type: 'local' 或 'v2v' 或 'v2i'
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
    
    def get_estimated_delay(self) -> float:
        """
        获取基于处理器共享模型的估计延迟（新版）
        
        使用ActiveTaskManager计算当前负载下的平均剩余时间。
        
        Returns:
            float: 估计延迟（秒）
        """
        return self.active_task_manager.get_estimated_delay()
    
    def update_pos(self, dt, map_size):
        """
        [物理更新] 更新位置并处理边界约束（道路模型：一维移动）

        Args:
            dt (float): 时间步长
            map_size (float): 道路长度 L (m)
        
        道路模型特点：
        - 车辆沿X轴正方向移动（一维移动）
        - Y坐标保持不变（在车道上）
        - 边界约束：X坐标在[0, L]内
        - 超出边界：标记为需要移除（由环境层处理）
        """
        # 道路模型：仅X方向移动
        # pos = [x, y]，只更新x坐标
        self.pos[0] += self.vel[0] * dt
        
        # 注意：不在这里clip，让车辆可以超出边界
        # 环境层在step()中会检查并移除超出边界的车辆
        # 这样可以让车辆自然超出边界后被移除
