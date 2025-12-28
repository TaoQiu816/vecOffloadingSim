import numpy as np
from configs.config import SystemConfig as Cfg
from envs.modules.queue_system import FIFOQueue


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

        # --- 资源约束 (Queue Constraint) ---
        # 使用FIFO队列系统管理任务（基于计算量的限制）
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

        # --- 通信状态 ---
        self.tx_power_dbm = Cfg.TX_POWER_MIN_DBM  # 发射功率 (受 RL 动作控制)
        self.last_target = None  # 记录上一次动作 (用于日志或调试)

        # --- CPU 频率初始化 ---
        # 异构算力：从配置范围中随机采样
        self.cpu_freq = np.random.uniform(Cfg.MIN_VEHICLE_CPU_FREQ, Cfg.MAX_VEHICLE_CPU_FREQ)

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