import numpy as np
from configs.config import SystemConfig as Cfg


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

        # --- 移动性初始化 (Mobility Model) ---
        # [修正] 使用 Config 中的参数，而不是硬编码 10.0/20.0
        speed = np.random.uniform(Cfg.VEL_MIN, Cfg.VEL_MAX)
        angle = np.random.uniform(0, 2 * np.pi)  # 随机方向
        self.vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])

        # --- 资源约束 (Queue Constraint) ---
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

    @property
    def is_queue_full(self):
        """
        [辅助属性] 判断队列是否已满
        用于 Env 中的 Action Masking。
        """
        return self.task_queue_len >= self.max_queue_size

    def update_pos(self, dt, map_size):
        """
        [物理更新] 更新位置并处理边界反弹

        Args:
            dt (float): 时间步长
            map_size (float): 地图边界
        """
        self.pos += self.vel * dt

        # 简单的边界反弹逻辑 (Bouncing Box)
        # 保证车辆始终在地图范围内
        for i in range(2):
            if self.pos[i] < 0:
                self.pos[i] = 0
                self.vel[i] *= -1  # 撞墙反弹
            if self.pos[i] > map_size:
                self.pos[i] = map_size
                self.vel[i] *= -1