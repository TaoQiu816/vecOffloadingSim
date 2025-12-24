import numpy as np


class SystemConfig:
    """
    [系统参数配置类] SystemConfig

    作用:
    集成通信模型、计算资源、物理约束、DAG生成参数及强化学习归一化常量。
    基于参考文献 (DVTP, Dependency-Aware) 及 VEC 通用标准进行优化。

    单位标准 (SI):
    - Time: Seconds (s)
    - Frequency: Hertz (Hz)
    - Data: Bits (bit)
    - Power: Watts (W) / dBm
    - Distance: Meters (m)
    - Computation: Cycles
    """

    # =========================================================================
    # 1. 仿真场景与物理设置 (Simulation & Physics)
    # =========================================================================
    SEED = 42  # 随机种子
    NUM_VEHICLES = 20  # 车辆总数 (DVTP: 10-50)
    MAP_SIZE = 2000.0  # 区域大小 (2000m x 2000m)
    RSU_POS = np.array([1000.0, 1000.0])  # RSU 坐标 (地图中心)

    DT = 0.1  # 仿真时间步长 (s) (标准车联网仿真步长)
    MAX_STEPS = 100  # 每个 Episode 步数

    # 车辆移动性参数
    VEL_MIN = 10.0  # 最小速度 (m/s) ~ 36 km/h
    VEL_MAX = 20.0  # 最大速度 (m/s) ~ 72 km/h
    # 兼容旧代码引用
    MAX_VELOCITY = VEL_MAX

    # =========================================================================
    # 2. 通信参数 (Communication Model)
    # =========================================================================
    # 载波频率 (C-V2X / DSRC)
    FC = 5.9e9  # 5.9 GHz
    C_LIGHT = 3e8  # 光速

    # 带宽 (Hz)
    BW_V2I = 25e6  # 25 MHz
    BW_V2V = 10e6  # V2V 保持 10 MHz 或提升至 20 MHz 均可

    # 噪声参数
    # 噪声功率谱密度 (dBm/Hz) -> -174 + Noise Figure (e.g., 9dB)
    NOISE_POWER_DENSITY_DBM = -174
    # 接收机噪声系数 (Noise Figure)
    NOISE_FIGURE = 9
    # 计算总噪声功率 (dBm) = -174 + 10log10(BW) + NF
    # 约为 -174 + 70 + 9 = -95 dBm
    NOISE_POWER_DBM = -174 + 10 * np.log10(BW_V2I) + NOISE_FIGURE

    # 发射功率 (dBm) -> IEEE 802.11p Class C (23dBm) / LTE-V2X (23dBm)
    # 范围放宽至 20-27 dBm (100mW - 500mW)
    TX_POWER_MIN_DBM = 20
    TX_POWER_MAX_DBM = 30

    # 路径损耗模型参数 (Urban Scenario)
    # PL = Alpha + Beta * log10(d) + Shadowing
    # 3GPP TR 36.885 V2X Urban Line-of-Sight (LOS)
    PL_ALPHA_V2I = 128.1
    PL_BETA_V2I = 37.6

    # V2V 模型通常更为复杂，此处简化为 Log-distance
    ALPHA_V2V = 3.0  # 路损指数 (遮挡严重)

    # 兼容旧代码的简单模型参数 (若 ChannelModel 未更新)
    ALPHA_V2I = 2.5
    RICIAN_K_DB = 6.0
    BETA_0_DB = -30

    # =========================================================================
    # 3. 计算资源参数 (Computation Model)
    # 异构性体现: RSU 算力 >> 车辆算力
    # =========================================================================
    # CPU 时钟频率 (Cycles/s)
    # F_VEHICLE = 1.0 * 1e9  # 车辆: 1 GHz
    MAX_VEHICLE_CPU_FREQ = 3.0e9  # 最大 CPU 频率，2 GHz
    MIN_VEHICLE_CPU_FREQ = 1.0e9  # 最小 CPU 频率，1 GHz
    F_RSU = 25.0 * 1e9  # RSU: 20 GHz (20倍于车辆)
    # 2. 依然保留 F_VEHICLE，但将其注释为“基准参考值”
    # 这样既能满足代码的调用需求（不报错），又不影响你每辆车的随机性
    F_VEHICLE = 2.0e9             # 设为平均值 2 GHz

    # 能耗系数 (Effective Switched Capacitance)
    K_ENERGY = 1e-28  # Energy = k * f^2 * cycles

    # =========================================================================
    # 4. 物理约束与掩码 (Constraints & Masking)
    # =========================================================================
    # 通信范围
    RSU_RANGE = 1000.0  # RSU 覆盖半径 (m)
    V2V_RANGE = 300.0  # V2V 通信半径 (m) (DVTP等文献常用值)

    # 队列约束 (拥堵控制)
    VEHICLE_QUEUE_LIMIT = 10  # 车辆任务队列上限 (避免排队过长)
    RSU_QUEUE_LIMIT = 50  # RSU 全局队列上限

    # =========================================================================
    # 5. DAG 任务生成参数 (Task Generation)
    # 目标: 生成时延敏感型任务，迫使 Agent 进行卸载
    # =========================================================================
    # 任务节点数范围
    MIN_NODES = 12
    MAX_NODES = 16

    # 单个子任务数据量 (Bits) -> 1 Mbit ~ 3 Mbit
    # 明确乘以 8，转换为 bit
    MIN_DATA = 50 * 1024 * 8  # 50 KB -> bits
    MAX_DATA = 500 * 1024 * 8  # 500 KB -> bits

    # 对应文献 100 KB ~ 500 KB（边传输量）
    MIN_EDGE_DATA = 100 * 1024 * 8
    MAX_EDGE_DATA = 500 * 1024 * 8

    # 单个子任务计算量 (Cycles)
    MIN_COMP = 1.0 * 1e7
    MAX_COMP = 1.0 * 1e8

    # 统计参考值
    MEAN_DATA_SIZE = (MIN_DATA + MAX_DATA) / 2
    MEAN_COMP_LOAD: float = (MIN_COMP + MAX_COMP) / 2

    # DAG 结构参数
    DAG_FAT = 0.6  # 宽度
    DAG_DENSITY = 0.4  # 密度
    DAG_REGULAR = 0.5
    DAG_CCR = 0.5  # 通信计算比

    # Deadline 因子
    # Deadline = (Total_Comp / F_VEHICLE) * Factor
    # 0.8 ~ 1.5: 涵盖了"本地无法完成"到"本地勉强完成"的范围
    # DEADLINE_FACTOR_MIN = 0.8
    # DEADLINE_FACTOR_MAX = 1.5
    # # 兼容旧代码单一值
    # DEADLINE_FACTOR = 1.2  # 默认平均值
    # [修改] Deadline 更加宽容一点，减少因物理环境不可能完成导致的“无助感”
    DEADLINE_FACTOR_MIN = 0.8  # 0.8
    DEADLINE_FACTOR_MAX = 1.5  # 1.5
    DEADLINE_FACTOR = 1.4

    # =========================================================================
    # 6. 强化学习归一化常量 (RL Normalization)
    # =========================================================================
    # 资源特征归一化
    NORM_MAX_CPU = 25.0 * 1e9  # 基准: RSU 频率
    NORM_MAX_COMP = 1.2 * 1e8  # 适应 1e8 的计算量
    NORM_MAX_DATA = 5.0 * 1e6  # 适应约 4e6 bit 的数据量

    # 速率特征归一化
    NORM_MAX_RATE_V2I = 50e6  # 50 Mbps
    NORM_MAX_RATE_V2V = 20e6  # 20 Mbps

    # 负载/等待时间归一化
    NORM_MAX_WAIT_TIME = 1.0

    # 1. 任务失败惩罚
    # 保持 -50.0 没问题，这是一个巨大的“死亡惩罚”，平均到20辆车上是 -2.5，
    # 相比于单步奖励 (0.1~0.5) 依然非常有震慑力。
    PENALTY_FAILURE = -50.0

    # 2. 拥堵惩罚权重 (W_QUEUE)
    # [关键修改] 从 0.5 降到 0.05
    # 逻辑: 40 个排队任务 * 0.05 = -2.0。
    # 这样惩罚量级(2.0) 和 收益量级(后面会放大到 1.0) 就在同一个数量级了。
    W_QUEUE = 0.05

    # 3. 奖励缩放 (REWARD_SCALE)
    # [关键修改] 放大时间收益。
    # 原始 cft_diff 只有 0.1s 左右。我们需要把它放大 10 倍甚至 20 倍，
    # 让 Agent 觉得"节省时间"是有利可图的。
    # 建议设置为 10.0 或 20.0
    REWARD_SCALE = 10.0


    # =========================================================================
    # 8. 辅助方法
    # =========================================================================
    @staticmethod
    def dbm2watt(dbm):
        """
        [工具函数] dBm 转 Watts
        Formula: P(W) = 10 ^ ((P(dBm) - 30) / 10)
        """
        return 10 ** ((dbm - 30) / 10.0)

    @staticmethod
    def watt2dbm(watt):
        """
        [工具函数] Watts 转 dBm
        """
        if watt <= 0: return -float('inf')
        return 10.0 * np.log10(watt) + 30.0