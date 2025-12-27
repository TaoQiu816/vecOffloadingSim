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
    
    # =========================================================================
    # 道路模型参数 (Road Model)
    # =========================================================================
    MAP_SIZE = 2000.0  # 道路长度 L (m) - 城市快速路
    NUM_LANES = 3  # 车道数量 N_lane
    LANE_WIDTH = 3.5  # 车道宽度 W_lane (m) - 标准车道宽度
    
    # 车辆参数
    NUM_VEHICLES = 20  # 车辆总数 (初始车辆数，后续动态生成)
    
    # =========================================================================
    # RSU部署参数 (RSU Deployment)
    # =========================================================================
    NUM_RSU = 5  # RSU数量 M
    RSU_Y_DIST = 20.0  # RSU离路边的距离 (m)
    # RSU覆盖半径在下面定义（RSU_RANGE）
    # RSU部署间距 D_inter 需要满足：D_inter <= 2 * sqrt(R_rsu^2 - Y_RSU^2) * 0.9
    # Y_RSU = ROAD_WIDTH + RSU_Y_DIST，其中ROAD_WIDTH = NUM_LANES * LANE_WIDTH
    # 实际部署间距：D_inter = MAP_SIZE / NUM_RSU
    # 需要断言：NUM_RSU * D_inter >= MAP_SIZE 以确保全覆盖
    
    # RSU配置（向后兼容：保留单个RSU的默认位置）
    RSU_POS = np.array([500.0, 500.0])  # 单个RSU的默认坐标（已废弃，改用RSU列表）
    
    # =========================================================================
    # 车辆到达参数（泊松过程）
    # =========================================================================
    VEHICLE_ARRIVAL_RATE = 0.1  # 车辆到达率 λ (vehicles/s)，即平均每10秒到达1辆车
    # 如果设为0，则禁用动态生成，只在reset时生成初始车辆

    DT = 0.1  # 仿真时间步长 (s) - 建议0.1s或1.0s，需保证 v_max * DT << R_rsu
    MAX_STEPS = 500  # 每个 Episode 步数

    # =========================================================================
    # 车辆移动性参数（截断正态分布）
    # =========================================================================
    VEL_MEAN = 16.6  # 速度均值 μ_v (m/s) ≈ 60 km/h
    VEL_STD = 3.0  # 速度标准差 σ_v (m/s)
    VEL_MIN = 10.0  # 最小速度 (m/s) ~ 36 km/h
    VEL_MAX = 25.0  # 最大速度 (m/s) ~ 90 km/h
    # 兼容旧代码引用
    MAX_VELOCITY = VEL_MAX

    # =========================================================================
    # 2. 通信参数 (Communication Model)
    # =========================================================================
    # 载波频率 (C-V2X / DSRC)
    FC = 5.9e9  # 5.9 GHz
    C_LIGHT = 3e8  # 光速

    # 带宽 (Hz)
    BW_V2I = 40e6  # 40 MHz
    BW_V2V = 10e6  # 10 MHz

    # 噪声参数
    NOISE_POWER_DENSITY_DBM = -174
    NOISE_FIGURE = 9
    NOISE_POWER_DBM = -174 + 10 * np.log10(BW_V2I) + NOISE_FIGURE

    # 发射功率 (dBm)
    TX_POWER_MIN_DBM = 23
    TX_POWER_MAX_DBM = 30

    # 路径损耗模型 (简化版: 自由空间 + 环境因子)
    # PL = PL_0 + 10 * n * log10(d/d_0)
    # 使用更实际的参数，确保 RSU 覆盖范围内 SNR >= 0
    PL_ALPHA_V2I = 60.0  # PL_0: 1m 处的基准路径损耗 (dB)
    PL_BETA_V2I = 2.2    # n: 路径损耗指数 (接近自由空间)

    # V2V 路径损耗
    PL_ALPHA_V2V = 60.0
    PL_BETA_V2V = 3.0

    # V2V 干扰因子
    V2V_INTERFERENCE_FACTOR = 10.0

    # 速率估计 SNR 参数
    # SNR 到速率映射: rate = BW * log2(1 + SNR_linear)
    # 为简化，使用分段线性映射
    SNR_MIN_DB = -10.0   # 最低可检测 SNR
    SNR_MAX_DB = 20.0    # 饱和 SNR
    SNR_OFFSET_DB = 10.0 # 偏移量用于保证正值 SNR

    # 兼容旧代码
    ALPHA_V2I = 2.5
    RICIAN_K_DB = 6.0
    BETA_0_DB = -30

    # V2V 路径损耗指数 (兼容旧代码引用)
    ALPHA_V2V = 3.0

    # =========================================================================
    # 3. 计算资源参数 (Computation Model)
    # 异构性体现: RSU 算力 > 车辆算力 (但不是压倒性优势)
    # =========================================================================
    # CPU 时钟频率 (Cycles/s)
    # 调整策略: RSU频率设为车辆的2倍，确保在高排队时形成梯度反转
    # 平均车辆频率 = (3+1)/2 = 2GHz, RSU = 4GHz (2倍)
    # 队列35时: 排队时间 = 35*0.3e9/4e9 = 2.6s, 总RSU时间 ≈ 2.7s
    # 队列50时: 排队时间 = 50*0.3e9/4e9 = 3.75s, 总RSU时间 ≈ 3.85s
    # 本地时间 ≈ 4.5s, 在队列35-50时形成梯度反转
    MAX_VEHICLE_CPU_FREQ = 3.0e9  # 最大 CPU 频率，3 GHz
    MIN_VEHICLE_CPU_FREQ = 1.0e9  # 最小 CPU 频率，1 GHz
    F_RSU = 4.0 * 1e9  # RSU: 4 GHz (约2倍于车辆平均频率)
    # 2. 依然保留 F_VEHICLE，但将其注释为“基准参考值”
    # 这样既能满足代码的调用需求（不报错），又不影响你每辆车的随机性
    #F_VEHICLE = 2.0e9  # 设为平均值 2 GHz

    # 能耗系数 (Effective Switched Capacitance)
    K_ENERGY = 1e-28  # Energy = k * f^2 * cycles

    # =========================================================================
    # 4. 物理约束与掩码 (Constraints & Masking)
    # =========================================================================
    # 通信范围
    RSU_RANGE = 500.0  # RSU 覆盖半径 R_rsu (m) - 已调整为适合道路模型的参数
    V2V_RANGE = 300.0  # V2V 通信半径 (m) (DVTP等文献常用值)

    # 队列约束 (拥堵控制)
    # 调整策略: 增加RSU队列限制，创造足够的排队时间来形成梯度反转
    # 当RSU队列>50时，排队时间(>0.17s)可能超过本地执行时间(0.05-0.17s)
    VEHICLE_QUEUE_LIMIT = 20  # 车辆任务队列上限
    RSU_QUEUE_LIMIT = 100  # RSU全局队列上限 (从40增加到100)
    
    # RSU多处理器配置
    RSU_NUM_PROCESSORS = 4  # RSU处理器数量（多核处理器架构）

    # =========================================================================
    # 5. DAG 任务生成参数 (Task Generation)
    # 目标: 生成时延敏感型任务，迫使 Agent 进行卸载
    # =========================================================================
    # 任务节点数范围
    MIN_NODES = 8
    MAX_NODES = 12

    # 单个子任务数据量 (Bits) -> 1 Mbit ~ 3 Mbit
    # 明确乘以 8，转换为 bit
    MIN_DATA = 200 * 1024 * 8  # 50 KB -> bits
    MAX_DATA = 500 * 1024 * 8  # 500 KB -> bits

    # 对应文献 100 KB ~ 500 KB（边传输量）
    MIN_EDGE_DATA = 100 * 1024 * 8
    MAX_EDGE_DATA = 500 * 1024 * 8

    # [调整] 计算量: 模拟 AI 推理/复杂计算
    # 目标：创造梯度反转点
    # 本地执行时间: 2.0s - 6.0s (10个子任务 x 0.2-0.6亿次 @ 2GHz)
    # RSU执行时间: 0.5s - 1.5s (4GHz vs 2GHz avg = 2x)
    # RSU排队时间上限: 15s (队列100 x 0.6亿次 / 4GHz)
    # 当RSU队列>50时，排队时间(>7.5s)可能超过本地执行时间(2-6s)，创造梯度反转
    MIN_COMP = 0.2 * 1e9  # 2亿次
    MAX_COMP = 0.6 * 1e9  # 6亿次

    # 统一使用标准语法，移除内联类型注释
    MEAN_COMP_LOAD = (MIN_COMP + MAX_COMP) / 2  # 平均计算负载 (cycles)
    # 或在类文档中统一说明类型

    # DAG 结构参数
    DAG_FAT = 0.6  # 宽度
    DAG_DENSITY = 0.4  # 密度
    DAG_REGULAR = 0.5
    DAG_CCR = 0.5  # 通信计算比

    # =========================================================================
    # Deadline 计算参数 (Ideal Local Anchoring)
    # =========================================================================
    """
    设计原则: "Deadline是铁律，只看硬件能力，不听任何借口（不看排队）"

    Deadline计算公式: T_deadline = γ × (W_k / f_local)
    - W_k: 任务总计算量 (Cycles)
    - f_local: 车辆本地CPU频率 (Hz)
    - γ (紧缩因子): 核心参数，确保本地计算无法满足Deadline

    为什么排除本地排队？
    1. 业务属性: 防碰撞等时延敏感任务只关心响应速度，不关心CPU忙闲
    2. 防止作弊: 如果Deadline包含排队，智能体可通过堆积本地任务延长Deadline
    3. 卸载必要性: γ < 1.0 确保本地执行时间必然大于Deadline，强迫卸载
    """
    DEADLINE_TIGHTENING_FACTOR = 0.75  # γ: 紧缩因子 (0.7-0.8)
    DEADLINE_TIGHTENING_MIN = 0.70  # γ最小值
    DEADLINE_TIGHTENING_MAX = 0.80  # γ最大值

    # 验证: γ < 1.0 确保本地计算100%失败
    # 当γ=0.75，本地耗时1.0s的任务，Deadline为0.75s
    # 无论本地CPU频率多高、队列多空，都无法满足Deadline

    # =========================================================================
    # 6. 强化学习归一化常量 (RL Normalization)
    # =========================================================================
    # 资源特征归一化
    NORM_MAX_CPU = 25.0 * 1e9  # 基准: RSU 频率
    NORM_MAX_COMP = 2.0 * 1e9  # 适应 1e8 的计算量
    NORM_MAX_DATA = 5.0 * 1e6  # 适应约 4e6 bit 的数据量

    # 速率特征归一化
    NORM_MAX_RATE_V2I = 50e6  # 50 Mbps
    NORM_MAX_RATE_V2V = 20e6  # 20 Mbps

    # =========================================================================
    # 7. 奖励函数参数 (MAPPO Reward Function)
    # =========================================================================
    # A. 效率收益参数 (Efficiency Gain)
    EFF_WEIGHT = 1.0  # α: 效率收益权重系数
    EFF_SCALE = 3.0   # λ: tanh缩放因子 (推荐 2.0-5.0)

    # B. 拥塞惩罚参数 (Congestion Penalty)
    CONG_WEIGHT = 0.5  # β: 拥塞惩罚权重系数
    CONG_GAMMA = 2.0   # γ: 拥塞敏感度指数 (推荐 ≥2.0)

    # C. 软约束惩罚参数 (Soft Constraint - Timeout)
    PENALTY_TIMEOUT_WEIGHT = 10.0  # η: 超时惩罚系数 (动态惩罚)

    # D. 硬约束惩罚参数 (Hard Constraint)
    PENALTY_LINK_BREAK = -50.0   # 链路断开惩罚
    PENALTY_OVERFLOW = -50.0     # 队列溢出惩罚
    PENALTY_FAILURE = -50.0      # 任务失败惩罚 (保留向后兼容)

    # E. 奖励范围控制
    REWARD_MAX = 10.0    # 奖励上限 (防止奖励爆炸)
    REWARD_MIN = -10.0   # 奖励下限

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

    # 删除硬编码的 NORM_MAX_WAIT_TIME
    # 用动态计算值完全替代
    _RSU_MAX_WAIT = RSU_QUEUE_LIMIT * MEAN_COMP_LOAD / F_RSU
    _VEHICLE_MAX_WAIT = VEHICLE_QUEUE_LIMIT * MEAN_COMP_LOAD / MIN_VEHICLE_CPU_FREQ
    DYNAMIC_MAX_WAIT_TIME = max(_RSU_MAX_WAIT, _VEHICLE_MAX_WAIT) * 1.2

    # 更新归一化常量引用
    NORM_MAX_WAIT_TIME = DYNAMIC_MAX_WAIT_TIME  # 保持向后兼容性
