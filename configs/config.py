import numpy as np
import os


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
    MAP_SIZE = 1000.0  # 道路长度 L (m) - 1公里路段，避免稀疏/拥堵
    NUM_LANES = 2  # 车道数量 N_lane - 双车道，减少换道干扰
    LANE_WIDTH = 3.5  # 车道宽度 W_lane (m) - 标准车道宽度
    
    # 车辆参数（降低密度以减少V2V目标数量，避免动作空间失衡）
    NUM_VEHICLES = 12  # 车辆总数（从20降到12，减少V2V目标约40%）
    MAX_VEHICLE_ID = 1000  # 最大车辆ID（用于Embedding表大小，需覆盖动态生成的所有车辆）
    V2V_TOP_K = 11  # 每个智能体最多考虑的V2V候选数量
    MAX_NEIGHBORS = max(0, min(NUM_VEHICLES - 1, V2V_TOP_K))
    MAX_TARGETS = 2 + MAX_NEIGHBORS
    RESOURCE_RAW_DIM = 11  # 资源原始特征维度（用于ResourceFeatureEncoder）
    
    # =========================================================================
    # RSU部署参数 (RSU Deployment)
    # =========================================================================
    NUM_RSU = 3  # RSU数量 M - 1000m全覆盖+重叠区
    RSU_Y_DIST = 10.0  # RSU离路边的距离 (m)
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
    VEHICLE_ARRIVAL_RATE = 0.2  # 车辆到达率 λ (vehicles/s)，即平均每5秒到达1辆车
    # 如果设为0，则禁用动态生成，只在reset时生成初始车辆

    DT = 0.05  # 仿真时间步长 (s) - 建议0.05s或0.1s，需保证 v_max * DT << R_rsu
    MAX_STEPS = 200  # 每个 Episode 步数

    # =========================================================================
    # 车辆移动性参数（截断正态分布）
    # =========================================================================
    VEL_MEAN = 11.1  # 速度均值 μ_v (m/s) ≈ 40 km/h
    VEL_STD = 2.0  # 速度标准差 σ_v (m/s)
    VEL_MIN = 5.0  # 最小速度 (m/s) ~ 18 km/h
    VEL_MAX = 16.6  # 最大速度 (m/s) ~ 60 km/h
    # 兼容旧代码引用
    MAX_VELOCITY = VEL_MAX

    # =========================================================================
    # 2. 通信参数 (Communication Model)
    # =========================================================================
    # 载波频率 (C-V2X / DSRC)
    FC = 5.9e9  # 5.9 GHz
    C_LIGHT = 3e8  # 光速

    # 带宽 (Hz)
    BW_V2I = 20e6  # 20 MHz
    BW_V2V = 10e6  # 10 MHz

    # 噪声参数
    NOISE_POWER_DENSITY_DBM = -174
    NOISE_FIGURE = 9
    NOISE_POWER_DBM = -95.0  # 热噪声底噪

    # 发射功率 (dBm)
    TX_POWER_UP_DBM = 23.0  # 车辆上行发射功率 (200mW)
    TX_POWER_V2V_DBM = 23.0  # V2V侧链发射功率 (200mW)
    TX_POWER_MIN_DBM = 20.0  # 功率控制下限
    TX_POWER_MAX_DBM = 23.0  # 功率控制上限

    # 路径损耗模型 (Log-Distance)
    # PL = PL_0 + 10 * n * log10(d/d_0)
    PL_ALPHA_V2I = 28.0  # V2I 参考路损 (LOS)
    PL_BETA_V2I = 2.5    # V2I 衰减指数

    # V2V 路径损耗
    PL_ALPHA_V2V = 28.0  # V2V 参考路损
    PL_BETA_V2V = 3.5    # V2V 衰减指数 (高衰减，模拟遮挡/干扰)

    # V2V 背景干扰强度 (环境干扰底噪)
    V2V_INTERFERENCE_DBM = -90.0  # V2V背景干扰强度 (dBm)

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
    F_RSU = 10.0 * 1e9  # RSU: 4 GHz (约2倍于车辆平均频率)
    # 2. 依然保留 F_VEHICLE，但将其注释为“基准参考值”
    # 这样既能满足代码的调用需求（不报错），又不影响你每辆车的随机性
    #F_VEHICLE = 2.0e9  # 设为平均值 2 GHz

    # 能耗系数 (Effective Switched Capacitance)
    K_ENERGY = 1e-28  # Energy = k * f^2 * cycles

    # =========================================================================
    # 4. 物理约束与掩码 (Constraints & Masking)
    # =========================================================================
    # 通信范围
    RSU_RANGE = 400.0  # RSU 覆盖半径 R_rsu (m) - 已调整为适合道路模型的参数
    V2V_RANGE = 300.0  # V2V 通信半径 (m) (DVTP等文献常用值)

    # 队列约束 (拥堵控制)
    # [向后兼容] 保留旧的参数名（但不再用于队列限制检查）
    # 这些参数仅用于计算归一化特征时的参考
    VEHICLE_QUEUE_LIMIT = 20  # 车辆任务队列上限（已废弃，改用VEHICLE_QUEUE_CYCLES_LIMIT）
    RSU_QUEUE_LIMIT = 100  # RSU全局队列上限（已废弃，改用RSU_QUEUE_CYCLES_LIMIT）
    
    # RSU多处理器配置
    RSU_NUM_PROCESSORS = 4  # RSU处理器数量（多核处理器架构）

    # =========================================================================
    # 5. DAG 任务生成参数 (Task Generation)
    # 目标: 生成时延敏感型任务，迫使 Agent 进行卸载
    # =========================================================================
    # 任务节点数范围（降维打击：减少依赖链深度）
    MIN_NODES = 4  # 从8降到4
    MAX_NODES = 6  # 从12降到6，极大降低依赖链深度
    
    # =========================================================================
    # DAG任务优先级算法权重
    # 优先级公式：Score(i) = PRIORITY_W1 * L_bwd[i] + PRIORITY_W2 * (total_comp[i] / NORM_MAX_COMP) + PRIORITY_W3 * (out_degree[i] / MAX_NODES)
    PRIORITY_W1 = 100.0  # 后向层级权重（主导，关键路径）
    PRIORITY_W2 = 1.0    # 计算量权重（同层级下的tie-breaking）
    PRIORITY_W3 = 1.0    # 出度权重（同计算量下的tie-breaking）

    # 单个子任务数据量 (Bits) -> 1 Mbit ~ 3 Mbit
    MIN_DATA = 1.0e6  # 1 Mbits 输入
    MAX_DATA = 3.0e6  # 3 Mbits 输入

    # 边传输数据量 (Bits) - 关键：设为计算量的 1/5 ~ 1/3
    MIN_EDGE_DATA = 0.2e6  # 200 Kbits (传输约0.02s)
    MAX_EDGE_DATA = 0.6e6  # 600 Kbits (传输约0.06s)

    # 计算量 (Cycles) - 负重减轻：让任务更轻量，减少排队积压
    # 提高单任务计算量，迫使多步或卸载：平均约1.5e9 cycles（本地约0.5s @3GHz）
    MIN_COMP = 1.0e9
    MAX_COMP = 2.0e9

    # 统一使用标准语法，移除内联类型注释
    MEAN_COMP_LOAD = (MIN_COMP + MAX_COMP) / 2  # 平均计算负载 (cycles)
    
    # [新设计] 基于计算量（Cycles）的队列限制，比任务个数更科学准确
    # 平均任务计算量
    AVG_COMP = (MIN_COMP + MAX_COMP) / 2  # 0.4e9 cycles (4亿次)
    
    # 基于计算量的队列限制（从原来的任务个数转换）
    # 车辆：原来20个任务 → 20 * AVG_COMP = 4.0e9 cycles
    # RSU：原来100个任务 → 100 * AVG_COMP = 20.0e9 cycles
    VEHICLE_QUEUE_CYCLES_LIMIT = 4.0e9  # 车辆队列最大计算量 (≈ 20 个平均任务)
    RSU_QUEUE_CYCLES_LIMIT = 20.0e9  # RSU队列最大计算量 (≈ 100 个平均任务)
    # 或在类文档中统一说明类型

    # DAG 结构参数（简化结构：降低连接密度）
    DAG_FAT = 0.6  # 宽度
    DAG_DENSITY = 0.2  # 密度（从0.4降到0.2，减少依赖复杂度）
    DAG_REGULAR = 0.5
    DAG_CCR = 0.2  # 通信计算比 (结构生成参数)

    # =========================================================================
    # Deadline 计算参数 (Ideal Local Anchoring)
    # =========================================================================
    """
    设计原则: "Deadline是铁律，只看硬件能力，不听任何借口（不看排队）"

    Deadline计算公式: T_deadline = γ × (W_k / f_local)
    - W_k: 任务总计算量 (Cycles)
    - f_local: 车辆本地CPU频率 (Hz)
    - γ (松紧因子): 控制Deadline严格程度，γ<1趋向强制卸载，γ>1放宽Deadline

    为什么排除本地排队？
    1. 业务属性: 防碰撞等时延敏感任务只关心响应速度，不关心CPU忙闲
    2. 防止作弊: 如果Deadline包含排队，智能体可通过堆积本地任务延长Deadline
    3. 卸载必要性: 通过设置γ<1可显式强迫卸载，设置γ>1用于放宽
    """
    DEADLINE_TIGHTENING_FACTOR = 0.85  # γ: 紧缩因子 (强迫卸载)
    DEADLINE_TIGHTENING_MIN = 5.0  # γ最小值（大幅放宽以确保V2V+排队+依赖链有充足时间）
    DEADLINE_TIGHTENING_MAX = 8.0  # γ最大值（暴力打通物理层，必须看到Veh% > 0）
    DEADLINE_SLACK_SECONDS = 0.0  # 额外松弛时间（秒），在关键路径基础上附加

    # 说明: γ<1时，本地计算更难满足Deadline；γ>1时Deadline更宽松

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
    EFF_SCALE = 5.0   # λ: tanh缩放因子 (高灵敏度)

    # B. 拥塞惩罚参数 (Congestion Penalty)
    CONG_WEIGHT = 0.5  # β: 拥塞惩罚权重系数
    CONG_GAMMA = 2.0   # γ: 拥塞敏感度指数 (平方增长)

    # C. 软约束惩罚参数 - 距离预警 (Soft Constraint - Distance Warning)
    DIST_PENALTY_WEIGHT = 2.0    # λ_d: 距离预警权重，最大产生-2.0惩罚
    DIST_SAFE_FACTOR = 0.8       # 安全距离因子，通信半径的80%为安全区
    DIST_SENSITIVITY = 2.0       # κ: 距离敏感度，平方增长
    DIST_PENALTY_MODE = "on"     # {"on","off","risk"} 默认使用距离预警
    
    # D. 软约束惩罚参数 - 超时惩罚 (Soft Constraint - Timeout)
    TIMEOUT_PENALTY_WEIGHT = 1.0  # η: 超时惩罚上限（大幅降低，避免step惩罚淹没terminal奖励）
    TIMEOUT_STEEPNESS = 3.0       # σ: 超时陡峭度，确保微小超时也有显著惩罚

    # E. 硬约束惩罚参数 (Hard Constraint) - 直接覆盖
    PENALTY_LINK_BREAK = -10.0   # 链路断开/超出范围（触发后直接返回）
    PENALTY_OVERFLOW = -10.0     # 队列溢出（触发后直接返回）
    PENALTY_FAILURE = -10.0      # 任务失败惩罚（超时）
    TIME_LIMIT_PENALTY = -1.0    # episode 因时间截断的额外终端惩罚
    # time_limit penalty模式
    TIME_LIMIT_PENALTY_MODE = "fixed"  # {"fixed","scaled"}
    TIME_LIMIT_PENALTY_K = 2.0
    TIME_LIMIT_PENALTY_RATIO_CLIP = 3.0

    # F. 成功奖励参数 (Success Bonus) - 稀疏奖励强化
    SUCCESS_BONUS = 20.0  # 任务成功完成时的固定奖励（增大以提高V2V探索动力）
    SUBTASK_SUCCESS_BONUS = 2.0  # 单个V2V/RSU子任务成功完成时的奖励（计件工资）
    BONUS_MODE = "both"  # {"none","subtask","success","both"} 默认保持当前双重奖励

    # G. 时延/能耗代价权重（用于负增量成本奖励）
    DELAY_WEIGHT = 1.0
    ENERGY_WEIGHT = 0.5

    # H. Reward 模式（绝对潜在值成形）
    REWARD_MODE = "delta_cft"  # {"incremental_cost","delta_cft"} —— 已切到绝对时间差公式
    # Scales absolute time savings (seconds) to reward points. 10.0 means 0.1s saved = +1.0 reward.
    DELTA_CFT_SCALE = 10.0
    # Penalty weight for normalized energy. 0.5 means max power usage cost is equivalent to 0.05s delay penalty.
    DELTA_CFT_ENERGY_WEIGHT = 0.5
    # 绝对时间差裁剪范围，防止估计跳变导致梯度爆炸
    DELTA_CFT_CLIP_MIN = -1.0
    DELTA_CFT_CLIP_MAX = 1.0

    # H. 奖励范围控制
    REWARD_MAX = 30.0    # 奖励上限（扩大以容纳SUCCESS_BONUS）
    REWARD_MIN = -15.0   # 奖励下限（扩大以容纳大惩罚）

    # I. 动作/奖励调试开关
    DEBUG_ASSERT_ILLEGAL_ACTION = False  # True时illegal_action直接断言
    DEBUG_ASSERT_METRICS = False  # True时对成功率/决策分布做范围断言
    EPISODE_JSONL_STDOUT = True  # 是否在stdout打印每个episode的JSONL

    # =========================================================================
    # 8. 动态归一化与统计
    # =========================================================================
    NORM_RATE_MODE = "static"  # {"static","ema_p95"} max_rate归一化策略
    RATE_EMA_ALPHA = 0.05      # EMA平滑系数
    RATE_RESERVOIR_SIZE = 256  # 速率P95估计样本大小
    RATE_MIN_SAMPLES = 32      # P95/EMA最小样本数（冷启动回退静态）

    STATS_RESERVOIR_SIZE = 256  # 统计P95估计样本大小
    STATS_SEED = 0

    # =========================================================================
    # 9. 辅助方法
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


PROFILE_REGISTRY = {
    "train_v2v_competitive_v1": {
        "F_RSU": 6.0e9,
        "RSU_NUM_PROCESSORS": 2,
        "RSU_QUEUE_CYCLES_LIMIT": 8.0e9,
        "BW_V2I": 15.0e6,
        "BW_V2V": 40.0e6,
        "V2V_INTERFERENCE_DBM": -110.0,
        "V2V_RANGE": 350.0,
        "VEHICLE_QUEUE_CYCLES_LIMIT": 6.0e9,
        "REWARD_MODE": "delta_cft",
        "BONUS_MODE": "none",
        "DELTA_CFT_SCALE": 8.0,
        "DELTA_CFT_ENERGY_WEIGHT": 0.03,
    },
    # train_ready_v1: keep V2V competitiveness, relax completion budget modestly.
    # Typical Tcmp≈E[C]/E[f], E[C]=(MIN_COMP+MAX_COMP)/2; Ttx≈E[D]/E[R], E[D]=(MIN_DATA+MAX_DATA)/2.
    # Ensure MAX_STEPS*DT covers several (Ttx+Wait+Tcmp) to avoid zero success at startup.
    "train_ready_v1": {
        "F_RSU": 6.0e9,
        "RSU_NUM_PROCESSORS": 2,
        "RSU_QUEUE_CYCLES_LIMIT": 8.0e9,
        "BW_V2I": 15.0e6,
        "BW_V2V": 40.0e6,
        "V2V_INTERFERENCE_DBM": -110.0,
        "V2V_RANGE": 350.0,
        "VEHICLE_QUEUE_CYCLES_LIMIT": 6.0e9,
        "MAX_STEPS": 300,
        "VEHICLE_ARRIVAL_RATE": 0.05,
        "MIN_COMP": 0.4e8,
        "MAX_COMP": 1.6e8,
        "MIN_DATA": 0.8e6,
        "MAX_DATA": 2.4e6,
        "MIN_EDGE_DATA": 0.16e6,
        "MAX_EDGE_DATA": 0.48e6,
        "DEADLINE_TIGHTENING_FACTOR": 1.05,
        "DEADLINE_TIGHTENING_MIN": 8.0,
        "DEADLINE_TIGHTENING_MAX": 12.0,
        "REWARD_MODE": "delta_cft",
        "BONUS_MODE": "none",
        "DELTA_CFT_SCALE": 8.0,
        "DELTA_CFT_ENERGY_WEIGHT": 0.03,
        "EPISODE_JSONL_STDOUT": False,
    },
    # train_ready_v2: reduce difficulty variance, tighten DAG load, boost RSU, and slightly relax deadlines.
    "train_ready_v2": {
        # 1) Disable dynamic arrivals to reduce episode variability
        "VEHICLE_ARRIVAL_RATE": 0.0,
        # 2) Narrow DAG load distribution
        "MIN_COMP": 0.8e8,
        "MAX_COMP": 1.2e8,
        "MIN_DATA": 1.0e6,
        "MAX_DATA": 2.0e6,
        "MIN_EDGE_DATA": 0.2e6,
        "MAX_EDGE_DATA": 0.4e6,
        # 3) Stronger RSU to mitigate queue-induced misses
        "F_RSU": 10.0e9,
        "RSU_NUM_PROCESSORS": 4,
        "RSU_QUEUE_CYCLES_LIMIT": 20.0e9,
        "VEHICLE_QUEUE_CYCLES_LIMIT": 6.0e9,
        # 4) Deadline lightly relaxed vs train_ready_v1 (~+20%)
        "DEADLINE_TIGHTENING_FACTOR": 1.05,
        "DEADLINE_TIGHTENING_MIN": 12.0,
        "DEADLINE_TIGHTENING_MAX": 16.0,
        # 5) Time discretization unchanged
        "MAX_STEPS": 300,
        # 6) Reward mode unchanged (delta_cft)
        "REWARD_MODE": "delta_cft",
        "BONUS_MODE": "none",
        "DELTA_CFT_SCALE": 8.0,
        "DELTA_CFT_ENERGY_WEIGHT": 0.03,
    }
    ,
    # train_ready_v3: heavier tasks to avoid 6~9 step easy wins; moderate RSU; stronger V2V; relaxed deadlines.
    "train_ready_v3": {
        "MAX_STEPS": 300,
        # Heavier DAG loads to reduce fast completion
        "MIN_COMP": 0.8e9,
        "MAX_COMP": 2.4e9,
        "MIN_DATA": 8.0e6,
        "MAX_DATA": 2.0e7,
        "MIN_EDGE_DATA": 1.6e6,
        "MAX_EDGE_DATA": 4.0e6,
        # RSU moderate
        "F_RSU": 6.0e9,
        "RSU_NUM_PROCESSORS": 2,
        "RSU_QUEUE_CYCLES_LIMIT": 12.0e9,
        # Keep vehicle queue in line with larger tasks
        "VEHICLE_QUEUE_CYCLES_LIMIT": 6.0e9,
        # Communication: stronger V2V, moderate V2I
        "BW_V2I": 15.0e6,
        "BW_V2V": 40.0e6,
        "V2V_INTERFERENCE_DBM": -110.0,
        "V2V_RANGE": 350.0,
        # Deadline relaxed for heavier load
        "DEADLINE_TIGHTENING_FACTOR": 1.05,
        "DEADLINE_TIGHTENING_MIN": 18.0,
        "DEADLINE_TIGHTENING_MAX": 24.0,
        # Reward scale for delta_cft
        "REWARD_MODE": "delta_cft",
        "BONUS_MODE": "none",
        "DELTA_CFT_SCALE": 15.0,
        "DELTA_CFT_ENERGY_WEIGHT": 0.03,
    }
    ,
    # train_ready_v4: literature-scale tasks, critical-path deadlines in seconds-scale.
    "train_ready_v4": {
        "MAX_STEPS": 300,
        "DT": 0.05,
        # DAG size
        "MIN_NODES": 8,
        "MAX_NODES": 15,
        # Task loads (cycles)
        "MIN_COMP": 2.0e7,
        "MAX_COMP": 6.0e7,
        # Node input data (bits) ~ 500KB-1500KB
        "MIN_DATA": 4.096e6,
        "MAX_DATA": 12.288e6,
        # Edge data (bits) ~ 100KB-500KB
        "MIN_EDGE_DATA": 0.8192e6,
        "MAX_EDGE_DATA": 2.4576e6,
        # Vehicle CPU range (Hz)
        "MIN_VEHICLE_CPU_FREQ": 0.5e9,
        "MAX_VEHICLE_CPU_FREQ": 2.0e9,
        # RSU CPU (Hz)
        "F_RSU": 8.0e9,
        "RSU_NUM_PROCESSORS": 4,
        "RSU_QUEUE_CYCLES_LIMIT": 12.0e9,
        "VEHICLE_QUEUE_CYCLES_LIMIT": 2.0e9,
        # Wireless
        "BW_V2I": 100.0e6,
        "BW_V2V": 40.0e6,
        "RSU_RANGE": 250.0,
        "V2V_RANGE": 350.0,
        "NOISE_POWER_DBM": -100.0,
        "TX_POWER_UP_DBM": 20.0,
        "TX_POWER_V2V_DBM": 20.0,
        "TX_POWER_MIN_DBM": 20.0,
        "TX_POWER_MAX_DBM": 20.0,
        # Deadlines (critical-path based)
        "DEADLINE_TIGHTENING_MIN": 2.0,
        "DEADLINE_TIGHTENING_MAX": 4.0,
        "DEADLINE_SLACK_SECONDS": 0.3,
        # Reward
        "REWARD_MODE": "delta_cft",
        "BONUS_MODE": "none",
        "DELTA_CFT_SCALE": 15.0,
        "DELTA_CFT_ENERGY_WEIGHT": 0.03,
    }
}


def _recompute_derived(cls):
    cls.MAX_NEIGHBORS = max(0, min(cls.NUM_VEHICLES - 1, cls.V2V_TOP_K))
    cls.MAX_TARGETS = 2 + cls.MAX_NEIGHBORS
    cls.MAX_VELOCITY = cls.VEL_MAX

    cls.MEAN_COMP_LOAD = (cls.MIN_COMP + cls.MAX_COMP) / 2
    cls.AVG_COMP = (cls.MIN_COMP + cls.MAX_COMP) / 2

    cls._RSU_MAX_WAIT = cls.RSU_QUEUE_LIMIT * cls.MEAN_COMP_LOAD / cls.F_RSU
    cls._VEHICLE_MAX_WAIT = cls.VEHICLE_QUEUE_LIMIT * cls.MEAN_COMP_LOAD / cls.MIN_VEHICLE_CPU_FREQ
    cls.DYNAMIC_MAX_WAIT_TIME = max(cls._RSU_MAX_WAIT, cls._VEHICLE_MAX_WAIT) * 1.2
    cls.NORM_MAX_WAIT_TIME = cls.DYNAMIC_MAX_WAIT_TIME


def apply_profile(profile_name):
    if not profile_name:
        return False
    profile = PROFILE_REGISTRY.get(profile_name)
    if profile is None:
        if os.environ.get("CFG_PROFILE_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
            print(f"[Cfg] profile not found: {profile_name}")
        return False
    for key, value in profile.items():
        setattr(SystemConfig, key, value)
    _recompute_derived(SystemConfig)
    if os.environ.get("CFG_PROFILE_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
        print(f"[Cfg] applied profile={profile_name} overrides={len(profile)}")
    return True


apply_profile(os.environ.get("CFG_PROFILE"))


def _env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


SystemConfig.EPISODE_JSONL_STDOUT = _env_flag(
    "EPISODE_JSONL_STDOUT", SystemConfig.EPISODE_JSONL_STDOUT
)
