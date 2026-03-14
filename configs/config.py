import numpy as np


class SystemConfig:
    """
    [系统参数配置类] SystemConfig
    System Configuration Class

    作用 (Purpose):
    集成通信模型、计算资源、物理约束、DAG生成参数及强化学习归一化常量。
    Integrates communication model, computing resources, physical constraints, 
    DAG generation parameters, and RL normalization constants.

    参考文献 (References):
    - VEC: Mao et al., "A Survey on Mobile Edge Computing" (2017)
    - DAG Offloading: Chen et al., "Dependency-Aware Task Offloading" (2020)
    - C-V2X: 3GPP TS 36.213, ETSI EN 302 637-2

    单位标准 (SI Units):
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
    SEED = 42  # 随机种子 (Random seed for reproducibility)
    
    # -------------------------------------------------------------------------
    # 1.1 道路模型参数 (Road Model)
    # -------------------------------------------------------------------------
    MAP_SIZE = 1000.0       # 道路长度 (m) - Road segment length
                            # 主线基础环境固定为 1 km 量级直路，兼容现有单路段推进流程
    
    NUM_LANES = 2           # 车道数量 - Number of lanes
                            # 影响: 车辆横向分布，减少换道干扰
                            # Impact: Vehicle lateral distribution, reduces lane-change interference
    
    LANE_WIDTH = 3.5        # 车道宽度 (m) - Lane width (standard)
                            # 影响: 标准车道宽度，影响V2V距离计算
                            # Impact: Standard lane width, affects V2V distance calculation
    
    # -------------------------------------------------------------------------
    # 1.2 车辆参数 (Vehicle Parameters)
    # -------------------------------------------------------------------------
    NUM_VEHICLES = 20       # 初始车辆数 - Initial number of vehicles
                            # 主线保持固定 agent 数，只通过外生几何采样改变有效邻居数

    V2V_TOP_K = 5           # V2V候选数上限 - Max V2V candidates per agent [调优: 11→5]
    # 候选集模式（锁定为ALL_FEASIBLE）
    # ALL: 保留通信范围内所有可达邻居，仅通过action_mask屏蔽不可行动作
    CANDIDATE_MODE = "ALL"
    TOPK_K = V2V_TOP_K
    RANDOMK_K = V2V_TOP_K
    CANDIDATE_SORT_BY = "t_finish"  # t_finish | rate | distance (V2V候选排序规则)
    DEBUG_CANDIDATE_SET = False     # 是否打印候选集排序与映射
    V2V_DYNAMIC_K = True    # 是否启用动态邻居数选择 - Enable dynamic V2V candidate filtering
    V2V_TOP_K_MIN = 2       # 动态筛选最少保留的V2V候选数 - Min V2V candidates to keep
    V2V_CANDIDATE_REL_TOL = 0.25  # 相对最优时间允许阈值 - Relative time tolerance (best*(1+rel))
    V2V_CANDIDATE_ABS_TOL = 0.2   # 绝对时间允许阈值(秒) - Absolute time tolerance (seconds)
                            # 影响: 减少V2V冗余，使邻居选择更有意义
                            # Impact: Reduces V2V redundancy, makes neighbor selection more meaningful
    
    # ALL-feasible模式：使用固定上界 NUM_VEHICLES-1 作为V2V slot数
    # TOPK/RANDOMK模式：使用 V2V_TOP_K 作为上限
    ALL_FEASIBLE = (CANDIDATE_MODE == "ALL")
    MAX_NEIGHBORS = (NUM_VEHICLES - 1) if ALL_FEASIBLE else max(0, min(NUM_VEHICLES - 1, V2V_TOP_K))
    
    # Action space配置移至NUM_RSU定义之后（见1.4节末尾）
    # ENABLE_RSU_SELECTION和MAX_TARGETS定义在NUM_RSU之后
    
    VEHICLE_ARRIVAL_RATE = 0.0  # 泊松到达率 (veh/s) - Poisson arrival rate
                                # 影响: 动态车辆生成，0表示禁用动态到达
                                # Impact: Dynamic vehicle spawning; 0 disables dynamic arrivals
    
    # -------------------------------------------------------------------------
    # 1.3 车辆移动性参数 (Vehicle Mobility - Truncated Normal Distribution)
    # -------------------------------------------------------------------------
    VEL_MEAN = 10.0         # 速度均值 (m/s) ≈ 36 km/h
    VEL_STD = 2.0           # 速度标准差 (m/s)
    VEL_MIN = 5.56          # 最小速度 (m/s) ≈ 20 km/h
    VEL_MAX = 13.89         # 最大速度 (m/s) ≈ 50 km/h
    
    MAX_VELOCITY = VEL_MAX  # 最大速度 - Maximum velocity
    
    # 车辆生成位置范围 (相对于MAP_SIZE的比例)
    # Vehicle spawn position range (ratio of MAP_SIZE)
    VEHICLE_SPAWN_X_MIN = 0.0   # 最小X位置比例 - Min X position ratio
    VEHICLE_SPAWN_X_MAX = 1.0   # 最大X位置比例 - Max X position ratio
    
    DT = 0.1                # 仿真时间步长 (s) - Simulation time step (文献二)
                            # 影响: 精度与计算开销权衡，0.1s降低50%开销
                            # Impact: Accuracy vs. computation tradeoff; 0.1s reduces 50% overhead

    MAX_STEPS = 300         # Episode最大步数 - Max steps per episode
                            # 影响: Episode总时长 = MAX_STEPS × DT = 30秒
                            # Impact: Total episode duration = MAX_STEPS × DT = 30s
    TERMINATE_ON_ALL_FINISHED = True  # 是否允许任务全部完成时提前终止

    # -------------------------------------------------------------------------
    # 1.3.1 调度与决策稀疏控制 (Scheduling Controls)
    # -------------------------------------------------------------------------
    MAX_SCHEDULE_PER_STEP = 1  # 每车每步最多调度的READY子任务数 (1=保持现状)
    MAX_INFLIGHT_SUBTASKS_PER_VEHICLE = 0  # 每车在途子任务上限 (0=不限制)
    
    # -------------------------------------------------------------------------
    # 1.4 RSU部署参数 (RSU Deployment)
    # -------------------------------------------------------------------------
    NUM_RSU = 2             # RSU数量 - Number of RSUs
    
    RSU_Y_DIST = 10.0       # RSU离路边距离 (m) - RSU distance from roadside
                            # 影响: V2I路径损耗，10m为标准高度
                            # Impact: V2I path loss; 10m is standard height
    
    RSU_RANGE = 300.0       # RSU覆盖半径 (m) - RSU coverage radius
    
    RSU_POS = np.array([500.0, 500.0])  # 默认RSU位置（向后兼容）
                                         # Default RSU position (backward compatibility)
    
    # -------------------------------------------------------------------------
    # 1.5 Action Space配置 (Action Space Configuration)
    # -------------------------------------------------------------------------
    # 主线固定为 concrete target selection:
    # Local(1) + specific RSU(NUM_RSU) + specific V2V helper(MAX_NEIGHBORS)
    ENABLE_RSU_SELECTION = True
    MAX_TARGETS = 1 + NUM_RSU + MAX_NEIGHBORS

    # =========================================================================
    # 2. 通信参数 (Communication Model)
    # =========================================================================
    # V2V 小尺度衰落开关（关闭=只用大尺度路损，训练可复现、方差可控）
    USE_BLOCK_FADING = False
    # CFT缓存哈希严格模式：包含DAG状态/队列摘要，避免错误复用
    CFT_CACHE_STRICT_KEY = True
    # -------------------------------------------------------------------------
    # 2.1 无线物理层参数 (Wireless Physical Layer - C-V2X / DSRC)
    # -------------------------------------------------------------------------
    FC = 5.9e9              # 载波频率 (Hz) - Carrier frequency (5.9 GHz for C-V2X)
                            # 影响: 路径损耗计算基准
                            # Impact: Path loss calculation baseline
    
    C_LIGHT = 3e8           # 光速 (m/s) - Speed of light
    
    BW_V2I = 10.0e6         # V2I带宽 (Hz) - V2I bandwidth (10 MHz)
                            # 主线: RB_SINR + 每RSU内竞争 + ICI

    BW_V2V = 20.0e6         # V2V带宽 (Hz) - V2V bandwidth (20 MHz)
                            # 主线: resource-pool / RB reuse + same-RB interference

    # 主线固定为 RB_SINR:
    # RSU 内 RB 正交 + 超载 round-robin 时分 + 跨 RSU 同 RB 干扰
    V2I_RATE_MODEL = "RB_SINR"
    V2I_RB_BW_HZ = 180e3               # 单RB带宽口径（ETSI/3GPP常用）
    V2I_NUM_RB = 24         # 主线默认：RB_SINR 竞争口径
    V2I_ICI_ENABLED = True             # 轻量跨RSU上行干扰开关
    V2I_FREQ_REUSE_FACTOR = 1          # reuse=1 表示全频复用
    
    # -------------------------------------------------------------------------
    # 2.2 噪声参数 (Noise Parameters)
    # -------------------------------------------------------------------------
    NOISE_POWER_DENSITY_DBM = -174  # 热噪声功率谱密度 (dBm/Hz) - Thermal noise PSD
    NOISE_FIGURE = 9                # 噪声系数 (dB) - Noise figure
    NOISE_POWER_DBM = -95.0         # UNUSED_IN_MAINLINE: 主线信道使用 PSD+NF 计算噪声
                                    # 仅保留作旧实验/诊断参考，不代表主线执行模型
    
    # -------------------------------------------------------------------------
    # 2.3 发射功率参数 (Transmit Power - FCC Compliant)
    # -------------------------------------------------------------------------
    TX_POWER_UP_DBM = 20.0      # ESTIMATE_ONLY: 仅供handover参考信号排序/诊断使用
                                # 主线远程传输实际走统一动作映射 [TX_POWER_MIN_DBM, TX_POWER_MAX_DBM]

    TX_POWER_V2V_DBM = 20.0     # UNUSED_IN_MAINLINE: 主线V2V也走统一动作映射功率
                                # 保留仅为旧实验兼容，不代表主线执行模型

    TX_POWER_MIN_DBM = 13.0     # 功控下限 (dBm) - Power control lower bound [审计调优v2]
                                # 影响: 扩大功率范围，增强功率梯度可见性
                                # Impact: Expands power range; enhances power gradient visibility

    TX_POWER_MAX_DBM = 23.0     # 功控上限 (dBm) - Power control upper bound [审计调优]
                                # 影响: 避免SNR过早饱和，保持功率梯度
                                # Impact: Prevents SNR saturation; maintains power gradient
    
    # -------------------------------------------------------------------------
    # 2.4 路径损耗模型 (Path Loss Model - Log-Distance)
    # 公式: PL(d) = PL_ALPHA + 10 * PL_BETA * log10(d/d_0)
    # -------------------------------------------------------------------------
    PL_ALPHA_V2I = 28.0     # UNUSED_IN_MAINLINE: 主线信道不使用该截距参数
                            # 当前主线路损实现见 envs/modules/channel.py 中 beta0 * d^{-alpha}
    
    PL_BETA_V2I = 2.5       # V2I路损指数 - V2I path loss exponent (LOS environment)
                            # 影响: V2I衰减速率，2.5为LOS标准
                            # Impact: V2I attenuation rate; 2.5 is standard for LOS
    
    PL_ALPHA_V2V = 28.0     # UNUSED_IN_MAINLINE: 主线V2V也不使用该截距参数
    PL_BETA_V2V = 3.3       # V2V路损指数 - V2V path loss exponent (仍高于V2I=2.5，保持更差链路)
                            # 影响: V2V衰减速率，3.5模拟NLOS遮挡/干扰
                            # Impact: V2V attenuation rate; 3.5 models NLOS obstruction/interference
    
    V2V_INTERFERENCE_DBM = -95.0  # ESTIMATE_ONLY: 仅供单链路估计/诊断fallback使用
                                  # 主线多链路V2V干扰来自 same-RB reuse 的显式干扰和
    
    V2V_RANGE = 200.0       # V2V通信半径 (m) - V2V communication range

    # -------------------------------------------------------------------------
    # 2.5.1 V2V RB 干扰模型 (V2V Resource Block Interference)
    # -------------------------------------------------------------------------
    V2V_NUM_RB = 10                 # V2V 子信道（RB）数量
    V2V_BW_PER_RB = BW_V2V / V2V_NUM_RB  # 每 RB 带宽 (Hz)
    # 功率映射硬编码对数域: P = Pmin*(Pmax/Pmin)^a_power, a_power∈[0,1]

    # -------------------------------------------------------------------------
    # 2.5.2 RSU 切换代价 (RSU Handover Cost)
    # -------------------------------------------------------------------------
    HO_FREEZE_STEPS = 2            # 切换后冻结步数（V2I 速率=0）
    HO_HYST_DB = 3.0               # 滞回门限 (dB)，新 RSU 参考信号须超过当前+此值
    MIN_RSU_STAY_STEPS = 5         # 最小驻留步数，切换后至少驻留此步数才可再切换
    V2I_HANDOVER_DELAY_STEPS = 2
    MAX_V2V_RETRY = 2

    # -------------------------------------------------------------------------
    # 2.5.3 信誉外生过程 (Reputation / Trust External Process)
    # -------------------------------------------------------------------------
    TRUST_ENABLED = True            # 主线启用 trust 风险约束与可行性掩码
    TRUST_P_RELIABLE_RANGE = (0.90, 1.0)  # 可靠节点 p_j 范围（无恶意场景：高可靠性）
    TRUST_P_UNRELIABLE_RANGE = (0.3, 0.7) # 不可靠节点 p_j 范围（RELIABLE_PROB=1.0时不生效）
    # [类型化可靠性覆盖] 若非None，则按节点类型覆盖上面的混合分布采样：
    # RSU: TRUST_P_RSU_RANGE, VEH: TRUST_P_VEH_RANGE
    TRUST_P_RSU_RANGE = None      # e.g. (0.95, 0.995)
    TRUST_P_VEH_RANGE = None      # e.g. (0.60, 0.95)
    TRUST_RELIABLE_PROB = 1.0       # 无恶意场景：100%节点均为可靠节点（消除unreliable采样路径）
    TRUST_PRIOR_A = 4.0             # Beta 先验 a：初始hat_rho=4/5=0.8，与高信任场景一致
    TRUST_PRIOR_B = 1.0             # Beta 先验 b（uncertainty=1/5=0.2，随证据快速收敛）
    TRUST_DELAY_STEPS = 3           # 证据延迟步数 tau_rep_steps
    TRUST_RESET_PER_EPISODE = True  # 每 episode 重采样隐藏可靠性
    TRUST_LCB_Z = 1.0
    TRUST_LCB_MIN = 0.35

    # [Chain x Trust] 将信誉证据延迟与链确认时延耦合（不引入共识/打包/费用，只用proxy时延）
    # 当 CHAIN_ENABLED=True 且此开关打开时：
    #   delay_steps = clamp(round(p95_confirm / DT) + base, min, max)
    CHAIN_TRUST_DELAY_COUPLED = False
    CHAIN_TRUST_DELAY_BASE_STEPS = 0
    CHAIN_TRUST_DELAY_MIN_STEPS = 0
    CHAIN_TRUST_DELAY_MAX_STEPS = 50

    # [恶意车辆比例 + 信誉分布 + 风险影响] p_m=0 时行为与原版完全一致
    MALICIOUS_RATIO = 0.0              # 恶意车辆比例 p_m，[0,1]
    REP_INIT_MODE = "beta"             # "beta"=Beta分布采样 | "uniform"=保持原Uniform行为
    REP_HONEST_BETA = (18, 4)          # 诚实节点 Beta(a,b)，均值≈0.818
    REP_MAL_BETA = (4, 18)             # 恶意节点 Beta(a,b)，均值≈0.182
    REP_OVERLAP = 0.05                 # 轻微重叠扰动（加性高斯 std），clip[0,1]
    RSU_REPUTATION = 0.95              # RSU 固定隐藏可靠性（p_j）
    RHO_MAP_MODE = "sigmoid"           # "identity" | "sigmoid"（当前reward使用identity通路）
    RHO_SIGMOID_K = 8.0                # sigmoid 斜率
    RHO_SIGMOID_TAU = 0.5              # sigmoid 中心点
    TRUST_OUTCOME_MODE = "prob_fail"   # 失败注入模式（当前已是 Bernoulli(p_j) 机制）
    TRUST_FAIL_SCOPE = "v2v_only"      # "v2v_only"=RSU不受失败注入 | "all"=全部节点
    TRUST_FAIL_COST_MODE = "no_progress"  # 失败时子任务回 READY（现有行为）

    # -------------------------------------------------------------------------
    # 2.6 域随机化 (Domain Randomization)
    # -------------------------------------------------------------------------
    DOMAIN_RANDOMIZATION = False
    DR_V2V_RANGE_MIN = 200.0
    DR_V2V_RANGE_MAX = 300.0
    DR_RSU_RANGE_MIN = 280.0
    DR_RSU_RANGE_MAX = 420.0
    DR_DAG_FAT_MIN = 0.3
    DR_DAG_FAT_MAX = 0.7
    DR_DAG_DENSITY_MIN = 0.1
    DR_DAG_DENSITY_MAX = 0.4
    
    # -------------------------------------------------------------------------
    # 2.5 速率估计参数 (Rate Estimation - Shannon Capacity Approximation)
    # -------------------------------------------------------------------------
    SNR_MIN_DB = -10.0      # 最低可检测SNR (dB) - Minimum detectable SNR
    SNR_MAX_DB = 25.0       # 饱和SNR (dB) - Saturation SNR [审计调优: 避免近距离功率截断]
    SNR_OFFSET_DB = 10.0    # SNR偏移量 (dB) - SNR offset for positive values
    RICIAN_K_DB = 6.0
    BETA_0_DB = -30

    # =========================================================================
    # 3. 计算资源参数 (Computation Model)
    # =========================================================================
    # -------------------------------------------------------------------------
    # 3.1 CPU频率设定 (CPU Frequency - Heterogeneous Configuration)
    # -------------------------------------------------------------------------
    MIN_VEHICLE_CPU_FREQ = 2.0e9    # 车辆主流算力下界 (Hz)
    MAX_VEHICLE_CPU_FREQ = 4.0e9    # 车辆主流算力上界 (Hz)

    VEH_CPU_DIST_MODE = "uniform"   # uniform | bimodal_helper
    VEH_CPU_HELPER_PROB = 0.25
    VEH_CPU_WEAK_MIN = 1.0e9        # 弱车算力下界 (Hz)
    VEH_CPU_WEAK_MAX = 2.0e9        # 弱车算力上界 (Hz)
    VEH_CPU_HELPER_MIN = 3.0e9      # helper车算力下界 (Hz)
    VEH_CPU_HELPER_MAX = 4.0e9      # helper车算力上界 (Hz)

    # -------------------------------------------------------------------------
    # 3.1.1 统一基础环境 + 外生因素采样协议
    # -------------------------------------------------------------------------
    WORKLOAD_PROFILE_SPECS = {
        "light": {
            "node_range": (5, 8),
            "total_comp": (4.0e8, 1.2e9),
            "total_data": (5.0e5, 2.0e6),
            "edge_data": (3.0e4, 1.8e5),
            "deadline_alpha": (1.50, 1.85),
            "deadline_slack": 0.30,
        },
        "medium": {
            "node_range": (7, 11),
            "total_comp": (0.9e9, 2.2e9),
            "total_data": (1.2e6, 4.6e6),
            "edge_data": (0.8e5, 6.2e5),
            "deadline_alpha": (1.82, 2.22),
            "deadline_slack": 0.78,
        },
        "heavy": {
            "node_range": (9, 12),
            "total_comp": (1.9e9, 3.3e9),
            "total_data": (2.8e6, 6.4e6),
            "edge_data": (1.8e5, 8.8e5),
            "deadline_alpha": (2.1e+00, 2.5e+00),
            "deadline_slack": 1.15,
        },
    }

    EXOGENOUS_LEVELS = ("low", "medium", "high")
    EXOGENOUS_FACTOR_PROBS = {
        "workload_level": (0.34, 0.33, 0.33),
        "traffic_density": (0.33, 0.34, 0.33),
        "helper_availability": (0.33, 0.34, 0.33),
        "contact_stability": (0.33, 0.34, 0.33),
        "rsu_accessibility_or_load": (0.33, 0.34, 0.33),
    }

    WORKLOAD_LEVEL_SPECS = {
        "light": {
            "task_weights": {"light": 0.55, "medium": 0.30, "heavy": 0.15},
            "comp_scale": 0.92,
            "data_scale": 0.92,
            "edge_scale": 0.92,
            "node_shift": -1,
            "deadline_alpha_shift": 0.05,
            "deadline_slack_scale": 1.05,
        },
        "medium": {
            "task_weights": {"light": 0.30, "medium": 0.45, "heavy": 0.25},
            "comp_scale": 0.98,
            "data_scale": 0.98,
            "edge_scale": 0.98,
            "node_shift": 0,
            "deadline_alpha_shift": 0.05,
            "deadline_slack_scale": 1.06,
        },
        "heavy": {
            "task_weights": {"light": 0.15, "medium": 0.35, "heavy": 0.50},
            "comp_scale": 0.98,
            "data_scale": 0.98,
            "edge_scale": 0.98,
            "node_shift": 0,
            "deadline_alpha_shift": 0.10,
            "deadline_slack_scale": 1.15,
        },
    }

    TRAFFIC_DENSITY_SPECS = {
        "low": {
            "spawn_mode": "uniform",
            "anchor_prob": 0.15,
            "cluster_spread": 120.0,
        },
        "medium": {
            "spawn_mode": "anchored",
            "anchor_prob": 0.65,
            "cluster_spread": 85.0,
        },
        "high": {
            "spawn_mode": "anchored",
            "anchor_prob": 0.90,
            "cluster_spread": 50.0,
        },
    }

    HELPER_AVAILABILITY_SPECS = {
        "low": {
            "role_probs": {"weak": 0.22, "regular": 0.75, "helper": 0.03},
            "role_cpu": {
                "weak": (1.0e9, 1.8e9),
                "regular": (2.1e9, 3.0e9),
                "helper": (2.1e+09, 2.7e+09),
            },
            "helper_cpu_scale": 0.96,
        },
        "medium": {
            "role_probs": {"weak": 0.20, "regular": 0.75, "helper": 0.05},
            "role_cpu": {
                "weak": (1.1e9, 1.9e9),
                "regular": (2.2e9, 3.1e9),
                "helper": (2.2e+09, 2.8e+09),
            },
            "helper_cpu_scale": 0.97,
        },
        "high": {
            "role_probs": {"weak": 0.18, "regular": 0.75, "helper": 0.07},
            "role_cpu": {
                "weak": (1.2e9, 2.0e9),
                "regular": (2.3e9, 3.2e9),
                "helper": (2.3e+09, 2.9e+09),
            },
            "helper_cpu_scale": 0.96,
        },
    }

    CONTACT_STABILITY_SPECS = {
        "low": {
            "speed_mean": 10.6,
            "speed_std": 3.0,
        },
        "medium": {
            "speed_mean": 10.0,
            "speed_std": 2.0,
        },
        "high": {
            "speed_mean": 9.4,
            "speed_std": 3.6,
        },
    }

    RSU_ACCESSIBILITY_LOAD_SPECS = {
        "low": {
            "anchor_source": "boundaries",
            "rsu_background_cycles": (0.45e9, 0.70e9),
        },
        "medium": {
            "anchor_source": "mixed",
            "rsu_background_cycles": (0.35e9, 0.60e9),
        },
        "high": {
            "anchor_source": "rsu_centers",
            "rsu_background_cycles": (0.30e9, 0.45e9),
        },
    }

    F_RSU = 5.0e9           # RSU单核频率 (Hz)
                            # 总服务能力由 F_RSU × RSU_NUM_PROCESSORS 给出，主线约 15 GHz

    RSU_NUM_PROCESSORS = 3  # RSU处理器核心数，主线总服务能力约 12 GHz
    
    K_ENERGY = 1e-27        # 能耗系数 - Energy coefficient (Effective Switched Capacitance)
                            # 公式: Energy = K_ENERGY * f^2 * cycles
                            # 影响: 计算能耗估计，频率平方关系
                            # Impact: Computation energy estimation; quadratic frequency relationship
    KAPPA = K_ENERGY        # 兼容别名（cpu_queue_service读取KAPPA）
    
    # -------------------------------------------------------------------------
    # 3.2 队列限制 (Queue Limits - Cycle-Based)
    # -------------------------------------------------------------------------
    VEHICLE_QUEUE_CYCLES_LIMIT = 10.0e9  # 车辆队列上限 (cycles) - Vehicle queue limit [审计调优]
                                        # 影响: 约8个平均任务(1.25G each)，适应新负载
                                        # Impact: ~8 average tasks (1.25G each); adapted to new load

    # [N=20对比建议] 收紧RSU队列上限，提升“集中卸载到RSU”的拥塞代价，
    # 促使策略进行更细粒度分流（Local/RSU/V2V）。
    RSU_QUEUE_CYCLES_LIMIT = 30.0e9     # RSU队列上限 (cycles) - RSU queue limit
                                        # 影响: 适度收紧RSU负载，避免过度偏向RSU
                                        # Impact: Moderately tightens RSU load to avoid over-reliance

    MAX_VEH_QUEUE_SIZE = 20             # 车辆任务缓冲区大小 - Vehicle task buffer size (count)
                                        # 影响: 限制车辆本地任务数，防止内存溢出
                                        # Impact: Limits local task count per vehicle; prevents memory overflow

    # =========================================================================
    # 4. DAG任务生成参数 (Task Generation)
    # =========================================================================
    # -------------------------------------------------------------------------
    # 4.1 DAG结构参数 (DAG Structure)
    # -------------------------------------------------------------------------
    DAG_SOURCE = "synthetic_small"  # synthetic_small | synthetic_large | workflow_json
    DAG_LARGE_NODE_OPTIONS = [20, 50, 100]  # synthetic_large节点数候选
    WORKFLOW_JSON_PATH = "data/workflows/sample_workflow.json"  # workflow_json路径

    MIN_NODES = 8           # DAG最小节点数 - Min DAG nodes
                            # 目标: 关键路径深度落在 4~8 的可学习区间
                            # Target: Keep critical-path depth in a learnable 4~8 regime

    MAX_NODES = 16          # DAG最大节点数 - Max DAG nodes
                            # 目标: 保持结构复杂性但避免200-step窗口内结构性超时
                            # Target: Preserve complexity while avoiding structural timeout in 200-step horizon
    
    DAG_FAT = 1.0           # DAG宽度参数 - DAG width parameter
                            # 目标: 在8~16节点下维持关键路径占比约40%~70%
                            # Target: Keep critical-path ratio around 40%~70% with 8~16 nodes
    
    DAG_DENSITY = 0.24      # DAG连接密度 - DAG edge density
                            # 目标: 在可行域内保留适度依赖与并行性
                            # Target: Preserve moderate dependency/parallelism in feasible regime
    
    DAG_REGULAR = 0.5       # DAG规则性 - DAG regularity
                            # 影响: 结构规则性，0.5为半规则
                            # Impact: Structure regularity; 0.5 is semi-regular
    
    DAG_CCR = 0.2           # 通信计算比 - Communication-to-Computation Ratio
                            # 影响: 结构生成参数，影响边数据量
                            # Impact: Structure generation parameter, affects edge data volume

    # -------------------------------------------------------------------------
    # 4.1.1 两类任务混合 (Two-Class Workload Mixture)
    # -------------------------------------------------------------------------
    # 说明:
    # - 面向 DT=0.1s 的离散仿真，deadline 主体设为 0.4~2.5s（4~25 step），
    #   避免 0.1~0.3s 在该离散粒度下退化为“1~3步硬阈值”问题。
    # - 口径对应:
    #   ETSI TS 122 185/3GPP TS 22.185 的消息级时延等级(20ms/100ms/1000ms)；
    #   Deadline-aware VEC 的任务级 deadline(100ms~5s)；
    #   以及 VEC 轻任务量级(0.1~10Mb, 1e8 cycles)。
    TASK_CLASS_MIX_ENABLE = False
    TASK_CLASS_B_PROB = 0.20

    # A类（协同感知/轻量推理，主流）
    TASK_A_TOTAL_DATA_MIN = 3.0e5    # bits, 0.3 Mb
    TASK_A_TOTAL_DATA_MAX = 1.2e7    # bits, 12 Mb
    TASK_A_TOTAL_COMP_MIN = 1.0e9    # cycles
    TASK_A_TOTAL_COMP_MAX = 4.0e9    # cycles
    TASK_A_DEADLINE_MIN = 0.4        # s (DT=0.1 -> 4 steps)
    TASK_A_DEADLINE_MAX = 1.0        # s (DT=0.1 -> 10 steps)

    # B类（较重任务，少量）
    TASK_B_TOTAL_DATA_MIN = 1.2e7    # bits, 12 Mb
    TASK_B_TOTAL_DATA_MAX = 4.5e7    # bits, 45 Mb
    TASK_B_TOTAL_COMP_MIN = 3.0e9    # cycles
    TASK_B_TOTAL_COMP_MAX = 8.0e9    # cycles
    TASK_B_DEADLINE_MIN = 1.0        # s
    TASK_B_DEADLINE_MAX = 2.5        # s

    # 总量拆分到节点/入口节点时的随机抖动
    TASK_COMP_SPLIT_JITTER = 0.6
    TASK_DATA_SPLIT_JITTER = 0.8
    
    # -------------------------------------------------------------------------
    # 4.2 任务负载参数 (Task Load Parameters)
    # -------------------------------------------------------------------------
    MIN_COMP = 8.0e7        # 子任务最小计算量 (cycles) - Min subtask computation (80 Mcycles)
                            # 口径参考: 常见VEC设定中100M cycles量级任务
                            # Reference: Typical VEC settings include ~100M-cycle task scale

    MAX_COMP = 6.0e8        # 子任务最大计算量 (cycles) - Max subtask computation (600 Mcycles)
                            # 目标: 200-step窗口内保持“可行但不必胜”
                            # Target: Keep tasks feasible-but-nontrivial under 200-step horizon

    # IMPORTANT UNIT: BITS（不是 bytes）
    MIN_DATA = 8.0e5        # 子任务最小数据量 (bits) - Min subtask data (0.8 Mbit, 100 KB)
                            # 口径参考: VEC任务数据量常见 0.1~10 Mb 区间
                            # Reference: VEC workloads often use 0.1~10 Mb data scale

    MAX_DATA = 6.0e6        # 子任务最大数据量 (bits) - Max subtask data (6.0 Mbit, 750 KB)
                            # 目标: 覆盖主任务并包含少量较重样本
                            # Target: Cover main workloads plus a modest heavy tail

    MIN_EDGE_DATA = 5.0e4   # DAG边最小数据量 (bits)
                            # 目标: 保持依赖传输代价可学习且不过度主导
                            # Target: Keep dependency-transfer cost learnable but not dominant

    MAX_EDGE_DATA = 2.0e6   # DAG边最大数据量 (bits)
                            # 目标: StageB中可点亮并发冲突与链路选择差异
                            # Target: Enable concurrency pressure/link-choice contrast in StageB
    
    MEAN_COMP_LOAD = (MIN_COMP + MAX_COMP) / 2  # 平均计算负载 (cycles) = 2.0e9
    AVG_COMP = MEAN_COMP_LOAD
    
    # -------------------------------------------------------------------------
    # 4.3 Deadline计算参数 (Deadline Calculation - Ideal Local Anchoring)
    # -------------------------------------------------------------------------
    """
    设计原则 (Design Principle): 
    "Deadline是铁律，只看硬件能力，不看排队" 
    "Deadline is absolute; only hardware capability matters, not queueing"

    计算公式 (Formula): T_deadline = γ × (W_k / f_local)
    - W_k: 任务总计算量 (Cycles) - Total computation load
    - f_local: 车辆本地CPU频率 (Hz) - Vehicle local CPU frequency
    - γ: 松紧因子 - Tightening factor
        - γ < 1: 趋向强制卸载 (Forces offloading)
        - γ > 1: 放宽Deadline (Relaxes deadline)
        - γ = 1: 理想本地执行时间 (Ideal local execution time)

    为什么排除本地排队？(Why exclude local queueing?)
    1. 业务属性: 防碰撞等时延敏感任务只关心响应速度，不关心CPU忙闲
       Business nature: Safety-critical tasks only care about response time, not CPU status
    2. 防止作弊: 如果Deadline包含排队，智能体可通过堆积本地任务延长Deadline
       Prevent gaming: If deadline includes queueing, agent can extend deadline by piling local tasks
    3. 卸载必要性: 通过设置γ<1可显式强迫卸载
       Offloading necessity: Setting γ<1 explicitly forces offloading
    """
    # -------------------------------------------------------------------------
    # Deadline计算模式选择
    # -------------------------------------------------------------------------
    DEADLINE_MODE = 'LB_ALPHA'          # 选择deadline计算模式:
                                        # - 'CRITICAL_PATH': CP_total / f_median (关键路径，推荐)
                                        # - 'TOTAL_MEDIAN': total_comp / f_median (总量，向后兼容)
                                        # - 'TOTAL_LOCAL': total_comp / f_local (本地算力)
                                        # - 'FIXED_RANGE': 直接从固定范围随机 (秒)

    # 基于计算量的deadline (使用γ因子)
    # 公式: deadline = max(γ × T_base + slack, (1+eps) × LB0)
    # 其中 T_base = CP_total / f_ref, LB0 = CP_total / f_max
    DEADLINE_TIGHTENING_MIN = 1.2       # γ最小值（基于T_base）
                                        # 目标: 形成可行但紧迫的deadline
    DEADLINE_TIGHTENING_MAX = 1.5       # γ最大值（基于T_base）
                                        # 目标: 保持样本多样性与可学习性
    # 可选模式：deadline = alpha * Tmin + slack
    # 其中 Tmin = CP_total / f_max（物理下界）
    # 在保持均值3.0不变的前提下收窄方差，降低episode间任务难度波动（场景更均衡）
    DEADLINE_ALPHA_MIN = 4.0
    DEADLINE_ALPHA_MAX = 6.0
    
    DEADLINE_LB_EPS = 0.02              # 物理下界裕量 eps
                                        # deadline ≥ (1+eps) × LB0 保证不先天不可行
                                        # 推荐范围: 0.05~0.1
    DEADLINE_STEP_GUARD_DELTA = 3.0     # step量化下界保护: LB* = max(LB0, (L_cp+delta)*DT)
                                        # 目的: 避免DT=0.1下过小deadline退化为1~2步硬阈值
    
    # 模式FIXED_RANGE: 固定范围的deadline (秒)
    DEADLINE_FIXED_MIN = 2.0            # 最小deadline (秒)
    DEADLINE_FIXED_MAX = 5.0            # 最大deadline (秒)
    
    DEADLINE_SLACK_SECONDS = 1.0        # 额外松弛时间 (s) - Additional slack time (RC1 calibrated by baseline feasibility)
                                        # 影响: 在关键路径基础上附加
                                        # Impact: Added on top of critical path
    
    # -------------------------------------------------------------------------
    # 4.4 优先级调度权重 (Priority Scheduling Weights)
    # 优先级公式: Score(i) = W1 * L_bwd[i] + W2 * (comp[i] / NORM_MAX_COMP) + W3 * (out_degree[i] / MAX_NODES)
    # -------------------------------------------------------------------------
    PRIORITY_W1 = 100.0     # 后向层级权重 - Backward level weight (dominant, critical path)
                            # 影响: 主导关键路径，确保依赖顺序
                            # Impact: Dominates critical path, ensures dependency order
    
    PRIORITY_W2 = 1.0       # 计算量权重 - Computation weight (tie-breaking within same level)
                            # 影响: 同层级下的tie-breaking
                            # Impact: Tie-breaking within same level
    
    PRIORITY_W3 = 1.0       # 出度权重 - Out-degree weight (tie-breaking with same computation)
                            # 影响: 同计算量下的tie-breaking
                            # Impact: Tie-breaking with same computation

    # =========================================================================
    # 5. 强化学习归一化常量 (RL Normalization Constants)
    # =========================================================================
    NORM_MAX_CPU = 25.0e9       # CPU频率归一化基准 (Hz) - CPU frequency normalization baseline
                                # 影响: RSU频率基准，确保归一化值在[0,1]
                                # Impact: RSU frequency baseline, ensures normalized values in [0,1]
    
    NORM_MAX_COMP = 4.0e9       # 计算量归一化基准 (cycles) - Computation normalization baseline [优化: 2.0→4.0]
                                # 影响: 适应新的MAX_COMP=3.5e9
                                # Impact: Adapts to new MAX_COMP=3.5e9
    
    NORM_MAX_DATA = 4.0e7       # 数据量归一化基准 (bits) - Data normalization baseline
                                # 影响: 覆盖A/B混合任务的输入数据上界，避免长期饱和
                                # Impact: Covers mixed workload upper bound to avoid persistent saturation
    
    NORM_MAX_RATE_V2I = 50e6    # V2I速率归一化基准 (bps) - V2I rate normalization baseline
    NORM_MAX_RATE_V2V = 20e6    # V2V速率归一化基准 (bps) - V2V rate normalization baseline
    
    # 动态计算的等待时间归一化常量 (Dynamically computed wait time normalization)
    # 基于队列上限和处理能力计算最大等待时间
    _RSU_MAX_WAIT = RSU_QUEUE_CYCLES_LIMIT / F_RSU
    _VEHICLE_MAX_WAIT = VEHICLE_QUEUE_CYCLES_LIMIT / MIN_VEHICLE_CPU_FREQ
    NORM_MAX_WAIT_TIME = max(_RSU_MAX_WAIT, _VEHICLE_MAX_WAIT) * 1.2

    # Snapshot oracle / regret diagnostics
    SNAPSHOT_ORACLE_EPS_ABS = 0.01   # 近最优绝对阈值（秒）
    SNAPSHOT_ORACLE_EPS_REL = 0.05   # 近最优相对阈值（5%）

    # =========================================================================
    # 6. 奖励函数参数 (Reward Function - Delta CFT Mode)
    # =========================================================================
    # -------------------------------------------------------------------------
    # 6.1 Delta CFT参数 (Delta Counterfactual Time - Absolute Potential-Based Shaping)
    # -------------------------------------------------------------------------
    """
    奖励模式 (Reward Mode): Delta Counterfactual Time (固定)
    - 基于绝对时间节省的势能成形 (Absolute time-saving potential-based shaping)
    - 公式: r = DELTA_CFT_SCALE * Δt_saved - DELTA_CFT_ENERGY_WEIGHT * power_norm
    - Δt_saved: 实际时间节省 (秒) - Actual time saved (seconds)
    - power_norm: 归一化功率 [0,1] - Normalized power
    """
    DELTA_CFT_SCALE = 10.0          # 时间节省缩放系数 - Time-saving scale factor
                                    # 影响: 0.1s节省 = +1.0奖励
                                    # Impact: 0.1s saved = +1.0 reward
    
    DELTA_CFT_ENERGY_WEIGHT = 0.2   # 能耗惩罚权重 - Energy penalty weight
                                    # 影响: 降低能耗权重，聚焦时延优化（最大功率≈0.02s时延惩罚）
                                    # Impact: Reduced energy weight, focus on latency (max power ≈ 0.02s delay penalty)
    
    DELTA_CFT_CLIP_MIN = -1.0       # 奖励下限裁剪 - Reward lower bound clipping
                                    # 影响: 防止估计跳变导致梯度爆炸
                                    # Impact: Prevents gradient explosion from estimation jumps
    
    DELTA_CFT_CLIP_MAX = 1.0        # 奖励上限裁剪 - Reward upper bound clipping
                                    # 影响: 防止估计跳变导致梯度爆炸
                                    # Impact: Prevents gradient explosion from estimation jumps
    
    # -------------------------------------------------------------------------
    # 6.2 距离预警惩罚 (Distance Warning Penalty - Soft Constraint)
    # -------------------------------------------------------------------------
    DIST_PENALTY_WEIGHT = 2.0       # 距离预警权重 - Distance warning weight
                                    # 影响: 最大产生-2.0惩罚
                                    # Impact: Max penalty of -2.0
    
    DIST_SAFE_FACTOR = 0.8          # 安全距离因子 - Safe distance factor
                                    # 影响: 通信半径的80%为安全区
                                    # Impact: 80% of communication range is safe zone
    
    DIST_SENSITIVITY = 2.0          # 距离敏感度 - Distance sensitivity
                                    # 影响: 平方增长，接近边界时惩罚陡增
                                    # Impact: Quadratic growth; penalty steepens near boundary
    
    # -------------------------------------------------------------------------
    # 6.3 超时惩罚 (Timeout Penalty - Soft Constraint)
    # -------------------------------------------------------------------------
    TIMEOUT_PENALTY_WEIGHT = 1.0    # 超时惩罚上限 - Timeout penalty upper bound
                                    # 影响: 避免step惩罚淹没terminal奖励
                                    # Impact: Prevents step penalty from drowning terminal reward
    
    TIMEOUT_STEEPNESS = 3.0         # 超时陡峭度 - Timeout steepness
                                    # 影响: 微小超时也有显著惩罚
                                    # Impact: Even minor timeout has significant penalty
    
    # -------------------------------------------------------------------------
    # 6.4 硬约束惩罚 (Hard Constraint Penalties - Episode Termination)
    # -------------------------------------------------------------------------
    PENALTY_LINK_BREAK = -20.0      # 链路断开惩罚 - Link break penalty
                                    # 影响: 超出通信范围，直接终止Episode（加重惩罚）
                                    # Impact: Out of communication range; terminates episode (heavier penalty)
    
    PENALTY_OVERFLOW = -20.0        # 队列溢出惩罚 - Queue overflow penalty
                                    # 影响: 队列超限，直接终止Episode（加重惩罚）
                                    # Impact: Queue limit exceeded; terminates episode (heavier penalty)
    
    PENALTY_FAILURE = -20.0         # 任务失败惩罚 - Task failure penalty
                                    # 影响: 超时失败，终止Episode（加重惩罚以强化时延约束）
                                    # Impact: Timeout failure; terminates episode (heavier penalty to enforce deadline)
    
    TIME_LIMIT_PENALTY = -1.0       # 时间截断惩罚 - Time truncation penalty
                                    # 影响: Episode因时间截断的额外terminal惩罚
                                    # Impact: Additional terminal penalty for time truncation
    TIME_LIMIT_PENALTY_MODE = "fixed"      # fixed / scaled
    TIME_LIMIT_PENALTY_K = 2.0             # scaled模式下的惩罚斜率
    TIME_LIMIT_PENALTY_RATIO_CLIP = 3.0    # scaled模式时间比例裁剪上限
    
    # -------------------------------------------------------------------------
    # 6.5 成功奖励 (Success Bonuses - Sparse Reward)
    # -------------------------------------------------------------------------
    SUCCESS_BONUS = 30.0            # 任务成功完成奖励 - Task success bonus
                                    # 影响: 强稀疏奖励强化，显著提高V2V/RSU探索动力
                                    # Impact: Strong sparse reward reinforcement; significantly boosts V2V/RSU exploration
    
    SUBTASK_SUCCESS_BONUS = 2.0     # 子任务成功完成奖励 - Subtask success bonus
                                    # 影响: 计件工资，鼓励单步V2V/RSU成功
                                    # Impact: Piece-rate reward; encourages single-step V2V/RSU success
    
    # -------------------------------------------------------------------------
    # 6.6 奖励范围控制 (Reward Range Control)
    # -------------------------------------------------------------------------
    REWARD_MAX = 40.0               # 奖励上限 - Reward upper bound
                                    # 影响: 容纳SUCCESS_BONUS(30)+时延奖励，防止裁剪
                                    # Impact: Accommodates SUCCESS_BONUS(30)+latency reward; prevents clipping
    
    REWARD_MIN = -25.0              # 奖励下限 - Reward lower bound
                                    # 影响: 容纳PENALTY_FAILURE(-20)，防止裁剪
                                    # Impact: Accommodates PENALTY_FAILURE(-20); prevents clipping
    REWARD_SCALE = 10.0             # 奖励缩放因子 - Reward scaling factor (replace hard clipping)
    # -------------------------------------------------------------------------
    # 6.7 新奖励方案 (Reward Scheme Switch & PBRS Parameters)
    # -------------------------------------------------------------------------
    REWARD_SCHEME = "UNIFIED"       # 奖励方案: "LEGACY_CFT" / "PBRS_KP" / "PBRS_KP_V2" / "UNIFIED"
    REWARD_ALPHA = 1.5              # 基础奖励系数 alpha
    REWARD_BETA = 0.1               # PBRS 系数 beta [审计调优: 降低shape噪声]
    REWARD_GAMMA = 0.99             # PBRS 折扣 gamma（应与训练端 TC.GAMMA 保持一致）
    T_REF = 0.7                     # 时间归一化参考尺度 (s) [审计标定: 取|Δt| p90≈0.69s以降低剪裁]
    PHI_CLIP = 5.0                  # ϕ 裁剪上界（负值幅度上限）
    SHAPE_CLIP = 10.0               # 潜势差分裁剪上界
    R_CLIP = 40.0                   # 总奖励裁剪上界（绝对值）
    EPS_RATE = 1e-9                 # 速率下界，防止除0
    TX_TIMEOUT_SECONDS = 2.0        # 单次传输时间物理上限(s)；超过视为信道中断，用于截断 t_tx 防止数值爆炸（C-V2X 协议级合理上界）
    ILLEGAL_PENALTY = -2.0          # 非法动作额外惩罚
    PBRS_PHI_MODE = "STATE_ONLY"    # STATE_ONLY / LEGACY_ACTION_SNAPSHOT
    PBRS_PHI_POWER_DBM = TX_POWER_MAX_DBM  # state-only Phi固定功率
    PBRS_PHI_ACTION_INVARIANT_CHECK = False  # Phi与action无关性检查

    # -------------------------------------------------------------------------
    # 6.7.1 非法动作惩罚细分 (Illegal Action Penalty Refinement)
    # -------------------------------------------------------------------------
    NO_TASK_PENALTY_DAG_DONE = 0.0      # DAG完成后no_task惩罚（不可控，应为0）
    NO_TASK_PENALTY_BLOCKED = 0.0       # 所有任务依赖阻塞惩罚（部分可控，暂设为0）
    NO_TASK_PENALTY_ASSIGNED = 0.0      # 所有READY任务已分配惩罚（边界情况，暂设为0）
    ILLEGAL_ENABLE_DYNAMIC_PENALTY = False  # 启用动态惩罚（Stage 2）

    TERMINAL_BONUS_SUCC = SUCCESS_BONUS       # 成功终局奖励
    TERMINAL_PENALTY_FAIL = PENALTY_FAILURE   # 失败终局惩罚
    ENERGY_LAMBDA_PBRS = 0.0       # PBRS_KP 通信能耗权重（默认关闭，保持旧方案）
    P_MAX_WATT = 10 ** ((TX_POWER_MAX_DBM - 30) / 10.0)  # 最大功率对应瓦特
    E_REF = 1.0                     # 能耗归一化参考
    E_CLIP = 10.0                   # 能耗裁剪上界

    # -------------------------------------------------------------------------
    # 6.7.2 PBRS_KP_V2 奖励参数 (Reward Scheme V2 Parameters)
    # -------------------------------------------------------------------------
    LAT_ALPHA = 2.0                 # 时延优势奖励系数 (tanh)
    TIMEOUT_L1 = 1.5                # 超时惩罚第一段系数
    TIMEOUT_L2 = 1.5                # 超时惩罚二段系数 (二次)
    TIMEOUT_O0 = 0.2                # 超时分段阈值 (ratio)
    TIMEOUT_K = 3.0                 # 超时惩罚tanh陡峭度
    ENERGY_LAMBDA = 0.03            # V2通信能耗权重
    POWER_LAMBDA = 0.015             # V2功率正则权重

    # -------------------------------------------------------------------------
    # 6.7.3 新统一奖励参数 (Unified Reward - nonlinear, dimensionless)
    # -------------------------------------------------------------------------
    # 终局 Terminal（注意：UNIFIED 路径使用的是 R_SUCC/R_FAIL，不是 TERMINAL_BONUS_*）
    # 当前主奖励为 margin-term-illegal；终局项需要足够强作为目标锚点，避免 reward 过稀疏。
    # 取 24 作为折中：高于过弱的 12，但显著低于会长期主导的 50。
    R_SUCC = 24.0                   # 成功奖励系数 R_s（UNIFIED terminal）
    R_FAIL = 24.0                   # 失败惩罚系数 R_f（UNIFIED terminal）
    P_SUCC = 1.0                    # 成功奖励指数 p_s
    P_FAIL = 1.0                    # 失败惩罚指数 p_f
    # 每步 Step-wise
    W_TIME = 0.35                   # 时间推进惩罚权重 w_t（保守默认；更大尺度需经中长程验证后再固化）
    DT_IDLE = 0.01                  # 时间项最小步长，dt_used=max(dt,DT_IDLE) 防止dt=0套利
    # 主奖励固定为:
    # 真实时间推进 + 终局成功/失败 + 轻量能耗/真实干扰 + illegal
    W_ENERGY = 0.05                 # 能耗惩罚权重 w_e
    P_ENERGY = 1.5                  # 能耗惩罚指数 p_e (>1 超线性惩罚极端功率)
    W_INTERF = 0.03                 # 干扰惩罚权重 w_I
    P_INTERF = 1.5                  # 干扰惩罚指数 p_I (>1 超线性惩罚极端干扰)
    W_ILLEGAL = 30.0                # 非法动作惩罚 w_ill
    REWARD_PROGRESS_TNORM = 2.5
    REWARD_PROGRESS_RMAX = 1.0
    R_SUCCESS_ANCHOR = 3.0
    ALPHA_SUCCESS = 2.0
    R_FAIL_ANCHOR = 3.0
    ALPHA_FAIL = 2.0
    ALPHA_MISS = 2.0
    MISS_CAP = 1.0
    # E_ref / I_ref
    # 奖励中能耗项仅统计 INPUT_TX 发射能耗（comm_queue_service: energy = p_tx * time_used）。
    # 实测上界: P_MAX_WATT * DT = 0.1995W * 0.1s ≈ 0.02J。将 E_REF 设为与上界同量级，
    # 使 energy_ratio 在正常工作点附近为 O(1)，确保梯度有效传播。
    E_REF_UNIFIED = 0.02
    I_REF_D0 = V2V_RANGE / 2.0     # I_ref 参考距离 d0
    # EMA能耗下界: 必须 >= E_REF_UNIFIED，防止policy全-Local时energy_ref漂移到1e-8，
    # 导致任何远程发射都触发max-clip，彻底摧毁功率控制梯度。
    REWARD_REF_ENERGY_MIN = 0.02    # = E_REF_UNIFIED，锚定物理上界（P_MAX × DT）
    REWARD_REF_EMA_ALPHA = 0.05     # 奖励参考尺度EMA平滑系数
    REWARD_REF_EMA_CAP = 1e3        # 奖励参考尺度EMA上界
    REWARD_REF_WARMUP_EPISODES = 200  # 前N个episode更新EMA，之后冻结
    REWARD_REF_FREEZE_AFTER_WARMUP = True
    # 干扰EMA下界按当前日志口径重标定: i_caused_input典型量级在1e-10~1e-9 W，
    # floor过高会把I_caused/I_ref压到近0，导致r_interf长期失活。
    I_REF_MIN_UNIFIED = 3e-9
    # 对 I_caused/I_ref 做上界裁剪，避免单项奖励主导训练。
    INTERF_RATIO_CLIP_UNIFIED = 3.0
    REWARD_SCALE_E_RATIO_CAP = 1.5   # 训练前尺度核验用：energy_norm上界假设
    # PBRS
    ENABLE_PBRS = False             # 主线固定关闭 PBRS
    PBRS_BETA = 0.02                # 仅供 archive/diagnostics 对照
    PBRS_GAMMA = 0.99               # PBRS 折扣 gamma
    PBRS_Q = 1.0                    # Phi 指数 q=1（线性势能，简化参数；q=1.2与1.0行为接近）
    PBRS_EPS = 1e-3                 # Phi 下界 eps
    PBRS_PHI_CLIP_UNIFIED = 5.0     # Phi 裁剪

    # -------------------------------------------------------------------------
    # 6.8 结算风险层 (Settlement Risk Layer - Chain Proxy)
    # -------------------------------------------------------------------------
    CHAIN_ENABLED = False
    CHAIN_MODE = "NONE"             # NONE / CONST / SWITCH
    CHAIN_OBS_DIM = 5               # [p50, p95, p_fail, mempool_len, rho]
    CHAIN_USE_IN_OBS = True
    CHAIN_RISK_WEIGHT_DEPOSIT = 0.0  # alpha_D
    CHAIN_RISK_WEIGHT_FAIL = 0.0     # alpha_F
    CHAIN_DEPOSIT_BASE = 1.0
    CHAIN_DEPOSIT_SCALE = 0.0
    CHAIN_CHARGE_RSU = True
    CHAIN_BATCH_K = 1

    # Level-1: switch mode parameters
    CHAIN_SWITCH_PERIOD_STEPS = 200
    CHAIN_P50_LOW = 0.0
    CHAIN_P95_LOW = 0.0
    CHAIN_PFAIL_LOW = 0.0
    CHAIN_P50_HIGH = 0.0
    CHAIN_P95_HIGH = 0.0
    CHAIN_PFAIL_HIGH = 0.0
    CHAIN_NOISE_STD = 0.0
    

    # =========================================================================
    # 7. 调试与日志参数 (Debug & Logging)
    # =========================================================================
    DEBUG_PBRS_AUDIT = False                # PBRS一致性审计/打点开关
    AUDIT_PER_DECISION_REWARD = False       # per-decision奖励分项审计开关
    AUDIT_FATAL_ON_SCHEME_ERROR = False     # 审计检查失败时是否中断训练（False=仅警告）
    EDGE_RATE_RECOMPUTE_AUDIT = False       # EDGE速率重算审计开关
    DEBUG_ASSERT_METRICS = False            # 训练端指标范围断言（用于调试）
    
    EPISODE_JSONL_STDOUT = True             # Episode JSONL输出 - Episode JSONL output
                                            # 影响: 是否在stdout打印每个episode的JSONL
                                            # Impact: Whether to print episode JSONL to stdout

    # =========================================================================
    # 8. 模型结构参数 (Model Architecture)
    # =========================================================================
    RESOURCE_RAW_DIM = 16           # 资源原始特征维度
                                    # [0] cpu_norm
                                    # [1] comp_backlog_norm
                                    # [2] tx_backlog_norm
                                    # [3] dist_norm
                                    # [4] rel_x
                                    # [5] rel_y
                                    # [6] rel_speed_norm
                                    # [7] contact_norm
                                    # [8] contention_norm
                                    # [9] occupancy_norm
                                    # [10] trust_mean
                                    # [11] trust_uncertainty
                                    # [12] trust_lcb
                                    # [13] contact_survival_slack
                                    # [14] v2i_coverage_handover_risk
                                    # [15] wait_norm (current instantaneous wait proxy)

    # -------------------------------------------------------------------------
    # 8.1 通信等待时间归一化 (CommWait Normalization)
    # -------------------------------------------------------------------------
    NORM_MAX_COMM_WAIT = 2.0        # 通信等待时间归一化基准 (s) - CommWait normalization baseline
                                    # 影响: 基于 episode 时长上限 (10s) 的 20%，防止饱和
                                    # Impact: Based on 20% of episode duration (10s), prevents saturation

    # =========================================================================
    # 9. 统计与采样参数 (Statistics & Sampling)
    # =========================================================================
    RATE_EMA_ALPHA = 0.05           # EMA平滑系数 - EMA smoothing coefficient
    RATE_RESERVOIR_SIZE = 256       # 速率P95估计样本大小 - Rate P95 estimation sample size
    RATE_MIN_SAMPLES = 32           # P95最小样本数 - P95 minimum samples
    STATS_RESERVOIR_SIZE = 256      # 统计P95估计样本大小 - Stats P95 estimation sample size
    STATS_SEED = 0                  # 统计随机种子 - Stats random seed

    # =========================================================================
    # 10. 辅助方法 (Utility Methods)
    # =========================================================================
    @staticmethod
    def dbm2watt(dbm):
        """
        [工具函数] dBm 转 Watts
        Converts dBm to Watts
        Formula: P(W) = 10 ^ ((P(dBm) - 30) / 10)
        """
        return 10 ** ((dbm - 30) / 10.0)

    @staticmethod
    def watt2dbm(watt):
        """
        [工具函数] Watts 转 dBm
        Converts Watts to dBm
        """
        if watt <= 0:
            return -float('inf')
        return 10.0 * np.log10(watt) + 30.0


# 全局配置实例 (Global config instance)
Cfg = SystemConfig
