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
                            # 影响: 仿真区域大小，影响车辆密度和RSU覆盖范围
                            # Impact: Simulation area size, affects vehicle density and RSU coverage
    
    NUM_LANES = 2           # 车道数量 - Number of lanes
                            # 影响: 车辆横向分布，减少换道干扰
                            # Impact: Vehicle lateral distribution, reduces lane-change interference
    
    LANE_WIDTH = 3.5        # 车道宽度 (m) - Lane width (standard)
                            # 影响: 标准车道宽度，影响V2V距离计算
                            # Impact: Standard lane width, affects V2V distance calculation
    
    # -------------------------------------------------------------------------
    # 1.2 车辆参数 (Vehicle Parameters)
    # -------------------------------------------------------------------------
    NUM_VEHICLES = 12       # 初始车辆数 - Initial number of vehicles
                            # 影响: 网络负载和V2V候选数量，降低以减少动作空间不平衡
                            # Impact: Network load and V2V candidates; reduced to balance action space
    
    V2V_TOP_K = 11          # V2V候选数上限 - Max V2V candidates per agent
                            # 影响: 限制每个Agent的V2V目标数，控制动作空间大小
                            # Impact: Limits V2V targets per agent, controls action space size
    
    MAX_NEIGHBORS = max(0, min(NUM_VEHICLES - 1, V2V_TOP_K))  # 派生值 - Derived value
    MAX_TARGETS = 2 + MAX_NEIGHBORS  # Local + RSU + Neighbors
    
    VEHICLE_ARRIVAL_RATE = 0.2  # 泊松到达率 (veh/s) - Poisson arrival rate
                                # 影响: 动态车辆生成，0表示禁用动态到达
                                # Impact: Dynamic vehicle spawning; 0 disables dynamic arrivals
    
    # -------------------------------------------------------------------------
    # 1.3 车辆移动性参数 (Vehicle Mobility - Truncated Normal Distribution)
    # -------------------------------------------------------------------------
    VEL_MEAN = 13.8         # 速度均值 (m/s) ≈ 50 km/h - Mean velocity
                            # 影响: 城市快速路标准速度，增加拓扑动态性
                            # Impact: Urban expressway standard speed, increases topology dynamics
    
    VEL_STD = 3.0           # 速度标准差 (m/s) - Velocity standard deviation
                            # 影响: 更大的速度异构性，模拟多样化驾驶风格
                            # Impact: Greater velocity heterogeneity, simulates diverse driving styles
    
    VEL_MIN = 5.0           # 最小速度 (m/s) ≈ 18 km/h - Minimum velocity
                            # 影响: 防止车辆静止，确保拓扑动态性
                            # Impact: Prevents static vehicles, ensures topology dynamics
    
    VEL_MAX = 20.0          # 最大速度 (m/s) ≈ 72 km/h - Maximum velocity
                            # 影响: 城市快速路限速，控制最大拓扑变化速率
                            # Impact: Urban expressway speed limit, controls max topology change rate
    
    MAX_VELOCITY = VEL_MAX  # 最大速度 - Maximum velocity
    
    DT = 0.05               # 仿真时间步长 (s) - Simulation time step
                            # 影响: 精度与计算开销权衡，0.05s保证 v_max * DT << R_rsu
                            # Impact: Accuracy vs. computation tradeoff; 0.05s ensures v_max * DT << R_rsu
    
    MAX_STEPS = 400         # Episode最大步数 - Max steps per episode (延长至20s)
                            # 影响: Episode总时长 = MAX_STEPS * DT = 20秒 (适应处理器共享降速)
                            # Impact: Total episode duration = MAX_STEPS * DT = 20s (accommodates processor sharing)
    
    # -------------------------------------------------------------------------
    # 1.4 RSU部署参数 (RSU Deployment)
    # -------------------------------------------------------------------------
    NUM_RSU = 3             # RSU数量 - Number of RSUs
                            # 影响: 覆盖率和卸载能力，3个RSU确保1000m全覆盖+重叠区
                            # Impact: Coverage and offloading capacity; 3 RSUs ensure full coverage
    
    RSU_Y_DIST = 10.0       # RSU离路边距离 (m) - RSU distance from roadside
                            # 影响: V2I路径损耗，10m为标准高度
                            # Impact: V2I path loss; 10m is standard height
    
    RSU_RANGE = 400.0       # RSU覆盖半径 (m) - RSU coverage radius
                            # 影响: V2I通信范围，400m确保全覆盖
                            # Impact: V2I communication range; 400m ensures full coverage
    
    RSU_POS = np.array([500.0, 500.0])  # 默认RSU位置（向后兼容）
                                         # Default RSU position (backward compatibility)

    # =========================================================================
    # 2. 通信参数 (Communication Model)
    # =========================================================================
    # -------------------------------------------------------------------------
    # 2.1 无线物理层参数 (Wireless Physical Layer - C-V2X / DSRC)
    # -------------------------------------------------------------------------
    FC = 5.9e9              # 载波频率 (Hz) - Carrier frequency (5.9 GHz for C-V2X)
                            # 影响: 路径损耗计算基准
                            # Impact: Path loss calculation baseline
    
    C_LIGHT = 3e8           # 光速 (m/s) - Speed of light
    
    BW_V2I = 20e6           # V2I带宽 (Hz) - V2I bandwidth (20 MHz)
                            # 影响: V2I最大速率上限，20MHz为LTE标准带宽
                            # Impact: V2I max rate limit; 20 MHz is standard LTE bandwidth
    
    BW_V2V = 10e6           # V2V带宽 (Hz) - V2V bandwidth (10 MHz)
                            # 影响: V2V最大速率上限，低于V2I以模拟边链劣势
                            # Impact: V2V max rate limit; lower than V2I to model sidelink disadvantage
    
    # -------------------------------------------------------------------------
    # 2.2 噪声参数 (Noise Parameters)
    # -------------------------------------------------------------------------
    NOISE_POWER_DENSITY_DBM = -174  # 热噪声功率谱密度 (dBm/Hz) - Thermal noise PSD
    NOISE_FIGURE = 9                # 噪声系数 (dB) - Noise figure
    NOISE_POWER_DBM = -95.0         # 热噪声功率 (dBm) - Thermal noise power
                                    # 影响: SNR计算基准，影响速率估计
                                    # Impact: SNR calculation baseline, affects rate estimation
    
    # -------------------------------------------------------------------------
    # 2.3 发射功率参数 (Transmit Power - FCC Compliant)
    # -------------------------------------------------------------------------
    TX_POWER_UP_DBM = 23.0      # 上行发射功率 (dBm) ≈ 200mW - Uplink transmit power
                                # 影响: V2I SINR，200mW为FCC限制
                                # Impact: V2I SINR; 200mW is FCC limit
    
    TX_POWER_V2V_DBM = 23.0     # V2V发射功率 (dBm) ≈ 200mW - V2V transmit power
                                # 影响: V2V SINR和干扰强度
                                # Impact: V2V SINR and interference strength
    
    TX_POWER_MIN_DBM = 20.0     # 功控下限 (dBm) - Power control lower bound
                                # 影响: 能耗优化范围下限
                                # Impact: Energy optimization lower bound
    
    TX_POWER_MAX_DBM = 23.0     # 功控上限 (dBm) - Power control upper bound
                                # 影响: 能耗优化范围上限
                                # Impact: Energy optimization upper bound
    
    NUM_POWER_LEVELS = 4        # 功率离散等级数 - Number of discrete power levels
                                # 影响: 动作空间维度，将[MIN,MAX]均匀离散化
                                # Impact: Action space dimension; uniformly discretizes [MIN,MAX]
    
    # -------------------------------------------------------------------------
    # 2.4 路径损耗模型 (Path Loss Model - Log-Distance)
    # 公式: PL(d) = PL_ALPHA + 10 * PL_BETA * log10(d/d_0)
    # -------------------------------------------------------------------------
    PL_ALPHA_V2I = 28.0     # V2I参考路损 (dB) - V2I reference path loss
                            # 影响: V2I链路预算基准
                            # Impact: V2I link budget baseline
    
    PL_BETA_V2I = 2.5       # V2I路损指数 - V2I path loss exponent (LOS environment)
                            # 影响: V2I衰减速率，2.5为LOS标准
                            # Impact: V2I attenuation rate; 2.5 is standard for LOS
    
    PL_ALPHA_V2V = 28.0     # V2V参考路损 (dB) - V2V reference path loss
    PL_BETA_V2V = 3.5       # V2V路损指数 - V2V path loss exponent (NLOS, high attenuation)
                            # 影响: V2V衰减速率，3.5模拟NLOS遮挡/干扰
                            # Impact: V2V attenuation rate; 3.5 models NLOS obstruction/interference
    
    V2V_INTERFERENCE_DBM = -95.0  # V2V背景干扰 (dBm) - V2V background interference
                                  # 影响: V2V SINR，降低V2V链路质量
                                  # Impact: V2V SINR; degrades V2V link quality
    
    V2V_RANGE = 300.0       # V2V通信半径 (m) - V2V communication range
                            # 影响: 邻居发现范围，DVTP等文献常用值
                            # Impact: Neighbor discovery range; common value in DVTP literature
    
    # -------------------------------------------------------------------------
    # 2.5 速率估计参数 (Rate Estimation - Shannon Capacity Approximation)
    # -------------------------------------------------------------------------
    SNR_MIN_DB = -10.0      # 最低可检测SNR (dB) - Minimum detectable SNR
    SNR_MAX_DB = 20.0       # 饱和SNR (dB) - Saturation SNR
    SNR_OFFSET_DB = 10.0    # SNR偏移量 (dB) - SNR offset for positive values
    RICIAN_K_DB = 6.0
    BETA_0_DB = -30

    # =========================================================================
    # 3. 计算资源参数 (Computation Model)
    # =========================================================================
    # -------------------------------------------------------------------------
    # 3.1 CPU频率设定 (CPU Frequency - Heterogeneous Configuration)
    # -------------------------------------------------------------------------
    MIN_VEHICLE_CPU_FREQ = 1.0e9    # 车辆最小CPU频率 (Hz) - Min vehicle CPU freq (1 GHz)
                                    # 影响: 异构性下界，最弱车辆算力
                                    # Impact: Heterogeneity lower bound, weakest vehicle computing power
    
    MAX_VEHICLE_CPU_FREQ = 3.0e9    # 车辆最大CPU频率 (Hz) - Max vehicle CPU freq (3 GHz)
                                    # 影响: 异构性上界，最强车辆算力
                                    # Impact: Heterogeneity upper bound, strongest vehicle computing power
    
    F_RSU = 12.0e9          # RSU CPU频率 (Hz) - RSU CPU frequency (12 GHz)
                            # 影响: RSU算力优势，约为车辆平均频率的6倍，显著强于车辆
                            # Impact: RSU computing advantage, ~6x vehicle average frequency, significantly stronger
    
    RSU_NUM_PROCESSORS = 4  # RSU处理器核心数 - RSU processor cores
                            # 影响: RSU并行处理能力，4核可同时处理4个任务
                            # Impact: RSU parallel processing capacity; 4 cores can handle 4 tasks simultaneously
    
    K_ENERGY = 1e-28        # 能耗系数 - Energy coefficient (Effective Switched Capacitance)
                            # 公式: Energy = K_ENERGY * f^2 * cycles
                            # 影响: 计算能耗估计，频率平方关系
                            # Impact: Computation energy estimation; quadratic frequency relationship
    
    # -------------------------------------------------------------------------
    # 3.2 队列限制 (Queue Limits - Cycle-Based)
    # -------------------------------------------------------------------------
    VEHICLE_QUEUE_CYCLES_LIMIT = 5.0e9  # 车辆队列上限 (cycles) - Vehicle queue limit
                                        # 影响: 约3个平均任务，严格限制防止车辆队列过载
                                        # Impact: ~3 average tasks; strict limit prevents vehicle queue overload
    
    RSU_QUEUE_CYCLES_LIMIT = 50.0e9     # RSU队列上限 (cycles) - RSU queue limit
                                        # 影响: 约33个平均任务，RSU高承载能力
                                        # Impact: ~33 average tasks; RSU high capacity

    # =========================================================================
    # 4. DAG任务生成参数 (Task Generation)
    # =========================================================================
    # -------------------------------------------------------------------------
    # 4.1 DAG结构参数 (DAG Structure)
    # -------------------------------------------------------------------------
    MIN_NODES = 4           # DAG最小节点数 - Min DAG nodes
                            # 影响: 降低依赖链复杂度，从8降到4
                            # Impact: Reduces dependency complexity; decreased from 8 to 4
    
    MAX_NODES = 6           # DAG最大节点数 - Max DAG nodes
                            # 影响: 极大降低依赖链深度，从12降到6
                            # Impact: Greatly reduces dependency depth; decreased from 12 to 6
    
    DAG_FAT = 0.5           # DAG宽度参数 - DAG width parameter
                            # 影响: 控制并行度，0.5为中等偏低宽度，减少并行任务数
                            # Impact: Controls parallelism; 0.5 is medium-low width, reduces parallel tasks
    
    DAG_DENSITY = 0.2       # DAG连接密度 - DAG edge density
                            # 影响: 低依赖复杂度，简化调度决策
                            # Impact: Low dependency complexity, simplifies scheduling decisions
    
    DAG_REGULAR = 0.5       # DAG规则性 - DAG regularity
                            # 影响: 结构规则性，0.5为半规则
                            # Impact: Structure regularity; 0.5 is semi-regular
    
    DAG_CCR = 0.2           # 通信计算比 - Communication-to-Computation Ratio
                            # 影响: 结构生成参数，影响边数据量
                            # Impact: Structure generation parameter, affects edge data volume
    
    # -------------------------------------------------------------------------
    # 4.2 任务负载参数 (Task Load Parameters)
    # -------------------------------------------------------------------------
    MIN_COMP = 0.3e9        # 子任务最小计算量 (cycles) - Min subtask computation (0.3 Gcycles) [降低62%]
                            # 影响: 本地执行约0.10s @3GHz，轻量负载
                            # Impact: Local execution ~0.10s @3GHz; light load
    
    MAX_COMP = 1.2e9        # 子任务最大计算量 (cycles) - Max subtask computation (1.2 Gcycles) [降低52%]
                            # 影响: 本地执行约0.40s @3GHz，中等负载
                            # Impact: Local execution ~0.40s @3GHz; medium load
    
    MIN_DATA = 1.0e6        # 子任务最小数据量 (bits) - Min subtask data (1 Mbit)
                            # 影响: V2I传输约0.017s @60Mbps
                            # Impact: V2I transmission ~0.017s @60Mbps
    
    MAX_DATA = 4.0e6        # 子任务最大数据量 (bits) - Max subtask data (4 Mbit)
                            # 影响: V2I传输约0.067s @60Mbps
                            # Impact: V2I transmission ~0.067s @60Mbps
    
    MIN_EDGE_DATA = 0.2e6   # DAG边最小数据量 (bits) - Min edge data (200 Kbit)
                            # 影响: 依赖传输开销约0.004s
                            # Impact: Dependency transmission overhead ~0.004s
    
    MAX_EDGE_DATA = 0.6e6   # DAG边最大数据量 (bits) - Max edge data (600 Kbit)
                            # 影响: 依赖传输开销约0.012s
                            # Impact: Dependency transmission overhead ~0.012s
    
    MEAN_COMP_LOAD = (0.8e9 + 2.5e9) / 2  # 平均计算负载 (cycles) - Average computation load
                                         # 动态计算：(MIN_COMP + MAX_COMP) / 2 = 1.65e9
                                         # Dynamically computed: (MIN_COMP + MAX_COMP) / 2 = 1.65e9
    AVG_COMP = MEAN_COMP_LOAD            # 同上 - Same as above
    
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
    DEADLINE_TIGHTENING_MIN = 1.8       # γ最小值（无量纲系数！）- γ minimum value (dimensionless coefficient!)
                                        # 影响: deadline = γ × (关键路径/本地CPU)，2.0表示本地时间的200%（Warmup宽松，先保证可训练）
                                        # Impact: deadline = γ × (critical_path/local_CPU), 2.0 means 200% of local time (Warmup relaxed, ensures trainability)
    
    DEADLINE_TIGHTENING_MAX = 2.2       # γ最大值（无量纲系数！）- γ maximum value (dimensionless coefficient!)
                                        # 影响: deadline = γ × (关键路径/本地CPU)，2.5表示本地时间的250%（Warmup阶段，适应处理器共享降速）
                                        # Impact: deadline = γ × (critical_path/local_CPU), 2.5 means 250% of local time (Warmup phase, accommodates processor sharing slowdown)
    
    DEADLINE_SLACK_SECONDS = 0.0        # 额外松弛时间 (s) - Additional slack time
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
    
    NORM_MAX_COMP = 2.0e9       # 计算量归一化基准 (cycles) - Computation normalization baseline
                                # 影响: 适应1e9级别计算量
                                # Impact: Adapts to 1e9-level computation
    
    NORM_MAX_DATA = 5.0e6       # 数据量归一化基准 (bits) - Data normalization baseline
                                # 影响: 适应约4e6 bit数据量
                                # Impact: Adapts to ~4e6 bit data volume
    
    NORM_MAX_RATE_V2I = 50e6    # V2I速率归一化基准 (bps) - V2I rate normalization baseline
    NORM_MAX_RATE_V2V = 20e6    # V2V速率归一化基准 (bps) - V2V rate normalization baseline
    
    # 动态计算的等待时间归一化常量 (Dynamically computed wait time normalization)
    # 基于队列上限和处理能力计算最大等待时间
    _RSU_MAX_WAIT = RSU_QUEUE_CYCLES_LIMIT / F_RSU
    _VEHICLE_MAX_WAIT = VEHICLE_QUEUE_CYCLES_LIMIT / MIN_VEHICLE_CPU_FREQ
    NORM_MAX_WAIT_TIME = max(_RSU_MAX_WAIT, _VEHICLE_MAX_WAIT) * 1.2

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
    

    # =========================================================================
    # 7. 调试与日志参数 (Debug & Logging)
    # =========================================================================
    DEBUG_ASSERT_ILLEGAL_ACTION = False     # 非法动作断言 - Illegal action assertion
                                            # 影响: True时illegal_action直接断言崩溃
                                            # Impact: True causes assertion crash on illegal action
    
    DEBUG_ASSERT_METRICS = False            # 指标范围断言 - Metrics range assertion
                                            # 影响: True时对成功率/决策分布做范围断言
                                            # Impact: True asserts success rate/decision distribution ranges
    
    EPISODE_JSONL_STDOUT = True             # Episode JSONL输出 - Episode JSONL output
                                            # 影响: 是否在stdout打印每个episode的JSONL
                                            # Impact: Whether to print episode JSONL to stdout

    # =========================================================================
    # 8. 模型结构参数 (Model Architecture)
    # =========================================================================
    RESOURCE_RAW_DIM = 14           # 资源原始特征维度 - Resource raw feature dimension
                                    # 影响: 11原始特征 + 3时间预估特征
                                    # Impact: 11 raw features + 3 time estimation features

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
