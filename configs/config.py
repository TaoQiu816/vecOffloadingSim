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
    NUM_VEHICLES = 30       # 初始车辆数 - Initial number of vehicles (提高密度)
                            # 影响: 网络负载和V2V候选数量，对齐文献参数
                            # Impact: Network load and V2V candidates; aligned with literature

    V2V_TOP_K = 5           # V2V候选数上限 - Max V2V candidates per agent [调优: 11→5]
    CANDIDATE_SORT_BY = "t_finish"  # t_finish | rate | distance (V2V候选排序规则)
    DEBUG_CANDIDATE_SET = False     # 是否打印候选集排序与映射
    V2V_DYNAMIC_K = True    # 是否启用动态邻居数选择 - Enable dynamic V2V candidate filtering
    V2V_TOP_K_MIN = 2       # 动态筛选最少保留的V2V候选数 - Min V2V candidates to keep
    V2V_CANDIDATE_REL_TOL = 0.25  # 相对最优时间允许阈值 - Relative time tolerance (best*(1+rel))
    V2V_CANDIDATE_ABS_TOL = 0.2   # 绝对时间允许阈值(秒) - Absolute time tolerance (seconds)
                            # 影响: 减少V2V冗余，使邻居选择更有意义
                            # Impact: Reduces V2V redundancy, makes neighbor selection more meaningful
    
    MAX_NEIGHBORS = max(0, min(NUM_VEHICLES - 1, V2V_TOP_K))  # 派生值 - Derived value
    MAX_TARGETS = 2 + MAX_NEIGHBORS  # Local + RSU + Neighbors
    
    VEHICLE_ARRIVAL_RATE = 0.0  # 泊松到达率 (veh/s) - Poisson arrival rate
                                # 影响: 动态车辆生成，0表示禁用动态到达
                                # Impact: Dynamic vehicle spawning; 0 disables dynamic arrivals
    
    # -------------------------------------------------------------------------
    # 1.3 车辆移动性参数 (Vehicle Mobility - Truncated Normal Distribution)
    # -------------------------------------------------------------------------
    VEL_MEAN = 15.0         # 速度均值 (m/s) ≈ 54 km/h - Mean velocity (文献二)
                            # 影响: 城市快速路标准速度，增加拓扑动态性
                            # Impact: Urban expressway standard speed, increases topology dynamics

    VEL_STD = 3.0           # 速度标准差 (m/s) - Velocity standard deviation
                            # 影响: 更大的速度异构性，模拟多样化驾驶风格
                            # Impact: Greater velocity heterogeneity, simulates diverse driving styles

    VEL_MIN = 5.0           # 最小速度 (m/s) ≈ 18 km/h - Minimum velocity
                            # 影响: 防止车辆静止，确保拓扑动态性
                            # Impact: Prevents static vehicles, ensures topology dynamics

    VEL_MAX = 25.0          # 最大速度 (m/s) ≈ 90 km/h - Maximum velocity (文献二)
                            # 影响: 城市快速路限速，控制最大拓扑变化速率
                            # Impact: Urban expressway speed limit, controls max topology change rate
    
    MAX_VELOCITY = VEL_MAX  # 最大速度 - Maximum velocity
    
    DT = 0.1                # 仿真时间步长 (s) - Simulation time step (文献二)
                            # 影响: 精度与计算开销权衡，0.1s降低50%开销
                            # Impact: Accuracy vs. computation tradeoff; 0.1s reduces 50% overhead

    MAX_STEPS = 200         # Episode最大步数 - Max steps per episode
                            # 影响: Episode总时长 = MAX_STEPS × DT = 20秒
                            # Impact: Total episode duration = MAX_STEPS × DT = 20s
    TERMINATE_ON_ALL_FINISHED = True  # 是否允许任务全部完成时提前终止

    # -------------------------------------------------------------------------
    # 1.3.1 调度与决策稀疏控制 (Scheduling Controls)
    # -------------------------------------------------------------------------
    MAX_SCHEDULE_PER_STEP = 1  # 每车每步最多调度的READY子任务数 (1=保持现状)
    MAX_INFLIGHT_SUBTASKS_PER_VEHICLE = 0  # 每车在途子任务上限 (0=不限制)
    
    # -------------------------------------------------------------------------
    # 1.4 RSU部署参数 (RSU Deployment)
    # -------------------------------------------------------------------------
    NUM_RSU = 3             # RSU数量 - Number of RSUs
                            # 影响: 覆盖率和卸载能力，3个RSU确保1000m全覆盖+重叠区
                            # Impact: Coverage and offloading capacity; 3 RSUs ensure full coverage
    
    RSU_Y_DIST = 10.0       # RSU离路边距离 (m) - RSU distance from roadside
                            # 影响: V2I路径损耗，10m为标准高度
                            # Impact: V2I path loss; 10m is standard height
    
    RSU_RANGE = 250.0       # RSU覆盖半径 (m) - RSU coverage radius (文献二: 500m直径)
                            # 影响: V2I通信范围，250m半径对齐文献
                            # Impact: V2I communication range; 250m radius aligned with literature
    
    RSU_POS = np.array([500.0, 500.0])  # 默认RSU位置（向后兼容）
                                         # Default RSU position (backward compatibility)

    # =========================================================================
    # 2. 通信参数 (Communication Model)
    # =========================================================================
    # Block Fading 开关：同一时隙内复用 V2V 小尺度衰落（保证口径一致）
    USE_BLOCK_FADING = True
    # CFT缓存哈希严格模式：包含DAG状态/队列摘要，避免错误复用
    CFT_CACHE_STRICT_KEY = True
    # -------------------------------------------------------------------------
    # 2.1 无线物理层参数 (Wireless Physical Layer - C-V2X / DSRC)
    # -------------------------------------------------------------------------
    FC = 5.9e9              # 载波频率 (Hz) - Carrier frequency (5.9 GHz for C-V2X)
                            # 影响: 路径损耗计算基准
                            # Impact: Path loss calculation baseline
    
    C_LIGHT = 3e8           # 光速 (m/s) - Speed of light
    
    BW_V2I = 20e6           # V2I带宽 (Hz) - V2I bandwidth (20 MHz) [文献一标准]
                            # 影响: V2I最大速率上限，20MHz对齐文献实际吞吐
                            # Impact: V2I max rate limit; 20 MHz aligned with literature throughput

    BW_V2V = 10e6           # V2V带宽 (Hz) - V2V bandwidth (10 MHz) [文献二]
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
    TX_POWER_UP_DBM = 20.0      # 上行发射功率 (dBm) ≈ 100mW - Uplink transmit power (文献二)
                                # 影响: V2I SINR，100mW对齐文献
                                # Impact: V2I SINR; 100mW aligned with literature

    TX_POWER_V2V_DBM = 20.0     # V2V发射功率 (dBm) ≈ 100mW - V2V transmit power (文献二)
                                # 影响: V2V SINR和干扰强度
                                # Impact: V2V SINR and interference strength

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
    MIN_VEHICLE_CPU_FREQ = 2.0e9    # 车辆最小CPU频率 (Hz) - Min vehicle CPU freq (2 GHz) [审计调优]
                                    # 影响: 异构性下界，弱车需卸载才能在deadline内完成
                                    # Impact: Heterogeneity lower bound, weak vehicles need offloading

    MAX_VEHICLE_CPU_FREQ = 8.0e9    # 车辆最大CPU频率 (Hz) - Max vehicle CPU freq (8 GHz) [审计调优]
                                    # 影响: 异构性上界，强车可作为V2V卸载目标
                                    # Impact: Heterogeneity upper bound, strong vehicles as V2V targets

    F_RSU = 12.0e9          # RSU CPU频率 (Hz) - RSU CPU frequency (12 GHz) [审计调优]
                            # 影响: RSU算力优势明显但不绝对，与强车形成竞争
                            # Impact: RSU computing advantage significant but not absolute
    
    RSU_NUM_PROCESSORS = 6  # RSU处理器核心数 - RSU processor cores
                             # 影响: 降低并行度，提高调度压力
                             # Impact: Lower parallelism, higher scheduling pressure
    
    K_ENERGY = 1e-28        # 能耗系数 - Energy coefficient (Effective Switched Capacitance)
                            # 公式: Energy = K_ENERGY * f^2 * cycles
                            # 影响: 计算能耗估计，频率平方关系
                            # Impact: Computation energy estimation; quadratic frequency relationship
    
    # -------------------------------------------------------------------------
    # 3.2 队列限制 (Queue Limits - Cycle-Based)
    # -------------------------------------------------------------------------
    VEHICLE_QUEUE_CYCLES_LIMIT = 10.0e9  # 车辆队列上限 (cycles) - Vehicle queue limit [审计调优]
                                        # 影响: 约8个平均任务(1.25G each)，适应新负载
                                        # Impact: ~8 average tasks (1.25G each); adapted to new load

    RSU_QUEUE_CYCLES_LIMIT = 150.0e9    # RSU队列上限 (cycles) - RSU queue limit
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

    MIN_NODES = 18          # DAG最小节点数 - Min DAG nodes [训练默认提高密度]
                            # 影响: 提升决策密度，减少空转步
                            # Impact: Increases decision density, reduces idle steps

    MAX_NODES = 24          # DAG最大节点数 - Max DAG nodes [训练默认提高密度]
                            # 影响: 提升决策密度，减少空转步
                            # Impact: Increases decision density, reduces idle steps
    
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
    MIN_COMP = 5.0e8        # 子任务最小计算量 (cycles) - Min subtask computation (0.5 Gcycles) [优化: 0.8→0.5]
                            # 影响: 小任务强车Local 62ms，保持卸载必要性
                            # Impact: Small tasks on strong vehicles 62ms; maintains offloading relevance

    MAX_COMP = 3.5e9        # 子任务最大计算量 (cycles) - Max subtask computation (3.5 Gcycles) [优化: 2.5→3.5]
                            # 影响: 大任务必须卸载，增加决策复杂度
                            # Impact: Large tasks require offloading; increases decision complexity

    MIN_DATA = 2.0e5        # 子任务最小数据量 (bits) - Min subtask data (25 KB) [审计调优]
                            # 影响: 传输时间适中，确保功率梯度可见
                            # Impact: Moderate transmission time; ensures power gradient visibility

    MAX_DATA = 1.0e6        # 子任务最大数据量 (bits) - Max subtask data (125 KB) [审计调优]
                            # 影响: V2I传输约0.03s @33Mbps，计算仍占主导
                            # Impact: V2I transmission ~0.03s @33Mbps; computation still dominant

    MIN_EDGE_DATA = 1.0e5   # DAG边最小数据量 (bits) - Min edge data (12.5 KB) [审计调优]
                            # 影响: 依赖数据传输开销适中
                            # Impact: Moderate dependency transmission overhead

    MAX_EDGE_DATA = 5.0e5   # DAG边最大数据量 (bits) - Max edge data (62.5 KB) [审计调优]
                            # 影响: 依赖数据传输开销适中
                            # Impact: Moderate dependency transmission overhead
    
    MEAN_COMP_LOAD = (8.0e8 + 2.5e9) / 2  # 平均计算负载 (cycles) - Average computation load
                                          # 动态计算：(MIN_COMP + MAX_COMP) / 2 = 1.65e9
                                          # Dynamically computed: (MIN_COMP + MAX_COMP) / 2 = 1.65e9
    AVG_COMP = MEAN_COMP_LOAD             # 同上 - Same as above
    
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
    DEADLINE_MODE = 'TOTAL_MEDIAN'      # 选择deadline计算模式:
                                        # - 'CRITICAL_PATH': CP_total / f_median (关键路径，推荐)
                                        # - 'TOTAL_MEDIAN': total_comp / f_median (总量，向后兼容)
                                        # - 'TOTAL_LOCAL': total_comp / f_local (本地算力)
                                        # - 'FIXED_RANGE': 直接从固定范围随机 (秒)

    # 基于计算量的deadline (使用γ因子)
    # 公式: deadline = max(γ × T_base + slack, (1+eps) × LB0)
    # 其中 T_base = CP_total / f_ref, LB0 = CP_total / f_max
    DEADLINE_TIGHTENING_MIN = 0.7       # γ最小值 [再次下调: 提高deadline压力]
                                        # 当前f_max/f_median=2.4，需要gamma<2.4才有卸载压力
    DEADLINE_TIGHTENING_MAX = 1.0       # γ最大值 [再次下调: 提高deadline压力]
                                        # 目标: 基线有明显超时，便于体现训练优势
    
    DEADLINE_LB_EPS = 0.02              # 物理下界裕量 eps
                                        # deadline ≥ (1+eps) × LB0 保证不先天不可行
                                        # 推荐范围: 0.05~0.1
    
    # 模式FIXED_RANGE: 固定范围的deadline (秒)
    DEADLINE_FIXED_MIN = 2.0            # 最小deadline (秒)
    DEADLINE_FIXED_MAX = 5.0            # 最大deadline (秒)
    
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
    
    NORM_MAX_COMP = 4.0e9       # 计算量归一化基准 (cycles) - Computation normalization baseline [优化: 2.0→4.0]
                                # 影响: 适应新的MAX_COMP=3.5e9
                                # Impact: Adapts to new MAX_COMP=3.5e9
    
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
    # -------------------------------------------------------------------------
    # 6.7 新奖励方案 (Reward Scheme Switch & PBRS Parameters)
    # -------------------------------------------------------------------------
    REWARD_SCHEME = "PBRS_KP"       # 奖励方案: "LEGACY_CFT" (旧) / "PBRS_KP" / "PBRS_KP_V2"
    REWARD_ALPHA = 1.5              # 基础奖励系数 alpha
    REWARD_BETA = 0.1               # PBRS 系数 beta [审计调优: 降低shape噪声]
    REWARD_GAMMA = 0.99             # PBRS 折扣 gamma（应与训练端 TC.GAMMA 保持一致）
    T_REF = 0.7                     # 时间归一化参考尺度 (s) [审计标定: 取|Δt| p90≈0.69s以降低剪裁]
    PHI_CLIP = 5.0                  # ϕ 裁剪上界（负值幅度上限）
    SHAPE_CLIP = 10.0               # 潜势差分裁剪上界
    R_CLIP = 40.0                   # 总奖励裁剪上界（绝对值）
    EPS_RATE = 1e-9                 # 速率下界，防止除0
    ILLEGAL_PENALTY = -2.0          # 非法动作额外惩罚

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
    LAT_ALPHA = 1.5                 # 时延优势奖励系数 (tanh)
    TIMEOUT_L1 = 1.0                # 超时惩罚第一段系数
    TIMEOUT_L2 = 1.0                # 超时惩罚二段系数 (二次)
    TIMEOUT_O0 = 0.2                # 超时分段阈值 (ratio)
    TIMEOUT_K = 3.0                 # 超时惩罚tanh陡峭度
    ENERGY_LAMBDA = 0.02            # V2通信能耗权重
    POWER_LAMBDA = 0.01             # V2功率正则权重
    

    # =========================================================================
    # 7. 调试与日志参数 (Debug & Logging)
    # =========================================================================
    DEBUG_ASSERT_ILLEGAL_ACTION = False     # 非法动作断言 - Illegal action assertion
                                            # 影响: True时illegal_action直接断言崩溃
                                            # Impact: True causes assertion crash on illegal action
    
    DEBUG_ASSERT_METRICS = False            # 指标范围断言 - Metrics range assertion
                                            # 影响: True时对成功率/决策分布做范围断言
                                            # Impact: True asserts success rate/decision distribution ranges

    DEBUG_REWARD_ASSERTS = False            # 奖励/速率快照强一致性断言
    DEBUG_PBRS_AUDIT = False                # PBRS一致性审计/打点开关
    DEBUG_PHI_MONO_PROB = 0.1               # Phi单调性抽样概率
    
    EPISODE_JSONL_STDOUT = True             # Episode JSONL输出 - Episode JSONL output
                                            # 影响: 是否在stdout打印每个episode的JSONL
                                            # Impact: Whether to print episode JSONL to stdout

    # =========================================================================
    # 8. 模型结构参数 (Model Architecture)
    # =========================================================================
    RESOURCE_RAW_DIM = 14           # 资源原始特征维度 - Resource raw feature dimension (CommWait 4维已移除)
                                    # 影响: 14原始特征 + 4 CommWait特征 (total/edge × v2i/v2v)
                                    # Impact: 14 raw features + 4 CommWait features (total/edge × v2i/v2v)

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
