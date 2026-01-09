class TrainConfig:
    """
    [训练超参数配置类] TrainConfig
    Training Hyperparameter Configuration

    作用 (Purpose):
    定义模型结构、优化器参数、PPO算法参数及训练流程控制。
    Defines model architecture, optimizer parameters, PPO algorithm settings, and training control.

    参考文献 (References):
    - PPO: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    - GAE: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
    - Transformer: Vaswani et al., "Attention Is All You Need" (2017)
    """

    # =========================================================================
    # 1. 网络结构参数 (Model Architecture)
    # =========================================================================
    """
    输入特征维度 (Input Feature Dimensions)
    - 必须与 data_utils.py 中的 process_env_obs 输出一致
    - Must match process_env_obs output in data_utils.py
    """
    TASK_INPUT_DIM = 7      # 任务特征维度 - Task feature dimension
                            # 特征: [comp, data, status, in_degree, t_rem, overdue, urgency]
                            # Features: [comp, data, status, in_degree, t_rem, overdue, urgency]
                            # 影响: 输入层大小，必须与实际观测维度匹配
                            # Impact: Input layer size; must match actual observation dimension
    
    VEH_INPUT_DIM = 7       # 车辆特征维度 - Vehicle feature dimension
                            # 特征: [vx, vy, queue, cap, pos_x, pos_y, ...]
                            # Features: [vx, vy, queue, cap, pos_x, pos_y, ...]
                            # 影响: 车辆编码器输入大小
                            # Impact: Vehicle encoder input size
    
    EDGE_INPUT_DIM = 2      # DAG边特征维度 - DAG edge feature dimension
                            # 特征: [edge_data, edge_type]
                            # Features: [edge_data, edge_type]
                            # 影响: 边特征编码器输入大小
                            # Impact: Edge feature encoder input size
    
    RSU_INPUT_DIM = 1       # RSU特征维度 - RSU feature dimension
                            # 特征: [load]
                            # Features: [load]
                            # 影响: RSU编码器输入大小
                            # Impact: RSU encoder input size

    EMBED_DIM = 128         # 嵌入维度 / 隐藏层维度 - Embedding dimension / Hidden dimension
                            # 影响: Transformer和GNN的特征维度，更大的维度增加模型容量但也增加计算开销
                            # Impact: Feature dimension for Transformer and GNN; larger increases capacity but also cost
                            # 推荐范围: 64-256 (64 for fast prototyping, 256 for production)
                            # Recommended range: 64-256
    USE_SUBTASK_COND_CRITIC = True  # Critic是否使用当前子任务上下文
    USE_SIMPLIFIED_CRITIC = True    # 是否使用简化版Critic Head
    USE_NO_READY_EMBEDDING = True   # subtask_index<0 时使用专用嵌入（避免误用节点0）
    COMMWAIT_DIRECT_TO_CRITIC = False  # 是否将CommWait特征直连拼接到Critic输入

    NUM_HEADS = 4           # 注意力头数 - Number of attention heads (Multi-Head Attention)
                            # 影响: 更多头数可以捕获更多样化的依赖关系，必须被EMBED_DIM整除
                            # Impact: More heads capture diverse dependencies; must divide EMBED_DIM evenly
                            # 推荐范围: 4-8 (for EMBED_DIM=128)
                            # Recommended range: 4-8

    NUM_LAYERS = 2          # Transformer层数 - Number of Transformer encoder layers [文献一]
                            # 影响: 更深的网络可以学习更复杂的特征，但可能过拟合或训练困难
                            # Impact: Deeper networks learn complex features but may overfit or be hard to train
                            # 推荐范围: 2-6 (2 for simple tasks, 6 for complex dependencies)
                            # Recommended range: 2-6

    D_FF = 512              # 前馈层维度 - Feed-forward layer dimension
                            # 影响: Transformer FFN的隐藏层大小，通常是EMBED_DIM的4倍
                            # Impact: Transformer FFN hidden size; typically 4x EMBED_DIM
                            # 推荐范围: 256-1024 (512 for EMBED_DIM=128)
                            # Recommended range: 256-1024

    DROPOUT = 0.1           # Dropout率 - Dropout rate for regularization
                            # 影响: 正则化强度，防止过拟合
                            #       - 过大: 损失模型表达能力
                            #       - 过小: 可能过拟合
                            # Impact: Regularization strength; prevents overfitting
                            #       - Too large: Loses model expressiveness
                            #       - Too small: May overfit
                            # 推荐范围: 0.05-0.2 (0.1 is standard)
                            # Recommended range: 0.05-0.2

    # =========================================================================
    # 2. 优化器参数 (Optimizer Parameters)
    # =========================================================================
    LR_ACTOR = 1e-4         # Actor学习率 - Actor learning rate [调优: 从3e-4降至1e-4稳定梯度]
                            # 影响: 控制策略网络的更新速度
                            #       - 过大: 训练不稳定，策略震荡
                            #       - 过小: 收敛慢，需要更多训练时间
                            # Impact: Controls policy network update speed
                            #       - Too large: Unstable training, policy oscillation
                            #       - Too small: Slow convergence, needs more training time
                            # 推荐范围: 1e-4 ~ 1e-3 (稳定性优先)
                            # Recommended range: 1e-4 ~ 1e-3 (stability first)

    LR_CRITIC = 3e-4        # Critic学习率 - Critic learning rate [审慎: 保持与Actor一致]
                            # 影响: 控制价值网络的更新速度，与Actor保持一致
                            # Impact: Controls value network update speed; consistent with Actor
                            # 推荐范围: 1e-4 ~ 1e-3
                            # Recommended range: 1e-4 ~ 1e-3

    USE_LR_DECAY = True     # 是否启用学习率衰减 - Enable learning rate decay
                            # 影响: 训练后期降低学习率有助于收敛和稳定性
                            # Impact: Reducing LR in late training aids convergence and stability
    
    LR_DECAY_STEPS = 100    # 学习率衰减间隔 (Episodes) - Learning rate decay interval [适应短期训练]
                            # 影响: 每N个Episodes衰减一次，短期训练使用较快衰减
                            # Impact: Decay every N episodes; faster decay for short-term training
                            # 推荐范围: 50-100 (短期), 100-200 (长期)
                            # Recommended range: 50-100 (short-term), 100-200 (long-term)
    
    LR_DECAY_RATE = 0.92    # 学习率衰减率 - Learning rate decay rate
                            # 影响: 指数衰减系数 (new_lr = lr * decay_rate)
                            #       - 0.9: 快速衰减，适合短期训练
                            #       - 0.95: 慢速衰减，适合长期训练
                            # Impact: Exponential decay coefficient (new_lr = lr * decay_rate)
                            #       - 0.9: Fast decay, suits short-term training
                            #       - 0.95: Slow decay, suits long-term training
                            # 推荐范围: 0.90-0.95
                            # Recommended range: 0.90-0.95

    MAX_GRAD_NORM = 2.0     # 梯度裁剪阈值 - Gradient clipping threshold [审慎调优: 0.5→2.0]
                            # 影响: 防止梯度爆炸，保证训练稳定性
                            #       配合Value Loss归一化，温和放宽即可
                            #       - 过大: 可能梯度爆炸
                            # Impact: Prevents gradient explosion, ensures training stability
                            #       - Too small: Limits learning speed (current: 100% clipped)
                            #       - Too large: May still explode
                            # 诊断: 当前梯度均值90，全部被裁剪，提高阈值配合降低LR
                            # Diagnostic: Current grad norm avg 90, all clipped, raise threshold with lower LR

    # =========================================================================
    # 3. PPO 算法参数 (PPO Algorithm Parameters)
    # =========================================================================
    GAMMA = 0.99            # 折扣因子 - Discount factor for future rewards
                            # 影响: 控制对未来奖励的重视程度
                            #       - 接近1: 重视长期回报，适合长期规划任务
                            #       - 接近0: 重视即时回报，适合短期决策任务
                            # Impact: Controls importance of future rewards
                            #       - Close to 1: Values long-term returns, suits long-term planning
                            #       - Close to 0: Values immediate returns, suits short-term decisions
                            # 推荐范围: 0.90-0.99 (0.98 for longer horizon planning)
                            # Recommended range: 0.90-0.99 (0.98 for longer horizon planning)
    
    GAE_LAMBDA = 0.95       # GAE平滑因子 - Generalized Advantage Estimation lambda
                            # 影响: 权衡优势估计的偏差-方差
                            #       - 接近1: 低偏差，高方差（使用完整回报）
                            #       - 接近0: 高偏差，低方差（仅使用TD残差）
                            # Impact: Balances bias-variance in advantage estimation
                            #       - Close to 1: Low bias, high variance (uses full returns)
                            #       - Close to 0: High bias, low variance (uses only TD residuals)
                            # 推荐范围: 0.90-0.99 (0.95 balances immediate and long-term)
                            # Recommended range: 0.90-0.99

    CLIP_PARAM = 0.2        # PPO裁剪阈值 epsilon - PPO clipping threshold (PPO_CLIP = 0.2)
                            # 影响: 限制策略更新幅度，防止破坏性更新
                            #       - 过小: 更新保守，学习慢
                            #       - 过大: 更新激进，可能不稳定
                            # Impact: Limits policy update magnitude, prevents destructive updates
                            #       - Too small: Conservative updates, slow learning
                            #       - Too large: Aggressive updates, may destabilize
                            # 推荐范围: 0.1-0.3 (0.2 is PPO default)
                            # Recommended range: 0.1-0.3

    PPO_EPOCH = 5           # 每次采样的更新轮数 - Number of optimization epochs per batch (NUM_EPOCHS = 5)
                            # 影响: 重复利用经验，提高样本效率
                            #       - 过少: 样本利用不足
                            #       - 过多: 可能过拟合采样数据
                            # Impact: Reuses experience for sample efficiency
                            #       - Too few: Underutilizes samples
                            #       - Too many: May overfit sampled data
                            # 推荐范围: 3-10 (5 for better sample utilization)
                            # Recommended range: 3-10 (5 for better sample utilization)
    
    MINI_BATCH_SIZE = 256   # 小批次大小 - Mini-batch size for SGD updates [增至256提高稳定性]
                            # 影响: 梯度估计的方差和计算效率
                            #       - 较大: 梯度稳定，但内存占用高
                            #       - 较小: 增加随机性，适应动态图结构
                            # Impact: Variance of gradient estimation and computational efficiency
                            #       - Larger: Stable gradients, but higher memory usage
                            #       - Smaller: Increases stochasticity, adapts to dynamic graphs
                            # 推荐范围: 64-256 (256 for better stability)
                            # Recommended range: 64-256

    ENTROPY_COEF = 0.003    # 熵正则化系数 - Entropy coefficient for exploration [审慎调优: 0.01→0.003]
                            # 影响: 增加动作探索性，应对动态环境
                            #       - 过大: 策略过于随机，难以收敛（当前问题）
                            #       - 过小: 策略过早收敛到局部最优
                            # Impact: Increases action exploration for dynamic environments
                            #       - Too large: Policy too random, hard to converge (current issue)
                            #       - Too small: Policy converges prematurely to local optimum
                            # 诊断: 当前entropy持续上升至最大值96%，需要降低
                            # Diagnostic: Current entropy rising to 96% of max, needs reduction

    VF_COEF = 0.5           # 价值函数损失系数 - Value function loss coefficient
                            # 影响: 平衡Actor-Critic训练，控制值函数更新权重
                            #       已通过Value Loss归一化解决梯度主导问题
                            # Impact: Balances Actor-Critic training, controls value function update weight
                            # 推荐范围: 0.5-1.0 (标准值，配合归一化的Value Loss)
                            # Recommended range: 0.5-1.0
    
    TARGET_KL = 0.02        # 目标KL散度（用于early stop）- Target KL divergence for early stopping
                            # 影响: 如果KL散度超过此值，提前停止policy update（若实现）
                            # Impact: If KL divergence exceeds this, early stop policy update (if implemented)
                            # 推荐范围: 0.01-0.05
                            # Recommended range: 0.01-0.05
                            # 影响: 平衡Actor-Critic训练，控制值函数更新权重
                            # Impact: Balances Actor-Critic training, controls value function update weight
                            # 推荐范围: 0.5-1.0 (0.5 is standard)
                            # Recommended range: 0.5-1.0
    
    # -------------------------------------------------------------------------
    # Logit Bias (用于解决动作空间不平衡问题)
    # Logit Bias (Addresses action space imbalance caused by V2V numerical advantage)
    # -------------------------------------------------------------------------
    USE_LOGIT_BIAS = True   # 是否启用Logit偏置 - Enable logit bias for action balancing
                            # 影响: 对抗V2V数量优势（11个V2V vs 1个Local + 1个RSU）
                            #       强制Agent探索Local和RSU选项，防止"只选V2V"的退化策略
                            # Impact: Counters V2V numerical advantage (11 V2V vs 1 Local + 1 RSU)
                            #       Forces agent to explore Local and RSU; prevents "V2V-only" degenerate policy
    
    LOGIT_BIAS_RSU = 2.4    # RSU的Logit偏置 - Logit bias for RSU action [数学推导: ln(11)≈2.4]
                            # 数学推导: 动作空间 1 Local + 1 RSU + 11 V2V
                            #   要使 P(RSU) = P(Local) = P(V2V_total) = 1/3
                            #   需要 exp(b) / [2*exp(b) + 11] = 1/3
                            #   解得 b = ln(11) ≈ 2.3979
                            # Impact: Mathematically derived for balanced exploration
                            # 推荐范围: 1.7-2.4 (2.4 for equal 33%/33%/33% distribution)
    
    LOGIT_BIAS_LOCAL = 2.4  # Local的Logit偏置 - Logit bias for Local action [数学推导: ln(11)≈2.4]
                            # 数学推导: 与RSU相同，确保初始状态三类动作均衡
                            #   无Bias时: Local 7.7%, RSU 7.7%, V2V 84.6%
                            #   有Bias=2.4时: Local 33.3%, RSU 33.3%, V2V 33.3%
                            # Impact: Mathematically derived for balanced exploration
                            # 推荐范围: 1.7-2.4 (2.4 for equal distribution)
    
    # -------------------------------------------------------------------------
    # Bias退火参数 (Bias Annealing) [适应短期训练]
    # -------------------------------------------------------------------------
    BIAS_DECAY_EVERY_EP = 100  # 每N个episode退火一次 - Decay bias every N episodes [适应短期]
                               # 影响: 控制退火频率，短期训练使用较快退火
                               # Impact: Controls decay frequency; faster decay for short-term training
    
    BIAS_DECAY_RSU = 0.5       # RSU bias每次退火减少量 - RSU bias decay amount per step [降至0.5]
                               # 影响: 每次退火时LOGIT_BIAS_RSU减少的量，更缓慢的衰减
                               # Impact: Amount to reduce LOGIT_BIAS_RSU per decay step, slower decay
    
    BIAS_DECAY_LOCAL = 0.3     # Local bias每次退火减少量 - Local bias decay amount per step [降至0.3]
                               # 影响: 每次退火时LOGIT_BIAS_LOCAL减少的量，更缓慢的衰减
                               # Impact: Amount to reduce LOGIT_BIAS_LOCAL per decay step, slower decay
    
    BIAS_MIN_RSU = 0.5         # RSU bias最小值 - Minimum RSU bias [设为0.5]
                               # 影响: LOGIT_BIAS_RSU不会低于此值，保持RSU探索
                               # Impact: LOGIT_BIAS_RSU will not go below this value, maintains RSU exploration
    
    BIAS_MIN_LOCAL = 0.5       # Local bias最小值 - Minimum Local bias [设为0.5]
                               # 影响: LOGIT_BIAS_LOCAL不会低于此值，保持Local探索
                               # Impact: LOGIT_BIAS_LOCAL will not go below this value, maintains Local exploration

    # =========================================================================
    # 4. 训练流程参数 (Training Loop Control)
    # =========================================================================
    MAX_EPISODES = 500      # 总训练Episodes - Total training episodes [审计调优: 增加训练量]
                            # 影响: 足够的训练量以验证收敛性
                            # Impact: Sufficient training for convergence verification
                            # 验证目标: Task Success Rate > 0%, Decision分布合理, Entropy收敛
                            # Validation goals: Task Success Rate > 0%, balanced decisions, entropy converges
                            # 推荐范围: 500 (验证), 1000-3000 (完整训练)
                            # Recommended range: 500 (validation), 1000-3000 (full training)
    
    MAX_STEPS = 200        # 每个Episode最大步数 - Max steps per episode
                            # 影响: 必须与 SystemConfig.MAX_STEPS 一致
                            #       MAX_STEPS * DT = Episode总时长 (400 * 0.05s = 20s)
                            # Impact: Must match SystemConfig.MAX_STEPS
                            #       MAX_STEPS * DT = Total episode duration (400 * 0.05s = 20s)

    # -------------------------------------------------------------------------
    # 评估与保存 (Evaluation and Checkpointing) [适应短期验证训练]
    # -------------------------------------------------------------------------
    LOG_INTERVAL = 10       # 日志打印间隔 (Episodes) - Log printing interval
                            # 影响: 每N个Episodes打印一次训练日志到终端
                            # Impact: Prints training log to terminal every N episodes
                            # 推荐范围: 5-10 (验证训练频繁监控)
                            # Recommended range: 5-10 (frequent monitoring for validation)
    
    SAVE_INTERVAL = 100     # 模型保存间隔 (Episodes) - Model checkpoint interval [适应短期]
                            # 影响: 每N个Episodes保存一次模型检查点
                            # Impact: Saves model checkpoint every N episodes
                            # 推荐范围: 50-100 (短期), 100-200 (长期)
                            # Recommended range: 50-100 (short-term), 100-200 (long-term)
    
    EVAL_INTERVAL = 1       # 评估间隔 (Episodes) - Evaluation interval [调优: 每episode评估以公平对比]
                            # 影响: 每N个Episodes进行一次验证，短期训练更频繁评估
                            # Impact: Evaluates training progress every N episodes; more frequent for validation
                            # 推荐范围: 20-50 (短期), 50-100 (长期)
                            # Recommended range: 20-50 (short-term), 50-100 (long-term)

    DEVICE_NAME = "cuda"    # 训练设备 - Training device
                            # 选项: "cuda" (GPU), "cpu", "mps" (Apple Silicon)
                            # Options: "cuda" (GPU), "cpu", "mps" (Apple Silicon)
                            # 影响: GPU训练快10-100倍，强烈推荐使用CUDA
                            # Impact: GPU training is 10-100x faster; CUDA strongly recommended

    # =========================================================================
    # 5. Rank-Guided Attention Bias（GA-DRL启发的邻居重要性先验）
    # Rank-Guided Attention Bias (GA-DRL inspired neighbor importance prior)
    # =========================================================================
    """
    【设计说明 - 方案A: Rank Bias 增强】
    作用: 在attention logits中新增rank_prior_bias，作为可控的注意力先验偏置，
         提升DAG表征质量与跨拓扑泛化能力。
    不做: 不改Transformer结构、不做稀疏采样删边、不改DRL action/reward/env.step逻辑。

    【数学原理】
    原始attention: softmax(QK^T + edge_bias + spatial_bias)
    增强attention: softmax(QK^T + edge_bias + spatial_bias + rank_bias)

    【Rank Bias计算】
    1. 得到节点先验权重: w_j = softmax(priority_j / tau) 或 softmax(-rank_j / tau)
    2. 转换为可加bias: rank_bias_{i,j} = kappa * log(w_j + eps)
    3. 物理意义: attention ∝ exp(logits) * (w_j)^kappa
       - kappa控制先验强度
       - 方向一致性: priority大(或rank小) => w大 => bias大 => attention高

    【一致性保证】
    forward() 与 evaluate_actions() 必须一致使用 rank_bias（通过同一forward实现）

    Reference: GA-DRL Rank-Guided Neighbor Sampling (adapted for attention bias)
    """

    # -------------------------------------------------------------------------
    # 5.1 总开关与模式 (Master Switch and Mode)
    # -------------------------------------------------------------------------
    USE_RANK_BIAS = True        # 是否启用Rank偏置 - Enable rank attention bias
                                # 设为False时网络输出应与当前主干完全一致（消融基线）
                                # Set False for ablation baseline (output identical to current)

    RANK_BIAS_MODE = 'priority' # Rank/Priority来源 - Rank/Priority source
                                # 'priority': 复用compute_task_priority()的输出（推荐第一版）
                                # 'gadrl_rank': 使用GA-DRL风格rank递推（第二阶段增强）
                                # 'priority': Reuse compute_task_priority() output (recommended v1)
                                # 'gadrl_rank': Use GA-DRL style rank recursion (v2 enhancement)

    RANK_BIAS_COVER = 'all'     # Bias覆盖模式 - Bias coverage mode
                                # 'all': 对所有i生效，仅依赖key节点j（M1，最稳，推荐）
                                # 'adj': 只对adj[i,j]==1的位置加bias（M2，更贴DAG结构）
                                # 'all': Apply to all i, depends only on key j (M1, most stable, recommended)
                                # 'adj': Only add bias where adj[i,j]==1 (M2, closer to DAG structure)

    # -------------------------------------------------------------------------
    # 5.2 温度与强度参数 (Temperature and Strength Parameters)
    # -------------------------------------------------------------------------
    RANK_BIAS_TAU = 1.0         # Softmax温度参数 - Softmax temperature
                                # 影响: 控制先验分布的尖锐程度
                                #       - tau < 1: 分布更尖锐，强区分重要节点
                                #       - tau > 1: 分布更平滑，弱区分
                                #       - tau = 1: 标准softmax
                                # Impact: Controls sharpness of prior distribution
                                # 推荐范围: 0.5-2.0 (1.0 default)

    RANK_BIAS_KAPPA = 0.5       # Bias强度系数 - Bias strength coefficient
                                # 影响: rank_bias = kappa * log(w + eps)
                                #       attention ∝ exp(logits) * (w)^kappa
                                #       - kappa=0: 无rank先验
                                #       - kappa=0.5: 温和先验
                                #       - kappa=1.0: 完全使用先验
                                # Impact: Controls prior strength
                                # 推荐范围: 0.1-1.0 (0.5 default, start small)

    # -------------------------------------------------------------------------
    # 5.3 归一化参数 (Normalization Parameters)
    # -------------------------------------------------------------------------
    RANK_NORM = 'minmax'        # Priority/Rank归一化方式 - Normalization method
                                # 'minmax': (x - min) / (max - min + eps) -> [0,1]
                                # 'zscore': (x - mean) / (std + eps)
                                # 'none': 不归一化（不推荐，数值尺度不稳定）
                                # 'minmax': (recommended) normalized to [0,1]
                                # 'zscore': z-score normalization
                                # 'none': No normalization (not recommended)

    # -------------------------------------------------------------------------
    # 5.4 GA-DRL Rank递推参数（仅当RANK_BIAS_MODE='gadrl_rank'时使用）
    # GA-DRL Rank Recursion Parameters (only used when RANK_BIAS_MODE='gadrl_rank')
    # -------------------------------------------------------------------------
    RANK_MEAN_CPU_FREQ = 2.0e9  # 平均CPU频率(Hz)用于static模式 - Mean CPU freq for static mode
    RANK_MEAN_RATE = 20e6       # 平均通信速率(bps)用于static模式 - Mean rate for static mode
    RANK_EXEC_COST_MODE = 'mean_cpu'  # exec_cost估计方式
    RANK_COMM_COST_MODE = 'mean_rate' # comm_cost估计方式
