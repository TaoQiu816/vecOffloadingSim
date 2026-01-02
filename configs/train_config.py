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

    NUM_HEADS = 4           # 注意力头数 - Number of attention heads (Multi-Head Attention)
                            # 影响: 更多头数可以捕获更多样化的依赖关系，必须被EMBED_DIM整除
                            # Impact: More heads capture diverse dependencies; must divide EMBED_DIM evenly
                            # 推荐范围: 4-8 (for EMBED_DIM=128)
                            # Recommended range: 4-8

    NUM_LAYERS = 3          # Transformer层数 - Number of Transformer encoder layers
                            # 影响: 更深的网络可以学习更复杂的特征，但可能过拟合或训练困难
                            # Impact: Deeper networks learn complex features but may overfit or be hard to train
                            # 推荐范围: 2-6 (2 for simple tasks, 6 for complex dependencies)
                            # Recommended range: 2-6

    # =========================================================================
    # 2. 优化器参数 (Optimizer Parameters)
    # =========================================================================
    LR_ACTOR = 3e-4         # Actor学习率 - Actor learning rate (LR = 3e-4)
                            # 影响: 控制策略网络的更新速度
                            #       - 过大: 训练不稳定，策略震荡
                            #       - 过小: 收敛慢，需要更多训练时间
                            # Impact: Controls policy network update speed
                            #       - Too large: Unstable training, policy oscillation
                            #       - Too small: Slow convergence, needs more training time
                            # 推荐范围: 1e-4 ~ 5e-4 (PPO标准)
                            # Recommended range: 1e-4 ~ 5e-4 (PPO standard)
    
    LR_CRITIC = 1e-3        # Critic学习率 - Critic learning rate
                            # 影响: 控制价值网络的更新速度，通常比Actor略快以快速拟合值函数
                            # Impact: Controls value network update speed; typically faster than actor
                            # 推荐范围: 3e-4 ~ 3e-3
                            # Recommended range: 3e-4 ~ 3e-3

    USE_LR_DECAY = True     # 是否启用学习率衰减 - Enable learning rate decay
                            # 影响: 训练后期降低学习率有助于收敛和稳定性
                            # Impact: Reducing LR in late training aids convergence and stability
    
    LR_DECAY_STEPS = 500    # 学习率衰减间隔 (Episodes) - Learning rate decay interval
                            # 影响: 每N个Episodes衰减一次，控制衰减频率
                            # Impact: Decay every N episodes; controls decay frequency
                            # 推荐范围: 100-1000 (取决于MAX_EPISODES)
                            # Recommended range: 100-1000 (depends on MAX_EPISODES)
    
    LR_DECAY_RATE = 0.92    # 学习率衰减率 - Learning rate decay rate
                            # 影响: 指数衰减系数 (new_lr = lr * decay_rate)
                            #       - 0.9: 快速衰减，适合短期训练
                            #       - 0.95: 慢速衰减，适合长期训练
                            # Impact: Exponential decay coefficient (new_lr = lr * decay_rate)
                            #       - 0.9: Fast decay, suits short-term training
                            #       - 0.95: Slow decay, suits long-term training
                            # 推荐范围: 0.90-0.95
                            # Recommended range: 0.90-0.95

    MAX_GRAD_NORM = 0.5     # 梯度裁剪阈值 - Gradient clipping threshold
                            # 影响: 防止梯度爆炸，保证训练稳定性
                            #       - 过小: 限制学习速度
                            #       - 过大: 可能梯度爆炸
                            # Impact: Prevents gradient explosion, ensures training stability
                            #       - Too small: Limits learning speed
                            #       - Too large: May still explode
                            # 推荐范围: 0.5-2.0 (0.5 conservative, 2.0 aggressive)
                            # Recommended range: 0.5-2.0

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
    
    MINI_BATCH_SIZE = 128   # 小批次大小 - Mini-batch size for SGD updates (MINIBATCHES = 4, 每批128样本)
                            # 影响: 梯度估计的方差和计算效率
                            #       - 较大: 梯度稳定，但内存占用高
                            #       - 较小: 增加随机性，适应动态图结构
                            # Impact: Variance of gradient estimation and computational efficiency
                            #       - Larger: Stable gradients, but higher memory usage
                            #       - Smaller: Increases stochasticity, adapts to dynamic graphs
                            # 推荐范围: 64-256 (128 for stable gradients)
                            # Recommended range: 64-256 (128 for stable gradients)

    ENTROPY_COEF = 0.01     # 熵正则化系数 - Entropy coefficient for exploration
                            # 影响: 增加动作探索性，应对动态环境
                            #       - 过大: 策略过于随机，难以收敛
                            #       - 过小: 策略过早收敛到局部最优
                            # Impact: Increases action exploration for dynamic environments
                            #       - Too large: Policy too random, hard to converge
                            #       - Too small: Policy converges prematurely to local optimum
                            # 推荐范围: 0.01-0.05 (0.03 for enhanced exploration)
                            # Recommended range: 0.01-0.05 (0.03 for enhanced exploration)

    VF_COEF = 0.5           # 价值函数损失系数 - Value function loss coefficient
                            # 影响: 平衡Actor-Critic训练，控制值函数更新权重
                            # Impact: Balances Actor-Critic training, controls value function update weight
                            # 推荐范围: 0.5-1.0 (0.5 is standard)
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
    
    LOGIT_BIAS_RSU = 5.0    # RSU的Logit偏置 - Logit bias for RSU action
                            # 影响: 在softmax前给RSU logit加上偏置，大幅提升选择概率以对抗11个邻居的概率淹没
                            #       提升至5.0以强制探索RSU，防止策略崩溃到V2V
                            # Impact: Adds bias to RSU logit before softmax, drastically boosts selection to counter neighbor swamping
                            #       Reduced to 2.0 to allow V2V exploration; 8.0 suppresses V2V to <0.2%
                            # 推荐范围: 1.0-3.0 (平衡探索与利用)
                            # Recommended range: 1.0-3.0 (balances exploration and exploitation)
    
    LOGIT_BIAS_LOCAL = 2.0  # Local的Logit偏置 - Logit bias for Local action
                            # 影响: 在softmax前给Local logit加上偏置，中等提升选择概率
                            #       提升至2.0以增加Local探索，配合Deadline放宽策略
                            # Impact: Adds bias to Local logit before softmax, moderately boosts selection
                            #       Reduced to 1.0 to encourage offloading; Local still ~15-20% initial prob
                            # 推荐范围: 0.5-2.0 (轻微偏置即可)
                            # Recommended range: 0.5-2.0 (light bias sufficient)
    
    # -------------------------------------------------------------------------
    # Bias退火参数 (Bias Annealing)
    # -------------------------------------------------------------------------
    BIAS_DECAY_EVERY_EP = 100  # 每N个episode退火一次 - Decay bias every N episodes
                               # 影响: 控制退火频率，100表示每100个episode降低一次bias
                               # Impact: Controls decay frequency; 100 means decay every 100 episodes
    
    BIAS_DECAY_RSU = 1.0       # RSU bias每次退火减少量 - RSU bias decay amount per step
                               # 影响: 每次退火时LOGIT_BIAS_RSU减少的量
                               # Impact: Amount to reduce LOGIT_BIAS_RSU per decay step
    
    BIAS_DECAY_LOCAL = 0.5     # Local bias每次退火减少量 - Local bias decay amount per step
                               # 影响: 每次退火时LOGIT_BIAS_LOCAL减少的量
                               # Impact: Amount to reduce LOGIT_BIAS_LOCAL per decay step
    
    BIAS_MIN_RSU = 0.0         # RSU bias最小值 - Minimum RSU bias
                               # 影响: LOGIT_BIAS_RSU不会低于此值
                               # Impact: LOGIT_BIAS_RSU will not go below this value
    
    BIAS_MIN_LOCAL = 0.0       # Local bias最小值 - Minimum Local bias
                               # 影响: LOGIT_BIAS_LOCAL不会低于此值
                               # Impact: LOGIT_BIAS_LOCAL will not go below this value

    # =========================================================================
    # 4. 训练流程参数 (Training Loop Control)
    # =========================================================================
    MAX_EPISODES = 5000     # 总训练Episodes - Total training episodes
                            # 影响: 控制训练总量，5000个Episodes约需数小时到数天（取决于硬件）
                            # Impact: Controls total training volume; 5000 episodes take hours to days (hardware-dependent)
                            # 推荐范围: 1000-10000 (1000 for quick experiments, 10000 for production)
                            # Recommended range: 1000-10000
    
    MAX_STEPS = 400         # 每个Episode最大步数 - Max steps per episode
                            # 影响: 必须与 SystemConfig.MAX_STEPS 一致
                            #       MAX_STEPS * DT = Episode总时长 (400 * 0.05s = 20s)
                            # Impact: Must match SystemConfig.MAX_STEPS
                            #       MAX_STEPS * DT = Total episode duration (400 * 0.05s = 20s)

    # -------------------------------------------------------------------------
    # 评估与保存 (Evaluation and Checkpointing)
    # -------------------------------------------------------------------------
    LOG_INTERVAL = 10       # 日志打印间隔 (Episodes) - Log printing interval
                            # 影响: 每N个Episodes打印一次训练日志到终端
                            # Impact: Prints training log to terminal every N episodes
                            # 推荐范围: 1-50 (1 for debugging, 50 for production)
                            # Recommended range: 1-50
    
    SAVE_INTERVAL = 100     # 模型保存间隔 (Episodes) - Model checkpoint interval
                            # 影响: 每N个Episodes保存一次模型检查点，影响磁盘占用
                            # Impact: Saves model checkpoint every N episodes; affects disk usage
                            # 推荐范围: 50-500 (取决于MAX_EPISODES和磁盘空间)
                            # Recommended range: 50-500 (depends on MAX_EPISODES and disk space)
    
    EVAL_INTERVAL = 50      # 评估间隔 (Episodes) - Evaluation interval
                            # 影响: 每N个Episodes进行一次验证，评估训练进度
                            # Impact: Evaluates training progress every N episodes
                            # 推荐范围: 10-100
                            # Recommended range: 10-100

    DEVICE_NAME = "cuda"    # 训练设备 - Training device
                            # 选项: "cuda" (GPU), "cpu", "mps" (Apple Silicon)
                            # Options: "cuda" (GPU), "cpu", "mps" (Apple Silicon)
                            # 影响: GPU训练快10-100倍，强烈推荐使用CUDA
                            # Impact: GPU training is 10-100x faster; CUDA strongly recommended
