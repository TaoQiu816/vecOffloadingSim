class TrainConfig:
    """
    [训练超参数配置类] TrainConfig

    作用:
    定义模型结构、优化器参数、PPO算法参数及训练流程控制。
    """

    # =========================
    # 1. 网络结构参数 (Model Architecture)
    # =========================
    # 输入特征维度 (必须与 data_utils.py 中的 process_env_obs 输出一致)
    # 对应特征: [comp, data, status, in_degree, t_rem, overdue, urgency]
    TASK_INPUT_DIM = 7

    # Vehicle: [vx, vy, queue, cap, pos_x, pos_y]
    VEH_INPUT_DIM = 7
    EDGE_INPUT_DIM = 2  # 修改为 2

    # RSU: [load, pos_x, pos_y]
    RSU_INPUT_DIM = 1

    # 隐藏层维度 (Transformer & GNN)
    EMBED_DIM = 128

    # 注意力头数 (Heads)
    NUM_HEADS = 4

    # Transformer 层数 (适中深度)
    NUM_LAYERS = 3

    # =========================
    # 2. 优化器参数 (Optimizer)
    # =========================
    # 学习率 (Actor 通常比 Critic 小)
    LR_ACTOR = 3e-4  # Actor 学习率 (PPO标准)
    LR_CRITIC = 1e-3  # Critic 学习率 (稍快)

    # 学习率衰减
    USE_LR_DECAY = True
    LR_DECAY_STEPS = 500  # 每多少个 Episode 衰减一次
    LR_DECAY_RATE = 0.92  # 衰减率

    # 梯度裁剪 (防止梯度爆炸)
    MAX_GRAD_NORM = 0.5

    # =========================
    # 3. PPO 算法参数 (Algorithm)
    # =========================
    GAMMA = 0.95  # 折扣因子 (关注短期Deadline)
    GAE_LAMBDA = 0.95  # GAE 平滑因子 (优势估计平滑)

    CLIP_PARAM = 0.2  # PPO 裁剪阈值 (epsilon)

    PPO_EPOCH = 4  # 每次采样的更新轮数
    MINI_BATCH_SIZE = 64  # 小批次 (适应动态图)

    # 熵正则化系数 (Entropy Coefficient)
    ENTROPY_COEF = 0.02  # 熵系数 (增加探索，应对动态环境)

    # 价值函数损失系数
    VF_COEF = 0.5
    
    # Logit Bias (用于解决动作空间不平衡问题)
    USE_LOGIT_BIAS = True  # 是否启用Logit Bias
    LOGIT_BIAS_RSU = 2.0   # RSU的Logit偏置 (Index 0)
    LOGIT_BIAS_LOCAL = 2.0  # Local的Logit偏置 (Index 1)

    # =========================
    # 4. 训练流程参数 (Training Loop)
    # =========================
    MAX_EPISODES = 5000  # 总训练 Episodes
    MAX_STEPS = 200  # 必须与 SystemConfig.MAX_STEPS 一致

    # 评估与保存
    LOG_INTERVAL = 10  # 日志打印间隔
    SAVE_INTERVAL = 100  # 模型保存间隔
    EVAL_INTERVAL = 50  # 评估间隔

    DEVICE_NAME = "cuda"  # 训练设备