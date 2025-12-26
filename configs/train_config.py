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

    # Transformer 层数
    NUM_LAYERS = 3

    # =========================
    # 2. 优化器参数 (Optimizer)
    # =========================
    # 学习率 (Actor 通常比 Critic 小)
    LR_ACTOR = 1e-4  # 降低学习率以求稳定
    LR_CRITIC = 1e-4  # Critic 需要更快收敛

    # 学习率衰减
    USE_LR_DECAY = True
    LR_DECAY_STEPS = 500  # 每多少个 Episode 衰减一次
    # [修改] 衰减率
    LR_DECAY_RATE = 0.92

    # 梯度裁剪 (防止梯度爆炸)
    MAX_GRAD_NORM = 0.5

    # =========================
    # 3. PPO 算法参数 (Algorithm)
    # =========================
    GAMMA = 0.99  # 折扣因子 (关注长期收益)
    GAE_LAMBDA = 0.95  # GAE 平滑因子 (平衡偏差与方差)

    CLIP_PARAM = 0.2  # PPO 裁剪阈值 (epsilon)

    PPO_EPOCH = 10  # 增加PPO迭代次数，提高训练稳定性
    MINI_BATCH_SIZE = 512  # 减小批次大小，降低计算成本

    # 熵正则化系数 (Entropy Coefficient)：值越大，策略越随机；值越小，策略越确定。
    # Reward 曲线完全是平的（不上升），说明探索不够，可以改到 0.02。
    # 如果发现 Reward 震荡极其剧烈且无法稳定，改到 0.005。
    # 0.01 是标准值。如果发现 Agent 过早收敛到单一动作，增大此值 (e.g., 0.05)
    ENTROPY_COEF = 0.05  # 增加探索以寻找更好的策略

    # 价值函数损失系数
    VF_COEF = 0.5

    # =========================
    # 4. 训练流程参数 (Training Loop)
    # =========================
    MAX_EPISODES = 3000  # 总训练轮次 (建议至少 2000-5000)
    MAX_STEPS = 100  # 每个 Episode 的步数 (与 SystemConfig 保持一致)

    # 评估与保存
    LOG_INTERVAL = 10  # 每多少轮打印一次日志
    SAVE_INTERVAL = 100  # 每多少轮保存一次模型
    EVAL_INTERVAL = 50  # 每多少轮评估一次 (不加噪声)

    DEVICE_NAME = "cuda"  # 强制使用 CUDA，如果可用