import os


def _recompute_derived(Cfg):
    Cfg.ALL_FEASIBLE = (str(getattr(Cfg, "CANDIDATE_MODE", "ALL")).upper() == "ALL")
    Cfg.MAX_NEIGHBORS = (Cfg.NUM_VEHICLES - 1) if Cfg.ALL_FEASIBLE else max(0, min(Cfg.NUM_VEHICLES - 1, Cfg.V2V_TOP_K))
    Cfg.MAX_TARGETS = (1 + Cfg.NUM_RSU + Cfg.MAX_NEIGHBORS) if Cfg.ENABLE_RSU_SELECTION else (2 + Cfg.MAX_NEIGHBORS)
    if int(getattr(Cfg, "V2V_NUM_RB", 1)) <= 0:
        Cfg.V2V_NUM_RB = 1
    Cfg.V2V_BW_PER_RB = float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)
    if int(getattr(Cfg, "V2I_NUM_RB", 1)) <= 0:
        Cfg.V2I_NUM_RB = 1


def _recompute_workload_and_norms(Cfg):
    # Keep derived workload / normalization scales consistent with overridden task ranges.
    Cfg.MEAN_COMP_LOAD = (float(Cfg.MIN_COMP) + float(Cfg.MAX_COMP)) / 2.0
    Cfg.AVG_COMP = Cfg.MEAN_COMP_LOAD
    Cfg.NORM_MAX_COMP = max(float(Cfg.MAX_COMP), 1.0)
    Cfg.NORM_MAX_DATA = max(float(Cfg.MAX_DATA), 1.0)
    try:
        Cfg._RSU_MAX_WAIT = float(Cfg.RSU_QUEUE_CYCLES_LIMIT) / max(float(Cfg.F_RSU), 1e-9)
        Cfg._VEHICLE_MAX_WAIT = float(Cfg.VEHICLE_QUEUE_CYCLES_LIMIT) / max(float(Cfg.MIN_VEHICLE_CPU_FREQ), 1e-9)
        Cfg.NORM_MAX_WAIT_TIME = max(float(Cfg._RSU_MAX_WAIT), float(Cfg._VEHICLE_MAX_WAIT)) * 1.2
    except Exception:
        pass


def apply_exp_dynamic_config(Cfg, TC):
    """
    静态车辆 + 动态任务持续生成 实验配置。

    场景设计：
    - 固定 8 辆车持续在场，不动态增减
    - 每辆车完成/失败当前 DAG 后立即获得新任务（TASK_RESPAWN_ON_COMPLETION）
    - 200 步截断作为唯一 episode 终止条件
    - 中等难度：MAX_NODES=15，ALPHA=1.8~2.5，打破零正反馈
    """

    # ------------------------------------------------------------------
    # 场景规模 / 移动性
    # ------------------------------------------------------------------
    Cfg.SEED = 123
    Cfg.NUM_VEHICLES = 8          # 固定 8 辆，不动态增减
    Cfg.NUM_RSU = 4               # 4 个 RSU，2000m 路段均匀分布（间距 500m）
    Cfg.NUM_LANES = 3
    # 4 RSU × RSU_RANGE=350m：间距 500m，重叠 200m，全路段无盲区
    Cfg.MAP_SIZE = 2000.0
    Cfg.RSU_Y_DIST = 10.0
    Cfg.RSU_RANGE = 350.0
    Cfg.V2V_RANGE = 220.0
    Cfg.VEHICLE_SPAWN_X_MIN = 0.02
    Cfg.VEHICLE_SPAWN_X_MAX = 0.98
    Cfg.VEHICLE_ARRIVAL_RATE = 0.0   # 关闭泊松车辆生成；任务重生由 TASK_RESPAWN_ON_COMPLETION 驱动
    Cfg.TASK_RESPAWN_ON_COMPLETION = True  # DAG 完成/失败后立即为该车分配新任务
    Cfg.VEL_MEAN = 18.0
    Cfg.VEL_STD = 5.0
    Cfg.VEL_MIN = 4.0
    Cfg.VEL_MAX = 32.0
    Cfg.MAX_VELOCITY = max(float(Cfg.VEL_MAX), 1.0)

    # ------------------------------------------------------------------
    # 通信 / 干扰 / 排队竞争
    # ------------------------------------------------------------------
    Cfg.ENABLE_RSU_SELECTION = True
    Cfg.CANDIDATE_MODE = "ALL"
    Cfg.V2V_TOP_K = 6
    Cfg.USE_BLOCK_FADING = True
    Cfg.V2I_ICI_ENABLED = True
    Cfg.V2I_RATE_MODEL = "SHARE"
    Cfg.V2I_NUM_RB = 5
    Cfg.V2V_NUM_RB = 5
    Cfg.BW_V2I = 10e6
    Cfg.BW_V2V = 10e6
    Cfg.VEHICLE_QUEUE_CYCLES_LIMIT = 12e9
    Cfg.RSU_QUEUE_CYCLES_LIMIT = 60e9

    # ------------------------------------------------------------------
    # DAG 工作负载（中等难度，打破零正反馈）
    # ------------------------------------------------------------------
    Cfg.DAG_SOURCE = "synthetic_small"
    Cfg.MIN_NODES = 6
    Cfg.MAX_NODES = 15             # 从 24 降至 15，缩短关键路径，使 200 步内可完成
    Cfg.MIN_VEHICLE_CPU_FREQ = 0.5e9
    Cfg.MAX_VEHICLE_CPU_FREQ = 1.5e9
    Cfg.F_RSU = 20.0e9
    Cfg.MIN_COMP = 1.0e8           # 0.1 Gcycles
    Cfg.MAX_COMP = 1.0e9           # 1.0 Gcycles（绝对不能是 4.0e9）
    Cfg.MIN_DATA = 8.0e5           # 100 KB
    Cfg.MAX_DATA = 8.0e6           # 1 MB
    Cfg.DOMAIN_RANDOMIZATION = True
    Cfg.DEADLINE_MODE = "LB_ALPHA"
    # 适度放宽 deadline，给智能体足够试错空间同时保持区分度
    Cfg.DEADLINE_ALPHA_MIN = 1.8
    Cfg.DEADLINE_ALPHA_MAX = 2.5
    Cfg.DEADLINE_SLACK_SECONDS = 0.10
    Cfg.DEADLINE_TIGHTENING_MIN = 1.05
    Cfg.DEADLINE_TIGHTENING_MAX = 1.40
    Cfg.MAX_INFLIGHT_SUBTASKS_PER_VEHICLE = 0

    # ------------------------------------------------------------------
    # 可选现实特征
    # ------------------------------------------------------------------
    Cfg.TRUST_ENABLED = True
    Cfg.CHAIN_ENABLED = False

    # ------------------------------------------------------------------
    # 训练 / 评估默认参数
    # ------------------------------------------------------------------
    TC.MAX_EPISODES = 2000
    Cfg.MAX_STEPS = 200
    TC.MAX_STEPS = 200
    TC.LOG_INTERVAL = 20
    TC.EVAL_INTERVAL = 100
    TC.SAVE_INTERVAL = 200
    TC.MINI_BATCH_SIZE = 256
    TC.PPO_EPOCH = 5
    TC.USE_LR_DECAY = True
    # 固定熵系数，关闭退火，保持非平稳环境持续探索
    TC.ENTROPY_COEF = 0.015
    TC.ENTROPY_COEF_START = 0.015
    TC.ENTROPY_COEF_END = 0.015
    TC.ENTROPY_ANNEAL_STEPS = 0

    _recompute_derived(Cfg)
    _recompute_workload_and_norms(Cfg)
    return {
        "profile": "exp_dynamic",
        "num_vehicles": int(Cfg.NUM_VEHICLES),
        "num_rsu": int(Cfg.NUM_RSU),
        "max_targets": int(Cfg.MAX_TARGETS),
        "dag_source": str(Cfg.DAG_SOURCE),
    }
