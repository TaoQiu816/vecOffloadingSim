import os


def _recompute_derived(Cfg):
    """Recompute a small set of derived config fields after overrides."""
    Cfg.ALL_FEASIBLE = (str(getattr(Cfg, "CANDIDATE_MODE", "ALL")).upper() == "ALL")
    Cfg.MAX_NEIGHBORS = (Cfg.NUM_VEHICLES - 1) if Cfg.ALL_FEASIBLE else max(0, min(Cfg.NUM_VEHICLES - 1, Cfg.V2V_TOP_K))
    Cfg.MAX_TARGETS = 1 + Cfg.NUM_RSU + Cfg.MAX_NEIGHBORS
    if int(getattr(Cfg, "V2V_NUM_RB", 1)) <= 0:
        Cfg.V2V_NUM_RB = 1
    Cfg.V2V_BW_PER_RB = float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)
    if int(getattr(Cfg, "V2I_NUM_RB", 0)) <= 0:
        Cfg.V2I_NUM_RB = 1


def apply_overfit_config(Cfg, TC):
    """
    Apply a deterministic single-sample micro-overfitting profile.

    Goal:
    - 1 vehicle, 1 RSU, no mobility, fixed resources
    - fixed 3-node workflow DAG every episode
    - compact PPO settings for quick overfitting diagnostics
    """
    workflow_path = os.path.join(os.path.dirname(__file__), "workflows", "overfit_linear3.json")

    # ------------------------------------------------------------------
    # Deterministic environment topology / mobility
    # ------------------------------------------------------------------
    Cfg.SEED = 123
    Cfg.NUM_VEHICLES = 1
    Cfg.NUM_RSU = 1
    Cfg.NUM_LANES = 1
    Cfg.MAP_SIZE = 100.0
    Cfg.RSU_Y_DIST = 8.0
    Cfg.RSU_RANGE = 500.0
    Cfg.VEHICLE_SPAWN_X_MIN = 0.50
    Cfg.VEHICLE_SPAWN_X_MAX = 0.50
    Cfg.VEHICLE_ARRIVAL_RATE = 0.0

    # Freeze mobility: sampled speed is always 0, but keep MAX_VELOCITY > 0 for normalization
    Cfg.VEL_MEAN = 0.0
    Cfg.VEL_STD = 0.0
    Cfg.VEL_MIN = 0.0
    Cfg.VEL_MAX = 0.0
    Cfg.MAX_VELOCITY = 1.0

    # ------------------------------------------------------------------
    # Fixed resources / comm
    # ------------------------------------------------------------------
    Cfg.MIN_VEHICLE_CPU_FREQ = 0.5e9
    Cfg.MAX_VEHICLE_CPU_FREQ = 0.5e9
    Cfg.F_RSU = 5.0e9
    Cfg.RSU_NUM_PROCESSORS = 1
    Cfg.BW_V2I = 20e6
    Cfg.BW_V2V = 10e6
    Cfg.V2I_RB_BW_HZ = 20e6
    Cfg.V2I_NUM_RB = 1
    Cfg.V2V_NUM_RB = 1
    Cfg.V2I_RATE_MODEL = "RB_SINR"
    Cfg.USE_BLOCK_FADING = False
    Cfg.V2I_ICI_ENABLED = False
    Cfg.DOMAIN_RANDOMIZATION = False
    Cfg.TRUST_ENABLED = False
    Cfg.CHAIN_ENABLED = False

    # Queue capacity kept finite but high enough to avoid incidental rejection in micro test
    Cfg.VEHICLE_QUEUE_CYCLES_LIMIT = 20e9
    Cfg.RSU_QUEUE_CYCLES_LIMIT = 20e9

    # ------------------------------------------------------------------
    # Fixed DAG / deadlines
    # ------------------------------------------------------------------
    Cfg.DAG_SOURCE = "workflow_json"
    Cfg.WORKFLOW_JSON_PATH = workflow_path
    Cfg.MIN_NODES = 3
    Cfg.MAX_NODES = 4  # keep padding headroom while task is fixed 3 nodes
    Cfg.DEADLINE_MODE = "FIXED_RANGE"
    Cfg.DEADLINE_FIXED_MIN = 1.0
    Cfg.DEADLINE_FIXED_MAX = 1.0
    Cfg.DEADLINE_TIGHTENING_MIN = 1.0
    Cfg.DEADLINE_TIGHTENING_MAX = 1.0
    Cfg.MAX_INFLIGHT_SUBTASKS_PER_VEHICLE = 0

    # Action set becomes {Local, RSU0}; no V2V candidates
    Cfg.CANDIDATE_MODE = "ALL"
    Cfg.V2V_TOP_K = 0
    Cfg.ENABLE_RSU_SELECTION = True

    # Keep unified reward path with scaling (from previous fix)
    Cfg.REWARD_SCHEME = "UNIFIED"
    Cfg.REWARD_SCALE = 10.0
    Cfg.MAX_STEPS = 40

    # ------------------------------------------------------------------
    # PPO / network settings for fast micro-overfit
    # ------------------------------------------------------------------
    TC.MAX_EPISODES = 300
    TC.MAX_STEPS = 40
    TC.LOG_INTERVAL = 10
    TC.SAVE_INTERVAL = 1000
    TC.EVAL_INTERVAL = 1000

    TC.EMBED_DIM = 64
    TC.NUM_HEADS = 2
    TC.NUM_LAYERS = 1
    TC.D_FF = 128
    TC.DROPOUT = 0.0

    TC.LR_ACTOR = 3e-4
    TC.LR_CRITIC = 1e-3
    TC.USE_LR_DECAY = False
    TC.PPO_EPOCH = 8
    TC.MINI_BATCH_SIZE = 32
    TC.MIN_ACTIVE_SAMPLES = 1

    TC.ENTROPY_COEF = 0.01
    TC.ENTROPY_COEF_START = 0.01
    TC.ENTROPY_COEF_END = 0.0
    TC.ENTROPY_ANNEAL_STEPS = 2000
    TC.USE_LOGIT_BIAS = False
    TC.CMDP_ENABLE = False
    TC.LATE_GUARD_ENABLE = False

    _recompute_derived(Cfg)
    return {"workflow_path": workflow_path, "num_targets": Cfg.MAX_TARGETS}
