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
    Restore a realistic dynamic VEC experiment profile for paper-scale runs.

    Goals:
    - multi-vehicle + multi-RSU with mobility
    - stochastic DAGs (varied topology/size/load)
    - queueing + interference contention enabled
    - periodic evaluation enabled (incl. baseline eval)
    """

    # ------------------------------------------------------------------
    # Scenario scale / mobility (high-dynamic)
    # ------------------------------------------------------------------
    Cfg.SEED = 123
    Cfg.NUM_VEHICLES = 10
    Cfg.NUM_RSU = 3
    Cfg.NUM_LANES = 3
    # For NUM_RSU=3 and fixed RSU_RANGE=350m, use a longer road so adjacent RSUs
    # keep only small overlap while still guaranteeing full coverage.
    Cfg.MAP_SIZE = 2000.0
    Cfg.RSU_Y_DIST = 10.0
    Cfg.RSU_RANGE = 350.0
    Cfg.V2V_RANGE = 220.0
    # Spawn vehicles almost across the whole covered road while avoiding exact boundaries.
    Cfg.VEHICLE_SPAWN_X_MIN = 0.02
    Cfg.VEHICLE_SPAWN_X_MAX = 0.98
    Cfg.VEHICLE_ARRIVAL_RATE = 0.30  # high-load dynamic arrivals (veh/s)
    Cfg.VEL_MEAN = 18.0
    Cfg.VEL_STD = 5.0
    Cfg.VEL_MIN = 4.0
    Cfg.VEL_MAX = 32.0
    Cfg.MAX_VELOCITY = max(float(Cfg.VEL_MAX), 1.0)

    # ------------------------------------------------------------------
    # Communication / interference / queueing contention
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
    # finite queue budgets -> real contention
    Cfg.VEHICLE_QUEUE_CYCLES_LIMIT = 12e9
    Cfg.RSU_QUEUE_CYCLES_LIMIT = 60e9

    # ------------------------------------------------------------------
    # DAG workload diversity (random generator)
    # ------------------------------------------------------------------
    Cfg.DAG_SOURCE = "synthetic_small"  # stochastic DAG generator
    Cfg.MIN_NODES = 10
    Cfg.MAX_NODES = 24
    # High-pressure compute / task profile (top-tier VEC setting)
    Cfg.MIN_VEHICLE_CPU_FREQ = 0.5e9
    Cfg.MAX_VEHICLE_CPU_FREQ = 1.5e9
    Cfg.F_RSU = 20.0e9
    Cfg.MIN_COMP = 1.0e8
    Cfg.MAX_COMP = 1.0e9
    # Input data uses bits in codebase: 100KB=8e5 bits, 1MB=8e6 bits
    Cfg.MIN_DATA = 8.0e5
    Cfg.MAX_DATA = 8.0e6
    Cfg.DOMAIN_RANDOMIZATION = True
    # keep workload/deadline stochastic with moderate difficulty
    Cfg.DEADLINE_MODE = "LB_ALPHA"
    # With 10 vehicles / 3 RSUs on a 2km road, pure LB-based deadlines can become too harsh
    # because LB omits queueing/communication contention. Use a slightly looser but still
    # learnable alpha range plus small slack.
    Cfg.DEADLINE_ALPHA_MIN = 1.2
    Cfg.DEADLINE_ALPHA_MAX = 1.8
    Cfg.DEADLINE_SLACK_SECONDS = 0.10
    Cfg.DEADLINE_TIGHTENING_MIN = 1.05
    Cfg.DEADLINE_TIGHTENING_MAX = 1.40
    Cfg.MAX_INFLIGHT_SUBTASKS_PER_VEHICLE = 0

    # ------------------------------------------------------------------
    # Optional realism features
    # ------------------------------------------------------------------
    Cfg.TRUST_ENABLED = True
    Cfg.CHAIN_ENABLED = False  # keep chain off for main dynamic smoke unless specifically studied

    # ------------------------------------------------------------------
    # Training / evaluation defaults for paper runs
    # ------------------------------------------------------------------
    TC.MAX_EPISODES = 2000
    # Use a longer horizon for 8-vehicle concurrent stochastic DAGs (up to 24 nodes)
    # so most episodes can reach meaningful terminal outcomes rather than truncation.
    Cfg.MAX_STEPS = 200
    TC.MAX_STEPS = 200
    TC.LOG_INTERVAL = 20
    TC.EVAL_INTERVAL = 100
    TC.SAVE_INTERVAL = 200
    TC.MINI_BATCH_SIZE = 256
    TC.PPO_EPOCH = 5
    TC.USE_LR_DECAY = True
    # Entropy schedule: start near 0.01 (recommended), decay linearly for late-stage stabilization.
    TC.ENTROPY_COEF = 0.015
    TC.ENTROPY_COEF_START = 0.015
    TC.ENTROPY_COEF_END = 0.015
    TC.ENTROPY_ANNEAL_STEPS = 0
    # Preserve existing model size defaults unless user overrides via CLI/env

    _recompute_derived(Cfg)
    _recompute_workload_and_norms(Cfg)
    return {
        "profile": "exp_dynamic",
        "num_vehicles": int(Cfg.NUM_VEHICLES),
        "num_rsu": int(Cfg.NUM_RSU),
        "max_targets": int(Cfg.MAX_TARGETS),
        "dag_source": str(Cfg.DAG_SOURCE),
    }
