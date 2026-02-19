"""
Final paper profile (single-reward, CTDE-MAPPO) for IoV/VEC.

This profile is intended as the default reproducible scenario for paper runs.
"""

from __future__ import annotations

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


FINAL_ENV = {
    # Scenario scale
    "NUM_VEHICLES": 20,
    "NUM_RSU": 3,
    "MAX_STEPS": 200,
    "CANDIDATE_MODE": "ALL",
    "ALL_FEASIBLE": True,
    # Coverage hierarchy: RSU must dominate coverage radius.
    "RSU_RANGE": 350.0,
    "V2V_RANGE": 250.0,
    # V2I/V2V physical models
    "V2I_RATE_MODEL": "RB_SINR",
    "V2I_ICI_ENABLED": True,
    "V2I_FREQ_REUSE_FACTOR": 1,
    "BW_V2I": 20.0e6,
    "V2I_NUM_RB": 80,
    "V2V_NUM_RB": 2,
    # Route1: fixed one DAG per vehicle per episode (no mid-episode arrivals).
    "VEHICLE_ARRIVAL_RATE": 0.0,
    # CPU envelope (avoid trivial pure-local dominance)
    "MIN_VEHICLE_CPU_FREQ": 1.5e9,
    "MAX_VEHICLE_CPU_FREQ": 4.0e9,
    "F_RSU": 12.0e9,
    # DAG/task generation (A/B mixture)
    "MIN_NODES": 8,
    "MAX_NODES": 16,
    "DAG_FAT": 1.00,
    "DAG_DENSITY": 0.24,
    "TASK_CLASS_MIX_ENABLE": True,
    "TASK_CLASS_B_PROB": 0.11,
    # Class A: 0.5~6 Mb, 4e8~1.8e9 cycles, 0.5~1.8 s
    "TASK_A_TOTAL_DATA_MIN": 5.0e5,
    "TASK_A_TOTAL_DATA_MAX": 6.0e6,
    "TASK_A_TOTAL_COMP_MIN": 4.0e8,
    "TASK_A_TOTAL_COMP_MAX": 1.8e9,
    "TASK_A_DEADLINE_MIN": 0.5,
    "TASK_A_DEADLINE_MAX": 2.0,
    # Class B: 6~25 Mb, 1.5e9~6e9 cycles, 1.2~3.0 s
    "TASK_B_TOTAL_DATA_MIN": 6.0e6,
    "TASK_B_TOTAL_DATA_MAX": 2.5e7,
    "TASK_B_TOTAL_COMP_MIN": 1.5e9,
    "TASK_B_TOTAL_COMP_MAX": 6.0e9,
    "TASK_B_DEADLINE_MIN": 1.2,
    "TASK_B_DEADLINE_MAX": 3.0,
    # split jitter for heterogeneity
    "TASK_COMP_SPLIT_JITTER": 0.6,
    "TASK_DATA_SPLIT_JITTER": 0.8,
    # Deadline lower-bound safety with DT quantization
    "DEADLINE_MODE": "LB_ALPHA",
    "DEADLINE_LB_EPS": 0.02,
    "DEADLINE_STEP_GUARD_DELTA": 3.0,
    "DEADLINE_ALPHA_MIN": 2.0,
    "DEADLINE_ALPHA_MAX": 4.0,
    "DEADLINE_SLACK_SECONDS": 0.2,
    # Trust heterogeneity (sampling only; beta update unchanged)
    "TRUST_ENABLED": True,
    "TRUST_P_RSU_RANGE": (0.95, 0.995),
    "TRUST_P_VEH_RANGE": (0.60, 0.95),
    "TRUST_RELIABLE_PROB": 0.8,
    "TRUST_PRIOR_A": 1.0,
    "TRUST_PRIOR_B": 1.0,
    # Paper-unique reward weights (single formula)
    "REWARD_SCHEME": "UNIFIED",
    "W_TIME": 0.35,
    "W_ENERGY": 0.10,
    "W_INTERF": 0.10,
    "W_RISK": 0.10,
    "W_ILLEGAL": 30.0,
    "R_SUCC": 20.0,
    "R_FAIL": 20.0,
    "P_SUCC": 1.0,
    "P_FAIL": 1.0,
    # robust scale controls for unified step terms (Fix-A)
    "ENERGY_RATIO_CLIP_UNIFIED": 1.0,
    "INTERF_RATIO_CLIP_UNIFIED": 3.0,
    "RISK_RATIO_CLIP_UNIFIED": 1.0,
    "REWARD_REF_EMA_ALPHA": 0.05,
    "REWARD_REF_ENERGY_MIN": 1e-8,
    "REWARD_REF_RISK_MIN": 0.05,
    "RISK_REF_UNIFIED_INIT": 0.25,
    "E_REF_UNIFIED": 2.0,
}


FINAL_TRAIN = {
    "MAX_STEPS": 200,
    # CTDE fixed path
    "USE_SIMPLIFIED_CRITIC": False,
    "COMMWAIT_DIRECT_TO_CRITIC": True,
    "CTDE_GLOBAL_DIM": 30,
    # reward-only: disable CMDP
    "CMDP_ENABLE": False,
    "CMDP_LAMBDA_LR": 0.0,
    "CMDP_LAMBDA_MAX": 0.0,
    "CMDP_WARMUP_EPISODES": 0,
    # category-level mild prior + group-size correction (implemented in policy)
    "USE_LOGIT_BIAS": True,
    "LOGIT_BIAS_LOCAL_INIT": 0.20,
    "LOGIT_BIAS_RSU_INIT": 0.05,
    "LOGIT_BIAS_V2V_INIT": 0.05,
    "BIAS_ANNEAL_FRAC": 0.65,
    "LOGIT_BIAS_LOCAL_END": 0.00,
    "LOGIT_BIAS_RSU_END": 0.03,
    "LOGIT_BIAS_V2V_END": 0.03,
    "LOGIT_BIAS_LOCAL_ANNEAL_STEPS": 0,
    "LOGIT_BIAS_RSU_ANNEAL_STEPS": 0,
    "LOGIT_BIAS_V2V_ANNEAL_STEPS": 0,
    # PPO stability tightening (Fix-A)
    "TARGET_KL": 0.015,
    "TARGET_KL_STOP_MULT": 1.25,
    "PPO_EPOCH": 4,
    "MINI_BATCH_SIZE": 512,
    "USE_LR_DECAY": True,
    "LR_DECAY_STEPS": 100,
    "LR_DECAY_RATE": 0.95,
    # exploration schedule
    "ENTROPY_COEF_START": 0.0030,
    "ENTROPY_COEF_END": 0.0018,
    "ENTROPY_ANNEAL_STEPS": 120000,
}


def apply_final_paper_profile() -> None:
    for k, v in FINAL_ENV.items():
        setattr(Cfg, k, v)
    for k, v in FINAL_TRAIN.items():
        setattr(TC, k, v)

    if float(Cfg.RSU_RANGE) <= float(Cfg.V2V_RANGE):
        raise ValueError(
            f"Final profile invalid: RSU_RANGE({Cfg.RSU_RANGE}) must be > V2V_RANGE({Cfg.V2V_RANGE})."
        )
    if int(Cfg.V2V_NUM_RB) <= 0:
        raise ValueError(f"Final profile invalid: V2V_NUM_RB must be positive, got {Cfg.V2V_NUM_RB}")
    Cfg.V2V_BW_PER_RB = float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)
    if int(getattr(Cfg, "V2I_NUM_RB", 0)) <= 0:
        rb_bw = float(max(getattr(Cfg, "V2I_RB_BW_HZ", 180e3), 1.0))
        Cfg.V2I_NUM_RB = max(int(round(float(Cfg.BW_V2I) / rb_bw)), 1)
    Cfg.ALL_FEASIBLE = str(getattr(Cfg, "CANDIDATE_MODE", "TOPK")).upper() == "ALL"
