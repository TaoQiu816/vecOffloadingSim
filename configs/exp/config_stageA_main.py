"""
StageA profile: main-paper training with feasible mixed-link regime.

Scope:
- Parameter-only overlay (no queue/SINR dynamics changes).
- Enforces RSU_RANGE > V2V_RANGE.
"""

from __future__ import annotations

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


STAGE_A_ENV = {
    "NUM_VEHICLES": 20,
    "MAX_STEPS": 200,
    "NUM_RSU": 3,
    "CANDIDATE_MODE": "ALL",
    "RSU_RANGE": 350.0,
    "V2V_RANGE": 250.0,
    "V2V_NUM_RB": 3,
    "BW_V2I": 20.0e6,
    "V2I_RATE_MODEL": "RB_SINR",
    "V2I_ICI_ENABLED": True,
    "V2I_FREQ_REUSE_FACTOR": 2,
    "V2I_NUM_RB": 100,
    # Route-1: fixed one DAG per vehicle per episode (no mid-episode arrivals)
    "VEHICLE_ARRIVAL_RATE": 0.0,
    # Tighten local CPU envelope to avoid trivial pure-local dominance.
    "MIN_VEHICLE_CPU_FREQ": 1.5e9,
    "MAX_VEHICLE_CPU_FREQ": 4.0e9,
    # DAG workload calibration for MAX_STEPS=200 (TH=20s):
    # - two-class workload mixture for feasible local/remote crossover
    # - deadline mainly in 0.4~2.5s (DT=0.1 compatible)
    "MIN_NODES": 8,
    "MAX_NODES": 16,
    "DAG_FAT": 1.00,
    "DAG_DENSITY": 0.24,
    "MIN_COMP": 4.0e8,
    "MAX_COMP": 6.0e9,
    "MIN_DATA": 5.0e5,
    "MAX_DATA": 2.5e7,
    "MIN_EDGE_DATA": 5.0e4,
    "MAX_EDGE_DATA": 2.0e6,
    "TASK_CLASS_MIX_ENABLE": True,
    "TASK_CLASS_B_PROB": 0.10,
    "TASK_A_TOTAL_DATA_MIN": 5.0e5,
    "TASK_A_TOTAL_DATA_MAX": 6.0e6,
    "TASK_A_TOTAL_COMP_MIN": 4.0e8,
    "TASK_A_TOTAL_COMP_MAX": 1.8e9,
    "TASK_A_DEADLINE_MIN": 0.5,
    "TASK_A_DEADLINE_MAX": 1.8,
    "TASK_B_TOTAL_DATA_MIN": 6.0e6,
    "TASK_B_TOTAL_DATA_MAX": 6.0e6,
    "TASK_B_TOTAL_COMP_MIN": 1.5e9,
    "TASK_B_TOTAL_COMP_MAX": 1.0e10,
    "TASK_B_DEADLINE_MIN": 1.5,
    "TASK_B_DEADLINE_MAX": 3.5,
    "TASK_COMP_SPLIT_JITTER": 0.6,
    "TASK_DATA_SPLIT_JITTER": 0.8,
    # Trust hidden reliability by node type (sampling only; Beta update unchanged).
    "TRUST_P_RSU_RANGE": (0.95, 0.995),
    "TRUST_P_VEH_RANGE": (0.60, 0.95),
    # fallback mode params (legacy compatibility)
    "DEADLINE_MODE": "LB_ALPHA",
    "DEADLINE_ALPHA_MIN": 2.0,
    "DEADLINE_ALPHA_MAX": 4.0,
    "DEADLINE_SLACK_SECONDS": 0.20,
    # Unified paper reward (single reward definition for training/baselines comparison).
    "W_TIME": 0.35,
    "W_ENERGY": 0.05,
    "W_INTERF": 0.03,
    "W_RISK": 0.04,
    "W_ILLEGAL": 30.0,
    "R_SUCC": 50.0,
    "R_FAIL": 50.0,
    "P_SUCC": 1.0,
    "P_FAIL": 1.0,
    "INTERF_RATIO_CLIP_UNIFIED": 3.0,
}

STAGE_A_TRAIN = {
    "MAX_STEPS": 200,
    "LOGIT_BIAS_RSU": 0.05,
    "LOGIT_BIAS_LOCAL": 0.20,
    "LOGIT_BIAS_LOCAL_INIT": 0.20,
    "LOGIT_BIAS_LOCAL_END": 0.0,
    "LOGIT_BIAS_LOCAL_ANNEAL_STEPS": 0,
    "LOGIT_BIAS_RSU_INIT": 0.05,
    "LOGIT_BIAS_RSU_END": 0.0,
    "LOGIT_BIAS_RSU_ANNEAL_STEPS": 0,
    "LOGIT_BIAS_V2V_INIT": 0.10,
    "LOGIT_BIAS_V2V_END": 0.0,
    "LOGIT_BIAS_V2V_ANNEAL_STEPS": 0,
    "BIAS_ANNEAL_FRAC": 0.50,
    "ENTROPY_COEF_START": 0.0020,
    "ENTROPY_COEF_END": 0.0012,
    "ENTROPY_ANNEAL_STEPS": 240000,
    # CTDE on, CMDP off (mainline reward-only training).
    "USE_SIMPLIFIED_CRITIC": False,
    "COMMWAIT_DIRECT_TO_CRITIC": True,
    "CTDE_GLOBAL_DIM": 30,
    "CMDP_ENABLE": False,
    "CMDP_LAMBDA_LR": 0.0,
    "CMDP_LAMBDA_MAX": 0.0,
    "CMDP_WARMUP_EPISODES": 0,
    "CMDP_BUDGET_ENERGY": 1.0e9,
    "CMDP_BUDGET_INTERF": 1.0e9,
    "CMDP_BUDGET_RISK": 1.0e9,
}


def apply_stage_a_profile() -> None:
    for k, v in STAGE_A_ENV.items():
        setattr(Cfg, k, v)
    for k, v in STAGE_A_TRAIN.items():
        setattr(TC, k, v)

    if float(Cfg.RSU_RANGE) <= float(Cfg.V2V_RANGE):
        raise ValueError(
            f"StageA invalid: RSU_RANGE({Cfg.RSU_RANGE}) must be > V2V_RANGE({Cfg.V2V_RANGE})."
        )

    if int(Cfg.V2V_NUM_RB) <= 0:
        raise ValueError(f"StageA invalid: V2V_NUM_RB must be positive, got {Cfg.V2V_NUM_RB}")
    Cfg.V2V_BW_PER_RB = float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)
    if int(getattr(Cfg, "V2I_NUM_RB", 0)) <= 0:
        Cfg.V2I_NUM_RB = max(int(round(float(Cfg.BW_V2I) / float(max(getattr(Cfg, "V2I_RB_BW_HZ", 180e3), 1.0)))), 1)
    Cfg.ALL_FEASIBLE = str(getattr(Cfg, "CANDIDATE_MODE", "TOPK")).upper() == "ALL"
