"""
StageC profile: trust/risk visibility with stronger reliability heterogeneity.

Scope:
- Parameter-only overlay (no dynamics formula changes).
- Enforces RSU_RANGE > V2V_RANGE.
"""

from __future__ import annotations

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


STAGE_C_ENV = {
    "NUM_VEHICLES": 20,
    "MAX_STEPS": 200,
    "NUM_RSU": 3,
    "CANDIDATE_MODE": "ALL",
    "RSU_RANGE": 350.0,
    "V2V_RANGE": 250.0,
    "V2V_NUM_RB": 3,
    # 双链路统一为RB-SINR口径（StageC干扰压力适中）
    "BW_V2I": 18.0e6,
    "V2I_RATE_MODEL": "RB_SINR",
    "V2I_ICI_ENABLED": True,
    "V2I_FREQ_REUSE_FACTOR": 2,
    "V2I_NUM_RB": 100,
    # Route-1: fixed one DAG per vehicle per episode (no mid-episode arrivals)
    "VEHICLE_ARRIVAL_RATE": 0.0,
    # DAG workload calibration for MAX_STEPS=200 (TH=20s):
    # two-class workload mixture + DT=0.1 compatible deadlines.
    "MIN_NODES": 8,
    "MAX_NODES": 16,
    "DAG_FAT": 1.00,
    "DAG_DENSITY": 0.24,
    "MIN_COMP": 1.0e8,
    "MAX_COMP": 4.0e9,
    "MIN_DATA": 1.0e5,
    "MAX_DATA": 1.0e7,
    "MIN_EDGE_DATA": 5.0e4,
    "MAX_EDGE_DATA": 2.0e6,
    "TASK_CLASS_MIX_ENABLE": True,
    "TASK_CLASS_B_PROB": 0.15,
    "TASK_A_TOTAL_DATA_MIN": 3.0e5,
    "TASK_A_TOTAL_DATA_MAX": 1.2e7,
    "TASK_A_TOTAL_COMP_MIN": 1.0e9,
    "TASK_A_TOTAL_COMP_MAX": 4.0e9,
    "TASK_A_DEADLINE_MIN": 0.5,
    "TASK_A_DEADLINE_MAX": 1.5,
    "TASK_B_TOTAL_DATA_MIN": 1.2e7,
    "TASK_B_TOTAL_DATA_MAX": 4.5e7,
    "TASK_B_TOTAL_COMP_MIN": 3.0e9,
    "TASK_B_TOTAL_COMP_MAX": 8.0e9,
    "TASK_B_DEADLINE_MIN": 1.2,
    "TASK_B_DEADLINE_MAX": 3.0,
    "TASK_COMP_SPLIT_JITTER": 0.6,
    "TASK_DATA_SPLIT_JITTER": 0.8,
    # fallback mode params (legacy compatibility)
    "DEADLINE_MODE": "LB_ALPHA",
    "DEADLINE_ALPHA_MIN": 2.0,
    "DEADLINE_ALPHA_MAX": 4.0,
    "DEADLINE_SLACK_SECONDS": 0.20,
    # 风险可学习性增强（对应当前TrustManager字段）
    "TRUST_RELIABLE_PROB": 0.6,
    "TRUST_PRIOR_A": 2.0,
    "TRUST_PRIOR_B": 2.0,
    # 论文统一奖励（与StageA/StageB保持同一套）
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

STAGE_C_TRAIN = {
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
    "ENTROPY_COEF_START": 0.0034,
    "ENTROPY_COEF_END": 0.001,
    "ENTROPY_ANNEAL_STEPS": 160000,
    # CTDE fixed + CMDP off (single paper algorithm)
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


def apply_stage_c_profile() -> None:
    for k, v in STAGE_C_ENV.items():
        setattr(Cfg, k, v)
    # 可靠性范围拉开到约[0.65, 0.99]
    Cfg.TRUST_P_RELIABLE_RANGE = (0.88, 0.99)
    Cfg.TRUST_P_UNRELIABLE_RANGE = (0.65, 0.82)
    for k, v in STAGE_C_TRAIN.items():
        setattr(TC, k, v)

    if float(Cfg.RSU_RANGE) <= float(Cfg.V2V_RANGE):
        raise ValueError(
            f"StageC invalid: RSU_RANGE({Cfg.RSU_RANGE}) must be > V2V_RANGE({Cfg.V2V_RANGE})."
        )

    if int(Cfg.V2V_NUM_RB) <= 0:
        raise ValueError(f"StageC invalid: V2V_NUM_RB must be positive, got {Cfg.V2V_NUM_RB}")
    Cfg.V2V_BW_PER_RB = float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)
    if int(getattr(Cfg, "V2I_NUM_RB", 0)) <= 0:
        Cfg.V2I_NUM_RB = max(int(round(float(Cfg.BW_V2I) / float(max(getattr(Cfg, "V2I_RB_BW_HZ", 180e3), 1.0)))), 1)
    Cfg.ALL_FEASIBLE = str(getattr(Cfg, "CANDIDATE_MODE", "TOPK")).upper() == "ALL"
