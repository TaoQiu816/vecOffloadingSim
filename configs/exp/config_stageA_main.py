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
    # DAG workload calibration for MAX_STEPS=200 (TH=20s):
    # - nodes: 8~16
    # - deadline mainly in 0.1~1.0s (vehicular latency regime)
    "MIN_NODES": 8,
    "MAX_NODES": 16,
    "DAG_FAT": 1.00,
    "DAG_DENSITY": 0.24,
    "MIN_COMP": 1.0e7,
    "MAX_COMP": 1.0e8,
    "MIN_DATA": 5.0e4,
    "MAX_DATA": 4.0e6,
    "MIN_EDGE_DATA": 2.0e4,
    "MAX_EDGE_DATA": 1.5e6,
    # deadline = alpha*Tmin + slack (keep feasible but non-trivial)
    "DEADLINE_MODE": "LB_ALPHA",
    "DEADLINE_ALPHA_MIN": 25.0,
    "DEADLINE_ALPHA_MAX": 35.0,
    "DEADLINE_SLACK_SECONDS": 0.60,
    # Reward main objective stays completion/latency.
    "W_INTERF": 0.12,
    "W_TIME": 0.38,
    "W_RISK": 0.28,
    "R_SUCC": 16.0,
    "R_FAIL": 16.0,
}

STAGE_A_TRAIN = {
    "MAX_STEPS": 200,
    "LOGIT_BIAS_RSU": 0.0,
    "LOGIT_BIAS_V2V_INIT": 0.14,
    "LOGIT_BIAS_V2V_END": 0.08,
    "LOGIT_BIAS_V2V_ANNEAL_STEPS": 96000,
    "ENTROPY_COEF_START": 0.0035,
    "ENTROPY_COEF_END": 0.001,
    "ENTROPY_ANNEAL_STEPS": 180000,
    # CTDE + CMDP
    "USE_SIMPLIFIED_CRITIC": False,
    "COMMWAIT_DIRECT_TO_CRITIC": True,
    "CTDE_GLOBAL_DIM": 12,
    "CMDP_ENABLE": True,
    "CMDP_LAMBDA_LR": 0.02,
    "CMDP_BUDGET_ENERGY": 0.22,
    "CMDP_BUDGET_INTERF": 0.05,
    "CMDP_BUDGET_RISK": 0.35,
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
