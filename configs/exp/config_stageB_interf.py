"""
StageB profile: interference/power visibility under realistic V2X ranges.

Scope:
- Parameter-only overlay (no dynamics formula changes).
- Enforces RSU_RANGE > V2V_RANGE.
"""

from __future__ import annotations

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC


STAGE_B_ENV = {
    "NUM_VEHICLES": 20,
    "MAX_STEPS": 200,
    "NUM_RSU": 2,
    "CANDIDATE_MODE": "ALL",
    "RSU_RANGE": 350.0,
    "V2V_RANGE": 250.0,
    "V2V_NUM_RB": 2,
    # 双链路统一为RB-SINR口径（V2I含跨RSU ICI）
    "BW_V2I": 16.0e6,
    "V2I_RATE_MODEL": "RB_SINR",
    "V2I_ICI_ENABLED": True,
    "V2I_FREQ_REUSE_FACTOR": 1,
    "V2I_NUM_RB": 90,
    # DAG workload calibration for MAX_STEPS=200 (TH=20s):
    # keep interference scenario non-trivial while avoiding all-fail attractor.
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
    # deadline = alpha*Tmin + slack
    "DEADLINE_MODE": "LB_ALPHA",
    "DEADLINE_ALPHA_MIN": 25.0,
    "DEADLINE_ALPHA_MAX": 35.0,
    "DEADLINE_SLACK_SECONDS": 0.60,
    # StageB 目标是“可完成前提下的干扰权衡”，避免干扰项压制全部目标
    "W_INTERF": 0.10,
    "W_TIME": 0.40,
    "W_RISK": 0.28,
    "R_SUCC": 16.0,
    "R_FAIL": 16.0,
}

STAGE_B_TRAIN = {
    "MAX_STEPS": 200,
    # 仅最小探索适配，不改算法结构
    "LOGIT_BIAS_RSU": 0.0,
    # 线性退火到0的实现下：通过“更大anneal_steps”维持中后段V2V探索
    # 600ep(12万步)时 bias≈0.24；800ep(16万步)时 bias≈0.12
    "LOGIT_BIAS_V2V_INIT": 0.18,
    "LOGIT_BIAS_V2V_END": 0.10,
    "LOGIT_BIAS_V2V_ANNEAL_STEPS": 96000,
    "ENTROPY_COEF_START": 0.0036,
    "ENTROPY_COEF_END": 0.001,
    "ENTROPY_ANNEAL_STEPS": 200000,
    # CTDE + CMDP
    "USE_SIMPLIFIED_CRITIC": False,
    "COMMWAIT_DIRECT_TO_CRITIC": True,
    "CTDE_GLOBAL_DIM": 12,
    "CMDP_ENABLE": True,
    "CMDP_LAMBDA_LR": 0.02,
    "CMDP_BUDGET_ENERGY": 0.25,
    "CMDP_BUDGET_INTERF": 0.08,
    "CMDP_BUDGET_RISK": 0.40,
}


def apply_stage_b_profile() -> None:
    for k, v in STAGE_B_ENV.items():
        setattr(Cfg, k, v)
    for k, v in STAGE_B_TRAIN.items():
        setattr(TC, k, v)

    if float(Cfg.RSU_RANGE) <= float(Cfg.V2V_RANGE):
        raise ValueError(
            f"StageB invalid: RSU_RANGE({Cfg.RSU_RANGE}) must be > V2V_RANGE({Cfg.V2V_RANGE})."
        )

    # Keep derived bandwidth split consistent with RB count.
    if int(Cfg.V2V_NUM_RB) <= 0:
        raise ValueError(f"StageB invalid: V2V_NUM_RB must be positive, got {Cfg.V2V_NUM_RB}")
    Cfg.V2V_BW_PER_RB = float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)
    if int(getattr(Cfg, "V2I_NUM_RB", 0)) <= 0:
        Cfg.V2I_NUM_RB = max(int(round(float(Cfg.BW_V2I) / float(max(getattr(Cfg, "V2I_RB_BW_HZ", 180e3), 1.0)))), 1)
    Cfg.ALL_FEASIBLE = str(getattr(Cfg, "CANDIDATE_MODE", "TOPK")).upper() == "ALL"
