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
    "NUM_RSU": 3,
    "CANDIDATE_MODE": "ALL",
    "RSU_RANGE": 350.0,
    "V2V_RANGE": 250.0,
    "V2V_NUM_RB": 3,
    # 轻度V2I拥塞等效，避免RSU一统但不做极限压力
    "BW_V2I": 17.0e6,
    # 保持与当前主线一致的数据规模
    "MIN_DATA": 8.0e5,
    "MAX_DATA": 3.2e6,
    "MIN_EDGE_DATA": 4.0e5,
    "MAX_EDGE_DATA": 1.6e6,
    "DEADLINE_SLACK_SECONDS": 0.0,
    # 风险可学习性增强（对应当前TrustManager字段）
    "TRUST_RELIABLE_PROB": 0.6,
    "TRUST_PRIOR_A": 2.0,
    "TRUST_PRIOR_B": 2.0,
    # StageC以风险为主，不让干扰项过强
    "W_INTERF": 0.30,
    "W_TIME": 0.30,
    "W_RISK": 0.60,
}

STAGE_C_TRAIN = {
    "LOGIT_BIAS_RSU": 0.0,
    "LOGIT_BIAS_V2V_INIT": 0.40,
    "LOGIT_BIAS_V2V_ANNEAL_STEPS": 90000,
    "ENTROPY_COEF_START": 0.0038,
    "ENTROPY_COEF_END": 0.001,
    "ENTROPY_ANNEAL_STEPS": 90000,
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
    Cfg.ALL_FEASIBLE = str(getattr(Cfg, "CANDIDATE_MODE", "TOPK")).upper() == "ALL"
