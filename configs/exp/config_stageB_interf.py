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
    "NUM_RSU": 2,
    "CANDIDATE_MODE": "ALL",
    "RSU_RANGE": 350.0,
    "V2V_RANGE": 250.0,
    "V2V_NUM_RB": 2,
    # V2I拥塞等效：相较默认20MHz下调约10%
    "BW_V2I": 18.0e6,
    # 负载与主线持平，压力主要来自 RSU数下降 + RB收紧
    "MIN_DATA": 8.0e5,
    "MAX_DATA": 3.2e6,
    "MIN_EDGE_DATA": 4.0e5,
    "MAX_EDGE_DATA": 1.6e6,
    # 适度放宽，避免“全失败”极端
    "DEADLINE_SLACK_SECONDS": 0.8,
    # 抑制干扰项单独主导，保持时延目标主导性
    "W_INTERF": 0.35,
    "W_TIME": 0.30,
}

STAGE_B_TRAIN = {
    # 仅最小探索适配，不改算法结构
    "LOGIT_BIAS_RSU": 0.0,
    "LOGIT_BIAS_V2V_INIT": 0.45,
    "LOGIT_BIAS_V2V_ANNEAL_STEPS": 100000,
    "ENTROPY_COEF_START": 0.0038,
    "ENTROPY_COEF_END": 0.001,
    "ENTROPY_ANNEAL_STEPS": 100000,
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
    Cfg.ALL_FEASIBLE = str(getattr(Cfg, "CANDIDATE_MODE", "TOPK")).upper() == "ALL"
