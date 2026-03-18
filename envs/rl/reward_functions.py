import numpy as np

from configs.config import SystemConfig as Cfg


def is_benign_trust_mode(config=None):
    cfg = config or Cfg
    malicious_ratio = float(max(getattr(cfg, "MALICIOUS_RATIO", 0.0), 0.0))
    reliable_prob = float(np.clip(getattr(cfg, "TRUST_RELIABLE_PROB", 1.0), 0.0, 1.0))
    return bool(malicious_ratio <= 1e-12 and reliable_prob >= 1.0 - 1e-12)


def clip_reward(reward, config=None):
    cfg = config or Cfg
    return float(np.clip(float(reward), cfg.REWARD_MIN, cfg.REWARD_MAX))


def compute_progress_reward(phi_prev, phi_curr, t_norm=None, rmax=None):
    if t_norm is None:
        t_norm = float(max(getattr(Cfg, "REWARD_PROGRESS_TNORM", 1.0), 1e-6))
    if rmax is None:
        rmax = float(max(getattr(Cfg, "REWARD_PROGRESS_RMAX", 2.0), 1e-6))
    delta = (float(phi_prev) - float(phi_curr)) / t_norm
    return float(np.clip(delta, -rmax, rmax))


def compute_unified_step_reward(r_prog, illegal=False):
    r_illegal = -float(getattr(Cfg, "W_ILLEGAL", 30.0)) if illegal else 0.0
    r_step = float(r_prog) + float(r_illegal)
    return float(r_step), {
        "r_prog": float(r_prog),
        "r_illegal": float(r_illegal),
        "r_step": float(r_step),
    }


def compute_failure_severity(finish_time, deadline, remaining_ratio=1.0, unrecoverable=False):
    if deadline <= 0:
        deadline = 1.0
    miss_ratio = max((float(finish_time) - float(deadline)) / float(deadline), 0.0)
    severity = max(float(remaining_ratio), miss_ratio)
    if unrecoverable:
        severity = max(severity, 1.0)
    return float(np.clip(severity, 0.0, float(getattr(Cfg, "MISS_CAP", 2.0))))


def compute_unified_terminal_reward(success, finish_time, deadline, severity_fail=0.0):
    """
    计算终止奖励，重标定到[-2, 2]范围
    - Success: [1.0, 2.0] = 1.0 + early_ratio
    - Miss: [-2.0, -1.0] = -(1.0 + miss_ratio)
    - Fail: [-2.0, -1.0] = -(1.0 + fail_ratio)
    """
    if deadline <= 0:
        deadline = 1.0
    finish_time = float(max(finish_time, 0.0))
    deadline = float(deadline)
    
    if success and finish_time <= deadline:
        # 成功且未超时：[1.0, 2.0]
        early_ratio = np.clip((deadline - finish_time) / deadline, 0.0, 1.0)
        r_term = 1.0 + early_ratio
        term_type = "success"
    elif finish_time > deadline:
        # 超时：[-2.0, -1.0]
        miss_ratio = np.clip((finish_time - deadline) / deadline, 0.0, 1.0)
        r_term = -(1.0 + miss_ratio)
        term_type = "miss"
    else:
        # 失败：[-2.0, -1.0]
        fail_ratio = np.clip(float(severity_fail), 0.0, 1.0)
        r_term = -(1.0 + fail_ratio)
        term_type = "fail"

    return float(r_term), {
        "r_term": float(r_term),
        "finish_time": finish_time,
        "deadline": deadline,
        "term_type": term_type,
    }


def compute_cost_power(tx_energy, dt=None, e_ref=None):
    if dt is None:
        dt = float(max(getattr(Cfg, "DT", 0.1), 1e-6))
    if e_ref is None:
        e_ref = float(max(getattr(Cfg, "E_REF_UNIFIED", 1.0), 1e-9))
    return float(np.clip(float(max(tx_energy, 0.0)) / e_ref, 0.0, 10.0))


def compute_cost_trust(trust_lcb, config=None):
    if is_benign_trust_mode(config=config):
        return 0.0
    return float(np.clip(1.0 - float(np.clip(trust_lcb, 0.0, 1.0)), 0.0, 1.0))


__all__ = [
    "clip_reward",
    "compute_progress_reward",
    "compute_unified_step_reward",
    "compute_failure_severity",
    "compute_unified_terminal_reward",
    "compute_cost_power",
    "compute_cost_trust",
    "is_benign_trust_mode",
]
