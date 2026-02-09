"""
[奖励函数] envs/rl/reward_functions.py
Unified Reward: 少项 + 非线性 + 无量纲 + 外部性内生化; PBRS 默认 ON

终局: succ  +R_s * ((Td-Tf)/Td)^p_s
      fail  -R_f * ((Tf-Td)/Td)^p_f

每步: -w_t*(dt/Td) - w_e*(E_tx/E_ref)^p_e - w_I*(I_caused/I_ref)^p_I
      - w_risk*(1-rho_target)^p_risk * 1_remote - w_ill*illegal

PBRS: r_pbrs = beta*(gamma*Phi(s') - Phi(s))
      Phi = -(eps + LB/Td)^q
      LB 只能用快照下界（不得用未来）

旧接口 compute_absolute_reward 保留兼容。
"""

import numpy as np
from configs.config import SystemConfig as Cfg


# ============================================================
# 旧接口（向后兼容，保留 LEGACY/PBRS_KP_V2 调用路径）
# ============================================================

def compute_absolute_reward(dT_rem, t_tx, power_ratio, dt, p_max_watt,
                           reward_min, reward_max,
                           hard_triggered=False, illegal_action=False):
    """旧 Delta-CFT 奖励（保持兼容）"""
    dT_clipped = float(np.clip(
        float(np.nan_to_num(dT_rem, nan=0.0, posinf=0.0, neginf=0.0)),
        Cfg.DELTA_CFT_CLIP_MIN, Cfg.DELTA_CFT_CLIP_MAX
    ))
    dT_eff = dT_clipped - float(dt)
    t_tx_clipped = float(np.clip(np.nan_to_num(t_tx, nan=0.0), 0.0, dt))

    p_watt = float(np.nan_to_num(p_max_watt, nan=0.0))
    p_circuit = float(getattr(Cfg, "P_CIRCUIT_WATT", 0.0))
    p_tx = float(np.nan_to_num(power_ratio, nan=0.0)) * p_watt
    e_step = (p_tx + p_circuit) * float(dt)
    e_max = max((p_watt + p_circuit) * float(dt), 1e-12)
    energy_norm = float(np.clip(e_step / e_max, 0.0, 1.0))

    if hard_triggered or illegal_action:
        reward = reward_min
    else:
        reward = (Cfg.DELTA_CFT_SCALE * dT_clipped -
                 Cfg.DELTA_CFT_ENERGY_WEIGHT * energy_norm)
    reward = float(np.clip(reward, reward_min, reward_max))

    return reward, {
        "dT": dT_clipped, "dT_eff": dT_eff,
        "energy_norm": energy_norm, "t_tx": t_tx_clipped, "dt_used": float(dt),
    }


# ============================================================
# 新统一奖励
# ============================================================

def compute_unified_step_reward(
    dt, Td, E_tx, I_caused, illegal,
    is_remote=False, rho_target=1.0,
    E_ref=None, I_ref=None,
):
    """
    每步奖励（不含终局和 PBRS）。

    Returns:
        (reward_step, info_dict)
    """
    if Td <= 0:
        Td = 1.0
    w_t = getattr(Cfg, 'W_TIME', 0.5)
    w_e = getattr(Cfg, 'W_ENERGY', 0.3)
    p_e = getattr(Cfg, 'P_ENERGY', 1.0)
    w_I = getattr(Cfg, 'W_INTERF', 0.2)
    p_I = getattr(Cfg, 'P_INTERF', 1.0)
    w_risk = getattr(Cfg, 'W_RISK', 0.0)
    p_risk = getattr(Cfg, 'P_RISK', 1.0)
    w_ill = getattr(Cfg, 'W_ILLEGAL', 2.0)

    if E_ref is None:
        E_ref = getattr(Cfg, 'E_REF_UNIFIED', Cfg.P_MAX_WATT * Cfg.DT)
    if I_ref is None:
        d0 = getattr(Cfg, 'I_REF_D0', Cfg.V2V_RANGE / 2.0)
        beta0 = 10 ** (getattr(Cfg, 'BETA_0_DB', -30) / 10.0)
        g_d0 = beta0 * (max(d0, 1.0) ** (-Cfg.PL_BETA_V2V))
        I_ref = Cfg.P_MAX_WATT * g_d0

    E_ref = max(E_ref, 1e-12)
    I_ref = max(I_ref, 1e-12)

    r_time = -w_t * (dt / Td)
    r_energy = -w_e * (max(E_tx, 0.0) / E_ref) ** p_e
    r_interf = -w_I * (max(I_caused, 0.0) / I_ref) ** p_I
    rho_target = float(np.clip(rho_target, 0.0, 1.0))
    r_risk = -w_risk * ((1.0 - rho_target) ** p_risk) if bool(is_remote) else 0.0
    r_illegal = -w_ill * float(illegal)

    r_step = r_time + r_energy + r_interf + r_risk + r_illegal

    return float(r_step), {
        'r_time': float(r_time),
        'r_energy': float(r_energy),
        'r_interf': float(r_interf),
        'r_risk': float(r_risk),
        'r_illegal': float(r_illegal),
        'E_tx': float(E_tx),
        'I_caused': float(I_caused),
        'rho_target': float(rho_target),
        'is_remote': bool(is_remote),
    }


def compute_unified_terminal_reward(success, Tf, Td):
    """
    终局奖励。

    Args:
        success: bool
        Tf: 实际完成时间
        Td: deadline

    Returns:
        (reward_terminal, info_dict)
    """
    if Td <= 0:
        Td = 1.0
    R_s = getattr(Cfg, 'R_SUCC', 10.0)
    R_f = getattr(Cfg, 'R_FAIL', 10.0)
    p_s = getattr(Cfg, 'P_SUCC', 1.0)
    p_f = getattr(Cfg, 'P_FAIL', 1.5)

    if success:
        margin = max((Td - Tf) / Td, 0.0)
        r_term = R_s * (margin ** p_s)
    else:
        overtime = max((Tf - Td) / Td, 0.0)
        r_term = -R_f * (overtime ** p_f)

    return float(r_term), {
        'success': bool(success),
        'Tf': float(Tf),
        'Td': float(Td),
        'r_terminal': float(r_term),
    }


def compute_unified_pbrs(phi_prev, phi_next, terminated=False):
    """
    PBRS: r = beta * (gamma * Phi(s') - Phi(s))

    当 terminated=True 时 Phi(s') = 0（吸收态）。

    Returns:
        float: PBRS 增量奖励
    """
    beta = getattr(Cfg, 'PBRS_BETA', 0.1)
    gamma = getattr(Cfg, 'PBRS_GAMMA', 0.99)
    clip = getattr(Cfg, 'PBRS_PHI_CLIP_UNIFIED', 5.0)

    if terminated:
        phi_next = 0.0
    delta = gamma * phi_next - phi_prev
    delta = float(np.clip(delta, -clip, clip))
    return beta * delta


def compute_phi_lb(LB, Td):
    """
    Phi = -(eps + LB/Td)^q

    Args:
        LB: 关键路径下界时间 (秒)
        Td: deadline (秒)

    Returns:
        float: 势值 (<=0)
    """
    if Td <= 0:
        Td = 1.0
    eps = getattr(Cfg, 'PBRS_EPS', 1e-3)
    q = getattr(Cfg, 'PBRS_Q', 0.5)
    clip = getattr(Cfg, 'PBRS_PHI_CLIP_UNIFIED', 5.0)
    phi = -((eps + max(LB, 0.0) / Td) ** q)
    return float(np.clip(phi, -clip, 0.0))


def clip_reward(reward, config=None):
    if config is None:
        config = Cfg
    return float(np.clip(reward, config.REWARD_MIN, config.REWARD_MAX))


__all__ = [
    'compute_absolute_reward', 'clip_reward',
    'compute_unified_step_reward', 'compute_unified_terminal_reward',
    'compute_unified_pbrs', 'compute_phi_lb',
]
