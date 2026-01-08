"""
[奖励函数] envs/rl/reward_functions.py
Reward Functions for VEC Offloading

作用 (Purpose):
    提供纯函数形式的奖励计算逻辑，包括：
    - 绝对潜在值奖励（Delta CFT）
    - 能耗惩罚计算

设计原则 (Design Principles):
    - 纯函数，无副作用
    - 所有参数显式传入
    - 支持配置驱动的权重调整
"""

import numpy as np
from configs.config import SystemConfig as Cfg


def compute_absolute_reward(dT_rem, t_tx, power_ratio, dt, p_max_watt,
                           reward_min, reward_max,
                           hard_triggered=False, illegal_action=False):
    """
    绝对潜在值奖励：仅依赖剩余时间差与动作功率，便于权重外调。

    能耗计算说明：
    - 当前实现：E = (p_tx + p_circuit) * dt （传输能耗）
    - 计算能耗：E_comp = K * f^2 * cycles （未显式计入）
    - 设计理由：
      1. 对于单车辆offloading决策，传输能耗是主要因素
      2. Local执行无传输能耗，只有固定计算能耗
      3. RSU/V2V执行的远程计算能耗由远程节点承担
    - 扩展方向：如需系统级能耗优化，可传入comp_cycles和cpu_freq参数

    Args:
        dT_rem: 剩余时间变化量（秒）
        t_tx: 传输时间（秒）
        power_ratio: 功率比例 [0, 1]
        dt: 时间步长（秒）
        p_max_watt: 最大功率（瓦）
        reward_min: 奖励下限
        reward_max: 奖励上限
        hard_triggered: 是否触发硬约束
        illegal_action: 是否为非法动作

    Returns:
        tuple: (reward, info_dict)
            - reward: 计算得到的奖励值
            - info_dict: 包含中间计算结果的字典
    """
    # 安全处理NaN和Inf
    dT_clipped = float(np.clip(
        float(np.nan_to_num(dT_rem, nan=0.0, posinf=0.0, neginf=0.0)),
        Cfg.DELTA_CFT_CLIP_MIN,
        Cfg.DELTA_CFT_CLIP_MAX
    ))
    dT_eff = dT_clipped - float(dt)
    t_tx_clipped = float(np.clip(
        np.nan_to_num(t_tx, nan=0.0, posinf=0.0, neginf=0.0),
        0.0, dt
    ))

    # 功率计算
    p_watt = float(np.nan_to_num(p_max_watt, nan=0.0, posinf=0.0, neginf=0.0))
    p_circuit = float(getattr(Cfg, "P_CIRCUIT_WATT", 0.0))
    p_tx = float(np.nan_to_num(power_ratio, nan=0.0, posinf=0.0, neginf=0.0)) * p_watt

    # 能耗组成：传输能耗（当前）+ 计算能耗（可扩展）
    e_step = (p_tx + p_circuit) * float(dt)
    e_max = max((p_watt + p_circuit) * float(dt), 1e-12)
    energy_norm = float(np.clip(e_step / e_max, 0.0, 1.0))

    # 线性组合：时间收益 - 能耗惩罚；权重完全由配置驱动
    if hard_triggered or illegal_action:
        reward = reward_min
    else:
        reward = (Cfg.DELTA_CFT_SCALE * dT_clipped -
                 Cfg.DELTA_CFT_ENERGY_WEIGHT * energy_norm)

    reward = float(np.clip(reward, reward_min, reward_max))

    return reward, {
        "dT": dT_clipped,
        "dT_eff": dT_eff,
        "energy_norm": energy_norm,
        "t_tx": t_tx_clipped,
        "dt_used": float(dt),
    }


def clip_reward(reward, config=None):
    """
    裁剪奖励到配置范围内

    Args:
        reward: 原始奖励值
        config: 配置对象（默认使用Cfg）

    Returns:
        float: 裁剪后的奖励值
    """
    if config is None:
        config = Cfg
    return float(np.clip(reward, config.REWARD_MIN, config.REWARD_MAX))


# 导出列表
__all__ = ['compute_absolute_reward', 'clip_reward']
