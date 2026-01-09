"""
CommWait 敏感性审计脚本

目的：
- 在不改动业务逻辑/模型结构的前提下，用真实 rollout 数据检验 CommWait 特征与回报/价值的相关性。
- 输出统计：ΔV 方向、Corr(CommWait, return)、线性回归系数。

运行示例：
    PYTHONPATH=. python scripts/audit/commwait_sensitivity.py
"""

import math
import numpy as np
import torch

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork


def discounted_returns(rewards, gamma):
    """计算每个时间步的折扣回报（按episode分段）。"""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))


def corr_with_p(x, y):
    """计算 Pearson r 及近似 p 值（无依赖版）。"""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0, 1.0
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    if n <= 2:
        return r, 1.0
    t_stat = r * math.sqrt((n - 2) / (1 - r * r + 1e-9))
    # 双侧p值近似
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / math.sqrt(2))))
    return float(r), float(p)


def run_rollouts(num_episodes=20, min_steps=10000, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CommWait特征已从观测中移除，本脚本不再适用
    raise SystemExit("CommWait features have been removed from observations; commwait_sensitivity script is obsolete.")

    gamma = TC.GAMMA if hasattr(TC, "GAMMA") else 0.99

    comm_means = []
    rewards_all = []
    values_all = []
    returns_all = []
    steps_collected = 0

    for ep in range(num_episodes):
        obs_list, _ = env.reset(seed=seed + ep)
        ep_rewards = []
        ep_comm = []
        ep_values = []

        while True:
            # 取当前obs的CommWait均值（末尾4维）
            comm_mean_step = []
            for obs in obs_list:
                comm = np.asarray(obs["resource_raw"], dtype=np.float32)[:, -4:]
                comm_mean_step.append(comm.mean())
            comm_mean_step = float(np.mean(comm_mean_step))

            # forward取得价值
            inputs = net.prepare_inputs(obs_list, device="cpu")
            with torch.no_grad():
                _, _, _, values = net.forward(**inputs)
                values_np = values.squeeze(-1).cpu().numpy()

            # 随机策略：用网络输出采样动作
            target_logits, alpha, beta, _ = net.forward(**inputs)
            target_dist = torch.distributions.Categorical(logits=target_logits)
            power_dist = torch.distributions.Beta(alpha.squeeze(-1), beta.squeeze(-1))
            targets = target_dist.sample().cpu().numpy()
            powers = power_dist.sample().cpu().numpy()
            actions = [{"target": int(t), "power": float(p)} for t, p in zip(targets, powers)]

            next_obs, rewards, terminated, truncated, _ = env.step(actions)

            ep_comm.append(comm_mean_step)
            ep_rewards.append(np.mean(rewards))
            ep_values.append(np.mean(values_np))

            steps_collected += 1
            obs_list = next_obs
            if terminated or truncated or steps_collected >= min_steps:
                # 结束时计算回报
                ep_returns = discounted_returns(ep_rewards, gamma)
                comm_means.extend(ep_comm)
                rewards_all.extend(ep_rewards)
                values_all.extend(ep_values)
                returns_all.extend(ep_returns)
                break
        if steps_collected >= min_steps:
            break

    env.close()
    return {
        "comm_mean": np.array(comm_means, dtype=np.float32),
        "rewards": np.array(rewards_all, dtype=np.float32),
        "values": np.array(values_all, dtype=np.float32),
        "returns": np.array(returns_all, dtype=np.float32),
    }


def linear_regression(x, y, controls=None):
    """
    最小二乘回归: y ~ a*x + controls
    返回 a 系数及最小二乘解。
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    X = x
    if controls is not None:
        ctrl_list = []
        for c in controls:
            c_arr = np.asarray(c).reshape(-1, 1)
            ctrl_list.append(c_arr)
        X = np.concatenate([x] + ctrl_list, axis=1)
    # 加1常数项
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a = float(coef[0])
    return a, coef.flatten()


def main():
    data = run_rollouts()
    comm = data["comm_mean"]
    ret = data["returns"]
    val = data["values"]

    # 相关性
    r_ret, p_ret = corr_with_p(comm, ret)
    # advantage-like: return - value
    adv = ret - val
    r_adv, p_adv = corr_with_p(comm, adv)

    # 简单线性回归，control 使用 time index
    time_idx = np.arange(len(comm)).reshape(-1, 1)
    a_coef, coefs = linear_regression(comm, ret, controls=[time_idx])

    print("=== CommWait Sensitivity (COMMWAIT_DIRECT_TO_CRITIC default={}) ===".format(
        getattr(TC, "COMMWAIT_DIRECT_TO_CRITIC", False)))
    print(f"N samples: {len(comm)}")
    print(f"Corr(comm_mean, return): r={r_ret:.4f}, p≈{p_ret:.4f}")
    print(f"Corr(comm_mean, return - value): r={r_adv:.4f}, p≈{p_adv:.4f}")
    print(f"Linear reg G_t ~ a*comm + b*time + c: a={a_coef:.4f}, coefs={coefs}")
    print("Stats comm_mean:", {
        "mean": float(comm.mean()),
        "median": float(np.median(comm)),
        "p10": float(np.percentile(comm, 10)),
        "p90": float(np.percentile(comm, 90)),
    })

    # 判定
    if p_ret < 0.05:
        direction = "positive" if r_ret > 0 else "negative"
        print(f"[判定] CommWait 与回报显著 {direction} 相关 (p<0.05)")
    else:
        print("[判定] CommWait 与回报相关性不显著")


if __name__ == "__main__":
    main()
