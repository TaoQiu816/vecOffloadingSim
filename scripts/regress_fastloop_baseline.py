import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import SystemConfig as BaseCfg
from envs.vec_offloading_env import VecOffloadingEnv


class Cfg(BaseCfg):
    CHAIN_ENABLED = False
    CHAIN_RISK_WEIGHT_DEPOSIT = 0.0
    CHAIN_RISK_WEIGHT_FAIL = 0.0
    REWARD_SCHEME = "LEGACY_CFT"


def main():
    env = VecOffloadingEnv(config=Cfg)
    obs, _ = env.reset(seed=123)
    env.action_space.seed(999)
    actions = [env.action_space.sample() for _ in range(10)]
    rewards = []
    risk_costs = []
    tx_steps = []
    for step_idx, action in enumerate(actions):
        obs, rew, term, trunc, info = env.step(action)
        rewards.append(rew)
        risk_costs.append(info.get("risk_cost_sum", 0.0))
        tx_steps.append(info.get("tx_arrivals_step", 0))
        print(
            f"step={step_idx} reward={np.round(rew, 6).tolist()} "
            f"risk_cost_sum={risk_costs[-1]} tx_arrivals_step={tx_steps[-1]}"
        )
    print("rewards_seq=", rewards)
    print("risk_cost_seq=", risk_costs)
    print("tx_arrivals_seq=", tx_steps)


if __name__ == "__main__":
    main()
