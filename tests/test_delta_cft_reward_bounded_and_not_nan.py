import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from baselines.random_policy import RandomPolicy


def test_delta_cft_reward_bounded_and_not_nan():
    orig_mode = Cfg.REWARD_MODE
    try:
        Cfg.REWARD_MODE = "delta_cft"
        env = VecOffloadingEnv()
        policy = RandomPolicy(seed=0)
        obs_list, _ = env.reset(seed=0)
        for _ in range(20):
            actions = policy.select_action(obs_list)
            for i, act in enumerate(actions):
                if not act:
                    continue
                if "obs_stamp" in obs_list[i] and "obs_stamp" not in act:
                    act["obs_stamp"] = int(obs_list[i]["obs_stamp"])
            obs_list, rewards, terminated, truncated, _ = env.step(actions)
            for r in rewards:
                assert np.isfinite(r)
                assert Cfg.REWARD_MIN - 1e-6 <= r <= Cfg.REWARD_MAX + 1e-6
            if terminated or truncated:
                break
    finally:
        Cfg.REWARD_MODE = orig_mode
