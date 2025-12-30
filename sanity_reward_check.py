import random
import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def run_once(reward_mode="incremental_cost", steps=200, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    Cfg.REWARD_MODE = reward_mode
    env = VecOffloadingEnv()
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False

    for _ in range(steps):
        actions = []
        for ob in obs:
            mask = ob["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                actions.append(None)
                continue
            target = int(np.random.choice(valid))
            power = float(np.random.uniform(0.0, 1.0))
            actions.append({"target": target, "power": power})
        obs, rewards, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break

    if not (terminated or truncated):
        truncated = True
    env._log_episode_stats(terminated, truncated)


if __name__ == "__main__":
    run_once("incremental_cost", steps=200, seed=0)
    run_once("delta_cft", steps=200, seed=1)
