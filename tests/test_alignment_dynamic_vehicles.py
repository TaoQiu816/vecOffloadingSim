import os
import numpy as np

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg


def test_actions_obs_vehicle_alignment_dynamic(monkeypatch):
    monkeypatch.setattr(Cfg, "VEHICLE_ARRIVAL_RATE", 2.0, raising=False)
    monkeypatch.setenv("AUDIT_ASSERTS", "1")

    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)

    for _ in range(5):
        prev_count = len(env.vehicles)
        actions = []
        for obs in obs_list:
            valid = np.where(obs["action_mask"])[0]
            target = int(valid[0]) if len(valid) > 0 else 0
            actions.append({"target": target, "power": 0.5})
        assert len(actions) == prev_count
        obs_list, _, _, _, _ = env.step(actions)
        assert len(obs_list) == len(env.vehicles)
