import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_mask_resource_alignment_strict():
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)

    for _ in range(10):
        for obs, v in zip(obs_list, env.vehicles):
            target_mask = obs["target_mask"]
            action_mask = obs["action_mask"]
            resource_ids = obs["resource_ids"]

            assert len(target_mask) == Cfg.MAX_TARGETS
            assert len(action_mask) == Cfg.MAX_TARGETS
            assert len(resource_ids) == Cfg.MAX_TARGETS
            assert np.array_equal(target_mask, action_mask)

            assert int(resource_ids[0]) == 1
            assert int(resource_ids[1]) == 2

            candidate_ids = env._last_candidates.get(v.id)
            assert candidate_ids is not None
            assert len(candidate_ids) == Cfg.MAX_NEIGHBORS

            for j in range(Cfg.MAX_NEIGHBORS):
                if candidate_ids[j] is not None and candidate_ids[j] >= 0:
                    assert target_mask[2 + j]
                    assert int(resource_ids[2 + j]) == 3 + int(candidate_ids[j])
                else:
                    assert not target_mask[2 + j]

        actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        obs_list, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break
