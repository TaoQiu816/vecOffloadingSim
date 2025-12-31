from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_canonical_order_stable_across_steps():
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)

    for _ in range(10):
        for obs in obs_list:
            target_mask = obs["target_mask"]
            resource_ids = obs["resource_ids"]
            assert len(target_mask) == Cfg.MAX_TARGETS
            assert len(resource_ids) == Cfg.MAX_TARGETS
            assert int(resource_ids[0]) == 1
            assert int(resource_ids[1]) == 2
            assert bool(target_mask[0]) is True

        actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        obs_list, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break
