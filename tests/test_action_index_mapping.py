import numpy as np

from envs.vec_offloading_env import VecOffloadingEnv
from utils.data_utils import process_env_obs


def test_action_mask_and_resource_ids_alignment():
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)

    dag_list, _, _ = process_env_obs(obs_list, device="cpu")
    obs0 = obs_list[0]
    dag0 = dag_list[0]

    assert dag0.target_mask.shape[1] == obs0["target_mask"].shape[0]
    assert np.array_equal(dag0.target_mask[0].numpy(), obs0["target_mask"])

    candidate_ids = env._last_candidates.get(env.vehicles[0].id, [])
    resource_ids = obs0["resource_ids"]
    for idx, candidate_id in enumerate(candidate_ids):
        expected = 0 if candidate_id < 0 else 3 + candidate_id
        assert resource_ids[2 + idx] == expected
