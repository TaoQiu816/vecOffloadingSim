import numpy as np

from envs.vec_offloading_env import VecOffloadingEnv


def test_v2v_target_maps_to_candidate():
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)

    # find a vehicle with available V2V target
    chosen = None
    for i, obs in enumerate(obs_list):
        v2v_indices = np.where(obs["action_mask"][2:])[0]
        if len(v2v_indices) > 0:
            chosen = (i, int(v2v_indices[0] + 2))
            break

    if chosen is None:
        return  # no V2V available in this seed; skip silently

    i, target_idx = chosen
    v = env.vehicles[i]
    candidate_ids = env._last_candidates.get(v.id, [])
    expected = candidate_ids[target_idx - 2]

    actions = [None for _ in obs_list]
    actions[i] = {"target": int(target_idx), "power": 0.5}
    env.step(actions)

    assert isinstance(v.curr_target, int)
    assert v.curr_target == expected
