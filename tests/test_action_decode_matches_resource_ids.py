from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_action_decode_matches_resource_ids():
    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)

    found = False
    max_attempts = 200
    episode_seed = 1

    for _ in range(max_attempts):
        chosen = None
        for i, obs in enumerate(obs_list):
            if obs.get("subtask_index", -1) is None or obs.get("subtask_index", -1) < 0:
                continue
            mask = obs["target_mask"]
            for j in range(Cfg.MAX_NEIGHBORS):
                tgt = 2 + j
                if tgt < len(mask) and mask[tgt]:
                    token = int(obs["resource_ids"][tgt])
                    if token >= 3:
                        chosen = (i, env.vehicles[i].id, tgt, token)
                        break
            if chosen:
                break

        if chosen:
            idx, veh_id, tgt, token = chosen
            actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
            actions[idx] = {"target": tgt, "power": 1.0}
            obs_list, _, terminated, truncated, _ = env.step(actions)

            veh = env._get_vehicle_by_id(veh_id)
            if veh is not None and not getattr(veh, "illegal_action", False):
                assert isinstance(veh.curr_target, int)
                assert veh.curr_target == token - 3
                found = True
                break
        else:
            actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
            obs_list, _, terminated, truncated, _ = env.step(actions)

        if terminated or truncated:
            obs_list, _ = env.reset(seed=episode_seed)
            episode_seed += 1

    assert found, "no valid V2V candidate found to validate decode mapping"
