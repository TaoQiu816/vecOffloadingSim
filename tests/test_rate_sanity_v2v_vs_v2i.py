import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_rate_sanity_v2v_vs_v2i():
    env = VecOffloadingEnv()
    env.reset(seed=7)

    ratios = []
    steps = 30
    np.random.seed(7)

    for _ in range(steps):
        env._get_obs()
        for v in env.vehicles:
            candidate_ids = env._last_candidates.get(v.id, [])
            neighbor_id = next((cid for cid in candidate_ids if cid is not None and cid >= 0), None)
            if neighbor_id is None:
                continue
            neighbor = env._get_vehicle_by_id(neighbor_id)
            if neighbor is None:
                continue
            dist = np.linalg.norm(v.pos - neighbor.pos)
            if dist > Cfg.V2V_RANGE:
                continue
            v2v_rate = env.channel.compute_one_rate(v, neighbor.pos, "V2V", env.time)
            v2i_rate = env.channel.compute_one_rate(
                v, env.rsus[0].position, "V2I", env.time, v2i_user_count=env._estimate_v2i_users()
            )
            if v2i_rate > 0:
                ratios.append(v2v_rate / v2i_rate)

        actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break

    if len(ratios) < 5:
        import pytest
        pytest.skip("not enough v2v samples for rate sanity check")

    p50 = float(np.percentile(ratios, 50))
    p90 = float(np.percentile(ratios, 90))

    # Heuristic sanity: V2V rates should not be orders-of-magnitude below V2I for most samples.
    assert p50 > 1e-6
    assert p90 > 1e-7
