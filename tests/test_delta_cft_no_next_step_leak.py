from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_delta_cft_no_next_step_leak():
    orig = {
        "REWARD_MODE": Cfg.REWARD_MODE,
        "ENERGY_IN_DELTA_CFT": Cfg.ENERGY_IN_DELTA_CFT,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.REWARD_MODE = "delta_cft"
        Cfg.ENERGY_IN_DELTA_CFT = False
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        env = VecOffloadingEnv()
        env.reset(seed=21)
        env._get_obs()

        calls = []
        t0 = env.time

        orig_compute = env._compute_mean_cft_pi0
        def wrapped_compute(snapshot_time=None, v2i_user_count=None):
            calls.append(("cft", snapshot_time, env.time))
            return orig_compute(snapshot_time=snapshot_time, v2i_user_count=v2i_user_count)
        env._compute_mean_cft_pi0 = wrapped_compute

        orig_get_obs = env._get_obs
        def wrapped_get_obs(*args, **kwargs):
            calls.append(("obs", env.time))
            return orig_get_obs(*args, **kwargs)
        env._get_obs = wrapped_get_obs

        actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        env.step(actions)

        cft_calls = [c for c in calls if c[0] == "cft"]
        assert len(cft_calls) == 2
        assert abs(cft_calls[0][1] - t0) < 1e-6
        assert abs(cft_calls[0][2] - t0) < 1e-6

        t1 = t0 + Cfg.DT
        assert abs(cft_calls[1][1] - t1) < 1e-6
        assert abs(cft_calls[1][2] - t1) < 1e-6

        obs_idx = next(i for i, c in enumerate(calls) if c[0] == "obs")
        assert obs_idx > calls.index(cft_calls[1])
    finally:
        Cfg.REWARD_MODE = orig["REWARD_MODE"]
        Cfg.ENERGY_IN_DELTA_CFT = orig["ENERGY_IN_DELTA_CFT"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
