from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_delta_cft_remaining_time_formula_prevents_time_drift(monkeypatch):
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
        env.reset(seed=11)
        env._get_obs()

        def fake_compute(snapshot_time=None, v2i_user_count=None, vehicle_ids=None):
            if snapshot_time is None:
                snapshot_time = env.time
            return snapshot_time + 10.0

        monkeypatch.setattr(env, "_compute_mean_cft_pi0", fake_compute)

        actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        _, rewards, _, _, _ = env.step(actions)
        assert all(abs(r) < 1e-6 for r in rewards)
    finally:
        Cfg.REWARD_MODE = orig["REWARD_MODE"]
        Cfg.ENERGY_IN_DELTA_CFT = orig["ENERGY_IN_DELTA_CFT"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
