from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_delta_cft_excludes_new_spawn_from_credit(monkeypatch):
    orig = {
        "REWARD_MODE": Cfg.REWARD_MODE,
        "ENERGY_IN_DELTA_CFT": Cfg.ENERGY_IN_DELTA_CFT,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.REWARD_MODE = "delta_cft"
        Cfg.ENERGY_IN_DELTA_CFT = False
        Cfg.VEHICLE_ARRIVAL_RATE = 1.0

        env = VecOffloadingEnv()
        env.reset(seed=13)
        env._next_vehicle_arrival_time = 0.0
        env._get_obs()

        def fake_compute(snapshot_time=None, v2i_user_count=None, vehicle_ids=None):
            assert vehicle_ids is not None
            if snapshot_time is None:
                snapshot_time = env.time
            return snapshot_time + 50.0 + 100.0 * len(vehicle_ids)

        monkeypatch.setattr(env, "_compute_mean_cft_pi0", fake_compute)

        prev_count = len(env.vehicles)
        actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        _, rewards, _, _, _ = env.step(actions)
        assert len(env.vehicles) >= prev_count + 1
        assert all(abs(r) < 1e-6 for r in rewards)
    finally:
        Cfg.REWARD_MODE = orig["REWARD_MODE"]
        Cfg.ENERGY_IN_DELTA_CFT = orig["ENERGY_IN_DELTA_CFT"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
