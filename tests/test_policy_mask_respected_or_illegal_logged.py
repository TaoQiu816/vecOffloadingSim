from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_policy_mask_respected_or_illegal_logged():
    orig = {
        "REWARD_MODE": Cfg.REWARD_MODE,
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "NUM_RSU": Cfg.NUM_RSU,
        "RSU_RANGE": Cfg.RSU_RANGE,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.REWARD_MODE = "incremental_cost"
        Cfg.NUM_VEHICLES = 1
        Cfg.NUM_RSU = 1
        Cfg.RSU_RANGE = 0.0
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        env = VecOffloadingEnv()
        obs_list, _ = env.reset(seed=2)
        obs_list = env._get_obs()

        actions = [{"target": 1, "power": 1.0} for _ in env.vehicles]
        env.step(actions)

        v = env.vehicles[0]
        assert v.curr_target == "Local"
        assert v.illegal_reason in ("rsu_unavailable", "rsu_out_of_coverage")
        assert env._reward_stats.counters.get("illegal_action_count", 0) >= 1
    finally:
        Cfg.REWARD_MODE = orig["REWARD_MODE"]
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.NUM_RSU = orig["NUM_RSU"]
        Cfg.RSU_RANGE = orig["RSU_RANGE"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
