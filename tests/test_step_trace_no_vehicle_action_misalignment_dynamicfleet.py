import math

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_step_trace_no_vehicle_action_misalignment_dynamicfleet():
    orig = {
        "MAP_SIZE": Cfg.MAP_SIZE,
        "DT": Cfg.DT,
        "VEL_MEAN": Cfg.VEL_MEAN,
        "VEL_STD": Cfg.VEL_STD,
        "VEL_MIN": Cfg.VEL_MIN,
        "VEL_MAX": Cfg.VEL_MAX,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.MAP_SIZE = 50.0
        Cfg.DT = 1.0
        Cfg.VEL_MEAN = 20.0
        Cfg.VEL_STD = 0.0
        Cfg.VEL_MIN = 20.0
        Cfg.VEL_MAX = 20.0
        Cfg.VEHICLE_ARRIVAL_RATE = 1.0

        env = VecOffloadingEnv()
        env.reset(seed=0)

        for _ in range(5):
            power_map = {}
            ready_map = {}
            actions = []
            for idx, v in enumerate(env.vehicles):
                p = (idx + 1) / (len(env.vehicles) + 1.0)
                power_map[v.id] = p
                ready_map[v.id] = v.task_dag.get_top_priority_task() is not None
                actions.append({"target": 0, "power": p})

            _, _, terminated, truncated, _ = env.step(actions)

            for v in env.vehicles:
                if v.id in power_map and ready_map.get(v.id, False):
                    expected = Cfg.TX_POWER_MIN_DBM + power_map[v.id] * (
                        Cfg.TX_POWER_MAX_DBM - Cfg.TX_POWER_MIN_DBM
                    )
                    assert math.isclose(v.tx_power_dbm, expected, rel_tol=0.0, abs_tol=1e-6)

            if terminated or truncated:
                break
    finally:
        Cfg.MAP_SIZE = orig["MAP_SIZE"]
        Cfg.DT = orig["DT"]
        Cfg.VEL_MEAN = orig["VEL_MEAN"]
        Cfg.VEL_STD = orig["VEL_STD"]
        Cfg.VEL_MIN = orig["VEL_MIN"]
        Cfg.VEL_MAX = orig["VEL_MAX"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
