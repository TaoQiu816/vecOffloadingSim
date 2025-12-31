from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_delta_cft_credit_assignment_current_step():
    orig = {
        "REWARD_MODE": Cfg.REWARD_MODE,
        "ENERGY_IN_DELTA_CFT": Cfg.ENERGY_IN_DELTA_CFT,
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "NUM_RSU": Cfg.NUM_RSU,
        "RSU_RANGE": Cfg.RSU_RANGE,
        "RSU_NUM_PROCESSORS": Cfg.RSU_NUM_PROCESSORS,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
        "F_RSU": Cfg.F_RSU,
    }
    try:
        Cfg.REWARD_MODE = "delta_cft"
        Cfg.ENERGY_IN_DELTA_CFT = False
        Cfg.NUM_VEHICLES = 1
        Cfg.NUM_RSU = 1
        Cfg.RSU_RANGE = 1e9
        Cfg.RSU_NUM_PROCESSORS = 1
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0
        Cfg.F_RSU = 1.0e9

        env_local = VecOffloadingEnv()
        env_rsu = VecOffloadingEnv()

        env_local.reset(seed=3)
        env_rsu.reset(seed=3)

        for env in (env_local, env_rsu):
            v = env.vehicles[0]
            v.cpu_freq = 1.0e6
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8
            v.task_dag.total_data[:] = 0.0
            v.task_dag.rem_data[:] = 0.0
            env.rsus[0].cpu_freq = 1.0e9
            env._get_obs()

        r_local = env_local.step([{"target": 0, "power": 1.0}])[1][0]
        r_rsu = env_rsu.step([{"target": 1, "power": 1.0}])[1][0]

        assert r_rsu > r_local
    finally:
        Cfg.REWARD_MODE = orig["REWARD_MODE"]
        Cfg.ENERGY_IN_DELTA_CFT = orig["ENERGY_IN_DELTA_CFT"]
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.NUM_RSU = orig["NUM_RSU"]
        Cfg.RSU_RANGE = orig["RSU_RANGE"]
        Cfg.RSU_NUM_PROCESSORS = orig["RSU_NUM_PROCESSORS"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
        Cfg.F_RSU = orig["F_RSU"]
