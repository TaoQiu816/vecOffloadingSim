import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_reward_direction_consistency():
    orig = {
        "DIST_PENALTY_MODE": Cfg.DIST_PENALTY_MODE,
        "NUM_RSU": Cfg.NUM_RSU,
        "RSU_NUM_PROCESSORS": Cfg.RSU_NUM_PROCESSORS,
        "RSU_QUEUE_CYCLES_LIMIT": Cfg.RSU_QUEUE_CYCLES_LIMIT,
        "RSU_RANGE": Cfg.RSU_RANGE,
        "F_RSU": Cfg.F_RSU,
    }
    try:
        Cfg.DIST_PENALTY_MODE = "off"
        Cfg.NUM_RSU = 1
        Cfg.RSU_NUM_PROCESSORS = 1
        Cfg.RSU_QUEUE_CYCLES_LIMIT = 1.0e9
        Cfg.RSU_RANGE = 1.0e9
        Cfg.F_RSU = 1.0e8

        env = VecOffloadingEnv()
        env.reset(seed=5)

        v = env.vehicles[0]
        v.cpu_freq = 1.0e9
        task_idx = v.task_dag.get_top_priority_task()
        assert task_idx is not None

        # Force a non-zero data size to make RSU tx_time > 0
        v.task_dag.total_data[task_idx] = max(v.task_dag.total_data[task_idx], 1.0e6)
        v.task_dag.total_comp[task_idx] = 1.0e8
        v.task_dag.rem_comp[task_idx] = 1.0e8

        rsu = env.rsus[0]
        # Preload RSU queue to increase wait_time but avoid hard constraint
        preload = 0.6 * Cfg.RSU_QUEUE_CYCLES_LIMIT
        rsu.queue_manager.clear()
        rsu.queue_manager.processor_queues[0].enqueue(preload)

        r_local, comp_local = env.calculate_agent_reward(v.id, "Local", task_idx, return_components=True)
        r_rsu, comp_rsu = env.calculate_agent_reward(v.id, ("RSU", 0), task_idx, return_components=True)

        cost_local = Cfg.DELAY_WEIGHT * comp_local["delay_norm"] + Cfg.ENERGY_WEIGHT * comp_local["energy_norm"]
        cost_rsu = Cfg.DELAY_WEIGHT * comp_rsu["delay_norm"] + Cfg.ENERGY_WEIGHT * comp_rsu["energy_norm"]

        assert cost_rsu >= cost_local
        assert r_local >= r_rsu
    finally:
        Cfg.DIST_PENALTY_MODE = orig["DIST_PENALTY_MODE"]
        Cfg.NUM_RSU = orig["NUM_RSU"]
        Cfg.RSU_NUM_PROCESSORS = orig["RSU_NUM_PROCESSORS"]
        Cfg.RSU_QUEUE_CYCLES_LIMIT = orig["RSU_QUEUE_CYCLES_LIMIT"]
        Cfg.RSU_RANGE = orig["RSU_RANGE"]
        Cfg.F_RSU = orig["F_RSU"]
