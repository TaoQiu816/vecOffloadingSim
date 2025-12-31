from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_mask_execution_consistency_under_conflict():
    orig = {
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "V2V_TOP_K": Cfg.V2V_TOP_K,
        "MAX_NEIGHBORS": Cfg.MAX_NEIGHBORS,
        "MAX_TARGETS": Cfg.MAX_TARGETS,
        "RSU_NUM_PROCESSORS": Cfg.RSU_NUM_PROCESSORS,
        "RSU_QUEUE_CYCLES_LIMIT": Cfg.RSU_QUEUE_CYCLES_LIMIT,
        "RSU_RANGE": Cfg.RSU_RANGE,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.NUM_VEHICLES = 3
        Cfg.V2V_TOP_K = 0
        Cfg.MAX_NEIGHBORS = 0
        Cfg.MAX_TARGETS = 2
        Cfg.RSU_NUM_PROCESSORS = 1
        # allow exactly one task of 1e8 cycles
        Cfg.RSU_QUEUE_CYCLES_LIMIT = 1.0e8
        Cfg.RSU_RANGE = 1e9
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        env1 = VecOffloadingEnv()
        env2 = VecOffloadingEnv()

        env1.reset(seed=42)
        env2.reset(seed=42)

        # align task comp to enforce deterministic conflicts
        for v in env1.vehicles:
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8
        for v in env2.vehicles:
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8

        env1._get_obs()
        env2._get_obs()

        actions_by_id = {v.id: {"target": 1, "power": 1.0} for v in env1.vehicles}
        actions1 = [actions_by_id[v.id] for v in env1.vehicles]
        actions2 = [actions_by_id[v.id] for v in env2.vehicles]

        env1.step(actions1)
        env2.step(actions2)

        accepted_1 = {v.id for v in env1.vehicles if isinstance(v.curr_target, tuple)}
        accepted_2 = {v.id for v in env2.vehicles if isinstance(v.curr_target, tuple)}

        # capacity allows only one RSU assignment
        assert len(accepted_1) == 1
        assert len(accepted_2) == 1
        assert accepted_1 == accepted_2

        rejected_1 = [v for v in env1.vehicles if v.id not in accepted_1]
        rejected_2 = [v for v in env2.vehicles if v.id not in accepted_2]

        assert all(v.curr_target == "Local" for v in rejected_1)
        assert all(v.curr_target == "Local" for v in rejected_2)
        assert all(v.illegal_reason == "queue_full_conflict" for v in rejected_1)
        assert all(v.illegal_reason == "queue_full_conflict" for v in rejected_2)
    finally:
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.V2V_TOP_K = orig["V2V_TOP_K"]
        Cfg.MAX_NEIGHBORS = orig["MAX_NEIGHBORS"]
        Cfg.MAX_TARGETS = orig["MAX_TARGETS"]
        Cfg.RSU_NUM_PROCESSORS = orig["RSU_NUM_PROCESSORS"]
        Cfg.RSU_QUEUE_CYCLES_LIMIT = orig["RSU_QUEUE_CYCLES_LIMIT"]
        Cfg.RSU_RANGE = orig["RSU_RANGE"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
