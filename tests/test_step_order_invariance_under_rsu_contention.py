from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_step_order_invariance_under_rsu_contention():
    orig = {
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "NUM_RSU": Cfg.NUM_RSU,
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
        Cfg.NUM_RSU = 1
        Cfg.V2V_TOP_K = 0
        Cfg.MAX_NEIGHBORS = 0
        Cfg.MAX_TARGETS = 2
        Cfg.RSU_NUM_PROCESSORS = 1
        Cfg.RSU_QUEUE_CYCLES_LIMIT = 1.0e8
        Cfg.RSU_RANGE = 1e9
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        envA = VecOffloadingEnv()
        envB = VecOffloadingEnv()

        envA.reset(seed=7)
        envB.reset(seed=7)

        for v in envA.vehicles:
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8
        for v in envB.vehicles:
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8

        envA._get_obs()
        envB._get_obs()

        envB.vehicles = list(reversed(envB.vehicles))

        actions_by_vid = {v.id: {"target": 1, "power": 1.0} for v in envA.vehicles}
        actionsA = [actions_by_vid[v.id] for v in envA.vehicles]
        actionsB = [actions_by_vid[v.id] for v in envB.vehicles]

        envA.step(actionsA)
        envB.step(actionsB)

        accepted_A = {v.id for v in envA.vehicles if isinstance(v.curr_target, tuple)}
        accepted_B = {v.id for v in envB.vehicles if isinstance(v.curr_target, tuple)}

        assert len(accepted_A) == 1
        assert len(accepted_B) == 1
        assert accepted_A == accepted_B

        rejected_A = [v for v in envA.vehicles if v.id not in accepted_A]
        rejected_B = [v for v in envB.vehicles if v.id not in accepted_B]

        assert all(v.curr_target == "Local" for v in rejected_A)
        assert all(v.curr_target == "Local" for v in rejected_B)
        assert all(v.illegal_reason == "queue_full_conflict" for v in rejected_A)
        assert all(v.illegal_reason == "queue_full_conflict" for v in rejected_B)

        cA = envA._reward_stats.counters
        cB = envB._reward_stats.counters
        assert cA.get("decision_rsu", 0) == cB.get("decision_rsu", 0)
        assert cA.get("decision_local", 0) == cB.get("decision_local", 0)
        assert cA.get("illegal_action_count", 0) == cB.get("illegal_action_count", 0)
        assert cA.get("illegal_reason.queue_full_conflict", 0) == cB.get("illegal_reason.queue_full_conflict", 0)
        assert cA.get("illegal_reason.internal_invariant_violation", 0) == 0
        assert cB.get("illegal_reason.internal_invariant_violation", 0) == 0
    finally:
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.NUM_RSU = orig["NUM_RSU"]
        Cfg.V2V_TOP_K = orig["V2V_TOP_K"]
        Cfg.MAX_NEIGHBORS = orig["MAX_NEIGHBORS"]
        Cfg.MAX_TARGETS = orig["MAX_TARGETS"]
        Cfg.RSU_NUM_PROCESSORS = orig["RSU_NUM_PROCESSORS"]
        Cfg.RSU_QUEUE_CYCLES_LIMIT = orig["RSU_QUEUE_CYCLES_LIMIT"]
        Cfg.RSU_RANGE = orig["RSU_RANGE"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
