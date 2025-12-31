from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_step_order_bias_queue_full():
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
        # deterministic, minimal setup
        Cfg.NUM_VEHICLES = 2
        Cfg.V2V_TOP_K = 0
        Cfg.MAX_NEIGHBORS = 0
        Cfg.MAX_TARGETS = 2
        Cfg.RSU_NUM_PROCESSORS = 1
        Cfg.RSU_QUEUE_CYCLES_LIMIT = 1.5e8
        Cfg.RSU_RANGE = 1e9
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        env1 = VecOffloadingEnv()
        env2 = VecOffloadingEnv()

        env1.reset(seed=123)
        env2.reset(seed=123)

        # align task comp to enforce queue_full after one enqueue
        for v in env1.vehicles:
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8
        for v in env2.vehicles:
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8

        env1._get_obs()

        # reverse vehicle order in env2 to change processing order
        env2.vehicles = list(reversed(env2.vehicles))
        env2._get_obs()

        actions_by_id = {v.id: {"target": 1, "power": 1.0} for v in env1.vehicles}
        actions1 = [actions_by_id[v.id] for v in env1.vehicles]
        actions2 = [actions_by_id[v.id] for v in env2.vehicles]

        env1.step(actions1)
        env2.step(actions2)

        accepted_1 = {v.id for v in env1.vehicles if isinstance(v.curr_target, tuple)}
        accepted_2 = {v.id for v in env2.vehicles if isinstance(v.curr_target, tuple)}

        # exactly one should be accepted due to queue capacity
        assert len(accepted_1) == 1
        assert len(accepted_2) == 1
        # no order bias: accepted vehicle should match
        assert accepted_1 == accepted_2

        c1 = env1._reward_stats.counters
        c2 = env2._reward_stats.counters
        assert c1.get("decision_rsu", 0) == c2.get("decision_rsu", 0)
        assert c1.get("decision_local", 0) == c2.get("decision_local", 0)
        assert c1.get("illegal_action_count", 0) == c2.get("illegal_action_count", 0)
        assert c1.get("illegal_reason.queue_full_conflict", 0) == c2.get("illegal_reason.queue_full_conflict", 0)
    finally:
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.V2V_TOP_K = orig["V2V_TOP_K"]
        Cfg.MAX_NEIGHBORS = orig["MAX_NEIGHBORS"]
        Cfg.MAX_TARGETS = orig["MAX_TARGETS"]
        Cfg.RSU_NUM_PROCESSORS = orig["RSU_NUM_PROCESSORS"]
        Cfg.RSU_QUEUE_CYCLES_LIMIT = orig["RSU_QUEUE_CYCLES_LIMIT"]
        Cfg.RSU_RANGE = orig["RSU_RANGE"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
