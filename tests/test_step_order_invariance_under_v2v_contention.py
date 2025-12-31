import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_step_order_invariance_under_v2v_contention():
    orig = {
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "NUM_RSU": Cfg.NUM_RSU,
        "V2V_TOP_K": Cfg.V2V_TOP_K,
        "MAX_NEIGHBORS": Cfg.MAX_NEIGHBORS,
        "MAX_TARGETS": Cfg.MAX_TARGETS,
        "VEHICLE_QUEUE_CYCLES_LIMIT": Cfg.VEHICLE_QUEUE_CYCLES_LIMIT,
        "V2V_RANGE": Cfg.V2V_RANGE,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.NUM_VEHICLES = 3
        Cfg.NUM_RSU = 1
        Cfg.V2V_TOP_K = 1
        Cfg.MAX_NEIGHBORS = 1
        Cfg.MAX_TARGETS = 3
        Cfg.VEHICLE_QUEUE_CYCLES_LIMIT = 1.0e8
        Cfg.V2V_RANGE = 1.0e9
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        envA = VecOffloadingEnv()
        envB = VecOffloadingEnv()

        envA.reset(seed=11)
        envB.reset(seed=11)

        # enforce deterministic positions: both v0/v1 choose v2 as the only V2V candidate
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, 10.0]),
            2: np.array([0.0, 1.0]),
        }
        for v in envA.vehicles:
            v.pos = positions[v.id].copy()
            v.vel[:] = 0.0
            v.cpu_freq = 1.0e9
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8
            v.task_dag.total_data[:] = 0.0
            v.task_dag.rem_data[:] = 0.0
        for v in envB.vehicles:
            v.pos = positions[v.id].copy()
            v.vel[:] = 0.0
            v.cpu_freq = 1.0e9
            v.task_dag.total_comp[:] = 1.0e8
            v.task_dag.rem_comp[:] = 1.0e8
            v.task_dag.total_data[:] = 0.0
            v.task_dag.rem_data[:] = 0.0

        # refresh candidates deterministically (clear dist cache after manual pos edits)
        envA._dist_matrix_cache = None
        envA._dist_matrix_time = -1.0
        np.random.seed(123)
        envA._get_obs()

        envB.vehicles = list(reversed(envB.vehicles))
        envB._dist_matrix_cache = None
        envB._dist_matrix_time = -1.0
        np.random.seed(123)
        envB._get_obs()

        candidate_a0 = envA._last_candidates[0][0]
        candidate_a1 = envA._last_candidates[1][0]
        candidate_b0 = envB._last_candidates[0][0]
        candidate_b1 = envB._last_candidates[1][0]
        assert candidate_a0 == 2
        assert candidate_a1 == 2
        assert candidate_b0 == 2
        assert candidate_b1 == 2

        actions_by_vid = {
            0: {"target": 2, "power": 1.0},
            1: {"target": 2, "power": 1.0},
            2: {"target": 0, "power": 1.0},
        }
        actionsA = [actions_by_vid[v.id] for v in envA.vehicles]
        actionsB = [actions_by_vid[v.id] for v in envB.vehicles]

        envA.step(actionsA)
        envB.step(actionsB)

        accepted_A = {v.id for v in envA.vehicles if isinstance(v.curr_target, int)}
        accepted_B = {v.id for v in envB.vehicles if isinstance(v.curr_target, int)}

        assert len(accepted_A) == 1
        assert len(accepted_B) == 1
        assert accepted_A == accepted_B

        rejected_A = [v for v in envA.vehicles if v.id not in accepted_A and v.id != 2]
        rejected_B = [v for v in envB.vehicles if v.id not in accepted_B and v.id != 2]

        assert all(v.curr_target == "Local" for v in rejected_A)
        assert all(v.curr_target == "Local" for v in rejected_B)
        assert all(v.illegal_reason == "queue_full_conflict" for v in rejected_A)
        assert all(v.illegal_reason == "queue_full_conflict" for v in rejected_B)

        cA = envA._reward_stats.counters
        cB = envB._reward_stats.counters
        assert cA.get("decision_v2v", 0) == cB.get("decision_v2v", 0)
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
        Cfg.VEHICLE_QUEUE_CYCLES_LIMIT = orig["VEHICLE_QUEUE_CYCLES_LIMIT"]
        Cfg.V2V_RANGE = orig["V2V_RANGE"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
