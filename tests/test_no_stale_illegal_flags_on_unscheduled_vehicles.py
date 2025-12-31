from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_no_stale_illegal_flags_on_unscheduled_vehicles():
    orig = {
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        Cfg.NUM_VEHICLES = 2
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        env = VecOffloadingEnv()
        env.reset(seed=9)
        env._get_obs()

        v0 = env.vehicles[0]
        v1 = env.vehicles[1]

        # make v0 unschedulable this step
        v0.task_dag.status[:] = 3
        v0.illegal_action = True
        v0.illegal_reason = "forced_stale"

        actions = [
            {"target": 1, "power": 1.0},
            {"target": 0, "power": 1.0},
        ]

        env.step(actions)

        assert v0.task_dag.get_top_priority_task() is None
        assert v0.illegal_action is False
        assert v0.illegal_reason is None
        # ensure scheduled vehicle still has valid state
        assert v1.illegal_reason in (None, "queue_full_conflict", "rsu_unavailable", "rsu_out_of_coverage")
    finally:
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
