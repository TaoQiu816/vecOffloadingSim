import os
import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_train_rollout_action_alignment_dynamicfleet():
    orig = {
        "NUM_VEHICLES": Cfg.NUM_VEHICLES,
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    prev_audit = os.environ.get("AUDIT_ASSERTS")
    try:
        os.environ["AUDIT_ASSERTS"] = "1"
        Cfg.NUM_VEHICLES = 2
        Cfg.VEHICLE_ARRIVAL_RATE = 5.0

        env = VecOffloadingEnv()
        obs_list, _ = env.reset(seed=0)
        env._next_vehicle_arrival_time = 0.0

        initial_count = len(env.vehicles)
        max_steps = 20
        seen_counts = set([initial_count])

        for _ in range(max_steps):
            actions = []
            for obs in obs_list:
                mask = obs["action_mask"]
                valid = np.where(mask)[0]
                target = int(valid[0]) if len(valid) > 0 else 0
                act = {"target": target, "power": 1.0, "obs_stamp": int(obs["obs_stamp"])}
                actions.append(act)

            assert len(actions) == len(obs_list) == len(env.vehicles)
            for i, act in enumerate(actions):
                assert act["obs_stamp"] == obs_list[i]["obs_stamp"]

            obs_list, rewards, terminated, truncated, _ = env.step(actions)
            seen_counts.add(len(env.vehicles))
            if terminated or truncated:
                break

        assert max(seen_counts) > initial_count
    finally:
        Cfg.NUM_VEHICLES = orig["NUM_VEHICLES"]
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
        if prev_audit is None:
            os.environ.pop("AUDIT_ASSERTS", None)
        else:
            os.environ["AUDIT_ASSERTS"] = prev_audit
