import os
import pytest

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def test_no_stale_obs_used_for_action_selection():
    prev_audit = os.environ.get("AUDIT_ASSERTS")
    orig = {
        "VEHICLE_ARRIVAL_RATE": Cfg.VEHICLE_ARRIVAL_RATE,
    }
    try:
        os.environ["AUDIT_ASSERTS"] = "1"
        Cfg.VEHICLE_ARRIVAL_RATE = 0.0

        env = VecOffloadingEnv()
        obs_list, _ = env.reset(seed=1)

        actions = [{"target": 0, "power": 1.0, "obs_stamp": int(obs["obs_stamp"])} for obs in obs_list]
        obs_list2, _, _, _, _ = env.step(actions)

        stale_actions = [{"target": 0, "power": 1.0, "obs_stamp": int(obs["obs_stamp"])} for obs in obs_list]
        with pytest.raises(AssertionError):
            env.step(stale_actions)
    finally:
        Cfg.VEHICLE_ARRIVAL_RATE = orig["VEHICLE_ARRIVAL_RATE"]
        if prev_audit is None:
            os.environ.pop("AUDIT_ASSERTS", None)
        else:
            os.environ["AUDIT_ASSERTS"] = prev_audit
