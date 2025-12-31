import json
import os
import numpy as np

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg


def test_metrics_in_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("REWARD_JSONL_PATH", str(tmp_path / "episode.jsonl"))
    monkeypatch.setenv("MAX_EPISODES", "1")
    monkeypatch.setattr(Cfg, "MAX_STEPS", 1, raising=False)

    env = VecOffloadingEnv()
    obs_list, _ = env.reset(seed=0)
    actions = []
    for obs in obs_list:
        valid = np.where(obs["action_mask"])[0]
        target = int(valid[0]) if len(valid) > 0 else 0
        actions.append({"target": target, "power": 0.5})
    env.step(actions)

    data = (tmp_path / "episode.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert data, "JSONL should contain at least one line"
    payload = json.loads(data[-1])

    vehicle_success = payload.get("vehicle_success_rate", payload.get("success_rate"))
    deadline_miss = payload.get("deadline_miss_rate")
    subtask_success = payload.get("subtask_success_rate")

    assert 0.0 <= vehicle_success <= 1.0
    assert 0.0 <= deadline_miss <= 1.0
    assert 0.0 <= subtask_success <= 1.0
