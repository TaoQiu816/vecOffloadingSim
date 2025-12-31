import json
import numpy as np

from envs.vec_offloading_env import VecOffloadingEnv
from configs.config import SystemConfig as Cfg


def test_subtask_metrics_consistent(tmp_path, monkeypatch):
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

    payload = json.loads((tmp_path / "episode.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    total_subtasks = payload.get("total_subtasks", 0)
    completed_subtasks = payload.get("completed_subtasks", 0)
    rate = payload.get("subtask_success_rate", 0.0)

    assert total_subtasks >= completed_subtasks
    expected = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
    assert abs(rate - expected) < 1e-6
