import json
import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def _run_episode(tmp_path, monkeypatch, dynamic=False):
    log_file = tmp_path / "metrics_static.jsonl"
    monkeypatch.setenv("REWARD_JSONL_PATH", str(log_file))
    monkeypatch.setattr(Cfg, "DEBUG_ASSERT_METRICS", True, raising=False)
    if dynamic:
        monkeypatch.setattr(Cfg, "VEHICLE_ARRIVAL_RATE", 1.0, raising=False)
    else:
        monkeypatch.setattr(Cfg, "VEHICLE_ARRIVAL_RATE", 0.0, raising=False)

    env = VecOffloadingEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        actions = []
        for ob in obs:
            valid = np.where(ob["action_mask"])[0]
            target = int(valid[0]) if len(valid) > 0 else 0
            actions.append({"target": target, "power": 0.5})
        obs, _, _, _, _ = env.step(actions)
    env._log_episode_stats(terminated=True, truncated=False)
    data = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    return data[-1]


def test_metrics_bounds_staticfleet(tmp_path, monkeypatch):
    line = _run_episode(tmp_path, monkeypatch, dynamic=False)
    for key in ["vehicle_success_rate", "task_success_rate", "success_rate", "subtask_success_rate", "deadline_miss_rate"]:
        assert 0.0 <= line.get(key, 0.0) <= 1.0
    if line.get("total_decisions_count", 0) > 0:
        frac_sum = line.get("decision_frac_local", 0.0) + line.get("decision_frac_rsu", 0.0) + line.get("decision_frac_v2v", 0.0)
        assert 0.0 <= frac_sum <= 1.001


def test_metrics_bounds_dynamicfleet(tmp_path, monkeypatch):
    line = _run_episode(tmp_path, monkeypatch, dynamic=True)
    for key in ["vehicle_success_rate", "task_success_rate", "success_rate", "subtask_success_rate", "deadline_miss_rate"]:
        assert 0.0 <= line.get(key, 0.0) <= 1.0
    if line.get("total_decisions_count", 0) > 0:
        frac_sum = line.get("decision_frac_local", 0.0) + line.get("decision_frac_rsu", 0.0) + line.get("decision_frac_v2v", 0.0)
        assert 0.0 <= frac_sum <= 1.001
