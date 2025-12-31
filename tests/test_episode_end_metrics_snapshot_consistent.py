import json
import os

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def _read_last_json(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    assert lines, "no jsonl lines written"
    return json.loads(lines[-1])


def _assert_snapshot(env, rec):
    assert rec["episode_vehicle_count"] == len(env.vehicles)
    assert rec["episode_vehicle_seen"] == len(env._vehicles_seen)
    success_count = sum(1 for v in env.vehicles if v.task_dag.is_finished)
    fail_count = sum(1 for v in env.vehicles if v.task_dag.is_failed)
    denom = max(len(env.vehicles), 1)
    assert abs(rec["success_rate_end"] - (success_count / denom)) < 1e-6
    assert abs(rec["deadline_miss_rate"] - (fail_count / denom)) < 1e-6
    assert rec["total_decisions_count"] == env._reward_stats.counters.get("decision_total", 0)


def test_episode_end_metrics_snapshot_consistent(tmp_path):
    orig_max_steps = Cfg.MAX_STEPS
    try:
        # truncated episode
        Cfg.MAX_STEPS = 1
        trunc_path = tmp_path / "trunc.jsonl"
        env = VecOffloadingEnv()
        env._jsonl_path = str(trunc_path)
        env.reset(seed=0)
        actions = [{"target": 0, "power": 1.0} for _ in env.vehicles]
        _, _, terminated, truncated, _ = env.step(actions)
        assert truncated and not terminated
        rec = _read_last_json(trunc_path)
        assert rec["truncated"] is True
        _assert_snapshot(env, rec)

        # terminated episode
        Cfg.MAX_STEPS = 5
        term_path = tmp_path / "term.jsonl"
        env2 = VecOffloadingEnv()
        env2._jsonl_path = str(term_path)
        env2.reset(seed=1)
        for v in env2.vehicles:
            v.task_dag.status[:] = 3
            v.task_dag._is_failed = False
        actions = [{"target": 0, "power": 1.0} for _ in env2.vehicles]
        _, _, terminated, truncated, _ = env2.step(actions)
        assert terminated
        rec2 = _read_last_json(term_path)
        assert rec2["terminated"] is True
        _assert_snapshot(env2, rec2)
    finally:
        Cfg.MAX_STEPS = orig_max_steps
