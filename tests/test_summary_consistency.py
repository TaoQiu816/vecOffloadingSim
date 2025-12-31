import json
import os

from scripts import summarize_ablation as sa


def test_summary_consistency(tmp_path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    log_dir.mkdir()
    out_dir.mkdir()

    sample = {
        "reward_mode": "incremental_cost",
        "bonus_mode": "none",
        "episode_id": 1,
        "target_episodes": 2,
        "vehicle_success_rate": 0.5,
        "success_rate": 0.5,
        "deadline_miss_rate": 0.0,
        "subtask_success_rate": 0.25,
        "task_success_rate": 0.5,
        "decision_frac_local": 0.6,
        "decision_frac_rsu": 0.2,
        "decision_frac_v2v": 0.2,
        "total_decisions_count": 10,
        "episode_vehicle_count": 2,
        "episode_vehicle_seen": 2,
        "mean_cft": 5.0,
    }
    path = log_dir / "sample.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")
        sample2 = sample.copy()
        sample2["episode_id"] = 2
        f.write(json.dumps(sample2) + "\n")

    runs, excluded, missing, legacy = sa.load_runs(str(log_dir))
    assert not excluded
    summary = sa.summarize_groups(sa._dedup_latest(runs))
    md_path = out_dir / "ablation_summary.md"
    sa.write_md(summary, str(md_path), excluded, missing, legacy)
    md_text = md_path.read_text(encoding="utf-8")
    assert "decision_frac_local" in md_text
    assert "subtask_success_rate" in md_text
    assert "Missing Fields" in md_text
