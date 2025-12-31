import json
import os

from scripts import summarize_ablation as sa


def test_load_runs_accepts_legacy_jsonl(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    path = log_dir / "legacy.jsonl"
    sample = {
        "reward_mode": "incremental_cost",
        "bonus_mode": "none",
        "episode_id": 1,
        "vehicle_success_rate": 0.5,
        "success_rate": 0.5,
        "deadline_miss_rate": 0.0,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")
    runs, excluded, missing_counts, legacy_counts = sa.load_runs(str(log_dir))
    assert runs, "legacy file should not be excluded"
    assert not excluded
    assert legacy_counts.get("missing_target_episodes", 0) == 1


def test_md_written_when_no_runs(tmp_path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    log_dir.mkdir()
    out_dir.mkdir()
    runs, excluded, missing_counts, legacy_counts = sa.load_runs(str(log_dir))
    summary = sa.summarize_groups(sa._dedup_latest(runs)) if runs else {}
    md_path = out_dir / "ablation_summary.md"
    sa.write_md(summary, str(md_path), excluded, missing_counts, legacy_counts)
    text = md_path.read_text(encoding="utf-8")
    assert "**Excluded Files**" in text
    assert "**Missing Fields Summary**" in text
