import json
import os
import time

from scripts import summarize_ablation as sa


def _write_jsonl(path, reward_mode, bonus_mode, seed, run_id, target_episodes, lines):
    payloads = []
    for i in range(lines):
        payloads.append({
            "reward_mode": reward_mode,
            "bonus_mode": bonus_mode,
            "seed": seed,
            "run_id": run_id,
            "episode_id": i + 1,
            "target_episodes": target_episodes,
            "vehicle_success_rate": 0.5,
            "deadline_miss_rate": 0.1,
        })
    with open(path, "w", encoding="utf-8") as f:
        for p in payloads:
            f.write(json.dumps(p) + "\n")


def test_dedup_and_truncate_exclusion(tmp_path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "results"
    log_dir.mkdir()
    out_dir.mkdir()

    old_file = log_dir / "incremental_cost__none__seed0__run_old.jsonl"
    new_file = log_dir / "incremental_cost__none__seed0__run_new.jsonl"
    bad_file = log_dir / "incremental_cost__none__seed0__run_bad.jsonl"

    _write_jsonl(old_file, "incremental_cost", "none", 0, "old", 10, 10)
    _write_jsonl(new_file, "incremental_cost", "none", 0, "new", 10, 10)
    _write_jsonl(bad_file, "incremental_cost", "none", 0, "bad", 10, 3)

    now = time.time()
    os.utime(old_file, (now - 100, now - 100))
    os.utime(new_file, (now, now))
    os.utime(bad_file, (now - 50, now - 50))

    runs, excluded, missing, legacy = sa.load_runs(str(log_dir))
    latest = sa._dedup_latest(runs)

    assert any(item["path"] == str(bad_file) and item["reason"] == "too_short" for item in excluded)
    assert len(latest) == 1
    assert latest[0]["path"] == str(new_file)

    summary = sa.summarize_groups(latest)
    md_path = out_dir / "ablation_summary.md"
    sa.write_md(summary, str(md_path), excluded, missing, legacy)
    md_text = md_path.read_text(encoding="utf-8")
    assert "Excluded Files" in md_text
