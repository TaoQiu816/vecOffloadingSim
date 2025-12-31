import os
import sys
import subprocess
from pathlib import Path


def test_train_console_compact_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_compact"
    env = os.environ.copy()
    env.update({
        "RUN_DIR": str(run_dir),
        "RUN_ID": "test_compact",
        "MAX_EPISODES": "1",
        "DEVICE_NAME": "cpu",
        "DISABLE_BASELINE_EVAL": "1",
        "DISABLE_AUTO_PLOT": "1",
        "EPISODE_JSONL_STDOUT": "0",
    })
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    csv_path = run_dir / "metrics" / "train_metrics.csv"
    jsonl_path = run_dir / "logs" / "run.jsonl"
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert jsonl_path.exists() and jsonl_path.stat().st_size > 0

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
