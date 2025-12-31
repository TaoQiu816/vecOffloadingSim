import json
import os
import subprocess
import sys
from pathlib import Path


def test_train_smoke_outputs_metrics(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env.update({
        "CFG_PROFILE": "train_ready_v1",
        "REWARD_MODE": "delta_cft",
        "BONUS_MODE": "none",
        "DEVICE_NAME": "cpu",
        "SEED": "7",
        "RUN_ID": "smoke_train",
        "RUN_DIR": str(run_dir),
        "MAX_EPISODES": "2",
        "MAX_STEPS": "10",
        "DISABLE_BASELINE_EVAL": "1",
        "DISABLE_AUTO_PLOT": "1",
    })

    subprocess.run(
        [sys.executable, "train.py"],
        cwd=repo_root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    metrics_csv = run_dir / "metrics" / "train_metrics.csv"
    snapshot_path = run_dir / "config_snapshot.json"
    assert metrics_csv.exists() and metrics_csv.stat().st_size > 0
    assert snapshot_path.exists()

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["env"]["REWARD_MODE"] == "delta_cft"
    assert str(snapshot["env"]["SEED"]) == "7"
