import json
import os
import subprocess
import sys
from pathlib import Path


def test_diag_metrics_written(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_diag"
    env = os.environ.copy()
    env.update({
        "CFG_PROFILE": "train_ready_v4",
        "DEVICE_NAME": "cpu",
        "SEED": "7",
        "RUN_ID": "diag_metrics",
        "RUN_DIR": str(run_dir),
        "MAX_EPISODES": "2",
        "MAX_STEPS": "5",
        "DISABLE_BASELINE_EVAL": "1",
        "DISABLE_AUTO_PLOT": "1",
        "EPISODE_JSONL_STDOUT": "0",
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

    resolved = sorted(run_dir.parent.glob(run_dir.name + "*"))[-1]
    metrics_csv = resolved / "logs" / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = resolved / "metrics" / "metrics.csv"
    assert metrics_csv.exists()
    lines = metrics_csv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cols = ["avail_V", "neighbor_count_mean", "best_v2v_valid_rate", "policy_entropy", "entropy_loss"]
    for c in cols:
        assert c in header
    # parse csv rows
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    for c in ["avail_V", "neighbor_count_mean", "best_v2v_valid_rate", "policy_entropy", "entropy_loss"]:
        assert not (df[c].isna().all()), f"{c} all NaN"
    # jsonl keys exist
    jsonl = resolved / "logs" / "metrics.jsonl"
    if not jsonl.exists():
        jsonl = resolved / "metrics" / "metrics.jsonl"
    assert jsonl.exists()
    last = json.loads(jsonl.read_text(encoding="utf-8").strip().splitlines()[-1])
    for c in ["avail_V", "neighbor_count_mean", "best_v2v_valid_rate", "policy_entropy", "entropy_loss"]:
        assert c in last
