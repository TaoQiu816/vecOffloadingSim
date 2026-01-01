import os
import subprocess
import sys
from pathlib import Path


def test_deadline_fields_positive(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_deadline"
    env = os.environ.copy()
    env.update({
        "CFG_PROFILE": "train_ready_v4",
        "DEVICE_NAME": "cpu",
        "SEED": "7",
        "RUN_ID": "deadline_fields",
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

    candidates = sorted(
        run_dir.parent.glob(run_dir.name + "*"),
        key=lambda p: p.stat().st_mtime
    )
    assert candidates, "no run dir created"
    metrics_csv = candidates[-1] / "logs" / "metrics.csv"
    assert metrics_csv.exists()
    lines = metrics_csv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert "deadline_seconds" in header and "critical_path_cycles" in header
    # check positive values in rows
    has_positive = False
    for row in lines[1:]:
        cols = row.split(",")
        row_dict = dict(zip(header, cols))
        try:
            ddl = float(row_dict.get("deadline_seconds", "0"))
            cp = float(row_dict.get("critical_path_cycles", "0"))
            if ddl > 0 and cp > 0:
                has_positive = True
                break
        except ValueError:
            continue
    assert has_positive, "deadline/critical_path not positive"
