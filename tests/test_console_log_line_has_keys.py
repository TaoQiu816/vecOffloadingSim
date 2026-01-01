import os
import subprocess
import sys
from pathlib import Path


def test_console_log_line_has_keys(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        {
            "CFG_PROFILE": "train_ready_v4",
            "DEVICE_NAME": "cpu",
            "SEED": "7",
            "RUN_DIR": str(run_dir),
            "RUN_ID": "logline_smoke",
            "MAX_EPISODES": "2",
            "MAX_STEPS": "5",
            "DISABLE_BASELINE_EVAL": "1",
            "DISABLE_AUTO_PLOT": "1",
            "LOG_INTERVAL": "1",
        }
    )
    proc = subprocess.run(
        [sys.executable, "train.py"],
        cwd=repo,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("|")]
    # find a data row (long enough)
    data_lines = [ln for ln in lines if "r_sum" in ln or "term" in ln]
    assert data_lines, "no table rows found"
    sample = data_lines[-1]
    for key in ["r_sum", "succ", "miss", "L", "R", "V", "kl", "clip", "v_loss", "v2v_win"]:
        assert key in proc.stdout, f"{key} not found in stdout"
