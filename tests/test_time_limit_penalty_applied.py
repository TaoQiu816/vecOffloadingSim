import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_run_dir(base_dir: Path) -> Path:
    candidates = sorted(base_dir.parent.glob(base_dir.name + "*"), key=lambda p: p.stat().st_mtime)
    assert candidates, f"no run dir found for prefix {base_dir}"
    return candidates[-1]


def _load_metrics_df(run_dir: Path) -> pd.DataFrame:
    logs_csv = run_dir / "logs" / "metrics.csv"
    metrics_csv = run_dir / "metrics" / "metrics.csv"
    if logs_csv.exists():
        return pd.read_csv(logs_csv)
    if metrics_csv.exists():
        return pd.read_csv(metrics_csv)
    raise AssertionError(f"No metrics.csv found under {run_dir}")


def _run_training(tmp_path: Path, run_id: str, penalty_mode: str) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / run_id
    env = os.environ.copy()
    env.update(
        {
            "CFG_PROFILE": "train_ready_v4",
            "REWARD_MODE": "delta_cft",
            "BONUS_MODE": "none",
            "DEVICE_NAME": "cpu",
            "SEED": "7",
            "RUN_DIR": str(run_dir),
            "RUN_ID": run_id,
            "MAX_EPISODES": "2",
            "MAX_STEPS": "5",
            "DISABLE_BASELINE_EVAL": "1",
            "DISABLE_AUTO_PLOT": "1",
            "TIME_LIMIT_PENALTY_MODE": penalty_mode,
        }
    )
    subprocess.run(
        [sys.executable, "train.py"],
        cwd=repo_root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    resolved = _resolve_run_dir(run_dir)
    return _load_metrics_df(resolved)


def test_time_limit_penalty_applied_fixed_vs_scaled(tmp_path):
    # fixed mode should apply constant -1 penalty on time_limit episodes
    df_fixed = _run_training(tmp_path, "tl_fixed", "fixed")
    tl_fixed = df_fixed[df_fixed["termination_reason"] == "time_limit"]
    assert not tl_fixed.empty
    assert tl_fixed["time_limit_penalty_applied"].max() == 1
    assert np.isclose(tl_fixed["time_limit_penalty_value"], -1.0).all()

    # scaled mode should apply a negative penalty that is not all -1
    df_scaled = _run_training(tmp_path, "tl_scaled", "scaled")
    tl_scaled = df_scaled[df_scaled["termination_reason"] == "time_limit"]
    assert not tl_scaled.empty
    assert tl_scaled["time_limit_penalty_applied"].max() == 1
    # penalty must be negative and differ from fixed mode
    assert (tl_scaled["time_limit_penalty_value"] < 0).any()
    assert not np.isclose(tl_scaled["time_limit_penalty_value"], -1.0).all()
