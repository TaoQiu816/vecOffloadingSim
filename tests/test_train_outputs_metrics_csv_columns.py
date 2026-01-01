import os
import subprocess
import sys
from pathlib import Path


def _resolve_run_dir(base_dir: Path) -> Path:
    candidates = sorted(
        base_dir.parent.glob(base_dir.name + "*"),
        key=lambda p: p.stat().st_mtime,
    )
    assert candidates, f"no run dir found for prefix {base_dir}"
    return candidates[-1]


def test_train_outputs_metrics_csv_columns(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_metrics"
    env = os.environ.copy()
    env.update({
        "CFG_PROFILE": "train_ready_v1",
        "REWARD_MODE": "delta_cft",
        "BONUS_MODE": "none",
        "DEVICE_NAME": "cpu",
        "SEED": "7",
        "RUN_ID": "metrics_columns",
        "RUN_DIR": str(run_dir),
        "MAX_EPISODES": "2",
        "MAX_STEPS": "10",
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

    resolved_dir = _resolve_run_dir(run_dir)
    metrics_csv = resolved_dir / "logs" / "metrics.csv"
    assert metrics_csv.exists() and metrics_csv.stat().st_size > 0

    header = metrics_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    required = {
        "reward_mean",
        "reward_p95",
        "delta_cft_rem_mean",
        "success_rate_end",
        "entropy",
        "approx_kl",
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
        "deadline_seconds",
        "critical_path_cycles",
        "time_limit_rate",
        "mean_cft_est",
        "mean_cft_completed",
        "episode_time_seconds",
        "vehicle_cft_count",
    }
    assert required.issubset(set(header))
