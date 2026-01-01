import json
import os
import subprocess
import sys
from pathlib import Path


def _resolve_run_dir(base_dir: Path) -> Path:
    candidates = sorted(
        base_dir.parent.glob(base_dir.name + "*"),
        key=lambda p: p.stat().st_mtime
    )
    assert candidates, f"no run dir found for prefix {base_dir}"
    return candidates[-1]


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

    resolved_dir = _resolve_run_dir(run_dir)
    metrics_csv = resolved_dir / "logs" / "metrics.csv"
    snapshot_path = resolved_dir / "logs" / "config_snapshot.json"
    assert metrics_csv.exists() and metrics_csv.stat().st_size > 0
    assert snapshot_path.exists()

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["env"]["REWARD_MODE"] == "delta_cft"
    assert str(snapshot["env"]["SEED"]) == "7"

    header = metrics_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    required = {
        "reward_mean",
        "reward_p95",
        "delta_cft_rem_mean",
        "success_rate_end",
        "mean_cft_est",
        "mean_cft_completed",
        "episode_time_seconds",
        "vehicle_cft_count",
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
        "entropy",
        "approx_kl",
        "deadline_seconds",
        "critical_path_cycles",
        "time_limit_rate",
    }
    assert required.issubset(set(header))
