import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_dummy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_plot"
    logs = run_dir / "logs"
    plots = run_dir / "plots"
    logs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "episode": [1, 2, 3, 4, 5],
        "steps": [100, 120, 140, 160, 180],
        "reward_mean": [0.1, 0.2, 0.3, 0.25, 0.4],
        "success_rate_end": [0, 0.2, 0.4, 0.6, 0.8],
        "deadline_miss_rate": [1, 0.8, 0.6, 0.4, 0.2],
        "time_limit_rate": [1, 1, 1, 0, 0],
        "decision_frac_local": [0.5, 0.4, 0.3, 0.2, 0.1],
        "decision_frac_rsu": [0.4, 0.5, 0.6, 0.7, 0.8],
        "decision_frac_v2v": [0.1, 0.1, 0.1, 0.1, 0.1],
        "v2v_beats_rsu_rate": [0, 0, 0.1, 0.1, 0.2],
        "mean_cost_gap_v2v_minus_rsu": [0.05, 0.04, 0.03, 0.02, 0.01],
        "mean_cost_rsu": [0.5, 0.5, 0.5, 0.5, 0.5],
        "mean_cost_v2v": [0.55, 0.54, 0.53, 0.52, 0.51],
        "approx_kl": [0.01, 0.02, 0.03, 0.02, 0.01],
        "clip_frac": [0.1, 0.2, 0.15, 0.1, 0.05],
        "policy_entropy": [0.5, 0.6, 0.7, 0.65, 0.6],
        "value_loss": [1, 2, 3, 2, 1],
    })
    df.to_csv(logs / "metrics.csv", index=False)
    return run_dir


def test_plot_key_metrics_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = _write_dummy_run(tmp_path)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "plot_key_metrics_v4.py"),
        "--run-dir",
        str(run_dir),
        "--window",
        "3",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    plots = run_dir / "plots"
    required = [
        "01_reward_sum.png",
        "02_success_miss_timelimit.png",
        "03_action_frac_LRV.png",
        "04a_v2v_beats_rsu_rate.png",
        "04b_cost_gap_and_costs.png",
        "05_ppo_kl_clip_entropy.png",
        "06_value_loss.png",
        "07_steps_distribution.png",
        "08_reward_sum_vs_steps.png",
    ]
    for fname in required:
        p = plots / fname
        assert p.exists() and p.stat().st_size > 0, f"{fname} missing"
