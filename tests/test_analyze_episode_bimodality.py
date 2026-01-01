import subprocess
import sys
from pathlib import Path


def test_analyze_episode_bimodality(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_bi"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    metrics_csv = logs / "metrics.csv"
    metrics_csv.write_text(
        "episode,steps,reward_mean,task_success_rate,deadline_miss_rate,decision_frac_rsu,delta_cft_rem_mean\n"
        "1,100,0.5,0.4,0.6,0.3,0.1\n"
        "2,300,0.1,0.6,0.4,0.5,0.05\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "scripts/analyze_episode_bimodality.py", "--run_dir", str(run_dir)],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout
    assert "short_episode_ratio" in out
    assert "[SHORT]" in out and "[NORMAL]" in out
