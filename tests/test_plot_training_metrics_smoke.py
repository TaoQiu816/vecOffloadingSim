import subprocess
import sys
from pathlib import Path


def test_plot_training_metrics_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    csv_path = metrics_dir / "train_metrics.csv"
    csv_path.write_text(
        "episode,reward_abs.mean,success_rate_end,subtask_success_rate,deadline_miss_rate,decision_frac_local,decision_frac_rsu,decision_frac_v2v,clip_hit_ratio\n"
        "1,0.5,0.1,0.2,0.3,0.4,0.3,0.3,0.01\n"
        "2,0.6,0.2,0.3,0.2,0.35,0.35,0.3,0.02\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/plot_training_metrics.py", "--run_dir", str(run_dir)],
        cwd=repo_root,
        check=True,
    )

    plots_dir = run_dir / "plots"
    pngs = list(plots_dir.glob("*.png"))
    assert len(pngs) >= 3
