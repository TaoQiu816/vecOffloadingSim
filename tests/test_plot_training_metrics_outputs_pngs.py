import subprocess
import sys
from pathlib import Path


def test_plot_training_metrics_outputs_pngs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_plots"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    csv_path = logs_dir / "metrics.csv"
    csv_path.write_text(
        "episode,reward_mean,reward_p95,delta_cft_rem_mean,mean_cft,success_rate_end,task_success_rate,subtask_success_rate,"
        "deadline_miss_rate,illegal_action_rate,hard_trigger_rate,decision_frac_local,decision_frac_rsu,decision_frac_v2v,"
        "entropy,approx_kl,clip_frac,policy_loss,value_loss,total_loss,power_ratio_mean,power_ratio_p95\n"
        "1,0.5,0.6,0.1,5.0,0.1,0.2,0.3,0.3,0.01,0.02,0.4,0.3,0.3,0.9,0.01,0.02,0.8,1.1,2.0,0.4,0.6\n"
        "2,0.6,0.7,0.2,4.5,0.2,0.3,0.4,0.2,0.02,0.03,0.35,0.35,0.3,0.8,0.02,0.03,0.7,1.0,1.8,0.45,0.65\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/plot_training_metrics.py", "--run_dir", str(run_dir)],
        cwd=repo_root,
        check=True,
    )

    plots_dir = run_dir / "plots"
    expected = {
        "reward_mean.png",
        "delta_cft_rem_mean.png",
        "mean_cft.png",
        "success_rates.png",
        "deadline_miss_rate.png",
        "safety_rates.png",
        "decision_fracs.png",
        "ppo_diagnostics.png",
        "losses.png",
        "power_ratio_mean.png",
    }
    pngs = {p.name for p in plots_dir.glob("*.png")}
    assert expected.issubset(pngs)
    for name in expected:
        path = plots_dir / name
        assert path.exists() and path.stat().st_size > 0
