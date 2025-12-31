import subprocess
import sys
from pathlib import Path


def test_final_readiness_check_smoke(tmp_path):
    out_dir = tmp_path / "final_readiness"
    cmd = [
        sys.executable,
        "scripts/final_readiness_check.py",
        "--out_dir",
        str(out_dir),
        "--episodes",
        "1",
        "--steps",
        "20",
        "--seed",
        "0",
        "--audit",
        "1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    summary_md = out_dir / "summarize" / "ablation_summary.md"
    summary_csv = out_dir / "summarize" / "ablation_summary.csv"
    dd_md = out_dir / "decision_dominance" / "no_ckpt" / "decision_dominance_audit.md"
    dd_csv = out_dir / "decision_dominance" / "no_ckpt" / "decision_dominance_audit.csv"

    for path in (summary_md, summary_csv, dd_md, dd_csv):
        assert path.exists() and path.stat().st_size > 0
