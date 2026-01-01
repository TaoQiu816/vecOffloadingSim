import os
import sys
import subprocess
from pathlib import Path


def _resolve_run_dir(base_dir: Path) -> Path:
    candidates = sorted(
        base_dir.parent.glob(base_dir.name + "*"),
        key=lambda p: p.stat().st_mtime
    )
    assert candidates, f"no run dir found for prefix {base_dir}"
    return candidates[-1]


def test_train_console_compact_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_compact"
    env = os.environ.copy()
    env.update({
        "RUN_DIR": str(run_dir),
        "RUN_ID": "test_compact",
        "MAX_EPISODES": "1",
        "DEVICE_NAME": "cpu",
        "DISABLE_BASELINE_EVAL": "1",
        "DISABLE_AUTO_PLOT": "1",
        "EPISODE_JSONL_STDOUT": "0",
    })
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    resolved_dir = _resolve_run_dir(run_dir)
    csv_path = resolved_dir / "logs" / "metrics.csv"
    jsonl_candidates = [
        resolved_dir / "logs" / "run.jsonl",
        resolved_dir / "logs" / "env_reward.jsonl",
    ]
    jsonl_path = next((p for p in jsonl_candidates if p.exists()), None)
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert jsonl_path is not None and jsonl_path.stat().st_size > 0

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
