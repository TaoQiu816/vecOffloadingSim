import json
import subprocess
import sys
from pathlib import Path


def test_reward_mode_logged_correctly_delta_cft(tmp_path):
    out_dir = tmp_path / "final_readiness"
    cmd = [
        sys.executable,
        "scripts/final_readiness_check.py",
        "--out_dir",
        str(out_dir),
        "--episodes",
        "1",
        "--steps",
        "5",
        "--seed",
        "7",
        "--reward_mode",
        "delta_cft",
        "--audit",
        "1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, proc.stderr

    jsonl_path = out_dir / "logs" / "run.jsonl"
    assert jsonl_path.exists()
    last_line = None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    assert last_line is not None
    payload = json.loads(last_line)
    assert payload.get("reward_mode") == "delta_cft"
    assert int(payload.get("seed", -1)) == 7
