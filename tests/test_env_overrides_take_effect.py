import json
import os
import subprocess
import sys
from pathlib import Path


def test_env_overrides_take_effect(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    env = os.environ.copy()
    env.update(
        {
            "CFG_PROFILE": "train_ready_v4",
            "BONUS_MODE": "none",
            "DEVICE_NAME": "cpu",
            "SEED": "7",
            "RUN_DIR": str(run_dir),
            "RUN_ID": "override_smoke",
            "MAX_EPISODES": "2",
            "MAX_STEPS": "10",
            "DISABLE_BASELINE_EVAL": "1",
            "DISABLE_AUTO_PLOT": "1",
            # overrides
            "GAMMA": "0.995",
            "CLIP_PARAM": "0.10",
            "ENTROPY_COEF": "0.005",
            "LR_ACTOR": "0.0002",
            "LR_CRITIC": "0.0003",
            "MINI_BATCH_SIZE": "128",
            "USE_LOGIT_BIAS": "0",
            "VEHICLE_ARRIVAL_RATE": "0.0",
            "RSU_NUM_PROCESSORS": "2",
            "BW_V2V": "80000000",
            "MIN_CPU": "1000000000",
            "MAX_CPU": "4000000000",
        }
    )

    subprocess.run(
        [sys.executable, "train.py"],
        cwd=repo,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # resolve run dir
    candidates = sorted(run_dir.parent.glob(run_dir.name + "*"), key=lambda p: p.stat().st_mtime)
    assert candidates, "run dir not created"
    rd = candidates[-1]
    snapshot_path = rd / "logs" / "config_snapshot.json"
    assert snapshot_path.exists()
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tc = snap["train_config"]
    cfg = snap["system_config"]
    assert abs(float(tc["GAMMA"]) - 0.995) < 1e-6
    assert abs(float(tc["CLIP_PARAM"]) - 0.10) < 1e-6
    assert abs(float(tc["ENTROPY_COEF"]) - 0.005) < 1e-6
    assert abs(float(tc["LR_ACTOR"]) - 0.0002) < 1e-6
    assert abs(float(tc["LR_CRITIC"]) - 0.0003) < 1e-6
    assert int(tc["MINI_BATCH_SIZE"]) == 128
    assert tc["USE_LOGIT_BIAS"] in (False, 0, "False")
    assert abs(float(cfg["VEHICLE_ARRIVAL_RATE"]) - 0.0) < 1e-6
    assert int(cfg["RSU_NUM_PROCESSORS"]) == 2
    assert abs(float(cfg["BW_V2V"]) - 80000000.0) < 1e-3
    assert abs(float(cfg["MIN_VEHICLE_CPU_FREQ"]) - 1e9) < 1e-3
    assert abs(float(cfg["MAX_VEHICLE_CPU_FREQ"]) - 4e9) < 1e-3
