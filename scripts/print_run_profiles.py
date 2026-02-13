#!/usr/bin/env python3
"""
Print reproducible Stage A/B/C run profiles and write effective config previews.

This script does not run training. It only:
1) applies env overrides to config classes,
2) prints train.py commands (single seed=42),
3) writes resolved config snapshots under each run_dir.
"""

import copy
import json
import os
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from train import apply_env_overrides


def _snapshot_class_attrs(cls) -> Dict[str, Any]:
    out = {}
    for k, v in cls.__dict__.items():
        if k.startswith("__") or callable(v) or isinstance(v, (staticmethod, classmethod)):
            continue
        out[k] = copy.deepcopy(v)
    return out


def _restore_class_attrs(cls, snap: Dict[str, Any]):
    for k, v in snap.items():
        setattr(cls, k, copy.deepcopy(v))


@contextmanager
def _temp_env(overrides: Dict[str, Any]):
    backup = {}
    for k, v in overrides.items():
        backup[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k in overrides:
            old = backup.get(k)
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _dump_effective_config(run_dir: Path, stage: str, env_overrides: Dict[str, Any], cmd: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "run_dir": str(run_dir),
        "env_overrides": dict(env_overrides),
        "train_command": cmd,
        "system_config_effective": _snapshot_class_attrs(Cfg),
        "train_config_effective": _snapshot_class_attrs(TC),
    }
    out_path = run_dir / "profile_config_preview.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, default=str)
    return out_path


def _build_train_cmd(env_overrides: Dict[str, Any], run_dir: str, episodes: int, max_steps: int, seed: int = 42) -> str:
    env_parts = [f"{k}={shlex.quote(str(v))}" for k, v in env_overrides.items()]
    env_parts.append("DISABLE_AUTO_PLOT=1")
    cmd_parts = [
        "python",
        "train.py",
        "--max-episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--run-dir",
        shlex.quote(run_dir),
        "--exact-run-dir",
        "--disable-baseline-eval",
    ]
    return " ".join(env_parts + cmd_parts)


def main():
    base_cfg = _snapshot_class_attrs(Cfg)
    base_tc = _snapshot_class_attrs(TC)

    profiles = [
        {
            "name": "Stage A",
            "run_dir": "runs/stageA_n20_all_ep1500/seed_42",
            "episodes": 1500,
            "max_steps": 200,
            "env": {
                "NUM_VEHICLES": 20,
                "NUM_RSU": 3,
                "CANDIDATE_MODE": "ALL",
                "V2V_NUM_RB": 4,
                "V2V_RANGE": 250.0,
                "RSU_RANGE": 350.0,
                "DEADLINE_TIGHTENING_MIN": 1.2,
                "DEADLINE_TIGHTENING_MAX": 1.5,
            },
        },
        {
            "name": "Stage B",
            "run_dir": "runs/stageB_n20_interf_ep1500/seed_42",
            "episodes": 1500,
            "max_steps": 200,
            "env": {
                "NUM_VEHICLES": 20,
                "NUM_RSU": 3,
                "CANDIDATE_MODE": "ALL",
                "V2V_NUM_RB": 2,
                "V2V_RANGE": 300.0,
                "RSU_RANGE": 300.0,
                "MIN_DATA": 800000.0,
                "MAX_DATA": 3200000.0,
                "MIN_EDGE_DATA": 400000.0,
                "MAX_EDGE_DATA": 1600000.0,
                "DEADLINE_TIGHTENING_MIN": 1.2,
                "DEADLINE_TIGHTENING_MAX": 1.5,
            },
        },
        {
            "name": "Stage C",
            "run_dir": "runs/stageC_n40_all_ep1500/seed_42",
            "episodes": 1500,
            "max_steps": 200,
            "env": {
                "NUM_VEHICLES": 40,
                "NUM_RSU": 3,
                "CANDIDATE_MODE": "ALL",
                "V2V_NUM_RB": 4,
                "V2V_RANGE": 250.0,
                "RSU_RANGE": 350.0,
                "DEADLINE_TIGHTENING_MIN": 1.2,
                "DEADLINE_TIGHTENING_MAX": 1.5,
            },
        },
    ]

    print("=== Stage Run Profiles (single seed=42) ===")
    for prof in profiles:
        _restore_class_attrs(Cfg, base_cfg)
        _restore_class_attrs(TC, base_tc)

        env_overrides = prof["env"]
        run_dir = prof["run_dir"]
        cmd = _build_train_cmd(
            env_overrides=env_overrides,
            run_dir=run_dir,
            episodes=prof["episodes"],
            max_steps=prof["max_steps"],
            seed=42,
        )
        with _temp_env(env_overrides):
            apply_env_overrides()
            out_path = _dump_effective_config(Path(run_dir), prof["name"], env_overrides, cmd)

        print(f"\n[{prof['name']}]")
        print("env overrides:")
        for k, v in env_overrides.items():
            print(f"  {k}={v}")
        print("train command:")
        print(f"  {cmd}")
        print(f"effective config dump: {out_path}")


if __name__ == "__main__":
    main()

