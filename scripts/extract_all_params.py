#!/usr/bin/env python3
"""Purpose: extract full config/train parameters into machine-readable files.
Inputs: CLI args; respects CFG_PROFILE/REWARD_MODE env vars if set.
Outputs: JSON/CSV under results_dbg/extract_all_params by default.
Example: python scripts/extract_all_params.py --out_dir results_dbg/params
"""
import argparse
import os
import sys
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _add_repo_root():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)


def _collect_class_params(cls):
    params = {}
    for key in dir(cls):
        if key.startswith("_"):
            continue
        val = getattr(cls, key)
        if callable(val):
            continue
        params[key] = val
    return params


def _impact_hint(name):
    key = name.upper()
    if "MAX_STEPS" in key:
        return "增大回合长度，减少truncated"
    if key in ("DT",):
        return "增大单步推进量但降低时间精度"
    if "NUM_VEHICLES" in key or "ARRIVAL" in key:
        return "增大车流量，提升竞争与拥塞"
    if "BW" in key or "RATE" in key or "F_" in key or "CPU" in key:
        return "增大吞吐/算力，降低传输或计算时延"
    if "RANGE" in key:
        return "增大覆盖范围，提高可用性"
    if "QUEUE" in key or "BUFFER" in key or "LIMIT" in key:
        return "增大容量，降低溢出但可能增加等待"
    if "DEADLINE" in key:
        return "增大deadline，更易完成"
    if "WEIGHT" in key or "SCALE" in key:
        return "增大奖励权重，强化对应项"
    if "ENTROPY" in key:
        return "增大探索，降低策略坍缩风险"
    return "影响依赖具体逻辑"


def _categorize_cfg_params(params):
    groups = {
        "A) 场景/车辆/时间": {},
        "B) RSU": {},
        "C) V2V/V2I/信道": {},
        "D) DAG": {},
        "E) 队列": {},
        "F) Reward": {},
        "Other": {},
    }
    for name, val in params.items():
        key = name.upper()
        if any(tok in key for tok in ("REWARD", "BONUS", "PENALTY", "DELTA_CFT", "ENERGY_IN_DELTA_CFT")):
            groups["F) Reward"][name] = val
        elif "RSU" in key:
            groups["B) RSU"][name] = val
        elif any(tok in key for tok in ("V2V", "V2I", "BW", "ALPHA", "NOISE", "INTERFERENCE", "PATH")):
            groups["C) V2V/V2I/信道"][name] = val
        elif any(tok in key for tok in ("NODES", "COMP", "DATA", "DEADLINE", "DAG")):
            groups["D) DAG"][name] = val
        elif any(tok in key for tok in ("QUEUE", "BUFFER")):
            groups["E) 队列"][name] = val
        elif any(tok in key for tok in ("MAP", "DT", "NUM_VEHICLES", "ARRIVAL", "VEL", "LANE", "MAX_STEPS", "SEED")):
            groups["A) 场景/车辆/时间"][name] = val
        else:
            groups["Other"][name] = val
    return groups


def _render_section(title, params):
    lines = [f"#### {title}", "", "| 参数 | 默认值 | 影响方向 |", "|---|---|---|"]
    for name in sorted(params.keys()):
        val = params[name]
        lines.append(f"| `{name}` | `{val}` | {_impact_hint(name)} |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results_dbg/param_inventory")
    args = parser.parse_args()

    _add_repo_root()
    from configs.config import SystemConfig as Cfg
    from configs.train_config import TrainConfig as TC

    cfg_params = _collect_class_params(Cfg)
    tc_params = _collect_class_params(TC)

    grouped_cfg = _categorize_cfg_params(cfg_params)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "param_inventory.md"

    lines = ["# Parameter Inventory", ""]
    for group, params in grouped_cfg.items():
        if not params:
            continue
        lines.extend(_render_section(group, params))

    lines.extend(_render_section("G) Model/Train (train_config.py)", tc_params))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
