#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return raw


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a modified config_snapshot.json from a base snapshot.")
    ap.add_argument("--base-snapshot", required=True, type=Path)
    ap.add_argument("--out-snapshot", required=True, type=Path)
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help="Override config key as KEY=VALUE; auto-routes to system_config/train_config by existing key.",
    )
    args = ap.parse_args()

    with args.base_snapshot.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sys_cfg = dict(data.get("system_config", {}))
    train_cfg = dict(data.get("train_config", {}))

    for item in args.sets:
        if "=" not in item:
            raise SystemExit(f"Invalid --set item: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        value = _parse_value(raw.strip())
        if key in sys_cfg:
            sys_cfg[key] = value
        elif key in train_cfg:
            train_cfg[key] = value
        else:
            sys_cfg[key] = value

    data["system_config"] = sys_cfg
    data["train_config"] = train_cfg
    # Mark as derived for traceability; keep the rest untouched.
    data["derived_from_snapshot"] = str(args.base_snapshot)

    args.out_snapshot.parent.mkdir(parents=True, exist_ok=True)
    with args.out_snapshot.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
