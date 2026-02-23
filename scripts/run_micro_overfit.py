#!/usr/bin/env python3
"""
Run a deterministic micro-overfitting test for the VEC MAPPO stack.

This wrapper applies a fixed single-sample environment profile before training starts.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train
from configs.overfit_config import apply_overfit_config


def _patch_overrides():
    original_apply_env_overrides = train.apply_env_overrides

    def _patched():
        original_apply_env_overrides()
        info = apply_overfit_config(train.Cfg, train.TC)
        print(
            "[MicroOverfit] applied deterministic profile: "
            f"NUM_VEHICLES={train.Cfg.NUM_VEHICLES}, NUM_RSU={train.Cfg.NUM_RSU}, "
            f"MAX_TARGETS={train.Cfg.MAX_TARGETS}, DAG={info['workflow_path']}",
            flush=True,
        )

    train.apply_env_overrides = _patched


def _inject_default_args():
    # Keep wrapper simple: only append safe defaults if caller did not specify them.
    args = sys.argv[1:]
    has = set(args)
    if "--disable-baseline-eval" not in has and "--enable-baseline-eval" not in has:
        sys.argv.append("--disable-baseline-eval")
    if "--device" not in has:
        sys.argv.extend(["--device", "cpu"])
    if "--max-episodes" not in has:
        sys.argv.extend(["--max-episodes", "300"])
    if "--max-steps" not in has:
        sys.argv.extend(["--max-steps", "40"])
    if "--log-interval" not in has:
        sys.argv.extend(["--log-interval", "10"])
    if "--eval-interval" not in has:
        sys.argv.extend(["--eval-interval", "1000"])
    if "--save-interval" not in has:
        sys.argv.extend(["--save-interval", "1000"])


def main():
    _patch_overrides()
    _inject_default_args()
    train.main()


if __name__ == "__main__":
    main()
