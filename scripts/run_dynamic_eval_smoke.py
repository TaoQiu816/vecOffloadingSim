#!/usr/bin/env python3
"""
Run a dynamic VEC training smoke/mini-run with the paper-style experiment profile.

This wrapper applies `configs.exp_dynamic_config.apply_exp_dynamic_config` before train.main().
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train
from configs.exp_dynamic_config import apply_exp_dynamic_config


def _patch_overrides():
    original_apply_env_overrides = train.apply_env_overrides

    def _patched():
        original_apply_env_overrides()
        info = apply_exp_dynamic_config(train.Cfg, train.TC)
        print(
            "[DynamicExp] applied profile: "
            f"veh={info['num_vehicles']} rsu={info['num_rsu']} "
            f"targets={info['max_targets']} dag={info['dag_source']}",
            flush=True,
        )

    train.apply_env_overrides = _patched


def _inject_default_args():
    args = sys.argv[1:]
    has = set(args)
    if "--device" not in has:
        sys.argv.extend(["--device", "cpu"])
    if "--enable-baseline-eval" not in has and "--disable-baseline-eval" not in has:
        sys.argv.append("--enable-baseline-eval")
    if "--max-episodes" not in has:
        sys.argv.extend(["--max-episodes", "2000"])
    if "--max-steps" not in has:
        sys.argv.extend(["--max-steps", "120"])
    if "--log-interval" not in has:
        sys.argv.extend(["--log-interval", "20"])
    if "--eval-interval" not in has:
        sys.argv.extend(["--eval-interval", "100"])
    if "--save-interval" not in has:
        sys.argv.extend(["--save-interval", "200"])


def main():
    _patch_overrides()
    _inject_default_args()
    train.main()


if __name__ == "__main__":
    main()

