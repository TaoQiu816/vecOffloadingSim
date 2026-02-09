import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import SystemConfig as Cfg
import train


PRESETS = {
    "E1": {
        "CHAIN_ENABLED": False,
        "REWARD_SCHEME": "LEGACY_CFT",
    },
    "E2": {
        "CHAIN_ENABLED": False,
        "REWARD_SCHEME": "PBRS_KP_V2",
        "PBRS_PHI_MODE": "STATE_ONLY",
    },
    "E3": {
        "CHAIN_ENABLED": True,
        "CHAIN_MODE": "SWITCH",
        "CHAIN_RISK_WEIGHT_DEPOSIT": 0.1,
        "CHAIN_RISK_WEIGHT_FAIL": 0.0,
        "REWARD_SCHEME": "LEGACY_CFT",
    },
    "E4": {
        "CHAIN_ENABLED": True,
        "CHAIN_MODE": "SWITCH",
        "CHAIN_RISK_WEIGHT_DEPOSIT": 0.1,
        "CHAIN_RISK_WEIGHT_FAIL": 0.0,
        "REWARD_SCHEME": "PBRS_KP_V2",
        "PBRS_PHI_MODE": "STATE_ONLY",
    },
}


def apply_preset(preset):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    for key, value in PRESETS[preset].items():
        setattr(Cfg, key, value)


def main():
    parser = argparse.ArgumentParser(description="Run fast-loop presets E1-E4.")
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_AUTO_PLOT", "1")

    for seed in args.seeds:
        apply_preset(args.preset)
        run_dir = os.path.join(args.logdir, args.preset, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        argv = [
            "train.py",
            "--max-episodes",
            str(args.episodes),
            "--seed",
            str(seed),
            "--run-dir",
            run_dir,
        ]
        if args.device:
            argv.extend(["--device", args.device])
        if args.max_steps is not None:
            argv.extend(["--max-steps", str(args.max_steps)])

        old_argv = sys.argv
        try:
            sys.argv = argv
            train.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
