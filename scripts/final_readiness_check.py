#!/usr/bin/env python3
"""Purpose: run a minimal readiness smoke check for training pipelines.
Inputs: CLI args for episodes/steps/seed/reward_mode; uses env vars if set.
Outputs: results_dbg/final_readiness/* plus stdout status lines.
Example: python scripts/final_readiness_check.py --episodes 2 --steps 50
"""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from baselines.random_policy import RandomPolicy


def _run(cmd, check=True):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _check_file(path, min_size=1):
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size >= min_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results_dbg/final_readiness")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--audit", type=int, default=1)
    parser.add_argument("--reward_mode", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--run_delta_cft_audit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    summarize_dir = out_dir / "summarize"

    _ensure_dir(out_dir)
    _ensure_dir(log_dir)

    # resolve reward_mode/seed with CLI priority
    env_reward = os.environ.get("REWARD_MODE")
    env_seed = os.environ.get("SEED")
    effective_reward_mode = args.reward_mode or env_reward or Cfg.REWARD_MODE
    if args.seed is not None:
        effective_seed = args.seed
    elif env_seed is not None:
        effective_seed = int(env_seed)
    else:
        effective_seed = int(Cfg.SEED)

    # env vars for audit + jsonl logging
    if args.audit:
        os.environ["AUDIT_ASSERTS"] = "1"
    else:
        os.environ.pop("AUDIT_ASSERTS", None)
    os.environ["REWARD_JSONL_PATH"] = str(log_dir / "run.jsonl")
    os.environ["RUN_ID"] = "final_readiness"
    os.environ["MAX_EPISODES"] = str(args.episodes)
    os.environ["REWARD_MODE"] = str(effective_reward_mode)
    os.environ["SEED"] = str(effective_seed)

    orig_reward_mode = Cfg.REWARD_MODE
    orig_seed = Cfg.SEED
    if effective_reward_mode:
        Cfg.REWARD_MODE = effective_reward_mode
    Cfg.SEED = int(effective_seed)

    results = []

    try:
        # (1) minimal rollout
        env = VecOffloadingEnv()
        policy = RandomPolicy(seed=effective_seed)
        rollout_ok = True
        for ep in range(args.episodes):
            obs_list, _ = env.reset(seed=effective_seed + ep)
            done = False
            for _ in range(args.steps):
                actions = policy.select_action(obs_list)
                if len(actions) != len(obs_list) or len(obs_list) != len(env.vehicles):
                    rollout_ok = False
                    break
                # enforce obs_stamp when audit is on
                if args.audit:
                    for i, act in enumerate(actions):
                        if "obs_stamp" in obs_list[i] and "obs_stamp" not in act:
                            act["obs_stamp"] = int(obs_list[i]["obs_stamp"])
                obs_list, rewards, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated
                if done:
                    break
            if not done:
                # ensure JSONL line written for short episodes
                env._log_episode_stats(terminated=False, truncated=True)
            if not rollout_ok:
                break
        results.append(("rollout", rollout_ok, str(log_dir / "run.jsonl")))

        # JSONL self-check: reward_mode/seed must match this run
        jsonl_ok = True
        jsonl_path = log_dir / "run.jsonl"
        last_payload = None
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_payload = line
        if last_payload is None:
            jsonl_ok = False
        else:
            try:
                payload = json.loads(last_payload)
                if payload.get("reward_mode") != effective_reward_mode:
                    jsonl_ok = False
                if int(payload.get("seed", -1)) != int(effective_seed):
                    jsonl_ok = False
            except Exception:
                jsonl_ok = False
        results.append(("jsonl_check", jsonl_ok, str(jsonl_path)))

        # (2) summarize
        summarize_ok = True
        try:
            _run([
                sys.executable, "scripts/summarize_ablation.py",
                "--log_dir", str(log_dir),
                "--out_dir", str(summarize_dir)
            ])
        except Exception:
            summarize_ok = False
        summary_md = summarize_dir / "ablation_summary.md"
        summary_csv = summarize_dir / "ablation_summary.csv"
        if summarize_ok:
            summarize_ok = _check_file(summary_md) and _check_file(summary_csv)
        if summarize_ok:
            content = summary_md.read_text(encoding="utf-8")
            if "Excluded Files" not in content or "Missing Fields Summary" not in content:
                summarize_ok = False
        results.append(("summarize", summarize_ok, str(summary_md)))

        # (3) decision dominance audit (no ckpt)
        dd_out = out_dir / "decision_dominance" / "no_ckpt"
        _ensure_dir(dd_out)
        dd_ok = True
        try:
            _run([
                sys.executable, "scripts/decision_dominance_audit.py",
                "--episodes", "20",
                "--steps", "50",
                "--seed", str(effective_seed),
                "--out_dir", str(dd_out),
                "--policy_rollout", "type_balanced_random"
            ])
        except Exception:
            dd_ok = False
        dd_md = dd_out / "decision_dominance_audit.md"
        dd_csv = dd_out / "decision_dominance_audit.csv"
        if dd_ok:
            dd_ok = _check_file(dd_md) and _check_file(dd_csv)
        if dd_ok:
            content = dd_md.read_text(encoding="utf-8")
            if "Argmin Distribution" not in content:
                dd_ok = False
        results.append(("decision_dominance", dd_ok, str(dd_md)))

        # (4) delta_cft audit (conditional)
        delta_cft_ok = True
        if effective_reward_mode == "delta_cft" or args.run_delta_cft_audit:
            delta_out = out_dir / "delta_cft_audit"
            _ensure_dir(delta_out)
            try:
                _run([
                    sys.executable, "scripts/delta_cft_audit.py",
                    "--episodes", "2",
                    "--steps", "10",
                "--seed", str(effective_seed),
                "--out_dir", str(delta_out)
            ])
            except Exception:
                delta_cft_ok = False
            delta_md = delta_out / "delta_cft_audit.md"
            delta_csv = delta_out / "delta_cft_audit.csv"
            if delta_cft_ok:
                delta_cft_ok = _check_file(delta_md) and _check_file(delta_csv)
            results.append(("delta_cft_audit", delta_cft_ok, str(delta_md)))

    finally:
        Cfg.REWARD_MODE = orig_reward_mode
        Cfg.SEED = orig_seed

    all_ok = True
    for name, ok, path in results:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {path}")
        if not ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
