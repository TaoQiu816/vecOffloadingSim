#!/usr/bin/env python3
import os, sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(str(_ROOT))

import argparse
import csv
import os
import random

import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _sample_actions(obs_list, rng):
    actions = []
    decision_counts = {"local": 0, "rsu": 0, "v2v": 0}
    for obs in obs_list:
        mask = obs.get("target_mask", [])
        valid = [i for i, m in enumerate(mask) if m]
        if not valid:
            tgt = 0
        else:
            tgt = int(rng.choice(valid))
        if tgt == 0:
            decision_counts["local"] += 1
        elif tgt == 1:
            decision_counts["rsu"] += 1
        else:
            decision_counts["v2v"] += 1
        actions.append({"target": tgt, "power": 1.0})
    return actions, decision_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results_dbg/delta_cft_audit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "delta_cft_audit.csv")
    md_path = os.path.join(args.out_dir, "delta_cft_audit.md")

    orig_reward_mode = Cfg.REWARD_MODE
    try:
        Cfg.REWARD_MODE = "delta_cft"
        env = VecOffloadingEnv()
        env.reset(seed=args.seed)
        _set_seed(args.seed)
        rng = random.Random(args.seed)

        rows = []
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            for step_idx in range(args.steps):
                ids_prev = [v.id for v in env.vehicles]
                t_prev = env.time
                cft_prev_abs = env._compute_mean_cft_pi0(
                    snapshot_time=t_prev,
                    v2i_user_count=env._estimate_v2i_users(),
                    vehicle_ids=ids_prev
                )
                cft_prev_rem = max(cft_prev_abs - t_prev, 0.0)
                obs_list = env._get_obs()
                actions, counts = _sample_actions(obs_list, rng)
                # AUDIT_ASSERTS: propagate obs_stamp into actions
                for i, act in enumerate(actions):
                    if not act:
                        continue
                    obs = obs_list[i] if i < len(obs_list) else None
                    if isinstance(obs, dict) and 'obs_stamp' in obs and 'obs_stamp' not in act:
                        act['obs_stamp'] = int(obs['obs_stamp'])
                _, rewards, terminated, truncated, _ = env.step(actions)
                t_curr = env.time
                cft_curr_abs = env._compute_mean_cft_pi0(
                    snapshot_time=t_curr,
                    v2i_user_count=env._estimate_v2i_users(),
                    vehicle_ids=ids_prev
                )
                cft_curr_rem = max(cft_curr_abs - t_curr, 0.0)
                if Cfg.DELTA_CFT_REF_MODE == "prev":
                    t_ref = max(cft_prev_rem, Cfg.DELTA_CFT_REF_EPS)
                else:
                    t_ref = max(Cfg.DELTA_CFT_REF_CONST, Cfg.DELTA_CFT_REF_EPS)
                delta_cft = (cft_prev_rem - cft_curr_rem) / t_ref
                rows.append({
                    "episode": ep,
                    "step": step_idx,
                    "cft_prev_abs": float(cft_prev_abs),
                    "cft_curr_abs": float(cft_curr_abs),
                    "cft_prev_rem": float(cft_prev_rem),
                    "cft_curr_rem": float(cft_curr_rem),
                    "delta_cft_rem": float(delta_cft),
                    "reward_mean": float(sum(rewards) / max(len(rewards), 1)),
                    "dec_local": counts["local"],
                    "dec_rsu": counts["rsu"],
                    "dec_v2v": counts["v2v"],
                })
                if terminated or truncated:
                    break

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Delta CFT Audit\n\n")
            f.write("This audit logs remaining CFT (CFT_abs - t) on a stable vehicle set per step.\n\n")
            f.write("| episode | step | cft_prev_rem | cft_curr_rem | delta_cft_rem | reward_mean | dec_local | dec_rsu | dec_v2v |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for row in rows[:50]:
                f.write(
                    f"| {row['episode']} | {row['step']} | {row['cft_prev_rem']:.4f} | {row['cft_curr_rem']:.4f} | "
                    f"{row['delta_cft_rem']:.6f} | {row['reward_mean']:.6f} | {row['dec_local']} | "
                    f"{row['dec_rsu']} | {row['dec_v2v']} |\\n"
                )
            f.write("\\n")

        print(f"[delta_cft_audit] wrote {csv_path}")
        print(f"[delta_cft_audit] wrote {md_path}")
    finally:
        Cfg.REWARD_MODE = orig_reward_mode


if __name__ == "__main__":
    main()
