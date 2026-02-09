import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv


DEFAULT_SEED_COUNT = 50
DEFAULT_STEPS_PER_SEED = 800
DEFAULT_STALL_WINDOW = 50
EPS = 1e-6
LOG_DIR = ROOT_DIR / "logs"


def _set_cfg(**kwargs):
    orig = {k: getattr(Cfg, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(Cfg, k, v)
    return orig


def _restore_cfg(orig):
    for k, v in orig.items():
        setattr(Cfg, k, v)


def _total_comm_bytes(env):
    total = 0.0
    for q in env.txq_v2i.values():
        for job in q:
            total += float(getattr(job, "rem_bytes", 0.0))
    for q in env.txq_v2v.values():
        for job in q:
            total += float(getattr(job, "rem_bytes", 0.0))
    return total


def _queue_counts(env):
    veh_cpu = sum(len(q) for q in env.veh_cpu_q.values())
    rsu_cpu = sum(len(q) for proc in env.rsu_cpu_q.values() for q in proc.values())
    v2i = sum(len(q) for q in env.txq_v2i.values())
    v2v = sum(len(q) for q in env.txq_v2v.values())
    return veh_cpu, rsu_cpu, v2i, v2v


def _active_tx_set_size(env):
    active = set()
    for tx_node, q in env.txq_v2i.items():
        if q:
            active.add(tx_node)
    for tx_node, q in env.txq_v2v.items():
        if q:
            active.add(tx_node)
    return len(active)


def _action_mask_stats(obs_list):
    counts = [int(np.sum(obs["action_mask"])) for obs in obs_list]
    if not counts:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": int(np.min(counts)),
        "mean": float(np.mean(counts)),
        "max": int(np.max(counts)),
    }


def _status_distribution(env):
    counter = Counter()
    for v in env.vehicles:
        for s in v.task_dag.status:
            counter[int(s)] += 1
    return {
        "pending": int(counter.get(0, 0)),
        "ready": int(counter.get(1, 0)),
        "running": int(counter.get(2, 0)),
        "completed": int(counter.get(3, 0)),
    }


def _dump_stall(log_path, payload):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Diagnose stall events in long simulations.")
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_SEED)
    parser.add_argument("--stall-window", type=int, default=DEFAULT_STALL_WINDOW)
    return parser.parse_args()


def main():
    args = _parse_args()
    seed_count = max(int(args.seed_count), 1)
    steps_per_seed = max(int(args.steps), 1)
    stall_window = max(int(args.stall_window), 1)
    seeds = list(range(seed_count))
    orig = _set_cfg(
        VEHICLE_ARRIVAL_RATE=0,
        BW_V2I=20e6,
        REWARD_SCHEME="LEGACY_CFT",
    )
    stall_seeds = 0
    stall_steps = []
    stall_link_counts = Counter()

    try:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            env = VecOffloadingEnv(config=Cfg)
            obs_list, _ = env.reset(seed=seed)

            prev_cycles = env._get_total_W_remaining()
            prev_bytes = _total_comm_bytes(env)
            prev_completed = sum(int(np.sum(v.task_dag.status == 3)) for v in env.vehicles)
            stall_run = 0
            stall_fired = False

            for step in range(steps_per_seed):
                actions = []
                for obs in obs_list:
                    mask = obs["action_mask"]
                    valid = np.where(mask)[0]
                    target = int(rng.choice(valid)) if len(valid) else 0
                    actions.append({"target": target, "power": float(rng.random())})

                obs_list, rewards, dones, truncs, infos = env.step(actions)

                curr_cycles = env._get_total_W_remaining()
                curr_bytes = _total_comm_bytes(env)
                curr_completed = sum(int(np.sum(v.task_dag.status == 3)) for v in env.vehicles)

                delta_cycles = max(prev_cycles - curr_cycles, 0.0)
                delta_bytes = max(prev_bytes - curr_bytes, 0.0)
                delta_completed = max(curr_completed - prev_completed, 0)
                active_tasks = env._get_total_active_tasks()

                if active_tasks > 0 and delta_completed == 0 and delta_cycles <= EPS and delta_bytes <= EPS:
                    stall_run += 1
                else:
                    stall_run = 0

                if stall_run >= stall_window and not stall_fired:
                    veh_cpu, rsu_cpu, v2i_q, v2v_q = _queue_counts(env)
                    active_tx = _active_tx_set_size(env)
                    mask_stats = _action_mask_stats(obs_list)
                    status_dist = _status_distribution(env)

                    payload = {
                        "seed": seed,
                        "step": step,
                        "time": float(env.time),
                        "stall_window": stall_window,
                        "delta_cycles": float(delta_cycles),
                        "delta_bytes": float(delta_bytes),
                        "delta_completed": int(delta_completed),
                        "active_tasks": int(active_tasks),
                        "queues": {
                            "veh_cpu": int(veh_cpu),
                            "rsu_cpu": int(rsu_cpu),
                            "v2i": int(v2i_q),
                            "v2v": int(v2v_q),
                        },
                        "active_tx_set": int(active_tx),
                        "action_mask": mask_stats,
                        "status_dist": status_dist,
                    }
                    log_path = LOG_DIR / f"stall_seed{seed}_step{step}.json"
                    _dump_stall(log_path, payload)

                    stall_fired = True
                    stall_seeds += 1
                    stall_steps.append(step)
                    if v2i_q > 0:
                        stall_link_counts["v2i"] += 1
                    if v2v_q > 0:
                        stall_link_counts["v2v"] += 1
                    if (veh_cpu + rsu_cpu) > 0:
                        stall_link_counts["cpu"] += 1

                prev_cycles = curr_cycles
                prev_bytes = curr_bytes
                prev_completed = curr_completed

                if all(_normalize_flags(dones)) or all(_normalize_flags(truncs)):
                    break

            env.close()

        stall_prob = stall_seeds / max(len(seeds), 1)
        most_common_steps = Counter(stall_steps).most_common(5)

        print("=== Stall Diagnosis Summary ===")
        print(f"seeds={len(seeds)} steps_per_seed={steps_per_seed} window={stall_window}")
        print(f"stall_seeds={stall_seeds} stall_prob={stall_prob:.3f}")
        if most_common_steps:
            print(f"most_common_stall_steps={most_common_steps}")
        else:
            print("most_common_stall_steps=[]")
        print(f"stall_link_counts={dict(stall_link_counts)}")
        print(f"logs_dir={LOG_DIR}")
    finally:
        _restore_cfg(orig)


def _normalize_flags(flags):
    if isinstance(flags, dict):
        return [bool(v) for _, v in sorted(flags.items())]
    if isinstance(flags, (list, np.ndarray)):
        return [bool(v) for v in flags]
    return [bool(flags)]


if __name__ == "__main__":
    main()
