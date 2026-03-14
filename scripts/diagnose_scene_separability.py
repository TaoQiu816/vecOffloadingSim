#!/usr/bin/env python3
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from scripts.diagnose_phi_alignment import collect_snapshots, write_csv


def _best_rows(rows):
    best = {}
    for row in rows:
        sid = int(row["snapshot_id"])
        if sid not in best or float(row["J"]) < float(best[sid]["J"]):
            best[sid] = row
    return [best[sid] for sid in sorted(best.keys())]


def _dominant_gap_component(rows, best_rows):
    counts = Counter()
    for best in best_rows:
        sid = int(best["snapshot_id"])
        peers = [r for r in rows if int(r["snapshot_id"]) == sid and str(r["target"]) != str(best["target"])]
        if not peers:
            continue
        alt = min(peers, key=lambda r: float(r["J"]))
        deltas = {
            "cpu_exec": float(alt["cpu_exec"]) - float(best["cpu_exec"]),
            "cpu_wait": float(alt["cpu_wait"]) - float(best["cpu_wait"]),
            "tx_time": float(alt["tx_time"]) - float(best["tx_time"]),
            "comm_wait": float(alt["comm_wait"]) - float(best["comm_wait"]),
            "contact_penalty": float(alt["contact_penalty"]) - float(best["contact_penalty"]),
            "delta_ext": float(alt["delta_ext"]) - float(best["delta_ext"]),
        }
        dominant = max(deltas.items(), key=lambda kv: abs(kv[1]))[0]
        counts[dominant] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="100-snapshot scene separability gate under the current sampling protocol.")
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--min-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/scene_gate_20260313/scene_gate_snapshots.csv")
    args = parser.parse_args()

    env = VecOffloadingEnv(Cfg)
    rows = collect_snapshots(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        snapshots=args.snapshots,
        seed=args.seed,
        min_step=args.min_step,
        max_snapshots_per_episode=max(1, math.ceil(args.snapshots / max(args.episodes, 1))),
    )
    out_path = Path(args.out)
    write_csv(rows, out_path)

    best_rows = _best_rows(rows)
    total = max(len(best_rows), 1)
    mode_counts = Counter(str(row["mode"]) for row in best_rows)
    dom_counts = _dominant_gap_component(rows, best_rows)
    workload_counts = Counter(str(row.get("workload_level", "unknown")) for row in best_rows)
    helper_counts = Counter(str(row.get("helper_availability", "unknown")) for row in best_rows)
    rsu_counts = Counter(str(row.get("rsu_accessibility_or_load", "unknown")) for row in best_rows)

    local_ratio = float(mode_counts.get("local", 0) / total)
    rsu_ratio = float(mode_counts.get("rsu", 0) / total)
    v2v_ratio = float(mode_counts.get("v2v", 0) / total)
    passed = (
        0.10 <= local_ratio <= 0.30
        and 0.20 <= v2v_ratio <= 0.40
        and 0.30 <= rsu_ratio <= 0.60
    )

    print(f"Local_best_ratio {local_ratio:.3f}")
    print(f"RSU_best_ratio {rsu_ratio:.3f}")
    print(f"V2V_best_ratio {v2v_ratio:.3f}")
    print("best_mode_counts " + " ".join(f"{k}={mode_counts.get(k, 0)}" for k in ("local", "rsu", "v2v")))
    print("dominant_gap_component " + " ".join(f"{k}={dom_counts.get(k, 0)}" for k in ("cpu_exec", "cpu_wait", "tx_time", "comm_wait", "contact_penalty", "delta_ext")))
    print("workload_level_counts " + " ".join(f"{k}={workload_counts.get(k, 0)}" for k in ("light", "medium", "heavy")))
    print("helper_availability_counts " + " ".join(f"{k}={helper_counts.get(k, 0)}" for k in ("low", "medium", "high")))
    print("rsu_accessibility_or_load_counts " + " ".join(f"{k}={rsu_counts.get(k, 0)}" for k in ("low", "medium", "high")))
    print(f"gate_target_pass {int(passed)}")
    print(f"csv={out_path}")


if __name__ == "__main__":
    main()
