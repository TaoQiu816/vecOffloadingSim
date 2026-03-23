"""
Run AUTHORITATIVE_EVAL_PROTOCOL v1.0 on the fixed RC1 ablation suite.

This wrapper reuses the exact frozen-scene / deterministic / seeds / score-tuple
definition from scripts/run_authoritative_eval.py, but evaluates best_model and
last_model for the four ablation runs.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.train_config import TrainConfig as TC
from scripts import run_authoritative_eval as proto


SUITE_ROOT = os.path.join(ROOT, "runs", "rc1_ablation_1500ep_20260322_180707")
OUT_DIR = os.path.join(SUITE_ROOT, "authoritative_eval")

RUN_SPECS = [
    ("full", "full", "Full-MAPPO"),
    ("wo_dag", "no_dag", "w/o DAG-Feature"),
    ("wo_resource", "no_resource", "w/o Resource-Feature"),
    ("wo_dag_resource", "no_dag_resource", "w/o DAG & Resource"),
]


def _checkpoint_specs() -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    for run_name, ablation_mode, label in RUN_SPECS:
        run_dir = os.path.join(SUITE_ROOT, run_name)
        for ckpt_kind in ("best_model", "last_model"):
            ckpt_path = os.path.join(run_dir, "models", f"{ckpt_kind}.pth")
            specs.append(
                {
                    "run_name": run_name,
                    "ablation_mode": ablation_mode,
                    "label": label,
                    "checkpoint_kind": ckpt_kind,
                    "policy_name": f"{label}::{ckpt_kind}",
                    "checkpoint_path": ckpt_path,
                }
            )
    return specs


def _write_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _score_str(score) -> str:
    return f"({score[0]:.4f}, {score[1]:.4f}, {score[2]:.4f})"


def _generate_md_report(summaries: List[Dict[str, object]], out_dir: str) -> None:
    md_path = os.path.join(out_dir, "ABLATION_AUTHORITATIVE_REPORT.md")
    ranked = sorted(
        summaries,
        key=lambda s: (
            s["score_tuple"][0],
            s["score_tuple"][1],
            s["score_tuple"][2],
        ),
        reverse=True,
    )
    best_per_run: Dict[str, Dict[str, object]] = {}
    for s in summaries:
        run_name = s.get("run_name")
        if not run_name:
            continue
        prev = best_per_run.get(run_name)
        if prev is None or proto.lex_gt(s["score_tuple"], prev["score_tuple"]):
            best_per_run[run_name] = s

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RC1 Ablation AUTHORITATIVE Eval Report\n\n")
        f.write(f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"> Protocol: seeds {proto.SEEDS[0]}-{proto.SEEDS[-1]}, "
            f"episodes={proto.NUM_EPISODES}, MAX_STEPS={proto.MAX_STEPS}, deterministic, frozen scene\n\n"
        )

        f.write("## Summary Table\n\n")
        f.write("| Policy | task_success_rate_B | deadline_miss_rate | mean_cft | subtask_sr | local | rsu | v2v | score S |\n")
        f.write("|------|------:|------:|------:|------:|------:|------:|------:|------|\n")
        for s in ranked:
            f.write(
                f"| {s['policy']} "
                f"| {s['task_success_rate_B_mean']:.4f} "
                f"| {s['deadline_miss_rate_mean']:.4f} "
                f"| {s['mean_cft_mean']:.4f} "
                f"| {s['subtask_success_rate_mean']:.4f} "
                f"| {s['decision_frac_local_mean']:.4f} "
                f"| {s['decision_frac_rsu_mean']:.4f} "
                f"| {s['decision_frac_v2v_mean']:.4f} "
                f"| `{_score_str(s['score_tuple'])}` |\n"
            )

        f.write("\n## Best Checkpoint Per Ablation Run\n\n")
        f.write("| Run | Winner | Score S |\n")
        f.write("|------|------|------|\n")
        for run_name, winner in best_per_run.items():
            f.write(
                f"| {run_name} | {winner['policy']} | `{_score_str(winner['score_tuple'])}` |\n"
            )

        f.write("\n## Overall Ranking\n\n")
        for idx, s in enumerate(ranked, start=1):
            f.write(f"{idx}. `{s['policy']}` — S={_score_str(s['score_tuple'])}\n")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    env = proto.make_env()

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []

    print("\n=== AUTHORITATIVE baselines ===")
    for name, policy_fn in (
        ("Local-Only", proto.policy_local_only),
        ("Greedy-Local", proto.policy_greedy_local),
        ("Legal-Random", proto.policy_legal_random),
    ):
        if name == "Legal-Random":
            import numpy as np

            np.random.seed(9999)
        rows = proto.evaluate(name, env, policy_fn=policy_fn)
        all_rows.extend(rows)
        summaries.append(proto.compute_summary(name, rows))

    print("\n=== AUTHORITATIVE ablation checkpoints ===")
    for spec in _checkpoint_specs():
        ckpt_path = spec["checkpoint_path"]
        if not os.path.exists(ckpt_path):
            print(f"[skip] missing checkpoint: {ckpt_path}")
            continue
        TC.ABLATION_MODE = spec["ablation_mode"]
        print(
            f"\n=== {spec['policy_name']} "
            f"(ABLATION_MODE={spec['ablation_mode']}) ==="
        )
        agent = proto.load_agent(ckpt_path)
        rows = proto.evaluate(spec["policy_name"], env, agent=agent)
        for row in rows:
            row["run_name"] = spec["run_name"]
            row["ablation_mode"] = spec["ablation_mode"]
            row["checkpoint_kind"] = spec["checkpoint_kind"]
        all_rows.extend(rows)
        summary = proto.compute_summary(spec["policy_name"], rows)
        summary["run_name"] = spec["run_name"]
        summary["ablation_mode"] = spec["ablation_mode"]
        summary["checkpoint_kind"] = spec["checkpoint_kind"]
        summaries.append(summary)

    env.close()

    csv_path = os.path.join(OUT_DIR, "formal_eval_metrics.csv")
    json_path = os.path.join(OUT_DIR, "formal_eval_summary.json")
    _write_csv(all_rows, csv_path)
    _write_json(
        [
            {**s, "score_tuple": list(s["score_tuple"])}
            for s in summaries
        ],
        json_path,
    )
    _generate_md_report(summaries, OUT_DIR)

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD: {os.path.join(OUT_DIR, 'ABLATION_AUTHORITATIVE_REPORT.md')}")


if __name__ == "__main__":
    main()
