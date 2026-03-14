#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.py"
RUN_DIR = ROOT / "runs" / "exogenous_gate_20260314" / "helper_v2v_sweep"
GATE1_SNAPSHOTS = 40
GATE1_EPISODES = 8
GATE1_MAX_STEPS = 40
GATE1_MIN_STEP = 2
GATE2_EPISODES = 9


PROB_LEVELS = {
    "A": {"low": 0.02, "medium": 0.04, "high": 0.06},
    "B": {"low": 0.025, "medium": 0.045, "high": 0.065},
    "C": {"low": 0.03, "medium": 0.05, "high": 0.07},
}

CPU_LEVELS = {
    "L1": {
        "low": (2.1e9, 2.7e9),
        "medium": (2.2e9, 2.8e9),
        "high": (2.3e9, 2.9e9),
        "high_scale": 0.96,
    },
    "L2": {
        "low": (2.2e9, 2.8e9),
        "medium": (2.3e9, 2.9e9),
        "high": (2.4e9, 3.0e9),
        "high_scale": 0.97,
    },
}

STABILITY_LEVELS = {
    "S1": 3.2,
    "S2": 3.6,
}

HEAVY_RELAX_LEVELS = {
    "H0": {"alpha_add": 0.0, "slack_add": 0.0},
    "H1": {"alpha_add": 0.05, "slack_add": 0.05},
    "H2": {"alpha_add": 0.10, "slack_add": 0.10},
}

BASE_HEAVY_ALPHA = (2.10, 2.50)
BASE_HEAVY_SLACK = 1.15


def _run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc.stdout


def _replace_once(text, pattern, repl):
    new_text, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise RuntimeError(f"failed to replace pattern: {pattern}")
    return new_text


def _regular_prob(weak, helper):
    return 1.0 - weak - helper


def _format_prob(val):
    s = f"{val:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _format_tuple(pair):
    return f"({pair[0]:.1e}, {pair[1]:.1e})"


def _apply_combo_to_text(base_text, prob_key, cpu_key, stability_key, heavy_key):
    probs = PROB_LEVELS[prob_key]
    cpus = CPU_LEVELS[cpu_key]
    speed_std = STABILITY_LEVELS[stability_key]
    heavy_relax = HEAVY_RELAX_LEVELS[heavy_key]

    levels = [("low", 0.22), ("medium", 0.20), ("high", 0.18)]
    text = base_text
    for level, weak in levels:
        regular = _regular_prob(weak, probs[level])
        pattern = rf'("{level}": \{{\n\s+"role_probs": \{{"weak": {weak:.2f}, "regular": )[0-9.]+(, "helper": )[0-9.]+(\}},)'
        repl = rf'\g<1>{_format_prob(regular)}\g<2>{_format_prob(probs[level])}\g<3>'
        text = _replace_once(text, pattern, repl)

        cpu_pattern = rf'("{level}": \{{\n\s+"role_probs": \{{.*?\n\s+"role_cpu": \{{\n\s+"weak": \([^)]+\),\n\s+"regular": \([^)]+\),\n\s+"helper": )\([^)]+\)(,\n\s+\}},)'
        text = _replace_once(text, cpu_pattern, rf'\g<1>{_format_tuple(cpus[level])}\g<2>')

    text = _replace_once(
        text,
        r'("high": \{\n\s+"role_probs": \{.*?\n\s+"role_cpu": \{.*?\n\s+\},\n\s+"helper_cpu_scale": )[0-9.]+(,\n\s+\},)',
        rf'\g<1>{CPU_LEVELS[cpu_key]["high_scale"]:.2f}\g<2>',
    )
    text = _replace_once(
        text,
        r'("high": \{\n\s+"speed_mean": 9\.4,\n\s+"speed_std": )[0-9.]+(,\n\s+\},)',
        rf'\g<1>{speed_std:.1f}\g<2>',
    )

    heavy_alpha = (BASE_HEAVY_ALPHA[0] + heavy_relax["alpha_add"], BASE_HEAVY_ALPHA[1] + heavy_relax["alpha_add"])
    heavy_slack = BASE_HEAVY_SLACK + heavy_relax["slack_add"]
    text = _replace_once(
        text,
        r'("heavy": \{\n\s+"node_range": \(9, 12\),\n\s+"total_comp": \(1\.9e9, 3\.3e9\),\n\s+"total_data": \(2\.8e6, 6\.4e6\),\n\s+"edge_data": \(1\.8e5, 8\.8e5\),\n\s+"deadline_alpha": )\([^)]+\)(,\n\s+"deadline_slack": )[0-9.]+(,\n\s+\},)',
        rf'\g<1>{_format_tuple(heavy_alpha)}\g<2>{heavy_slack:.2f}\g<3>',
    )
    return text


def _parse_gate1(stdout):
    out = {}
    for line in stdout.splitlines():
        if line.startswith("Local_best_ratio "):
            out["Local_best_ratio"] = float(line.split()[1])
        elif line.startswith("RSU_best_ratio "):
            out["RSU_best_ratio"] = float(line.split()[1])
        elif line.startswith("V2V_best_ratio "):
            out["V2V_best_ratio"] = float(line.split()[1])
        elif line.startswith("best_mode_counts "):
            out["best_mode_counts"] = line.split(" ", 1)[1]
        elif line.startswith("dominant_gap_component "):
            out["dominant_gap_component"] = line.split(" ", 1)[1]
        elif line.startswith("gate_target_pass "):
            out["gate1_pass"] = int(line.split()[1])
    return out


def _summarize_gate2(csv_path):
    df = pd.read_csv(csv_path)
    overall = {
        "task_success_rate": float(df["task_success_rate"].mean()),
        "deadline_miss_rate": float(df["deadline_miss_rate"].mean()),
        "time_limit_rate": float(df["time_limit_rate"].mean()),
    }
    workload = (
        df.groupby("workload_level")[["task_success_rate", "deadline_miss_rate", "time_limit_rate"]]
        .mean()
        .to_dict(orient="index")
    )
    return overall, workload


def _parse_gate2(stdout, csv_path):
    gate2_pass = 0
    for line in stdout.splitlines():
        if line.startswith("gate2_pass "):
            gate2_pass = int(line.split()[1])
    overall, workload = _summarize_gate2(csv_path)
    heavy_sr = float(workload.get("heavy", {}).get("task_success_rate", 0.0))
    return {
        "overall": overall,
        "workload_level": workload,
        "gate2_pass": gate2_pass,
        "heavy_task_success_rate": heavy_sr,
    }


def _gate1_penalty(g1):
    return (
        3 * max(g1["V2V_best_ratio"] - 0.40, 0.0)
        + 3 * max(0.30 - g1["RSU_best_ratio"], 0.0)
        + 2 * max(0.10 - g1["Local_best_ratio"], 0.0)
    )


def _score(result):
    g1 = result["gate1"]
    g2 = result["gate2"]
    if not g2:
        return _gate1_penalty(g1)
    return (
        _gate1_penalty(g1)
        + 2 * max(0.55 - g2["heavy_task_success_rate"], 0.0)
        + 2 * max(g2["overall"]["deadline_miss_rate"] - 0.35, 0.0)
        + 2 * max(g2["overall"]["time_limit_rate"] - 0.05, 0.0)
    )


def _run_gate1(combo_id):
    gate1_csv = RUN_DIR / f"{combo_id}_gate1.csv"
    gate1_stdout = _run(
        [
            sys.executable,
            "scripts/diagnose_scene_separability.py",
            "--snapshots",
            str(GATE1_SNAPSHOTS),
            "--episodes",
            str(GATE1_EPISODES),
            "--max-steps",
            str(GATE1_MAX_STEPS),
            "--min-step",
            str(GATE1_MIN_STEP),
            "--seed",
            "42",
            "--out",
            str(gate1_csv),
        ]
    )
    return _parse_gate1(gate1_stdout)


def _run_gate2(combo_id):
    gate2_csv = RUN_DIR / f"{combo_id}_gate2.csv"
    gate2_stdout = _run(
        [
            sys.executable,
            "scripts/gate_episode_feasibility.py",
            "--episodes",
            str(GATE2_EPISODES),
            "--policy",
            "CP-EFT",
            "--seed",
            "42",
            "--out",
            str(gate2_csv),
        ]
    )
    return _parse_gate2(gate2_stdout, gate2_csv)


def _evaluate_combo(base_text, prob_key, cpu_key, stability_key, heavy_key="H0", run_gate2=False):
    combo_id = f"{prob_key}_{cpu_key}_{stability_key}_{heavy_key}"
    CONFIG_PATH.write_text(_apply_combo_to_text(base_text, prob_key, cpu_key, stability_key, heavy_key), encoding="utf-8")

    result = {
        "combo_id": combo_id,
        "prob_level": prob_key,
        "cpu_level": cpu_key,
        "stability_level": stability_key,
        "heavy_level": heavy_key,
        "gate1": _run_gate1(combo_id),
        "gate2": _run_gate2(combo_id) if run_gate2 else None,
    }
    result["score"] = _score(result)
    result["pass_both"] = bool(result["gate1"]["gate1_pass"] and result["gate2"] and result["gate2"]["gate2_pass"])
    return result


def _apply_best_config(base_text, result):
    CONFIG_PATH.write_text(
        _apply_combo_to_text(
            base_text,
            result["prob_level"],
            result["cpu_level"],
            result["stability_level"],
            result["heavy_level"],
        ),
        encoding="utf-8",
    )


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    results = []
    try:
        for prob_key, cpu_key, stability_key in product(PROB_LEVELS, CPU_LEVELS, STABILITY_LEVELS):
            results.append(_evaluate_combo(original_text, prob_key, cpu_key, stability_key, "H0", run_gate2=False))

        shortlist = sorted(results, key=lambda r: (r["score"], r["combo_id"]))[:4]
        shortlist_ids = {item["combo_id"] for item in shortlist}
        for item in results:
            if item["combo_id"] in shortlist_ids:
                refreshed = _evaluate_combo(
                    original_text,
                    item["prob_level"],
                    item["cpu_level"],
                    item["stability_level"],
                    item["heavy_level"],
                    run_gate2=True,
                )
                item.update(refreshed)

        passing = [r for r in results if r["pass_both"]]
        if not passing:
            top_candidates = sorted(
                [r for r in results if r["gate2"] is not None],
                key=lambda r: (r["score"], r["combo_id"]),
            )[:2]
            for candidate in top_candidates:
                for heavy_key in ("H1", "H2"):
                    results.append(
                        _evaluate_combo(
                            original_text,
                            candidate["prob_level"],
                            candidate["cpu_level"],
                            candidate["stability_level"],
                            heavy_key,
                            run_gate2=True,
                        )
                    )

        best = min(results, key=lambda r: (r["score"], not r["pass_both"], r["combo_id"]))
        _apply_best_config(original_text, best)

        summary = {
            "total_runs": len(results),
            "best_combo": best["combo_id"],
            "results": results,
        }
        out_path = RUN_DIR / "sweep_summary.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary={out_path}")
        print(f"total_runs={len(results)}")
        print(f"best_combo={best['combo_id']}")
        print(f"best_score={best['score']:.6f}")
    except Exception:
        CONFIG_PATH.write_text(original_text, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
