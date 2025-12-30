import argparse
import glob
import json
import os
import re
from statistics import mean, stdev


METRICS = [
    "success_rate",
    "vehicle_success_rate",
    "deadline_miss_rate",
    "task_success_rate",
    "mean_cft",
    "delay_norm.mean",
    "delay_norm.p95",
    "energy_norm.mean",
    "energy_norm.p95",
    "clip_hit_ratio",
    "illegal_action_rate",
    "hard_trigger_rate",
    "bonus_ratio.mean",
    "bonus_ratio.p95",
    "power_ratio.mean",
    "power_ratio.p95",
    "tx_power_dbm.mean",
    "tx_power_dbm.p95",
]

REQUIRED_FIELDS = ("reward_mode", "bonus_mode", "episode_id")
SUCCESS_FIELDS = ("vehicle_success_rate", "success_rate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/ablation")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args()


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def extract_metric(ep, key):
    if key in ep:
        return ep.get(key, 0.0)
    if key == "vehicle_success_rate":
        return ep.get("vehicle_success_rate", ep.get("success_rate", 0.0))
    if key == "task_success_rate":
        return ep.get("task_success_rate", ep.get("vehicle_success_rate", ep.get("success_rate", 0.0)))
    if "." in key:
        base, stat = key.split(".", 1)
        return ep.get("metrics", {}).get(base, {}).get(stat, 0.0)
    return ep.get("metrics", {}).get(key, {}).get("mean", 0.0)

def _parse_seed(path, first_line):
    seed = first_line.get("seed")
    if seed is not None:
        return int(seed)
    match = re.search(r"seed(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def _has_required_fields(line):
    if not all(field in line for field in REQUIRED_FIELDS):
        return False
    if not any(field in line for field in SUCCESS_FIELDS):
        return False
    return True


def _validate_lines(lines):
    target_eps = lines[0].get("target_episodes")
    if target_eps is None:
        return False, "missing_target_episodes"
    try:
        target_eps = int(target_eps)
    except (TypeError, ValueError):
        return False, "invalid_target_episodes"
    if target_eps > 0 and len(lines) < int(0.8 * target_eps):
        return False, "too_short"
    if not _has_required_fields(lines[-1]):
        return False, "missing_fields"
    return True, None


def load_runs(log_dir, run_id=None):
    runs = []
    excluded = []
    for path in glob.glob(os.path.join(log_dir, "*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
        except Exception as exc:
            excluded.append({"path": path, "reason": f"read_error:{exc}"})
            continue
        if not lines:
            excluded.append({"path": path, "reason": "empty"})
            continue
        if run_id is not None and lines[0].get("run_id") != run_id:
            excluded.append({"path": path, "reason": "run_id_mismatch"})
            continue
        valid, reason = _validate_lines(lines)
        if not valid:
            excluded.append({"path": path, "reason": reason})
            continue
        reward_mode = lines[0].get("reward_mode", "unknown")
        bonus_mode = lines[0].get("bonus_mode", "unknown")
        runs.append({
            "path": path,
            "reward_mode": reward_mode,
            "bonus_mode": bonus_mode,
            "seed": _parse_seed(path, lines[0]),
            "run_id": lines[0].get("run_id"),
            "mtime": os.path.getmtime(path),
            "lines": lines,
        })
    return runs, excluded


def aggregate_run(lines):
    agg = {}
    for key in METRICS:
        vals = [extract_metric(ep, key) for ep in lines]
        agg[key] = safe_mean(vals)
    return agg


def _dedup_latest(runs):
    latest = {}
    for run in runs:
        key = (run["reward_mode"], run["bonus_mode"], run["seed"])
        prev = latest.get(key)
        if prev is None or run["mtime"] > prev["mtime"]:
            latest[key] = run
    return list(latest.values())


def summarize_groups(runs):
    grouped = {}
    for run in runs:
        key = (run["reward_mode"], run["bonus_mode"])
        grouped.setdefault(key, []).append(aggregate_run(run["lines"]))

    summary = {}
    for key, run_metrics in grouped.items():
        summary[key] = {}
        for metric in METRICS:
            vals = [m.get(metric, 0.0) for m in run_metrics]
            summary[key][f"{metric}_mean"] = safe_mean(vals)
            summary[key][f"{metric}_std"] = safe_std(vals)
    return summary


def write_csv(summary, out_path):
    headers = ["reward_mode", "bonus_mode"]
    for metric in METRICS:
        headers.append(f"{metric}_mean")
        headers.append(f"{metric}_std")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for (reward_mode, bonus_mode), metrics in sorted(summary.items()):
            row = [reward_mode, bonus_mode]
            for metric in METRICS:
                row.append(f"{metrics.get(f'{metric}_mean', 0.0):.6f}")
                row.append(f"{metrics.get(f'{metric}_std', 0.0):.6f}")
            f.write(",".join(row) + "\n")


def write_md(summary, out_path, excluded):
    cols = [
        "reward_mode",
        "bonus_mode",
        "success_rate",
        "vehicle_success_rate",
        "deadline_miss_rate",
        "task_success_rate",
        "mean_cft",
        "delay_norm.mean",
        "delay_norm.p95",
        "energy_norm.mean",
        "energy_norm.p95",
        "clip_hit_ratio",
        "illegal_action_rate",
        "hard_trigger_rate",
        "bonus_ratio.mean",
        "bonus_ratio.p95",
        "power_ratio.mean",
        "power_ratio.p95",
        "tx_power_dbm.mean",
        "tx_power_dbm.p95",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for (reward_mode, bonus_mode), metrics in sorted(summary.items()):
            row = [reward_mode, bonus_mode]
            for metric in cols[2:]:
                mean_val = metrics.get(f"{metric}_mean", 0.0)
                std_val = metrics.get(f"{metric}_std", 0.0)
                row.append(f"{mean_val:.4f}±{std_val:.4f}")
            f.write("| " + " | ".join(row) + " |\n")

    # Sanity checks
    delta_groups = {k: v for k, v in summary.items() if k[0] == "delta_cft"}
    if delta_groups:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n**Sanity Checks**\n\n")
            for metric in ["mean_cft", "success_rate", "delay_norm.mean", "energy_norm.mean"]:
                vals = [v.get(f"{metric}_mean", 0.0) for v in delta_groups.values()]
                spread = max(vals) - min(vals) if vals else 0.0
                flag = "WARN" if spread > 1e-3 else "OK"
                f.write(f"- delta_cft {metric} spread: {spread:.6f} ({flag})\n")

            for (reward_mode, bonus_mode), metrics in sorted(summary.items()):
                clip_ratio = metrics.get("clip_hit_ratio_mean", 0.0)
                illegal_rate = metrics.get("illegal_action_rate_mean", 0.0)
                hard_rate = metrics.get("hard_trigger_rate_mean", 0.0)
                warn = []
                if clip_ratio > 0.10:
                    warn.append("clip_hit_ratio>10%")
                if illegal_rate > 0.001:
                    warn.append("illegal_action_rate>0.1%")
                if hard_rate > 0.001:
                    warn.append("hard_trigger_rate>0.1%")
                if warn:
                    f.write(f"- {reward_mode}/{bonus_mode}: WARN " + ", ".join(warn) + "\n")

    if excluded:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n**Excluded Files**\n\n")
            for item in excluded:
                f.write(f"- {item['path']}: {item['reason']}\n")


def plot_summary(summary, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Warn] matplotlib not available; skipping plots.")
        return

    labels = []
    mean_cft = []
    success = []
    clip_ratio = []
    bonus_ratio = []

    for (reward_mode, bonus_mode), metrics in sorted(summary.items()):
        labels.append(f"{reward_mode}\n{bonus_mode}")
        mean_cft.append(metrics.get("mean_cft_mean", 0.0))
        success.append(metrics.get("success_rate_mean", 0.0))
        clip_ratio.append(metrics.get("clip_hit_ratio_mean", 0.0))
        bonus_ratio.append(metrics.get("bonus_ratio.mean_mean", 0.0))

    plt.figure(figsize=(12, 4))
    plt.bar(labels, mean_cft)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("mean_cft")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_cft_bar.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(labels, success)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("success_rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "success_rate_bar.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    width = 0.4
    x = range(len(labels))
    plt.bar(x, clip_ratio, width=width, label="clip_hit_ratio")
    plt.bar([i + width for i in x], bonus_ratio, width=width, label="bonus_ratio")
    plt.xticks([i + width / 2 for i in x], labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clip_bonus_bar.png"))
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs, excluded = load_runs(args.log_dir, run_id=args.run_id)
    if not runs:
        print("[Error] No jsonl logs found.")
        return
    runs = _dedup_latest(runs)
    summary = summarize_groups(runs)
    csv_path = os.path.join(args.out_dir, "ablation_summary.csv")
    md_path = os.path.join(args.out_dir, "ablation_summary.md")
    write_csv(summary, csv_path)
    write_md(summary, md_path, excluded)
    if args.plot:
        plot_summary(summary, args.out_dir)
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {md_path}")
    if excluded:
        print("[Warn] Excluded files:")
        for item in excluded:
            print(f"  - {item['path']}: {item['reason']}")


if __name__ == "__main__":
    main()
