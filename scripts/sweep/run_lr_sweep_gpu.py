#!/usr/bin/env python3
"""
Run a multi-learning-rate sweep on a single GPU, then optionally git-push and
schedule a shutdown.

Typical use:
  python scripts/sweep/run_lr_sweep_gpu.py \
    --config-snapshot runs/run_1000ep_A_20260320/logs/config_snapshot.json \
    --critic-lrs 2e-4,3e-4,5e-4,8e-4 \
    --concurrency 2 \
    --out-root runs/lr_sweep_20260320 \
    --git-sync \
    --shutdown-after-push
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / "train.py"
POSTPROCESS_PY = ROOT / "scripts" / "postprocess_run.py"


@dataclass
class JobSpec:
    actor_lr: float
    critic_lr: float
    run_name: str
    run_dir: Path
    log_path: Path


@dataclass
class RunningJob:
    spec: JobSpec
    proc: subprocess.Popen
    log_fp: object
    started_at: float


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a learning-rate sweep on one GPU.")
    ap.add_argument("--config-snapshot", type=str, required=True, help="Base logs/config_snapshot.json.")
    ap.add_argument("--actor-lrs", type=str, default=None, help="Comma-separated actor LRs.")
    ap.add_argument("--critic-lrs", type=str, default=None, help="Comma-separated critic LRs.")
    ap.add_argument(
        "--lr-pairs",
        type=str,
        default=None,
        help="Comma-separated actor:critic LR pairs, e.g. 2e-4:2e-4,2e-4:3e-4",
    )
    ap.add_argument(
        "--grid",
        action="store_true",
        default=False,
        help="If both --actor-lrs and --critic-lrs are given, use Cartesian product.",
    )
    ap.add_argument("--concurrency", type=int, default=2, help="Concurrent training processes on the same GPU.")
    ap.add_argument("--out-root", type=str, default=None, help="Sweep output root under runs/.")
    ap.add_argument("--name-prefix", type=str, default="lr_sweep", help="Run name prefix.")
    ap.add_argument("--device", type=str, default="cuda", help="Device passed to train.py.")
    ap.add_argument("--cuda-visible-devices", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed override. Default: preserve snapshot.")
    ap.add_argument("--max-episodes", type=int, default=None, help="Optional episode override. Default: preserve snapshot.")
    ap.add_argument("--max-steps", type=int, default=None, help="Optional max-steps override. Default: preserve snapshot.")
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--save-interval", type=int, default=None)
    ap.add_argument("--stagger-seconds", type=float, default=10.0, help="Delay between launches when concurrency > 1.")
    ap.add_argument("--poll-seconds", type=float, default=15.0, help="Polling interval for launcher.")
    ap.add_argument("--postprocess", action="store_true", default=False, help="Run postprocess_run.py after each successful run.")
    ap.add_argument("--disable-baseline-eval", action="store_true", default=True, help="Forward --disable-baseline-eval to train.py.")
    ap.add_argument("--enable-baseline-eval", action="store_true", default=False, help="Override and keep baseline eval enabled.")
    ap.add_argument("--disable-auto-plot", action="store_true", default=False, help="Set DISABLE_AUTO_PLOT=1 for launched runs.")
    ap.add_argument("--git-sync", action="store_true", default=False, help="Commit and push generated sweep outputs.")
    ap.add_argument("--git-remote", type=str, default="origin")
    ap.add_argument("--git-branch", type=str, default=None, help="Explicit branch to push. Default: current branch.")
    ap.add_argument("--git-commit-message", type=str, default=None, help="Commit message. Default: auto-generated.")
    ap.add_argument(
        "--git-path",
        action="append",
        default=[],
        help="Extra path(s) to include in git add, relative to repo root.",
    )
    ap.add_argument("--shutdown-after-push", action="store_true", default=False, help="Schedule shutdown after a successful git push.")
    ap.add_argument(
        "--shutdown-cmd",
        type=str,
        default="shutdown -h +2",
        help="Shutdown command. Default is Linux-style delayed shutdown after 2 minutes.",
    )
    ap.add_argument("--dry-run", action="store_true", default=False, help="Print plan only; do not launch.")
    args = ap.parse_args()

    if args.enable_baseline_eval:
        args.disable_baseline_eval = False
    if args.concurrency < 1 or args.concurrency > 4:
        ap.error("--concurrency must be within [1, 4].")
    return args


def _parse_float_list(text: Optional[str]) -> List[float]:
    if not text:
        return []
    vals = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals


def _parse_lr_pairs(text: Optional[str]) -> List[Tuple[float, float]]:
    if not text:
        return []
    pairs: List[Tuple[float, float]] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid lr pair '{item}'. Expected actor:critic.")
        actor_raw, critic_raw = item.split(":", 1)
        pairs.append((float(actor_raw.strip()), float(critic_raw.strip())))
    return pairs


def _fmt_lr(value: float) -> str:
    text = f"{value:.0e}"
    text = text.replace("e-0", "e-").replace("e+0", "e+").replace("+", "")
    return text


def _safe_name(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace(" ", "_")


def _load_snapshot(snapshot_path: Path) -> Dict[str, object]:
    with snapshot_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_branch(explicit_branch: Optional[str]) -> str:
    if explicit_branch:
        return explicit_branch
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    branch = proc.stdout.strip()
    if branch == "HEAD":
        raise RuntimeError("Detached HEAD detected. Pass --git-branch explicitly.")
    return branch


def _build_jobs(args: argparse.Namespace, snapshot: Dict[str, object], out_root: Path) -> List[JobSpec]:
    actor_base = float(snapshot.get("train_config", {}).get("LR_ACTOR", snapshot.get("LR_ACTOR", 0.0)))
    critic_base = float(snapshot.get("train_config", {}).get("LR_CRITIC", snapshot.get("LR_CRITIC", 0.0)))
    actor_lrs = _parse_float_list(args.actor_lrs)
    critic_lrs = _parse_float_list(args.critic_lrs)
    lr_pairs = _parse_lr_pairs(args.lr_pairs)

    combos: List[Tuple[float, float]] = []
    if lr_pairs:
        combos = lr_pairs
    elif actor_lrs and critic_lrs:
        if args.grid:
            combos = [(a, c) for a in actor_lrs for c in critic_lrs]
        else:
            raise ValueError("Provide either one LR list, or use --grid, or pass explicit --lr-pairs.")
    elif actor_lrs:
        combos = [(a, critic_base) for a in actor_lrs]
    elif critic_lrs:
        combos = [(actor_base, c) for c in critic_lrs]
    else:
        raise ValueError("No sweep values provided. Use --actor-lrs, --critic-lrs, or --lr-pairs.")

    if not (2 <= len(combos) <= 4):
        raise ValueError(f"Expected 2-4 configs for this sweep, got {len(combos)}.")

    jobs: List[JobSpec] = []
    log_dir = out_root / "_launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for actor_lr, critic_lr in combos:
        run_name = _safe_name(
            f"{args.name_prefix}_actor_{_fmt_lr(actor_lr)}_critic_{_fmt_lr(critic_lr)}"
        )
        run_dir = out_root / run_name
        log_path = log_dir / f"{run_name}.log"
        jobs.append(
            JobSpec(
                actor_lr=float(actor_lr),
                critic_lr=float(critic_lr),
                run_name=run_name,
                run_dir=run_dir,
                log_path=log_path,
            )
        )
    return jobs


def _build_train_cmd(args: argparse.Namespace, spec: JobSpec, snapshot_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_PY),
        "--config-snapshot",
        str(snapshot_path),
        "--run-dir",
        str(spec.run_dir),
        "--exact-run-dir",
        "--device",
        str(args.device),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.max_episodes is not None:
        cmd.extend(["--max-episodes", str(args.max_episodes)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.log_interval is not None:
        cmd.extend(["--log-interval", str(args.log_interval)])
    if args.eval_interval is not None:
        cmd.extend(["--eval-interval", str(args.eval_interval)])
    if args.save_interval is not None:
        cmd.extend(["--save-interval", str(args.save_interval)])
    if args.disable_baseline_eval:
        cmd.append("--disable-baseline-eval")
    return cmd


def _build_env(args: argparse.Namespace, spec: JobSpec) -> Dict[str, str]:
    env = os.environ.copy()
    env["LR_ACTOR"] = str(spec.actor_lr)
    env["LR_CRITIC"] = str(spec.critic_lr)
    env["DEVICE_NAME"] = str(args.device)
    env.setdefault("MPLBACKEND", "Agg")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.disable_auto_plot:
        env["DISABLE_AUTO_PLOT"] = "1"
    return env


def _launch_job(args: argparse.Namespace, spec: JobSpec, snapshot_path: Path) -> RunningJob:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_train_cmd(args, spec, snapshot_path)
    env = _build_env(args, spec)
    log_fp = spec.log_path.open("w", encoding="utf-8")
    log_fp.write(f"# cwd: {ROOT}\n")
    log_fp.write(f"# actor_lr={spec.actor_lr} critic_lr={spec.critic_lr}\n")
    log_fp.write("# cmd: " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
    log_fp.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(spec=spec, proc=proc, log_fp=log_fp, started_at=time.time())


def _postprocess_run(run_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(POSTPROCESS_PY), "--run-dir", str(run_dir), "--overwrite"],
        cwd=str(ROOT),
        check=True,
    )


def _write_manifest(out_root: Path, snapshot_path: Path, args: argparse.Namespace, jobs: Sequence[JobSpec]) -> None:
    payload = {
        "created_at_unix": time.time(),
        "config_snapshot": str(snapshot_path),
        "concurrency": int(args.concurrency),
        "device": str(args.device),
        "cuda_visible_devices": args.cuda_visible_devices,
        "max_episodes": args.max_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "disable_baseline_eval": bool(args.disable_baseline_eval),
        "disable_auto_plot": bool(args.disable_auto_plot),
        "jobs": [
            {
                "run_name": spec.run_name,
                "run_dir": str(spec.run_dir),
                "actor_lr": spec.actor_lr,
                "critic_lr": spec.critic_lr,
                "launcher_log": str(spec.log_path),
            }
            for spec in jobs
        ],
    }
    with (out_root / "lr_sweep_plan.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _run_git(args: argparse.Namespace, out_root: Path, jobs: Sequence[JobSpec]) -> None:
    branch = _resolve_branch(args.git_branch)
    commit_message = args.git_commit_message
    if not commit_message:
        lr_part = ",".join(
            f"a{_fmt_lr(spec.actor_lr)}_c{_fmt_lr(spec.critic_lr)}" for spec in jobs
        )
        commit_message = f"Add LR sweep runs: {lr_part}"

    add_paths = [str(out_root.relative_to(ROOT))]
    script_rel = Path(__file__).resolve().relative_to(ROOT)
    add_paths.append(str(script_rel))
    for extra in args.git_path:
        add_paths.append(str(Path(extra)))

    subprocess.run(["git", "add", "-A", "--", *add_paths], cwd=str(ROOT), check=True)
    diff_proc = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *add_paths],
        cwd=str(ROOT),
        check=False,
    )
    if diff_proc.returncode == 0:
        print("[Git] No staged changes in sweep scope; skipping commit/push.", flush=True)
        return
    subprocess.run(["git", "commit", "-m", commit_message], cwd=str(ROOT), check=True)
    subprocess.run(["git", "push", args.git_remote, branch], cwd=str(ROOT), check=True)
    print(f"[Git] Pushed branch '{branch}' to remote '{args.git_remote}'.", flush=True)


def _schedule_shutdown(shutdown_cmd: str) -> None:
    subprocess.run(shutdown_cmd, cwd=str(ROOT), shell=True, check=True)
    print(f"[Shutdown] Scheduled via: {shutdown_cmd}", flush=True)


def _print_plan(snapshot_path: Path, out_root: Path, jobs: Sequence[JobSpec], args: argparse.Namespace) -> None:
    print(f"[Sweep] base snapshot: {snapshot_path}", flush=True)
    print(f"[Sweep] output root: {out_root}", flush=True)
    print(f"[Sweep] concurrency: {args.concurrency}", flush=True)
    for spec in jobs:
        print(
            f"[Sweep] {spec.run_name}: actor_lr={spec.actor_lr} critic_lr={spec.critic_lr} "
            f"run_dir={spec.run_dir}",
            flush=True,
        )
    if args.git_sync:
        print("[Sweep] git sync enabled", flush=True)
    if args.shutdown_after_push:
        print(f"[Sweep] shutdown command: {args.shutdown_cmd}", flush=True)


def main() -> int:
    args = _parse_args()
    snapshot_path = Path(args.config_snapshot).resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {snapshot_path}")
    snapshot = _load_snapshot(snapshot_path)

    out_root = Path(args.out_root).resolve() if args.out_root else (ROOT / "runs" / f"{args.name_prefix}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, snapshot, out_root)
    _write_manifest(out_root, snapshot_path, args, jobs)
    _print_plan(snapshot_path, out_root, jobs, args)

    if args.dry_run:
        return 0

    pending = list(jobs)
    running: List[RunningJob] = []
    completed: List[JobSpec] = []
    failed: List[Tuple[JobSpec, int]] = []

    while pending or running:
        while pending and len(running) < args.concurrency:
            spec = pending.pop(0)
            job = _launch_job(args, spec, snapshot_path)
            running.append(job)
            print(
                f"[Launch] {spec.run_name} pid={job.proc.pid} actor_lr={spec.actor_lr} critic_lr={spec.critic_lr}",
                flush=True,
            )
            if pending and args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)

        time.sleep(max(args.poll_seconds, 1.0))
        next_running: List[RunningJob] = []
        for job in running:
            code = job.proc.poll()
            if code is None:
                next_running.append(job)
                continue
            elapsed = time.time() - job.started_at
            job.log_fp.flush()
            job.log_fp.close()
            if code == 0:
                print(f"[Done] {job.spec.run_name} elapsed={elapsed:.1f}s", flush=True)
                if args.postprocess:
                    try:
                        _postprocess_run(job.spec.run_dir)
                        print(f"[Postprocess] {job.spec.run_name} done", flush=True)
                    except Exception as exc:
                        print(f"[Postprocess][WARN] {job.spec.run_name}: {exc}", flush=True)
                completed.append(job.spec)
            else:
                print(f"[Fail] {job.spec.run_name} exit={code} log={job.spec.log_path}", flush=True)
                failed.append((job.spec, int(code)))
        running = next_running
        if failed:
            break

    if running:
        for job in running:
            if job.proc.poll() is None:
                job.proc.terminate()
            job.log_fp.close()

    status = {
        "completed": [spec.run_name for spec in completed],
        "failed": [{"run_name": spec.run_name, "exit_code": code} for spec, code in failed],
    }
    with (out_root / "lr_sweep_status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=True, indent=2)

    if failed:
        print("[Sweep] Aborting git sync/shutdown because at least one run failed.", flush=True)
        return 1

    if args.git_sync:
        _run_git(args, out_root, jobs)
    if args.shutdown_after_push:
        if not args.git_sync:
            print("[Shutdown][WARN] shutdown requested without --git-sync; proceeding anyway.", flush=True)
        _schedule_shutdown(args.shutdown_cmd)
    print("[Sweep] All runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
