"""
[RC1论文表格] export_rc1_eval_tables.py

将 sweep_eval_scale_malicious.py 的 *_summary.csv 导出为论文可直接 \\input 的 LaTeX 表格。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _fmt_pm(x):
    try:
        return f"{float(x):.3g}"
    except Exception:
        return str(x)


def _fmt_mean_std(mean, std, digits=3):
    if mean is None or pd.isna(mean):
        return "-"
    try:
        m = float(mean)
    except Exception:
        return str(mean)
    if std is None or pd.isna(std):
        return f"{m:.{digits}f}"
    try:
        s = float(std)
    except Exception:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}$\\pm${s:.{digits}f}"


def _write_table(lines: List[str], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("".join(lines), encoding="utf-8")


def export_main(summary_csv: Path, out_tex: Path, policy_order: List[str]) -> None:
    df = pd.read_csv(summary_csv)
    if "sweep" in df.columns:
        df = df[df["sweep"] == "main"].copy()
    if df.empty:
        raise ValueError("Empty main summary.")
    df = df.set_index("policy")

    lines = []
    lines.append("% Auto-generated. Do not edit by hand.\n")
    lines.append("\\begin{tabular}{lccc}\n")
    lines.append("\\hline\n")
    lines.append("方法 & 成功率 & 平均完工时间（估计）/s & 平均风险代价 \\\\\n")
    lines.append("\\hline\n")
    for p in policy_order:
        if p not in df.index:
            continue
        r = df.loc[p]
        lines.append(
            f"{p} & "
            f"{_fmt_mean_std(r.get('task_success_rate_mean'), r.get('task_success_rate_std'), digits=3)} & "
            f"{_fmt_mean_std(r.get('mean_cft_est_mean'), r.get('mean_cft_est_std'), digits=3)} & "
            f"{_fmt_mean_std(r.get('risk_penalty_mean_mean'), r.get('risk_penalty_mean_std'), digits=4)} \\\\\n"
        )
    lines.append("\\hline\n")
    lines.append("\\end{tabular}\n")
    _write_table(lines, out_tex)


def export_sweep(summary_csv: Path, sweep: str, out_tex: Path, x_label: str, policy_order: List[str]) -> None:
    df = pd.read_csv(summary_csv)
    df = df[df["sweep"] == sweep].copy()
    if df.empty:
        raise ValueError(f"Empty sweep={sweep} summary.")
    # stable ordering
    df["policy_order"] = df["policy"].map({p: i for i, p in enumerate(policy_order)}).fillna(9999).astype(int)
    df = df.sort_values(["value", "policy_order", "policy"])

    lines = []
    lines.append("% Auto-generated. Do not edit by hand.\n")
    lines.append("\\begin{tabular}{l l c c c}\n")
    lines.append("\\hline\n")
    lines.append(f"{x_label} & 方法 & 成功率 & 平均完工时间（估计）/s & 平均风险代价 \\\\\n")
    lines.append("\\hline\n")
    for _, r in df.iterrows():
        x = r["value"]
        lines.append(
            f"{_fmt_pm(x)} & {r['policy']} & "
            f"{_fmt_mean_std(r.get('task_success_rate_mean'), r.get('task_success_rate_std'), digits=3)} & "
            f"{_fmt_mean_std(r.get('mean_cft_est_mean'), r.get('mean_cft_est_std'), digits=3)} & "
            f"{_fmt_mean_std(r.get('risk_penalty_mean_mean'), r.get('risk_penalty_mean_std'), digits=4)} \\\\\n"
        )
    lines.append("\\hline\n")
    lines.append("\\end{tabular}\n")
    _write_table(lines, out_tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="Run dir that contains paper_exports/*.csv")
    ap.add_argument("--main-summary", type=str, default="paper_exports/main_eval_episode_summary.csv")
    ap.add_argument("--scale-summary", type=str, default="paper_exports/scale_eval_episode_summary.csv")
    ap.add_argument("--malicious-summary", type=str, default="paper_exports/malicious_eval_episode_summary.csv")
    ap.add_argument("--out-dir", type=str, default="paper_exports")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = (run_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_order_main = ["mappo", "Greedy", "EFT", "LB-Greedy", "Local-Only"]
    policy_order_small = ["mappo", "Greedy", "Local-Only"]

    export_main(run_dir / args.main_summary, out_dir / "tab_rc1_main_compare.tex", policy_order_main)
    export_sweep(run_dir / args.scale_summary, "scale", out_dir / "tab_rc1_scale_compare.tex", "$U$", policy_order_small)
    export_sweep(run_dir / args.malicious_summary, "malicious", out_dir / "tab_rc1_malicious_compare.tex", "$p_m$", policy_order_small)
    print(f"✓ Wrote tables under: {out_dir}")


if __name__ == "__main__":
    main()

