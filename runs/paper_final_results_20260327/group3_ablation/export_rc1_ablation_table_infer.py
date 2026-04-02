"""
[RC1论文表格] export_rc1_ablation_table_infer.py

导出“评估期模块开关（不重训）”的消融对比表。
注意：该表反映的是推理/评估阶段的开关影响，不等价于训练期消融。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _load_one(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if len(df) != 1:
        df = df[df["policy"] == "mappo"]
    if df.empty:
        raise ValueError(f"No mappo row in {path}")
    return df.iloc[0]


def _fmt(mean, std, digits=3):
    try:
        m = float(mean)
    except Exception:
        return "-"
    if std is None or pd.isna(std):
        return f"{m:.{digits}f}"
    try:
        s = float(std)
    except Exception:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}$\\pm${s:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-summary", type=str, required=True)
    ap.add_argument("--no-pbca-summary", type=str, required=True)
    ap.add_argument("--fixed-power-summary", type=str, required=True)
    ap.add_argument("--no-trust-summary", type=str, required=True)
    ap.add_argument("--out-tex", type=str, required=True)
    args = ap.parse_args()

    items: List[Tuple[str, pd.Series]] = [
        ("完整模型", _load_one(Path(args.full_summary))),
        ("去PBCA(物理偏置)", _load_one(Path(args.no_pbca_summary))),
        ("固定功率", _load_one(Path(args.fixed_power_summary))),
        ("去信誉/风险输入", _load_one(Path(args.no_trust_summary))),
    ]

    out_tex = Path(args.out_tex).resolve()
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("% Auto-generated. Do not edit by hand.\n")
    lines.append("\\begin{tabular}{lccc}\n")
    lines.append("\\hline\n")
    lines.append("设置 & 成功率 & 平均完工时间（估计）/s & 平均风险代价 \\\\\n")
    lines.append("\\hline\n")
    for name, r in items:
        lines.append(
            f"{name} & "
            f"{_fmt(r.get('task_success_rate_mean'), r.get('task_success_rate_std'), digits=3)} & "
            f"{_fmt(r.get('mean_cft_est_mean'), r.get('mean_cft_est_std'), digits=3)} & "
            f"{_fmt(r.get('risk_penalty_mean_mean'), r.get('risk_penalty_mean_std'), digits=4)} \\\\\n"
        )
    lines.append("\\hline\n")
    lines.append("\\end{tabular}\n")
    out_tex.write_text("".join(lines), encoding="utf-8")
    print(f"✓ Wrote: {out_tex}")


if __name__ == "__main__":
    main()

