"""
[RC1论文表格] export_rc1_metric_mapping_table.py

导出“论文指标 <-> 日志字段”的映射表（LaTeX tabular）。
该表用于第3.5节说明：核心指标/辅助指标来自哪些CSV字段以及统计口径。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tex", type=str, required=True)
    args = ap.parse_args()

    out_tex = Path(args.out_tex).resolve()
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # (metric_name, paper_notation, csv_field, scope_note)
    rows: List[Tuple[str, str, str, str]] = [
        ("按时完成率/成功率", r"$\\mathbb{I}[T_u^{\\mathrm{finish}}\\le T_u^{\\mathrm{ddl}}]$", "metrics.csv:task_success_rate", "episode级（按车辆聚合）"),
        ("超时率", "-", "metrics.csv:deadline_miss_rate", "episode级"),
        ("平均完工时间（估计）", r"$T_u^{\\mathrm{finish}}$", "metrics.csv:mean_cft_est", "episode级（按车辆聚合）"),
        ("已完成任务完工时间", "-", "metrics.csv:mean_cft_completed", "episode级（仅统计已完成车辆）"),
        ("风险代价（回合均值）", r"$\\sum_k \\mathrm{risk}_{x_{u,k}}$", "metrics.csv:risk_penalty_mean", "episode级（与reward风险项一致）"),
        ("可信估计（选择目标）", r"$\\hat\\rho_j$", "metrics.csv:rho_selected_mean / p10 / p95", "episode级（对被选远端节点统计）"),
        ("不确定性（选择目标）", "-", "metrics.csv:uncertainty_selected_mean / p90", "episode级"),
        ("远端卸载比例", "-", "metrics.csv:decision_frac_local/rsu/v2v", "episode级（按决策次数归一化）"),
        ("远端功率比例", r"$p_u(t)$", "metrics.csv:power_ratio_mean / power_ratio_p95", "episode级（仅统计远端决策）"),
        ("信誉失败率", "-", "metrics.csv:trust_failure_rate", "episode级（按尝试次数归一化）"),
        ("链上风险代理（可选）", "-", "metrics.csv:chain_risk_cost_total / chain_pfail_mean", "episode级（若启用链网代理）"),
        ("能耗（评估记录）", "-", "metrics.csv:energy_norm_mean", "episode级（不作为显式优化项时仅记录）"),
        ("干扰（评估记录）", "-", "metrics.csv:I_caused_mean / I_total_mean", "episode级（不作为显式优化项时仅记录）"),
    ]

    lines = []
    lines.append("% Auto-generated. Do not edit by hand.\n")
    lines.append("\\begin{tabular}{llll}\n")
    lines.append("\\hline\n")
    lines.append("指标 & 记号/含义 & 日志字段 & 统计口径 \\\\\n")
    lines.append("\\hline\n")
    for m, sym, field, note in rows:
        lines.append(f"{m} & {sym} & {field} & {note} \\\\\n")
    lines.append("\\hline\n")
    lines.append("\\end{tabular}\n")

    out_tex.write_text("".join(lines), encoding="utf-8")
    print(f"✓ Wrote: {out_tex}")


if __name__ == "__main__":
    main()

