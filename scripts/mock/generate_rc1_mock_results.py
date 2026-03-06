"""
[进度检查用] generate_rc1_mock_results.py

用途（重要）：
  - 生成“合成/示意”的实验数据与对比图，用于阶段性进度检查与后续真实仿真的参考指引；
  - 不代表真实仿真或可复现实验结论，禁止直接作为论文最终结果。

输出（默认生成一个 mock run 目录，结构尽量对齐真实 run，便于替换/对照）：
  mock_runs/rc1_mock_progress_<timestamp>/
    logs/metrics.csv                          # 训练收敛曲线（示意）
    paper_exports/*.csv                       # 主对比/规模/恶意/消融（示意）
    paper_exports/*.tex                       # 可直接 \\input 的表格（示意）
    paper_figs_cn/*.png                       # 中文对比图（示意，带水印）
    MOCK_DISCLAIMER.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def _set_cn_style():
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.size"] = 11


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _watermark(ax, text: str = "示意图（合成数据）", alpha: float = 0.10):
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        color="gray",
        alpha=alpha,
        rotation=18,
        zorder=10,
    )


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _pos(x: float, eps: float = 1e-6) -> float:
    return float(max(x, eps))


def _dirichlet_mean(rng: np.random.Generator, mean: Tuple[float, float, float], strength: float = 50.0) -> Tuple[float, float, float]:
    m = np.array(mean, dtype=float)
    m = m / max(m.sum(), 1e-9)
    alpha = np.maximum(m * strength, 1e-3)
    s = rng.dirichlet(alpha).astype(float)
    return float(s[0]), float(s[1]), float(s[2])


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    # Base performance at "main" condition
    sr_mean: float
    cft_mean: float
    risk_mean: float
    # Behavior priors (Local/RSU/V2V)
    ratio_mean: Tuple[float, float, float]
    # Power (remote)
    power_mean: float


def _policy_profiles() -> List[PolicyProfile]:
    # “理想趋势”设定：MAPPO在风险代价更低的同时接近/超过启发式的完成表现。
    return [
        PolicyProfile("mappo",      sr_mean=0.92, cft_mean=2.55, risk_mean=0.0060, ratio_mean=(0.10, 0.82, 0.08), power_mean=0.55),
        PolicyProfile("Greedy",     sr_mean=0.90, cft_mean=2.35, risk_mean=0.0135, ratio_mean=(0.05, 0.92, 0.03), power_mean=1.00),
        PolicyProfile("EFT",        sr_mean=0.86, cft_mean=2.55, risk_mean=0.0105, ratio_mean=(0.10, 0.80, 0.10), power_mean=0.60),
        PolicyProfile("LB-Greedy",  sr_mean=0.84, cft_mean=2.62, risk_mean=0.0095, ratio_mean=(0.12, 0.78, 0.10), power_mean=0.60),
        PolicyProfile("Local-Only", sr_mean=0.55, cft_mean=3.45, risk_mean=0.0000, ratio_mean=(1.00, 0.00, 0.00), power_mean=0.00),
    ]


def _condition_effect_scale(U: int) -> Dict[str, float]:
    """
    理想推导：规模增大 → 竞争/排队更强 → 成功率下降、完工时间上升、风险略上升；
    且学习方法相对更鲁棒（退化幅度略小）。
    """
    # Normalize relative to U=20 as baseline.
    if U <= 10:
        return {"sr_delta": +0.03, "cft_mul": 0.92, "risk_mul": 0.95, "remote_delta": +0.02}
    if U <= 20:
        return {"sr_delta": 0.00, "cft_mul": 1.00, "risk_mul": 1.00, "remote_delta": 0.00}
    if U <= 40:
        return {"sr_delta": -0.07, "cft_mul": 1.18, "risk_mul": 1.08, "remote_delta": -0.04}
    return {"sr_delta": -0.10, "cft_mul": 1.25, "risk_mul": 1.12, "remote_delta": -0.06}


def _condition_effect_malicious(pm: float) -> Dict[str, float]:
    """
    理想推导：恶意比例上升 → V2V可靠性下降 → 风险代价上升、策略更保守（远端比例下降），
    完工时间上升、成功率下降。
    """
    pm = float(pm)
    return {
        "sr_delta": -0.10 * pm / 0.3,   # pm=0.3 -> -0.10
        "cft_mul": 1.0 + 0.12 * pm / 0.3,
        "risk_add": 0.010 * pm / 0.3,   # pm=0.3 -> +0.010
        "v2v_mul": 1.0 - 0.65 * pm / 0.3,
        "remote_delta": -0.08 * pm / 0.3,
    }


def _sample_episode_metrics(
    rng: np.random.Generator,
    prof: PolicyProfile,
    cond: Dict[str, float],
    *,
    policy_kind: str,
) -> Dict[str, float]:
    # Success rate
    sr = prof.sr_mean + cond.get("sr_delta", 0.0)
    # Policy robustness tweak
    if policy_kind == "learned":
        sr += 0.01
    else:
        sr -= 0.01
    sr = _clip01(sr + rng.normal(0.0, 0.03))

    # Deadline miss rate correlates with (1 - sr) but not identical
    dmr = _clip01((1.0 - sr) * 0.95 + rng.normal(0.0, 0.02))

    # Completion time
    cft = prof.cft_mean * cond.get("cft_mul", 1.0)
    cft = _pos(cft + rng.normal(0.0, 0.12))

    # Risk penalty
    risk = prof.risk_mean * cond.get("risk_mul", 1.0) + cond.get("risk_add", 0.0)
    risk = _pos(risk + rng.normal(0.0, 0.0012))
    if prof.name == "Local-Only":
        risk = 0.0

    # Behavior ratios (Local/RSU/V2V)
    base_L, base_R, base_V = prof.ratio_mean
    # remote adjustment (keep within [0,1])
    remote_delta = cond.get("remote_delta", 0.0)
    # For simplicity: shift from remote to local when remote_delta < 0
    base_L = _clip01(base_L - remote_delta)
    base_R = _clip01(base_R + remote_delta * 0.7)
    base_V = _clip01(base_V + remote_delta * 0.3)

    # malicious reduces V2V proportion specifically
    if "v2v_mul" in cond:
        base_V = _clip01(base_V * cond["v2v_mul"])
        # re-normalize the removed mass to RSU (safer remote)
        rem = max(0.0, 1.0 - (base_L + base_R + base_V))
        base_R = _clip01(base_R + rem)

    L, R, V = _dirichlet_mean(rng, (base_L, base_R, base_V), strength=70.0 if prof.name != "Local-Only" else 200.0)

    # Power (remote)
    power = prof.power_mean
    if prof.name == "Greedy":
        power = 1.0
    elif prof.name == "Local-Only":
        power = 0.0
    else:
        power = float(np.clip(power + rng.normal(0.0, 0.04), 0.0, 1.0))

    # Trust stats
    if prof.name == "Local-Only":
        rho = 0.0
        unc = 0.0
        trust_fail = 0.0
    else:
        rho = float(np.clip(0.93 - 0.08 * cond.get("risk_add", 0.0) / 0.010 + rng.normal(0.0, 0.01), 0.5, 0.99))
        unc = float(np.clip(0.08 + rng.normal(0.0, 0.006), 0.0, 1.0))
        # trust failure rises with pm and v2v usage
        trust_fail = float(np.clip(0.01 + 0.20 * cond.get("risk_add", 0.0) / 0.010 + 0.15 * V + rng.normal(0.0, 0.01), 0.0, 1.0))

    return {
        "task_success_rate": sr,
        "deadline_miss_rate": dmr,
        "mean_cft_est": cft,
        "risk_penalty_mean": risk,
        "decision_frac_local": L,
        "decision_frac_rsu": R,
        "decision_frac_v2v": V,
        "power_ratio_mean": power if (1.0 - L) > 1e-6 else 0.0,
        "trust_failure_rate": trust_fail,
        "rho_selected_mean": rho,
        "uncertainty_selected_mean": unc,
        "illegal_action_rate": 0.0,
        "time_limit_rate": float(np.clip(dmr * 0.25 + rng.normal(0.0, 0.02), 0.0, 1.0)),
    }


def _aggregate_summary(df: pd.DataFrame, group_keys: List[str], metric_keys: List[str]) -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(group_keys):
        if not isinstance(g, tuple):
            g = (g,)
        row = {k: v for k, v in zip(group_keys, g)}
        row["episodes"] = int(len(d))
        for mk in metric_keys:
            row[f"{mk}_mean"] = float(d[mk].mean())
            row[f"{mk}_std"] = float(d[mk].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_eval_pair(out_dir: Path, stem: str, df_ep: pd.DataFrame) -> Tuple[Path, Path]:
    ep_csv = out_dir / f"{stem}.csv"
    sum_csv = out_dir / f"{stem}_summary.csv"
    df_ep.to_csv(ep_csv, index=False)
    metric_keys = [
        "task_success_rate",
        "deadline_miss_rate",
        "mean_cft_est",
        "risk_penalty_mean",
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
        "power_ratio_mean",
        "trust_failure_rate",
        "rho_selected_mean",
        "uncertainty_selected_mean",
        "illegal_action_rate",
        "time_limit_rate",
    ]
    df_sum = _aggregate_summary(df_ep, ["sweep", "value", "policy"], metric_keys)
    df_sum.to_csv(sum_csv, index=False)
    return ep_csv, sum_csv


def _export_latex_tables(exports_dir: Path, main_sum: Path, scale_sum: Path, mal_sum: Path) -> None:
    # Minimal LaTeX exporters (avoid dependency on other scripts).
    def fmt(m, s, digits=3):
        return f"{m:.{digits}f}$\\pm${s:.{digits}f}"

    def write_main(path: Path, df: pd.DataFrame, order: List[str]):
        df = df[df["sweep"] == "main"].set_index("policy")
        lines = ["% Auto-generated (mock).\n", "\\begin{tabular}{lccc}\n", "\\hline\n",
                 "方法 & 成功率 & 平均完工时间（估计）/s & 平均风险代价 \\\\\n", "\\hline\n"]
        for p in order:
            if p not in df.index:
                continue
            r = df.loc[p]
            lines.append(f"{p} & {fmt(r['task_success_rate_mean'], r['task_success_rate_std'], 3)} & "
                         f"{fmt(r['mean_cft_est_mean'], r['mean_cft_est_std'], 3)} & "
                         f"{fmt(r['risk_penalty_mean_mean'], r['risk_penalty_mean_std'], 4)} \\\\\n")
        lines += ["\\hline\n", "\\end{tabular}\n"]
        path.write_text("".join(lines), encoding="utf-8")

    def write_sweep(path: Path, df: pd.DataFrame, sweep: str, xlab: str, order: List[str]):
        df = df[df["sweep"] == sweep].copy()
        df["policy_order"] = df["policy"].map({p: i for i, p in enumerate(order)}).fillna(999).astype(int)
        df = df.sort_values(["value", "policy_order"])
        lines = ["% Auto-generated (mock).\n", "\\begin{tabular}{l l c c c}\n", "\\hline\n",
                 f"{xlab} & 方法 & 成功率 & 平均完工时间（估计）/s & 平均风险代价 \\\\\n", "\\hline\n"]
        for _, r in df.iterrows():
            lines.append(f"{r['value']:.3g} & {r['policy']} & "
                         f"{fmt(r['task_success_rate_mean'], r['task_success_rate_std'], 3)} & "
                         f"{fmt(r['mean_cft_est_mean'], r['mean_cft_est_std'], 3)} & "
                         f"{fmt(r['risk_penalty_mean_mean'], r['risk_penalty_mean_std'], 4)} \\\\\n")
        lines += ["\\hline\n", "\\end{tabular}\n"]
        path.write_text("".join(lines), encoding="utf-8")

    main_df = pd.read_csv(main_sum)
    scale_df = pd.read_csv(scale_sum)
    mal_df = pd.read_csv(mal_sum)

    write_main(exports_dir / "tab_rc1_main_compare_mock.tex", main_df, ["mappo", "Greedy", "EFT", "LB-Greedy", "Local-Only"])
    write_sweep(exports_dir / "tab_rc1_scale_compare_mock.tex", scale_df, "scale", "$U$", ["mappo", "Greedy", "Local-Only"])
    write_sweep(exports_dir / "tab_rc1_malicious_compare_mock.tex", mal_df, "malicious", "$p_m$", ["mappo", "Greedy", "Local-Only"])


def _plot_main_convergence(figs_dir: Path, metrics_csv: Path) -> None:
    df = pd.read_csv(metrics_csv)
    ep = df["episode"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 8.4), sharex=True)
    axes[0].plot(ep, df["task_success_rate"].to_numpy(), linewidth=1.8)
    axes[0].set_ylabel("成功率")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.25)
    _watermark(axes[0])

    axes[1].plot(ep, df["mean_cft_est"].to_numpy(), linewidth=1.8, color="#d55e00")
    axes[1].set_ylabel("平均完工时间（估计）/s")
    axes[1].grid(True, alpha=0.25)
    _watermark(axes[1])

    axes[2].plot(ep, df["risk_penalty_mean"].to_numpy(), linewidth=1.8, color="#0072b2")
    axes[2].set_ylabel("平均风险代价")
    axes[2].set_xlabel("训练轮次 / Episode")
    axes[2].grid(True, alpha=0.25)
    _watermark(axes[2])

    fig.tight_layout()
    fig.savefig(figs_dir / "fig_main_convergence_mock_cn.png", dpi=220)
    plt.close(fig)


def _bar_compare(figs_dir: Path, main_sum: Path) -> None:
    df = pd.read_csv(main_sum)
    df = df[df["sweep"] == "main"].copy()
    order = ["mappo", "Greedy", "EFT", "LB-Greedy", "Local-Only"]
    df = df.set_index("policy").loc[[p for p in order if p in set(df["policy"])]]
    labels = df.index.to_list()

    def bar(metric: str, ylabel: str, fname: str):
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        y = df[f"{metric}_mean"].to_numpy()
        yerr = df[f"{metric}_std"].to_numpy()
        ax.bar(range(len(labels)), y, yerr=yerr, capsize=3, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        _watermark(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / fname, dpi=220)
        plt.close(fig)

    bar("task_success_rate", "成功率", "fig_baseline_success_mock_cn.png")
    bar("mean_cft_est", "平均完工时间（估计）/s", "fig_baseline_cft_mock_cn.png")
    bar("risk_penalty_mean", "平均风险代价", "fig_baseline_risk_mock_cn.png")


def _plot_sweep_lines(figs_dir: Path, sum_csv: Path, sweep: str, xlabel: str, tag: str) -> None:
    d = pd.read_csv(sum_csv)
    d = d[d["sweep"] == sweep].copy()
    d["x"] = d["value"]
    d = d.sort_values(["policy", "x"])

    def line(metric: str, ylabel: str, fname: str):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for p, g in d.groupby("policy"):
            ax.errorbar(
                g["x"].to_numpy(),
                g[f"{metric}_mean"].to_numpy(),
                yerr=g[f"{metric}_std"].to_numpy(),
                marker="o",
                capsize=3,
                linewidth=1.6,
                label=p,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        _watermark(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / fname, dpi=220)
        plt.close(fig)

    line("task_success_rate", "成功率", f"fig_{tag}_success_mock_cn.png")
    line("mean_cft_est", "平均完工时间（估计）/s", f"fig_{tag}_cft_mock_cn.png")
    line("risk_penalty_mean", "平均风险代价", f"fig_{tag}_risk_mock_cn.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default="mock_runs")
    ap.add_argument("--episodes", type=int, default=20, help="每种设置评估回合数（示意）。")
    ap.add_argument("--train-episodes", type=int, default=200, help="收敛曲线长度（示意）。")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    _set_cn_style()
    rng = np.random.default_rng(int(args.seed))
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root).resolve() / f"rc1_mock_progress_{ts}"
    logs_dir = run_dir / "logs"
    exports_dir = run_dir / "paper_exports"
    figs_dir = run_dir / "paper_figs_cn"
    for p in (logs_dir, exports_dir, figs_dir):
        _ensure_dir(p)

    disclaimer = (
        "本目录为“进度检查用合成/示意数据”，不代表真实仿真结果。\n"
        "请勿将其中数值/图表直接用于论文最终结论。\n"
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"随机种子：{args.seed}\n"
    )
    (run_dir / "MOCK_DISCLAIMER.txt").write_text(disclaimer, encoding="utf-8")

    # ------------------------------------------------------------------
    # 1) 生成训练收敛曲线（示意）：metrics.csv
    # ------------------------------------------------------------------
    ep_list = np.arange(1, int(args.train_episodes) + 1)
    # Ideal learning curve: SR rises, CFT/risk fall
    sr_curve = 0.35 + 0.60 * (1.0 - np.exp(-ep_list / 35.0)) + rng.normal(0.0, 0.01, size=len(ep_list))
    cft_curve = 3.60 - 1.15 * (1.0 - np.exp(-ep_list / 45.0)) + rng.normal(0.0, 0.03, size=len(ep_list))
    risk_curve = 0.020 - 0.014 * (1.0 - np.exp(-ep_list / 55.0)) + rng.normal(0.0, 0.0006, size=len(ep_list))
    sr_curve = np.clip(sr_curve, 0.0, 1.0)
    cft_curve = np.clip(cft_curve, 0.6, None)
    risk_curve = np.clip(risk_curve, 0.0, None)
    df_metrics = pd.DataFrame({
        "episode": ep_list,
        "task_success_rate": sr_curve,
        "mean_cft_est": cft_curve,
        "risk_penalty_mean": risk_curve,
    })
    metrics_csv = logs_dir / "metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)

    # ------------------------------------------------------------------
    # 2) 主实验对比（main）
    # ------------------------------------------------------------------
    rows: List[Dict[str, float]] = []
    profiles = _policy_profiles()
    for prof in profiles:
        kind = "learned" if prof.name == "mappo" else "baseline"
        for i in range(int(args.episodes)):
            m = _sample_episode_metrics(rng, prof, cond={}, policy_kind=kind)
            rows.append({"sweep": "main", "value": 0.0, "policy": prof.name, "episode": i + 1, **m})
    df_main = pd.DataFrame(rows)
    _, main_sum = _write_eval_pair(exports_dir, "main_eval_episode_mock", df_main)

    # ------------------------------------------------------------------
    # 3) 规模 sweep（U=10/20/40）
    # ------------------------------------------------------------------
    rows = []
    for U in (10, 20, 40):
        cond = _condition_effect_scale(U)
        for prof in [p for p in profiles if p.name in ("mappo", "Greedy", "Local-Only")]:
            kind = "learned" if prof.name == "mappo" else "baseline"
            for i in range(int(args.episodes)):
                m = _sample_episode_metrics(rng, prof, cond=cond, policy_kind=kind)
                rows.append({"sweep": "scale", "value": float(U), "policy": prof.name, "episode": i + 1, **m})
    df_scale = pd.DataFrame(rows)
    _, scale_sum = _write_eval_pair(exports_dir, "scale_eval_episode_mock", df_scale)

    # ------------------------------------------------------------------
    # 4) 恶意比例 sweep（pm=0/0.1/0.2/0.3）
    # ------------------------------------------------------------------
    rows = []
    for pm in (0.0, 0.1, 0.2, 0.3):
        cond = _condition_effect_malicious(pm)
        for prof in [p for p in profiles if p.name in ("mappo", "Greedy", "Local-Only")]:
            kind = "learned" if prof.name == "mappo" else "baseline"
            mcond = dict(cond)
            # Local-Only is insensitive to maliciousness in this simplified mock
            if prof.name == "Local-Only":
                mcond = {"sr_delta": 0.0, "cft_mul": 1.0, "risk_add": 0.0, "v2v_mul": 1.0, "remote_delta": 0.0}
            for i in range(int(args.episodes)):
                m = _sample_episode_metrics(rng, prof, cond=mcond, policy_kind=kind)
                rows.append({"sweep": "malicious", "value": float(pm), "policy": prof.name, "episode": i + 1, **m})
    df_mal = pd.DataFrame(rows)
    _, mal_sum = _write_eval_pair(exports_dir, "malicious_eval_episode_mock", df_mal)

    # ------------------------------------------------------------------
    # 5) “消融”（示意，不重训）：对 mappo 的模块开关做趋势化扰动
    # ------------------------------------------------------------------
    ablations = {
        "ablation_no_pbca_mock":  {"sr_delta": -0.03, "cft_mul": 1.06, "risk_add": +0.0015, "remote_delta": -0.01},
        "ablation_fixed_power_mock": {"sr_delta": -0.02, "cft_mul": 1.04, "risk_add": +0.0008, "remote_delta": -0.005},
        "ablation_no_trust_mock": {"sr_delta": +0.00, "cft_mul": 1.02, "risk_add": -0.0060, "remote_delta": +0.01},
    }
    for stem, cond in ablations.items():
        rows = []
        prof = next(p for p in profiles if p.name == "mappo")
        for i in range(int(args.episodes)):
            m = _sample_episode_metrics(rng, prof, cond=cond, policy_kind="learned")
            rows.append({"sweep": "main", "value": 0.0, "policy": "mappo", "episode": i + 1, **m})
        df_ab = pd.DataFrame(rows)
        _write_eval_pair(exports_dir, stem, df_ab)

    # ------------------------------------------------------------------
    # 6) 导出简版 LaTeX 表（示意）
    # ------------------------------------------------------------------
    _export_latex_tables(exports_dir, main_sum, scale_sum, mal_sum)

    # ------------------------------------------------------------------
    # 7) 生成图（示意 + 水印）
    # ------------------------------------------------------------------
    _plot_main_convergence(figs_dir, metrics_csv)
    _bar_compare(figs_dir, main_sum)
    _plot_sweep_lines(figs_dir, scale_sum, "scale", "车辆数量 $U$", "scale")
    _plot_sweep_lines(figs_dir, mal_sum, "malicious", "恶意邻车比例 $p_m$", "malicious")

    print(f"✓ Mock run dir: {run_dir}")
    print(f"  - logs: {logs_dir}")
    print(f"  - exports: {exports_dir}")
    print(f"  - figs: {figs_dir}")


if __name__ == "__main__":
    main()

