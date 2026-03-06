"""
[进度检查用] generate_rc1_mock_suite.py

用途（重要）：
  - 生成“合成/示意”的实验数据与对比图，用于阶段性进度检查与后续真实仿真的参考指引；
  - 支持“参考真实训练曲线的形态”，并推导出：
      1) 不同超参数（学习率等）下的收敛曲线（理想趋势，用于“挑最优超参”）
      2) 不同消融（去Transformer/去资源编码/去PBCA/固定功率/去风险输入等）下的收敛曲线
      3) 不同规模/恶意比例/算力倍率等场景下，对比 Ours + Ablations + Baselines 的多指标结果
  - 不代表真实仿真或可复现实验结论，禁止直接作为论文最终结果。

输出目录：
  mock_runs/rc1_mock_suite_<timestamp>/
    MOCK_DISCLAIMER.txt
    reference_meta.json
    logs/reference_metrics_preview.csv
    paper_exports/*.csv      # 合成数据（episode级/summary）
    paper_figs_cn/*.png      # 中文图（带水印）
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# ------------------------------ style & utils ------------------------------

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


def _watermark(ax, text: str = "示意图（合成数据）", alpha: float = 0.12):
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


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _smooth_ma(y: np.ndarray, win: int = 15) -> np.ndarray:
    if win <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(win, min_periods=max(1, win // 3)).mean().to_numpy()


def _latest_1000ep_metrics_csv(runs_dir: Path) -> Optional[Path]:
    # Heuristic: pick the latest run that has logs/metrics.csv with >= 900 rows.
    candidates = list(runs_dir.glob("**/logs/metrics.csv"))
    best: Optional[Tuple[float, Path]] = None
    for p in candidates:
        try:
            st = p.stat()
            # quick filter by size
            if st.st_size < 50_000:
                continue
            df = pd.read_csv(p, usecols=["episode"])
            if len(df) < 900:
                continue
            # prefer most recently modified
            key = st.st_mtime
            if best is None or key > best[0]:
                best = (key, p)
        except Exception:
            continue
    return best[1] if best else None


def _read_reference(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    want = []
    for c in [
        "episode",
        "r_total",
        "reward_mean",
        "task_success_rate",
        "subtask_success_rate",
        "mean_cft_est",
        "energy_norm_mean",
        "I_caused_mean",
        "risk_penalty_mean",
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v",
    ]:
        if c in df.columns:
            want.append(c)
    df = df[want].copy()
    if "episode" in df.columns:
        df = df.sort_values("episode")
    return df


# ------------------------------ convergence synthesis ------------------------------

@dataclass(frozen=True)
class CurveVariant:
    group: str  # "lr" or "ablation"
    name: str
    label: str
    # modifiers (relative to reference)
    speed: float  # >1 faster convergence, <1 slower
    reward_gain: float  # additive bump to reward_mean/r_total after convergence
    sr_gain: float  # additive bump to task_success_rate after convergence
    cft_mul: float  # multiply mean_cft_est
    risk_mul: float  # multiply risk_penalty_mean
    noise: float  # extra noise


def _build_variant_curves(
    rng: np.random.Generator,
    ref: pd.DataFrame,
    variants: List[CurveVariant],
    out_len: Optional[int] = None,
) -> pd.DataFrame:
    ep = ref["episode"].to_numpy()
    n = int(out_len or len(ref))
    ep = np.arange(1, n + 1)

    # Build smoothed base signals from reference (extend by extrapolation if needed).
    def _get_base(col: str, default: float):
        if col in ref.columns:
            y = ref[col].to_numpy(dtype=float)
            if len(y) >= n:
                y = y[:n]
            else:
                # simple tail extension
                tail = float(np.nanmean(y[-20:])) if len(y) > 0 else default
                y = np.concatenate([y, np.full(n - len(y), tail)])
            return _smooth_ma(y, 21)
        return np.full(n, default, dtype=float)

    base_reward = _get_base("reward_mean", default=0.005)
    base_r_total = _get_base("r_total", default=float(np.nanmean(base_reward)))
    base_sr = _get_base("task_success_rate", default=0.4)
    base_cft = _get_base("mean_cft_est", default=4.0)
    base_risk = _get_base("risk_penalty_mean", default=0.012)

    # Normalize "time axis" by warping episode index
    t = np.linspace(0, 1, n)

    rows: List[Dict[str, float]] = []
    for v in variants:
        # time warp: faster convergence means using smaller effective t earlier
        t_eff = np.clip(t ** (1.0 / max(v.speed, 1e-6)), 0.0, 1.0)
        idx = (t_eff * (n - 1)).astype(int)

        reward = base_reward[idx] + (v.reward_gain * (1.0 - np.exp(-5 * t))) + rng.normal(0.0, 0.002 + v.noise, size=n)
        r_total = base_r_total[idx] + (v.reward_gain * (1.0 - np.exp(-5 * t))) + rng.normal(0.0, 0.002 + v.noise, size=n)

        sr = base_sr[idx] + (v.sr_gain * (1.0 - np.exp(-4 * t))) + rng.normal(0.0, 0.015 + 0.6 * v.noise, size=n)
        sr = _clip01(sr)

        cft = np.clip(base_cft[idx] * v.cft_mul + rng.normal(0.0, 0.10 + 0.4 * v.noise, size=n), 0.5, None)
        risk = np.clip(base_risk[idx] * v.risk_mul + rng.normal(0.0, 0.0009 + 0.4 * v.noise, size=n), 0.0, None)

        # produce per-episode row set
        for i in range(n):
            rows.append(
                {
                    "episode": int(ep[i]),
                    "group": v.group,
                    "variant": v.name,
                    "label": v.label,
                    "reward_mean": float(reward[i]),
                    "r_total": float(r_total[i]),
                    "task_success_rate": float(sr[i]),
                    "mean_cft_est": float(cft[i]),
                    "risk_penalty_mean": float(risk[i]),
                }
            )

    return pd.DataFrame(rows)


def _plot_convergence_panel(df: pd.DataFrame, figs_dir: Path, group: str, title: str, prefix: str) -> None:
    d = df[df["group"] == group].copy()
    if d.empty:
        return
    variants = list(dict.fromkeys(d["label"].to_list()))

    def plot_metric(metric: str, ylabel: str, fname: str):
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        for label, g in d.groupby("label"):
            g = g.sort_values("episode")
            ax.plot(g["episode"].to_numpy(), _smooth_ma(g[metric].to_numpy(dtype=float), 15), linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_xlabel("训练轮次 / Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        _watermark(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / fname, dpi=220)
        plt.close(fig)

    plot_metric("reward_mean", "平均奖励（示意）", f"{prefix}_reward_cn.png")
    plot_metric("task_success_rate", "成功率（示意）", f"{prefix}_success_cn.png")
    plot_metric("mean_cft_est", "平均完工时间（估计，示意）/s", f"{prefix}_cft_cn.png")
    plot_metric("risk_penalty_mean", "平均风险代价（示意）", f"{prefix}_risk_cn.png")


# ------------------------------ scenario sweeps synthesis ------------------------------

@dataclass(frozen=True)
class PolicyPerf:
    policy: str
    kind: str  # ours/ablation/baseline
    base_sr: float
    base_subsr: float
    base_cft: float
    base_energy: float
    base_interf: float
    base_risk: float
    ratio_mean: Tuple[float, float, float]  # L/R/V
    power_mean: float


def _policies_for_sweeps() -> List[PolicyPerf]:
    # 强调“理想结果”：ours最佳；ablation次之；baseline可比但略差；Local-Only风险最低但完成差
    return [
        PolicyPerf("EET-PBCA-MAPPO", "ours",     0.92, 0.96, 2.55, 0.030, 1.6e-10, 0.0065, (0.10, 0.82, 0.08), 0.55),
        PolicyPerf("w/o-Transformer", "ablation",0.86, 0.92, 2.85, 0.034, 1.7e-10, 0.0078, (0.12, 0.80, 0.08), 0.55),
        PolicyPerf("w/o-ResourceEnc", "ablation",0.84, 0.91, 2.95, 0.035, 1.8e-10, 0.0085, (0.13, 0.79, 0.08), 0.55),
        PolicyPerf("w/o-PBCA",        "ablation",0.89, 0.94, 2.70, 0.032, 1.7e-10, 0.0072, (0.11, 0.83, 0.06), 0.55),
        PolicyPerf("Fixed-Power",     "ablation",0.88, 0.94, 2.72, 0.031, 1.6e-10, 0.0069, (0.11, 0.83, 0.06), 0.50),
        PolicyPerf("w/o-RiskFeat",    "ablation",0.90, 0.95, 2.62, 0.031, 1.6e-10, 0.0000, (0.09, 0.84, 0.07), 0.55),
        PolicyPerf("Greedy",          "baseline",0.90, 0.95, 2.35, 0.040, 2.0e-10, 0.0135,(0.05, 0.92, 0.03), 1.00),
        PolicyPerf("EFT",             "baseline",0.86, 0.93, 2.55, 0.036, 1.8e-10, 0.0105,(0.10, 0.80, 0.10), 0.60),
        PolicyPerf("LB-Greedy",       "baseline",0.84, 0.92, 2.62, 0.035, 1.8e-10, 0.0095,(0.12, 0.78, 0.10), 0.60),
        PolicyPerf("Local-Only",      "baseline",0.55, 0.88, 3.45, 0.028, 0.5e-10, 0.0000,(1.00, 0.00, 0.00), 0.00),
        PolicyPerf("Random",          "baseline",0.62, 0.89, 3.25, 0.045, 2.2e-10, 0.0120,(0.33, 0.34, 0.33), 0.70),
    ]


def _scenario_factor_scale(U: int) -> Dict[str, float]:
    # U increases -> worse cft/energy/interf, lower sr/subsr, higher risk
    if U <= 10:
        return {"sr": +0.03, "subsr": +0.01, "cft_mul": 0.92, "energy_mul": 0.92, "interf_mul": 0.90, "risk_mul": 0.96}
    if U <= 20:
        return {"sr": 0.00, "subsr": 0.00, "cft_mul": 1.00, "energy_mul": 1.00, "interf_mul": 1.00, "risk_mul": 1.00}
    if U <= 40:
        return {"sr": -0.07, "subsr": -0.03, "cft_mul": 1.18, "energy_mul": 1.12, "interf_mul": 1.20, "risk_mul": 1.10}
    return {"sr": -0.10, "subsr": -0.05, "cft_mul": 1.25, "energy_mul": 1.18, "interf_mul": 1.30, "risk_mul": 1.15}


def _scenario_factor_malicious(pm: float) -> Dict[str, float]:
    pm = float(pm)
    return {
        "sr": -0.10 * pm / 0.3,
        "subsr": -0.05 * pm / 0.3,
        "cft_mul": 1.0 + 0.12 * pm / 0.3,
        "energy_mul": 1.0 + 0.06 * pm / 0.3,
        "interf_mul": 1.0 + 0.05 * pm / 0.3,
        "risk_add": 0.010 * pm / 0.3,
        "v2v_mul": 1.0 - 0.65 * pm / 0.3,
    }


def _scenario_factor_compute(mult_rsu: float = 1.0, mult_veh: float = 1.0) -> Dict[str, float]:
    # More compute -> better cft/sr, slightly lower energy (less waiting), lower risk exposure
    m = float(mult_rsu)
    n = float(mult_veh)
    eff = 0.6 * (m - 1.0) + 0.4 * (n - 1.0)
    return {
        "sr": +0.05 * eff,
        "subsr": +0.02 * eff,
        "cft_mul": max(0.75, 1.0 - 0.18 * eff),
        "energy_mul": max(0.80, 1.0 - 0.10 * eff),
        "interf_mul": 1.0,
        "risk_mul": max(0.85, 1.0 - 0.08 * eff),
    }


def _sample_policy_episode(
    rng: np.random.Generator,
    p: PolicyPerf,
    factor: Dict[str, float],
) -> Dict[str, float]:
    sr = float(np.clip(p.base_sr + factor.get("sr", 0.0) + rng.normal(0.0, 0.02), 0.0, 1.0))
    subsr = float(np.clip(p.base_subsr + factor.get("subsr", 0.0) + rng.normal(0.0, 0.015), 0.0, 1.0))
    cft = float(max(0.5, p.base_cft * factor.get("cft_mul", 1.0) + rng.normal(0.0, 0.10)))
    energy = float(max(0.0, p.base_energy * factor.get("energy_mul", 1.0) + rng.normal(0.0, 0.002)))
    interf = float(max(0.0, p.base_interf * factor.get("interf_mul", 1.0) + rng.normal(0.0, 0.15e-10)))
    risk = p.base_risk
    if "risk_mul" in factor:
        risk *= factor["risk_mul"]
    if "risk_add" in factor:
        risk += factor["risk_add"]
    risk = float(max(0.0, risk + rng.normal(0.0, 0.0012)))
    if p.policy in ("Local-Only", "w/o-RiskFeat"):
        risk = 0.0

    # decision ratios
    L, R, V = p.ratio_mean
    if "v2v_mul" in factor:
        V = float(np.clip(V * factor["v2v_mul"], 0.0, 1.0))
        # move removed mass to RSU
        rem = max(0.0, 1.0 - (L + R + V))
        R = float(np.clip(R + rem, 0.0, 1.0))
    # sample with Dirichlet
    mean = np.array([L, R, V], dtype=float)
    mean = mean / max(mean.sum(), 1e-9)
    alpha = np.maximum(mean * 70.0, 1e-3)
    s = rng.dirichlet(alpha)
    L, R, V = float(s[0]), float(s[1]), float(s[2])

    power = float(np.clip(p.power_mean + rng.normal(0.0, 0.04), 0.0, 1.0))
    if p.policy == "Greedy":
        power = 1.0
    if p.policy == "Fixed-Power":
        power = 0.5
    if p.policy == "Local-Only":
        power = 0.0

    # derived
    dmr = float(np.clip((1.0 - sr) * 0.9 + rng.normal(0.0, 0.02), 0.0, 1.0))
    time_limit_rate = float(np.clip(dmr * 0.25 + rng.normal(0.0, 0.02), 0.0, 1.0))
    return {
        "task_success_rate": sr,
        "subtask_success_rate": subsr,
        "deadline_miss_rate": dmr,
        "mean_cft_est": cft,
        "energy_norm_mean": energy,
        "I_caused_mean": interf,
        "risk_penalty_mean": risk,
        "decision_frac_local": L,
        "decision_frac_rsu": R,
        "decision_frac_v2v": V,
        "power_ratio_mean": power if (1.0 - L) > 1e-6 else 0.0,
        "illegal_action_rate": 0.0,
        "time_limit_rate": time_limit_rate,
    }


def _build_sweep_dataset(
    rng: np.random.Generator,
    sweep: str,
    x_values: List[float],
    episodes: int,
    factor_fn,
    policies: List[PolicyPerf],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for x in x_values:
        factor = factor_fn(x)
        for pol in policies:
            for ep in range(1, episodes + 1):
                m = _sample_policy_episode(rng, pol, factor)
                rows.append({"sweep": sweep, "value": float(x), "policy": pol.policy, "episode": ep, "kind": pol.kind, **m})
    df_ep = pd.DataFrame(rows)
    metric_keys = [
        "task_success_rate",
        "subtask_success_rate",
        "mean_cft_est",
        "energy_norm_mean",
        "I_caused_mean",
        "risk_penalty_mean",
    ]
    df_sum = (
        df_ep.groupby(["sweep", "value", "policy", "kind"], as_index=False)[metric_keys]
        .agg(["mean", "std"])
    )
    # flatten columns
    df_sum.columns = ["_".join([c for c in col if c]) for col in df_sum.columns.to_flat_index()]
    df_sum = df_sum.rename(columns={"sweep_": "sweep", "value_": "value", "policy_": "policy", "kind_": "kind"})
    return df_ep, df_sum


def _plot_sweep_multi_metrics(figs_dir: Path, df_sum: pd.DataFrame, sweep: str, xlabel: str, prefix: str) -> None:
    d = df_sum[df_sum["sweep"] == sweep].copy()
    if d.empty:
        return
    d = d.sort_values(["policy", "value"])

    metrics = [
        ("task_success_rate", "成功率"),
        ("subtask_success_rate", "子任务成功率"),
        ("mean_cft_est", "平均完工时间（估计）/s"),
        ("energy_norm_mean", "能耗（归一化）"),
        ("I_caused_mean", "可控干扰（均值）"),
        ("risk_penalty_mean", "风险代价（均值）"),
    ]
    # Produce 3x2 panel
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 10.2), sharex=True)
    axes = axes.reshape(-1)
    for ax, (m, ylabel) in zip(axes, metrics):
        for pol, g in d.groupby("policy"):
            x = g["value"].to_numpy(dtype=float)
            y = g[f"{m}_mean"].to_numpy(dtype=float)
            yerr = g[f"{m}_std"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.6, label=pol)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        _watermark(ax)
    for ax in axes[-2:]:
        ax.set_xlabel(xlabel)
    # Legend once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"{sweep} 场景指标对比（合成示意）", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figs_dir / f"{prefix}_{sweep}_panel_cn.png", dpi=220)
    plt.close(fig)


def _plot_sweep_per_metric(
    figs_dir: Path,
    df_sum: pd.DataFrame,
    sweep: str,
    xlabel: str,
    *,
    prefix: str,
    policy_filter: Optional[List[str]] = None,
    title_suffix: str = "",
) -> None:
    d = df_sum[df_sum["sweep"] == sweep].copy()
    if policy_filter:
        d = d[d["policy"].isin(policy_filter)].copy()
    if d.empty:
        return
    d = d.sort_values(["policy", "value"])

    metrics = [
        ("task_success_rate", "成功率"),
        ("subtask_success_rate", "子任务成功率"),
        ("mean_cft_est", "平均完工时间（估计）/s"),
        ("energy_norm_mean", "能耗（归一化）"),
        ("I_caused_mean", "可控干扰（均值）"),
        ("risk_penalty_mean", "风险代价（均值）"),
    ]
    for m, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        for pol, g in d.groupby("policy"):
            x = g["value"].to_numpy(dtype=float)
            y = g[f"{m}_mean"].to_numpy(dtype=float)
            yerr = g[f"{m}_std"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.6, label=pol)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{sweep} 场景：{ylabel}对比{title_suffix}")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=False)
        _watermark(ax)
        fig.tight_layout()
        fig.savefig(figs_dir / f"{prefix}_{sweep}_{m}_cn.png", dpi=220)
        plt.close(fig)


# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default="mock_runs")
    ap.add_argument("--reference-metrics", type=str, default=None, help="参考的真实metrics.csv路径（可选）。")
    ap.add_argument("--seed", type=int, default=20260303)
    ap.add_argument("--conv-episodes", type=int, default=1000, help="收敛曲线长度（示意）。")
    ap.add_argument("--eval-episodes", type=int, default=30, help="每个设置评估episode数（示意）。")
    args = ap.parse_args()

    _set_cn_style()
    rng = np.random.default_rng(int(args.seed))

    runs_dir = Path("runs").resolve()
    ref_path = Path(args.reference_metrics).resolve() if args.reference_metrics else _latest_1000ep_metrics_csv(runs_dir)
    if ref_path is None or not ref_path.exists():
        raise FileNotFoundError("找不到参考 metrics.csv。请用 --reference-metrics 显式指定。")

    ref = _read_reference(ref_path)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root).resolve() / f"rc1_mock_suite_{ts}"
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
        f"参考曲线：{str(ref_path)}\n"
    )
    (run_dir / "MOCK_DISCLAIMER.txt").write_text(disclaimer, encoding="utf-8")
    (run_dir / "reference_meta.json").write_text(
        json.dumps({"reference_metrics_csv": str(ref_path), "seed": int(args.seed)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Save a preview for quick inspection
    ref.head(60).to_csv(logs_dir / "reference_metrics_preview.csv", index=False)

    # ---------------- convergence: lr sweep + ablations ----------------
    lr_variants = [
        CurveVariant("lr", "lr_1e-4", "学习率=1e-4（偏慢）", speed=0.85, reward_gain=0.002, sr_gain=0.015, cft_mul=1.02, risk_mul=1.00, noise=0.001),
        CurveVariant("lr", "lr_2e-4", "学习率=2e-4（最优）", speed=1.10, reward_gain=0.006, sr_gain=0.030, cft_mul=0.97, risk_mul=0.98, noise=0.001),
        CurveVariant("lr", "lr_5e-4", "学习率=5e-4（快但抖）", speed=1.35, reward_gain=0.003, sr_gain=0.020, cft_mul=0.99, risk_mul=1.00, noise=0.003),
        CurveVariant("lr", "lr_8e-5", "学习率=8e-5（更稳但更慢）", speed=0.75, reward_gain=0.001, sr_gain=0.010, cft_mul=1.03, risk_mul=1.00, noise=0.0008),
    ]
    ab_variants = [
        CurveVariant("ablation", "full", "完整模型（最优）", speed=1.10, reward_gain=0.006, sr_gain=0.030, cft_mul=0.97, risk_mul=0.98, noise=0.001),
        CurveVariant("ablation", "no_transformer", "去Transformer（退化）", speed=0.95, reward_gain=0.001, sr_gain=-0.010, cft_mul=1.06, risk_mul=1.05, noise=0.002),
        CurveVariant("ablation", "no_resource_enc", "去资源编码（退化）", speed=0.90, reward_gain=0.0005, sr_gain=-0.015, cft_mul=1.08, risk_mul=1.06, noise=0.002),
        CurveVariant("ablation", "no_pbca", "去PBCA（退化）", speed=1.00, reward_gain=0.002, sr_gain=-0.008, cft_mul=1.04, risk_mul=1.03, noise=0.0015),
        CurveVariant("ablation", "fixed_power", "固定功率（退化）", speed=1.05, reward_gain=0.0015, sr_gain=-0.006, cft_mul=1.03, risk_mul=1.00, noise=0.0015),
    ]

    df_conv = _build_variant_curves(rng, ref, lr_variants + ab_variants, out_len=int(args.conv_episodes))
    conv_csv = exports_dir / "convergence_variants_mock.csv"
    df_conv.to_csv(conv_csv, index=False)
    _plot_convergence_panel(df_conv, figs_dir, "lr", "不同学习率下的收敛曲线对比（合成示意）", "fig_hp_lr")
    _plot_convergence_panel(df_conv, figs_dir, "ablation", "消融设置下的收敛曲线对比（合成示意）", "fig_ablation_conv")

    # ---------------- scenario sweeps: scale / malicious / compute ----------------
    policies = _policies_for_sweeps()

    # scale
    df_scale_ep, df_scale_sum = _build_sweep_dataset(
        rng, "scale", [10, 20, 40], int(args.eval_episodes), lambda x: _scenario_factor_scale(int(x)), policies
    )
    df_scale_ep.to_csv(exports_dir / "scale_eval_episode_mock_plus.csv", index=False)
    df_scale_sum.to_csv(exports_dir / "scale_eval_episode_mock_plus_summary.csv", index=False)
    _plot_sweep_multi_metrics(figs_dir, df_scale_sum, "scale", "车辆数量 $U$", "fig")
    ours_abls = [p.policy for p in policies if p.kind in ("ours", "ablation")]
    ours_bases = [p.policy for p in policies if p.kind in ("ours", "baseline")]
    _plot_sweep_per_metric(figs_dir, df_scale_sum, "scale", "车辆数量 $U$", prefix="fig_ours_abls", policy_filter=ours_abls, title_suffix="（本文方法+消融）")
    _plot_sweep_per_metric(figs_dir, df_scale_sum, "scale", "车辆数量 $U$", prefix="fig_ours_bases", policy_filter=ours_bases, title_suffix="（本文方法+基线）")

    # malicious
    df_mal_ep, df_mal_sum = _build_sweep_dataset(
        rng, "malicious", [0.0, 0.1, 0.2, 0.3], int(args.eval_episodes), lambda x: _scenario_factor_malicious(float(x)), policies
    )
    df_mal_ep.to_csv(exports_dir / "malicious_eval_episode_mock_plus.csv", index=False)
    df_mal_sum.to_csv(exports_dir / "malicious_eval_episode_mock_plus_summary.csv", index=False)
    _plot_sweep_multi_metrics(figs_dir, df_mal_sum, "malicious", "恶意比例 $p_m$", "fig")
    _plot_sweep_per_metric(figs_dir, df_mal_sum, "malicious", "恶意比例 $p_m$", prefix="fig_ours_abls", policy_filter=ours_abls, title_suffix="（本文方法+消融）")
    _plot_sweep_per_metric(figs_dir, df_mal_sum, "malicious", "恶意比例 $p_m$", prefix="fig_ours_bases", policy_filter=ours_bases, title_suffix="（本文方法+基线）")

    # compute sweep (RSU multiplier)
    rsu_mults = [0.7, 1.0, 1.3, 1.6]
    df_rsu_ep, df_rsu_sum = _build_sweep_dataset(
        rng, "rsu_compute", rsu_mults, int(args.eval_episodes),
        lambda x: _scenario_factor_compute(mult_rsu=float(x), mult_veh=1.0),
        policies,
    )
    df_rsu_ep.to_csv(exports_dir / "rsu_compute_eval_episode_mock_plus.csv", index=False)
    df_rsu_sum.to_csv(exports_dir / "rsu_compute_eval_episode_mock_plus_summary.csv", index=False)
    _plot_sweep_multi_metrics(figs_dir, df_rsu_sum, "rsu_compute", "RSU算力倍率", "fig")
    _plot_sweep_per_metric(figs_dir, df_rsu_sum, "rsu_compute", "RSU算力倍率", prefix="fig_ours_abls", policy_filter=ours_abls, title_suffix="（本文方法+消融）")
    _plot_sweep_per_metric(figs_dir, df_rsu_sum, "rsu_compute", "RSU算力倍率", prefix="fig_ours_bases", policy_filter=ours_bases, title_suffix="（本文方法+基线）")

    # compute sweep (Vehicle multiplier)
    veh_mults = [0.7, 1.0, 1.3, 1.6]
    df_veh_ep, df_veh_sum = _build_sweep_dataset(
        rng, "veh_compute", veh_mults, int(args.eval_episodes),
        lambda x: _scenario_factor_compute(mult_rsu=1.0, mult_veh=float(x)),
        policies,
    )
    df_veh_ep.to_csv(exports_dir / "veh_compute_eval_episode_mock_plus.csv", index=False)
    df_veh_sum.to_csv(exports_dir / "veh_compute_eval_episode_mock_plus_summary.csv", index=False)
    _plot_sweep_multi_metrics(figs_dir, df_veh_sum, "veh_compute", "车辆算力倍率", "fig")
    _plot_sweep_per_metric(figs_dir, df_veh_sum, "veh_compute", "车辆算力倍率", prefix="fig_ours_abls", policy_filter=ours_abls, title_suffix="（本文方法+消融）")
    _plot_sweep_per_metric(figs_dir, df_veh_sum, "veh_compute", "车辆算力倍率", prefix="fig_ours_bases", policy_filter=ours_bases, title_suffix="（本文方法+基线）")

    print(f"✓ Mock suite dir: {run_dir}")
    print(f"  - DISCLAIMER: {run_dir / 'MOCK_DISCLAIMER.txt'}")
    print(f"  - convergence csv: {conv_csv}")
    print(f"  - figs: {figs_dir}")


if __name__ == "__main__":
    main()
