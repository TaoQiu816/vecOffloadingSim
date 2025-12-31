#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import numpy as np


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _add_repo_root():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)


def _fmt_range(lo, hi, unit="s"):
    return f"[{lo:.4g}, {hi:.4g}] {unit}"


def _safe_div(num, den):
    return num / max(den, 1e-9)


def _sample_v2v_rate():
    from configs.config import SystemConfig as Cfg
    from envs.modules.channel import ChannelModel
    from envs.entities.vehicle import Vehicle

    np.random.seed(0)
    channel = ChannelModel()
    v = Vehicle(0, np.array([0.0, 0.0]))
    v.tx_power_dbm = getattr(Cfg, "TX_POWER_DEFAULT_DBM", Cfg.TX_POWER_MIN_DBM)
    dist = min(50.0, Cfg.V2V_RANGE * 0.3)
    tgt_pos = np.array([dist, 0.0])
    rates = [channel.compute_one_rate(v, tgt_pos, "V2V", curr_time=0.0) for _ in range(200)]
    return float(np.median(rates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results_dbg/param_inventory")
    args = parser.parse_args()

    _add_repo_root()
    from configs.config import SystemConfig as Cfg

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "param_sanity_report.md"

    comp_min = float(Cfg.MIN_COMP)
    comp_max = float(Cfg.MAX_COMP)
    freq_min = float(Cfg.MIN_VEHICLE_CPU_FREQ)
    freq_max = float(Cfg.MAX_VEHICLE_CPU_FREQ)
    f_rsu = float(Cfg.F_RSU)

    comp_vehicle_min = _safe_div(comp_min, freq_max)
    comp_vehicle_max = _safe_div(comp_max, freq_min)
    comp_rsu_min = _safe_div(comp_min, f_rsu)
    comp_rsu_max = _safe_div(comp_max, f_rsu)

    data_min = float(Cfg.MIN_DATA)
    data_max = float(Cfg.MAX_DATA)

    users_list = [1, 6, 12]
    tx_v2i_rows = []
    for users in users_list:
        bw_eff = _safe_div(Cfg.BW_V2I, users)
        tx_min = _safe_div(data_min, bw_eff)
        tx_max = _safe_div(data_max, bw_eff)
        tx_v2i_rows.append((users, tx_min, tx_max))

    v2v_rate_est = None
    try:
        v2v_rate_est = _sample_v2v_rate()
    except Exception:
        v2v_rate_est = float(Cfg.NORM_MAX_RATE_V2V)

    tx_v2v_min = _safe_div(data_min, v2v_rate_est)
    tx_v2v_max = _safe_div(data_max, v2v_rate_est)

    avg_comp = (comp_min + comp_max) / 2.0
    avg_data = (data_min + data_max) / 2.0
    comp_vehicle_mid = _safe_div(avg_comp, (freq_min + freq_max) / 2.0)
    comp_rsu_mid = _safe_div(avg_comp, f_rsu)
    tx_v2i_mid = _safe_div(avg_data, _safe_div(Cfg.BW_V2I, 6))
    tx_v2v_mid = _safe_div(avg_data, v2v_rate_est)

    lines = [
        "# Parameter Sanity Report",
        "",
        "## 1) 计算时延量级",
        "",
        f"- comp_time_vehicle = C / f_vehicle, C∈[{comp_min:.3g},{comp_max:.3g}], "
        f"f∈[{freq_min:.3g},{freq_max:.3g}]",
        f"  - range: {_fmt_range(comp_vehicle_min, comp_vehicle_max)}",
        f"- comp_time_rsu = C / F_RSU, F_RSU={f_rsu:.3g}",
        f"  - range: {_fmt_range(comp_rsu_min, comp_rsu_max)}",
        "",
        "## 2) 传输时延量级",
        "",
        "- tx_time_v2i = D / (BW_V2I / users)",
    ]
    for users, tx_min, tx_max in tx_v2i_rows:
        lines.append(f"  - users={users}: {_fmt_range(tx_min, tx_max)}")
    lines.extend([
        "- tx_time_v2v = D / R_v2v_typ",
        f"  - R_v2v_typ ≈ {v2v_rate_est:.3g} bps",
        f"  - range: {_fmt_range(tx_v2v_min, tx_v2v_max)}",
        "",
        "## 3) 量级对比结论",
        "",
        f"- mid comp_time_vehicle ≈ {comp_vehicle_mid:.4g}s, comp_time_rsu ≈ {comp_rsu_mid:.4g}s",
        f"- mid tx_time_v2i (users=6) ≈ {tx_v2i_mid:.4g}s, mid tx_time_v2v ≈ {tx_v2v_mid:.4g}s",
        "- 若 comp_time_rsu << comp_time_vehicle 且 tx_time_v2i 不高，RSU 将在成本上占优。",
        "",
        "## 4) 让 V2V 有机会成为 argmin 的必要条件",
        "",
        "- 在部分状态满足: (tx + wait + comp)_v2v < (tx + wait + comp)_rsu。",
        "- 这通常要求：V2V 链路速率/距离优势 + 邻居排队不高 + RSU 排队或算力相对变弱。",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
