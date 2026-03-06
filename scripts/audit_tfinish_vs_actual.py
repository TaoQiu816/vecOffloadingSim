#!/usr/bin/env python3
"""White-box audit: t_finish_norm estimate fairness vs realized completion time.

Scope (minimal intrusion):
- Pure inference only (no training)
- Same seeds for MAPPO(best checkpoint) and Greedy
- No environment main-logic / physical-layer changes

Outputs:
- decisions_*.csv: on-task decision-level candidate snapshot and selected target
- completions_*.csv: completed-subtask t_est/t_actual records from env audit callback
- completed_joined_*.csv: decision-completion joined samples
- per_type_stats.csv: Local/RSU/V2V correlation & error stats
- candidate_structure_stats.csv: min_nonrsu_vs_rsu window statistics
- policy_compare_summary.csv: compact policy comparison
- conclusion.md: one-page verdict following requested rules
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mappo_agent import MAPPOAgent
from baselines import GreedyPolicy
from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from models.offloading_policy import OffloadingPolicyNetwork


TYPE_NAME = {1: "Local", 2: "RSU", 3: "V2V"}


@dataclass
class PolicyRunner:
    name: str
    env: VecOffloadingEnv
    policy_obj: object
    is_mappo: bool


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_mappo_agent(checkpoint: str, device: str) -> MAPPOAgent:
    net = OffloadingPolicyNetwork(
        d_model=TC.EMBED_DIM,
        num_heads=TC.NUM_HEADS,
        num_layers=TC.NUM_LAYERS,
    )
    agent = MAPPOAgent(net, device=device)
    agent.load(checkpoint, restore_optimizer=False, restore_scheduler=False)
    agent.network.eval()
    return agent


def _candidate_rows_from_obs(obs: Dict) -> List[Dict]:
    mask = np.asarray(obs.get("action_mask", []), dtype=np.float32).reshape(-1)
    types = np.asarray(obs.get("candidate_types", []), dtype=np.int64).reshape(-1)
    ids = np.asarray(obs.get("candidate_ids", []), dtype=np.int64).reshape(-1)
    raw = np.asarray(obs.get("resource_raw", []), dtype=np.float32)

    out: List[Dict] = []
    n = min(len(mask), len(types), raw.shape[0] if raw.ndim == 2 else 0)
    for i in range(n):
        if mask[i] <= 0:
            continue
        t = int(types[i])
        if t not in (1, 2, 3):
            continue
        rid = int(ids[i]) if i < len(ids) else -1
        rr = raw[i]
        out.append(
            {
                "idx": int(i),
                "candidate_id": rid,
                "type_code": t,
                "type": TYPE_NAME.get(t, "Unknown"),
                "t_finish_norm": float(rr[8]) if rr.shape[0] > 8 else float("nan"),
                "queue_norm": float(rr[1]) if rr.shape[0] > 1 else float("nan"),
                "dist_norm": float(rr[2]) if rr.shape[0] > 2 else float("nan"),
                "contact_norm": float(rr[10]) if rr.shape[0] > 10 else float("nan"),
            }
        )
    return out


def _find_selected_idx(obs: Dict, plan: Dict) -> Optional[int]:
    mask = np.asarray(obs.get("action_mask", []), dtype=np.float32).reshape(-1)
    types = np.asarray(obs.get("candidate_types", []), dtype=np.int64).reshape(-1)
    ids = np.asarray(obs.get("candidate_ids", []), dtype=np.int64).reshape(-1)

    def _valid(i: int) -> bool:
        return 0 <= i < len(mask) and mask[i] > 0 and 0 <= i < len(types)

    kind = str(plan.get("planned_kind", "")).lower()
    target = plan.get("planned_target")

    if kind == "local":
        for i in range(min(len(mask), len(types))):
            if _valid(i) and int(types[i]) == 1:
                return int(i)
        return 0 if _valid(0) else None

    if kind == "rsu" and isinstance(target, tuple) and len(target) >= 2:
        rsu_id = int(target[1])
        for i in range(min(len(mask), len(types), len(ids))):
            if _valid(i) and int(types[i]) == 2 and int(ids[i]) == rsu_id:
                return int(i)

    if kind == "v2v" and isinstance(target, int):
        nid = int(target)
        for i in range(min(len(mask), len(types), len(ids))):
            if _valid(i) and int(types[i]) == 3 and int(ids[i]) == nid:
                return int(i)

    raw_idx = plan.get("target_idx")
    if raw_idx is not None:
        try:
            i = int(raw_idx)
            if _valid(i):
                return i
        except Exception:
            pass
    return None


def _decision_row(
    *,
    policy: str,
    ep_idx: int,
    seed: int,
    step: int,
    obs: Dict,
    plan: Dict,
) -> Tuple[Dict, List[Dict]]:
    candidates = _candidate_rows_from_obs(obs)

    t_rsu = [c["t_finish_norm"] for c in candidates if c["type_code"] == 2 and np.isfinite(c["t_finish_norm"])]
    t_non = [c["t_finish_norm"] for c in candidates if c["type_code"] in (1, 3) and np.isfinite(c["t_finish_norm"])]
    min_rsu = float(min(t_rsu)) if t_rsu else float("nan")
    min_non = float(min(t_non)) if t_non else float("nan")
    comparable = int(np.isfinite(min_rsu) and np.isfinite(min_non))
    nonrsu_beats_rsu = int(min_non < min_rsu) if comparable else -1

    sel_idx = _find_selected_idx(obs, plan)
    sel_type = str(plan.get("planned_kind", "unknown")).lower()
    sel_type_name = {"local": "Local", "rsu": "RSU", "v2v": "V2V"}.get(sel_type, "Unknown")

    raw = np.asarray(obs.get("resource_raw", []), dtype=np.float32)
    cids = np.asarray(obs.get("candidate_ids", []), dtype=np.int64).reshape(-1)
    t_finish_sel = float("nan")
    queue_sel = float("nan")
    dist_sel = float("nan")
    contact_sel = float("nan")
    cand_id_sel = -1
    if sel_idx is not None and raw.ndim == 2 and 0 <= sel_idx < raw.shape[0]:
        rr = raw[sel_idx]
        t_finish_sel = float(rr[8]) if rr.shape[0] > 8 else float("nan")
        queue_sel = float(rr[1]) if rr.shape[0] > 1 else float("nan")
        dist_sel = float(rr[2]) if rr.shape[0] > 2 else float("nan")
        contact_sel = float(rr[10]) if rr.shape[0] > 10 else float("nan")
        if 0 <= sel_idx < len(cids):
            cand_id_sel = int(cids[sel_idx])

    decision = {
        "policy": policy,
        "episode": int(ep_idx),
        "seed": int(seed),
        "step": int(step),
        "vehicle_id": int(plan.get("vehicle_id", -1)),
        "subtask_id": int(plan.get("subtask_idx", -1)),
        "selected_type": sel_type_name,
        "selected_idx": int(sel_idx) if sel_idx is not None else -1,
        "selected_candidate_id": int(cand_id_sel),
        "t_finish_norm_sel": float(t_finish_sel),
        "queue_norm_sel": float(queue_sel),
        "dist_norm_sel": float(dist_sel),
        "contact_norm_sel": float(contact_sel),
        "min_t_finish_rsu": float(min_rsu),
        "min_t_finish_nonrsu": float(min_non),
        "comparable_nonrsu_vs_rsu": int(comparable),
        "nonrsu_beats_rsu_est": int(nonrsu_beats_rsu),
        "illegal_reason": str(plan.get("illegal_reason", "")) if plan.get("illegal_reason") is not None else "",
        "candidates_json": json.dumps(candidates, ensure_ascii=True),
    }
    candidate_rows: List[Dict] = []
    for c in candidates:
        candidate_rows.append(
            {
                "policy": policy,
                "episode": int(ep_idx),
                "seed": int(seed),
                "step": int(step),
                "vehicle_id": int(plan.get("vehicle_id", -1)),
                "subtask_id": int(plan.get("subtask_idx", -1)),
                "candidate_idx": int(c["idx"]),
                "candidate_id": int(c["candidate_id"]),
                "candidate_type": str(c["type"]),
                "candidate_type_code": int(c["type_code"]),
                "t_finish_norm": float(c["t_finish_norm"]),
                "queue_norm": float(c["queue_norm"]),
                "dist_norm": float(c["dist_norm"]),
                "contact_norm": float(c["contact_norm"]),
                "is_selected": int(sel_idx is not None and int(c["idx"]) == int(sel_idx)),
                "nonrsu_beats_rsu_est": int(nonrsu_beats_rsu),
            }
        )
    return decision, candidate_rows


def _run_policy(
    *,
    name: str,
    env: VecOffloadingEnv,
    policy_obj: object,
    is_mappo: bool,
    episodes: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions: List[Dict] = []
    completions: List[Dict] = []
    candidates: List[Dict] = []

    for ep in range(episodes):
        ep_seed = base_seed + ep
        _set_seed(ep_seed)
        obs_list, _ = env.reset(seed=ep_seed)
        if hasattr(policy_obj, "reset"):
            try:
                policy_obj.reset()
            except TypeError:
                policy_obj.reset(ep_seed)

        done = False
        step = 0
        while not done:
            pre_obs = obs_list
            if is_mappo:
                out = policy_obj.select_action(obs_list, deterministic=True)
                actions = out["actions"]
            else:
                actions = policy_obj.select_action(obs_list)

            obs_list, _, terminated, truncated, _ = env.step(actions)
            done = bool(terminated or truncated)

            obs_by_vid: Dict[int, Dict] = {}
            for i, ob in enumerate(pre_obs):
                if i < len(env.vehicles):
                    obs_by_vid[int(env.vehicles[i].id)] = ob

            for plan in getattr(env, "_last_commit_plans", []):
                if plan.get("subtask_idx") is None:
                    continue
                vid = int(plan.get("vehicle_id", -1))
                ob = obs_by_vid.get(vid)
                if ob is None:
                    continue
                drow, crows = _decision_row(
                        policy=name,
                        ep_idx=ep + 1,
                        seed=ep_seed,
                        step=step,
                        obs=ob,
                        plan=plan,
                    )
                decisions.append(drow)
                candidates.extend(crows)
            step += 1

        for rec in getattr(env, "_audit_t_est_records", []):
            completions.append(
                {
                    "policy": name,
                    "episode": int(ep + 1),
                    "seed": int(ep_seed),
                    "vehicle_id": int(rec.get("vehicle_id", -1)),
                    "subtask_id": int(rec.get("subtask_id", -1)),
                    "action_type": str(rec.get("action_type", "Unknown")),
                    "t_est_env": float(rec.get("t_actual_est", float("nan"))),
                    "t_actual": float(rec.get("t_actual_real", float("nan"))),
                    "est_error_env": float(rec.get("est_error", float("nan"))),
                }
            )

        if (ep + 1) % 10 == 0:
            print(f"[{name}] episode {ep + 1}/{episodes}", flush=True)

    return pd.DataFrame(decisions), pd.DataFrame(completions), pd.DataFrame(candidates)


def _t_est_from_norm(t_finish_norm: pd.Series) -> pd.Series:
    # inverse of env _finish_norm: log1p(t) / log1p(MAX_STEPS*DT)
    denom = float(np.log1p(max(float(Cfg.MAX_STEPS) * float(Cfg.DT), 1e-6)))
    x = pd.to_numeric(t_finish_norm, errors="coerce").astype(float)
    return np.expm1(np.clip(x, 0.0, 1.0) * max(denom, 1e-9))


def _per_type_stats(decisions: pd.DataFrame, joined: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows: List[Dict] = []
    eps = 1e-9

    for tname in ["Local", "RSU", "V2V"]:
        d_t = decisions[decisions["selected_type"] == tname]
        j_t = joined[joined["action_type"] == tname].copy()

        n_decisions = int(len(d_t))
        n_completed = int(len(j_t))
        completion_rate = float(n_completed / n_decisions) if n_decisions > 0 else float("nan")
        fail_rate = float(1.0 - completion_rate) if np.isfinite(completion_rate) else float("nan")

        if n_completed > 0:
            j_t["t_est_used"] = pd.to_numeric(j_t["t_est_env"], errors="coerce")
            j_t["err_used"] = j_t["t_actual"] - j_t["t_est_used"]
            j_t["ratio_used"] = j_t["t_actual"] / (j_t["t_est_used"] + eps)

            valid_corr = j_t[["t_finish_norm_sel", "t_actual"]].dropna()
            if len(valid_corr) >= 2:
                sp = float(valid_corr.corr(method="spearman").iloc[0, 1])
            else:
                sp = float("nan")
            err = pd.to_numeric(j_t["err_used"], errors="coerce")
            ratio = pd.to_numeric(j_t["ratio_used"], errors="coerce")

            row = {
                "policy": policy,
                "type": tname,
                "n_decisions": n_decisions,
                "n_completed": n_completed,
                "completion_rate": completion_rate,
                "failure_rate": fail_rate,
                "spearman_tfinishnorm_vs_tactual": sp,
                "mean_err_sec": float(err.mean()),
                "median_err_sec": float(err.median()),
                "p50_err_sec": float(err.quantile(0.50)),
                "p90_err_sec": float(err.quantile(0.90)),
                "p95_err_sec": float(err.quantile(0.95)),
                "mean_ratio": float(ratio.mean()),
                "median_ratio": float(ratio.median()),
            }
        else:
            row = {
                "policy": policy,
                "type": tname,
                "n_decisions": n_decisions,
                "n_completed": n_completed,
                "completion_rate": completion_rate,
                "failure_rate": fail_rate,
                "spearman_tfinishnorm_vs_tactual": float("nan"),
                "mean_err_sec": float("nan"),
                "median_err_sec": float("nan"),
                "p50_err_sec": float("nan"),
                "p90_err_sec": float("nan"),
                "p95_err_sec": float("nan"),
                "mean_ratio": float("nan"),
                "median_ratio": float("nan"),
            }
        rows.append(row)

    return pd.DataFrame(rows)


def _candidate_structure_stats(decisions: pd.DataFrame, policy: str) -> pd.DataFrame:
    d = decisions.copy()
    if d.empty:
        return pd.DataFrame(
            [
                {
                    "policy": policy,
                    "on_task_decisions": 0,
                    "comparable_count": 0,
                    "p_min_nonrsu_lt_rsu_given_comparable": float("nan"),
                    "p_min_nonrsu_lt_rsu_all": float("nan"),
                    "choose_nonrsu_given_win": float("nan"),
                    "choose_rsu_given_win": float("nan"),
                    "actual_nonrsu_beats_rsu_rate": float("nan"),
                    "actual_nonrsu_beats_rsu_note": "counterfactual_not_available",
                }
            ]
        )

    d["is_nonrsu"] = d["selected_type"].isin(["Local", "V2V"]).astype(int)
    d["is_rsu"] = (d["selected_type"] == "RSU").astype(int)
    comp = d[d["comparable_nonrsu_vs_rsu"] == 1]
    win = comp[comp["nonrsu_beats_rsu_est"] == 1]

    on_task = len(d)
    comparable_count = len(comp)
    win_count = len(win)

    row = {
        "policy": policy,
        "on_task_decisions": int(on_task),
        "comparable_count": int(comparable_count),
        "p_min_nonrsu_lt_rsu_given_comparable": float(win_count / comparable_count) if comparable_count > 0 else float("nan"),
        "p_min_nonrsu_lt_rsu_all": float(win_count / on_task) if on_task > 0 else float("nan"),
        "choose_nonrsu_given_win": float(win["is_nonrsu"].mean()) if win_count > 0 else float("nan"),
        "choose_rsu_given_win": float(win["is_rsu"].mean()) if win_count > 0 else float("nan"),
        "actual_nonrsu_beats_rsu_rate": float("nan"),
        "actual_nonrsu_beats_rsu_note": "counterfactual_not_available_same_subtask",
    }
    return pd.DataFrame([row])


def _build_conclusion(
    *,
    out_dir: Path,
    per_type: pd.DataFrame,
    cand_stats: pd.DataFrame,
    compare: pd.DataFrame,
) -> None:
    def _pick(df: pd.DataFrame, policy: str, tname: str, col: str) -> float:
        x = df[(df["policy"] == policy) & (df["type"] == tname)]
        if x.empty:
            return float("nan")
        try:
            return float(x.iloc[0][col])
        except Exception:
            return float("nan")

    def _cand(policy: str, col: str) -> float:
        x = cand_stats[cand_stats["policy"] == policy]
        if x.empty:
            return float("nan")
        try:
            return float(x.iloc[0][col])
        except Exception:
            return float("nan")

    p_win_m = _cand("mappo", "p_min_nonrsu_lt_rsu_given_comparable")
    p_win_g = _cand("greedy", "p_min_nonrsu_lt_rsu_given_comparable")
    choose_non_m = _cand("mappo", "choose_nonrsu_given_win")
    choose_non_g = _cand("greedy", "choose_nonrsu_given_win")

    ratio_local = _pick(per_type, "mappo", "Local", "mean_ratio")
    ratio_rsu = _pick(per_type, "mappo", "RSU", "mean_ratio")
    ratio_v2v = _pick(per_type, "mappo", "V2V", "mean_ratio")

    # Rule-based decision
    # severe estimation bias heuristic: non-RSU ratio deviates strongly from RSU
    severe_bias = False
    bias_reasons: List[str] = []
    for tname, rv in [("Local", ratio_local), ("V2V", ratio_v2v)]:
        if np.isfinite(rv) and np.isfinite(ratio_rsu):
            if abs(rv - 1.0) > 0.5 and abs(rv - ratio_rsu) > 0.3:
                severe_bias = True
                bias_reasons.append(f"{tname} mean_ratio={rv:.3f} vs RSU={ratio_rsu:.3f}")

    structural = np.isfinite(p_win_m) and p_win_m < 0.05
    learning_impl = (np.isfinite(p_win_m) and p_win_m >= 0.05 and np.isfinite(choose_non_m) and choose_non_m < 0.20)

    if severe_bias:
        verdict = "估计偏差问题"
        next_step = "先修 t_finish_est 公式或归一化，再做场景参数调整。"
        why = "; ".join(bias_reasons)
    elif structural:
        verdict = "环境结构性 RSU 优势"
        next_step = "优先走 P1b（提升 V2V 窗口）或进一步削弱 RSU 相对优势。"
        why = f"P(min_nonrsu < rsu)={p_win_m:.3f} 接近 0。"
    elif learning_impl:
        verdict = "学习/实现问题"
        next_step = "优先检查训练稳定性与特征使用（mask、归一化、熵/塌缩控制），而非继续改物理参数。"
        why = (
            f"存在非RSU胜窗 P={p_win_m:.3f}，但 MAPPO 在胜窗下选 nonRSU 概率仅 {choose_non_m:.3f}，"
            f"Greedy 为 {choose_non_g:.3f}。"
        )
    else:
        verdict = "混合因素（需小步复核）"
        next_step = "先做 1-episode 白盒相关性核查（t_finish_norm 与真实完工时序一致性），再决定走场景或训练侧。"
        why = "结构性与偏差信号均不极端。"

    lines: List[str] = []
    lines.append("# t_finish_norm 白盒审计结论（一页）")
    lines.append("")
    lines.append("## 1) 核心结论")
    lines.append(f"- 判定：**{verdict}**")
    lines.append(f"- 主要证据：{why}")
    lines.append(f"- 下一步（仅1个）：{next_step}")
    lines.append("")
    lines.append("## 2) 关键统计（同 seeds, 纯推理）")
    for p in ["mappo", "greedy"]:
        lines.append(
            f"- {p}: P(min_nonrsu<rsu|comparable)={_cand(p, 'p_min_nonrsu_lt_rsu_given_comparable'):.3f}, "
            f"P(select_nonrsu|min_nonrsu<rsu)={_cand(p, 'choose_nonrsu_given_win'):.3f}"
        )
    lines.append("")
    lines.append("## 3) 估计误差摘要（MAPPO）")
    for tname in ["Local", "RSU", "V2V"]:
        r = per_type[(per_type["policy"] == "mappo") & (per_type["type"] == tname)]
        if r.empty:
            continue
        rr = r.iloc[0]
        lines.append(
            f"- {tname}: n_completed={int(rr['n_completed'])}, "
            f"Spearman={rr['spearman_tfinishnorm_vs_tactual']:.3f}, "
            f"mean(t_actual/t_est)={rr['mean_ratio']:.3f}, "
            f"mean_err={rr['mean_err_sec']:.3f}s"
        )
    lines.append("")
    lines.append("## 4) 反事实说明")
    lines.append(
        "- \"Local/V2V 真正赢 RSU\"需要同一子任务双动作反事实样本；当前在线轨迹不可同时观测，"
        "本次按要求标记为 unavailable，不做伪反事实推断。"
    )

    (out_dir / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")

    # compact compare table for quick check
    compare.to_csv(out_dir / "policy_compare_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit t_finish_norm fairness via white-box inference runs")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to MAPPO best checkpoint")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42, help="Base seed; episode seed = seed + ep_idx")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("audit_results") / f"tfinish_audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    m_agent = _load_mappo_agent(str(ckpt), device=args.device)

    env_m = VecOffloadingEnv(config=Cfg)
    env_g = VecOffloadingEnv(config=Cfg)

    try:
        print(f"[Audit] output dir: {out_dir}", flush=True)
        print("[Audit] running MAPPO...", flush=True)
        d_m, c_m, k_m = _run_policy(
            name="mappo",
            env=env_m,
            policy_obj=m_agent,
            is_mappo=True,
            episodes=int(args.episodes),
            base_seed=int(args.seed),
        )

        print("[Audit] running Greedy...", flush=True)
        g_policy = GreedyPolicy(env_g)
        d_g, c_g, k_g = _run_policy(
            name="greedy",
            env=env_g,
            policy_obj=g_policy,
            is_mappo=False,
            episodes=int(args.episodes),
            base_seed=int(args.seed),
        )
    finally:
        env_m.close()
        env_g.close()

    # save raw
    d_m.to_csv(out_dir / "decisions_mappo.csv", index=False)
    d_g.to_csv(out_dir / "decisions_greedy.csv", index=False)
    k_m.to_csv(out_dir / "candidates_mappo.csv", index=False)
    k_g.to_csv(out_dir / "candidates_greedy.csv", index=False)
    c_m.to_csv(out_dir / "completions_mappo.csv", index=False)
    c_g.to_csv(out_dir / "completions_greedy.csv", index=False)

    # join completion with selected t_finish_norm
    keys = ["policy", "episode", "seed", "vehicle_id", "subtask_id"]
    d_all = pd.concat([d_m, d_g], ignore_index=True)
    c_all = pd.concat([c_m, c_g], ignore_index=True)

    j_all = c_all.merge(
        d_all[[*keys, "selected_type", "t_finish_norm_sel", "min_t_finish_rsu", "min_t_finish_nonrsu", "comparable_nonrsu_vs_rsu", "nonrsu_beats_rsu_est"]],
        on=keys,
        how="left",
    )

    j_all[j_all["policy"] == "mappo"].to_csv(out_dir / "completed_joined_mappo.csv", index=False)
    j_all[j_all["policy"] == "greedy"].to_csv(out_dir / "completed_joined_greedy.csv", index=False)

    # stats
    per_type = pd.concat(
        [
            _per_type_stats(d_m, j_all[j_all["policy"] == "mappo"], "mappo"),
            _per_type_stats(d_g, j_all[j_all["policy"] == "greedy"], "greedy"),
        ],
        ignore_index=True,
    )
    per_type.to_csv(out_dir / "per_type_stats.csv", index=False)

    cand_stats = pd.concat(
        [
            _candidate_structure_stats(d_m, "mappo"),
            _candidate_structure_stats(d_g, "greedy"),
        ],
        ignore_index=True,
    )
    cand_stats.to_csv(out_dir / "candidate_structure_stats.csv", index=False)

    # compact compare
    comp_rows: List[Dict] = []
    for p in ["mappo", "greedy"]:
        d = d_all[d_all["policy"] == p]
        j = j_all[j_all["policy"] == p]
        comp_rows.append(
            {
                "policy": p,
                "on_task_decisions": int(len(d)),
                "completed_samples": int(len(j)),
                "task_success_rate_proxy": float(pd.to_numeric(j["t_actual"], errors="coerce").notna().mean()) if len(d) > 0 else float("nan"),
                "decision_frac_rsu": float((d["selected_type"] == "RSU").mean()) if len(d) > 0 else float("nan"),
                "decision_frac_local": float((d["selected_type"] == "Local").mean()) if len(d) > 0 else float("nan"),
                "decision_frac_v2v": float((d["selected_type"] == "V2V").mean()) if len(d) > 0 else float("nan"),
                "p_min_nonrsu_lt_rsu": float(cand_stats[cand_stats["policy"] == p]["p_min_nonrsu_lt_rsu_given_comparable"].iloc[0]),
                "choose_nonrsu_given_win": float(cand_stats[cand_stats["policy"] == p]["choose_nonrsu_given_win"].iloc[0]),
            }
        )
    compare = pd.DataFrame(comp_rows)

    _build_conclusion(out_dir=out_dir, per_type=per_type, cand_stats=cand_stats, compare=compare)

    print("[Audit] done")
    print(f"[Audit] decisions_mappo: {out_dir / 'decisions_mappo.csv'}")
    print(f"[Audit] decisions_greedy: {out_dir / 'decisions_greedy.csv'}")
    print(f"[Audit] per_type_stats: {out_dir / 'per_type_stats.csv'}")
    print(f"[Audit] candidate_structure_stats: {out_dir / 'candidate_structure_stats.csv'}")
    print(f"[Audit] conclusion: {out_dir / 'conclusion.md'}")


if __name__ == "__main__":
    main()
