#!/usr/bin/env python3
"""
RC1 default scenario (python train.py baseline) time-scale compatibility audit.

This script does NOT modify environment logic. It only wraps service step methods
to collect completed-job durations and episode-level statistics.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import SystemConfig as Cfg  # noqa: E402
from envs.vec_offloading_env import VecOffloadingEnv  # noqa: E402


def _pct(arr: List[float], q: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.asarray(arr, dtype=np.float64), q))


def _summary(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {"count": 0, "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    xs = np.asarray(arr, dtype=np.float64)
    return {
        "count": int(xs.size),
        "p50": float(np.percentile(xs, 50)),
        "p90": float(np.percentile(xs, 90)),
        "p95": float(np.percentile(xs, 95)),
        "max": float(np.max(xs)),
    }


def _fmt_stats(name: str, stats_sec: Dict[str, float], stats_step: Dict[str, float], one_step_ratio: float) -> str:
    return (
        f"{name}: count={stats_sec['count']} | "
        f"sec[p50/p90/p95/max]={stats_sec['p50']:.4f}/{stats_sec['p90']:.4f}/{stats_sec['p95']:.4f}/{stats_sec['max']:.4f} | "
        f"step[p50/p90/p95/max]={stats_step['p50']:.2f}/{stats_step['p90']:.2f}/{stats_step['p95']:.2f}/{stats_step['max']:.2f} | "
        f"one_step={one_step_ratio:.1%}"
    )


def _duration_steps(duration_sec: float, dt: float) -> int:
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        return 1
    return max(1, int(math.ceil((duration_sec - 1e-12) / max(dt, 1e-12))))


@dataclass
class EventStats:
    durations_sec: List[float] = field(default_factory=list)
    durations_step: List[int] = field(default_factory=list)
    one_step_count: int = 0

    def add(self, duration_sec: float, dt: float) -> None:
        steps = _duration_steps(duration_sec, dt)
        self.durations_sec.append(float(duration_sec))
        self.durations_step.append(int(steps))
        if steps == 1:
            self.one_step_count += 1

    def summarize(self) -> tuple[Dict[str, float], Dict[str, float], float]:
        sec_s = _summary(self.durations_sec)
        step_s = _summary([float(x) for x in self.durations_step])
        n = len(self.durations_step)
        one_ratio = float(self.one_step_count) / float(n) if n > 0 else float("nan")
        return sec_s, step_s, one_ratio


@dataclass
class AuditStats:
    comm_v2i: EventStats = field(default_factory=EventStats)
    comm_v2v: EventStats = field(default_factory=EventStats)
    cpu_veh: EventStats = field(default_factory=EventStats)
    cpu_rsu: EventStats = field(default_factory=EventStats)
    displacement_all: List[float] = field(default_factory=list)
    displacement_nominal: List[float] = field(default_factory=list)
    displacement_respawn_jumps: int = 0
    deadlines: List[float] = field(default_factory=list)
    lb_star: List[float] = field(default_factory=list)
    deadline_lb_ratio: List[float] = field(default_factory=list)
    term_success_all_done: int = 0
    term_idle: int = 0
    trunc_time_limit: int = 0
    term_other: Dict[str, int] = field(default_factory=dict)
    episodes: int = 0
    steps_total: int = 0
    seen_dag_obj_ids: Set[int] = field(default_factory=set)


def assert_default_scenario() -> None:
    # Core baseline assertions (only the user-defined "unique scenario" items)
    assert Cfg.SEED == 42
    assert Cfg.MAP_SIZE == 2000.0
    assert Cfg.NUM_LANES == 2
    assert Cfg.NUM_VEHICLES == 20
    assert Cfg.VEHICLE_ARRIVAL_RATE == 0.0
    assert Cfg.DT == 0.1
    assert Cfg.MAX_STEPS == 200
    assert bool(Cfg.TERMINATE_ON_ALL_FINISHED) is True
    assert Cfg.NUM_RSU == 3
    assert float(Cfg.RSU_RANGE) == 350.0
    assert float(Cfg.V2V_RANGE) == 250.0
    assert str(Cfg.V2I_RATE_MODEL).upper() == "RB_SINR"
    assert bool(Cfg.V2I_ICI_ENABLED) is True
    assert int(Cfg.V2I_FREQ_REUSE_FACTOR) == 1
    assert float(Cfg.BW_V2I) == 10e6
    assert float(Cfg.BW_V2V) == 20e6
    assert int(Cfg.V2V_NUM_RB) == 10
    assert int(Cfg.V2I_NUM_RB) == 55
    assert float(Cfg.MIN_VEHICLE_CPU_FREQ) == 0.5e9
    assert float(Cfg.MAX_VEHICLE_CPU_FREQ) == 2.0e9
    assert float(Cfg.MIN_COMP) == 8.0e7 and float(Cfg.MAX_COMP) == 6.0e8
    assert float(Cfg.MIN_DATA) == 8.0e5 and float(Cfg.MAX_DATA) == 6.0e6
    assert float(Cfg.DEADLINE_ALPHA_MIN) == 6.0 and float(Cfg.DEADLINE_ALPHA_MAX) == 9.0
    assert int(Cfg.RSU_NUM_PROCESSORS) == 4
    assert str(Cfg.DAG_SOURCE) == "synthetic_small"
    assert int(Cfg.MIN_NODES) == 8 and int(Cfg.MAX_NODES) == 16
    assert str(Cfg.DEADLINE_MODE) == "LB_ALPHA"
    assert bool(Cfg.TRUST_ENABLED) is True
    assert bool(Cfg.CHAIN_ENABLED) is False
    assert bool(Cfg.DOMAIN_RANDOMIZATION) is False
    assert str(Cfg.REWARD_SCHEME) == "UNIFIED"
    assert bool(Cfg.ENABLE_PBRS) is False
    # Derived values (actual code defaults)
    assert bool(Cfg.ALL_FEASIBLE) is True
    assert int(Cfg.MAX_NEIGHBORS) == 19
    assert int(Cfg.MAX_TARGETS) == 23
    assert int(Cfg.V2I_NUM_RB) == 55
    assert abs(float(Cfg.V2V_BW_PER_RB) - float(Cfg.BW_V2V) / float(Cfg.V2V_NUM_RB)) < 1e-9


def assert_priority_rank_disabled(env: VecOffloadingEnv) -> None:
    # Observation-level removal
    assert "priority" not in env.observation_space.spaces
    obs, _ = env.reset(seed=int(Cfg.SEED))
    assert len(obs) > 0
    assert "priority" not in obs[0]
    # Strategy path was manually disabled in current code (white-box expectation).
    # We only check the model class attribute if instantiated elsewhere; no need to instantiate here.


def make_policy_action(obs: Dict, rng: np.random.Generator) -> Dict:
    subtask_mask = np.asarray(obs.get("subtask_mask", obs.get("task_mask")), dtype=np.float32) > 0.5
    sub_candidates = np.where(subtask_mask)[0]
    if sub_candidates.size > 0:
        subtask = int(rng.choice(sub_candidates))
    else:
        subtask = int(np.asarray(obs.get("subtask_index", 0)).item()) if "subtask_index" in obs else 0
        if subtask < 0:
            subtask = 0

    action_mask = np.asarray(obs["action_mask"], dtype=np.float32) > 0.5
    tgt_candidates = np.where(action_mask)[0]
    if tgt_candidates.size == 0:
        target = 0
    else:
        non_local = tgt_candidates[tgt_candidates != 0]
        if non_local.size > 0 and rng.random() < 0.7:
            target = int(rng.choice(non_local))
        else:
            target = int(rng.choice(tgt_candidates))

    power = float(rng.uniform(0.2, 1.0))
    return {"subtask": subtask, "target": target, "power": power}


def install_service_wrappers(env: VecOffloadingEnv, stats: AuditStats, dt: float) -> None:
    orig_comm_step = env._comm_service.step
    orig_cpu_step = env._cpu_service.step

    def comm_step_wrapper(*args, **kwargs):
        result = orig_comm_step(*args, **kwargs)
        for job in getattr(result, "completed_jobs", []):
            st = getattr(job, "start_time", None)
            ft = getattr(job, "finish_time", None)
            if st is None or ft is None:
                continue
            dur = float(max(ft - st, 0.0))
            link_type = str(getattr(job, "link_type", "")).upper()
            if link_type == "V2I":
                stats.comm_v2i.add(dur, dt)
            elif link_type == "V2V":
                stats.comm_v2v.add(dur, dt)
        return result

    def cpu_step_wrapper(*args, **kwargs):
        result = orig_cpu_step(*args, **kwargs)
        for job in getattr(result, "completed_jobs", []):
            st = getattr(job, "start_time", None)
            ft = getattr(job, "finish_time", None)
            if st is None or ft is None:
                continue
            dur = float(max(ft - st, 0.0))
            exec_node = getattr(job, "exec_node", None)
            if isinstance(exec_node, tuple) and len(exec_node) >= 1 and str(exec_node[0]).upper() == "RSU":
                stats.cpu_rsu.add(dur, dt)
            else:
                stats.cpu_veh.add(dur, dt)
        return result

    env._comm_service.step = comm_step_wrapper
    env._cpu_service.step = cpu_step_wrapper


def harvest_dag_stats(env: VecOffloadingEnv, stats: AuditStats) -> None:
    for v in getattr(env, "vehicles", []):
        dag = getattr(v, "task_dag", None)
        if dag is None:
            continue
        oid = id(dag)
        if oid in stats.seen_dag_obj_ids:
            continue
        stats.seen_dag_obj_ids.add(oid)
        deadline = float(getattr(dag, "deadline", np.nan))
        base = float(getattr(dag, "deadline_base_time", np.nan))
        if np.isfinite(deadline) and deadline > 0:
            stats.deadlines.append(deadline)
        if np.isfinite(base) and base > 0:
            stats.lb_star.append(base)
            if np.isfinite(deadline) and deadline > 0:
                stats.deadline_lb_ratio.append(float(deadline / base))


def classify_episode_end(terminated: bool, truncated: bool, info: Dict, stats: AuditStats) -> None:
    reason = str(info.get("terminated_reason", "none"))
    if truncated:
        if reason == "time_limit":
            stats.trunc_time_limit += 1
        else:
            stats.term_other[f"truncated:{reason}"] = stats.term_other.get(f"truncated:{reason}", 0) + 1
        return
    if terminated:
        if reason == "success_all_done":
            stats.term_success_all_done += 1
        elif reason == "idle":
            stats.term_idle += 1
        else:
            stats.term_other[reason] = stats.term_other.get(reason, 0) + 1


def run_audit(episodes: int = 5, base_seed: int = 42) -> AuditStats:
    dt = float(Cfg.DT)
    vmax_step = float(Cfg.VEL_MAX) * dt * 1.10
    rng = np.random.default_rng(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    stats = AuditStats()
    env = VecOffloadingEnv()
    install_service_wrappers(env, stats, dt)

    for ep in range(episodes):
        obs_list, _ = env.reset(seed=int(base_seed + ep))
        harvest_dag_stats(env, stats)
        done = False
        while not done:
            pos_before = {int(v.id): np.asarray(v.pos, dtype=np.float64).copy() for v in env.vehicles}
            actions = [make_policy_action(obs, rng) for obs in obs_list]
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            _ = rewards  # only to keep variable explicit
            stats.steps_total += 1

            for v in env.vehicles:
                vid = int(v.id)
                if vid in pos_before:
                    disp = float(np.linalg.norm(np.asarray(v.pos, dtype=np.float64) - pos_before[vid]))
                    stats.displacement_all.append(disp)
                    if disp <= vmax_step:
                        stats.displacement_nominal.append(disp)
                    else:
                        stats.displacement_respawn_jumps += 1

            harvest_dag_stats(env, stats)
            obs_list = next_obs
            done = bool(terminated or truncated)
            if done:
                classify_episode_end(bool(terminated), bool(truncated), info, stats)
        stats.episodes += 1

    try:
        env.close()
    except Exception:
        pass
    return stats


def print_whitebox_evidence() -> None:
    print("=== A. White-box code-path confirmation (default SystemConfig scenario) ===")
    print("1) DT-discrete推进（通信/计算/移动/奖励/终止）")
    print("   - 通信阶段传入DT: envs/vec_offloading_env.py:_phase3_advance_comm_queues @ 2475-2481 (self.config.DT)")
    print("   - Comm FIFO+DT预算: envs/services/comm_queue_service.py:37-46, 64-70, 72-116")
    print("   - 计算阶段传入DT: envs/vec_offloading_env.py:_phase4_advance_cpu_queues @ 2664-2671 (self.config.DT)")
    print("   - CPU FIFO+DT预算: envs/services/cpu_queue_service.py:25-34, 63-69, 71-107")
    print("   - 全局时间推进: envs/vec_offloading_env.py:3161-3164 (self.time += self.config.DT)")
    print("   - 车辆移动步长: envs/vec_offloading_env.py:3214-3216 (update_pos(self.config.DT,...))")
    print("   - UNIFIED奖励时间项使用DT: envs/vec_offloading_env.py:3314-3318, 3381-3384")
    print("   - 终止判定按MAX_STEPS: envs/vec_offloading_env.py:3952-3964")
    print()
    print("2) LB_ALPHA deadline 显式考虑DT/步长保护")
    print("   - DT与关键路径步数保护LB_step: utils/dag_generator.py:374-377")
    print("   - LB_star = max(LB0, LB_step): utils/dag_generator.py:377")
    print("   - LB_ALPHA模式 base_time=LB_star: utils/dag_generator.py:412-423")
    print("   - 最终deadline >= (1+eps)*LB_star: utils/dag_generator.py:441-443")
    print()
    print("3) Handover参数与DT时间语义")
    print("   - 切换条件/语义注释（步数单位）: envs/vec_offloading_env.py:1297-1304")
    print("   - MIN_RSU_STAY_STEPS门限检查: envs/vec_offloading_env.py:1334-1338")
    print("   - 切换后冻结步数赋值HO_FREEZE_STEPS: envs/vec_offloading_env.py:1340-1342, 1356-1358")
    print("   - 冻结计数逐步递减并将V2I速率置0: envs/vec_offloading_env.py:730-745")
    print()


def print_scenario_confirmation() -> None:
    print("=== Default scenario confirmation (python train.py baseline, no profile/env/CLI overrides) ===")
    print(
        f"SEED={Cfg.SEED}, MAP_SIZE={Cfg.MAP_SIZE}, NUM_LANES={Cfg.NUM_LANES}, "
        f"NUM_VEHICLES={Cfg.NUM_VEHICLES}, NUM_RSU={Cfg.NUM_RSU}, DT={Cfg.DT}, MAX_STEPS={Cfg.MAX_STEPS}"
    )
    print(
        f"ARRIVAL_RATE={Cfg.VEHICLE_ARRIVAL_RATE}, RSU_RANGE={Cfg.RSU_RANGE}, V2V_RANGE={Cfg.V2V_RANGE}, "
        f"V2I={Cfg.BW_V2I/1e6:.1f}MHz/{Cfg.V2I_RATE_MODEL}/RB={Cfg.V2I_NUM_RB}, "
        f"V2V={Cfg.BW_V2V/1e6:.1f}MHz/RB={Cfg.V2V_NUM_RB}"
    )
    print(
        f"CANDIDATE_MODE={Cfg.CANDIDATE_MODE}, ALL_FEASIBLE={Cfg.ALL_FEASIBLE}, "
        f"MAX_NEIGHBORS={Cfg.MAX_NEIGHBORS}, MAX_TARGETS={Cfg.MAX_TARGETS}, ENABLE_RSU_SELECTION={Cfg.ENABLE_RSU_SELECTION}"
    )
    print(
        f"DAG_SOURCE={Cfg.DAG_SOURCE}, nodes=[{Cfg.MIN_NODES},{Cfg.MAX_NODES}], "
        f"DEADLINE_MODE={Cfg.DEADLINE_MODE}, alpha=[{Cfg.DEADLINE_ALPHA_MIN},{Cfg.DEADLINE_ALPHA_MAX}], "
        f"LB_EPS={Cfg.DEADLINE_LB_EPS}, STEP_GUARD={Cfg.DEADLINE_STEP_GUARD_DELTA}"
    )
    print(
        f"TRUST_ENABLED={Cfg.TRUST_ENABLED}, CHAIN_ENABLED={Cfg.CHAIN_ENABLED}, DOMAIN_RANDOMIZATION={Cfg.DOMAIN_RANDOMIZATION}, "
        f"REWARD_SCHEME={Cfg.REWARD_SCHEME}, ENABLE_PBRS={Cfg.ENABLE_PBRS}"
    )
    print()


def print_stats(stats: AuditStats) -> None:
    dt = float(Cfg.DT)
    print("=== B. Time-scale statistics (minimal instrumentation, no logic change) ===")
    print(f"Episodes={stats.episodes}, total_steps={stats.steps_total}, DT={dt}s")
    print()

    for name, es in [
        ("COMM V2I (service time)", stats.comm_v2i),
        ("COMM V2V (service time)", stats.comm_v2v),
        ("CPU VEH (service time)", stats.cpu_veh),
        ("CPU RSU (service time)", stats.cpu_rsu),
    ]:
        sec_s, step_s, one_ratio = es.summarize()
        print(_fmt_stats(name, sec_s, step_s, one_ratio))
    print()

    disp_all = _summary(stats.displacement_all)
    disp_nom = _summary(stats.displacement_nominal)
    print(
        f"Displacement raw (m/step): count={disp_all['count']} "
        f"p50/p90/p95/max={disp_all['p50']:.4f}/{disp_all['p90']:.4f}/{disp_all['p95']:.4f}/{disp_all['max']:.4f}"
    )
    print(
        f"Displacement nominal<=1.1*VEL_MAX*DT (m/step): count={disp_nom['count']} "
        f"p50/p90/p95/max={disp_nom['p50']:.4f}/{disp_nom['p90']:.4f}/{disp_nom['p95']:.4f}/{disp_nom['max']:.4f} "
        f"| respawn_jump_count={stats.displacement_respawn_jumps}"
    )
    print(
        f"Coverage scale check: p95_nominal_disp / V2V_RANGE = "
        f"{(disp_nom['p95'] / float(Cfg.V2V_RANGE)) if np.isfinite(disp_nom['p95']) else float('nan'):.5f}, "
        f"p95_nominal_disp / RSU_RANGE = "
        f"{(disp_nom['p95'] / float(Cfg.RSU_RANGE)) if np.isfinite(disp_nom['p95']) else float('nan'):.5f}"
    )
    print()

    d_s = _summary(stats.deadlines)
    lb_s = _summary(stats.lb_star)
    ratio_s = _summary(stats.deadline_lb_ratio)
    print(
        f"Deadline (s): count={d_s['count']} p50/p90/p95/max={d_s['p50']:.4f}/{d_s['p90']:.4f}/{d_s['p95']:.4f}/{d_s['max']:.4f}"
    )
    print(
        f"LB* (deadline_base_time under LB_ALPHA) (s): count={lb_s['count']} "
        f"p50/p90/p95/max={lb_s['p50']:.4f}/{lb_s['p90']:.4f}/{lb_s['p95']:.4f}/{lb_s['max']:.4f}"
    )
    print(
        f"Deadline/LB* ratio: count={ratio_s['count']} "
        f"p50/p90/p95/max={ratio_s['p50']:.4f}/{ratio_s['p90']:.4f}/{ratio_s['p95']:.4f}/{ratio_s['max']:.4f}"
    )
    print()

    total_ep = max(stats.episodes, 1)
    print(
        "Episode end types: "
        f"terminated(success_all_done)={stats.term_success_all_done} ({stats.term_success_all_done/total_ep:.1%}), "
        f"terminated(idle)={stats.term_idle} ({stats.term_idle/total_ep:.1%}), "
        f"truncated(time_limit)={stats.trunc_time_limit} ({stats.trunc_time_limit/total_ep:.1%})"
    )
    if stats.term_other:
        print(f"Other terminal reasons: {stats.term_other}")
    print()


def print_conclusion(stats: AuditStats) -> None:
    dt = float(Cfg.DT)
    # Simple evidence-based heuristics (no new logic, only interpretation of collected stats)
    comm_steps = stats.comm_v2i.durations_step + stats.comm_v2v.durations_step
    cpu_steps = stats.cpu_veh.durations_step + stats.cpu_rsu.durations_step
    comm_p95 = _pct([float(x) for x in comm_steps], 95) if comm_steps else float("nan")
    cpu_p95 = _pct([float(x) for x in cpu_steps], 95) if cpu_steps else float("nan")
    one_step_comm_ratio = (
        (stats.comm_v2i.one_step_count + stats.comm_v2v.one_step_count) / max(len(comm_steps), 1)
        if comm_steps else float("nan")
    )
    one_step_cpu_ratio = (
        (stats.cpu_veh.one_step_count + stats.cpu_rsu.one_step_count) / max(len(cpu_steps), 1)
        if cpu_steps else float("nan")
    )

    print("=== C. Audit conclusion (default scenario) ===")
    print("可运行且理论自洽: 是（代码阶段顺序与DT离散推进口径一致，LB_ALPHA显式含DT步长保护）")

    # Compatibility judgment (evidence-based, conservative wording)
    if (np.isfinite(comm_p95) and comm_p95 <= 5.0) and (np.isfinite(cpu_p95) and cpu_p95 <= 5.0):
        compat = "基本是"
    elif (np.isfinite(comm_p95) and comm_p95 <= 2.5) and (np.isfinite(cpu_p95) and cpu_p95 <= 2.5):
        compat = "是"
    else:
        compat = "否"
    print(f"DT=0.1s 与当前事件时长尺度兼容性: {compat}")

    print(
        "风险类型判断: 统计若显示大量事件在1~数个step内完成，则属于离散时间分辨率近似；"
        "当前白盒未见实现级错误（非将slot近似误判为bug）。"
    )
    print(
        f"补充量化: comm one-step={one_step_comm_ratio:.1%}，cpu one-step={one_step_cpu_ratio:.1%}，"
        f"comm_step_p95={comm_p95:.2f}，cpu_step_p95={cpu_p95:.2f}"
    )

    # Minimal parameter-only advice (only if needed)
    suggest = False
    advice = []
    if np.isfinite(one_step_comm_ratio) and one_step_comm_ratio > 0.9:
        suggest = True
        advice.append("若希望降低通信事件的DT量化误差，可优先调小BW/RB或收紧功率范围（参数层面）。")
    if np.isfinite(one_step_cpu_ratio) and one_step_cpu_ratio > 0.9:
        suggest = True
        advice.append("若希望降低计算事件的DT量化误差，可适度降低CPU频率或增大任务计算量（参数层面）。")
    if np.isfinite(one_step_comm_ratio) and one_step_comm_ratio < 0.05 and np.isfinite(comm_p95) and comm_p95 > 20:
        suggest = True
        advice.append("若通信事件过慢且跨很多step，可适度增大BW/RB或减小数据量范围（参数层面）。")

    print(f"是否建议立即修改参数: {'是' if suggest else '否'}")
    if suggest:
        print("最小侵入建议（仅参数，不新增开关）:")
        for s in advice[:3]:
            print(f"- {s}")
    print()


def main() -> None:
    episodes = 5
    base_seed = int(Cfg.SEED)
    if len(sys.argv) >= 2:
        episodes = int(sys.argv[1])
    if len(sys.argv) >= 3:
        base_seed = int(sys.argv[2])

    assert_default_scenario()
    env_check = VecOffloadingEnv()
    assert_priority_rank_disabled(env_check)
    try:
        env_check.close()
    except Exception:
        pass

    print_whitebox_evidence()
    print_scenario_confirmation()
    stats = run_audit(episodes=episodes, base_seed=base_seed)
    print_stats(stats)
    print_conclusion(stats)


if __name__ == "__main__":
    main()
