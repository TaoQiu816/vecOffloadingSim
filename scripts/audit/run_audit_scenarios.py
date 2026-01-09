"""
系统核验脚手架：最小四个集成场景的运行、日志与不变量断言。

场景：
1) 单车Local
2) 单车RSU
3) 双车V2V
4) 强依赖EDGE阻塞

特点：
- 不改业务逻辑，只通过配置覆盖/动作序列/自定义DAG注入
- 每step打印关键队列长度、队头job剩余量、已完成job数、DAG状态
- 不变量断言（10条）

运行示例：
    python scripts/audit/run_audit_scenarios.py
"""

from __future__ import annotations

import dataclasses
import functools
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.append(".")

from configs.config import SystemConfig as Cfg  # noqa: E402
from envs.vec_offloading_env import VecOffloadingEnv  # noqa: E402
from envs.entities.task_dag import DAGTask  # noqa: E402
from envs.jobs import TransferJob, ComputeJob  # noqa: E402


# --------------------------------------------------------------------------- #
# 配置覆写辅助
# --------------------------------------------------------------------------- #


def make_config(**overrides):
    """创建一个临时配置类，基于当前Cfg克隆并覆写字段。"""

    class TempCfg(Cfg):  # type: ignore[misc, valid-type]
        pass

    for k, v in overrides.items():
        setattr(TempCfg, k, v)
    return TempCfg


# --------------------------------------------------------------------------- #
# DAG 构造辅助
# --------------------------------------------------------------------------- #


def build_dag(
    num_nodes: int,
    edges: List[Tuple[int, int, float]],
    comp: List[float],
    input_data: List[float],
    deadline: float = 5.0,
) -> DAGTask:
    """构造一个确定性的 DAGTask 对象。

    edges: [(u, v, data_bits)], data_matrix[u, v] = data_bits
    comp:  每个节点的计算量 (cycles)
    input_data: 每个节点的输入数据量 (bits)
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    data_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    for u, v, data_bits in edges:
        adj[u, v] = 1
        data_matrix[u, v] = data_bits

    profiles = []
    for i in range(num_nodes):
        profiles.append({"comp": comp[i], "input_data": input_data[i]})

    dag = DAGTask(task_id=0, adj=adj, profiles=profiles, data_matrix=data_matrix, deadline=deadline)
    dag.start_time = 0.0
    return dag


def reset_dag_fields(dag: DAGTask):
    """显式重置 DAG 状态，确保存在 READY 子任务。"""
    n = dag.num_subtasks
    dag.status = np.zeros(n, dtype=int)
    dag.status[:] = 1  # READY
    dag.exec_locations = [None] * n
    dag.task_locations = [None] * n
    dag.in_degree = np.sum(dag.adj, axis=0)
    dag.inter_task_transfers = {}
    dag.waiting_for_data = [False] * n
    dag.rem_comp = dag.total_comp.copy()
    dag.rem_data = dag.total_data.copy()
    dag._is_failed = False
    dag.fail_reason = None
    if hasattr(dag, "priority") and dag.priority is None:
        dag.priority = dag._compute_priority_internal()


def diagnose_rsu_queue(env: VecOffloadingEnv, rsu_id: int, task_comp: float):
    """打印 _is_rsu_queue_full 内部关键变量。"""
    rsu = env.rsus[rsu_id]
    proc_dict = env.rsu_cpu_q[rsu_id]
    per_proc_limit = env.config.RSU_QUEUE_CYCLES_LIMIT / rsu.num_processors
    loads = {pid: sum(job.rem_cycles for job in q) for pid, q in proc_dict.items()}
    bools = []
    for pid, load in loads.items():
        ok = (load + task_comp) <= per_proc_limit
        bools.append(ok)
    print(f"[diag/rsu_queue] rsu_id={rsu_id} task_comp={task_comp:.2e} per_proc_limit={per_proc_limit:.2e} loads={loads} any_ok={any(bools)} return={not any(bools)}")


# --------------------------------------------------------------------------- #
# Recorder & Monkeypatch
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class StepRecord:
    comm_completed: List[TransferJob]
    cpu_completed: List[ComputeJob]
    comm_energy_cost: Dict[int, float]
    cpu_energy_cost: Dict[int, float]


def attach_recorders(env: VecOffloadingEnv):
    """为 comm/cpu service 添加record钩子，不改业务逻辑。"""
    recorder = StepRecord(comm_completed=[], cpu_completed=[], comm_energy_cost={}, cpu_energy_cost={})

    orig_comm_step = env._comm_service.step
    orig_cpu_step = env._cpu_service.step
    orig_on_transfer_done = env._dag_handler.on_transfer_done
    orig_on_compute_done = env._dag_handler.on_compute_done

    @functools.wraps(orig_comm_step)
    def comm_step_with_record(*args, **kwargs):
        result = orig_comm_step(*args, **kwargs)
        recorder.comm_completed = list(result.completed_jobs)
        recorder.comm_energy_cost = dict(result.energy_delta_cost)
        return result

    @functools.wraps(orig_cpu_step)
    def cpu_step_with_record(*args, **kwargs):
        result = orig_cpu_step(*args, **kwargs)
        recorder.cpu_completed = list(result.completed_jobs)
        recorder.cpu_energy_cost = dict(result.energy_delta_cost_local)
        return result

    @functools.wraps(orig_on_transfer_done)
    def on_transfer_done_with_log(job, *args, **kwargs):
        print(f"[hook/on_transfer_done] owner={job.owner_vehicle_id} subtask={job.subtask_id} kind={job.kind} dst={job.dst_node}")
        return orig_on_transfer_done(job, *args, **kwargs)

    @functools.wraps(orig_on_compute_done)
    def on_compute_done_with_log(job, *args, **kwargs):
        print(f"[hook/on_compute_done] owner={job.owner_vehicle_id} subtask={job.subtask_id} exec_node={job.exec_node}")
        return orig_on_compute_done(job, *args, **kwargs)

    env._comm_service.step = comm_step_with_record  # type: ignore[assignment]
    env._cpu_service.step = cpu_step_with_record  # type: ignore[assignment]
    env._dag_handler.on_transfer_done = on_transfer_done_with_log  # type: ignore[assignment]
    env._dag_handler.on_compute_done = on_compute_done_with_log  # type: ignore[assignment]
    return recorder


# --------------------------------------------------------------------------- #
# 不变量断言
# --------------------------------------------------------------------------- #


def assert_invariants(env: VecOffloadingEnv, recorder: StepRecord, prev_status: Dict[int, np.ndarray]):
    # 1. rem_bits 非负
    for q in list(env.txq_v2i.values()) + list(env.txq_v2v.values()):
        for job in q:
            assert job.rem_bytes >= -1e-9, f"rem_bits negative: {job}"
    # 2. rem_cycles 非负
    for q in list(env.veh_cpu_q.values()):
        for job in q:
            assert job.rem_cycles >= -1e-9, f"rem_cycles negative: {job}"
    for proc_dict in env.rsu_cpu_q.values():
        for q in proc_dict.values():
            for job in q:
                assert job.rem_cycles >= -1e-9, f"rem_cycles negative: {job}"
    # 3. status 单调不降
    for v in env.vehicles:
        prev = prev_status.get(v.id)
        if prev is not None:
            assert np.all(prev <= v.task_dag.status), f"status decreased veh {v.id}"
        prev_status[v.id] = v.task_dag.status.copy()
    # 4. exec_locations 一旦写入不可变
    for v in env.vehicles:
        if not hasattr(v.task_dag, "_exec_prev"):
            v.task_dag._exec_prev = [None] * v.task_dag.num_subtasks  # type: ignore[attr-defined]
        for i, loc in enumerate(v.task_dag.exec_locations):
            if v.task_dag._exec_prev[i] is None and loc is not None:
                v.task_dag._exec_prev[i] = loc
            if v.task_dag._exec_prev[i] is not None:
                assert v.task_dag._exec_prev[i] == loc, f"exec_locations changed veh {v.id} task {i}"
    # 5. COMPLETED位置一致
    for v in env.vehicles:
        for i, st in enumerate(v.task_dag.status):
            if st == 3:
                assert v.task_dag.task_locations[i] == v.task_dag.exec_locations[i], "task_locations mismatch"
    # 6. in_degree 非负
    for v in env.vehicles:
        assert np.all(v.task_dag.in_degree >= 0), "in_degree negative"
    # 7. mask[0]=True (Local可用)
    obs_list, _ = env._get_obs(), {}
    for obs in obs_list:
        assert bool(obs["target_mask"][0]) is True, "Local mask false"
    # 8. 时间单调
    if not hasattr(env, "_last_time_check"):
        env._last_time_check = -1.0  # type: ignore[attr-defined]
    assert env.time >= env._last_time_check, "time not monotonic"
    env._last_time_check = env.time  # type: ignore[attr-defined]
    # 9. 功率范围
    for v in env.vehicles:
        assert Cfg.TX_POWER_MIN_DBM - 1e-6 <= v.tx_power_dbm <= Cfg.TX_POWER_MAX_DBM + 1e-6, "tx_power out of range"
    # 10. 完成job rem≈0
    for job in recorder.comm_completed:
        assert abs(job.rem_bytes) < 1e-6, f"completed comm job rem!=0 {job}"
    for job in recorder.cpu_completed:
        assert abs(job.rem_cycles) < 1e-6, f"completed cpu job rem!=0 {job}"


# --------------------------------------------------------------------------- #
# 日志打印
# --------------------------------------------------------------------------- #


def log_step(env: VecOffloadingEnv, recorder: StepRecord, rewards: List[float]):
    tx_v2i_len = sum(len(q) for q in env.txq_v2i.values())
    tx_v2v_len = sum(len(q) for q in env.txq_v2v.values())
    cpu_veh_len = sum(len(q) for q in env.veh_cpu_q.values())
    cpu_rsu_len = sum(len(q) for pd in env.rsu_cpu_q.values() for q in pd.values())

    def head_info(queue):
        if not queue:
            return None
        job = queue[0]
        return {"kind": job.kind, "rem": getattr(job, "rem_bytes", None), "subtask": job.subtask_id}

    print(
        f"[t={env.time:.2f}] tx_v2i={tx_v2i_len} tx_v2v={tx_v2v_len} cpu_veh={cpu_veh_len} cpu_rsu={cpu_rsu_len} "
        f"comm_done={len(recorder.comm_completed)} cpu_done={len(recorder.cpu_completed)} "
        f"rew_mean={np.mean(rewards) if rewards else 0:.3f}"
    )
    if env.txq_v2i:
        first_key = next(iter(env.txq_v2i))
        print(f"  txq_v2i head {first_key}: {head_info(env.txq_v2i[first_key])}")
    if env.txq_v2v:
        first_key = next(iter(env.txq_v2v))
        print(f"  txq_v2v head {first_key}: {head_info(env.txq_v2v[first_key])}")
    if env.vehicles:
        v0 = env.vehicles[0]
        print(f"  DAG status veh0: {v0.task_dag.status}, exec={v0.task_dag.exec_locations}, loc={v0.task_dag.task_locations}")
    # 打印 completed_jobs 详情
    if recorder.comm_completed:
        print("  comm_completed:", [(j.kind, j.dst_node, getattr(j, 'rem_bytes', None)) for j in recorder.comm_completed])
    if recorder.cpu_completed:
        print("  cpu_completed:", [(j.exec_node, getattr(j, 'rem_cycles', None)) for j in recorder.cpu_completed])


def log_step_rsu_detail(env: VecOffloadingEnv, obs_list, actions):
    """单车RSU场景的细粒度证据输出。"""
    v = env.vehicles[0]
    obs = obs_list[0] if obs_list else {}
    sub_idx = obs.get("subtask_index", None)
    tgt_mask = obs.get("target_mask", None)
    mask_local = tgt_mask[0] if tgt_mask is not None else None
    mask_rsu = tgt_mask[1] if (tgt_mask is not None and len(tgt_mask) > 1) else None
    act = actions[0] if actions else {}
    plans = env._plan_actions_snapshot(actions)
    plan0 = plans[0] if plans else {}
    desired = plan0.get("desired_target")
    planned = plan0.get("planned_target")
    illegal_reason = plan0.get("illegal_reason")
    rsu_pos = env.rsus[0].position if env.rsus else np.array([0.0, 0.0])
    rsu_range = env.config.RSU_RANGE
    dist = float(np.linalg.norm(v.pos - rsu_pos))
    speed = float(np.linalg.norm(v.vel))
    contact = (rsu_range - dist) / speed if speed > 1e-6 else env._max_rsu_contact_time if hasattr(env, "_max_rsu_contact_time") else float("inf")
    print(
        f"  [RSU detail] subtask_index={sub_idx} action={act} mask(Local,RSU)=({mask_local},{mask_rsu}) "
        f"desired={desired} planned={planned} illegal={illegal_reason} "
        f"rsu_dist={dist:.2f} rsu_range={rsu_range:.2f} contact_time={contact:.2f}"
    )
    exec_locs = v.task_dag.exec_locations if hasattr(v.task_dag, "exec_locations") else None
    print(f"  [RSU detail] exec_locations={exec_locs}")
    # 检查INPUT是否入队
    src_node = ("VEH", v.id)
    txq_len = len(env.txq_v2i.get(src_node, []))
    print(f"  [RSU detail] txq_v2i[{src_node}] len={txq_len}")
    # 打印CommWait
    cw = env._compute_comm_wait(v.id)
    print(f"  [RSU detail] CommWait total_v2i={cw['total_v2i']:.6f} edge_v2i={cw['edge_v2i']:.6f} total_v2v={cw['total_v2v']:.6f}")


def log_obs_brief(tag: str, obs):
    tgt_mask = obs.get("target_mask") if obs else None
    task_mask = obs.get("task_mask") if obs else None
    sub_idx = obs.get("subtask_index") if obs else None
    print(f"[{tag}] subtask_index={sub_idx} target_mask={tgt_mask} task_mask={task_mask}")


# --------------------------------------------------------------------------- #
# 场景定义
# --------------------------------------------------------------------------- #


def scenario_single_local() -> Tuple[VecOffloadingEnv, List[List[Dict]]]:
    cfg = make_config(
        NUM_VEHICLES=1,
        NUM_RSU=1,
        MAX_STEPS=5,
        VEHICLE_ARRIVAL_RATE=0,
        VEL_MEAN=0.0,
        VEL_MAX=0.0,
        BW_V2I=20e6,
    )
    env = VecOffloadingEnv(config=cfg)
    env.reset(seed=42)
    # 构造简单DAG：单节点
    dag = build_dag(num_nodes=1, edges=[], comp=[1e6], input_data=[1.2e6], deadline=5.0)
    env.vehicles[0].task_dag = dag
    # 所有动作选 Local target=0
    actions = [[{"target": 0, "power": 0.5}] for _ in range(5)]
    return env, actions


def scenario_single_rsu() -> Tuple[VecOffloadingEnv, List[List[Dict]]]:
    cfg = make_config(
        NUM_VEHICLES=1,
        NUM_RSU=1,
        MAX_STEPS=8,
        VEHICLE_ARRIVAL_RATE=0,
        VEL_MEAN=0.0,
        VEL_MAX=0.0,
        BW_V2I=20e6,
    )
    env = VecOffloadingEnv(config=cfg)
    # 开启审计日志
    setattr(env.config, "AUDIT_LOG_RSU_SELECTOR", True)
    env.reset(seed=43)
    # 强制车辆处于RSU覆盖内，速度为0，距离RSU 50m
    if env.rsus:
        rsu_pos = env.rsus[0].position
        env.vehicles[0].pos = rsu_pos + np.array([50.0, 0.0])
    env.vehicles[0].vel = np.array([0.0, 0.0])
    # 设置足够大的数据/计算量，确保跨步观察
    dag = build_dag(num_nodes=1, edges=[], comp=[5.0e9], input_data=[5e7], deadline=20.0)
    reset_dag_fields(dag)
    # 关键断言：确保READY且字段正确
    assert dag.status[0] == 1, f"status not READY: {dag.status}"
    assert dag.exec_locations[0] is None, f"exec_locations not None: {dag.exec_locations}"
    assert dag.task_locations[0] is None, f"task_locations not None: {dag.task_locations}"
    assert dag.in_degree[0] == 0, f"in_degree not 0: {dag.in_degree}"
    assert dag.rem_comp[0] >= 2.0e8, f"rem_comp too small: {dag.rem_comp}"
    assert dag.rem_data[0] > 0, f"rem_data not positive: {dag.rem_data}"
    env.vehicles[0].task_dag = dag
    # 触发一次观测以填充 _last_rsu_choice/mask，并打印诊断
    obs = env._get_obs()
    if obs:
        o0 = obs[0]
        print(f"[diag/reset_obs] subtask_index={o0.get('subtask_index')} target_mask={o0.get('target_mask')} status={dag.status}")
        if o0.get("target_mask", [False, False])[1] is False:
            rsu_choice = env._select_best_rsu(env.vehicles[0], dag.total_comp[0], dag.total_data[0])
            print(f"[diag] initial rsu_choice={rsu_choice}")
            # 诊断输出 _last_rsu_choice 当前值
            print(f"[diag] _last_rsu_choice={getattr(env, '_last_rsu_choice', {})}")
            # 打印车辆/RSU距离等
            rsu_pos = env.rsus[0].position if env.rsus else np.array([0.0, 0.0])
            dist = float(np.linalg.norm(env.vehicles[0].pos - rsu_pos))
            print(f"[diag] rsu_dist={dist:.2f} rsu_range={env.config.RSU_RANGE:.2f}")
        else:
            print(f"[diag] target_mask RSU already True, _last_rsu_choice={getattr(env, '_last_rsu_choice', {})}")
    # 手动调用一次 _select_best_rsu 以获取完整诊断
    rsu_choice_diag = env._select_best_rsu(env.vehicles[0], dag.total_comp[0], dag.total_data[0])
    print(f"[diag/manual_select] rsu_choice={rsu_choice_diag} _last_rsu_choice={getattr(env, '_last_rsu_choice', {})}")
    diagnose_rsu_queue(env, 0, dag.total_comp[0])
    actions = [[{"target": 1, "power": 1.0}] for _ in range(8)]  # target=RSU
    # 断言确保存在READY任务且RSU可用（在覆盖内）
    assert env.vehicles[0].task_dag.get_top_priority_task() is not None, "No READY subtask"
    return env, actions


def scenario_double_v2v() -> Tuple[VecOffloadingEnv, List[List[Dict]]]:
    cfg = make_config(
        NUM_VEHICLES=2,
        NUM_RSU=0,
        MAX_STEPS=10,
        VEHICLE_ARRIVAL_RATE=0,
        V2V_RANGE=300.0,
        VEL_MEAN=0.0,
        VEL_MAX=0.0,
        BW_V2I=20e6,
    )
    env = VecOffloadingEnv(config=cfg)
    env.reset(seed=44)
    # 固定位置，确保在V2V范围内
    env.vehicles[0].pos = np.array([0.0, 0.0])
    env.vehicles[1].pos = np.array([100.0, 0.0])
    dag0 = build_dag(num_nodes=1, edges=[], comp=[1e6], input_data=[1.5e6], deadline=5.0)
    env.vehicles[0].task_dag = dag0
    # 车1无任务，但存在以接收
    dag1 = build_dag(num_nodes=1, edges=[], comp=[1e3], input_data=[0.0], deadline=1.0)
    dag1.status[0] = 3  # 标记为已完成，避免调度
    dag1.exec_locations[0] = "Local"
    dag1.task_locations[0] = "Local"
    env.vehicles[1].task_dag = dag1
    # 车0卸载到车1：target=2 (neighbor 0索引)
    actions = [[{"target": 2, "power": 1.0}, {"target": 0, "power": 0.0}] for _ in range(10)]
    return env, actions


def scenario_edge_blocking() -> Tuple[VecOffloadingEnv, List[List[Dict]]]:
    cfg = make_config(
        NUM_VEHICLES=1,
        NUM_RSU=1,
        MAX_STEPS=12,
        VEHICLE_ARRIVAL_RATE=0,
        VEL_MEAN=0.0,
        VEL_MAX=0.0,
        BW_V2I=20e6,
    )
    env = VecOffloadingEnv(config=cfg)
    env.reset(seed=45)
    # DAG: 0 -> 1 -> 2; 边(1->2)有数据，制造 EDGE 传输
    edges = [(0, 1, 5e5), (1, 2, 8e5)]
    dag = build_dag(num_nodes=3, edges=edges, comp=[5e5, 8e5, 5e5], input_data=[1e6, 0.0, 0.0], deadline=8.0)
    env.vehicles[0].task_dag = dag
    # 计划: 0 Local, 1 RSU, 2 Local
    actions = [
        [{"target": 0, "power": 0.5}],  # step0: assign task0 Local
        [{"target": 1, "power": 1.0}],  # step1: assign task1 RSU (after task0 done)
        [{"target": 0, "power": 0.5}],  # step2+: keep Local for task2
    ]
    actions = actions + [actions[-1]] * 9  # pad to MAX_STEPS
    return env, actions


def scenario_commwait_stress() -> Tuple[VecOffloadingEnv, List[List[Dict]]]:
    """
    压测场景：强制所有车辆卸载到RSU，数据量>步容量以产生队列积压。
    """
    cfg = make_config(
        NUM_VEHICLES=6,
        NUM_RSU=1,
        MAX_STEPS=30,
        VEHICLE_ARRIVAL_RATE=0,
        VEL_MEAN=0.0,
        VEL_MAX=0.0,
        BW_V2I=20e6,
    )
    env = VecOffloadingEnv(config=cfg)
    env.reset(seed=50)
    # 固定所有车在RSU覆盖内且静止
    if env.rsus:
        rsu_pos = env.rsus[0].position
        for v in env.vehicles:
            v.pos = rsu_pos + np.array([50.0, 0.0])
            v.vel = np.array([0.0, 0.0])
    # 动态计算单步容量，构造至少5倍容量的数据量以产生稳定积压
    sample_v = env.vehicles[0]
    rate = env.channel.compute_one_rate(
        sample_v, env.rsus[0].position, "V2I", curr_time=env.time,
        v2i_user_count=max(len(env.vehicles), 1),
    )
    step_capacity = rate * env.config.DT
    # 取至少10倍单步容量，且不低于1e9 bits，确保跨多步积压
    big_data = max(step_capacity * 10.0, 1e9)
    big_comp = 5e9
    for v in env.vehicles:
        dag = build_dag(num_nodes=1, edges=[], comp=[big_comp], input_data=[big_data], deadline=50.0)
        reset_dag_fields(dag)
        v.task_dag = dag
    # 动作：全RSU，power=1.0
    actions = [[{"target": 1, "power": 1.0} for _ in env.vehicles] for _ in range(cfg.MAX_STEPS)]
    return env, actions


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #


def run_scenario(name: str, builder: Callable[[], Tuple[VecOffloadingEnv, List[List[Dict]]]]):
    print(f"\n===== Running scenario: {name} =====")
    env, action_plan = builder()
    recorder = attach_recorders(env)
    if name == "commwait_stress":
        setattr(env.config, "AUDIT_LOG_COMMWAIT", True)
    prev_status: Dict[int, np.ndarray] = {}
    original_compute_one_rate = env.channel.compute_one_rate
    v2v_rate_log: List[Tuple[int, str, float, float]] = []

    def _wrap_compute_one_rate():
        import inspect

        def wrapper(*args, **kwargs):
            rate = original_compute_one_rate(*args, **kwargs)
            origin = "unknown"
            for frame in inspect.stack():
                if frame.function in {
                    "_build_action_mask",
                    "_get_obs",
                    "_compute_comm_wait",
                    "_serve_tx_queue",
                    "_phase3_advance_comm_queues",
                    "_calculate_global_cft_critical_path",
                }:
                    origin = frame.function
                    break
            # time_key用于区分时隙
            curr_time = kwargs.get("curr_time", getattr(env, "time", 0.0))
            slot = int(round(curr_time / env.config.DT)) if env.config.DT > 0 else 0
            v2v_rate_log.append((slot, origin, curr_time, rate))
            return rate

        env.channel.compute_one_rate = wrapper

    if name == "double_v2v":
        _wrap_compute_one_rate()
    try:
        # 额外打印 reset 返回的 obs 信息（针对 single_rsu）
        reset_obs = env._get_obs()
        if name == "single_rsu" and reset_obs:
            log_obs_brief("reset_obs", reset_obs[0])
            # 验证 _is_rsu_queue_full 返回
            rsu_full = env._is_rsu_queue_full(0, new_task_cycles=env.vehicles[0].task_dag.total_comp[0])
            print(f"[diag/reset] _is_rsu_queue_full={rsu_full} keys={list(env.rsu_cpu_q[0].keys()) if env.rsu_cpu_q else None}")

        for step_idx, acts in enumerate(action_plan):
            # step 前的 obs 打印
            pre_obs = env._get_obs()
            if name == "single_rsu" and pre_obs:
                log_obs_brief(f"pre_step{step_idx}", pre_obs[0])
                if env.vehicles:
                    cw = env._compute_comm_wait(env.vehicles[0].id)
                    print(f"[pre_step{step_idx}] CommWait v2i={cw['total_v2i']:.6f} v2v={cw['total_v2v']:.6f}")
            obs, rewards, term, trunc, info = env.step(acts)
            log_step(env, recorder, rewards)
            if name == "single_rsu":
                log_step_rsu_detail(env, obs, acts)
                if obs:
                    log_obs_brief(f"post_step{step_idx}", obs[0])
            if name == "double_v2v" and v2v_rate_log:
                current_slot = int(round(env.time / env.config.DT))
                same_slot = [x for x in v2v_rate_log if x[0] == current_slot]
                if same_slot:
                    seen = {}
                    for slot, origin, t, r in same_slot:
                        seen.setdefault(origin, []).append(r)
                    print(f"[v2v_rate/slot={current_slot}] " + "; ".join([f"{k}: {seen[k]}" for k in seen]))
            assert_invariants(env, recorder, prev_status)
            if term or trunc:
                break
    finally:
        env.close()
        env.channel.compute_one_rate = original_compute_one_rate


def main():
    scenarios = [
        ("single_local", scenario_single_local),
        ("single_rsu", scenario_single_rsu),
        ("double_v2v", scenario_double_v2v),
        ("edge_blocking", scenario_edge_blocking),
    ]
    for name, builder in scenarios:
        run_scenario(name, builder)


if __name__ == "__main__":
    main()
