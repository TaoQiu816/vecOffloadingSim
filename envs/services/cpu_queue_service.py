"""
计算队列推进服务（纯逻辑，行为等同于原 env._serve_cpu_queue/_phase4_advance_cpu_queues）。

职责：
- 车辆/RSU 计算队列 FIFO + work-conserving 推进
- 每处理器/车辆拥有独立 DT 预算
- 本地计算能耗计成本，RSU 仅记录 cycles
"""
from typing import Dict, Tuple, TYPE_CHECKING
from collections import deque

from configs.config import SystemConfig as Cfg
from envs.services.step_results import CpuStepResult

if TYPE_CHECKING:
    from envs.vec_offloading_env import ComputeJob  # type: ignore
else:
    ComputeJob = object


class CpuQueueService:
    def __init__(self, config):
        self.config = config

    def step(self, veh_cpu_q: Dict, rsu_cpu_q: Dict, dt: float, time_now: float,
             veh_cpu_hz_fn, rsu_cpu_hz_fn) -> CpuStepResult:
        """
        推进所有计算队列，返回 CpuStepResult。

        Args:
            veh_cpu_q: {veh_id: deque[ComputeJob]}
            rsu_cpu_q: {rsu_id: {proc_id: deque[ComputeJob]}}
            dt: 步长 DT
            time_now: 当前时间
            veh_cpu_hz_fn: lambda veh_id -> cpu_freq
            rsu_cpu_hz_fn: lambda rsu_id -> cpu_freq
        """
        result = CpuStepResult()
        # 每步清零step级统计
        for q in veh_cpu_q.values():
            for job in q:
                job.step_time_used = 0.0
                job.step_cycles_done = 0.0
        for proc_dict in rsu_cpu_q.values():
            for q in proc_dict.values():
                for job in q:
                    job.step_time_used = 0.0
                    job.step_cycles_done = 0.0

        # 车辆 CPU
        for veh_id, q in veh_cpu_q.items():
            speed = veh_cpu_hz_fn(veh_id)
            self._serve_queue(q, dt, time_now, speed, ("VEH", veh_id), result)

        # RSU CPU
        for rsu_id, proc_dict in rsu_cpu_q.items():
            speed = rsu_cpu_hz_fn(rsu_id)
            for proc_id, q in proc_dict.items():
                self._serve_queue(q, dt, time_now, speed, ("RSU", rsu_id), result)

        return result

    def _serve_queue(self, queue: deque, dt: float, time_now: float, speed_cycles_per_s: float,
                     processor_node: Tuple, cpu_result: CpuStepResult):
        """
        单条计算队列推进（FIFO + work-conserving），收集到 cpu_result。
        """
        remaining = dt
        eps = 1e-9

        while remaining > eps and queue:
            job = queue[0]
            if job.start_time is None:
                job.start_time = time_now + (dt - remaining)

            can = speed_cycles_per_s * remaining
            do = min(job.rem_cycles, can)
            time_used = do / speed_cycles_per_s

            job.rem_cycles -= do
            job.step_cycles_done += do
            job.step_time_used += time_used
            remaining -= time_used

            if processor_node[0] == "VEH":
                u = processor_node[1]
                kappa = getattr(Cfg, 'KAPPA', 1e-28)
                f_u = speed_cycles_per_s
                energy = kappa * do * (f_u ** 2)
                cpu_result.energy_delta_cost_local[u] = cpu_result.energy_delta_cost_local.get(u, 0.0) + energy
                cpu_result.cycles_done_local[u] = cpu_result.cycles_done_local.get(u, 0.0) + do
            else:
                rsu_id = processor_node[1]
                cpu_result.cycles_done_rsu_record[rsu_id] = cpu_result.cycles_done_rsu_record.get(rsu_id, 0.0) + do

            if job.rem_cycles <= eps:
                job.finish_time = time_now + (dt - remaining)
                queue.popleft()
                cpu_result.completed_jobs.append(job)
