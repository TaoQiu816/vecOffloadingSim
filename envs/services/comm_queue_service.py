"""
通信队列推进服务（纯逻辑，行为等同于原 env._serve_tx_queue/_phase3_advance_comm_queues）。

职责：
- 对每个 sender 的 V2I/V2V 队列并行推进，FIFO + work-conserving
- 同一 sender 的 V2I 与 V2V 各自拥有 DT 预算（不互相抢占）
- INPUT 功率来源于动作映射的 tx_power_dbm；EDGE 使用 TX_POWER_MAX_DBM（通过 power_dbm_override）
- 完成后不直接修改 DAG/队列外部状态，只返回 CommStepResult 由调用方处理
"""
from typing import Dict, Set, Tuple
from collections import deque

from typing import TYPE_CHECKING, Callable, Optional
from envs.services.step_results import CommStepResult
if TYPE_CHECKING:
    from envs.vec_offloading_env import TransferJob  # type: ignore
else:
    TransferJob = object
from configs.config import SystemConfig as Cfg


class CommQueueService:
    def __init__(self, channel, config):
        self.channel = channel
        self.config = config

    def step(self, txq_v2i: Dict, txq_v2v: Dict, dt: float, time_now: float,
             rate_fn: Optional[Callable[[TransferJob, tuple], float]] = None) -> CommStepResult:
        """
        推进所有通信队列，返回结果汇总。

        Args:
            txq_v2i: {tx_node: deque[TransferJob]}
            txq_v2v: {tx_node: deque[TransferJob]}
            dt: 步长 DT
            time_now: 当前时间
        """
        # 每步清零job的step级统计，确保时间预算按步计算
        for q_dict in (txq_v2i, txq_v2v):
            for q in q_dict.values():
                for job in q:
                    job.step_time_used = 0.0
                    job.step_bytes_sent = 0.0

        result = CommStepResult()
        all_tx_nodes: Set = set(txq_v2i.keys()) | set(txq_v2v.keys())
        for tx_node in all_tx_nodes:
            if tx_node in txq_v2i:
                self._serve_tx_queue(txq_v2i[tx_node], dt, time_now, tx_node, "V2I", result, rate_fn)
            if tx_node in txq_v2v:
                self._serve_tx_queue(txq_v2v[tx_node], dt, time_now, tx_node, "V2V", result, rate_fn)
        return result

    def _serve_tx_queue(self, queue: deque, dt: float, time_now: float, tx_node: Tuple, queue_type: str, comm_result: CommStepResult,
                        rate_fn: Optional[Callable[[TransferJob, tuple], float]] = None):
        """
        推进单条通信队列（FIFO + work-conserving）
        """
        remaining = dt
        eps = 1e-9

        while remaining > eps and queue:
            job = queue[0]

            # 首次推进：记录start_time
            if job.start_time is None:
                job.start_time = time_now + (dt - remaining)

            # 计算速率（EDGE用 power_dbm_override=MAX，INPUT 用 job.tx_power_dbm）
            rate = rate_fn(job, tx_node) if rate_fn is not None else self._compute_job_rate(job, tx_node, time_now, queue_type)
            if rate <= eps:
                break  # 无法推进

            # 推进传输
            send = min(job.rem_bytes, rate * remaining)
            time_used = send / rate

            job.rem_bytes -= send
            job.step_bytes_sent += send
            job.step_time_used += time_used
            remaining -= time_used

            # 能耗记账：仅车辆发送端计入
            if tx_node[0] == "VEH":
                u = tx_node[1]
                p_tx = self.config.dbm2watt(job.tx_power_dbm)
                energy = p_tx * time_used
                if job.kind == "INPUT":
                    comm_result.energy_delta_cost[u] = comm_result.energy_delta_cost.get(u, 0.0) + energy
                else:
                    comm_result.energy_delta_record_edge[u] = comm_result.energy_delta_record_edge.get(u, 0.0) + energy

            # 完成处理
            if job.rem_bytes <= eps:
                job.finish_time = time_now + (dt - remaining)
                queue.popleft()
                comm_result.completed_jobs.append(job)

    def _compute_job_rate(self, job: TransferJob, tx_node: Tuple, time_now: float, queue_type: str) -> float:
        """
        计算TransferJob的传输速率
        """
        # 获取src和dst位置
        if tx_node[0] == "VEH":
            src_veh = tx_node[1]
        else:
            src_veh = None

        # 对于 V2I/V2V，我们只需要 dst 坐标，旧逻辑在 env 中计算
        # 这里直接复用 channel.compute_one_rate，遵循 power_dbm_override 规则
        power_override = job.tx_power_dbm
        if job.kind == "EDGE":
            power_override = self.config.TX_POWER_MAX_DBM

        # channel.compute_one_rate 接口期望 vehicle 对象；此 service 保持签名兼容，仍由外部调用时提供 vehicle 对象
        # 为最小改动，这里假设 channel.compute_one_rate 能接受 tx_node 里的车辆/RSU
        return self.channel.compute_one_rate(
            job.src_node[1] if job.src_node[0] == "VEH" else self._build_rsu_proxy(job.src_node[1]),
            self._resolve_dst_pos(job.dst_node),
            queue_type,
            time_now,
            power_dbm_override=power_override,
        )

    def _resolve_dst_pos(self, dst_node: Tuple):
        # 在 service 中仅用于计算速率的辅助函数
        # 由调用方确保 dst_node 合法
        if dst_node[0] == "VEH":
            # 直接返回车辆对象，由 channel 处理位置获取
            return dst_node[1]
        else:
            # RSU 由 channel 内部处理
            return dst_node[1]

    def _build_rsu_proxy(self, rsu_id: int):
        # 占位：若需要为 RSU 发送端构造代理，可在此扩展
        return rsu_id
