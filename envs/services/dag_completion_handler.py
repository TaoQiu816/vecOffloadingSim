"""
DAG完成处理器（集中化状态机更新逻辑）。

职责：
- 传输完成后的处理（INPUT/EDGE）
- 计算完成后的处理（写task_locations，调用_mark_done）
- 计算就绪检查与入队（统一入口）
- EDGE激活逻辑（child未assign不激活，同位置瞬时清零）

【重要语义】：
- status=READY(1) = "传输就绪"（前驱完成，可开始INPUT/EDGE传输）
- "计算就绪" = input_ready && edge_ready && exec_location已确定 && in_degree==0
- 任务只能调度一次（exec_locations只写一次）
- 同位置EDGE瞬时清零，不入队列
"""
from typing import TYPE_CHECKING, Dict, Set, Tuple, Optional, List, Any

if TYPE_CHECKING:
    from envs.vec_offloading_env import TransferJob, ComputeJob  # type: ignore
    from envs.entities.vehicle import Vehicle  # type: ignore
    from envs.entities.task_dag import DAGTask  # type: ignore
else:
    TransferJob = object
    ComputeJob = object
    Vehicle = object
    DAGTask = object


class DagCompletionHandler:
    """
    DAG完成处理器：集中处理传输/计算完成后的状态机更新。
    
    设计原则：
    - 不直接修改队列（由调用方管理）
    - 不直接修改账本（由调用方管理）
    - 只负责DAG状态更新和ComputeJob创建建议
    """
    
    def __init__(self, config):
        self.config = config
    
    def on_transfer_done(self, job: TransferJob, vehicle: Vehicle, 
                        time_now: float, active_edge_keys: Set,
                        veh_cpu_q: Dict, rsu_cpu_q: Dict, rsus: List,
                        debug_trace: bool = False) -> List[ComputeJob]:
        """
        传输完成后的处理。
        
        Args:
            job: 完成的TransferJob
            vehicle: 任务所属车辆
            time_now: 当前时间
            active_edge_keys: EDGE去重集合（用于清除）
            veh_cpu_q: 车辆计算队列容器
            rsu_cpu_q: RSU计算队列容器
            rsus: RSU列表（用于FAT选择）
            debug_trace: 是否记录trace事件
            
        Returns:
            list[ComputeJob]: 新创建的ComputeJob列表（已入队）
        """
        created_jobs = []
        
        if job.kind == "INPUT":
            # [INPUT完成] 回写rem_data=0
            vehicle.task_dag.rem_data[job.subtask_id] = 0.0
            if hasattr(vehicle.task_dag, 'input_ready'):
                vehicle.task_dag.input_ready[job.subtask_id] = True
            
            # [时隙内联动] 尝试入计算队列
            new_job = self._try_enqueue_compute_if_ready(
                vehicle, job.subtask_id, time_now, veh_cpu_q, rsu_cpu_q, rsus
            )
            if new_job is not None:
                created_jobs.append(new_job)
        
        else:  # EDGE
            # [EDGE完成] 回写inter_task_transfers
            child_id = job.subtask_id
            parent_id = job.parent_task_id
            
            if child_id in vehicle.task_dag.inter_task_transfers:
                if parent_id in vehicle.task_dag.inter_task_transfers[child_id]:
                    vehicle.task_dag.inter_task_transfers[child_id][parent_id]['rem_bytes'] = 0.0
            
            # 调用边到齐检查
            vehicle.task_dag.step_inter_task_transfers(child_id, 0.0, 0.0)
            
            # 清除去重标记
            edge_key = (job.owner_vehicle_id, child_id, parent_id)
            active_edge_keys.discard(edge_key)
            
            # [时隙内联动] 尝试入计算队列
            new_job = self._try_enqueue_compute_if_ready(
                vehicle, child_id, time_now, veh_cpu_q, rsu_cpu_q, rsus
            )
            if new_job is not None:
                created_jobs.append(new_job)
        
        return created_jobs
    
    def on_compute_done(self, job: ComputeJob, vehicle: Vehicle, 
                        time_now: float, veh_cpu_q: Dict = None, rsu_cpu_q: Dict = None, 
                        rsus: List = None, debug_trace: bool = False) -> Dict[str, Any]:
        """
        计算完成后的处理。
        
        Args:
            job: 完成的ComputeJob
            vehicle: 任务所属车辆
            time_now: 当前时间
            veh_cpu_q: 车辆计算队列（用于入队新解锁节点）
            rsu_cpu_q: RSU计算队列（用于入队新解锁节点）
            rsus: RSU列表（用于入队新解锁节点）
            debug_trace: 是否记录trace事件
            
        Returns:
            dict: 事件信息（用于trace/audit）
        """
        subtask_id = job.subtask_id
        dag = vehicle.task_dag
        
        # [完成位置落地] 从exec_locations读取位置码写入task_locations
        exec_loc = vehicle.task_dag.exec_locations[subtask_id]
        dag.task_locations[subtask_id] = exec_loc
        
        # 调用DAG的_mark_done（解锁后续节点：PENDING→READY）
        dag._mark_done(subtask_id)
        
        # 【关键修复】尝试将新解锁的READY节点入队
        if veh_cpu_q is not None and rsu_cpu_q is not None and rsus is not None:
            # 找到所有READY且已分配的节点
            for child_id in range(dag.num_subtasks):
                if (dag.status[child_id] == 1 and  # READY
                    dag.exec_locations[child_id] is not None):  # 已分配
                    # 尝试入队（会检查input/edge数据是否到齐）
                    self._try_enqueue_compute_if_ready(
                        vehicle, child_id, time_now, veh_cpu_q, rsu_cpu_q, rsus
                    )
        
        # 返回事件信息
        return {
            "type": "COMPUTE_DONE",
            "owner_vehicle_id": job.owner_vehicle_id,
            "subtask_id": subtask_id,
            "exec_location": exec_loc,
            "time": time_now
        }
    
    def _try_enqueue_compute_if_ready(self, vehicle: Vehicle, subtask_id: int,
                                      time_now: float, veh_cpu_q: Dict, 
                                      rsu_cpu_q: Dict, rsus: List) -> Optional[ComputeJob]:
        """
        [统一入口] 尝试将subtask入计算队列。
        
        【重要】计算就绪 vs 传输就绪：
        - DAG.status=READY(1) 仅表示"传输就绪"（前驱完成，可以开始传输）
        - "计算就绪"需要额外的数据条件：input_ready + edge_ready
        
        计算就绪条件（全部满足才入计算队列）：
        1. exec_locations[subtask] 已确定（任务只能调度一次）
        2. dag.status[subtask] == READY(1)（传输就绪/依赖满足）
        3. input_ready[subtask] == True（INPUT数据到齐）
        4. edge_ready[subtask] == True（所有EDGE数据到齐）
        5. subtask 尚未入队/未完成（去重）
        
        Returns:
            ComputeJob or None: 如果成功入队返回job，否则None
        """
        dag = vehicle.task_dag
        
        # 条件1: exec_locations已确定
        if vehicle.task_dag.exec_locations[subtask_id] is None:
            return None
        
        # 条件2: input_ready
        input_ready = (dag.rem_data[subtask_id] <= 1e-9)
        if not input_ready:
            return None
        
        # 条件3: edge_ready（所有入边数据到齐）
        edge_ready = True
        if hasattr(dag, 'inter_task_transfers') and subtask_id in dag.inter_task_transfers:
            for parent_id, transfer_info in dag.inter_task_transfers[subtask_id].items():
                if transfer_info.get('rem_bytes', 0) > 1e-9:
                    edge_ready = False
                    break
        if not edge_ready:
            return None
        
        # 条件4: 依赖满足（拓扑依赖）
        if dag.in_degree[subtask_id] > 0:
            return None
        if dag.status[subtask_id] != 1:  # 1=READY
            return None
        
        # 条件5: 去重（检查是否已入队或已完成）
        if dag.status[subtask_id] >= 2:  # 2=RUNNING, 3=COMPLETED
            return None
        
        # 所有条件满足，创建ComputeJob并入队
        exec_loc = vehicle.task_dag.exec_locations[subtask_id]
        task_comp = dag.total_comp[subtask_id]
        
        # 确定执行节点和处理器
        if exec_loc == 'Local':
            exec_node = ("VEH", vehicle.id)
            processor_id = 0
            queue = veh_cpu_q[vehicle.id]
        elif isinstance(exec_loc, tuple) and exec_loc[0] == 'RSU':
            rsu_id = exec_loc[1]
            exec_node = ("RSU", rsu_id)
            # RSU多处理器：选择最早可用时间（FAT）的队列
            if rsu_id < len(rsus):
                processor_id = rsus[rsu_id].get_earliest_available_processor()
                queue = rsu_cpu_q[rsu_id][processor_id]
            else:
                return None
        elif isinstance(exec_loc, int):
            # V2V卸载到其他车辆
            exec_node = ("VEH", exec_loc)
            processor_id = 0
            queue = veh_cpu_q[exec_loc]
        else:
            return None
        
        # 创建ComputeJob（延迟导入避免循环依赖）
        if TYPE_CHECKING:
            from envs.vec_offloading_env import ComputeJob
        else:
            # 运行时动态导入
            import sys
            if 'envs.vec_offloading_env' in sys.modules:
                ComputeJob = sys.modules['envs.vec_offloading_env'].ComputeJob
            else:
                from envs.vec_offloading_env import ComputeJob
        
        job = ComputeJob(
            owner_vehicle_id=vehicle.id,
            subtask_id=subtask_id,
            rem_cycles=task_comp,
            exec_node=exec_node,
            processor_id=processor_id,
            enqueue_time=time_now
        )
        
        # 入队
        queue.append(job)
        
        # 更新状态为RUNNING
        dag.status[subtask_id] = 2
        
        return job

