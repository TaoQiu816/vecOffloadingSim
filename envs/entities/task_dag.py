import numpy as np
from configs.config import SystemConfig as Cfg
from models.dag_features import compute_forward_levels, compute_backward_levels, compute_shortest_path_matrix, normalize_distance_matrix
from configs.config import SystemConfig as Cfg


class DAGTask:
    """
    DAG任务实体类
    
    功能：
    - 维护任务状态，包括依赖关系、剩余计算量、数据量和完成状态
    - 管理子任务状态转换：PENDING -> READY -> RUNNING -> COMPLETED
    - 记录每个子任务的执行位置（确保唯一性）
    - 跟踪任务间依赖数据传输状态
    
    核心原则：
    - 只负责任务状态管理，不负责调度决策（调度逻辑在环境层）
    - 每个子任务只能分配一次执行位置，一旦分配不能再次调度
    
    状态定义：
    - 0 = PENDING (等待依赖满足)
    - 1 = READY (依赖已满足，可调度)
    - 2 = RUNNING (已调度并正在执行)
    - 3 = COMPLETED (执行完成)
    """

    def __init__(self, task_id, adj, profiles, data_matrix, deadline):
        """
        Args:
            task_id: 任务/DAG ID
            adj: 邻接矩阵 (NxN), adj[i][j]=1 表示任务 i 指向任务 j (i -> j)
            profiles: 节点属性列表 [{'comp':..., 'input_data':...}]
            data_matrix: 边传输数据量矩阵 (NxN)
            deadline: 截止时间 (s)
        """
        self.id = task_id
        self.adj = np.array(adj)
        self.data_matrix = data_matrix
        self.deadline = deadline
        self.num_subtasks = len(profiles)
        self.start_time = 0.0  # 由环境在reset时设置

        # 状态管理：PENDING(0) -> READY(1) -> RUNNING(2) -> COMPLETED(3)
        self.status = np.zeros(self.num_subtasks, dtype=int)

        # 计算量和数据量
        self.total_comp = np.array([p['comp'] for p in profiles], dtype=np.float32)
        self.rem_comp = self.total_comp.copy()
        self.total_data = np.array([p.get('input_data', 0.0) for p in profiles], dtype=np.float32)

        # 依赖管理：入度和出度
        self.in_degree = np.sum(self.adj, axis=0)   # 前驱任务数
        self.out_degree = np.sum(self.adj, axis=1)  # 后继任务数

        entry_mask = (self.in_degree == 0)
        self.total_data[entry_mask] = np.maximum(self.total_data[entry_mask], 8000.0)
        self.total_data[~entry_mask] = np.maximum(self.total_data[~entry_mask], 0.0)
        self.rem_data = self.total_data.copy()

        # 执行位置跟踪：每个子任务只能分配一次执行位置
        # 值: 'Local' | 'RSU' | int(车辆ID) | None(未分配)
        self.task_locations = [None] * self.num_subtasks

        # 依赖数据传输跟踪
        # 结构: {child_id: {parent_id: {'rem_data': float, 'transfer_speed': float}}}
        self.inter_task_transfers = {}
        self.waiting_for_data = [False] * self.num_subtasks

        # 时间计算：EST（执行开始时间）和CT（完成时间）
        # 每个子任务存储其EST和CT，用于CFT计算
        self.EST = np.full(self.num_subtasks, -1.0, dtype=np.float32)  # -1表示未计算
        self.CT = np.full(self.num_subtasks, -1.0, dtype=np.float32)  # -1表示未计算

        # 初始化：入度为0的节点设为READY，其他为PENDING
        self.status[self.in_degree == 0] = 1
        self._is_failed = False
        self.timeout_logged = False  # 用于死因诊断，避免重复打印
        
        # 拓扑特征：前向层级、后向层级、最短路径距离矩阵
        try:
            from models.dag_features import compute_forward_levels, compute_backward_levels, compute_shortest_path_matrix, normalize_distance_matrix
            self.L_fwd = compute_forward_levels(self.adj)
            self.L_bwd = compute_backward_levels(self.adj)
            self.Delta = normalize_distance_matrix(
                compute_shortest_path_matrix(self.adj, Cfg.MAX_NODES),
                Cfg.MAX_NODES
            )
        except ImportError:
            # 如果导入失败（例如在初始化阶段），延迟计算
            self.L_fwd = None
            self.L_bwd = None
            self.Delta = None

    @property
    def is_finished(self):
        """判断整个DAG是否完成（所有子任务均为COMPLETED状态）"""
        return np.all(self.status == 3)

    @property
    def is_failed(self):
        """
        判断任务是否失败 (超时)
        注意: 需要外部显式调用 check_deadline 或设置
        """
        return self._is_failed

    def set_failed(self):
        self._is_failed = True

    def get_action_mask(self):
        """
        获取可调度的任务掩码
        
        只有状态为READY且尚未分配执行位置的任务可被调度
        
        Returns:
            np.array(bool): [Num_Subtasks], True表示可调度
        """
        ready_mask = (self.status == 1)
        not_assigned_mask = np.array([loc is None for loc in self.task_locations])
        return ready_mask & not_assigned_mask
    
    def compute_task_priority(self, task_id: int) -> float:
        """
        计算任务的优先级分数
        
        公式：Score(i) = W1 * L_bwd[i] + W2 * (total_comp[i] / NORM_MAX_COMP) + W3 * (out_degree[i] / MAX_NODES)
        
        Args:
            task_id: 任务索引
            
        Returns:
            float: 优先级分数（越大优先级越高）
        """
        # 确保拓扑特征已计算
        if self.L_bwd is None:
            from models.dag_features import compute_backward_levels
            self.L_bwd = compute_backward_levels(self.adj)
        
        w1 = Cfg.PRIORITY_W1
        w2 = Cfg.PRIORITY_W2
        w3 = Cfg.PRIORITY_W3
        
        # 后向层级（主导因素）
        score_bwd = w1 * self.L_bwd[task_id]
        
        # 计算量（归一化）
        comp_norm = self.total_comp[task_id] / Cfg.NORM_MAX_COMP
        score_comp = w2 * comp_norm
        
        # 出度（归一化）
        out_deg_norm = self.out_degree[task_id] / Cfg.MAX_NODES
        score_out = w3 * out_deg_norm
        
        return score_bwd + score_comp + score_out
    
    def get_top_priority_task(self):
        """
        获取优先级最高的可调度任务
        
        Returns:
            int or None: 优先级最高的任务索引，如果没有可调度的任务则返回None
        """
        action_mask = self.get_action_mask()
        ready_tasks = np.where(action_mask)[0]
        
        if len(ready_tasks) == 0:
            return None
        
        # 计算所有就绪任务的优先级
        priorities = np.array([self.compute_task_priority(tid) for tid in ready_tasks])
        
        # 选择优先级最高的任务
        top_idx = np.argmax(priorities)
        return int(ready_tasks[top_idx])
        
    def step_inter_task_transfers(self, subtask_id, transfer_speed, dt):
        """
        处理任务间依赖数据的传输进度
        
        Args:
            subtask_id: 接收数据的子任务ID
            transfer_speed: 传输速度 (bits/s)
            dt: 时间步长 (s)
            
        Returns:
            bool: 所有依赖数据是否传输完成
        """
        if subtask_id not in self.inter_task_transfers:
            return True
            
        all_transfers_completed = True
        completed_parents = []
        
        # 处理所有依赖数据传输
        for parent_id, transfer_info in self.inter_task_transfers[subtask_id].items():
            if transfer_info['rem_data'] > 0:
                # 更新传输速度
                if transfer_speed > 0:
                    transfer_info['transfer_speed'] = transfer_speed
                
                # 计算传输数据量
                transmitted = transfer_info['transfer_speed'] * dt
                
                if transmitted >= transfer_info['rem_data']:
                    # 传输完成
                    transfer_info['rem_data'] = 0
                    completed_parents.append(parent_id)
                else:
                    # 传输未完成
                    transfer_info['rem_data'] -= transmitted
                    all_transfers_completed = False
            else:
                completed_parents.append(parent_id)
        
        # 移除已完成的传输
        for parent_id in completed_parents:
            del self.inter_task_transfers[subtask_id][parent_id]
            
        # 如果所有传输都完成
        if all_transfers_completed:
            del self.inter_task_transfers[subtask_id]
            self.waiting_for_data[subtask_id] = False
            
            # 如果所有依赖都已满足且任务处于PENDING状态，转为READY
            if self.in_degree[subtask_id] == 0 and self.status[subtask_id] == 0:
                self.status[subtask_id] = 1  # PENDING (0) -> READY (1)
        
        return all_transfers_completed

    def assign_task(self, subtask_id, target):
        """
        将子任务分配给目标执行位置
        
        每个子任务只能分配一次，一旦分配就不能再次调度
        
        Args:
            subtask_id: 子任务索引
            target: 执行位置 ('Local' | 'RSU' | int车辆ID)
        
        Returns:
            bool: 是否成功分配（False表示任务已分配过或状态不允许）
        """
        # 安全检查：越界、重复分配、状态检查
        if subtask_id < 0 or subtask_id >= self.num_subtasks:
            return False
        if self.task_locations[subtask_id] is not None:
            return False
        if self.status[subtask_id] != 1:
            return False

        # 记录执行位置并转换状态
        self.task_locations[subtask_id] = target
        self.status[subtask_id] = 2  # READY -> RUNNING
        return True

    def step_progress(self, subtask_id, comp_speed, comm_speed, dt):
        """
        推进单个子任务的执行进度
        
        先传输数据（如果需要），然后进行计算。如果传输在本step内完成，
        剩余时间用于计算，避免时间浪费。
        
        Args:
            subtask_id: 子任务索引
            comp_speed: 计算速度 (cycles/s)
            comm_speed: 通信速度 (bits/s)
            dt: 时间步长 (s)

        Returns:
            bool: 该子任务是否在本step完成
        """
        if self.status[subtask_id] != 2:
            return False

        remaining_dt = dt

        # 传输阶段
        if self.rem_data[subtask_id] > 0:
            time_to_finish_data = self.rem_data[subtask_id] / (comm_speed + 1e-9)
            if time_to_finish_data > remaining_dt:
                # 传输未完成，消耗全部时间
                transmitted = comm_speed * remaining_dt
                self.rem_data[subtask_id] -= transmitted
                return False
            else:
                # 传输完成，剩余时间用于计算
                self.rem_data[subtask_id] = 0.0
                remaining_dt -= time_to_finish_data

        # 计算阶段（数据传输完成后）
        if self.rem_comp[subtask_id] > 0 and remaining_dt > 0:
            computed = comp_speed * remaining_dt
            self.rem_comp[subtask_id] -= computed
            if self.rem_comp[subtask_id] <= 0:
                self.rem_comp[subtask_id] = 0.0
                self._mark_done(subtask_id)
                return True

        return False

    def _mark_done(self, subtask_id):
        """
        标记任务完成并解锁后继任务
        
        状态转换: RUNNING(2) -> COMPLETED(3)
        """
        self.status[subtask_id] = 3

        # 查找后继节点 (Children)
        # adj[i, j] = 1 表示 i -> j
        children = np.where(self.adj[subtask_id, :] == 1)[0]

        for child in children:
            self.in_degree[child] -= 1
            
            # 获取依赖传输的数据量
            transfer_data = self.data_matrix[subtask_id, child]
            
            if transfer_data > 0:
                # 需要传输依赖数据，创建传输任务
                if child not in self.inter_task_transfers:
                    self.inter_task_transfers[child] = {}
                
                # 初始化传输状态
                self.inter_task_transfers[child][subtask_id] = {
                    'rem_data': transfer_data,
                    'transfer_speed': 0.0  # 初始为0，由环境在step中设置
                }
                
                # 标记子任务正在等待数据传输
                self.waiting_for_data[child] = True
            
            # 如果依赖全部满足（入度减为0），检查数据传输状态
            if self.in_degree[child] == 0:
                has_pending_transfers = (child in self.inter_task_transfers and 
                                       len(self.inter_task_transfers[child]) > 0)
                if not has_pending_transfers and self.status[child] == 0:
                    self.status[child] = 1  # PENDING -> READY
