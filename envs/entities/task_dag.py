import numpy as np


class DAGTask:
    """
    DAG 任务实体类
    
    功能：
    - 维护一个车辆的任务状态，包括依赖关系、剩余计算量、数据量和完成状态
    - 管理子任务的状态转换（PENDING -> READY -> RUNNING -> COMPLETED）
    - 记录每个子任务的执行位置（确保唯一性）
    - 跟踪任务间依赖数据传输状态
    
    核心原则：
    - 只负责"生"任务，不负责"做"任务（调度逻辑在环境层）
    - 每个子任务只能分配一次执行位置，一旦分配不能再次调度
    
    状态定义：
    - 0 = PENDING (等待依赖满足)
    - 1 = READY (依赖已满足，可调度)
    - 2 = RUNNING (已调度并正在执行)
    - 3 = COMPLETED (执行完成)
    
    修改记录:
    - [关键修复] 在 __init__ 中增加了 out_degree 的计算，用于 GraphBuilder 生成 7 维特征
    - [关键修复] 添加防止重复调度机制，确保每个子任务只能分配一次执行位置
    - [关键修复] 统一状态命名（PENDING/READY/RUNNING/COMPLETED）
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
        self.adj = np.array(adj) # 确保转换为 numpy array
        self.data_matrix = data_matrix
        self.deadline = deadline

        self.num_subtasks = len(profiles)
        self.start_time = 0.0  # 由 Env 在 reset 时赋值

        # 状态码定义 (状态转换: PENDING -> READY -> RUNNING -> COMPLETED)
        # 0 = PENDING (等待依赖满足)
        # 1 = READY (依赖已满足，可调度)
        # 2 = RUNNING (已调度并正在执行)
        # 3 = COMPLETED (执行完成)
        self.status = np.zeros(self.num_subtasks, dtype=int)

        # 初始化计算量 (Cycles)
        self.total_comp = np.array([p['comp'] for p in profiles], dtype=np.float32)
        self.rem_comp = self.total_comp.copy()

        # 初始化数据量 (Bits) - 用于卸载传输
        # 简化模型: 这里的 rem_data 代表需要传输到计算节点的"任务体"大小
        # 如果在本地执行，传输速率无限大，此项瞬间归零
        self.total_data = np.array([p.get('input_data', 0.0) for p in profiles], dtype=np.float32)
        # 确保数据量不为0以避免除零错误 (最小 1KB)
        self.total_data = np.maximum(self.total_data, 8000.0)
        self.rem_data = self.total_data.copy()

        # [依赖管理]
        # 1. 入度 (In-Degree): 表示有多少个前驱任务未完成
        #    axis=0 (列和) -> 指向该节点的边数
        self.in_degree = np.sum(self.adj, axis=0)

        # 2. [关键新增] 出度 (Out-Degree): 表示该任务影响多少个后继任务
        #    axis=1 (行和) -> 该节点发出的边数
        #    这对于 GNN 提取拓扑特征至关重要，反映了任务的"影响力"
        self.out_degree = np.sum(self.adj, axis=1)

        # 新增: 任务位置跟踪
        # 记录每个子任务的执行位置: 'Local' | 'RSU' | int (目标车辆ID)
        self.task_locations = [None] * self.num_subtasks

        # 新增: 依赖数据传输状态跟踪
        # 存储每个依赖关系的数据传输状态
        # 结构: {child_task_id: {parent_task_id: {'rem_data': float, 'transfer_speed': float}}}
        self.inter_task_transfers = {}
        # 标记哪些任务正在等待依赖数据传输
        self.waiting_for_data = [False] * self.num_subtasks

        # 初始化: 入度为 0 的节点状态设为 READY (1)，可以直接调度
        # 入度 > 0 的节点状态为 PENDING (0)，等待依赖满足
        self.status[self.in_degree == 0] = 1

        # 失败标记 (超时或显式丢弃)
        self._is_failed = False

    @property
    def is_finished(self):
        """
        [关键属性] 判断整个 DAG 是否完成
        所有子任务状态均为 3 (COMPLETED)
        """
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
        获取当前可调度的任务掩码
        
        注意：只有状态为 READY (1) 且尚未分配执行位置的任务可以被调度
        Returns:
            np.array (bool): [Num_Subtasks], True 表示可调度 (READY且未分配)
        """
        # 只有状态为 READY (1) 且 task_locations 为 None 的任务可以被调度
        # 一旦任务被分配（task_locations不为None），就不能再次调度
        # RUNNING (2) 和 COMPLETED (3) 的任务不可调度
        ready_mask = (self.status == 1)
        not_assigned_mask = np.array([loc is None for loc in self.task_locations])
        return ready_mask & not_assigned_mask
        
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
        
        注意：每个子任务只能分配一次，一旦分配就不能再次调度
        
        Args:
            subtask_id: 子任务索引
            target: 执行位置 ('Local' | 'RSU' | int车辆ID)
        
        Returns:
            bool: 是否成功分配 (False表示任务已分配过或状态不允许)
        """
        # [安全检查1] 越界检查
        if subtask_id < 0 or subtask_id >= self.num_subtasks:
            return False
        
        # [安全检查2] 防止重复调度：如果任务已经分配过执行位置，拒绝再次分配
        if self.task_locations[subtask_id] is not None:
            # 任务已经被分配过，不允许再次调度
            return False
        
        # [安全检查3] 只有READY状态的任务才能被分配
        if self.status[subtask_id] != 1:
            # 非READY状态的任务不能分配
            return False

        # 记录任务执行位置（每个子任务只能选择一个执行位置）
        self.task_locations[subtask_id] = target

        # 状态转换: READY (1) -> RUNNING (2)
        self.status[subtask_id] = 2
        
        return True

    def step_progress(self, subtask_id, comp_speed, comm_speed, dt):
        """
        [修复版] 推进单个子任务的进度
        修复了"传输完成后立即返回导致时间浪费"的 Bug。

        Args:
            subtask_id: 子任务索引
            comp_speed: 计算速度 (cycles/s)
            comm_speed: 通信速度 (bits/s)
            dt: 时间步长 (s)

        Returns:
            bool: 该子任务是否在本 Step 完成
        """
        if self.status[subtask_id] != 2:
            return False  # 非 RUNNING 状态不更新

        # 用于记录当前 step 剩余可用的时间
        # 初始为完整的时间步长
        remaining_dt = dt

        # ==========================================
        # 1. 传输阶段 (Transmission)
        # ==========================================
        if self.rem_data[subtask_id] > 0:
            # 计算传完剩余数据需要多少时间
            # 加上 1e-9 防止除零
            time_to_finish_data = self.rem_data[subtask_id] / (comm_speed + 1e-9)

            if time_to_finish_data > remaining_dt:
                # 情况 A: 本 step 时间不够传完数据
                transmitted = comm_speed * remaining_dt
                self.rem_data[subtask_id] -= transmitted
                return False  # 时间耗尽，仍在传输，直接返回
            else:
                # 情况 B: 本 step 内可以传完数据
                # 扣除传输消耗的时间，保留剩余时间给计算阶段
                self.rem_data[subtask_id] = 0.0
                remaining_dt -= time_to_finish_data
                # 【关键修改】不要 return False，继续向下执行计算逻辑！

        # ==========================================
        # 2. 计算阶段 (Computation)
        # ==========================================
        # 只有当数据传输完成（rem_data == 0）且还有剩余时间时，才进行计算
        if self.rem_comp[subtask_id] > 0 and remaining_dt > 0:
            # 利用剩余时间进行计算
            computed = comp_speed * remaining_dt
            self.rem_comp[subtask_id] -= computed

            # 如果计算完成
            if self.rem_comp[subtask_id] <= 0:
                self.rem_comp[subtask_id] = 0.0
                self._mark_done(subtask_id)
                return True # 任务在本 step 完成

        return False

    def _mark_done(self, subtask_id):
        """
        内部方法: 标记任务完成并解锁后继任务
        
        状态转换: RUNNING (2) -> COMPLETED (3)
        """
        self.status[subtask_id] = 3  # COMPLETED

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
            
            # 依赖全部满足 (入度减为0)
            if self.in_degree[child] == 0:
                # 检查是否有未完成的数据传输
                if child in self.inter_task_transfers and len(self.inter_task_transfers[child]) > 0:
                    # 有未完成的数据传输，保持PENDING状态
                    if self.status[child] == 0:
                        self.status[child] = 0  # 继续PENDING，等待数据传输完成
                else:
                    # 没有未完成的数据传输，转为READY状态
                    if self.status[child] == 0:
                        self.status[child] = 1  # PENDING (0) -> READY (1)