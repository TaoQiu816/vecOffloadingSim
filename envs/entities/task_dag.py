import numpy as np


class DAGTask:
    """
    DAG 任务实体类
    维护一个车辆的任务状态，包括依赖关系、剩余计算量、数据量和完成状态。

    修改记录:
    - [关键修复] 在 __init__ 中增加了 out_degree 的计算，用于 GraphBuilder 生成 7 维特征。
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

        # 状态码: 0=WAIT, 1=READY, 2=RUNNING, 3=DONE
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

        # 初始化: 入度为 0 的节点状态设为 READY (1)
        self.status[self.in_degree == 0] = 1

        # 失败标记 (超时或显式丢弃)
        self._is_failed = False

    @property
    def is_finished(self):
        """
        [关键属性] 判断整个 DAG 是否完成
        所有子任务状态均为 3 (DONE)
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
        Returns:
            np.array (bool): [Num_Subtasks], True 表示可调度 (READY)
        """
        # 只有状态为 READY (1) 的任务可以被调度
        # RUNNING (2) 的任务如果支持抢占也可以是 True，这里简化为不支持抢占
        return self.status == 1

    def assign_task(self, subtask_id, target):
        """
        将子任务分配给目标
        """
        # [新增] 越界检查 (Safety Guard)
        if subtask_id < 0 or subtask_id >= self.num_subtasks:
            # print(f"[Warning] Agent selected invalid subtask_id {subtask_id} (Max: {self.num_subtasks-1}). Ignored.")
            return

        # 原有逻辑: 如果是 READY 状态，则转为 RUNNING
        if self.status[subtask_id] == 1:
            self.status[subtask_id] = 2 # 标记为 RUNNING

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
        """
        self.status[subtask_id] = 3  # DONE

        # 查找后继节点 (Children)
        # adj[i, j] = 1 表示 i -> j
        children = np.where(self.adj[subtask_id, :] == 1)[0]

        for child in children:
            self.in_degree[child] -= 1
            # 依赖全部满足 (入度减为0)，状态转为 READY
            if self.in_degree[child] == 0:
                # 注意: 只有之前是 WAIT (0) 的才转 READY
                # 避免逻辑错误覆盖了其他状态 (虽然理论上入度为0不可能是其他状态)
                if self.status[child] == 0:
                    self.status[child] = 1