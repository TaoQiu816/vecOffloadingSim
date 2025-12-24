import gymnasium as gym
import numpy as np
import math
from configs.config import SystemConfig as Cfg
from envs.modules.channel import ChannelModel
from envs.entities.vehicle import Vehicle
from envs.entities.task_dag import DAGTask
from utils.dag_generator import DAGGenerator


class VecOffloadingEnv(gym.Env):
    """
    [VecOffloadingEnv]
    车联网边缘计算任务卸载环境 (OpenAI Gym 接口兼容)

    特性:
    1. DAG 依赖感知: 奖励函数基于 DAG 关键路径的预期完成时间 (CFT)。
    2. 移动性管理: 通过 Time-to-Leave (TTL) 预测防止车辆驶出 RSU 范围导致的任务中断。
    3. 拥堵控制: 包含 RSU 与 车辆 (V2V) 的队列限制，并通过 Mask 屏蔽非法动作。
    4. 异构资源: 区分 RSU (高算力、高带宽) 与 车辆 (低算力、V2V共享) 的资源模型。
    """

    def __init__(self):
        # 初始化物理信道模型 (负责真实的 SINR 和 Rate 计算)
        self.channel = ChannelModel()
        # 初始化 DAG 生成器
        self.dag_gen = DAGGenerator()

        # 车辆列表与全局时间
        self.vehicles = []
        self.time = 0.0

        # RSU 状态追踪
        self.rsu_queue_curr = 0  # 当前 RSU 任务队列长度 (任务数)

        # 奖励计算辅助变量 (上一时刻的全局 CFT)
        self.last_global_cft = 0.0

    def reset(self, seed=None, options=None):
        """
        [环境重置]
        重置全局时间、清空队列，并根据 Config 生成新的车辆和 DAG 任务。
        """
        if seed:
            np.random.seed(seed)

        self.vehicles = []
        self.time = 0.0
        self.rsu_queue_curr = 0

        # 生成车辆实体
        for i in range(Cfg.NUM_VEHICLES):
            # 随机分布在 MAP 范围内
            pos = np.random.rand(2) * Cfg.MAP_SIZE
            v = Vehicle(i, pos)

            v.cpu_freq = np.random.uniform(Cfg.MIN_VEHICLE_CPU_FREQ, Cfg.MAX_VEHICLE_CPU_FREQ)

            # 生成 DAG 任务
            # 使用 Config 定义的节点范围，保证训练难度的统一性
            # 2. [关键修改] 生成任务时传入该车的 cpu_freq
            # 这样生成的 Deadline 将完美契合该车的本地处理能力
            n_node = np.random.randint(Cfg.MIN_NODES, Cfg.MAX_NODES + 1)
            adj, prof, data, ddl = self.dag_gen.generate(n_node, veh_f=v.cpu_freq)

            # 绑定任务并初始化
            v.task_dag = DAGTask(0, adj, prof, data, ddl)
            v.task_dag.start_time = 0.0
            v.task_queue_len = 0  # 本地排队任务数

            # 移动性初始化
            # 随机生成速度向量，范围 [-MaxVel, +MaxVel]
            v.vel = (np.random.rand(2) - 0.5) * 2 * Cfg.MAX_VELOCITY

            self.vehicles.append(v)

        # [基准计算] 计算初始状态下的全局预计完成时间 (CFT)
        # 此时所有任务默认为本地执行或未调度状态
        self.last_global_cft = self._calculate_global_cft_critical_path()

        return self._get_obs(), {}

    def step(self, actions):
        """
        [环境步进]
        1. 动作执行: 目标选择与功率控制。
        2. 物理演化: 信道计算、任务进度更新、车辆移动。
        3. 奖励反馈: 基于 CFT 缩减量的奖励计算。
        """
        # --- 1. 应用动作 (Action Application) ---
        for i, v in enumerate(self.vehicles):
            act = actions[i]
            if not act: continue  # 跳过无效动作

            # 解析动作: Target (离散) & Power (连续)
            tgt_idx = int(act['target'])
            subtask_idx = int(act['subtask'])

            # 目标映射: 0->Local, 1->RSU, 2+->Neighbor
            if tgt_idx == 0:
                target = 'Local'
            elif tgt_idx == 1:
                target = 'RSU'
            else:
                # 映射回 Neighbor List (注意: 动作空间需与 Observation 中的邻居顺序对齐)
                neighbor_id = tgt_idx - 2
                if 0 <= neighbor_id < len(self.vehicles):
                    target = neighbor_id  # 使用车辆 ID (int)
                else:
                    target = 'Local'  # Fallback

            # 功率控制: 映射 [0, 1] -> [Min_dBm, Max_dBm]
            p_norm = np.clip(act.get('power', 1.0), 0.0, 1.0)
            v.tx_power_dbm = Cfg.TX_POWER_MIN_DBM + p_norm * (Cfg.TX_POWER_MAX_DBM - Cfg.TX_POWER_MIN_DBM)

            # 状态更新
            v.curr_subtask = subtask_idx
            v.curr_target = target

            # [Core Logic] 在 DAG 层面分配任务，这会影响 get_ready_time 的计算
            # 实际的传输延迟将在物理计算环节处理，这里先标记“意图”
            v.task_dag.assign_task(subtask_idx, target)

            # 队列负载更新 (用于拥堵模拟)
            if target == 'RSU':
                self.rsu_queue_curr += 1
            elif isinstance(target, int):
                self.vehicles[target].task_queue_len += 1

        # --- 2. 物理计算 (Physics & Channel) ---
        # 计算真实物理速率 (基于当前位置、功率和干扰)
        # rates 返回字典: {veh_id: rate_bps} (V2I) 或 V2V 矩阵，视 ChannelModel 实现而定
        # 此处假设 compute_rates 返回 V2I 速率，V2V 按需计算
        rates_v2i = self.channel.compute_rates(self.vehicles, Cfg.RSU_POS)

        # --- 3. 任务处理与移动 (Processing & Mobility) ---
        for v in self.vehicles:
            c_spd = 0.0  # 通信速率
            comp_spd = 0.0  # 计算速率
            tgt = v.curr_target

            # 车辆的 CPU 频率影响计算速率
            if tgt == 'Local':
                comp_spd = v.cpu_freq  # 使用车辆的 CPU 频率代替固定值
                c_spd = 1e12  # 本地总线极快，忽略传输延迟
            elif tgt == 'RSU':
                comp_spd = Cfg.F_RSU  # RSU 算力仍然使用配置中的固定值
                c_spd = rates_v2i.get(v.id, 1e-6)  # 获取 V2I 速率
            elif isinstance(tgt, int):
                # V2V 情况
                comp_spd = v.cpu_freq  # 使用车辆的 CPU 频率代替固定值
                target_veh = self.vehicles[tgt]
                # 计算点对点 V2V 速率
                c_spd = self.channel.compute_one_rate(v, target_veh.pos, link_type='V2V')

            # 防止除零错误
            c_spd = max(c_spd, 1e-6)

            # [DAG 步进] 更新子任务进度
            # step_progress 内部应处理数据传输 (c_spd) 和计算 (comp_spd) 的时间消耗
            task_finished = False
            if v.curr_subtask is not None:
                task_finished = v.task_dag.step_progress(v.curr_subtask, comp_spd, c_spd, Cfg.DT)

                if task_finished:
                    # 任务完成，释放队列资源
                    if tgt == 'RSU':
                        self.rsu_queue_curr = max(0, self.rsu_queue_curr - 1)
                    elif isinstance(tgt, int):
                        target_veh = self.vehicles[tgt]
                        target_veh.task_queue_len = max(0, target_veh.task_queue_len - 1)

                    # 重置当前动作状态
                    v.curr_subtask = None

            # [Mobility] 位置更新
            v.update_pos(Cfg.DT, Cfg.MAP_SIZE)

        self.time += Cfg.DT

        # --- 4. 奖励计算 (Reward Function) ---
        # 基于公式 (28): Reward = (CFT_prev - CFT_curr)
        # 我们使用基于关键路径的 CFT 估算，更能体现 DAG 的并行优势
        current_global_cft = self._calculate_global_cft_critical_path()

        # 变化量: 正值代表时间缩短 (好)，负值代表时间增加 (坏)
        cft_diff = self.last_global_cft - current_global_cft

        # 归一化与缩放
        step_reward = (cft_diff / len(self.vehicles)) * Cfg.REWARD_SCALE

        # 更新基准
        self.last_global_cft = current_global_cft

        # 广播全作奖励 (Cooperative Reward)
        rewards = [step_reward] * len(self.vehicles)

        # 终止条件 (由外部 Max Steps 控制，此处默认 False)
        done = False
        # 如果所有 DAG 完成，也可以 done=True
        if all(v.task_dag.is_finished for v in self.vehicles):
            done = True

        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        """
        [观测生成]
        生成包含 DAG 状态、物理环境、邻居信息及 Mask 的综合观测。

        修改记录:
        - [关键修正] node_feats 维度修正为 7，严格对齐 GraphBuilder 和 TrainConfig。
        - [特征顺序] [comp, data, status, in_degree, out_degree, t_rem, urgency]
        - [容错处理] 增加了 out_degree 的现场计算，防止 DAGTask 类未更新导致的 AttributeError。
        """
        obs_list = []

        # RSU 全局负载归一化
        rsu_wait_time = (self.rsu_queue_curr * Cfg.MEAN_DATA_SIZE) / Cfg.F_RSU
        rsu_load_norm = np.clip(rsu_wait_time / Cfg.NORM_MAX_WAIT_TIME, 0.0, 1.0)

        # RSU 是否满载 (Mask 用)
        rsu_is_full = self.rsu_queue_curr >= Cfg.RSU_QUEUE_LIMIT

        for v in self.vehicles:
            # =========================================================
            # 1. DAG Node Features (节点特征) -> 目标维度: [Num_Nodes, 7]
            # =========================================================
            num_nodes = v.task_dag.num_subtasks

            # A. 时间相关特征 (标量 -> 向量广播)
            elapsed = self.time - v.task_dag.start_time
            t_rem = v.task_dag.deadline - elapsed
            t_total = v.task_dag.deadline if v.task_dag.deadline > 0 else 1.0

            # 标量计算
            val_t_rem = np.clip(t_rem, -10.0, 10.0)
            val_urgency = np.clip(t_rem / t_total, 0.0, 1.0) if t_rem > 0 else 0.0

            # 广播为数组 [N]
            feat_t_rem = np.full(num_nodes, val_t_rem)
            feat_urgency = np.full(num_nodes, val_urgency)

            # B. 拓扑特征 (数组)
            # 入度 (In-Degree)
            feat_in_degree = v.task_dag.in_degree

            # [关键] 出度 (Out-Degree)
            # 容错逻辑: 如果 DAGTask 对象还没更新 out_degree 属性，则现场计算
            if hasattr(v.task_dag, 'out_degree'):
                feat_out_degree = v.task_dag.out_degree
            else:
                # axis=1 行和 = 出度
                feat_out_degree = np.sum(v.task_dag.adj, axis=1)

            # C. 特征堆叠 (Stacking)
            # 必须与 TrainConfig.TASK_INPUT_DIM = 7 严格一致
            node_feats = np.stack([
                v.task_dag.rem_comp / Cfg.NORM_MAX_COMP,  # 1. 剩余计算量
                v.task_dag.rem_data / Cfg.NORM_MAX_DATA,  # 2. 剩余数据量
                v.task_dag.status,  # 3. 状态
                feat_in_degree,  # 4. 入度
                feat_out_degree,  # 5. 出度 [新增/修复]
                feat_t_rem,  # 6. 剩余时间
                feat_urgency  # 7. 紧迫度
            ], axis=1)

            # =========================================================
            # 2. Self Info (自身状态) -> [6 Dim]
            # =========================================================
            # 预估 V2I 速率用于观测
            dist_rsu = np.linalg.norm(v.pos - Cfg.RSU_POS)
            est_v2i_rate = self._simulate_channel_rate(dist_rsu, 'V2I')

            # 本地等待时间
            self_wait = (v.task_queue_len * Cfg.MEAN_DATA_SIZE) / v.cpu_freq

            self_info = np.array([
                v.vel[0] / Cfg.MAX_VELOCITY, v.vel[1] / Cfg.MAX_VELOCITY,  # 归一化速度
                np.clip(self_wait / Cfg.NORM_MAX_WAIT_TIME, 0, 1),  # 本地负载
                v.cpu_freq / Cfg.NORM_MAX_CPU,  # 本地算力
                np.clip(est_v2i_rate / Cfg.NORM_MAX_RATE_V2I, 0, 1),  # V2I 信道质量
                v.pos[0] / Cfg.MAP_SIZE, v.pos[1] / Cfg.MAP_SIZE  # 绝对位置
            ], dtype=np.float32)

            # =========================================================
            # 3. Neighbor Info (邻居信息)
            # =========================================================
            neighbors = []
            # valid_targets_base: [Local, RSU, Neighbor_1, Neighbor_2, ...]
            # 1 表示物理可达 (距离范围内)，0 表示不可达
            valid_targets_base = [1, 1]

            for other in self.vehicles:
                if v.id == other.id: continue

                dist = np.linalg.norm(v.pos - other.pos)
                if dist <= Cfg.V2V_RANGE:
                    # 相对位置 & 速度
                    rel_pos = (other.pos - v.pos) / Cfg.V2V_RANGE

                    # 邻居负载
                    n_wait = (other.task_queue_len * Cfg.MEAN_DATA_SIZE) / other.cpu_freq

                    # 邻居 V2V 速率估算
                    est_v2v_rate = self._simulate_channel_rate(dist, 'V2V')

                    neighbors.append([
                        other.id,
                        rel_pos[0], rel_pos[1],
                        other.vel[0], other.vel[1],
                        np.clip(n_wait / Cfg.NORM_MAX_WAIT_TIME, 0, 1),
                        other.cpu_freq / Cfg.NORM_MAX_CPU, # 这里已正确使用邻居的个体频率
                        np.clip(est_v2v_rate / Cfg.NORM_MAX_RATE_V2V, 0, 1)
                    ])
                    valid_targets_base.append(1)  # 可达
                else:
                    valid_targets_base.append(0)  # 不可达

            neighbors = np.array(neighbors) if neighbors else np.zeros((0, 8))

            # =========================================================
            # 4. Mask Generation (不合法动作屏蔽)
            # =========================================================
            # 最终 Mask 矩阵: [Num_Subtasks, Num_Targets]
            num_targets = len(valid_targets_base)

            # A. Task Mask (DAG 依赖): 入度 > 0 或已完成的任务不能被调度
            task_schedulable = v.task_dag.get_action_mask()  # 返回 [Num_Subtasks] bool 数组

            # B. Target Mask (物理约束)
            # 基础物理连接掩码 (距离范围)
            target_mask_row = np.array(valid_targets_base, dtype=bool)

            # [Congestion Control] RSU 拥堵
            if rsu_is_full:
                target_mask_row[1] = False  # 禁止卸载到 RSU

            # [Mobility Management] 移动性约束 (Time-to-Leave)
            # 如果 (离开 RSU 范围时间) < (预计传输时间)，则屏蔽 RSU
            speed = np.linalg.norm(v.vel)
            if speed > 0.1:
                time_to_leave = (Cfg.RSU_RANGE - dist_rsu) / speed
                avg_data_size = np.mean(v.task_dag.rem_data)  # 估算数据量
                est_trans_time = avg_data_size / max(est_v2i_rate, 1e-6)

                if time_to_leave < est_trans_time:
                    target_mask_row[1] = False  # 还没传完就跑出去了，禁止动作

            # [Congestion Control] 邻居拥堵
            # 遍历 valid_targets_base 从 index 2 开始 (Neighbors)
            # 这里的逻辑是: valid_targets_base 的顺序对应 self.vehicles 列表(跳过自己)
            candidate_vehs = [x for x in self.vehicles if x.id != v.id]

            for i in range(2, num_targets):
                if valid_targets_base[i] == 1:
                    # 映射回 candidate_vehs 的索引
                    target_veh_idx = i - 2
                    if target_veh_idx < len(candidate_vehs):
                        n_veh = candidate_vehs[target_veh_idx]
                        # 如果邻居队列满了，Mask 掉
                        if n_veh.task_queue_len >= Cfg.VEHICLE_QUEUE_LIMIT:
                            target_mask_row[i] = False

            # 组合 Mask
            # 形状: [Num_Subtasks, Num_Targets]
            final_mask = np.tile(target_mask_row, (num_nodes, 1))

            # 应用 Task Mask (行屏蔽)
            for t_idx in range(num_nodes):
                if not task_schedulable[t_idx]:
                    final_mask[t_idx, :] = False  # 该任务不可被调度

            obs_list.append({
                'node_x': node_feats,
                'self_info': self_info,
                'rsu_info': [rsu_load_norm],
                'adj': v.task_dag.adj,
                'neighbors': neighbors,
                'task_mask': task_schedulable,
                'target_mask': final_mask
            })

        return obs_list

    def _simulate_channel_rate(self, distance, link_type='V2I'):
        """
        [辅助方法] 模拟信道速率 (Pre-Action Sensing)
        用于 Observation 中给 Agent 提供当前的参考速率，含 Config 参数。
        """
        if link_type == 'V2I':
            B = Cfg.BW_V2I
            P_tx = Cfg.TX_POWER_MAX_DBM
            alpha = Cfg.ALPHA_V2I
            std_dev = 3.0
        else:
            B = Cfg.BW_V2V
            P_tx = 20.0
            alpha = Cfg.ALPHA_V2V
            std_dev = 4.0

        noise_dbm = Cfg.NOISE_POWER_DBM
        d = max(distance, 5.0)

        # Log-distance Path Loss
        pl_db = 30.0 + 10.0 * alpha * np.log10(d)
        shadow = np.random.normal(0, std_dev)  # 阴影衰落

        snr_db = P_tx - pl_db + shadow - noise_dbm
        snr_linear = 10 ** (snr_db / 10.0)

        return B * np.log2(1 + snr_linear)

    def _calculate_global_cft_critical_path(self):
        """
        [关键逻辑] 计算全局 CFT (Calculated Finish Time)

        区别于简单的求和，这里使用简化的关键路径 (Critical Path) 估算，
        以体现 DAG 的并行性和依赖约束 (Dependency-Aware)。

        逻辑:
        1. 对于每个任务节点，计算其 (传输时间 + 计算时间)。
        2. 结合 DAG 拓扑，计算到达 End 节点的最长路径时间。
        """
        total_cft = 0.0

        for v in self.vehicles:
            if v.task_dag.is_failed:
                total_cft += (self.time + 1000.0)  # 失败惩罚
                continue

            # 如果已经完成
            if v.task_dag.is_finished:
                total_cft += self.time  # 已完成任务贡献当前时间
                continue

            # --- DAG 关键路径估算 ---
            # 这是一个简化版，假设当前分配的目标不变

            # 1. 获取所有子任务的剩余工作量
            rem_comps = v.task_dag.rem_comp
            rem_datas = v.task_dag.rem_data

            # 2. 估算每个节点的执行时间 (Execution Time)
            # exec_time[i] = trans_time + comp_time
            exec_times = np.zeros(v.task_dag.num_subtasks)

            # 获取当前所有任务的目标速率 (快照)
            # 注意: 这里为了效率，使用模拟速率或最近一次的真实速率
            # 为了奖励的准确性，最好使用 Config 的标称值进行估算

            tgt = v.curr_target
            # 计算每个任务的执行时间时，使用车辆的 CPU 频率
            if tgt == 'RSU':
                proc_speed = Cfg.F_RSU  # RSU 使用固定算力
                comm_speed = self.channel.compute_one_rate(v, Cfg.RSU_POS, 'V2I')
            elif isinstance(tgt, int):
                target_v = self.vehicles[tgt]
                proc_speed = target_v.cpu_freq  # 修改：使用目标的 CPU 频率，而不是发起者的
                comm_speed = self.channel.compute_one_rate(v, target_v.pos, 'V2V')
            else:  # Local
                proc_speed = v.cpu_freq  # 使用车辆的 CPU 频率
                comm_speed = 1e12  # 本地总线极快，忽略

            comm_speed = max(comm_speed, 1e-6)

            for i in range(v.task_dag.num_subtasks):
                # 仅计算未完成的任务
                if v.task_dag.status[i] != 3:  # 3 is DONE
                    t_trans = rem_datas[i] / comm_speed
                    t_comp = rem_comps[i] / proc_speed
                    exec_times[i] = t_trans + t_comp
                else:
                    exec_times[i] = 0.0

            # 3. 计算关键路径 (最长路径)
            # 简单的动态规划: Earliest Finish Time (EFT)
            num_tasks = v.task_dag.num_subtasks
            eft = np.zeros(num_tasks)

            # 假设 adj 是邻接矩阵 [i, j] = 1 代表 i -> j
            adj = v.task_dag.adj

            # 拓扑排序 (简化: 假设 index 顺序大致符合拓扑，或者多轮迭代)
            # 为了严谨，应该做一次 BFS/DFS，但这里用多轮松弛模拟
            for _ in range(num_tasks):
                for j in range(num_tasks):
                    max_prev_eft = 0.0
                    # 找所有前驱 i
                    predecessors = np.where(adj[:, j] == 1)[0]
                    if len(predecessors) > 0:
                        max_prev_eft = np.max(eft[predecessors])

                    eft[j] = max_prev_eft + exec_times[j]

            # 该 DAG 的预计完成时间 = 最后一个节点的 EFT
            dag_est_finish_time = np.max(eft)

            total_cft += (self.time + dag_est_finish_time)

        return total_cft