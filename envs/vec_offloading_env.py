import gymnasium as gym
import numpy as np
from configs.config import SystemConfig as Cfg
from envs.modules.channel import ChannelModel
from envs.modules.queue_system import FIFOQueue
from envs.entities.vehicle import Vehicle
from envs.entities.rsu import RSU
from envs.entities.task_dag import DAGTask
from utils.dag_generator import DAGGenerator


class VecOffloadingEnv(gym.Env):
    """车联网边缘计算任务卸载环境 (Gymnasium接口)

    核心特性:
    - 多车辆协同任务卸载: 每个车辆作为独立智能体，拥有独立DAG任务
    - 任务依赖感知: 子任务间存在数据依赖关系，需考虑传输时间
    - 动态V2V通信: 车辆间通信受距离和信道条件影响
    - RSU边缘计算: 可选择将任务卸载到RSU服务器处理

    状态空间 (Observation Space):
    - node_x: 子任务特征 (计算量、剩余数据、状态、入度、出度、剩余时间、紧急度)
    - self_info: 自身状态 (速度、等待时间、CPU频率、信道质量、位置)
    - rsu_info: RSU负载信息
    - adj: DAG邻接矩阵 (任务依赖关系)
    - neighbors: 邻居车辆特征 (固定维度填充)
    - task_mask: 可调度任务掩码 (用于Critic)
    - target_mask: 动作目标掩码 (Local/RSU/V2V)，仅对应当前选中的任务
    - subtask_index: 当前环境自动选择的任务索引 (标量)

    动作空间 (Action Space):
    - target: 卸载目标 (0=Local, 1=RSU, 2+k=Vehicle k)
    - subtask: 要调度的子任务索引
    - power: 传输功率控制

    奖励设计:
    - CFT奖励: 任务完成时间越短越好
    - 拥堵惩罚: V2V/V2I信道拥塞时产生惩罚
    """

    def __init__(self):
        self.channel = ChannelModel()
        self.dag_gen = DAGGenerator()
        self.vehicles = []
        self.time = 0.0
        
        # RSU实体列表（道路模型：等间距线性部署）
        self.rsus = []
        self._init_rsus()
        
        # CFT计算缓存
        self.last_global_cft = 0.0
        self._cft_cache = None
        self._cft_cache_time = 0.0
        self._cft_cache_valid = False
        self._cft_state_hash = None
        
        # 通信速率和距离缓存（用于性能优化）
        self._comm_rate_cache = {}
        self._cache_time_step = -1.0
        self._dist_matrix_cache = None
        self._dist_matrix_time = -1.0
        self._rsu_dist_cache = {}

        # 归一化常数（预先计算倒数以提高性能）
        self._inv_map_size = 1.0 / Cfg.MAP_SIZE
        self._inv_max_nodes = 1.0 / Cfg.MAX_NODES
        self._inv_max_cpu = 1.0 / Cfg.NORM_MAX_CPU
        self._inv_max_comp = 1.0 / Cfg.NORM_MAX_COMP
        self._inv_max_data = 1.0 / Cfg.NORM_MAX_DATA
        self._inv_max_wait = 1.0 / Cfg.NORM_MAX_WAIT_TIME
        self._inv_max_rate_v2i = 1.0 / Cfg.NORM_MAX_RATE_V2I
        self._inv_max_rate_v2v = 1.0 / Cfg.NORM_MAX_RATE_V2V
        self._inv_max_velocity = 1.0 / Cfg.MAX_VELOCITY
        self._inv_v2v_range = 1.0 / Cfg.V2V_RANGE
        self._mean_comp_load = Cfg.MEAN_COMP_LOAD
    
    def _init_rsus(self):
        """
        初始化RSU实体列表（道路模型：等间距线性部署）
        
        RSU部署公式：
        - Pos_RSU_j = ((j-1)*D_inter + D_inter/2, Y_RSU)
        - Y_RSU = ROAD_WIDTH + RSU_Y_DIST
        - D_inter = MAP_SIZE / NUM_RSU
        - 断言：NUM_RSU * D_inter >= MAP_SIZE 以确保全覆盖
        """
        num_rsu = getattr(Cfg, 'NUM_RSU', 1)
        self.rsus = []
        
        # 计算道路宽度和RSU Y坐标
        road_width = getattr(Cfg, 'NUM_LANES', 3) * getattr(Cfg, 'LANE_WIDTH', 3.5)
        rsu_y_dist = getattr(Cfg, 'RSU_Y_DIST', 20.0)
        y_rsu = road_width + rsu_y_dist  # RSU的Y坐标（固定值）
        
        # 计算RSU部署间距
        map_size = Cfg.MAP_SIZE
        d_inter = map_size / num_rsu  # 等间距部署
        
        # 断言：确保全覆盖
        assert num_rsu * d_inter >= map_size, \
            f"RSU部署间距不足：{num_rsu} * {d_inter} = {num_rsu * d_inter} < {map_size}"
        
        # 验证部署间距满足覆盖约束
        rsu_range = Cfg.RSU_RANGE
        max_d_inter = 2 * np.sqrt(rsu_range**2 - y_rsu**2) * 0.9  # 保留10%重叠
        if d_inter > max_d_inter:
            import warnings
            warnings.warn(
                f"RSU部署间距 {d_inter:.2f}m 超过推荐值 {max_d_inter:.2f}m，"
                f"可能导致覆盖盲区。建议增加RSU数量或增大覆盖半径。"
            )
        
        for i in range(num_rsu):
            # RSU位置公式：Pos_RSU_j = ((j-1)*D_inter + D_inter/2, Y_RSU)
            # j从1开始，但索引从0开始，所以j = i+1
            x_pos = (i * d_inter) + (d_inter / 2)
            pos = np.array([x_pos, y_rsu])
            
            rsu = RSU(
                rsu_id=i,
                position=pos,
                cpu_freq=Cfg.F_RSU,
                num_processors=getattr(Cfg, 'RSU_NUM_PROCESSORS', 4),
                queue_limit=Cfg.RSU_QUEUE_LIMIT,
                coverage_range=rsu_range
            )
            self.rsus.append(rsu)
    
    def _get_nearest_rsu(self, position):
        """
        获取距离指定位置最近的RSU
        
        Args:
            position: 位置坐标 [x, y]
        
        Returns:
            RSU or None: 最近的RSU，如果不在任何RSU覆盖范围内返回None
        """
        if len(self.rsus) == 0:
            return None
        
        nearest_rsu = None
        min_dist = float('inf')
        
        for rsu in self.rsus:
            if rsu.is_in_coverage(position):
                dist = rsu.get_distance(position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_rsu = rsu
        
        return nearest_rsu
    
    def _get_all_rsus_in_range(self, position):
        """
        获取覆盖范围内所有RSU
        
        Args:
            position: 位置坐标 [x, y]
        
        Returns:
            list: 覆盖范围内的RSU列表
        """
        return [rsu for rsu in self.rsus if rsu.is_in_coverage(position)]
    
    def _is_rsu_location(self, loc):
        """
        判断位置标识是否是RSU
        
        Args:
            loc: 位置标识（可能是'RSU'、('RSU', rsu_id)或其他）
        
        Returns:
            bool: 如果是RSU位置返回True
        """
        return loc == 'RSU' or (isinstance(loc, tuple) and loc[0] == 'RSU')
    
    def _get_rsu_id_from_location(self, loc):
        """
        从位置标识中提取RSU ID
        
        Args:
            loc: 位置标识
        
        Returns:
            int or None: RSU ID，如果不是RSU则返回None
        """
        if loc == 'RSU':
            # 兼容旧代码：单个RSU场景，返回第一个RSU的ID
            return 0 if len(self.rsus) > 0 else None
        elif isinstance(loc, tuple) and loc[0] == 'RSU':
            return loc[1]
        return None
    
    def _get_rsu_position(self, rsu_id):
        """
        获取RSU的位置
        
        Args:
            rsu_id: RSU ID
        
        Returns:
            np.array or None: RSU位置，如果ID无效返回None
        """
        if 0 <= rsu_id < len(self.rsus):
            return self.rsus[rsu_id].position
        return None

    def reset(self, seed=None, options=None):
        if seed:
            np.random.seed(seed)

        self.vehicles = []
        self.time = 0.0
        self.steps = 0
        
        # 重置RSU队列和FAT
        for rsu in self.rsus:
            rsu.clear_queue()
            rsu.reset_fat()
        self._comm_rate_cache = {}
        self._cache_time_step = -1.0
        self._dist_matrix_cache = None
        self._dist_matrix_time = -1.0
        self._rsu_dist_cache = {}
        if abs(self.time - self._cft_cache_time) > Cfg.DT * 0.5:
            self._cft_cache = None
            self._cft_cache_valid = False

        for i in range(Cfg.NUM_VEHICLES):
            # 车辆初始位置：在前30%道路上随机分布
            # X坐标：随机在[0, 0.3*MAP_SIZE]范围内
            # Y坐标：随机选择车道中心（道路模型：3条车道）
            x_pos = np.random.uniform(0, 0.3 * Cfg.MAP_SIZE)
            lane_centers = [(k + 0.5) * Cfg.LANE_WIDTH for k in range(Cfg.NUM_LANES)]
            y_pos = np.random.choice(lane_centers)
            pos = np.array([x_pos, y_pos])
            
            v = Vehicle(i, pos)
            v.cpu_freq = np.random.uniform(Cfg.MIN_VEHICLE_CPU_FREQ, Cfg.MAX_VEHICLE_CPU_FREQ)
            v.tx_power_dbm = Cfg.TX_POWER_DEFAULT_DBM if hasattr(Cfg, 'TX_POWER_DEFAULT_DBM') else Cfg.TX_POWER_MIN_DBM

            n_node = np.random.randint(Cfg.MIN_NODES, Cfg.MAX_NODES + 1)
            adj, prof, data, ddl = self.dag_gen.generate(n_node, veh_f=v.cpu_freq)
            v.task_dag = DAGTask(0, adj, prof, data, ddl)
            v.task_dag.start_time = 0.0
            v.task_queue.clear()  # 清空队列
            v.task_queue_len = 0  # 同步队列长度
            
            # 道路模型：速度已在Vehicle.__init__中设置（截断正态分布，沿X轴正方向）
            # 这里不需要重新设置速度，Vehicle类会自动处理
            
            v.last_scheduled_subtask = -1
            v.exec_locations = [None] * v.task_dag.num_subtasks

            self.vehicles.append(v)
        
        # 道路模型：初始化动态车辆生成的下一辆到达时间（泊松过程）
        # 如果VEHICLE_ARRIVAL_RATE > 0，则启用动态生成
        if hasattr(Cfg, 'VEHICLE_ARRIVAL_RATE') and Cfg.VEHICLE_ARRIVAL_RATE > 0:
            # 下一辆车的到达时间间隔服从指数分布：Δt ~ Exponential(λ)
            # 初始下一辆到达时间：从当前时间开始的第一个到达时间
            self._next_vehicle_arrival_time = np.random.exponential(1.0 / Cfg.VEHICLE_ARRIVAL_RATE)
            self._next_vehicle_id = Cfg.NUM_VEHICLES  # 车辆ID从初始数量开始
        else:
            self._next_vehicle_arrival_time = float('inf')  # 禁用动态生成
            self._next_vehicle_id = Cfg.NUM_VEHICLES

        self.last_global_cft = self._calculate_global_cft_critical_path()
        return self._get_obs(), {}

    def step(self, actions):
        self.steps += 1

        if abs(self.time - self._cache_time_step) > 1e-6:
            self._comm_rate_cache.clear()
            self._cache_time_step = self.time

        self._cft_cache = None
        self._cft_cache_valid = False
        self._dist_matrix_cache = None
        self._rsu_dist_cache.clear()

        step_congestion_cost = 0.0
        active_agents_count = 0

        for i, v in enumerate(self.vehicles):
            act = actions[i]
            if not act: continue

            target_idx = int(act['target'])
            # [新设计] 环境自动选择子任务，不再从action中读取subtask
            # subtask_idx = int(act['subtask'])  # 已废弃
            # 使用优先级算法自动选择最高优先级的就绪任务
            subtask_idx = v.task_dag.get_top_priority_task()
            if subtask_idx is None:
                # 没有可调度的任务，跳过
                continue

            if target_idx == 0:
                target = 'Local'
            elif target_idx == 1:
                target = 'RSU'
            else:
                neighbor_list_idx = target_idx - 2
                neighbor_id_map = v.neighbor_id_map if hasattr(v, 'neighbor_id_map') else list(range(len(self.vehicles)))
                if 0 <= neighbor_list_idx < len(neighbor_id_map):
                    actual_neighbor_id = neighbor_id_map[neighbor_list_idx]
                    target = actual_neighbor_id if 0 <= actual_neighbor_id < len(self.vehicles) else 'Local'
                else:
                    target = 'Local'

            p_norm = np.clip(act.get('power', 1.0), 0.0, 1.0)
            raw_power = Cfg.TX_POWER_MIN_DBM + p_norm * (Cfg.TX_POWER_MAX_DBM - Cfg.TX_POWER_MIN_DBM)
            v.tx_power_dbm = np.clip(raw_power, Cfg.TX_POWER_MIN_DBM, Cfg.TX_POWER_MAX_DBM)

            # [新设计] 不再需要检查last_scheduled_subtask，因为环境自动选择任务
            # 如果subtask_idx有效，则进行调度
            if subtask_idx is not None and subtask_idx >= 0:
                active_agents_count += 1
                actual_target = target

                # 使用FIFO队列进行溢出检查
                if target == 'RSU':
                    # 选择最近的RSU或负载最低的RSU
                    rsu_in_range = self._get_all_rsus_in_range(v.pos)
                    if len(rsu_in_range) == 0:
                        # 不在任何RSU覆盖范围内，回退到Local
                        actual_target = 'Local'
                    else:
                        # 选择负载最低的RSU（等待时间最小）
                        # 获取当前任务的计算量（如果可用）
                        task_comp = v.task_dag.total_comp[subtask_idx] if subtask_idx < len(v.task_dag.total_comp) else Cfg.MEAN_COMP_LOAD
                        min_wait = float('inf')
                        for rsu in rsu_in_range:
                            if not rsu.is_queue_full(new_task_cycles=task_comp):
                                wait = rsu.get_estimated_wait_time()
                                if wait < min_wait:
                                    min_wait = wait
                                    selected_rsu = rsu
                        
                        if selected_rsu is None:
                            # 所有RSU队列都满，回退到Local
                            actual_target = 'Local'
                        else:
                            actual_target = ('RSU', selected_rsu.id)  # 使用元组标识RSU
                elif isinstance(target, int):
                    t_veh = self.vehicles[target]
                    # 获取当前任务的计算量（如果可用）
                    task_comp = v.task_dag.total_comp[subtask_idx] if subtask_idx < len(v.task_dag.total_comp) else Cfg.MEAN_COMP_LOAD
                    if t_veh.is_queue_full(new_task_cycles=task_comp):
                        actual_target = 'Local'

                # 计算等待时间（使用FIFO队列的精确计算）
                wait_time = 0.0
                if isinstance(actual_target, tuple) and actual_target[0] == 'RSU':
                    # 多RSU场景：使用选定RSU的等待时间
                    rsu_id = actual_target[1]
                    if 0 <= rsu_id < len(self.rsus):
                        wait_time = self.rsus[rsu_id].get_estimated_wait_time()
                elif actual_target == 'Local':
                    wait_time = v.task_queue.get_estimated_wait_time(v.cpu_freq)
                elif isinstance(actual_target, int):
                    t_veh = self.vehicles[actual_target]
                    wait_time = t_veh.task_queue.get_estimated_wait_time(t_veh.cpu_freq)

                step_congestion_cost += wait_time

                v.curr_target = actual_target
                v.curr_subtask = subtask_idx
                # assign_task现在返回bool表示是否成功分配（防止重复调度）
                assign_success = v.task_dag.assign_task(subtask_idx, actual_target)
                if assign_success:
                    v.last_scheduled_subtask = subtask_idx
                    # 更新exec_locations记录任务执行位置（用于CFT计算）
                    # actual_target已经是正确的格式（'Local'、('RSU', rsu_id)或int vehicle_id）
                    if hasattr(v, 'exec_locations') and 0 <= subtask_idx < len(v.exec_locations):
                        v.exec_locations[subtask_idx] = actual_target
                else:
                    # 如果分配失败（任务已分配过或状态不允许），跳过队列更新
                    continue

                # 获取任务计算量并加入FIFO队列
                task_comp = v.task_dag.total_comp[subtask_idx]
                
                if isinstance(actual_target, tuple) and actual_target[0] == 'RSU':
                    # 多RSU场景：将任务加入选定RSU的队列
                    rsu_id = actual_target[1]
                    if 0 <= rsu_id < len(self.rsus):
                        rsu = self.rsus[rsu_id]
                        processor_id = rsu.enqueue_task(task_comp)
                        if processor_id is None:
                            # RSU队列满，回退到Local
                            actual_target = 'Local'
                            v.curr_target = 'Local'
                            if not v.is_queue_full(new_task_cycles=task_comp):
                                v.task_queue.enqueue(task_comp)
                elif isinstance(actual_target, int):
                    t_veh = self.vehicles[actual_target]
                    # 目标车辆队列
                    t_veh.task_queue.enqueue(task_comp)
                elif actual_target == 'Local':
                    # 本地队列
                    v.task_queue.enqueue(task_comp)
                
        # 队列长度同步（由FIFO队列管理，在所有任务分配完成后统一同步）
        for v_check in self.vehicles:
            v_check.update_queue_sync()
        
        # 计算V2I通信速率（对每个RSU计算）
        rates_v2i_dict = {}  # {(vehicle_id, rsu_id): rate}
        for rsu in self.rsus:
            rates = self.channel.compute_rates(self.vehicles, rsu.position)
            for veh_id, rate in rates.items():
                rates_v2i_dict[(veh_id, rsu.id)] = rate

        for v in self.vehicles:
            c_spd = 0.0
            comp_spd = 0.0
            tgt = v.curr_target

            if tgt == 'Local':
                comp_spd = v.cpu_freq
                c_spd = 1e12
            elif isinstance(tgt, tuple) and tgt[0] == 'RSU':
                # 多RSU场景：使用选定的RSU
                rsu_id = tgt[1]
                if 0 <= rsu_id < len(self.rsus):
                    rsu = self.rsus[rsu_id]
                    comp_spd = rsu.cpu_freq
                    c_spd = rates_v2i_dict.get((v.id, rsu_id), 1e-6)
            elif isinstance(tgt, int):
                target_veh = self.vehicles[tgt]
                comp_spd = target_veh.cpu_freq
                c_spd = self.channel.compute_one_rate(v, target_veh.pos, 'V2V', self.time)

            c_spd = max(c_spd, 1e-6)

            task_finished = False
            if v.curr_subtask is not None:
                task_finished = v.task_dag.step_progress(v.curr_subtask, comp_spd, c_spd, Cfg.DT)

                if task_finished:
                    # [新增] 子任务成功奖励：任何任务完成都给奖励（全域激励）
                    # 包括Local、RSU、V2V，避免智能体歧视Local任务导致依赖链阻塞
                    if not hasattr(v, 'subtask_reward_buffer'):
                        v.subtask_reward_buffer = 0.0
                    
                    # 无论在哪里执行，完成就给奖励
                    v.subtask_reward_buffer += Cfg.SUBTASK_SUCCESS_BONUS
                    
                    # 任务完成时，从队列中移除一个任务（FIFO顺序）
                    if isinstance(tgt, tuple) and tgt[0] == 'RSU':
                        # 多RSU场景：从选定的RSU队列移除
                        rsu_id = tgt[1]
                        if 0 <= rsu_id < len(self.rsus):
                            self.rsus[rsu_id].dequeue_task()
                    elif isinstance(tgt, int):
                        target_veh = self.vehicles[tgt]
                        if target_veh.task_queue.get_queue_length() > 0:
                            target_veh.task_queue.dequeue_one()
                        target_veh.update_queue_sync()
                    elif tgt == 'Local':
                        if v.task_queue.get_queue_length() > 0:
                            v.task_queue.dequeue_one()
                        v.update_queue_sync()

                    completed_task = v.curr_subtask
                    # 使用exec_locations获取parent_loc（如果已设置），否则使用curr_target
                    # 注意：exec_locations在assign_task时已经设置，更可靠
                    if v.exec_locations[completed_task] is not None:
                        parent_loc = v.exec_locations[completed_task]
                    else:
                        parent_loc = v.curr_target
                        v.exec_locations[completed_task] = parent_loc  # 同步更新
                    v.curr_subtask = None

                    for child_task_id in np.where(v.task_dag.adj[completed_task, :] == 1)[0]:
                        transfer_data = v.task_dag.data_matrix[completed_task, child_task_id]
                        if transfer_data > 0:
                            if v.exec_locations[child_task_id] is not None:
                                child_loc = v.exec_locations[child_task_id]
                            elif v.task_dag.task_locations[child_task_id] is not None:
                                child_loc = v.task_dag.task_locations[child_task_id]
                            else:
                                child_loc = None

                            # 如果子任务位置未定且父任务在RSU，默认子任务也在同一RSU
                            if child_loc is None and self._is_rsu_location(parent_loc):
                                child_loc = parent_loc

                            transfer_speed = 0.0
                            # 判断是否在同一位置
                            if self._is_rsu_location(parent_loc) and self._is_rsu_location(child_loc):
                                # 两个都在RSU，检查是否是同一个RSU
                                rsu_id_p = self._get_rsu_id_from_location(parent_loc)
                                rsu_id_c = self._get_rsu_id_from_location(child_loc)
                                same_location = (rsu_id_p is not None and rsu_id_p == rsu_id_c)
                            elif parent_loc == 'Local' and child_loc == 'Local':
                                same_location = True
                            elif isinstance(parent_loc, int) and isinstance(child_loc, int):
                                same_location = (parent_loc == child_loc)
                            else:
                                same_location = False

                            if same_location:
                                transfer_speed = float('inf')
                            elif parent_loc != child_loc:
                                tx_veh = None
                                rx_veh = None
                                tx_pos = None
                                rx_pos = None

                                # 确定发送方位置
                                if parent_loc == 'Local' or self._is_rsu_location(parent_loc):
                                    tx_veh = v
                                    tx_pos = v.pos
                                elif isinstance(parent_loc, int):
                                    tx_veh = self.vehicles[parent_loc]
                                    tx_pos = tx_veh.pos

                                # 确定接收方位置
                                if child_loc == 'Local':
                                    rx_veh = v
                                    rx_pos = v.pos
                                elif self._is_rsu_location(child_loc):
                                    rsu_id = self._get_rsu_id_from_location(child_loc)
                                    rx_pos = self._get_rsu_position(rsu_id)
                                elif isinstance(child_loc, int):
                                    rx_veh = self.vehicles[child_loc]
                                    rx_pos = rx_veh.pos

                                if tx_pos is not None and rx_pos is not None:
                                    if self._is_rsu_location(child_loc):
                                        transfer_speed = self.channel.compute_one_rate(tx_veh, rx_pos, 'V2I', self.time)
                                    else:
                                        transfer_speed = self.channel.compute_one_rate(tx_veh, rx_pos, 'V2V', self.time)

                            # 创建数据传输记录
                            v.active_transfers.append({
                                'child_id': child_task_id,
                                'parent_id': completed_task,
                                'rem_data': transfer_data,
                                'speed': transfer_speed
                            })

            completed_transfers = []
            for transfer in v.active_transfers:
                child_id = transfer['child_id']
                parent_id = transfer['parent_id']

                parent_loc = v.exec_locations[parent_id] if v.exec_locations[parent_id] is not None else v.task_dag.task_locations[parent_id]
                # 获取child_loc：优先使用exec_locations，然后task_locations，最后使用curr_target（如果当前正在执行）
                if v.exec_locations[child_id] is not None:
                    child_loc = v.exec_locations[child_id]
                elif v.task_dag.task_locations[child_id] is not None:
                    child_loc = v.task_dag.task_locations[child_id]
                elif child_id == v.curr_subtask:
                    child_loc = v.curr_target
                else:
                    child_loc = None
                
                # 如果child_loc仍未确定且parent_loc在RSU，默认child_loc也在同一RSU（与创建传输记录时的逻辑一致）
                if child_loc is None and self._is_rsu_location(parent_loc):
                    child_loc = parent_loc

                current_speed = 0.0
                # 判断是否在同一位置（复用辅助函数）
                if self._is_rsu_location(parent_loc) and self._is_rsu_location(child_loc):
                    rsu_id_p = self._get_rsu_id_from_location(parent_loc)
                    rsu_id_c = self._get_rsu_id_from_location(child_loc)
                    same_location = (rsu_id_p is not None and rsu_id_p == rsu_id_c)
                elif parent_loc == 'Local' and child_loc == 'Local':
                    same_location = True
                elif isinstance(parent_loc, int) and isinstance(child_loc, int):
                    same_location = (parent_loc == child_loc)
                else:
                    same_location = False

                if same_location:
                    current_speed = float('inf')
                elif parent_loc != child_loc:
                    tx_veh = None
                    rx_veh = None
                    tx_pos = None
                    rx_pos = None

                    # 确定发送方位置
                    if parent_loc == 'Local' or self._is_rsu_location(parent_loc):
                        tx_veh = v
                        tx_pos = v.pos
                    elif isinstance(parent_loc, int):
                        tx_veh = self.vehicles[parent_loc]
                        tx_pos = tx_veh.pos

                    # 确定接收方位置
                    if child_loc == 'Local':
                        rx_veh = v
                        rx_pos = v.pos
                    elif self._is_rsu_location(child_loc):
                        rsu_id = self._get_rsu_id_from_location(child_loc)
                        rx_pos = self._get_rsu_position(rsu_id)
                    elif isinstance(child_loc, int):
                        rx_veh = self.vehicles[child_loc]
                        rx_pos = rx_veh.pos

                    if tx_pos is not None and rx_pos is not None:
                        if self._is_rsu_location(child_loc):
                            current_speed = self.channel.compute_one_rate(tx_veh, rx_pos, 'V2I', self.time)
                        else:
                            current_speed = self.channel.compute_one_rate(tx_veh, rx_pos, 'V2V', self.time)

                transfer['speed'] = current_speed
                if current_speed == float('inf'):
                    transfer['rem_data'] = 0
                else:
                    transmitted = current_speed * Cfg.DT
                    transfer['rem_data'] = max(0, transfer['rem_data'] - transmitted)

                if transfer['rem_data'] <= 0:
                    completed_transfers.append(transfer)

            for transfer in completed_transfers:
                v.active_transfers.remove(transfer)

                child_id = transfer['child_id']
                parent_id = transfer['parent_id']
                if child_id in v.task_dag.inter_task_transfers:
                    if parent_id in v.task_dag.inter_task_transfers[child_id]:
                        v.task_dag.inter_task_transfers[child_id][parent_id]['rem_data'] = 0

                v.task_dag.step_inter_task_transfers(child_id, transfer['speed'], Cfg.DT)

            # 更新车辆位置（道路模型：一维移动）
            v.update_pos(Cfg.DT, Cfg.MAP_SIZE)
        
        # 移除超出边界的车辆（道路模型：车辆超出道路长度L后移除）
        vehicles_to_remove = []
        for i, v in enumerate(self.vehicles):
            if v.pos[0] >= Cfg.MAP_SIZE:  # 车辆X坐标达到或超过道路长度
                vehicles_to_remove.append(i)
        
        # 从后往前移除，避免索引问题
        for i in reversed(vehicles_to_remove):
            self.vehicles.pop(i)
        
        # 动态生成新车辆（泊松过程）
        if hasattr(self, '_next_vehicle_arrival_time') and hasattr(Cfg, 'VEHICLE_ARRIVAL_RATE') and Cfg.VEHICLE_ARRIVAL_RATE > 0:
            if self.time >= self._next_vehicle_arrival_time:
                # 生成新车辆
                # 位置：X=0（入口），Y=随机车道中心
                x_pos = 0.0
                lane_centers = [(k + 0.5) * Cfg.LANE_WIDTH for k in range(Cfg.NUM_LANES)]
                y_pos = np.random.choice(lane_centers)
                pos = np.array([x_pos, y_pos])
                
                new_vehicle = Vehicle(self._next_vehicle_id, pos)
                new_vehicle.cpu_freq = np.random.uniform(Cfg.MIN_VEHICLE_CPU_FREQ, Cfg.MAX_VEHICLE_CPU_FREQ)
                new_vehicle.tx_power_dbm = Cfg.TX_POWER_DEFAULT_DBM if hasattr(Cfg, 'TX_POWER_DEFAULT_DBM') else Cfg.TX_POWER_MIN_DBM
                
                # 生成DAG任务
                n_node = np.random.randint(Cfg.MIN_NODES, Cfg.MAX_NODES + 1)
                adj, prof, data, ddl = self.dag_gen.generate(n_node, veh_f=new_vehicle.cpu_freq)
                new_vehicle.task_dag = DAGTask(0, adj, prof, data, ddl)
                new_vehicle.task_dag.start_time = self.time
                new_vehicle.task_queue.clear()
                new_vehicle.task_queue_len = 0
                new_vehicle.last_scheduled_subtask = -1
                new_vehicle.exec_locations = [None] * new_vehicle.task_dag.num_subtasks
                
                self.vehicles.append(new_vehicle)
                self._next_vehicle_id += 1
                
                # 计算下一辆车的到达时间
                self._next_vehicle_arrival_time = self.time + np.random.exponential(1.0 / Cfg.VEHICLE_ARRIVAL_RATE)

        self.time += Cfg.DT

        rewards = []
        # 计算每个车辆的CFT（任务完成时间）
        vehicle_cfts = []
        for i, v in enumerate(self.vehicles):
            if v.task_dag.is_finished:
                vehicle_cfts.append(self.time)
            else:
                # 获取任务位置分配
                task_locations = ['Local'] * v.task_dag.num_subtasks
                if hasattr(v, 'exec_locations'):
                    for j in range(v.task_dag.num_subtasks):
                        if v.exec_locations[j] is not None:
                            task_locations[j] = v.exec_locations[j]
                
                # 当前正在处理的子任务
                if v.curr_subtask is not None and 0 <= v.curr_subtask < v.task_dag.num_subtasks:
                    task_locations[v.curr_subtask] = v.curr_target
                
                # 使用time_calculator计算该车辆任务的CFT
                try:
                    from envs.modules.time_calculator import calculate_est_ct
                    EST, CT, CFT = calculate_est_ct(
                        v, v.task_dag, task_locations,
                        self.channel, self.rsus, self.vehicles, self.time
                    )
                    vehicle_cfts.append(CFT)
                except Exception as e:
                    # 如果计算失败，使用旧的全局方法
                    print(f"警告: 计算车辆{i}的CFT失败: {e}")
                    vehicle_cfts.append(self.time + 100.0)  # 使用一个大的默认值
        
        # 保存每个车辆的CFT（用于观测和奖励计算）
        self.vehicle_cfts = vehicle_cfts
        # 全局CFT使用所有车辆的最大值（用于兼容旧代码）
        self.last_global_cft = max(vehicle_cfts) if len(vehicle_cfts) > 0 else self.time
        
        # 计算奖励
        for i, v in enumerate(self.vehicles):
            dag = v.task_dag
            target = v.curr_target if v.curr_subtask is not None else 'Local'
            task_idx = v.curr_subtask if v.curr_subtask is not None else 0
            data_size = dag.data_matrix[v.curr_subtask, :].sum() if v.curr_subtask is not None else 0

            # 获取任务计算量（用于基于计算量的队列限制检查）
            task_comp = dag.total_comp[task_idx] if task_idx is not None and task_idx < len(dag.total_comp) else Cfg.MEAN_COMP_LOAD
            r = self.calculate_agent_reward(i, target, task_idx, data_size, task_comp)
            
            # [新增] 加上本step累积的子任务奖励（计件工资）
            if hasattr(v, 'subtask_reward_buffer'):
                r += v.subtask_reward_buffer
                v.subtask_reward_buffer = 0.0  # 清零，避免重复计算
            
            rewards.append(r)

        all_finished = all(v.task_dag.is_finished for v in self.vehicles)
        time_limit_reached = self.steps >= Cfg.MAX_STEPS
        terminated = all_finished
        truncated = time_limit_reached
        info = {'timeout': time_limit_reached} if time_limit_reached else {}

        return self._get_obs(), rewards, terminated, truncated, info

    def _calculate_global_cft_critical_path(self):
        """
        [关键方法] 计算全局关键路径完成时间 (CFT)

        核心思想:
        - 考虑任务间的依赖关系 (DAG)
        - 考虑数据传输时间 (跨节点执行时)
        - 考虑历史执行位置 (通过 exec_locations 跟踪)

        缓存机制:
        - 如果仿真状态未变 (位置、队列等)，直接返回缓存的 CFT 值
        - 避免重复计算，提高效率
        """
        # 计算RSU队列状态（用于缓存哈希）
        rsu_queue_state = tuple(rsu.queue_length for rsu in self.rsus) if len(self.rsus) > 0 else (0,)
        
        current_state_hash = hash((
            round(self.time, 3),
            rsu_queue_state,
            tuple(round(v.pos[0], 2) for v in self.vehicles),
            tuple(round(v.pos[1], 2) for v in self.vehicles),
            tuple(v.task_queue_len for v in self.vehicles),
            tuple(v.curr_target if hasattr(v, 'curr_target') else None for v in self.vehicles)
        ))

        if (self._cft_cache is not None and
                self._cft_cache_valid and
                hasattr(self, '_cft_state_hash') and
                self._cft_state_hash == current_state_hash):
            return self._cft_cache

        self._cft_state_hash = current_state_hash

        total_cft = 0.0

        for v in self.vehicles:
            if v.task_dag.is_finished:
                total_cft = max(total_cft, self.time)
                continue

            dag = v.task_dag
            num_tasks = dag.num_subtasks
            if num_tasks == 0: continue

            adj = dag.adj
            data_matrix = dag.data_matrix
            rem_comps = dag.rem_comp

            task_locations = ['Local'] * num_tasks

            if hasattr(v, 'exec_locations'):
                for i in range(num_tasks):
                    if v.exec_locations[i] is not None:
                        task_locations[i] = v.exec_locations[i]

            for i in range(num_tasks):
                if task_locations[i] is None:
                    task_locations[i] = 'Local'

            if v.curr_subtask is not None and 0 <= v.curr_subtask < num_tasks:
                task_locations[v.curr_subtask] = v.curr_target

            # 使用FIFO队列计算等待时间
            local_wait = v.task_queue.get_estimated_wait_time(v.cpu_freq)
            # 多RSU场景：使用所有RSU中的最小等待时间
            if len(self.rsus) > 0:
                rsu_wait_global = min([rsu.get_estimated_wait_time() for rsu in self.rsus])
            else:
                rsu_wait_global = 0.0

            node_exec_times = np.zeros(num_tasks)
            cpu_fat = np.zeros(num_tasks)
            channel_fat = np.zeros(num_tasks)

            for i in range(num_tasks):
                if dag.status[i] == 3: continue

                loc = task_locations[i]
                if loc == 'Local':
                    node_exec_times[i] = rem_comps[i] / v.cpu_freq
                    cpu_fat[i] = local_wait
                    channel_fat[i] = 0.0
                elif self._is_rsu_location(loc):
                    # 多RSU场景：使用对应RSU的CPU频率
                    rsu_id = self._get_rsu_id_from_location(loc)
                    if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                        rsu = self.rsus[rsu_id]
                        node_exec_times[i] = rem_comps[i] / rsu.cpu_freq
                        cpu_fat[i] = rsu.get_estimated_wait_time()
                    else:
                        # 向后兼容：使用默认RSU频率
                        node_exec_times[i] = rem_comps[i] / Cfg.F_RSU
                        cpu_fat[i] = rsu_wait_global
                    channel_fat[i] = 0.0
                elif isinstance(loc, int):
                    target_veh = self.vehicles[loc] if loc < len(self.vehicles) else v
                    wait_target = target_veh.task_queue.get_estimated_wait_time(target_veh.cpu_freq)
                    node_exec_times[i] = rem_comps[i] / target_veh.cpu_freq
                    cpu_fat[i] = wait_target
                    channel_fat[i] = 0.0

            earliest_start = np.zeros(num_tasks)

            for i in range(num_tasks):
                if dag.status[i] == 3: continue

                preds = np.where(adj[:, i] == 1)[0]
                max_pred_finish = 0.0

                for p in preds:
                    pred_loc = task_locations[p]
                    curr_loc = task_locations[i]

                    pred_finish = earliest_start[p] + node_exec_times[p]

                    # 判断是否在同一位置（支持RSU元组）
                    if self._is_rsu_location(pred_loc) and self._is_rsu_location(curr_loc):
                        rsu_id_p = self._get_rsu_id_from_location(pred_loc)
                        rsu_id_c = self._get_rsu_id_from_location(curr_loc)
                        same_location = (rsu_id_p is not None and rsu_id_p == rsu_id_c)
                    elif pred_loc == 'Local' and curr_loc == 'Local':
                        same_location = True
                    elif isinstance(pred_loc, int) and isinstance(curr_loc, int):
                        same_location = (pred_loc == curr_loc)
                    else:
                        same_location = False

                    if same_location:
                        data_transfer_time = 0.0
                    else:
                        transfer_data = data_matrix[p, i]
                        if transfer_data <= 1e-9:
                            data_transfer_time = 0.0
                        else:
                            # 获取RSU位置（如果是RSU目标）
                            rsu_pos = Cfg.RSU_POS  # 默认使用配置位置（向后兼容）
                            if self._is_rsu_location(curr_loc):
                                rsu_id = self._get_rsu_id_from_location(curr_loc)
                                if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                                    rsu_pos = self.rsus[rsu_id].position
                            est_rate = self._get_comm_rate(v, p, curr_loc, rsu_pos)
                            data_transfer_time = transfer_data / est_rate

                    max_pred_finish = max(max_pred_finish, pred_finish + data_transfer_time)

                est_i = max(cpu_fat[i], channel_fat[i], max_pred_finish)
                earliest_start[i] = est_i

            max_completion_time = 0.0
            estimated_failed = False

            for i in range(num_tasks):
                if dag.status[i] == 3: continue

                completion_time = earliest_start[i] + node_exec_times[i]
                max_completion_time = max(max_completion_time, completion_time)

            if dag.deadline > 0 and max_completion_time > dag.deadline:
                estimated_failed = True

            total_cft = max(total_cft, max_completion_time)

        self._cft_cache = total_cft
        self._cft_cache_time = self.time
        self._cft_cache_valid = True
        return total_cft

    def _get_dist_matrix(self):
        """计算并缓存所有车辆间的距离矩阵

        优化: 避免在_get_obs中重复计算车辆间距离
        缓存: 同一时间步内复用
        """
        if (self._dist_matrix_cache is not None and
            abs(self.time - self._dist_matrix_time) < 1e-6):
            return self._dist_matrix_cache

        num_vehicles = len(self.vehicles)
        if num_vehicles == 0:
            self._dist_matrix_cache = np.zeros((0, 0))
            self._dist_matrix_time = self.time
            return self._dist_matrix_cache

        positions = np.array([v.pos for v in self.vehicles])
        # 使用numpy广播完全向量化计算距离矩阵
        # positions[:, None, :] 形状 (N, 1, 2)
        # positions[None, :, :] 形状 (1, N, 2)
        # 广播后 diff 形状 (N, N, 2)
        diff = positions[:, None, :] - positions[None, :, :]
        self._dist_matrix_cache = np.linalg.norm(diff, axis=2)  # (N, N)

        self._dist_matrix_time = self.time
        return self._dist_matrix_cache

    def _get_rsu_dist(self, vehicle):
        """获取车辆到最近RSU的距离（使用实际RSU列表）"""
        if vehicle.id in self._rsu_dist_cache:
            return self._rsu_dist_cache[vehicle.id]
        
        # 使用实际部署的RSU列表计算最近距离
        if len(self.rsus) > 0:
            min_dist = float('inf')
            for rsu in self.rsus:
                dist = rsu.get_distance(vehicle.pos)
                if dist < min_dist:
                    min_dist = dist
            dist = min_dist
        else:
            # 向后兼容：没有RSU时使用配置
            dist = np.linalg.norm(vehicle.pos - Cfg.RSU_POS)
        
        self._rsu_dist_cache[vehicle.id] = dist
        return dist

    def _get_obs(self):
        """[关键方法] 生成所有车辆的观测

        观测组成:
        - node_x: 子任务级特征 (DAG属性)
        - self_info: 车辆级特征 (自身状态)
        - rsu_info: 环境级特征 (RSU负载)
        - adj: 图结构 (任务依赖)
        - neighbors: 邻居特征 (V2V通信候选)
        - task_mask: 可调度任务 (READY状态)
        - target_mask: 合法动作 (位置有效、队列未满)

        维度约束:
        - neighbors使用固定维度填充 (MAX_NEIGHBORS, 8)
        - 满足Gymnasium批处理要求
        """
        obs_list = []
        # 多RSU场景：使用所有RSU中的最小等待时间（用于观测）
        # 对于观测空间，使用平均任务大小检查队列是否接近满（保守估计）
        if len(self.rsus) > 0:
            rsu_wait_time = min([rsu.get_estimated_wait_time() for rsu in self.rsus])
            # 使用平均任务大小进行保守检查（如果没有具体任务信息）
            avg_task_comp = Cfg.MEAN_COMP_LOAD
            rsu_is_full = all([rsu.is_queue_full(new_task_cycles=avg_task_comp) for rsu in self.rsus])
        else:
            rsu_wait_time = 0.0
            rsu_is_full = True
        rsu_load_norm = np.clip(rsu_wait_time / Cfg.DYNAMIC_MAX_WAIT_TIME, 0.0, 1.0)

        dist_matrix = self._get_dist_matrix()
        vehicle_ids = [veh.id for veh in self.vehicles]

        for v in self.vehicles:
            v_idx = vehicle_ids.index(v.id)
            num_nodes = v.task_dag.num_subtasks
            elapsed = self.time - v.task_dag.start_time
            t_rem = v.task_dag.deadline - elapsed
            t_total = v.task_dag.deadline if v.task_dag.deadline > 0 else 1.0

            val_t_rem = np.clip(t_rem, -10.0, 10.0)
            val_urgency = np.clip(t_rem / t_total, 0.0, 1.0) if t_rem > 0 else 0.0

            feat_t_rem = np.full(num_nodes, val_t_rem)
            feat_urgency = np.full(num_nodes, val_urgency)
            feat_in_degree = v.task_dag.in_degree * self._inv_max_nodes
            out_degree_arr = v.task_dag.out_degree if hasattr(v.task_dag, 'out_degree') else np.sum(v.task_dag.adj, axis=1)
            feat_out_degree = out_degree_arr * self._inv_max_nodes
            feat_status = v.task_dag.status / 3.0

            node_feats = np.stack([
                v.task_dag.rem_comp * self._inv_max_comp,
                v.task_dag.rem_data * self._inv_max_data,
                feat_status,
                feat_in_degree,
                feat_out_degree,
                (feat_t_rem + 10.0) / 20.0,
                feat_urgency
            ], axis=1)

            # [关键] 固定维度填充 - 适配批处理要求
            # 将node特征填充到固定维度MAX_NODES，确保所有车辆观测形状一致
            MAX_NODES = Cfg.MAX_NODES
            node_dim = 7
            padded_node_feats = np.zeros((MAX_NODES, node_dim), dtype=np.float32)
            padded_node_feats[:num_nodes, :] = node_feats

            dist_rsu = self._get_rsu_dist(v)
            # 多RSU场景：使用最近RSU的位置计算V2I速率
            if len(self.rsus) > 0:
                min_dist = float('inf')
                nearest_rsu_pos = Cfg.RSU_POS
                for rsu in self.rsus:
                    if rsu.is_in_coverage(v.pos):
                        dist = rsu.get_distance(v.pos)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_rsu_pos = rsu.position
                rsu_pos_for_v2i = nearest_rsu_pos
            else:
                rsu_pos_for_v2i = Cfg.RSU_POS
            est_v2i_rate = self.channel.compute_one_rate(v, rsu_pos_for_v2i, 'V2I', curr_time=self.time)
            self_wait = v.task_queue.get_estimated_wait_time(v.cpu_freq)

            self_info = np.array([
                v.vel[0] * self._inv_max_velocity, v.vel[1] * self._inv_max_velocity,
                np.clip(self_wait * self._inv_max_wait, 0, 1),
                v.cpu_freq * self._inv_max_cpu,
                np.clip(est_v2i_rate * self._inv_max_rate_v2i, 0, 1),
                v.pos[0] * self._inv_map_size, v.pos[1] * self._inv_map_size
            ], dtype=np.float32)

            neighbors = []
            valid_targets_base = [1, 1]
            neighbor_id_map = []
            # [关键修复] 构建索引到车辆ID的映射，用于后续队列检查
            target_index_to_veh_id = {}  # {target_mask_index: vehicle_id}

            for j, other in enumerate(self.vehicles):
                target_mask_idx = len(valid_targets_base)  # 当前要添加的索引位置
                dist = dist_matrix[v_idx, j]
                if v.id == other.id:
                    valid_targets_base.append(0)
                    continue

                if dist <= Cfg.V2V_RANGE:
                    rel_pos = (other.pos - v.pos) * self._inv_v2v_range
                    n_wait = other.task_queue.get_estimated_wait_time(other.cpu_freq)
                    est_v2v_rate = self.channel.compute_one_rate(v, other.pos, 'V2V', self.time)

                    neighbors.append([
                        other.id, rel_pos[0], rel_pos[1],
                        other.vel[0], other.vel[1],
                        np.clip(n_wait * self._inv_max_wait, 0, 1),
                        other.cpu_freq * self._inv_max_cpu,
                        np.clip(est_v2v_rate * self._inv_max_rate_v2v, 0, 1)
                    ])
                    valid_targets_base.append(1)
                    neighbor_id_map.append(other.id)
                    target_index_to_veh_id[target_mask_idx] = other.id
                else:
                    valid_targets_base.append(0)
                    target_index_to_veh_id[target_mask_idx] = other.id  # 即使不可达也要记录映射

            actual_num_neighbors = len(neighbor_id_map)
            num_targets = 2 + actual_num_neighbors

            MAX_NEIGHBORS = Cfg.NUM_VEHICLES
            neighbor_dim = 8
            neighbors_array = np.zeros((MAX_NEIGHBORS, neighbor_dim), dtype=np.float32)
            for idx, neighbor_feat in enumerate(neighbors):
                if idx < MAX_NEIGHBORS:
                    neighbors_array[idx] = neighbor_feat

            task_schedulable = v.task_dag.get_action_mask()
            
            # [新设计] 环境自动选择优先级最高的任务
            selected_subtask_idx = v.task_dag.get_top_priority_task()
            if selected_subtask_idx is None:
                # 没有可调度的任务，使用无效索引-1
                selected_subtask_idx = -1
            
            target_mask_row = np.array(valid_targets_base, dtype=bool)

            v.neighbor_id_map = neighbor_id_map.copy()

            # RSU队列满时禁用RSU卸载
            if rsu_is_full:
                target_mask_row[1] = False

            # [关键修复] RSU范围检查 - 车辆不在任何RSU覆盖范围内时禁用RSU
            in_rsu_range = False
            if len(self.rsus) > 0:
                for rsu in self.rsus:
                    if rsu.is_in_coverage(v.pos):
                        in_rsu_range = True
                        break
            else:
                # 向后兼容
                in_rsu_range = (dist_rsu <= Cfg.RSU_RANGE)
            
            if not in_rsu_range:
                target_mask_row[1] = False

            # [关键] V2I离开判断 - 车辆即将离开RSU覆盖范围时禁用RSU
            if in_rsu_range:
                speed = np.linalg.norm(v.vel)
                if speed > 0.1:
                    time_to_leave = (Cfg.RSU_RANGE - dist_rsu) / speed
                    avg_data_size = np.mean(v.task_dag.rem_data)
                    est_trans_time = avg_data_size / max(est_v2i_rate, 1e-6)
                    if time_to_leave < est_trans_time:
                        target_mask_row[1] = False

            # [关键] 目标车辆队列满检查 - 动态禁用过载车辆（基于计算量）
            # 使用平均任务大小进行保守检查
            avg_task_comp = Cfg.MEAN_COMP_LOAD
            for i in range(2, len(valid_targets_base)):
                if i >= len(target_mask_row):
                    break
                if target_mask_row[i]:  # 只检查当前仍可用的目标
                    target_veh_id = target_index_to_veh_id.get(i)
                    if target_veh_id is not None:
                        n_veh = next((veh for veh in self.vehicles if veh.id == target_veh_id), None)
                        if n_veh is None:
                            target_mask_row[i] = False
                            continue
                        
                        # 使用基于计算量的检查（更准确）
                        if n_veh.is_queue_full(new_task_cycles=avg_task_comp):
                            target_mask_row[i] = False
                            continue
                        
                        # [新增] V2V智能断链保护 - 预测任务能否在断开前完成
                        # 计算当前距离（需要通过索引映射找到对应的j）
                        j_idx = None
                        for j, other_veh in enumerate(self.vehicles):
                            if other_veh.id == target_veh_id:
                                j_idx = j
                                break
                        if j_idx is None:
                            target_mask_row[i] = False
                            continue
                        
                        current_dist = dist_matrix[v_idx, j_idx]
                        
                        # 估算任务总耗时（传输 + 排队 + 计算）
                        # 使用当前选中的任务数据量（如果可用），否则使用平均值
                        if selected_subtask_idx >= 0 and selected_subtask_idx < v.task_dag.num_subtasks:
                            task_data_size = v.task_dag.total_data[selected_subtask_idx]
                            task_comp_size = v.task_dag.total_comp[selected_subtask_idx]
                        else:
                            task_data_size = np.mean(v.task_dag.total_data)
                            task_comp_size = avg_task_comp
                        
                        # 传输时间（重新计算V2V速率）
                        est_v2v_rate = self.channel.compute_one_rate(v, n_veh.pos, 'V2V', self.time)
                        est_v2v_rate = max(est_v2v_rate, 1e-6)
                        trans_time = task_data_size / est_v2v_rate
                        
                        # 排队时间（估算）
                        queue_wait_time = n_veh.task_queue.get_estimated_wait_time(n_veh.cpu_freq)
                        
                        # 计算时间
                        comp_time = task_comp_size / max(n_veh.cpu_freq, 1e-6)
                        
                        # 总任务耗时（加安全系数）
                        total_task_time = (trans_time + queue_wait_time + comp_time) * 1.2  # 1.2是安全系数
                        
                        # 估算两车还能连多久（考虑相对速度方向）
                        # 计算相对速度向量
                        rel_vel = n_veh.vel - v.vel
                        # 计算位置差向量（从v指向n_veh）
                        pos_diff = n_veh.pos - v.pos
                        pos_diff_norm = np.linalg.norm(pos_diff)
                        if pos_diff_norm < 1e-6:
                            # 位置相同，使用大值
                            time_to_break = 1000.0
                        else:
                            # 计算相对速度在位置差方向上的投影（标量）
                            # 如果投影为正，两车在远离；如果为负，两车在靠近
                            rel_vel_proj = np.dot(rel_vel, pos_diff) / pos_diff_norm
                            
                            if rel_vel_proj > 0.1:  # 两车在远离
                                # 使用相对速度的投影分量计算断开时间
                                time_to_break = (Cfg.V2V_RANGE - current_dist) / rel_vel_proj
                            elif rel_vel_proj < -0.1:  # 两车在靠近
                                # 两车在靠近，连接时间会更长（设为大值表示不会断开）
                                time_to_break = 1000.0
                            else:  # 相对速度很小或垂直于位置差
                                # 相对速度很小，认为可以连很久
                                time_to_break = 1000.0
                        
                        # 如果任务耗时 > 连接时间，mask掉（注定失败）
                        if total_task_time > time_to_break:
                            target_mask_row[i] = False

            # [关键] 扩展掩码 - 每个子任务对应一行目标掩码
            # 非READY状态的任务对应的整行掩码设为False
            final_mask = np.tile(target_mask_row, (num_nodes, 1))
            for t_idx in range(num_nodes):
                if not task_schedulable[t_idx]:
                    final_mask[t_idx, :] = False

            # [关键] 固定维度填充 - 适配批处理要求
            # 将target_mask填充到固定维度MAX_NODES，确保所有车辆观测形状一致
            MAX_NODES = Cfg.MAX_NODES
            padded_mask = np.zeros((MAX_NODES, target_mask_row.shape[0]), dtype=bool)
            padded_mask[:num_nodes, :] = final_mask

            # [关键] 固定维度填充 - 适配批处理要求
            # 将邻接矩阵填充到固定维度MAX_NODES x MAX_NODES
            padded_adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
            padded_adj[:num_nodes, :num_nodes] = v.task_dag.adj

            # [关键] 固定维度填充 - 适配批处理要求
            # 将task_mask填充到固定维度MAX_NODES（保留用于Critic）
            MAX_NODES = Cfg.MAX_NODES
            padded_task_mask = np.zeros(MAX_NODES, dtype=bool)
            padded_task_mask[:num_nodes] = task_schedulable

            # [新设计] 将target_mask扩展到固定维度MAX_NEIGHBORS（如果实际邻居数小于MAX_NEIGHBORS）
            MAX_NEIGHBORS = Cfg.NUM_VEHICLES
            MAX_TARGETS = 2 + MAX_NEIGHBORS
            padded_target_mask = np.zeros(MAX_TARGETS, dtype=bool)
            # 只复制不超过MAX_TARGETS个元素（防止动态车辆生成导致的维度不匹配）
            copy_len = min(len(target_mask_row), MAX_TARGETS)
            padded_target_mask[:copy_len] = target_mask_row[:copy_len]
            
            # [关键] 死锁兜底：如果所有目标都不可用，强制开启Local
            if not np.any(padded_target_mask):
                padded_target_mask[0] = True  # Local总是可以作为fallback
            
            # [新增] 构建Resource ID列表（用于ID Embedding）
            # ID映射：Local=1, RSU=2, Neighbors=3+vehicle_id
            resource_id_list = np.zeros(MAX_TARGETS, dtype=np.int64)
            resource_id_list[0] = 1  # Local
            resource_id_list[1] = 2  # RSU
            # Neighbors: 使用neighbor_id_map中的实际车辆ID
            for idx, neighbor_veh_id in enumerate(neighbor_id_map):
                if 2 + idx < MAX_TARGETS:
                    resource_id_list[2 + idx] = 3 + neighbor_veh_id
            # Padding位置保持0（稍后会被mask掉）
            
            # [新增] DAG拓扑特征（用于网络特征工程）
            # L_fwd, L_bwd: [MAX_NODES], 前向/后向层级
            padded_L_fwd = np.zeros(MAX_NODES, dtype=np.int32)
            padded_L_bwd = np.zeros(MAX_NODES, dtype=np.int32)
            padded_L_fwd[:num_nodes] = v.task_dag.L_fwd
            padded_L_bwd[:num_nodes] = v.task_dag.L_bwd
            
            # data_matrix: [MAX_NODES, MAX_NODES], 边数据量
            padded_data_matrix = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
            padded_data_matrix[:num_nodes, :num_nodes] = v.task_dag.data_matrix
            
            # Delta: [MAX_NODES, MAX_NODES], 最短路径距离
            padded_Delta = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int32)
            padded_Delta[:num_nodes, :num_nodes] = v.task_dag.Delta
            
            # status: [MAX_NODES], 任务状态（0-3）
            padded_status = np.zeros(MAX_NODES, dtype=np.int32)
            padded_status[:num_nodes] = v.task_dag.status
            
            # location: [MAX_NODES], 任务执行位置编码
            # 0: Unscheduled, 1: Local, 2: RSU, 3+: Neighbor vehicle ID
            padded_location = np.zeros(MAX_NODES, dtype=np.int32)
            for t_idx in range(num_nodes):
                # 优先从v.exec_locations获取（Vehicle属性），其次是v.task_dag.task_locations
                if hasattr(v, 'exec_locations') and v.exec_locations[t_idx] is not None:
                    loc = v.exec_locations[t_idx]
                elif hasattr(v.task_dag, 'task_locations') and v.task_dag.task_locations[t_idx] is not None:
                    loc = v.task_dag.task_locations[t_idx]
                else:
                    loc = None
                
                if loc is None or loc == 'None':
                    padded_location[t_idx] = 0  # Unscheduled
                elif loc == 'Local':
                    padded_location[t_idx] = 1
                elif self._is_rsu_location(loc):
                    padded_location[t_idx] = 2
                elif isinstance(loc, int):
                    padded_location[t_idx] = 3 + loc  # Neighbor vehicle ID
                else:
                    padded_location[t_idx] = 0

            obs_list.append({
                'node_x': padded_node_feats,
                'self_info': self_info,
                'rsu_info': [rsu_load_norm],
                'adj': padded_adj,
                'neighbors': neighbors_array,
                'task_mask': padded_task_mask,
                'target_mask': padded_target_mask,  # [新设计] 简化为[2+MAX_NEIGHBORS]
                'action_mask': padded_target_mask.copy(),  # [新增] Actor专用动作掩码
                'resource_ids': resource_id_list,  # [新增] 资源节点ID列表
                'subtask_index': int(selected_subtask_idx),  # [新设计] 添加当前选中的任务索引
                # [新增] DAG拓扑特征
                'L_fwd': padded_L_fwd,
                'L_bwd': padded_L_bwd,
                'data_matrix': padded_data_matrix,
                'Delta': padded_Delta,
                'status': padded_status,
                'location': padded_location
            })

        return obs_list

    def _get_comm_rate(self, vehicle, pred_task_id, curr_loc, rsu_pos):
        """计算任务间通信速率（简化接口，向后兼容）"""
        return self._get_inter_task_comm_rate(vehicle, pred_task_id, 0, 'Local', curr_loc)

    def _get_inter_task_comm_rate(self, vehicle, pred_task_id, curr_task_id, pred_loc, curr_loc):
        """[关键方法] 计算任务间通信速率

        通信场景:
        - Local->Local: 同地执行，无传输需求 (返回inf)
        - V2V: 基于距离和信道模型计算速率
        - V2I: 基于V2I信道模型计算速率
        - RSU参与: 使用V2I链路

        缓存优化:
        - 同一时间步内相同位置对的查询结果会被缓存
        - 避免重复计算信道速率
        """
        if pred_loc == curr_loc:
            return float('inf')

        time_key = int(self.time * 10)
        cache_key = (vehicle.id, str(pred_loc), str(curr_loc), time_key)
        if cache_key in self._comm_rate_cache:
            return self._comm_rate_cache[cache_key]

        tx_veh = None
        rx_veh = None

        if pred_loc == 'Local':
            tx_veh = vehicle
        elif isinstance(pred_loc, int) and 0 <= pred_loc < len(self.vehicles):
            tx_veh = self.vehicles[pred_loc]

        if curr_loc == 'Local':
            rx_veh = vehicle
        elif isinstance(curr_loc, int) and 0 <= curr_loc < len(self.vehicles):
            rx_veh = self.vehicles[curr_loc]

        rate = 1e-6

        if pred_loc == 'RSU' or curr_loc == 'RSU':
            # 确定目标位置（如果是RSU）
            if self._is_rsu_location(pred_loc):
                rsu_id = self._get_rsu_id_from_location(pred_loc)
                if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                    rsu_pos = self.rsus[rsu_id].position
                else:
                    rsu_pos = Cfg.RSU_POS  # 向后兼容
                target_veh = rx_veh if rx_veh else (tx_veh if tx_veh else self.vehicles[0] if len(self.vehicles) > 0 else None)
                if target_veh:
                    rate = self.channel.compute_one_rate(target_veh, rsu_pos, 'V2I', self.time)
                else:
                    rate = 1e6
            elif self._is_rsu_location(curr_loc):
                rsu_id = self._get_rsu_id_from_location(curr_loc)
                if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                    rsu_pos = self.rsus[rsu_id].position
                else:
                    rsu_pos = Cfg.RSU_POS  # 向后兼容
                target_veh = tx_veh if tx_veh else (rx_veh if rx_veh else self.vehicles[0] if len(self.vehicles) > 0 else None)
                if target_veh:
                    rate = self.channel.compute_one_rate(target_veh, rsu_pos, 'V2I', self.time)
                else:
                    rate = 1e6
            else:
                # 向后兼容：使用默认RSU位置
                target_veh = rx_veh if rx_veh else tx_veh
                if target_veh:
                    rate = self.channel.compute_one_rate(target_veh, Cfg.RSU_POS, 'V2I', self.time)
                else:
                    rate = 1e6
        else:
            # V2V通信
            if tx_veh and rx_veh:
                dist = np.linalg.norm(tx_veh.pos - rx_veh.pos)
                if dist <= Cfg.V2V_RANGE:
                    rate = self.channel.compute_one_rate(tx_veh, rx_veh.pos, 'V2V', self.time)
                else:
                    rate = 1e-6
            else:
                rate = 1e-6
        final_rate = max(rate, 1e-6)
        self._comm_rate_cache[cache_key] = final_rate
        return final_rate

    def validate_environment(self):
        """环境验证方法（仅在DEBUG模式下使用）"""
        if not Cfg.DEBUG_MODE:
            return

        print(f"\n=== 环境验证 ===")
        print(f"时间: {self.time:.2f}s")
        print(f"车辆数量: {len(self.vehicles)}")
        total_rsu_queue = sum(rsu.queue_length for rsu in self.rsus) if len(self.rsus) > 0 else 0
        total_rsu_limit = Cfg.RSU_QUEUE_LIMIT * len(self.rsus) if len(self.rsus) > 0 else Cfg.RSU_QUEUE_LIMIT
        print(f"RSU队列: {total_rsu_queue}/{total_rsu_limit}")

        if self.vehicles:
            v = self.vehicles[0]
            if hasattr(v, 'task_dag') and v.task_dag.num_subtasks >= 2:
                print(f"\n=== 任务间通信验证 ===")

                test_cases = [
                    ('Local', 'Local', "本地到本地"),
                    ('Local', 'RSU', "本地到RSU"),
                    ('RSU', 'Local', "RSU到本地"),
                ]

                for pred_loc, curr_loc, desc in test_cases:
                    rate = self._get_inter_task_comm_rate(v, 0, 1, pred_loc, curr_loc)
                    print(f"  {desc}: {rate:.2f} bps")

                if len(self.vehicles) > 1:
                    rate = self._get_inter_task_comm_rate(v, 0, 1, 'Local', 1)
                    print(f"  本地到邻居: {rate:.2f} bps")

        print(f"\n=== CFT计算验证 ===")
        cft1 = self._calculate_global_cft_critical_path()
        cft2 = self._calculate_global_cft_critical_path()
        print(f"第一次CFT: {cft1:.2f}")
        print(f"第二次CFT: {cft2:.2f}")
        print(f"缓存是否生效: {cft1 == cft2}")

    def _calculate_local_execution_time(self, dag, vehicle_id=0):
        """
        [奖励函数辅助] 计算任务在本地执行的预估时间

        包含排队时延，与 _estimate_execution_time() 保持一致

        Args:
            dag: DAGTask 对象
            vehicle_id: 车辆ID，用于获取队列长度和CPU频率

        Returns:
            float: 本地执行总时间 (考虑关键路径和排队时延)
        """
        if dag.num_subtasks == 0:
            return 0.0

        if vehicle_id < len(self.vehicles):
            v = self.vehicles[vehicle_id]
            wait_time = v.task_queue.get_estimated_wait_time(v.cpu_freq)
            freq = v.cpu_freq
        else:
            freq = Cfg.MIN_VEHICLE_CPU_FREQ
            wait_time = 0.0

        node_comp = dag.total_comp
        local_exec_times = node_comp / freq

        critical_path_time = self._calc_critical_path_local(dag, local_exec_times)

        return critical_path_time + wait_time

    def _calc_critical_path_local(self, dag, exec_times):
        """
        [私有方法] 计算本地执行的关键路径时间（用于 r_eff 计算）
        """
        num_tasks = dag.num_subtasks
        in_degree = dag.in_degree.copy()

        ready_indices = np.where(in_degree == 0)[0]
        if len(ready_indices) == 0:
            return 0.0

        est = np.zeros(num_tasks)
        completed = set()

        for _ in range(num_tasks):
            if len(ready_indices) == 0:
                break
            i = ready_indices[0]
            ready_indices = ready_indices[1:]

            max_pred_finish = 0.0
            for pred in range(num_tasks):
                if dag.adj[pred, i] > 0:
                    data_size = dag.data_matrix[pred, i]
                    trans_time = 0.0
                    max_pred_finish = max(max_pred_finish, est[pred] + exec_times[pred] + trans_time)

            est[i] = max_pred_finish + exec_times[i]
            completed.add(i)

            for succ in range(num_tasks):
                if dag.adj[i, succ] > 0 and succ not in completed:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        ready_indices = np.append(ready_indices, succ)

        return np.max(est) if len(est) > 0 else 0.0

    def _calculate_efficiency_gain(self, dag, target, task_idx=None, vehicle_id=None):
        """
        [奖励函数组件] 计算效率收益 r_eff

        基于 MAPPO 设计:
        r_eff = tanh(λ * (T_local - T_exec) / T_local)

        Args:
            dag: 当前车辆的 DAG
            target: 目标执行位置 ('Local', 'RSU', 或车辆ID)
            task_idx: 当前调度的任务索引
            vehicle_id: 车辆ID（用于计算本地执行时间）

        Returns:
            float: 效率收益值 ∈ (-1, 1)
        """
        if vehicle_id is None:
            vehicle_id = 0
        
        t_local = self._calculate_local_execution_time(dag, vehicle_id)
        if t_local <= 0:
            return 0.0

        # 处理target格式：'Local', 'RSU', int车辆ID, 或('RSU', rsu_id)元组
        is_local = (target == 'Local')
        
        if is_local:
            # 本地执行时，效率收益为0（没有相对于本地执行的增益）
            gain_ratio = 0.0
        else:
            t_exec = self._estimate_execution_time(dag, target, task_idx, vehicle_id)
            if t_local > 0:
                gain_ratio = (t_local - t_exec) / t_local
            else:
                gain_ratio = 0.0

        eff_gain = np.tanh(Cfg.EFF_SCALE * gain_ratio)

        return eff_gain

    def _estimate_execution_time(self, dag, target, task_idx=None, vehicle_id=None):
        """
        [奖励函数辅助] 估计在目标节点执行的时间

        包含: 传输时间 + 排队时间 + 计算时间
        
        Args:
            dag: DAG任务对象
            target: 目标执行位置
            task_idx: 任务索引
            vehicle_id: 车辆ID（用于获取车辆位置等信息）
        """
        if vehicle_id is None:
            vehicle_id = 0
        
        # 处理target格式：'Local', 'RSU', int车辆ID, 或('RSU', rsu_id)元组
        if target == 'Local':
            return self._calculate_local_execution_time(dag, vehicle_id)
        elif self._is_rsu_location(target):
            # RSU执行：多RSU场景或单个RSU场景
            if isinstance(target, tuple) and len(target) == 2:
                # 多RSU场景：使用指定的RSU
                rsu_id = target[1]
                if 0 <= rsu_id < len(self.rsus):
                    wait_time = self.rsus[rsu_id].get_estimated_wait_time()
                    freq = self.rsus[rsu_id].cpu_freq
                else:
                    wait_time = 0.0
                    freq = Cfg.F_RSU
            else:
                # 单个RSU场景（向后兼容）
                if len(self.rsus) > 0:
                    wait_time = min([rsu.get_estimated_wait_time() for rsu in self.rsus])
                else:
                    wait_time = 0.0
                freq = Cfg.F_RSU
        elif isinstance(target, int) and 0 <= target < len(self.vehicles):
            # 其他车辆执行
            target_veh = self.vehicles[target]
            wait_time = target_veh.task_queue.get_estimated_wait_time(target_veh.cpu_freq)
            freq = target_veh.cpu_freq
        else:
            # 未知格式，默认本地执行
            return self._calculate_local_execution_time(dag, vehicle_id)

        total_comp = np.sum(dag.total_comp)
        comp_time = total_comp / freq

        trans_time = 0.0
        if task_idx is not None:
            input_data = dag.total_data[task_idx] if task_idx < len(dag.total_data) else 0.0
            if input_data > 0:
                # 使用正确的车辆位置
                if vehicle_id < len(self.vehicles):
                    veh_pos = self.vehicles[vehicle_id].pos
                else:
                    veh_pos = self.vehicles[0].pos if len(self.vehicles) > 0 else np.array([0, 0])
                
                # 处理target格式：'Local', 'RSU', int车辆ID, 或('RSU', rsu_id)元组
                if self._is_rsu_location(target):
                    # RSU执行
                    if isinstance(target, tuple) and len(target) == 2:
                        # 多RSU场景：使用指定的RSU位置
                        rsu_id = target[1]
                        if 0 <= rsu_id < len(self.rsus):
                            rsu_pos = self.rsus[rsu_id].position
                            dist = np.linalg.norm(veh_pos - rsu_pos)
                        else:
                            dist = 500.0  # 默认距离
                    else:
                        # 单个RSU场景（向后兼容）：使用最近RSU的距离
                        if len(self.rsus) > 0:
                            min_dist = min([np.linalg.norm(veh_pos - rsu.position) for rsu in self.rsus])
                            dist = min_dist
                        else:
                            dist = np.linalg.norm(veh_pos - Cfg.RSU_POS) if len(self.vehicles) > 0 else 500.0
                    rate = self._estimate_rate(dist, 'V2I', target)
                elif isinstance(target, int) and target < len(self.vehicles):
                    # 其他车辆执行
                    tx_pos = veh_pos
                    rx_pos = self.vehicles[target].pos
                    dist = np.linalg.norm(tx_pos - rx_pos)
                    rate = self._estimate_rate(dist, 'V2V', target)
                else:
                    # 未知格式，使用默认速率
                    rate = 1e6  # 默认高速率
                
                trans_time = input_data / max(rate, 1e-6)

        return wait_time + comp_time + trans_time

    def _estimate_rate(self, dist, link_type, target_id=None):
        """
        [通信模型辅助] 估计通信速率 (bits/s)

        使用简化的 Shannon 公式:
        rate = BW * log2(1 + SNR_linear)

        Args:
            dist: 通信距离 (m)
            link_type: 'V2I' 或 'V2V'
            target_id: 目标车辆ID (V2V 时使用)

        Returns:
            float: 通信速率 (bits/s)
        """
        if link_type == 'V2I':
            pl = Cfg.PL_ALPHA_V2I + Cfg.PL_BETA_V2I * np.log10(max(dist, 1.0))
            snr_db = Cfg.TX_POWER_MIN_DBM - pl - Cfg.NOISE_POWER_DBM
            snr_linear = 10 ** (snr_db / 10)
            rate = Cfg.BW_V2I * np.log2(1 + snr_linear)
            rate = max(rate, 0)
        else:
            pl = Cfg.PL_ALPHA_V2V + Cfg.PL_BETA_V2V * np.log10(max(dist, 1.0))
            interference_w = Cfg.dbm2watt(Cfg.V2V_INTERFERENCE_DBM)
            noise_w = Cfg.dbm2watt(Cfg.NOISE_POWER_DBM)
            snr_db = Cfg.TX_POWER_MIN_DBM - pl - 10 * np.log10(noise_w + interference_w)
            snr_linear = 10 ** (snr_db / 10)
            rate = Cfg.BW_V2V * np.log2(1 + snr_linear)
            rate = max(rate, 0)
        return rate

    def _calculate_congestion_penalty(self, target, task_comp=0, vehicle_id=None):
        """
        [奖励函数组件] 计算拥塞惩罚 r_cong（基于计算量）

        基于 MAPPO 设计:
        r_cong = -((Q_load + task_comp) / Q_max_load)^γ

        Args:
            target: 目标节点
            task_comp: 当前任务的计算量 (cycles)
            vehicle_id: 车辆ID（用于获取本地队列负载）

        Returns:
            float: 拥塞惩罚值 (≤ 0)
        """
        # 处理target格式：'Local', 'RSU', int车辆ID, 或('RSU', rsu_id)元组
        if target == 'Local':
            # 车辆本地队列
            if vehicle_id is not None and vehicle_id < len(self.vehicles):
                q_curr_load = self.vehicles[vehicle_id].task_queue.get_total_load()
                q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
            else:
                return 0.0
        elif self._is_rsu_location(target):
            # RSU执行
            if isinstance(target, tuple) and len(target) == 2:
                # 多RSU场景：使用指定的RSU队列计算量
                rsu_id = target[1]
                if 0 <= rsu_id < len(self.rsus):
                    q_curr_load = self.rsus[rsu_id].queue_manager.get_total_load()
                    q_max_load = Cfg.RSU_QUEUE_CYCLES_LIMIT
                else:
                    return 0.0
            else:
                # 单个RSU场景（向后兼容）：使用所有RSU的总计算量
                q_curr_load = sum([rsu.queue_manager.get_total_load() for rsu in self.rsus]) if len(self.rsus) > 0 else 0
                q_max_load = Cfg.RSU_QUEUE_CYCLES_LIMIT * len(self.rsus) if len(self.rsus) > 0 else Cfg.RSU_QUEUE_CYCLES_LIMIT
        elif isinstance(target, int):
            if 0 <= target < len(self.vehicles):
                q_curr_load = self.vehicles[target].task_queue.get_total_load()
                q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
            else:
                return 0.0
        else:
            return 0.0

        util_ratio = (q_curr_load + task_comp) / q_max_load
        util_ratio = np.clip(util_ratio, 0.0, 1.0)
        cong_penalty = -1.0 * (util_ratio ** Cfg.CONG_GAMMA)

        return cong_penalty

    def _calculate_constraint_penalty(self, vehicle_id, target, task_idx=None, task_comp=None):
        """
        [奖励函数组件] 计算约束惩罚 r_pen
        
        采用"掩码覆盖"设计：
        - 硬约束触发时直接返回REWARD_MIN，不再计算软约束
        - 软约束（距离预警）提供梯度信息

        Args:
            vehicle_id: 车辆ID
            target: 目标节点
            task_idx: 任务索引
            task_comp: 任务计算量 (cycles)

        Returns:
            tuple: (soft_penalty, hard_constraint_triggered)
                - soft_penalty: 软约束惩罚（距离预警）
                - hard_constraint_triggered: 是否触发硬约束
        """
        soft_penalty = 0.0
        hard_triggered = False

        v = self.vehicles[vehicle_id]
        
        if task_comp is None:
            task_comp = Cfg.MEAN_COMP_LOAD

        # ========== 硬约束检测 ==========
        # 1. RSU范围检查
        if self._is_rsu_location(target):
            in_range = False
            rsu_dist = float('inf')
            if isinstance(target, tuple) and len(target) == 2:
                rsu_id = target[1]
                if 0 <= rsu_id < len(self.rsus):
                    in_range = self.rsus[rsu_id].is_in_coverage(v.pos)
                    rsu_dist = np.linalg.norm(v.pos - self.rsus[rsu_id].position)
            else:
                if len(self.rsus) > 0:
                    for rsu in self.rsus:
                        if rsu.is_in_coverage(v.pos):
                            in_range = True
                            rsu_dist = min(rsu_dist, np.linalg.norm(v.pos - rsu.position))
                else:
                    dist = np.linalg.norm(v.pos - Cfg.RSU_POS)
                    in_range = (dist <= Cfg.RSU_RANGE)
                    rsu_dist = dist
            
            if not in_range:
                hard_triggered = True
            else:
                # 距离预警（软约束）
                safe_dist = Cfg.RSU_RANGE * Cfg.DIST_SAFE_FACTOR
                if rsu_dist > safe_dist:
                    dist_ratio = (rsu_dist - safe_dist) / (Cfg.RSU_RANGE - safe_dist + 1e-6)
                    dist_ratio = np.clip(dist_ratio, 0.0, 1.0)
                    soft_penalty += -Cfg.DIST_PENALTY_WEIGHT * (dist_ratio ** Cfg.DIST_SENSITIVITY)
        
        # 2. V2V范围检查
        elif isinstance(target, int):
            if target < len(self.vehicles):
                target_veh = self.vehicles[target]
                dist = np.linalg.norm(v.pos - target_veh.pos)
                
                if dist > Cfg.V2V_RANGE:
                    hard_triggered = True
                else:
                    # 距离预警（软约束）
                    safe_dist = Cfg.V2V_RANGE * Cfg.DIST_SAFE_FACTOR
                    if dist > safe_dist:
                        dist_ratio = (dist - safe_dist) / (Cfg.V2V_RANGE - safe_dist + 1e-6)
                        dist_ratio = np.clip(dist_ratio, 0.0, 1.0)
                        soft_penalty += -Cfg.DIST_PENALTY_WEIGHT * (dist_ratio ** Cfg.DIST_SENSITIVITY)

        # 3. 队列溢出检查（硬约束）
        if not hard_triggered:
            q_after_load = 0.0
            q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
            if target == 'Local':
                q_after_load = v.task_queue.get_total_load() + task_comp
                q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
            elif self._is_rsu_location(target):
                if isinstance(target, tuple) and len(target) == 2:
                    rsu_id = target[1]
                    if 0 <= rsu_id < len(self.rsus):
                        q_after_load = self.rsus[rsu_id].queue_manager.get_total_load() + task_comp
                    else:
                        q_after_load = task_comp
                    q_max_load = Cfg.RSU_QUEUE_CYCLES_LIMIT
                else:
                    q_after_load = (sum([rsu.queue_manager.get_total_load() for rsu in self.rsus]) + task_comp) if len(self.rsus) > 0 else task_comp
                    q_max_load = Cfg.RSU_QUEUE_CYCLES_LIMIT * len(self.rsus) if len(self.rsus) > 0 else Cfg.RSU_QUEUE_CYCLES_LIMIT
            elif isinstance(target, int):
                if target < len(self.vehicles):
                    q_after_load = self.vehicles[target].task_queue.get_total_load() + task_comp
                    q_max_load = Cfg.VEHICLE_QUEUE_CYCLES_LIMIT
            
            if q_after_load > q_max_load:
                hard_triggered = True

        return soft_penalty, hard_triggered

    def _clip_reward(self, reward):
        """
        [奖励函数辅助] 奖励裁剪，防止奖励爆炸

        Args:
            reward: 原始奖励值

        Returns:
            float: 裁剪后的奖励值
        """
        return np.clip(reward, Cfg.REWARD_MIN, Cfg.REWARD_MAX)

    def calculate_agent_reward(self, vehicle_id, target, task_idx=None, data_size=0, task_comp=None):
        """
        [MAPPO奖励函数] 计算单个智能体的奖励
        
        采用"掩码覆盖"设计：
        - 硬约束触发时直接返回REWARD_MIN
        - 否则计算 r = α*r_eff + β*r_cong + r_soft_pen

        Args:
            vehicle_id: 车辆ID
            target: 卸载目标 ('Local', 'RSU', 或车辆ID)
            task_idx: 当前调度的任务索引
            data_size: 任务数据量 (bits)
            task_comp: 任务计算量 (cycles)

        Returns:
            float: 归一化后的奖励值
        """
        v = self.vehicles[vehicle_id]
        dag = v.task_dag
        
        if task_comp is None:
            task_comp = Cfg.MEAN_COMP_LOAD

        # 1. 检查约束惩罚（区分硬约束和软约束）
        r_soft_pen, hard_triggered = self._calculate_constraint_penalty(vehicle_id, target, task_idx, task_comp)
        
        # 2. 硬约束触发时直接返回最小值（掩码覆盖）
        if hard_triggered:
            return Cfg.REWARD_MIN
        
        # 3. 计算效率收益和拥塞惩罚
        r_eff = self._calculate_efficiency_gain(dag, target, task_idx, vehicle_id)
        r_cong = self._calculate_congestion_penalty(target, task_comp, vehicle_id)

        # 4. 计算超时惩罚（软约束，仅在超时且未完成时应用）
        r_timeout = 0.0
        if dag.deadline > 0:
            elapsed = self.time - dag.start_time
            if elapsed > dag.deadline and not dag.is_finished:
                # 非线性超时惩罚：r_timeout = -η * tanh(σ * (t - D) / D)
                overtime_ratio = (elapsed - dag.deadline) / dag.deadline
                r_timeout = -Cfg.TIMEOUT_PENALTY_WEIGHT * np.tanh(Cfg.TIMEOUT_STEEPNESS * overtime_ratio)
                dag.set_failed()
                
                # [死因诊断] 已关闭，避免打印过多
                # if not dag.timeout_logged:
                #     completed = np.sum(dag.status == 3)
                #     running = np.sum(dag.status == 2)
                #     ready = np.sum(dag.status == 1)
                #     pending = np.sum(dag.status == 0)
                #     print(f"[AUTOPSY] Veh{vehicle_id} DAG timeout: "
                #           f"{completed}/{dag.num_subtasks} done, "
                #           f"{running} running, {ready} ready, {pending} pending | "
                #           f"elapsed={elapsed:.2f}s, deadline={dag.deadline:.2f}s, "
                #           f"overtime={elapsed-dag.deadline:.2f}s")
                #     dag.timeout_logged = True

        # 5. 组合奖励（所有软约束项）
        reward = (Cfg.EFF_WEIGHT * r_eff +
                  Cfg.CONG_WEIGHT * r_cong +
                  r_soft_pen +
                  r_timeout)

        # 6. [新增] 任务成功bonus（稀疏奖励强化，打破数学期望陷阱）
        # 只在任务完成且未失败时给予，提供强烈的正反馈信号
        if dag.is_finished and not dag.is_failed:
            reward += Cfg.SUCCESS_BONUS

        reward = self._clip_reward(reward)

        return reward