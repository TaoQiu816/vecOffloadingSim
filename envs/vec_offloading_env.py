import gymnasium as gym
import numpy as np
from configs.config import SystemConfig as Cfg
from envs.modules.channel import ChannelModel
from envs.modules.queue_system import FIFOQueue
from envs.entities.vehicle import Vehicle
from envs.entities.rsu import RSU
from envs.entities.road_network import RoadNetwork, RoadSegment
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
    - task_mask: 可调度任务掩码
    - target_mask: 动作目标掩码 (Local/RSU/V2V)

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
        
        # 道路网络
        self.road_network = RoadNetwork()
        
        # RSU实体列表
        self.rsus = []
        self._init_rsus()
        
        # 保持向后兼容（使用总队列长度，用于单个RSU场景）
        self.rsu_queue_curr = 0
        self.last_global_cft = 0.0
        self._comm_rate_cache = {}
        self._cache_time_step = -1.0
        self._cft_cache = None
        self._cft_cache_time = 0.0
        self._cft_cache_valid = False
        self._cft_state_hash = None
        self._dist_matrix_cache = None
        self._dist_matrix_time = -1.0
        self._rsu_dist_cache = {}

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
        初始化RSU实体列表
        
        根据配置创建RSU实例，默认在地图中心创建一个RSU
        后续可以扩展为在道路上部署多个RSU
        """
        num_rsu = getattr(Cfg, 'NUM_RSU', 1)
        self.rsus = []
        
        if num_rsu == 1:
            # 单个RSU：默认在地图中心
            rsu = RSU(
                rsu_id=0,
                position=Cfg.RSU_POS.copy(),
                cpu_freq=Cfg.F_RSU,
                num_processors=getattr(Cfg, 'RSU_NUM_PROCESSORS', 4),
                queue_limit=Cfg.RSU_QUEUE_LIMIT,
                coverage_range=Cfg.RSU_RANGE
            )
            self.rsus.append(rsu)
        else:
            # 多个RSU：在道路交叉口或关键位置部署
            # 这里使用简单的网格分布，后续可以根据道路网络优化
            map_size = Cfg.MAP_SIZE
            grid_size = int(np.ceil(np.sqrt(num_rsu)))
            spacing = map_size / (grid_size + 1)
            
            for i in range(num_rsu):
                row = i // grid_size
                col = i % grid_size
                pos = np.array([(col + 1) * spacing, (row + 1) * spacing])
                
                rsu = RSU(
                    rsu_id=i,
                    position=pos,
                    cpu_freq=Cfg.F_RSU,
                    num_processors=getattr(Cfg, 'RSU_NUM_PROCESSORS', 4),
                    queue_limit=Cfg.RSU_QUEUE_LIMIT,
                    coverage_range=Cfg.RSU_RANGE
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
        
        # 重置RSU队列
        for rsu in self.rsus:
            rsu.clear_queue()
        # 保持向后兼容
        self.rsu_queue_curr = 0
        self._comm_rate_cache = {}
        self._cache_time_step = -1.0
        self._dist_matrix_cache = None
        self._dist_matrix_time = -1.0
        self._rsu_dist_cache = {}
        if abs(self.time - self._cft_cache_time) > Cfg.DT * 0.5:
            self._cft_cache = None
            self._cft_cache_valid = False

        for i in range(Cfg.NUM_VEHICLES):
            # 车辆初始位置：随机选择一条道路段，在道路上随机生成位置
            if len(self.road_network.segments) > 0:
                seg = np.random.choice(self.road_network.segments)
                dist_along = np.random.uniform(0, seg.length)
                pos = seg.get_position_at_distance(dist_along)
                # 添加一些垂直于道路的随机偏移（模拟车道）
                offset = np.random.uniform(-seg.width * 0.3, seg.width * 0.3)
                pos = pos + seg.perpendicular * offset
                pos = np.clip(pos, 0, Cfg.MAP_SIZE)  # 确保在地图范围内
            else:
                pos = np.random.rand(2) * Cfg.MAP_SIZE
            
            v = Vehicle(i, pos)
            v.cpu_freq = np.random.uniform(Cfg.MIN_VEHICLE_CPU_FREQ, Cfg.MAX_VEHICLE_CPU_FREQ)
            v.tx_power_dbm = Cfg.TX_POWER_DEFAULT_DBM if hasattr(Cfg, 'TX_POWER_DEFAULT_DBM') else Cfg.TX_POWER_MIN_DBM

            n_node = np.random.randint(Cfg.MIN_NODES, Cfg.MAX_NODES + 1)
            adj, prof, data, ddl = self.dag_gen.generate(n_node, veh_f=v.cpu_freq)
            v.task_dag = DAGTask(0, adj, prof, data, ddl)
            v.task_dag.start_time = 0.0
            v.task_queue.clear()  # 清空队列
            v.task_queue_len = 0  # 同步队列长度
            speed = np.random.uniform(Cfg.VEL_MIN, Cfg.VEL_MAX)
            angle = np.random.uniform(0, 2 * np.pi)
            v.vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            v.last_scheduled_subtask = -1
            v.exec_locations = [None] * v.task_dag.num_subtasks

            self.vehicles.append(v)

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
            subtask_idx = int(act['subtask'])

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

            if subtask_idx != v.last_scheduled_subtask:
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
                        min_wait = float('inf')
                        for rsu in rsu_in_range:
                            if not rsu.is_queue_full():
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
                    if t_veh.task_queue.is_full():
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
                            if not v.task_queue.is_full():
                                v.task_queue.enqueue(task_comp)
                elif isinstance(actual_target, int):
                    t_veh = self.vehicles[actual_target]
                    # 目标车辆队列
                    t_veh.task_queue.enqueue(task_comp)
                elif actual_target == 'Local':
                    # 本地队列
                    v.task_queue.enqueue(task_comp)
                
                # 同步队列长度（向后兼容：使用第一个RSU的队列长度，如果存在）
                if len(self.rsus) > 0:
                    self.rsu_queue_curr = self.rsus[0].queue_length
                else:
                    self.rsu_queue_curr = 0
                for v_check in self.vehicles:
                    v_check.update_queue_sync()

        # 队列长度同步（由FIFO队列管理，这里只做同步）
        if len(self.rsus) > 0:
            self.rsu_queue_curr = self.rsus[0].queue_length
        else:
            self.rsu_queue_curr = 0
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
                    # 任务完成时，从队列中移除一个任务（FIFO顺序）
                    if isinstance(tgt, tuple) and tgt[0] == 'RSU':
                        # 多RSU场景：从选定的RSU队列移除
                        rsu_id = tgt[1]
                        if 0 <= rsu_id < len(self.rsus):
                            self.rsus[rsu_id].dequeue_task()
                            if len(self.rsus) > 0:
                                self.rsu_queue_curr = self.rsus[0].queue_length
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
                    parent_loc = v.curr_target
                    v.exec_locations[completed_task] = parent_loc
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
                child_loc = v.curr_target if child_id == v.curr_subtask else v.exec_locations[child_id]

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

            # 更新车辆位置（支持道路约束）
            v.update_pos(Cfg.DT, Cfg.MAP_SIZE)
            # 将位置约束到道路上（可选：如果启用道路约束）
            # v.pos = self.road_network.constrain_position_to_road(v.pos)

        self.time += Cfg.DT

        rewards = []
        for i, v in enumerate(self.vehicles):
            dag = v.task_dag
            target = v.curr_target if v.curr_subtask is not None else 'Local'
            task_idx = v.curr_subtask if v.curr_subtask is not None else 0
            data_size = dag.data_matrix[v.curr_subtask, :].sum() if v.curr_subtask is not None else 0

            r = self.calculate_agent_reward(i, target, task_idx, data_size)
            rewards.append(r)

        self.last_global_cft = self._calculate_global_cft_critical_path()

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
        current_state_hash = hash((
            round(self.time, 3),
            self.rsu_queue_curr,
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
        self._dist_matrix_cache = np.zeros((num_vehicles, num_vehicles))
        for i in range(num_vehicles):
            diff = positions - positions[i]
            self._dist_matrix_cache[i] = np.linalg.norm(diff, axis=1)

        self._dist_matrix_time = self.time
        return self._dist_matrix_cache

    def _get_rsu_dist(self, vehicle):
        """获取车辆到RSU的距离（带缓存）"""
        if vehicle.id in self._rsu_dist_cache:
            return self._rsu_dist_cache[vehicle.id]
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
        if len(self.rsus) > 0:
            rsu_wait_time = min([rsu.get_estimated_wait_time() for rsu in self.rsus])
            rsu_is_full = all([rsu.is_queue_full() for rsu in self.rsus])
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

            for j, other in enumerate(self.vehicles):
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
                else:
                    valid_targets_base.append(0)

            actual_num_neighbors = len(neighbor_id_map)
            num_targets = 2 + actual_num_neighbors

            MAX_NEIGHBORS = Cfg.NUM_VEHICLES
            neighbor_dim = 8
            neighbors_array = np.zeros((MAX_NEIGHBORS, neighbor_dim), dtype=np.float32)
            for idx, neighbor_feat in enumerate(neighbors):
                if idx < MAX_NEIGHBORS:
                    neighbors_array[idx] = neighbor_feat

            task_schedulable = v.task_dag.get_action_mask()
            target_mask_row = np.array(valid_targets_base, dtype=bool)

            v.neighbor_id_map = neighbor_id_map.copy()

            # RSU队列满时禁用RSU卸载
            if rsu_is_full:
                target_mask_row[1] = False

            # [关键] V2I离开判断 - 车辆即将离开RSU覆盖范围时禁用RSU
            speed = np.linalg.norm(v.vel)
            if speed > 0.1:
                time_to_leave = (Cfg.RSU_RANGE - dist_rsu) / speed
                avg_data_size = np.mean(v.task_dag.rem_data)
                est_trans_time = avg_data_size / max(est_v2i_rate, 1e-6)
                if time_to_leave < est_trans_time:
                    target_mask_row[1] = False

            # [关键] 目标车辆队列满检查 - 动态禁用过载车辆
            for i in range(2, num_targets):
                if valid_targets_base[i] == 1:
                    target_veh_id = i - 2
                    n_veh = next((veh for veh in self.vehicles if veh.id == target_veh_id), None)
                    if n_veh and n_veh.task_queue_len >= Cfg.VEHICLE_QUEUE_LIMIT:
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
            # 将task_mask填充到固定维度MAX_NODES
            padded_task_mask = np.zeros(MAX_NODES, dtype=bool)
            padded_task_mask[:num_nodes] = task_schedulable

            obs_list.append({
                'node_x': padded_node_feats,
                'self_info': self_info,
                'rsu_info': [rsu_load_norm],
                'adj': padded_adj,
                'neighbors': neighbors_array,
                'task_mask': padded_task_mask,
                'target_mask': padded_mask
            })

        return obs_list

    def _get_comm_rate(self, vehicle, pred_task_id, curr_loc, rsu_pos):
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
            else:
                target_veh = rx_veh if rx_veh else tx_veh
                if target_veh:
                    rate = self.channel.compute_one_rate(target_veh, Cfg.RSU_POS, 'V2I', self.time)
                else:
                    rate = 1e6
            else:
                rate = 1e6
        else:
            if tx_veh and rx_veh:
                dist = np.linalg.norm(tx_veh.pos - rx_veh.pos)
                if dist <= Cfg.V2V_RANGE:
                    rate = self.channel.compute_one_rate(tx_veh, rx_veh.pos, 'V2V', self.time)

        final_rate = max(rate, 1e-6)
        self._comm_rate_cache[cache_key] = final_rate
        return final_rate

    def validate_environment(self):
        if not Cfg.DEBUG_MODE:
            return

        print(f"\n=== 环境验证 ===")
        print(f"时间: {self.time:.2f}s")
        print(f"车辆数量: {len(self.vehicles)}")
        print(f"RSU队列: {self.rsu_queue_curr}/{Cfg.RSU_QUEUE_LIMIT}")

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

    def _calculate_efficiency_gain(self, dag, target, task_idx=None):
        """
        [奖励函数组件] 计算效率收益 r_eff

        基于 MAPPO 设计:
        r_eff = tanh(λ * (T_local - T_exec) / T_local)

        Args:
            dag: 当前车辆的 DAG
            target: 目标执行位置 ('Local', 'RSU', 或车辆ID)
            task_idx: 当前调度的任务索引

        Returns:
            float: 效率收益值 ∈ (-1, 1)
        """
        t_local = self._calculate_local_execution_time(dag)
        if t_local <= 0:
            return 0.0

        if target == 'Local':
            t_exec = t_local
        else:
            t_exec = self._estimate_execution_time(dag, target, task_idx)

        gain_ratio = (t_local - t_exec) / t_local
        eff_gain = np.tanh(Cfg.EFF_SCALE * gain_ratio)

        return eff_gain

    def _estimate_execution_time(self, dag, target, task_idx=None):
        """
        [奖励函数辅助] 估计在目标节点执行的时间

        包含: 传输时间 + 排队时间 + 计算时间
        """
        if target == 'RSU':
            # 多RSU场景：使用所有RSU中的最小等待时间
            if len(self.rsus) > 0:
                wait_time = min([rsu.get_estimated_wait_time() for rsu in self.rsus])
            else:
                wait_time = 0.0
            freq = Cfg.F_RSU
        elif isinstance(target, int) and 0 <= target < len(self.vehicles):
            target_veh = self.vehicles[target]
            wait_time = target_veh.task_queue.get_estimated_wait_time(target_veh.cpu_freq)
            freq = target_veh.cpu_freq
        else:
            return self._calculate_local_execution_time(dag)

        total_comp = np.sum(dag.total_comp)
        comp_time = total_comp / freq

        trans_time = 0.0
        if task_idx is not None:
            input_data = dag.total_data[task_idx] if task_idx < len(dag.total_data) else 0.0
            if input_data > 0:
                if target == 'RSU':
                    # 多RSU场景：使用最近RSU的距离
                    if len(self.vehicles) > 0 and len(self.rsus) > 0:
                        veh_pos = self.vehicles[0].pos
                        min_dist = min([np.linalg.norm(veh_pos - rsu.position) for rsu in self.rsus])
                        dist = min_dist
                    else:
                        dist = np.linalg.norm(self.vehicles[0].pos - Cfg.RSU_POS) if len(self.vehicles) > 0 else 500.0
                    rate = self._estimate_rate(dist, 'V2I', target)
                else:
                    tx_pos = self.vehicles[0].pos if len(self.vehicles) > 0 else np.array([0, 0])
                    rx_pos = self.vehicles[target].pos if isinstance(target, int) else tx_pos
                    dist = np.linalg.norm(tx_pos - rx_pos)
                    rate = self._estimate_rate(dist, 'V2V', target)
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
            interference = 10 ** (Cfg.V2V_INTERFERENCE_FACTOR / 10)
            snr_db = Cfg.TX_POWER_MIN_DBM - pl - Cfg.NOISE_POWER_DBM - 10 * np.log10(interference)
            snr_linear = 10 ** (snr_db / 10)
            rate = Cfg.BW_V2V * np.log2(1 + snr_linear)
            rate = max(rate, 0)
        return rate

    def _calculate_congestion_penalty(self, target, task_data_size=0):
        """
        [奖励函数组件] 计算拥塞惩罚 r_cong

        基于 MAPPO 设计:
        r_cong = -((Q + D) / Q_max)^γ

        Args:
            target: 目标节点
            task_data_size: 当前任务的数据量 (bits)

        Returns:
            float: 拥塞惩罚值 (≤ 0)
        """
        if target == 'Local':
            q_curr = 0
            q_max = Cfg.VEHICLE_QUEUE_LIMIT
        elif target == 'RSU':
            # 多RSU场景：使用所有RSU的总队列长度
            if len(self.rsus) > 0:
                q_curr = sum([rsu.queue_length for rsu in self.rsus])
            else:
                q_curr = self.rsu_queue_curr
            q_max = Cfg.RSU_QUEUE_LIMIT * len(self.rsus) if len(self.rsus) > 0 else Cfg.RSU_QUEUE_LIMIT
        elif isinstance(target, int):
            if 0 <= target < len(self.vehicles):
                q_curr = self.vehicles[target].task_queue_len
                q_max = Cfg.VEHICLE_QUEUE_LIMIT
            else:
                return 0.0
        else:
            return 0.0

        util_ratio = (q_curr + task_data_size / Cfg.MEAN_COMP_LOAD) / q_max
        util_ratio = np.clip(util_ratio, 0.0, 1.0)
        cong_penalty = -1.0 * (util_ratio ** Cfg.CONG_GAMMA)

        return cong_penalty

    def _calculate_constraint_penalty(self, vehicle_id, target, task_idx=None):
        """
        [奖励函数组件] 计算约束惩罚 r_pen

        只处理硬约束（物理违规），软约束（超时）在CFT计算中作为失败判断处理

        根据用户设计框架：
        - Deadline是"硬约束"，用于判断任务成功/失败（已在CFT计算中处理）
        - 奖励函数应该只关注相对效率收益，不应该重复惩罚超时

        Args:
            vehicle_id: 车辆ID
            target: 目标节点
            task_idx: 任务索引

        Returns:
            float: 总惩罚值 (负值或0)
        """
        penalty = 0.0

        v = self.vehicles[vehicle_id]

        if target == 'Local':
            pass
        elif target == 'RSU':
            # 多RSU场景：检查是否在任何RSU覆盖范围内
            in_range = False
            if len(self.rsus) > 0:
                for rsu in self.rsus:
                    if rsu.is_in_coverage(v.pos):
                        in_range = True
                        break
            else:
                # 向后兼容：使用配置的RSU位置
                dist = np.linalg.norm(v.pos - Cfg.RSU_POS)
                in_range = (dist <= Cfg.RSU_RANGE)
            
            if not in_range:
                penalty += Cfg.PENALTY_LINK_BREAK
        elif isinstance(target, int):
            if target < len(self.vehicles):
                target_veh = self.vehicles[target]
                dist = np.linalg.norm(v.pos - target_veh.pos)
                if dist > Cfg.V2V_RANGE:
                    penalty += Cfg.PENALTY_LINK_BREAK

        q_after = 0
        if target == 'Local':
            q_after = v.task_queue_len
        elif target == 'RSU':
            # 多RSU场景：估算总队列长度（使用第一个RSU作为参考）
            if len(self.rsus) > 0:
                q_after = sum([rsu.queue_length for rsu in self.rsus]) + 1
            else:
                q_after = self.rsu_queue_curr + 1
        elif isinstance(target, int):
            if target < len(self.vehicles):
                q_after = self.vehicles[target].task_queue_len + 1

        if target == 'RSU':
            q_limit = Cfg.RSU_QUEUE_LIMIT * len(self.rsus) if len(self.rsus) > 0 else Cfg.RSU_QUEUE_LIMIT
        else:
            q_limit = Cfg.VEHICLE_QUEUE_LIMIT
        if q_after > q_limit:
            penalty += Cfg.PENALTY_OVERFLOW

        return penalty

    def _clip_reward(self, reward):
        """
        [奖励函数辅助] 奖励裁剪，防止奖励爆炸

        Args:
            reward: 原始奖励值

        Returns:
            float: 裁剪后的奖励值
        """
        return np.clip(reward, Cfg.REWARD_MIN, Cfg.REWARD_MAX)

    def calculate_agent_reward(self, vehicle_id, target, task_idx=None, data_size=0):
        """
        [MAPPO奖励函数] 计算单个智能体的奖励

        基于 MAPPO 设计:
        r_i,t = α * r_eff + β * r_cong + r_pen

        Args:
            vehicle_id: 车辆ID
            target: 卸载目标 ('Local', 'RSU', 或车辆ID)
            task_idx: 当前调度的任务索引
            data_size: 任务数据量 (bits)

        Returns:
            float: 归一化后的奖励值
        """
        v = self.vehicles[vehicle_id]
        dag = v.task_dag

        r_eff = self._calculate_efficiency_gain(dag, target, task_idx)
        r_cong = self._calculate_congestion_penalty(target, data_size)
        r_pen = self._calculate_constraint_penalty(vehicle_id, target, task_idx)

        reward = (Cfg.EFF_WEIGHT * r_eff +
                  Cfg.CONG_WEIGHT * r_cong +
                  r_pen)

        reward = self._clip_reward(reward)

        return reward