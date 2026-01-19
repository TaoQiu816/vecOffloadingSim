import gymnasium as gym
import numpy as np
import os
from collections import deque, defaultdict
from configs.config import SystemConfig as Cfg
from envs.modules.channel import ChannelModel
from envs.modules.queue_system import FIFOQueue
from envs.entities.vehicle import Vehicle
from envs.entities.rsu import RSU
from envs.entities.task_dag import DAGTask
from envs.services.comm_queue_service import CommQueueService
from envs.services.cpu_queue_service import CpuQueueService
from envs.services.dag_completion_handler import DagCompletionHandler
from envs.rl.obs_builder import ObsBuilder
from envs.rl.reward_engine import RewardEngine
from envs.audit.trace import TraceCollector
from envs.audit.stats_collector import StatsCollector
from utils.dag_generator import DAGGenerator
from utils.reward_stats import RewardStats, ReservoirSampler
from envs.jobs import TransferJob, ComputeJob
from envs.rl.reward_functions import compute_absolute_reward
from envs.services.rsu_selector import RSUSelector
from envs.services.candidate_set_manager import CandidateSetManager


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

    def __init__(self, config=None):
        """初始化环境
        
        Args:
            config: 配置类（可选，默认使用全局Cfg）
        """
        # 使用传入的config或全局Cfg
        if config is not None:
            self.config = config
        else:
            from configs.config import SystemConfig as Cfg
            self.config = Cfg
        
        self.channel = ChannelModel()
        self.dag_gen = DAGGenerator()
        self.vehicles = []
        self.time = 0.0
        
        # RSU实体列表（道路模型：等间距线性部署）
        self.rsus = []
        self._init_rsus()
        # RSU选择器服务
        self.rsu_selector = RSUSelector(self.rsus, self.channel, self.config)
        # 候选集管理器（仅处理V2V候选）
        self.candidate_manager = CandidateSetManager(self.config)

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
        self._last_candidates = {}
        self._last_candidate_set = {}
        self._last_rsu_choice = {}
        self._reward_stats = RewardStats(sample_size=self.config.STATS_RESERVOIR_SIZE, seed=self.config.STATS_SEED)
        self._episode_id = 0
        self._episode_steps = 0
        self._rate_sampler_v2i = ReservoirSampler(size=self.config.RATE_RESERVOIR_SIZE, seed=self.config.STATS_SEED)
        self._rate_sampler_v2v = ReservoirSampler(size=self.config.RATE_RESERVOIR_SIZE, seed=self.config.STATS_SEED)
        self._rate_norm_v2i = self.config.NORM_MAX_RATE_V2I
        self._rate_norm_v2v = self.config.NORM_MAX_RATE_V2V
        self._rate_min_samples = self.config.RATE_MIN_SAMPLES
        # 通信推进服务（行为等同于原有Phase3推进，后续可独立成模块）
        self._comm_service = CommQueueService(self.channel, self.config)
        # 计算推进服务（行为等同于原有Phase4推进，后续可独立成模块）
        self._cpu_service = CpuQueueService(self.config)
        # DAG完成处理器（阶段5已集成）
        self._dag_handler = DagCompletionHandler(self.config)
        # 观测构造器（阶段6框架，当前未启用）
        self._obs_builder = ObsBuilder(self.config)
        # 奖励引擎（阶段6框架，当前未启用）
        self._reward_engine = RewardEngine(self.config)
        # Trace收集器（阶段7，默认关闭）
        self._trace_collector = TraceCollector(enabled=False)
        # 统计收集器（阶段7，默认开启）
        self._stats_collector = StatsCollector(enabled=True)
        self._episode_dT_eff_values = []
        self._episode_energy_norm_values = []
        self._episode_t_tx_values = []
        self._episode_task_durations = []  # [新增] 追踪真实任务完成时间（物理指标）
        self._last_obs_stamp = None
        # [P2性能统计] 运行期累积器
        self._p2_active_time = 0.0
        self._p2_idle_time = 0.0
        self._p2_deltaW_active = 0.0
        self._p2_zero_delta_steps = 0
        
        # [审计系统] 12项核心指标收集
        self._audit_step_info = {}
        # [Deadline检查计数] 用于诊断是否触发deadline判定
        self._audit_deadline_checks = 0
        self._audit_deadline_misses = 0
        # [P2性能统计] 仅在active时间段统计服务速率
        self._active_time_steps = []  # 记录active_tasks>0的步数
        self._delta_w_active = []  # 对应步的计算量减少
        self._audit_v2v_lifecycle = {
            'tx_started': set(),    # (owner_id, subtask_id)
            'tx_done': set(),
            'received': set(),
            'added_to_active': set(),
            'cpu_finished': set(),
            'dag_completed': set()
        }
        self._audit_task_registry = {}  # {(owner, subtask): {'state': , 'host': }}
        # PBRS诊断数据
        self._pbrs_debug_records = []
        self._last_phi_debug = {}
        self._episode_illegal_count = 0
        self._episode_no_task_count = 0
        self._episode_illegal_reasons = {}

        # 归一化常数（预先计算倒数以提高性能）
        self._inv_map_size = 1.0 / self.config.MAP_SIZE
        self._inv_max_nodes = 1.0 / self.config.MAX_NODES
        self._inv_max_cpu = 1.0 / self.config.NORM_MAX_CPU
        self._inv_max_comp = 1.0 / self.config.NORM_MAX_COMP
        self._inv_max_data = 1.0 / self.config.NORM_MAX_DATA
        self._inv_max_wait = 1.0 / self.config.NORM_MAX_WAIT_TIME
        self._inv_max_rate_v2i = 1.0 / self.config.NORM_MAX_RATE_V2I
        self._inv_max_rate_v2v = 1.0 / self.config.NORM_MAX_RATE_V2V
        self._inv_max_velocity = 1.0 / self.config.MAX_VELOCITY
        self._inv_v2v_range = 1.0 / self.config.V2V_RANGE
        self._mean_comp_load = self.config.MEAN_COMP_LOAD
        self._max_rsu_contact_time = self.config.RSU_RANGE / max(self.config.VEL_MIN, 1e-6)
        self._max_v2v_contact_time = self.config.V2V_RANGE / max(self.config.VEL_MIN, 1e-6)
        self._last_candidates = {}
        self._last_rsu_choice = {}
        # 动态车辆统计：记录整个episode出现过的车辆ID
        self._vehicles_seen = set()
        self._last_obs_stamp = None
        # 奖励/通信快照
        self._rate_snapshot = None
        self._rate_snapshot_step = -1
        self._rate_snapshot_token = None
        self._debug_rate_token_phase3 = None
        self._debug_rate_token_reward = None
        self._f_max_const = max(self.config.MAX_VEHICLE_CPU_FREQ, getattr(self.config, "F_RSU", 0.0))
        
        # =====================================================================
        # [Gymnasium接口] 定义动作空间和观测空间
        # =====================================================================
        # 动作空间：Tuple of Dict (每个车辆一个动作)
        # 每个车辆的动作:
        # - target: 0=Local, 1=RSU, 2...(2+MAX_NEIGHBORS-1)=V2V邻居
        # - power: 连续rho∈[0,1]（Beta输出），与环境解析保持一致
        single_agent_action_space = gym.spaces.Dict({
            "target": gym.spaces.Discrete(self.config.MAX_TARGETS),
            "power": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })
        self.action_space = gym.spaces.Tuple([single_agent_action_space] * self.config.NUM_VEHICLES)
        
        # 观测空间：Dict空间（具体维度在reset后确定）
        # 这里定义一个占位符，实际维度在reset()后根据DAG大小动态确定
        self.observation_space = gym.spaces.Dict({
            'node_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.MAX_NODES, 7), dtype=np.float32),
            'self_info': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'rsu_info': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.NUM_RSU,), dtype=np.float32),
            'serving_rsu_id': gym.spaces.Box(low=-1, high=max(self.config.NUM_RSU - 1, 0), shape=(), dtype=np.int64),
            'serving_rsu_onehot': gym.spaces.Box(low=0.0, high=1.0, shape=(self.config.NUM_RSU,), dtype=np.float32),
            'serving_rsu_info': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'candidate_ids': gym.spaces.Box(low=-1, high=max(self.config.NUM_VEHICLES, 1), shape=(self.config.MAX_TARGETS,), dtype=np.int64),
            'candidate_types': gym.spaces.Box(low=0, high=3, shape=(self.config.MAX_TARGETS,), dtype=np.int8),
            'candidate_mask': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_TARGETS,), dtype=np.float32),
            'adj': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_NODES, self.config.MAX_NODES), dtype=np.float32),
            'neighbors': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.MAX_NEIGHBORS, 4), dtype=np.float32),
            'task_mask': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_NODES,), dtype=np.float32)
        })
        
        # =====================================================================
        # [FIFO队列系统] 初始化队列容器
        # =====================================================================
        # 通信队列：每个发送实体维护两条并行FIFO队列（V2I与V2V）
        # key格式: ("VEH", vehicle_id) 或 ("RSU", rsu_id)
        self.txq_v2i = {}  # {tx_node: deque[TransferJob]}
        self.txq_v2v = {}  # {tx_node: deque[TransferJob]}
        
        # 计算队列：每个处理器维护一个FIFO队列
        # 车辆: veh_cpu_q[vehicle_id] = deque[ComputeJob]
        # RSU: rsu_cpu_q[rsu_id][processor_id] = deque[ComputeJob]
        self.veh_cpu_q = {}  # {vehicle_id: deque[ComputeJob]}
        self.rsu_cpu_q = {}  # {rsu_id: {processor_id: deque[ComputeJob]}}
        
        # EDGE去重：防止同一EDGE重复创建
        # key = (owner_vehicle_id, child_id, parent_id)
        self.active_edge_keys = set()
        
        # [数值trace] 端到端核验（默认关闭）
        self.DEBUG_TRACE_NUMERIC = False
        self.numeric_trace = []
        self.step_idx = 0
        
        # 能耗账本：严格分离INPUT/EDGE/本地计算/RSU计算
        self.E_tx_input_cost = {}    # INPUT上传能耗（纳入成本/奖励）
        self.E_tx_edge_record = {}   # EDGE传输能耗（仅记录，不纳入成本）
        self.E_cpu_local_cost = {}   # 本地计算能耗（纳入成本/奖励）
        self.CPU_cycles_local = {}   # 本地计算量（记录）
        self.CPU_cycles_rsu_record = {}  # RSU计算量（仅记录）
        
        # =====================================================================
        # 单位一致性检查（Units Sanity Check）
        # =====================================================================
        # 验证数据量（bits）和带宽（Hz -> bps）单位一致性
        mean_data_bits = (self.config.MIN_DATA + self.config.MAX_DATA) / 2  # bits
        mean_bandwidth = self.config.BW_V2I / max(self.config.NUM_VEHICLES // 5, 1)  # Hz (shared)
        # 假设SINR=10 (10dB) → log2(11)≈3.46
        typical_sinr = 10.0
        typical_rate_bps = mean_bandwidth * np.log2(1 + typical_sinr)  # bps
        typical_tx_time = mean_data_bits / typical_rate_bps  # seconds
        
        assert typical_tx_time > 0.005, (
            f"❌ 单位缩放错误：典型传输时间为 {typical_tx_time*1000:.2f}ms < 5ms！"
            f"请检查 DATA_SIZE (当前:{self.config.MIN_DATA:.2e}-{self.config.MAX_DATA:.2e} bits) "
            f"和 BW_V2I (当前:{self.config.BW_V2I:.2e} Hz) 的单位是否一致。"
        )
        assert typical_tx_time < 1.0, (
            f"⚠️  单位缩放警告：典型传输时间为 {typical_tx_time:.2f}s > 1s，"
            f"可能导致任务超时。考虑增加带宽或减少数据量。"
        )

    # =====================================================================
    # [P02修复] 统一队列查询方法 - 基于veh_cpu_q/rsu_cpu_q唯一事实源
    # =====================================================================

    def _get_veh_queue_load(self, veh_id: int) -> float:
        """
        获取车辆计算队列的总负载（cycles）

        [P02] 直接从veh_cpu_q计算，不使用capacity_tracker
        """
        if veh_id not in self.veh_cpu_q:
            return 0.0
        return sum(job.rem_cycles for job in self.veh_cpu_q[veh_id])

    def _get_veh_queue_wait_time(self, veh_id: int, cpu_freq: float = None) -> float:
        """
        获取车辆计算队列的估计等待时间（秒）

        [P02] 直接从veh_cpu_q计算
        """
        if cpu_freq is None:
            veh = self._get_vehicle_by_id(veh_id)
            cpu_freq = veh.cpu_freq if veh else self.config.MIN_VEHICLE_CPU_FREQ
        if cpu_freq <= 0:
            return 0.0
        load = self._get_veh_queue_load(veh_id)
        return load / cpu_freq

    def _get_rsu_queue_load(self, rsu_id: int, processor_id: int = None) -> float:
        """
        获取RSU计算队列的总负载（cycles）

        [P02] 直接从rsu_cpu_q计算

        Args:
            rsu_id: RSU ID
            processor_id: 处理器ID，如果为None则返回所有处理器总负载
        """
        if rsu_id not in self.rsu_cpu_q:
            return 0.0
        proc_dict = self.rsu_cpu_q[rsu_id]
        if processor_id is not None:
            if processor_id not in proc_dict:
                return 0.0
            return sum(job.rem_cycles for job in proc_dict[processor_id])
        # 所有处理器总负载
        total = 0.0
        for queue in proc_dict.values():
            total += sum(job.rem_cycles for job in queue)
        return total

    def _get_rsu_queue_wait_time(self, rsu_id: int) -> float:
        """
        获取RSU计算队列的估计等待时间（秒）- 取所有处理器中最小值

        [P02] 直接从rsu_cpu_q计算
        """
        if rsu_id not in self.rsu_cpu_q or rsu_id >= len(self.rsus):
            return 0.0
        rsu = self.rsus[rsu_id]
        cpu_freq = rsu.cpu_freq
        if cpu_freq <= 0:
            return 0.0

        proc_dict = self.rsu_cpu_q[rsu_id]
        if not proc_dict:
            return 0.0

        # 返回最小等待时间（FAT最早的处理器）
        min_wait = float('inf')
        for proc_id, queue in proc_dict.items():
            load = sum(job.rem_cycles for job in queue)
            wait = load / cpu_freq
            if wait < min_wait:
                min_wait = wait
        return min_wait if min_wait < float('inf') else 0.0

    def _refresh_f_max_const(self):
        """更新f_max常量上界（用于PBRS潜势归一化）"""
        vals = [
            getattr(self.config, "MAX_VEHICLE_CPU_FREQ", 0.0),
            getattr(self.config, "F_RSU", 0.0),
        ]
        vals.extend([getattr(v, "cpu_freq", 0.0) for v in self.vehicles])
        vals.extend([getattr(r, "cpu_freq", 0.0) for r in self.rsus])
        vals = [v for v in vals if v is not None]
        self._f_max_const = max(vals) if vals else 1.0

    def _rate_key(self, src_node, dst_node, link_type):
        """统一的速率key，确保phase3和reward使用同一标识"""
        return (
            link_type,
            src_node[0] if src_node else None,
            src_node[1] if src_node else None,
            dst_node[0] if dst_node else None,
            dst_node[1] if dst_node else None,
        )

    def _compute_pair_rate(self, src_node, dst_node, link_type, power_dbm=None):
        """仅在快照阶段调用，禁止在phase3/reward重复采样"""
        if src_node is None or dst_node is None:
            return 0.0
        if src_node[0] == "VEH":
            src_veh = self._get_vehicle_by_id(src_node[1])
            if src_veh is None:
                return 0.0
            if dst_node[0] == "VEH":
                dst_veh = self._get_vehicle_by_id(dst_node[1])
                if dst_veh is None:
                    return 0.0
                dst_pos = dst_veh.pos
            else:
                rsu_id = dst_node[1]
                rsu = self.rsus[rsu_id] if 0 <= rsu_id < len(self.rsus) else None
                dst_pos = rsu.position if rsu is not None else self.config.RSU_POS
            return self.channel.compute_one_rate(
                src_veh,
                dst_pos,
                link_type,
                self.time,
                power_dbm_override=power_dbm,
                v2i_user_count=self._estimate_v2i_users() if link_type == "V2I" else None,
            )
        # RSU作为发送端
        rsu_id = src_node[1]
        rsu = self.rsus[rsu_id] if 0 <= rsu_id < len(self.rsus) else None
        if dst_node[0] == "VEH":
            dst_veh = self._get_vehicle_by_id(dst_node[1])
            if dst_veh is None:
                return 0.0
            dst_pos = dst_veh.pos
        else:
            dst_rsu = self.rsus[dst_node[1]] if 0 <= dst_node[1] < len(self.rsus) else None
            dst_pos = dst_rsu.position if dst_rsu is not None else self.config.RSU_POS

        class RSUProxy:
            def __init__(self, position, tx_power_dbm):
                self.pos = position
                self.tx_power_dbm = tx_power_dbm

        rsu_proxy = RSUProxy(rsu.position if rsu is not None else self.config.RSU_POS, power_dbm)
        return self.channel.compute_one_rate(
            rsu_proxy, dst_pos, "V2I", self.time, power_dbm_override=power_dbm
        )

    def _capture_rate_snapshot(self, commit_plans):
        """在SNAPSHOT_PRE冻结本步速率，用于phase3与奖励"""
        rsu_pos_default = self.rsus[0].position if len(self.rsus) > 0 else self.config.RSU_POS
        # 1) 触发一次全局compute_rates（满足单次采样要求）
        rates_raw = self.channel.compute_rates(self.vehicles, rsu_pos_default)
        link_rates = {}

        def _add_pair(src_node, dst_node, link_type, power_dbm=None):
            key = self._rate_key(src_node, dst_node, link_type)
            if key in link_rates:
                return
            rate = self._compute_pair_rate(src_node, dst_node, link_type, power_dbm)
            rate = max(rate, getattr(self.config, "EPS_RATE", 1e-9))
            link_rates[key] = rate

        # 2) 现有通信队列中的所有job
        for q_dict, link_type in ((self.txq_v2i, "V2I"), (self.txq_v2v, "V2V")):
            for tx_node, queue in q_dict.items():
                for job in queue:
                    _add_pair(job.src_node, job.dst_node, link_type, getattr(job, "tx_power_dbm", None))

        # 3) 本步即将创建的INPUT传输（基于计划）
        for plan in commit_plans:
            if plan.get("subtask_idx") is None:
                continue
            tgt = plan.get("planned_target")
            if tgt is None or tgt == 'Local':
                continue
            src_node = ("VEH", plan["vehicle_id"])
            if isinstance(tgt, tuple) and tgt[0] == 'RSU':
                dst_node = ("RSU", tgt[1])
                link_type = "V2I"
            elif isinstance(tgt, int):
                dst_node = ("VEH", tgt)
                link_type = "V2V"
            else:
                continue
            power_dbm = plan.get("power_dbm", getattr(plan.get("vehicle"), "tx_power_dbm", None))
            _add_pair(src_node, dst_node, link_type, power_dbm)

        # 4) 潜在的EDGE传输（尚未入队，但拓扑已确定）
        for v in self.vehicles:
            dag = v.task_dag
            if not hasattr(dag, 'inter_task_transfers'):
                continue
            for child_id, parents_dict in dag.inter_task_transfers.items():
                child_exec_loc = dag.exec_locations[child_id] if child_id < len(dag.exec_locations) else None
                if child_exec_loc is None:
                    continue
                for parent_id, transfer_info in parents_dict.items():
                    if transfer_info.get('rem_bytes', 0) <= 0:
                        continue
                    parent_loc = dag.task_locations[parent_id] if parent_id < len(dag.task_locations) else None
                    if parent_loc is None or parent_loc == child_exec_loc:
                        continue

                    def location_to_node(loc):
                        if loc == 'Local':
                            return ("VEH", v.id)
                        if isinstance(loc, tuple) and loc[0] == 'RSU':
                            return ("RSU", loc[1])
                        if isinstance(loc, int):
                            return ("VEH", loc)
                        return None

                    src_node = location_to_node(parent_loc)
                    dst_node = location_to_node(child_exec_loc)
                    if src_node is None or dst_node is None:
                        continue
                    link_type = "V2I" if src_node[0] == "RSU" or dst_node[0] == "RSU" else "V2V"
                    _add_pair(src_node, dst_node, link_type, getattr(self.config, "TX_POWER_MAX_DBM", None))

        # 5) 候选集合的潜在链路（用于PBRS_KP_V2时延/势函数估计）
        for v in self.vehicles:
            candidate_set = self._last_candidate_set.get(v.id)
            tx_power_dbm = getattr(v, "tx_power_dbm", None)
            if candidate_set is None:
                rsu_id = self._last_rsu_choice.get(v.id)
                if rsu_id is not None and rsu_id >= 0:
                    _add_pair(("VEH", v.id), ("RSU", int(rsu_id)), "V2I", tx_power_dbm)
                continue
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            if len(ids) > 1 and len(mask) > 1 and bool(mask[1]):
                rsu_id = int(ids[1])
                if rsu_id >= 0:
                    _add_pair(("VEH", v.id), ("RSU", rsu_id), "V2I", tx_power_dbm)
            for idx in range(2, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id >= 0:
                        _add_pair(("VEH", v.id), ("VEH", cand_id), "V2V", tx_power_dbm)

        self._rate_snapshot = {
            "step": self.steps,
            "raw_rates": rates_raw,
            "links": link_rates,
        }
        self._rate_snapshot_step = self.steps
        self._rate_snapshot_token = (self.steps, id(self._rate_snapshot))

    def _clear_rate_snapshot(self):
        self._rate_snapshot = None
        self._rate_snapshot_step = -1
        self._rate_snapshot_token = None

    def _get_rate_from_snapshot(self, src_node, dst_node, link_type):
        snap = getattr(self, "_rate_snapshot", None)
        if snap is None or snap.get("step", -1) != self.steps:
            raise RuntimeError("[Assert] Snapshot missing when querying rate")
        key = self._rate_key(src_node, dst_node, link_type)
        rate = snap["links"].get(key)
        if rate is None:
            raise RuntimeError(f"[Assert] Snapshot rate missing for key={key}")
        if getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
            assert self._rate_snapshot_token is not None and id(self._rate_snapshot) == self._rate_snapshot_token[1], \
                "[Assert] Snapshot object mismatch in reward"
            self._debug_rate_token_reward = self._rate_snapshot_token
        return rate

    def _is_veh_queue_full(self, veh_id: int, new_task_cycles: float = 0) -> bool:
        """
        检查车辆队列是否已满（基于计算量限制）

        [P02] 直接从veh_cpu_q计算
        """
        current_load = self._get_veh_queue_load(veh_id)
        return (current_load + new_task_cycles) > self.config.VEHICLE_QUEUE_CYCLES_LIMIT

    def _is_rsu_queue_full(self, rsu_id: int, new_task_cycles: float = 0) -> bool:
        """
        检查RSU队列是否已满（所有处理器都满时返回True）

        [P02] 直接从rsu_cpu_q计算
        """
        if rsu_id not in self.rsu_cpu_q or rsu_id >= len(self.rsus):
            return True

        rsu = self.rsus[rsu_id]
        proc_dict = self.rsu_cpu_q[rsu_id]
        per_proc_limit = self.config.RSU_QUEUE_CYCLES_LIMIT / rsu.num_processors

        # 检查是否有任何处理器能接受新任务
        for proc_id, queue in proc_dict.items():
            load = sum(job.rem_cycles for job in queue)
            if (load + new_task_cycles) <= per_proc_limit:
                return False  # 至少一个处理器有空间
        return True  # 所有处理器都满

    def _get_node_delay(self, node):
        """
        统一获取节点的延迟估计

        [P02修复] 直接从veh_cpu_q/rsu_cpu_q计算，不依赖capacity_tracker

        Args:
            node: Vehicle or RSU实例
        Returns:
            float: 估计延迟（秒）
        """
        from envs.entities.vehicle import Vehicle
        from envs.entities.rsu import RSU

        if isinstance(node, Vehicle):
            return self._get_veh_queue_wait_time(node.id, node.cpu_freq)
        elif isinstance(node, RSU):
            return self._get_rsu_queue_wait_time(node.id)
        else:
            return 0.0

    def _plan_actions_snapshot(self, actions):
        plans = []
        max_schedule = max(1, int(getattr(self.config, "MAX_SCHEDULE_PER_STEP", 1)))
        inflight_limit = int(getattr(self.config, "MAX_INFLIGHT_SUBTASKS_PER_VEHICLE", 0))
        for i, v in enumerate(self.vehicles):
            schedule_limit = max_schedule
            if v.task_dag.is_finished or v.task_dag.is_failed:
                plan = {
                    "vehicle": v,
                    "vehicle_id": v.id,
                    "index": i,
                    "subtask_idx": None,
                    "extra_subtask_indices": [],
                    "task_comp": None,
                    "task_data": None,
                    "desired_target": None,
                    "desired_kind": "none",
                    "planned_target": None,
                    "planned_kind": "none",
                    "target_idx": None,
                    "illegal_reason": "task_done",
                    "power_ratio": None,
                    "power_dbm": None,
                }
                plans.append(plan)
                continue
            plan = {
                "vehicle": v,
                "vehicle_id": v.id,
                "index": i,
                "subtask_idx": None,
                "extra_subtask_indices": [],
                "task_comp": None,
                "task_data": None,
                "desired_target": None,
                "desired_kind": "none",
                "planned_target": None,
                "planned_kind": "none",
                "target_idx": None,
                "illegal_reason": None,
                "power_ratio": None,
                "power_dbm": None,
            }
            # 防御性处理：如果actions长度小于车辆数，缺失的动作默认回退到Local
            if i >= len(actions):
                act = {'target': 0, 'power': 1.0}
            else:
                act = actions[i]
            if act is None:
                plans.append(plan)
                continue

            # [改动C] 动作接口统一为 dict: {"target": int, "power": float in [0,1]}
            # 移除离散 power_level 分支，Agent 使用 Beta 分布输出连续功率 rho∈[0,1]
            if isinstance(act, dict):
                target_idx = int(act.get("target", 0))
                p_norm = float(np.clip(act.get("power", 1.0), 0.0, 1.0))
            else:
                # 兼容数组格式：[target_idx, power_ratio]，power 直接为连续值
                act_array = np.asarray(act).flatten()
                target_idx = int(act_array[0]) if len(act_array) > 0 else 0
                # [改动C] power 直接作为连续值 rho∈[0,1]，不再离散化
                p_norm = float(np.clip(act_array[1], 0.0, 1.0)) if len(act_array) > 1 else 1.0
            
            plan["target_idx"] = target_idx

            dag = v.task_dag
            if inflight_limit > 0:
                inflight_count = 0
                for idx, loc in enumerate(dag.exec_locations):
                    if loc is not None and dag.status[idx] < 3:
                        inflight_count += 1
                if inflight_count >= inflight_limit:
                    plan["illegal_reason"] = "inflight_limit"
                    plans.append(plan)
                    continue
                schedule_limit = min(schedule_limit, inflight_limit - inflight_count)
            if schedule_limit <= 0:
                plan["illegal_reason"] = "inflight_limit"
                plans.append(plan)
                continue

            subtask_idx = None
            extra_subtasks = []
            if schedule_limit == 1:
                subtask_idx = dag.get_top_priority_task()
            else:
                ready_mask = (dag.status == 1)
                unassigned_mask = np.array([loc is None for loc in dag.exec_locations])
                schedulable = np.where(ready_mask & unassigned_mask)[0]
                if len(schedulable) > 0:
                    priorities = np.array([dag.compute_task_priority(tid) for tid in schedulable])
                    order = schedulable[np.argsort(-priorities)]
                    subtask_idx = int(order[0])
                    for cand in order[1:]:
                        if len(extra_subtasks) >= schedule_limit - 1:
                            break
                        extra_subtasks.append(int(cand))
                else:
                    subtask_idx = None
            if subtask_idx is None:
                # [Stage 1] 细分 no_task 原因
                if dag.is_finished:
                    plan["illegal_reason"] = "no_task_dag_done"
                elif dag.is_failed:
                    plan["illegal_reason"] = "no_task_dag_failed"
                else:
                    # 进一步区分：所有任务阻塞 vs 所有READY任务已分配
                    action_mask = dag.get_action_mask()
                    ready_mask = (dag.status == 1)
                    if not np.any(ready_mask):
                        # 无 READY 任务 => 所有任务依赖阻塞
                        plan["illegal_reason"] = "no_task_blocked"
                    else:
                        # 有 READY 任务但已全部分配
                        plan["illegal_reason"] = "no_task_assigned"
                plans.append(plan)
                continue

            task_comp = v.task_dag.total_comp[subtask_idx] if subtask_idx < len(v.task_dag.total_comp) else self.config.MEAN_COMP_LOAD
            task_data = v.task_dag.total_data[subtask_idx] if subtask_idx < len(v.task_dag.total_data) else 0.0
            plan["subtask_idx"] = subtask_idx
            plan["extra_subtask_indices"] = list(extra_subtasks)
            plan["task_comp"] = task_comp
            plan["task_data"] = task_data

            raw_power = self.config.TX_POWER_MIN_DBM + p_norm * (self.config.TX_POWER_MAX_DBM - self.config.TX_POWER_MIN_DBM)
            plan["power_ratio"] = p_norm
            plan["power_dbm"] = np.clip(raw_power, self.config.TX_POWER_MIN_DBM, self.config.TX_POWER_MAX_DBM)

            desired_target = 'Local'
            desired_kind = "local"

            if target_idx >= self.config.MAX_TARGETS:
                plan["illegal_reason"] = "idx_out_of_range"
            elif target_idx == 0:
                desired_target = 'Local'
                desired_kind = "local"
            elif target_idx == 1:
                candidate_set = self._last_candidate_set.get(v.id)
                candidate_ids = candidate_set["ids"] if candidate_set is not None else None
                candidate_mask = candidate_set["mask"] if candidate_set is not None else None
                serving_rsu_id = getattr(v, "serving_rsu_id", None)
                if serving_rsu_id is None:
                    serving_rsu_id = self._update_serving_rsu(v)
                cached_choice = self._last_rsu_choice.get(v.id)
                if cached_choice is not None and cached_choice != serving_rsu_id:
                    raise RuntimeError(
                        f"[Assert] Non-serving RSU choice detected veh={v.id} "
                        f"cached={cached_choice} serving={serving_rsu_id}"
                    )
                if candidate_mask is not None and (target_idx >= len(candidate_mask) or not candidate_mask[target_idx]):
                    plan["illegal_reason"] = "masked_target"
                    desired_target = 'Local'
                    desired_kind = "local"
                elif serving_rsu_id is None:
                    plan["illegal_reason"] = "rsu_unavailable"
                    desired_target = 'Local'
                    desired_kind = "local"
                else:
                    if not (0 <= serving_rsu_id < len(self.rsus)):
                        raise RuntimeError(
                            f"[Assert] serving_rsu_id out of range veh={v.id} rsu_id={serving_rsu_id}"
                        )
                    rsu = self.rsus[serving_rsu_id]
                    if rsu is None:
                        plan["illegal_reason"] = "rsu_unavailable"
                        desired_target = 'Local'
                        desired_kind = "local"
                    elif not rsu.is_in_coverage(v.pos):
                        plan["illegal_reason"] = "rsu_out_of_coverage"
                        desired_target = 'Local'
                        desired_kind = "local"
                    elif self._is_rsu_queue_full(serving_rsu_id, task_comp):
                        plan["illegal_reason"] = "rsu_queue_full"
                        desired_target = 'Local'
                        desired_kind = "local"
                    else:
                        desired_target = ('RSU', serving_rsu_id)
                        desired_kind = "rsu"
            else:
                candidate_set = self._last_candidate_set.get(v.id)
                candidate_ids = candidate_set["ids"] if candidate_set is not None else None
                candidate_mask = candidate_set["mask"] if candidate_set is not None else None
                if candidate_ids is None or candidate_mask is None:
                    plan["illegal_reason"] = "no_candidate_cache"
                    desired_target = 'Local'
                    desired_kind = "local"
                elif target_idx >= len(candidate_ids) or target_idx >= len(candidate_mask):
                    plan["illegal_reason"] = "idx_out_of_range"
                    desired_target = 'Local'
                    desired_kind = "local"
                elif not candidate_mask[target_idx]:
                    plan["illegal_reason"] = "masked_target"
                    desired_target = 'Local'
                    desired_kind = "local"
                else:
                    neighbor_id = candidate_ids[target_idx]
                    if neighbor_id is None or neighbor_id < 0:
                        plan["illegal_reason"] = "id_mapping_fail"
                        desired_target = 'Local'
                        desired_kind = "local"
                    else:
                        assert 0 <= neighbor_id < self.config.NUM_VEHICLES and neighbor_id != v.id, "[Assert] neighbor id invalid"
                        target_veh = self._get_vehicle_by_id(neighbor_id)
                        if target_veh is None:
                            plan["illegal_reason"] = "id_mapping_fail"
                            desired_target = 'Local'
                            desired_kind = "local"
                        else:
                            desired_target = int(neighbor_id)
                            desired_kind = "v2v"

            plan["desired_target"] = desired_target
            plan["desired_kind"] = desired_kind
            plan["planned_target"] = desired_target
            plan["planned_kind"] = desired_kind
            plans.append(plan)

        # RSU conflict resolution (deterministic by vehicle_id)
        rsu_requests = {}
        for plan in plans:
            if plan["subtask_idx"] is None:
                continue
            if plan["illegal_reason"] is not None:
                continue
            if plan["desired_kind"] == "rsu":
                rsu_id = plan["desired_target"][1]
                rsu_requests.setdefault(rsu_id, []).append(plan)

        for rsu_id, reqs in rsu_requests.items():
            if not (0 <= rsu_id < len(self.rsus)):
                for plan in reqs:
                    plan["planned_target"] = 'Local'
                    plan["planned_kind"] = "local"
                    plan["illegal_reason"] = "rsu_unavailable"
                continue
            rsu = self.rsus[rsu_id]
            # [P02修复] 直接从rsu_cpu_q获取处理器负载，不使用capacity_tracker
            proc_dict = self.rsu_cpu_q.get(rsu_id, {})
            num_procs = rsu.num_processors
            per_proc_limit = self.config.RSU_QUEUE_CYCLES_LIMIT / num_procs
            proc_loads = []
            proc_sizes = []
            for pid in range(num_procs):
                queue = proc_dict.get(pid, [])
                proc_loads.append(sum(job.rem_cycles for job in queue))
                proc_sizes.append(len(queue))
            proc_limits = [per_proc_limit] * num_procs
            proc_caps = [100] * num_procs  # 默认队列任务数上限

            for plan in sorted(reqs, key=lambda p: p["vehicle_id"]):
                chosen_pid = None
                for pid, load in enumerate(proc_loads):
                    if proc_limits[pid] is not None:
                        can_accept = (load + plan["task_comp"]) <= proc_limits[pid]
                    else:
                        can_accept = (proc_sizes[pid] + 1) <= proc_caps[pid]
                    if not can_accept:
                        continue
                    if chosen_pid is None:
                        chosen_pid = pid
                    else:
                        if load < proc_loads[chosen_pid]:
                            chosen_pid = pid
                        elif load == proc_loads[chosen_pid] and pid < chosen_pid:
                            chosen_pid = pid
                if chosen_pid is None:
                    plan["planned_target"] = 'Local'
                    plan["planned_kind"] = "local"
                    plan["illegal_reason"] = "queue_full_conflict"
                else:
                    proc_loads[chosen_pid] += plan["task_comp"]
                    proc_sizes[chosen_pid] += 1

        # V2V conflict resolution (deterministic by vehicle_id)
        v2v_requests = {}
        for plan in plans:
            if plan["subtask_idx"] is None:
                continue
            if plan["illegal_reason"] is not None:
                continue
            if plan["desired_kind"] == "v2v":
                tgt_id = plan["desired_target"]
                v2v_requests.setdefault(tgt_id, []).append(plan)

        for tgt_id, reqs in v2v_requests.items():
            t_veh = self._get_vehicle_by_id(tgt_id)
            if t_veh is None:
                for plan in reqs:
                    plan["planned_target"] = 'Local'
                    plan["planned_kind"] = "local"
                    plan["illegal_reason"] = "id_mapping_fail"
                continue
            # [P02修复] 直接从veh_cpu_q获取队列负载，不使用capacity_tracker
            sim_load = self._get_veh_queue_load(tgt_id)
            sim_len = len(self.veh_cpu_q.get(tgt_id, []))
            limit_cycles = self.config.VEHICLE_QUEUE_CYCLES_LIMIT
            limit_size = 100  # 默认队列任务数上限

            for plan in sorted(reqs, key=lambda p: p["vehicle_id"]):
                if limit_cycles is not None:
                    can_accept = (sim_load + plan["task_comp"]) <= limit_cycles
                else:
                    can_accept = (sim_len + 1) <= limit_size
                if not can_accept:
                    plan["planned_target"] = 'Local'
                    plan["planned_kind"] = "local"
                    plan["illegal_reason"] = "queue_full_conflict"
                else:
                    sim_load += plan["task_comp"]
                    sim_len += 1

        return plans

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
        map_size = self.config.MAP_SIZE
        d_inter = map_size / num_rsu  # 等间距部署
        
        # 断言：确保全覆盖
        assert num_rsu * d_inter >= map_size, \
            f"RSU部署间距不足：{num_rsu} * {d_inter} = {num_rsu * d_inter} < {map_size}"
        
        # 验证部署间距满足覆盖约束
        rsu_range = self.config.RSU_RANGE
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
                cpu_freq=self.config.F_RSU,
                num_processors=getattr(Cfg, 'RSU_NUM_PROCESSORS', 4),
                queue_limit=100,  # 默认队列任务数上限
                coverage_range=rsu_range
            )
            self.rsus.append(rsu)

    def _get_nearest_rsu(self, position):
        """获取距离指定位置最近的RSU（委托给RSUSelector）"""
        return self.rsu_selector.get_nearest_rsu(position)

    def _get_all_rsus_in_range(self, position):
        """获取覆盖范围内所有RSU（委托给RSUSelector）"""
        return self.rsu_selector.get_all_rsus_in_range(position)

    def _update_serving_rsu(self, vehicle):
        """为车辆更新最近RSU (serving_rsu_id)，仅允许单RSU连接"""
        nearest = self._get_nearest_rsu(vehicle.pos)
        serving_id = nearest.id if nearest is not None else None
        prev_id = getattr(vehicle, "serving_rsu_id", None)
        vehicle.serving_rsu_id = serving_id
        if getattr(self.config, "DEBUG_ASSERT_ILLEGAL_ACTION", False) and prev_id != serving_id:
            print(f"[Debug] serving_rsu_id veh={vehicle.id} {prev_id} -> {serving_id}")
        return serving_id

    def _get_serving_rsu(self, vehicle):
        """获取车辆当前serving RSU实体与ID（若不在覆盖内返回None）"""
        rsu_id = getattr(vehicle, "serving_rsu_id", None)
        if rsu_id is None:
            rsu_id = self._update_serving_rsu(vehicle)
        if rsu_id is None:
            return None, None
        if not (0 <= rsu_id < len(self.rsus)):
            raise RuntimeError(f"[Assert] serving_rsu_id out of range veh={vehicle.id}, rsu_id={rsu_id}")
        rsu = self.rsus[rsu_id]
        if not rsu.is_in_coverage(vehicle.pos):
            vehicle.serving_rsu_id = None
            self._last_rsu_choice[vehicle.id] = None
            return None, None
        return rsu, rsu_id

    def _assert_serving_rsu(self, vehicle, rsu_id, context):
        """强制校验RSU目标必须等于serving_rsu_id"""
        serving_id = getattr(vehicle, "serving_rsu_id", None)
        if serving_id is None:
            return
        if rsu_id != serving_id:
            raise RuntimeError(
                f"[Assert] {context}: rsu_id {rsu_id} != serving_rsu_id {serving_id} (veh={vehicle.id})"
            )

    def _select_best_rsu(self, vehicle, task_comp, task_data):
        """
        选择当前车辆的最佳RSU（确定性规则）

        [改动B] metric 加入 CommWait_total_v2i，反映通信队列 backlog（含 EDGE 挤占）

        返回:
            tuple: (rsu_id, v2i_rate, wait_time, dist, contact_time)
        """
        if len(self.rsus) == 0:
            return None, 0.0, 0.0, 0.0, 0.0

        rsu, rsu_id = self._get_serving_rsu(vehicle)
        if rsu is None:
            return None, 0.0, 0.0, 0.0, 0.0

        if self._is_rsu_queue_full(rsu_id, task_comp):
            return None, 0.0, 0.0, 0.0, 0.0

        speed = np.linalg.norm(vehicle.vel)

        # [改动B] 计算 V2I 通信队列等待时间（含 EDGE 挤占效应）
        comm_wait = self._compute_comm_wait(vehicle.id)
        comm_wait_v2i = comm_wait['total_v2i']

        dist = rsu.get_distance(vehicle.pos)
        rate = self.channel.compute_one_rate(
            vehicle, rsu.position, 'V2I', self.time,
            v2i_user_count=self._estimate_v2i_users()
        )
        rate = max(rate, 1e-6)
        tx_time = task_data / rate if task_data > 0 else 0.0
        # [处理器共享] 使用新的延迟估算方法
        wait_time = self._get_node_delay(rsu)
        comp_time = task_comp / max(rsu.cpu_freq, 1e-6)
        # [改动B] T_finish_est = CommWait + CommTx + CPUWait + CPUExec
        metric = comm_wait_v2i + tx_time + wait_time + comp_time

        if speed > 0.1:
            contact_time = max(0.0, (rsu.coverage_range - dist) / speed)
        else:
            contact_time = self._max_rsu_contact_time

        # [改动B] 用 T_finish_est 与 contact_time 比较（RSU场景通常 contact 够长，但保留判断）
        if metric > contact_time and speed > 0.1:
            return None, 0.0, 0.0, 0.0, 0.0

        return rsu_id, rate, wait_time, dist, contact_time

    def _is_rsu_location(self, loc):
        """判断位置标识是否是RSU（委托给RSUSelector）"""
        return self.rsu_selector.is_rsu_location(loc)

    def _get_rsu_id_from_location(self, loc):
        """从位置标识中提取RSU ID（委托给RSUSelector）"""
        return self.rsu_selector.get_rsu_id_from_location(loc)

    def _get_rsu_position(self, rsu_id):
        """获取RSU的位置（委托给RSUSelector）"""
        return self.rsu_selector.get_rsu_position(rsu_id)

    def _get_vehicle_by_id(self, veh_id):
        """
        根据车辆ID获取车辆对象

        Args:
            veh_id: 车辆ID

        Returns:
            Vehicle or None
        """
        for veh in self.vehicles:
            if veh.id == veh_id:
                return veh
        return None
    
    def _get_total_W_remaining(self):
        """
        [P2辅助函数] 计算所有DAG的总剩余计算量（cycles）
        
        基于FIFO队列系统和DAG状态统计（不再使用active_task_manager）
        
        Returns:
            float: 总剩余计算量（cycles）
        """
        total_W = 0.0
        
        # 1. 队列中的任务（正在执行或等待执行）
        for veh_id, queue in self.veh_cpu_q.items():
            for job in queue:
                total_W += job.rem_cycles
        
        for rsu_id, proc_dict in self.rsu_cpu_q.items():
            for proc_id, queue in proc_dict.items():
                for job in queue:
                    total_W += job.rem_cycles
        
        # 2. DAG中未分配的任务（status < 2: PENDING或READY，尚未入队）
        for v in self.vehicles:
            dag = v.task_dag
            for i in range(dag.num_subtasks):
                if dag.status[i] < 2:  # PENDING(0) 或 READY(1)，尚未入计算队列
                    total_W += dag.rem_comp[i]
        
        return total_W
    
    def _get_total_active_tasks(self):
        """
        [P2辅助函数] 计算所有节点（车辆+RSU）的活跃任务总数
        
        基于FIFO队列系统统计（不再使用active_task_manager）
        
        Returns:
            int: 总活跃任务数（所有队列中的任务数）
        """
        total_active = 0
        
        # 车辆CPU队列
        for veh_id, queue in self.veh_cpu_q.items():
            total_active += len(queue)
        
        # RSU CPU队列
        for rsu_id, proc_dict in self.rsu_cpu_q.items():
            for proc_id, queue in proc_dict.items():
                total_active += len(queue)
        
        # 传输队列（INPUT + EDGE）
        for queue in self.txq_v2i.values():
            total_active += len(queue)
        for queue in self.txq_v2v.values():
            total_active += len(queue)
        
        return total_active

    def _is_system_idle(self):
        """
        判定系统是否完全空闲（无可调度、无在途、无队列）。
        """
        if self._get_total_active_tasks() > 0:
            return False
        for v in self.vehicles:
            dag = v.task_dag
            if dag.is_finished or dag.is_failed:
                continue
            ready_mask = (dag.status == 1)
            unassigned_mask = np.array([loc is None for loc in dag.exec_locations])
            if np.any(ready_mask & unassigned_mask):
                return False
            for idx, loc in enumerate(dag.exec_locations):
                if loc is not None and dag.status[idx] < 3:
                    return False
        return True

    def _build_vehicle(self, vehicle_id: int, start_time: float):
        x_pos = np.random.uniform(0, 0.3 * self.config.MAP_SIZE)
        lane_centers = [(k + 0.5) * self.config.LANE_WIDTH for k in range(self.config.NUM_LANES)]
        y_pos = np.random.choice(lane_centers)
        pos = np.array([x_pos, y_pos])

        v = Vehicle(vehicle_id, pos)
        v.cpu_freq = np.random.uniform(self.config.MIN_VEHICLE_CPU_FREQ, self.config.MAX_VEHICLE_CPU_FREQ)
        v.tx_power_dbm = self.config.TX_POWER_DEFAULT_DBM if hasattr(Cfg, 'TX_POWER_DEFAULT_DBM') else self.config.TX_POWER_MIN_DBM

        adj, prof, data, ddl, extra = self.dag_gen.generate_from_config(veh_f=v.cpu_freq)
        v.task_dag = DAGTask(0, adj, prof, data, ddl)
        v.task_dag.deadline_gamma = extra.get("deadline_gamma")
        v.task_dag.critical_path_cycles = extra.get("critical_path_cycles")
        v.task_dag.deadline_base_time = extra.get("deadline_base_time")
        v.task_dag.deadline_slack = extra.get("deadline_slack")
        v.task_dag.start_time = start_time

        v.capacity_tracker.clear()
        v.task_queue_len = 0
        v.last_scheduled_subtask = -1
        v.last_action_step = -1
        v.last_action_target = 'Local'
        v.subtask_reward_buffer = 0.0
        return v

    def _vehicle_has_active_jobs(self, vehicle_id: int) -> bool:
        if len(self.veh_cpu_q.get(vehicle_id, [])) > 0:
            return True
        for proc_dict in self.rsu_cpu_q.values():
            for queue in proc_dict.values():
                for job in queue:
                    if getattr(job, "owner_vehicle_id", None) == vehicle_id:
                        return True
        for q_dict in (self.txq_v2i, self.txq_v2v):
            for queue in q_dict.values():
                for job in queue:
                    if getattr(job, "owner_vehicle_id", None) == vehicle_id:
                        return True
        return False

    def _respawn_vehicle(self, vehicle_id: int):
        if vehicle_id in self.veh_cpu_q:
            self.veh_cpu_q[vehicle_id].clear()
        self.txq_v2i.pop(("VEH", vehicle_id), None)
        self.txq_v2v.pop(("VEH", vehicle_id), None)
        self._last_candidates.pop(vehicle_id, None)
        self._last_candidate_set.pop(vehicle_id, None)
        self._last_rsu_choice.pop(vehicle_id, None)

        new_v = self._build_vehicle(vehicle_id, start_time=self.time)
        for idx, v in enumerate(self.vehicles):
            if v.id == vehicle_id:
                self.vehicles[idx] = new_v
                break

    def _handle_dynamic_arrivals(self):
        rate = float(getattr(self.config, "VEHICLE_ARRIVAL_RATE", 0.0))
        if rate <= 0:
            return 0
        arrival_count = 0
        if not hasattr(self, "_next_vehicle_arrival_time"):
            self._next_vehicle_arrival_time = self.time + np.random.exponential(1.0 / rate)
        while self.time >= self._next_vehicle_arrival_time:
            candidate = None
            for v in self.vehicles:
                if v.task_dag.is_finished or v.task_dag.is_failed:
                    if not self._vehicle_has_active_jobs(v.id):
                        candidate = v
                        break
            if candidate is not None:
                self._respawn_vehicle(candidate.id)
                arrival_count += 1
            self._next_vehicle_arrival_time += np.random.exponential(1.0 / rate)
        return arrival_count
    
    def _update_rate_norm(self, rate, link_type):
        # 归一化模式已固化为static，此方法保留以兼容接口调用
        # Normalization mode is fixed to static; method kept for interface compatibility
        pass

    def _get_norm_rate(self, link_type):
        # 归一化模式已固化为static，直接返回静态常量
        # Normalization mode fixed to static; directly return static constants
        return self.config.NORM_MAX_RATE_V2I if link_type == 'V2I' else self.config.NORM_MAX_RATE_V2V

    def _compute_comm_wait(self, vehicle_id: int) -> dict:
        """
        [改动A核心] 计算车辆通信队列的等待时间（含 EDGE 挤占效应）

        基于 step 边界快照计算，时隙内状态固定（MDP 语义）。
        逐 job 计算剩余时间 t_rem = rem_bytes * 8 / R_hat(job)，
        R_hat 复用现有速率计算函数，保持口径一致。

        Args:
            vehicle_id: 车辆 ID

        Returns:
            dict: {
                'total_v2i': float,  # V2I 队列总等待时间 (s)
                'edge_v2i': float,   # V2I 队列中 EDGE 类型等待时间 (s)
                'total_v2v': float,  # V2V 队列总等待时间 (s)
                'edge_v2v': float,   # V2V 队列中 EDGE 类型等待时间 (s)
            }
        """
        result = {
            'total_v2i': 0.0,
            'edge_v2i': 0.0,
            'total_v2v': 0.0,
            'edge_v2v': 0.0,
        }

        src_node = ("VEH", vehicle_id)
        src_veh = self._get_vehicle_by_id(vehicle_id)
        if src_veh is None:
            return result

        v2i_user_count = self._estimate_v2i_users()

        # =====================================================================
        # V2I 队列：txq_v2i[src_node]
        # =====================================================================
        if src_node in self.txq_v2i:
            for job in self.txq_v2i[src_node]:
                # 计算 R_hat(job)：复用 channel.compute_one_rate
                # 获取目标位置
                if job.dst_node[0] == "RSU":
                    rsu_id = job.dst_node[1]
                    if 0 <= rsu_id < len(self.rsus):
                        dst_pos = self.rsus[rsu_id].position
                    else:
                        dst_pos = self.config.RSU_POS
                elif job.dst_node[0] == "VEH":
                    dst_veh = self._get_vehicle_by_id(job.dst_node[1])
                    if dst_veh is not None:
                        dst_pos = dst_veh.pos
                    else:
                        continue  # 目标车辆不存在，跳过
                else:
                    continue

                # 计算速率（使用 job 的功率）
                rate = self.channel.compute_one_rate(
                    src_veh, dst_pos, 'V2I', self.time,
                    v2i_user_count=v2i_user_count,
                    power_dbm_override=job.tx_power_dbm
                )
                rate = max(rate, 1e-6)

                # 计算剩余时间（rem_bytes 实为 bits，无需再乘8）
                t_rem = job.rem_bytes / rate
                result['total_v2i'] += t_rem

                if job.kind == "EDGE":
                    result['edge_v2i'] += t_rem
                

        # =====================================================================
        # V2V 队列：txq_v2v[src_node]
        # =====================================================================
        if src_node in self.txq_v2v:
            for job in self.txq_v2v[src_node]:
                # 获取目标位置
                if job.dst_node[0] == "VEH":
                    dst_veh = self._get_vehicle_by_id(job.dst_node[1])
                    if dst_veh is not None:
                        dst_pos = dst_veh.pos
                    else:
                        continue  # 目标车辆不存在，跳过
                elif job.dst_node[0] == "RSU":
                    rsu_id = job.dst_node[1]
                    if 0 <= rsu_id < len(self.rsus):
                        dst_pos = self.rsus[rsu_id].position
                    else:
                        dst_pos = self.config.RSU_POS
                else:
                    continue

                # 计算速率（使用 job 的功率）
                rate = self.channel.compute_one_rate(
                    src_veh, dst_pos, 'V2V', self.time,
                    power_dbm_override=job.tx_power_dbm
                )
                rate = max(rate, 1e-6)

                # 计算剩余时间（rem_bytes 实为 bits，无需再乘8）
                t_rem = job.rem_bytes / rate
                result['total_v2v'] += t_rem

                if job.kind == "EDGE":
                    result['edge_v2v'] += t_rem
                

        return result

    def _power_ratio_from_dbm(self, power_dbm):
        p_min = getattr(Cfg, "TX_POWER_MIN_DBM", power_dbm)
        p_max = getattr(Cfg, "TX_POWER_MAX_DBM", p_min)
        denom = p_max - p_min
        if denom <= 0:
            return 0.0
        return float(np.clip((power_dbm - p_min) / denom, 0.0, 1.0))

    def _get_p_max_watt(self, target):
        if target == 'Local':
            return 0.0
        if self._is_rsu_location(target):
            p_dbm = getattr(Cfg, "TX_POWER_UP_DBM", getattr(Cfg, "TX_POWER_MAX_DBM", getattr(Cfg, "TX_POWER_MIN_DBM", 20.0)))
        elif isinstance(target, int):
            p_dbm = getattr(Cfg, "TX_POWER_V2V_DBM", getattr(Cfg, "TX_POWER_MAX_DBM", getattr(Cfg, "TX_POWER_MIN_DBM", 20.0)))
        else:
            p_dbm = getattr(Cfg, "TX_POWER_MAX_DBM", getattr(Cfg, "TX_POWER_MIN_DBM", 20.0))
        return self.config.dbm2watt(p_dbm)

    def _build_task_locations_pi0(self, vehicle):
        num_tasks = vehicle.task_dag.num_subtasks
        task_locations = ['Local'] * num_tasks
        if hasattr(vehicle, 'exec_locations'):
            for i in range(num_tasks):
                if vehicle.task_dag.exec_locations[i] is not None:
                    task_locations[i] = vehicle.task_dag.exec_locations[i]
        if hasattr(vehicle.task_dag, 'task_locations'):
            for i in range(num_tasks):
                if task_locations[i] == 'Local' and vehicle.task_dag.task_locations[i] is not None:
                    task_locations[i] = vehicle.task_dag.task_locations[i]
        if vehicle.curr_subtask is not None and 0 <= vehicle.curr_subtask < num_tasks:
            task_locations[vehicle.curr_subtask] = vehicle.curr_target
        return task_locations

    def _compute_mean_cft_pi0(self, snapshot_time=None, v2i_user_count=None, vehicle_ids=None):
        if snapshot_time is None:
            snapshot_time = self.time
        if v2i_user_count is None:
            v2i_user_count = self._estimate_v2i_users()
        cft_list = []
        vehicles = self.vehicles
        if vehicle_ids is not None:
            vehicles = [self._get_vehicle_by_id(vid) for vid in vehicle_ids]
        for v in vehicles:
            if v is None:
                continue
            if v.task_dag.is_finished:
                cft_list.append(snapshot_time)
                continue
            task_locations = self._build_task_locations_pi0(v)
            try:
                from envs.modules.time_calculator import calculate_est_ct
                _, _, cft = calculate_est_ct(
                    v, v.task_dag, task_locations,
                    self.channel, self.rsus, self.vehicles, snapshot_time,
                    v2i_user_count=v2i_user_count
                )
                cft_list.append(cft)
            except Exception:
                cft_list.append(snapshot_time + 100.0)
        if not cft_list:
            return snapshot_time
        return float(np.mean(cft_list))

    def _compute_vehicle_cfts_snapshot(self, snapshot_time, vehicle_ids=None):
        vehicle_cfts = []
        vehicles = self.vehicles
        if vehicle_ids is not None:
            vehicles = [self._get_vehicle_by_id(vid) for vid in vehicle_ids]
        for v in vehicles:
            if v is None:
                vehicle_cfts.append(np.nan)
                continue
            if v.task_dag.is_finished:
                vehicle_cfts.append(snapshot_time)
                continue
            task_locations = self._build_task_locations_pi0(v)
            try:
                from envs.modules.time_calculator import calculate_est_ct
                _, _, cft = calculate_est_ct(
                    v,
                    v.task_dag,
                    task_locations,
                    self.channel,
                    self.rsus,
                    self.vehicles,
                    snapshot_time,
                    v2i_user_count=self._estimate_v2i_users(),
                )
                vehicle_cfts.append(cft)
            except Exception:
                vehicle_cfts.append(np.nan)
        return vehicle_cfts

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.vehicles = []
        self.time = 0.0
        self.steps = 0
        self._episode_steps = 0
        # [P2性能统计] Episode级统计清零
        self._p2_active_time = 0.0
        self._p2_idle_time = 0.0
        self._p2_deltaW_active = 0.0
        self._p2_zero_delta_steps = 0
        self._clear_rate_snapshot()
        self._debug_rate_token_phase3 = None
        self._debug_rate_token_reward = None
        self._episode_dT_eff_values = []
        self._episode_energy_norm_values = []
        self._episode_t_tx_values = []
        self._episode_task_durations = []
        self._pbrs_debug_records = []
        self._last_phi_debug = {}
        self._episode_illegal_count = 0
        self._episode_no_task_count = 0
        self._episode_illegal_reasons = {}
        self._idle_terminate_count = 0
        self._episode_obs_stats = {
            "steps": 0,
            "avail_L_sum": 0.0,
            "avail_R_sum": 0.0,
            "avail_V_sum": 0.0,
            "neighbor_sum": 0.0,
            "best_v2v_rate_sum": 0.0,
            "best_v2v_valid_sum": 0.0,
            "v2v_beats_rsu_sum": 0.0,
            "cost_gap_sum": 0.0,
            "cost_rsu_sum": 0.0,
            "cost_v2v_sum": 0.0,
            "cost_pair_count": 0.0,
        }
        self._last_candidates = {}
        self._last_candidate_set = {}
        self._last_rsu_choice = {}
        
        # =====================================================================
        # [FIFO队列系统] 清空所有队列与账本（防止跨episode污染）
        # =====================================================================
        self.txq_v2i = defaultdict(deque)
        self.txq_v2v = defaultdict(deque)
        self.veh_cpu_q = defaultdict(deque)
        self.rsu_cpu_q = defaultdict(lambda: defaultdict(deque))
        self.active_edge_keys = set()
        
        # FIFO能耗账本
        self.E_tx_input_cost = defaultdict(float)
        self.E_tx_edge_record = defaultdict(float)
        self.E_cpu_local_cost = defaultdict(float)
        self.CPU_cycles_local = defaultdict(float)
        self.CPU_cycles_rsu_record = defaultdict(float)
        # 预填充RSU CPU队列，避免空dict导致 _is_rsu_queue_full 误判为满
        for rsu in self.rsus:
            for pid in range(rsu.num_processors):
                _ = self.rsu_cpu_q[rsu.id][pid]
        # 重置RSU队列和FAT
        for rsu in self.rsus:
            rsu.clear_queue()
            rsu.reset_fat()
        if abs(self.time - self._cft_cache_time) > self.config.DT * 0.5:
            self._cft_cache = None
            self._cft_cache_valid = False

        for i in range(self.config.NUM_VEHICLES):
            v = self._build_vehicle(i, start_time=0.0)
            self.vehicles.append(v)
            self._vehicles_seen.add(v.id)
        
        # 道路模型：初始化动态车辆生成的下一辆到达时间（泊松过程）
        # 如果VEHICLE_ARRIVAL_RATE > 0，则启用动态生成
        if hasattr(Cfg, 'VEHICLE_ARRIVAL_RATE') and self.config.VEHICLE_ARRIVAL_RATE > 0:
            # 下一辆车的到达时间间隔服从指数分布：Δt ~ Exponential(λ)
            # 初始下一辆到达时间：从当前时间开始的第一个到达时间
            self._next_vehicle_arrival_time = np.random.exponential(1.0 / self.config.VEHICLE_ARRIVAL_RATE)
            self._next_vehicle_id = self.config.NUM_VEHICLES  # 车辆ID从初始数量开始
        else:
            self._next_vehicle_arrival_time = float('inf')  # 禁用动态生成
            self._next_vehicle_id = self.config.NUM_VEHICLES

        self.last_global_cft = self._calculate_global_cft_critical_path()
        # [P2性能统计] 初始化W_prev（在车辆生成后）
        self._p2_W_prev = self._get_total_W_remaining()
        self._refresh_f_max_const()

        # [Episode统计] 初始化episode级统计
        self._last_episode_metrics = {}
        self._decision_counts = {'local': 0, 'rsu': 0, 'v2v': 0}

        # [P03新增] 详细动作统计（用于诊断）
        self._p_target_raw = {'local': 0, 'rsu': 0, 'v2v': 0}  # policy输出的原始target类型
        self._p_target_effective = {'local': 0, 'rsu': 0, 'v2v': 0}  # 实际执行的target类型
        self._fallback_reasons = {}  # 非法原因计数 {reason: count}
        self._episode_delta_phi_values = []  # delta_phi值列表（用于p50/p95统计）
        self._episode_shape_clip_count = 0  # shape裁剪次数
        self._episode_r_total_clip_count = 0  # r_total裁剪次数
        self._episode_reward_count = 0  # 总奖励计数
        self._v2v_gain_sum = 0.0
        self._v2v_gain_pos_sum = 0.0
        self._v2v_gain_count = 0
        self._v2v_gain_pos_count = 0
        self._episode_illegal_count = 0
        self._episode_no_task_count = 0
        self._episode_illegal_reasons = {}

        return self._get_obs(), {}

    # =====================================================================
    # [FIFO队列系统] Phase 1-5 推进方法
    # =====================================================================
    
    def _try_enqueue_compute_if_ready(self, vehicle, subtask_id):
        """
        [兼容层] 委托到DAG完成处理器
        
        保留此方法以保持向后兼容，实际逻辑已迁移到DagCompletionHandler。
        """
        result = self._dag_handler._try_enqueue_compute_if_ready(
            vehicle, subtask_id, self.time, self.veh_cpu_q, self.rsu_cpu_q, self.rsus
        )
        return result is not None
    
    def _phase1_commit_offload_decisions(self, commit_plans):
        """
        [Phase1: Commit决策]
        
        职责：
        1. 写入exec_locations（权威事实源，表示"位置已确定"）
        2. 若target==Local：INPUT不入队列，标记input_ready，尝试入计算队列
        3. 若target!=Local：创建INPUT TransferJob并入对应通信队列
        
        硬断言：
        - Local目标不得创建INPUT TransferJob
        - exec_locations一旦写入不可改（由assign_task保证）
        """
        for plan in commit_plans:
            v = plan["vehicle"]
            subtask_indices = []
            if plan.get("subtask_idx") is not None:
                subtask_indices.append(plan["subtask_idx"])
            extra_subtasks = plan.get("extra_subtask_indices") or []
            subtask_indices.extend([idx for idx in extra_subtasks if idx is not None])
            actual_target = plan["planned_target"] if plan["planned_target"] is not None else 'Local'

            if not subtask_indices:
                continue

            for idx_pos, subtask_idx in enumerate(subtask_indices):
                if subtask_idx is None or subtask_idx < 0:
                    continue
                is_primary = (idx_pos == 0)

                # [Phase1职责] 写入exec_locations（由assign_task执行）
                assign_success = v.task_dag.assign_task(subtask_idx, actual_target)
                if not assign_success:
                    if is_primary:
                        v.illegal_action = True
                        v.illegal_reason = "assign_failed"
                    continue

                # 统计决策类型（已移除，使用StatsCollector）

                # [INPUT传输逻辑]
                task_data = v.task_dag.total_data[subtask_idx]

                if actual_target == 'Local':
                    # [Local路径] INPUT不入队列，视为input_ready（数据本地已存在）
                    v.task_dag.rem_data[subtask_idx] = 0.0
                    # 标记input_ready（可选字段，如果DAG支持）
                    if hasattr(v.task_dag, 'input_ready'):
                        v.task_dag.input_ready[subtask_idx] = True

                    # [关键修复] 清除同位置parent的pending EDGE传输
                    # 因为parent完成时child还没分配，创建了pending传输
                    # 现在child分配为Local，应该清除同位置（Local）parent的pending传输
                    if hasattr(v.task_dag, 'inter_task_transfers') and subtask_idx in v.task_dag.inter_task_transfers:
                        to_remove = []
                        for parent_id, transfer_info in v.task_dag.inter_task_transfers[subtask_idx].items():
                            parent_loc = v.task_dag.task_locations[parent_id]
                            if parent_loc == 'Local':  # 同位置：parent和child都在Local
                                to_remove.append(parent_id)
                        for parent_id in to_remove:
                            del v.task_dag.inter_task_transfers[subtask_idx][parent_id]
                        # 如果没有pending传输了，清理字典
                        if len(v.task_dag.inter_task_transfers[subtask_idx]) == 0:
                            del v.task_dag.inter_task_transfers[subtask_idx]
                            v.task_dag.waiting_for_data[subtask_idx] = False

                    # [Local路径] 立即尝试入计算队列（使用handler）
                    job = self._dag_handler._try_enqueue_compute_if_ready(
                        v, subtask_idx, self.time, self.veh_cpu_q, self.rsu_cpu_q, self.rsus
                    )

                else:
                    # [卸载路径] 创建INPUT TransferJob
                    # [关键修复] 先清除同位置parent的pending EDGE传输
                    # （与Local路径逻辑相同：parent完成时child还没分配，创建了pending传输；
                    #  现在child分配后，应该清除同位置parent的pending传输）
                    if hasattr(v.task_dag, 'inter_task_transfers') and subtask_idx in v.task_dag.inter_task_transfers:
                        to_remove = []
                        for parent_id, transfer_info in v.task_dag.inter_task_transfers[subtask_idx].items():
                            parent_loc = v.task_dag.task_locations[parent_id]
                            # 判断same_location: parent和child的位置编码相同
                            if parent_loc == actual_target:  # 同位置：parent和child都在同一位置（RSU或V2V peer）
                                to_remove.append(parent_id)
                        for parent_id in to_remove:
                            del v.task_dag.inter_task_transfers[subtask_idx][parent_id]
                        # 如果没有pending传输了，清理字典
                        if len(v.task_dag.inter_task_transfers[subtask_idx]) == 0:
                            del v.task_dag.inter_task_transfers[subtask_idx]
                            v.task_dag.waiting_for_data[subtask_idx] = False

                    # 确定src/dst节点
                    src_node = ("VEH", v.id)
                    if isinstance(actual_target, tuple) and actual_target[0] == 'RSU':
                        dst_node = ("RSU", actual_target[1])
                        link_type = "V2I"
                    elif isinstance(actual_target, int):
                        dst_node = ("VEH", actual_target)
                        # 判断link_type（V2I or V2V）
                        link_type = "V2V"
                    else:
                        # 异常情况，fallback到Local
                        if is_primary:
                            v.illegal_action = True
                            v.illegal_reason = "invalid_target"
                        continue

                    # 创建TransferJob
                    job = TransferJob(
                        kind="INPUT",
                        src_node=src_node,
                        dst_node=dst_node,
                        owner_vehicle_id=v.id,
                        subtask_id=subtask_idx,
                        rem_bytes=task_data,
                        tx_power_dbm=v.tx_power_dbm,  # INPUT使用动作映射功率
                        link_type=link_type,
                        enqueue_time=self.time,
                        parent_task_id=None  # INPUT无parent
                    )

                    # 入队到对应通信队列
                    if link_type == "V2I":
                        if src_node not in self.txq_v2i:
                            self.txq_v2i[src_node] = deque()
                        self.txq_v2i[src_node].append(job)
                    else:  # V2V
                        if src_node not in self.txq_v2v:
                            self.txq_v2v[src_node] = deque()
                        self.txq_v2v[src_node].append(job)
                    # snapshot 覆盖率检查
                    key = self._rate_key(src_node, dst_node, link_type)
                    if self._rate_snapshot is None or key not in self._rate_snapshot.get("links", {}):
                        raise RuntimeError(f"[Assert] Snapshot missing rate for new job key={key}, bytes={job.rem_bytes}, step={self.steps}")
    
    def _phase2_activate_edge_transfers(self):
        """
        [Phase2: 激活待传依赖边]
        
        职责：
        扫描每个DAG的inter_task_transfers，对于child_exec_loc已确定的边：
        1. 若parent_loc == child_loc：瞬时清零rem_bytes，不入队列
        2. 若parent_loc != child_loc：创建EDGE TransferJob（固定最大功率）
        
        硬断言：
        - child_exec_loc未确定（None）=> continue（绝对不创建/不推进/不清零）
        - 同位置EDGE不得入队列
        - EDGE唯一键不得重复（去重）
        """
        for v in self.vehicles:
            dag = v.task_dag
            if not hasattr(dag, 'inter_task_transfers'):
                continue
            
            # 扫描所有待传边
            for child_id, parents_dict in list(dag.inter_task_transfers.items()):
                # 获取child执行位置（必须已确定）
                child_exec_loc = v.task_dag.exec_locations[child_id] if child_id < len(v.task_dag.exec_locations) else None
                
                if child_exec_loc is None:
                    # [硬断言护栏] child未分配，绝不创建/推进/清零
                    continue
                
                # 扫描该child的所有parent边
                for parent_id, transfer_info in list(parents_dict.items()):
                    if transfer_info['rem_bytes'] <= 0:
                        continue  # 已完成
                    
                    # 获取parent完成位置
                    parent_task_loc = dag.task_locations[parent_id] if parent_id < len(dag.task_locations) else None
                    
                    if parent_task_loc is None:
                        # parent未完成，暂不处理（等待parent完成）
                        continue
                    
                    # [关键判断] 是否同位置
                    same_location = (parent_task_loc == child_exec_loc)
                    
                    if same_location:
                        # [同位置] 瞬时到齐，不入队列
                        transfer_info['rem_bytes'] = 0.0
                        # 调用DAG的边到齐函数（触发edge_ready检查）
                        dag.step_inter_task_transfers(child_id, 0.0, 0.0)
                    else:
                        # [不同位置] 创建EDGE TransferJob（固定最大功率）
                        
                        # [EDGE去重] 检查唯一键
                        edge_key = (v.id, child_id, parent_id)
                        if edge_key in self.active_edge_keys:
                            # 已存在，跳过（防止重复创建）
                            continue
                        
                        # 确定src/dst节点
                        def location_to_node(loc):
                            if loc == 'Local':
                                return ("VEH", v.id)
                            elif isinstance(loc, tuple) and loc[0] == 'RSU':
                                return ("RSU", loc[1])
                            elif isinstance(loc, int):
                                return ("VEH", loc)
                            else:
                                return None
                        
                        src_node = location_to_node(parent_task_loc)
                        dst_node = location_to_node(child_exec_loc)
                        
                        if src_node is None or dst_node is None:
                            continue  # 异常，跳过
                        
                        # 判断link_type
                        if src_node[0] == "RSU" or dst_node[0] == "RSU":
                            link_type = "V2I"
                        else:
                            link_type = "V2V"
                        
                        # 创建EDGE TransferJob（固定最大功率）
                        job = TransferJob(
                            kind="EDGE",
                            src_node=src_node,
                            dst_node=dst_node,
                            owner_vehicle_id=v.id,
                            subtask_id=child_id,
                            rem_bytes=transfer_info['rem_bytes'],
                            tx_power_dbm=self.config.TX_POWER_MAX_DBM,  # EDGE固定最大功率
                            link_type=link_type,
                            enqueue_time=self.time,
                            parent_task_id=parent_id
                        )
                        
                        # 入队到对应通信队列
                        if link_type == "V2I":
                            if src_node not in self.txq_v2i:
                                self.txq_v2i[src_node] = deque()
                            self.txq_v2i[src_node].append(job)
                        else:  # V2V
                            if src_node not in self.txq_v2v:
                                self.txq_v2v[src_node] = deque()
                            self.txq_v2v[src_node].append(job)
                        
                        # 标记已激活（防止重复）
                        self.active_edge_keys.add(edge_key)
    
    def _phase3_advance_comm_queues(self):
        """
        [Phase3: 推进通信队列]
        
        职责：
        对每个tx_node，并行推进txq_v2i和txq_v2v两条队列：
        1. 每条队列独立拥有DT时间预算
        2. FIFO串行：队头未完成，后续不推进
        3. work-conserving：队头完成后用剩余时间推进下一个
        4. 传输完成后调用finalize_transfer（INPUT入计算队列，EDGE清零）
        
        能耗记账（严格口径）：
        - INPUT发射能耗 => E_tx_input_cost[u]（纳入成本）
        - EDGE发射能耗 => E_tx_edge_record[u]（仅记录）
        """
        # 每步清零队列中job的step级统计，确保时间预算按步计算
        for q_dict in (self.txq_v2i, self.txq_v2v):
            for q in q_dict.values():
                for job in q:
                    job.step_time_used = 0.0
                    job.step_bytes_sent = 0.0
        # 合并所有tx_node并通过服务推进
        assert self._rate_snapshot is not None and self._rate_snapshot_step == self.steps, \
            "[Assert] rate snapshot missing before Phase3"
        # 补全所有队列job的速率（仍在同一步、同一次采样上下文）
        for q_dict, link_type in ((self.txq_v2i, "V2I"), (self.txq_v2v, "V2V")):
            for tx_node, queue in q_dict.items():
                for job in queue:
                    key = self._rate_key(job.src_node, job.dst_node, link_type)
                    if key not in self._rate_snapshot["links"]:
                        rate = self._compute_pair_rate(job.src_node, job.dst_node, link_type, getattr(job, "tx_power_dbm", None))
                        self._rate_snapshot["links"][key] = max(rate, getattr(self.config, "EPS_RATE", 1e-9))
        if getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
            assert self._rate_snapshot_token is None or self._rate_snapshot_token[0] == self.steps, \
                "[Assert] Rate snapshot token step mismatch before Phase3"
        comm_result = self._comm_service.step(
            self.txq_v2i,
            self.txq_v2v,
            self.config.DT,
            self.time,
            rate_fn=lambda job, tx_node: self._compute_job_rate(job, tx_node),
        )
        
        # 应用结果：能耗账本与完成回调
        for veh_id, delta in comm_result.energy_delta_cost.items():
            self.E_tx_input_cost[veh_id] = self.E_tx_input_cost.get(veh_id, 0.0) + delta
        for veh_id, delta in comm_result.energy_delta_record_edge.items():
            self.E_tx_edge_record[veh_id] = self.E_tx_edge_record.get(veh_id, 0.0) + delta
        # 使用DAG完成处理器处理传输完成
        for job in comm_result.completed_jobs:
            v = self._get_vehicle_by_id(job.owner_vehicle_id)
            if v is not None:
                dag = v.task_dag
                if dag.is_finished or dag.is_failed:
                    continue
                self._dag_handler.on_transfer_done(
                    job, v, self.time, self.active_edge_keys,
                    self.veh_cpu_q, self.rsu_cpu_q, self.rsus
                )
    
    def _compute_job_rate(self, job, tx_node):
        """
        计算TransferJob的传输速率；若存在本步速率快照，则直接复用，避免重复采样。
        """
        assert self._rate_snapshot is not None and self._rate_snapshot.get("step", -1) == self.steps, \
            "[Assert] missing rate snapshot in Phase3"
        key = self._rate_key(job.src_node, job.dst_node, job.link_type)
        snap_rate = self._rate_snapshot["links"].get(key)
        if snap_rate is not None:
            if getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
                assert self._rate_snapshot_token is not None and self._rate_snapshot_token[0] == self.steps, \
                    "[Assert] Snapshot token missing or step mismatch in Phase3"
                assert id(self._rate_snapshot) == self._rate_snapshot_token[1], \
                    "[Assert] Phase3 using different snapshot object"
                self._debug_rate_token_phase3 = self._rate_snapshot_token
            return snap_rate
        raise RuntimeError(f"[Assert] Rate snapshot miss in Phase3 key={key}")

    def _compute_job_rate_fresh(self, job, tx_node):
        """
        计算TransferJob的传输速率（无需快照回退路径）
        
        注意：
        - INPUT：使用job.tx_power_dbm（来自动作）
        - EDGE：使用job.tx_power_dbm=MAX（固定最大功率）
        - 通过power_dbm_override参数传递给channel
        """
        # 获取src和dst位置
        if tx_node[0] == "VEH":
            src_veh = self._get_vehicle_by_id(tx_node[1])
            if src_veh is None:
                return 0.0
            src_pos = src_veh.pos
        else:  # RSU
            rsu_id = tx_node[1]
            if 0 <= rsu_id < len(self.rsus):
                src_pos = self.rsus[rsu_id].position
            else:
                return 0.0
        
        if job.dst_node[0] == "VEH":
            dst_veh = self._get_vehicle_by_id(job.dst_node[1])
            if dst_veh is None:
                return 0.0
            dst_pos = dst_veh.pos
        else:  # RSU
            rsu_id = job.dst_node[1]
            if 0 <= rsu_id < len(self.rsus):
                dst_pos = self.rsus[rsu_id].position
            else:
                return 0.0
        
        # 计算速率
        # 注意：power_dbm_override允许覆盖功率（EDGE用）
        if tx_node[0] == "VEH":
            # 车辆作为发送端
            vehicle = src_veh
            rate = self.channel.compute_one_rate(
                vehicle, dst_pos, job.link_type, self.time,
                power_dbm_override=job.tx_power_dbm  # 显式传递功率
            )
        else:
            # RSU作为发送端（下行）
            # 使用channel的V2I速率计算，但需要构造proxy vehicle对象
            # 创建临时vehicle对象代表RSU的发送能力
            class RSUProxy:
                def __init__(self, position, tx_power_dbm):
                    self.pos = position
                    self.tx_power_dbm = tx_power_dbm
            
            rsu_proxy = RSUProxy(src_pos, job.tx_power_dbm)
            rate = self.channel.compute_one_rate(
                rsu_proxy, dst_pos, "V2I", self.time,
                power_dbm_override=job.tx_power_dbm
            )
        
        return rate
    
    def _finalize_transfer(self, job):
        """
        [兼容层] 委托到DAG完成处理器
        
        保留此方法以保持向后兼容，实际逻辑已迁移到DagCompletionHandler.on_transfer_done()。
        
        INPUT完成：
        1. 回写rem_data=0
        2. 若edge_ready，创建ComputeJob并入计算队列（时隙内联动）
        
        EDGE完成：
        1. 回写inter_task_transfers[child][parent].rem_bytes=0
        2. 调用DAG.step_inter_task_transfers触发edge_ready检查
        3. 清除active_edge_key
        """
        v = self._get_vehicle_by_id(job.owner_vehicle_id)
        if v is not None:
            self._dag_handler.on_transfer_done(
                job, v, self.time, self.active_edge_keys,
                self.veh_cpu_q, self.rsu_cpu_q, self.rsus
            )
    
    def _phase4_advance_cpu_queues(self):
        """
        [Phase4: 推进计算队列]
        
        职责：
        对每个处理器队列并行推进：
        1. FIFO串行：队头未完成，后续不推进
        2. work-conserving：队头完成后用剩余时间推进下一个
        3. 计算完成后调用finalize_compute（写task_locations，调用_mark_done）
        
        能耗记账（严格口径）：
        - 本地计算能耗 => E_cpu_local_cost[u]（纳入成本）
        - RSU计算：只记录cycles，不计入成本
        """
        # 通过服务推进所有计算队列，收集结果
        cpu_result = self._cpu_service.step(
            self.veh_cpu_q,
            self.rsu_cpu_q,
            self.config.DT,
            self.time,
            veh_cpu_hz_fn=lambda vid: getattr(self._get_vehicle_by_id(vid), "cpu_freq", self.config.MIN_VEHICLE_CPU_FREQ),
            rsu_cpu_hz_fn=lambda rid: self.rsus[rid].cpu_freq if 0 <= rid < len(self.rsus) else self.config.F_RSU,
        )

        # 统一应用结果
        for veh_id, delta in cpu_result.energy_delta_cost_local.items():
            self.E_cpu_local_cost[veh_id] = self.E_cpu_local_cost.get(veh_id, 0.0) + delta
        for veh_id, cycles in cpu_result.cycles_done_local.items():
            self.CPU_cycles_local[veh_id] = self.CPU_cycles_local.get(veh_id, 0.0) + cycles
        for rsu_id, cycles in cpu_result.cycles_done_rsu_record.items():
            self.CPU_cycles_rsu_record[rsu_id] = self.CPU_cycles_rsu_record.get(rsu_id, 0.0) + cycles
        # 使用DAG完成处理器处理计算完成（传递队列引用以便入队新解锁节点）
        for job in cpu_result.completed_jobs:
            v = self._get_vehicle_by_id(job.owner_vehicle_id)
            if v is not None:
                dag = v.task_dag
                if dag.is_finished or dag.is_failed:
                    continue
                self._dag_handler.on_compute_done(
                    job, v, self.time, 
                    veh_cpu_q=self.veh_cpu_q,
                    rsu_cpu_q=self.rsu_cpu_q,
                    rsus=self.rsus
                )
    
    # 兼容旧调用：委托到 cpu_service，用于测试/内部调用
    def _finalize_compute(self, job):
        """
        [兼容层] 委托到DAG完成处理器
        
        保留此方法以保持向后兼容，实际逻辑已迁移到DagCompletionHandler.on_compute_done()。
        
        【重要】位置编码一致性：
        - exec_locations 和 task_locations 都使用位置码：'Local' | ('RSU',id) | int(veh_id)
        - job.exec_node 是 node tuple: ("VEH",i) | ("RSU",j)，仅用于队列key
        - 必须从 exec_locations 读取位置码写入 task_locations
        """
        v = self._get_vehicle_by_id(job.owner_vehicle_id)
        if v is not None:
            self._dag_handler.on_compute_done(job, v, self.time)
    
    # =====================================================================
    # [主step方法] 调用5个Phase
    # =====================================================================
    
    def step(self, actions):
        self.steps += 1
        self._episode_steps += 1
        
        # 初始化决策统计（如果不存在）
        if not hasattr(self, '_decision_counts'):
            self._decision_counts = {'local': 0, 'rsu': 0, 'v2v': 0}

        # 清除旧的速率快照，避免跨步复用
        self._clear_rate_snapshot()
        self._debug_rate_token_phase3 = None
        self._debug_rate_token_reward = None

        snapshot_time = self.time  # 奖励时间轴：步前时间

        if abs(self.time - self._cache_time_step) > 1e-6:
            self._comm_rate_cache.clear()
            self._cache_time_step = self.time

        self._cft_cache = None
        self._cft_cache_valid = False
        self._dist_matrix_cache = None
        self._rsu_dist_cache.clear()

        ids_prev = [v.id for v in self.vehicles]
        v2i_users_prev = self._estimate_v2i_users()
        t_prev = snapshot_time
        cft_prev_abs = self._compute_mean_cft_pi0(
            snapshot_time=t_prev,
            v2i_user_count=v2i_users_prev,
            vehicle_ids=ids_prev
        )
        cft_prev_rem = max(cft_prev_abs - t_prev, 0.0) if cft_prev_abs is not None else 0.0
        if not np.isfinite(cft_prev_rem):
            cft_prev_rem = 0.0
        
        # 保存每辆车的CFT（用于per-vehicle奖励计算）
        vehicle_cfts_prev = self._compute_vehicle_cfts_snapshot(t_prev, vehicle_ids=ids_prev)

        step_congestion_cost = 0.0
        active_agents_count = 0

        for v in self.vehicles:
            v.illegal_action = False
            v.illegal_reason = None
        step_tx_time = {v.id: 0.0 for v in self.vehicles}
        step_power_ratio = {
            v.id: self._power_ratio_from_dbm(getattr(v, "tx_power_dbm", getattr(Cfg, "TX_POWER_MIN_DBM", 0.0)))
            for v in self.vehicles
        }

        # =====================================================================
        # [新FIFO队列系统] Phase 1-5 推进
        # =====================================================================
        
        # 解析动作并生成计划
        plans = self._plan_actions_snapshot(actions)
        no_task_reasons = {
            "no_task_dag_done",
            "no_task_dag_failed",
            "no_task_blocked",
            "no_task_assigned",
            "task_done",
        }
        active_agent_mask = []
        decision_step_mask = []
        no_task_step_mask = []
        for plan in plans:
            is_decision = plan.get("subtask_idx") is not None
            illegal_reason = plan.get("illegal_reason")
            is_no_task = bool(illegal_reason in no_task_reasons)
            decision_step_mask.append(bool(is_decision))
            active_agent_mask.append(1 if is_decision else 0)
            no_task_step_mask.append(bool(is_no_task))
        commit_plans = [p for p in plans if p["subtask_idx"] is not None]
        commit_plans.sort(key=lambda p: p["vehicle_id"])
        
        # 更新功率（在Phase1之前）
        for plan in commit_plans:
            v = plan["vehicle"]
            if plan["power_dbm"] is not None:
                v.tx_power_dbm = plan["power_dbm"]
                step_power_ratio[v.id] = plan["power_ratio"] if plan["power_ratio"] is not None else step_power_ratio.get(v.id, 0.0)
            
            if plan["illegal_reason"] is not None:
                v.illegal_action = True
                v.illegal_reason = plan["illegal_reason"]
            else:
                v.illegal_action = False
                v.illegal_reason = None

            # 统计决策分布（使用planned_kind而不是planned_target）
            kind = plan.get("planned_kind", "local")
            if kind == "local":
                self._decision_counts['local'] += 1
            elif kind == "rsu":
                self._decision_counts['rsu'] += 1
            elif kind == "v2v":
                self._decision_counts['v2v'] += 1

            # [P03新增] 统计p_target_raw/effective和fallback
            if plan["subtask_idx"] is not None:
                # 统计原始target类型（policy输出）
                desired_kind = plan.get("desired_kind", "local")
                self._p_target_raw[desired_kind] = self._p_target_raw.get(desired_kind, 0) + 1

                # 统计实际执行的target类型（可能因fallback而不同）
                self._p_target_effective[kind] = self._p_target_effective.get(kind, 0) + 1

                # 统计fallback原因
                if plan.get("illegal_reason"):
                    reason = plan["illegal_reason"]
                    self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1

        # 奖励快照/速率冻结（所有方案均需要，保证时隙冻结）
        scheme = getattr(self.config, "REWARD_SCHEME", "LEGACY_CFT")
        use_pbrs = scheme != "LEGACY_CFT"
        plan_by_vid = {p["vehicle_id"]: p for p in plans}
        reward_cache = {}
        self._refresh_f_max_const()
        self._capture_rate_snapshot(commit_plans)
        assert self._rate_snapshot is not None and self._rate_snapshot.get("step", -1) == self.steps, \
            "[Assert] rate snapshot missing before Phase3"
        for v in self.vehicles:
            dag = v.task_dag
            plan = plan_by_vid.get(v.id)
            subtask_idx = plan["subtask_idx"] if plan else None
            target = plan["planned_target"] if plan else 'Local'
            cycles = self._get_remaining_cycles(dag, subtask_idx) if subtask_idx is not None else 0.0
            t_local = 0.0
            t_actual = 0.0
            t_tx = 0.0
            if subtask_idx is not None:
                freq_self = max(getattr(v, "cpu_freq", self.config.MIN_VEHICLE_CPU_FREQ), 1e-9)
                t_local = self._get_veh_queue_wait_time(v.id, freq_self) + cycles / freq_self
                t_actual, t_tx = self._estimate_t_actual(
                    v,
                    subtask_idx,
                    target,
                    cycles,
                    plan.get("power_ratio") if plan else 1.0
                )
            if scheme == "PBRS_KP_V2":
                phi_prev, phi_debug = self._compute_phi_value_v2(dag, vehicle=v)
            else:
                phi_prev = self._compute_phi_value(dag, vehicle=v)
                phi_debug = {}
            reward_cache[v.id] = {
                "phi_prev": phi_prev,
                "phi_v2_debug": phi_debug,
                "finished_prev": dag.is_finished,
                "failed_prev": dag.is_failed,
                "subtask": subtask_idx,
                "target": target,
                "cycles": cycles,
                "t_local": t_local,
                "t_actual": t_actual,
                "t_tx": t_tx,
                "illegal": (plan and plan.get("illegal_reason") is not None) or getattr(v, "illegal_action", False),
                "illegal_reason": plan.get("illegal_reason") if plan else None,  # [Stage 1] 传播原因
                "power_ratio": plan.get("power_ratio") if plan else step_power_ratio.get(v.id, 0.0),
            }
            assert np.isfinite(reward_cache[v.id]["phi_prev"]), "[Assert] phi_prev not finite"
            assert np.isfinite(t_local) and np.isfinite(t_actual), "[Assert] time estimate not finite"
            if plan and plan.get("planned_kind") == "v2v" and subtask_idx is not None:
                delta_t = t_local - t_actual
                if np.isfinite(delta_t):
                    self._v2v_gain_sum += float(delta_t)
                    self._v2v_gain_count += 1
                    if delta_t > 0:
                        self._v2v_gain_pos_sum += float(delta_t)
                        self._v2v_gain_pos_count += 1

        # =====================================================================
        # [阶段1: 决策提交] Commit Decisions
        # 职责: 写入exec_locations + 创建INPUT传输任务
        # =====================================================================
        self._phase1_commit_offload_decisions(commit_plans)

        # =====================================================================
        # [阶段2: 边激活] Activate EDGE Transfers (首次)
        # 职责: 扫描pending边，为已分配child创建EDGE传输任务
        # =====================================================================
        self._phase2_activate_edge_transfers()

        # =====================================================================
        # [阶段3: 通信服务] Serve Communication Queues
        # 职责: FIFO并行推进V2I/V2V通信队列 (work-conserving)
        # =====================================================================
        self._phase3_advance_comm_queues()

        # =====================================================================
        # [阶段4: 计算服务] Serve Compute Queues
        # 职责: FIFO并行推进计算队列 (work-conserving)
        # 副作用: 任务完成时调用_mark_done()，可能创建新pending边
        # =====================================================================
        self._phase4_advance_cpu_queues()

        # =====================================================================
        # [阶段4.5: 边激活补偿] Activate EDGE Transfers (P01修复)
        # =====================================================================
        # 问题背景：
        #   当任务在阶段4完成时，_mark_done()会为其children创建inter_task_transfers
        #   但此时阶段2已执行，导致这些EDGE传输要等到下一step才能激活
        #   造成1个时隙的延迟，累积影响CFT计算准确性
        #
        # 修复方案：
        #   阶段4后再次调用边激活，处理刚创建的pending边
        #   该函数是幂等的（通过active_edge_keys去重），不会重复创建
        # =====================================================================
        self._phase2_activate_edge_transfers()

        # 队列长度同步（用于统计与可视化）
        for v in self.vehicles:
            queue = self.veh_cpu_q.get(v.id)
            if queue is not None:
                v.sync_capacity_from_queue(queue)
            else:
                v.task_queue_len = 0
        for rsu in self.rsus:
            proc_dict = self.rsu_cpu_q.get(rsu.id, {})
            rsu.sync_capacity_from_queues(proc_dict)

        # =====================================================================
        # [阶段5: 时间推进] Time Advance
        # 职责: 全局时间前进DT
        # =====================================================================
        self.time += self.config.DT

        # =====================================================================
        # [P03修复: Deadline检查] 在时间推进后立即检查所有任务的deadline
        # 修复问题: 原逻辑仅检查未完成任务，导致完成但超时的任务被误判为成功
        # 正确逻辑:
        #   1. 任务刚完成（is_finished且completion_time为None）：记录完成时间并检查
        #   2. 任务未完成且未失败：检查当前时间是否超过deadline
        # =====================================================================
        for v in self.vehicles:
            dag = v.task_dag
            if dag.deadline <= 0:
                continue  # 无deadline约束

            elapsed = self.time - dag.start_time

            # Case 1: 任务刚完成，记录完成时间并检查是否超时
            if dag.is_finished and dag.completion_time is None and not dag.is_failed:
                dag.completion_time = elapsed
                if not getattr(dag, "_completion_logged", False):
                    self._episode_task_durations.append(dag.completion_time)
                    dag._completion_logged = True
                if dag.completion_time > dag.deadline:
                    # 完成但超时，标记为失败
                    self._audit_deadline_checks += 1
                    self._audit_deadline_misses += 1
                    dag.set_failed(reason='deadline')
                    if hasattr(self, '_logger') and self._logger:
                        self._logger.warning(
                            f"[Deadline Miss-Completed] Vehicle{v.id}, DAG{dag.id}: "
                            f"completion_time={dag.completion_time:.3f}s > deadline={dag.deadline:.3f}s"
                        )

            # Case 2: 任务未完成且未失败，检查当前时间是否超过deadline
            elif not dag.is_finished and not dag.is_failed:
                self._audit_deadline_checks += 1
                if elapsed > dag.deadline:
                    self._audit_deadline_misses += 1
                    dag.set_failed(reason='deadline')
                    if hasattr(self, '_logger') and self._logger and not dag.timeout_logged:
                        dag.timeout_logged = True
                        self._logger.warning(
                            f"[Deadline Miss-Running] Vehicle{v.id}, DAG{dag.id}: "
                            f"elapsed={elapsed:.3f}s > deadline={dag.deadline:.3f}s, "
                            f"status_dist={np.bincount(dag.status, minlength=4)}"
                        )

        # =====================================================================
        # [车辆移动与动态管理]
        # =====================================================================
        for v in self.vehicles:
            # 更新车辆位置（道路模型：一维移动）
            v.update_pos(self.config.DT, self.config.MAP_SIZE)
        
        # 移除超出边界的车辆（道路模型：车辆超出道路长度L后移除）
        vehicles_to_remove = []

        rewards = []
        vehicle_cfts = self._compute_vehicle_cfts_snapshot(self.time)
        
        # 保存每个车辆的CFT（用于观测和奖励计算）
        self.vehicle_cfts = vehicle_cfts
        # 全局CFT使用所有车辆的最大值（用于兼容旧代码）
        self.last_global_cft = np.nanmax(vehicle_cfts) if len(vehicle_cfts) > 0 and np.any(np.isfinite(vehicle_cfts)) else np.nan
        v2i_users_curr = self._estimate_v2i_users()
        t_curr = self.time
        cft_curr_abs = self._compute_mean_cft_pi0(
            snapshot_time=t_curr,
            v2i_user_count=v2i_users_curr,
            vehicle_ids=ids_prev
        )
        cft_curr_rem = max(cft_curr_abs - t_curr, 0.0) if cft_curr_abs is not None else 0.0
        if not np.isfinite(cft_curr_rem):
            cft_curr_rem = 0.0
        cft_prev_rem = max(cft_prev_rem, 0.0)
        cft_curr_rem = max(cft_curr_rem, 0.0)
        dCFT_abs = float(cft_prev_abs - cft_curr_abs) if (cft_prev_abs is not None and cft_curr_abs is not None) else 0.0
        dT_rem = cft_prev_rem - cft_curr_rem
        dT = float(np.clip(dT_rem, self.config.DELTA_CFT_CLIP_MIN, self.config.DELTA_CFT_CLIP_MAX))
        dT_eff = dT - self.config.DT
        
        if scheme == "LEGACY_CFT":
            # 旧奖励路径（保持向后兼容）
            for i, v in enumerate(self.vehicles):
                dag = v.task_dag
                target = v.curr_target if v.curr_subtask is not None else None
                task_idx = v.curr_subtask if v.curr_subtask is not None else None
                if task_idx is None and getattr(v, 'last_action_step', -1) == self.steps:
                    pass  # 已清理
                    last_idx = getattr(v, 'last_scheduled_subtask', -1)
                    if 0 <= last_idx < dag.num_subtasks:
                        task_idx = last_idx
                        target = v.last_action_target
                if target is None:
                    target = 'Local'
                data_size = dag.total_data[task_idx] if task_idx is not None and task_idx < len(dag.total_data) else 0.0

                # 获取任务计算量（用于基于计算量的队列限制检查）
                task_comp = dag.total_comp[task_idx] if task_idx is not None and task_idx < len(dag.total_comp) else self.config.MEAN_COMP_LOAD
                power_ratio = float(np.clip(step_power_ratio.get(v.id, 0.0), 0.0, 1.0))
                t_tx_raw = float(step_tx_time.get(v.id, 0.0))
                if target == 'Local':
                    t_tx = 0.0
                else:
                    t_tx = float(np.clip(t_tx_raw, 0.0, self.config.DT))
                p_max_watt = self._get_p_max_watt(target)
                
                # 计算该车辆的CFT变化（per-vehicle reward）
                cft_v_prev = vehicle_cfts_prev[i] if i < len(vehicle_cfts_prev) else np.nan
                cft_v_curr = vehicle_cfts[i] if i < len(vehicle_cfts) else np.nan
                
                if np.isfinite(cft_v_prev) and np.isfinite(cft_v_curr):
                    cft_v_prev_rem = max(cft_v_prev - t_prev, 0.0)
                    cft_v_curr_rem = max(cft_v_curr - t_curr, 0.0)
                    dT_rem_v = cft_v_prev_rem - cft_v_curr_rem
                else:
                    # 如果CFT无效，使用全局CFT作为fallback
                    dT_rem_v = dT_rem
                
                reward_parts = None
                if getattr(v, 'illegal_action', False):
                    r = self.config.REWARD_MIN  # 非法动作给予最小奖励
                    components = {
                        "delay_norm": 0.0,
                        "energy_norm": 0.0,
                        "r_soft_pen": 0.0,
                        "r_timeout": 0.0,
                        "hard_triggered": False,
                    }
                    hard_triggered = False
                    reward_parts = compute_absolute_reward(
                        dT_rem_v, 0.0, power_ratio, self.config.DT, p_max_watt,
                        self.config.REWARD_MIN, self.config.REWARD_MAX, hard_triggered=True, illegal_action=True
                    )[1]
                    reward_parts["energy_norm"] = 0.0
                    r = self._clip_reward(r)
                else:
                    components = self._compute_cost_components(i, target, task_idx, task_comp)
                    hard_triggered = components.get("hard_triggered", False)
                    base_reward, reward_parts = compute_absolute_reward(
                        dT_rem_v, t_tx, power_ratio, self.config.DT, p_max_watt,
                        self.config.REWARD_MIN, self.config.REWARD_MAX, hard_triggered=hard_triggered, illegal_action=False
                    )
                    if hard_triggered:
                        reward_parts["energy_norm"] = 0.0
                    r = self._clip_reward(base_reward)
                if hasattr(v, 'subtask_reward_buffer'):
                    v.subtask_reward_buffer = 0.0

                self._episode_dT_eff_values.append(dT_eff)
                self._episode_energy_norm_values.append(reward_parts.get("energy_norm", 0.0) if reward_parts else 0.0)
                self._episode_t_tx_values.append(step_tx_time.get(v.id, 0.0))

                rewards.append(r)
        else:
            assert self._rate_snapshot is None or self._rate_snapshot.get("step", -1) == self.steps
            phi_list = []
            rem_list = []
            pbrs_step_flags = [] if getattr(self.config, "DEBUG_PBRS_AUDIT", False) else None
            for v in self.vehicles:
                dag = v.task_dag
                ctx = reward_cache.get(v.id, {
                    "phi_prev": self._compute_phi_value(dag, vehicle=v),
                    "finished_prev": dag.is_finished,
                    "failed_prev": dag.is_failed,
                    "subtask": None,
                    "target": 'Local',
                    "cycles": 0.0,
                    "t_local": 0.0,
                    "t_actual": 0.0,
                    "t_tx": 0.0,
                    "illegal": getattr(v, "illegal_action", False),
                    "power_ratio": step_power_ratio.get(v.id, 0.0),
                })
                # PBRS_KP_V2: latency advantage + LB shaping + timeout/power penalties
                if scheme == "PBRS_KP_V2":
                    phi_next, phi_debug = self._compute_phi_value_v2(dag, vehicle=v)
                else:
                    phi_next = self._compute_phi_value(dag, vehicle=v)
                    phi_debug = {}
                assert np.isfinite(phi_next) and np.isfinite(ctx.get("phi_prev", 0.0)), "[Assert] phi not finite"
                phi_list.append(phi_next)
                try:
                    rem_list.append(float(np.sum(dag.rem_comp[dag.status != 3])))
                except Exception:
                    pass

                delta_t = ctx.get("t_local", 0.0) - ctx.get("t_actual", 0.0)
                if not np.isfinite(delta_t):
                    delta_t = 0.0
                if scheme != "PBRS_KP_V2":
                    assert np.isfinite(delta_t), "[Assert] delta_t not finite"
                    if ctx.get("subtask") is not None and ctx.get("target") == 'Local' and not ctx.get("illegal"):
                        assert np.isclose(delta_t, 0.0, atol=1e-6), "[Assert] Local action delta_t should be ~0"

                r_base = 0.0
                raw_base = 0.0
                r_lat = 0.0
                lat_debug = {}
                if scheme == "PBRS_KP_V2":
                    r_lat, lat_debug = self._compute_latency_advantage(v, ctx)
                else:
                    r_base = self.config.REWARD_ALPHA * float(np.clip(delta_t / max(self.config.T_REF, 1e-9), -1.0, 1.0))
                    raw_base = self.config.REWARD_ALPHA * (delta_t / max(self.config.T_REF, 1e-9))

                # [Stage 1] 细分非法惩罚
                r_illegal = 0.0
                if ctx.get("illegal"):
                    illegal_reason = ctx.get("illegal_reason")
                    if illegal_reason in ["no_task_dag_done", "no_task_dag_failed", "task_done"]:
                        r_illegal = self.config.NO_TASK_PENALTY_DAG_DONE  # = 0.0
                    elif illegal_reason == "no_task_blocked":
                        r_illegal = self.config.NO_TASK_PENALTY_BLOCKED  # = 0.0
                    elif illegal_reason == "no_task_assigned":
                        r_illegal = self.config.NO_TASK_PENALTY_ASSIGNED  # = 0.0
                    else:
                        # 真正的非法动作（rsu_unavailable, idx_out_of_range 等）
                        r_illegal = self.config.ILLEGAL_PENALTY  # = -2.0

                r_term = 0.0
                if (not ctx.get("finished_prev")) and dag.is_finished:
                    r_term += self.config.TERMINAL_BONUS_SUCC
                if (not ctx.get("failed_prev")) and dag.is_failed:
                    r_term += self.config.TERMINAL_PENALTY_FAIL
                r_energy = 0.0
                r_power = 0.0
                e_tx = 0.0
                overtime_ratio = 0.0
                r_timeout = 0.0
                delta_phi = self.config.REWARD_GAMMA * phi_next - ctx.get("phi_prev", 0.0)
                raw_shape = self.config.REWARD_BETA * delta_phi
                r_shape = self.config.REWARD_BETA * float(np.clip(delta_phi, -self.config.SHAPE_CLIP, self.config.SHAPE_CLIP))

                if scheme == "PBRS_KP_V2":
                    target = ctx.get("target")
                    if not ctx.get("illegal") and target is not None and target != 'Local':
                        p_ratio = float(np.clip(ctx.get("power_ratio", 0.0), 0.0, 1.0))
                        p_watt = getattr(self.config, "P_MAX_WATT", 0.0) * p_ratio
                        e_tx = p_watt * ctx.get("t_tx", 0.0)
                        r_energy = -self.config.ENERGY_LAMBDA * float(np.tanh(e_tx / max(self.config.E_REF, 1e-9)))
                        r_power = -self.config.POWER_LAMBDA * float(p_ratio ** 2)
                    if dag.deadline > 0 and dag.is_failed and dag.fail_reason == 'deadline':
                        elapsed = self.time - dag.start_time
                        overtime_ratio = max((elapsed - dag.deadline) / dag.deadline, 0.0)
                        r_timeout = -self.config.TIMEOUT_L1 * np.tanh(self.config.TIMEOUT_K * overtime_ratio)
                        r_timeout += -self.config.TIMEOUT_L2 * float(max(overtime_ratio - self.config.TIMEOUT_O0, 0.0) ** 2)
                    r_total = r_lat + r_shape + r_timeout + r_energy + r_power + r_term + r_illegal
                else:
                    if getattr(self.config, "ENERGY_LAMBDA_PBRS", 0.0) > 0.0:
                        p_ratio = float(np.clip(ctx.get("power_ratio", 0.0), 0.0, 1.0))
                        p_watt = getattr(self.config, "P_MAX_WATT", 0.0) * p_ratio
                        e_tx = p_watt * ctx.get("t_tx", 0.0)
                        e_norm = e_tx / max(self.config.E_REF, 1e-9)
                        r_energy = -self.config.ENERGY_LAMBDA_PBRS * float(np.clip(e_norm, 0.0, self.config.E_CLIP))
                    if dag.deadline > 0 and dag.is_failed and dag.fail_reason == 'deadline':
                        elapsed = self.time - dag.start_time
                        overtime_ratio = max((elapsed - dag.deadline) / dag.deadline, 0.0)
                        r_timeout = -self.config.TIMEOUT_PENALTY_WEIGHT * np.tanh(
                            self.config.TIMEOUT_STEEPNESS * overtime_ratio
                        )
                    r_total = r_base + r_illegal + r_term + r_shape + r_energy + r_timeout

                r_total = float(np.clip(r_total, -self.config.R_CLIP, self.config.R_CLIP))
                assert np.isfinite(r_total) and np.isfinite(r_shape), "[Assert] reward not finite"
                if dag.is_finished:
                    assert np.isclose(phi_next, 0.0, atol=1e-6), "[Assert] Phi must be 0 when DAG finished"
                assert abs(r_total) <= self.config.R_CLIP + 1e-6, "[Assert] reward exceeded clip bound"

                if getattr(self.config, "DEBUG_PBRS_AUDIT", False):
                    if np.random.rand() < getattr(self.config, "DEBUG_PHI_MONO_PROB", 0.1):
                        tgt = ctx.get("target")
                        rsu_used = None
                        if isinstance(tgt, tuple) and tgt[0] == "RSU":
                            rsu_used = int(tgt[1])
                        print(
                            f"[PBRS] veh={v.id} phi_prev={ctx.get('phi_prev', 0.0):.6f} "
                            f"phi_next={phi_next:.6f} delta_phi={delta_phi:.6f} "
                            f"serving_rsu={getattr(v, 'serving_rsu_id', None)} rsu_used={rsu_used}"
                        )

                # PBRS诊断与非法统计
                if ctx.get("illegal"):
                    illegal_reason = ctx.get("illegal_reason")
                    no_task_reasons = {
                        "no_task_dag_done",
                        "no_task_dag_failed",
                        "no_task_blocked",
                        "no_task_assigned",
                        "task_done",
                    }
                    if illegal_reason in no_task_reasons:
                        self._episode_no_task_count += 1
                    else:
                        self._episode_illegal_count += 1
                    if illegal_reason:
                        self._episode_illegal_reasons[illegal_reason] = (
                            self._episode_illegal_reasons.get(illegal_reason, 0) + 1
                        )

                if getattr(self.config, "DEBUG_PBRS_AUDIT", False):
                    phi_prev = ctx.get("phi_prev", 0.0)
                    delta_t_norm = delta_t / max(self.config.T_REF, 1e-9)
                    base_clipped = abs(r_base - raw_base) > 1e-9
                    shape_clipped = abs(r_shape - raw_shape) > 1e-9
                    total_clipped = abs(r_total) >= self.config.R_CLIP - 1e-9
                    rsu_used = None
                    tgt = ctx.get("target")
                    if isinstance(tgt, tuple) and tgt[0] == "RSU":
                        rsu_used = int(tgt[1])
                    f_debug = self._last_phi_debug.get(v.id, {})
                    illegal_reason = ctx.get("illegal_reason")
                    no_task_reasons = {
                        "no_task_dag_done",
                        "no_task_dag_failed",
                        "no_task_blocked",
                        "no_task_assigned",
                        "task_done",
                    }
                    is_no_task_step = bool(ctx.get("illegal") and illegal_reason in no_task_reasons)
                    is_decision_step = ctx.get("subtask") is not None
                    active_agent_mask = bool(is_decision_step)
                    self._pbrs_debug_records.append({
                        "step": int(self._episode_steps),
                        "vehicle_id": int(v.id),
                        "delta_t": float(delta_t),
                        "delta_t_norm": float(delta_t_norm),
                        "phi_prev": float(phi_prev),
                        "phi_next": float(phi_next),
                        "delta_phi": float(delta_phi),
                        "r_base": float(r_base),
                        "r_lat": float(r_lat),
                        "r_shape": float(r_shape),
                        "r_illegal": float(r_illegal),
                        "r_term": float(r_term),
                        "r_energy": float(r_energy),
                        "r_power": float(r_power),
                        "r_timeout": float(r_timeout),
                        "r_total": float(r_total),
                        "base_clipped": bool(base_clipped),
                        "shape_clipped": bool(shape_clipped),
                        "total_clipped": bool(total_clipped),
                        "serving_rsu_id": int(getattr(v, "serving_rsu_id", -1) or -1),
                        "rsu_id_used": rsu_used if rsu_used is not None else -1,
                        "f_local": float(f_debug.get("f_local", 0.0)),
                        "f_serving_rsu": float(f_debug.get("f_serving_rsu", 0.0)),
                        "f_candidates_max": float(f_debug.get("f_candidates_max", 0.0)),
                        "f_max": float(f_debug.get("f_max", 0.0)),
                        "is_no_task_step": bool(is_no_task_step),
                        "is_decision_step": bool(is_decision_step),
                        "active_agent_mask": bool(active_agent_mask),
                    })
                    if pbrs_step_flags is not None:
                        pbrs_step_flags.append({
                            "vehicle_id": int(v.id),
                            "is_no_task_step": bool(is_no_task_step),
                            "is_decision_step": bool(is_decision_step),
                            "active_agent_mask": bool(active_agent_mask),
                        })
                rewards.append(r_total)

                if hasattr(self, '_reward_stats'):
                    def _add_metric(name, value):
                        if value is None:
                            return
                        try:
                            if not np.isfinite(value):
                                return
                        except Exception:
                            return
                        self._reward_stats.add_metric(name, value)

                    if scheme == "PBRS_KP_V2":
                        _add_metric("r_lat", r_lat)
                        _add_metric("r_shape", r_shape)
                        _add_metric("r_term", r_term)
                        _add_metric("r_illegal", r_illegal)
                        _add_metric("r_timeout", r_timeout)
                        _add_metric("r_energy", r_energy)
                        _add_metric("r_power", r_power)
                        _add_metric("r_total", r_total)
                        _add_metric("delta_phi", delta_phi)
                        _add_metric("overtime_ratio", overtime_ratio)
                        _add_metric("e_tx", e_tx)
                        _add_metric("t_L", lat_debug.get("t_L"))
                        _add_metric("t_R", lat_debug.get("t_R"))
                        _add_metric("t_V", lat_debug.get("t_V"))
                        _add_metric("t_a", lat_debug.get("t_a"))
                        _add_metric("t_alt", lat_debug.get("t_alt"))
                        _add_metric("A_t", lat_debug.get("A_t"))
                        if phi_debug:
                            _add_metric("cp_rem", phi_debug.get("cp_rem"))
                            _add_metric("f_max", phi_debug.get("f_max"))
                            _add_metric("d_cp_lb", phi_debug.get("d_cp_lb"))
                            _add_metric("rate_best", phi_debug.get("rate_best"))
                            _add_metric("comm_lb", phi_debug.get("comm_lb"))
                            _add_metric("queue_lb", phi_debug.get("queue_lb"))
                            _add_metric("lb", phi_debug.get("lb"))
                            _add_metric("phi", phi_debug.get("phi"))
                        if abs(r_shape - raw_shape) > 1e-9:
                            self._reward_stats.add_counter("r_shape_clipped", 1)
                        if abs(r_total) >= self.config.R_CLIP - 1e-9:
                            self._reward_stats.add_counter("r_total_clipped", 1)
                    else:
                        _add_metric("r_base", r_base)
                        _add_metric("r_shape", r_shape)
                        _add_metric("r_term", r_term)
                        _add_metric("r_illegal", r_illegal)
                        _add_metric("r_timeout", r_timeout)
                        _add_metric("r_total", r_total)
                        _add_metric("delta_phi", delta_phi)
                        if abs(r_base - raw_base) > 1e-9:
                            self._reward_stats.add_counter("r_base_clipped", 1)
                        if abs(r_shape - raw_shape) > 1e-9:
                            self._reward_stats.add_counter("r_shape_clipped", 1)
                        if abs(r_total) >= self.config.R_CLIP - 1e-9:
                            self._reward_stats.add_counter("r_total_clipped", 1)
                    # [Stage 1] 统计 no_task 细分分布
                    illegal_reason = ctx.get("illegal_reason")
                    if illegal_reason:
                        self._reward_stats.add_counter(f"illegal_{illegal_reason}", 1)
                p_ratio_raw = ctx.get("power_ratio", 0.0)
                if p_ratio_raw is None or not np.isfinite(p_ratio_raw):
                    p_ratio_raw = 0.0
                p_ratio = float(np.clip(p_ratio_raw, 0.0, 1.0))
                target = ctx.get("target") or 'Local'
                p_max_watt = self._get_p_max_watt(target)
                p_circuit = float(getattr(self.config, "P_CIRCUIT_WATT", 0.0))
                e_step = (p_ratio * p_max_watt + p_circuit) * float(self.config.DT)
                e_max = max((p_max_watt + p_circuit) * float(self.config.DT), 1e-12)
                energy_norm = float(np.clip(e_step / e_max, 0.0, 1.0))

                if hasattr(self, '_reward_stats'):
                    self._reward_stats.add_metric("power_ratio", p_ratio)

                self._episode_dT_eff_values.append(dT_eff)
                self._episode_energy_norm_values.append(energy_norm)
                self._episode_t_tx_values.append(ctx.get("t_tx", 0.0))
            if phi_list:
                phi_avg = float(np.mean(phi_list))
                total_rem = float(np.sum(rem_list)) if rem_list else 0.0
                self._check_phi_monotonicity(total_rem, phi_avg)

        # =====================================================================
        # [强制续航] Episode终止逻辑
        # =====================================================================
        # 设计原则：
        # - 不因任务完成而提前终止episode
        # - 让环境完整运行到MAX_TIME（MAX_STEPS * DT）
        # 优势：
        # 1. 反映长期平均性能（包括空闲期和新任务）
        # 2. 处理动态到达的新车辆任务
        # 3. 使Reward曲线更稳定、更真实
        # 4. 符合连续运行的真实场景（系统不会因为当前任务完成就停机）
        # =====================================================================
        all_finished = all(v.task_dag.is_finished for v in self.vehicles)
        time_limit_reached = self.steps >= self.config.MAX_STEPS
        allow_early_terminate = bool(getattr(self.config, "TERMINATE_ON_ALL_FINISHED", False))
        has_dynamic_arrival = getattr(self.config, "VEHICLE_ARRIVAL_RATE", 0.0) > 0
        is_idle = self._is_system_idle()

        terminated = False
        truncated = False
        terminated_reason = "none"
        if time_limit_reached:
            truncated = True
            terminated_reason = "time_limit"
        elif allow_early_terminate and not has_dynamic_arrival and all_finished:
            terminated = True
            terminated_reason = "success_all_done"
        elif allow_early_terminate and not has_dynamic_arrival and is_idle:
            terminated = True
            terminated_reason = "idle"
            self._idle_terminate_count += 1
        self._last_terminated_reason = terminated_reason
        
        # 在info中记录任务完成状态（用于分析）
        info = {
            'timeout': time_limit_reached,
            'all_finished': all_finished,
            'num_active_vehicles': len([v for v in self.vehicles if not v.task_dag.is_finished]),
            'terminated_trigger': terminated_reason,
            'terminated_reason': terminated_reason,
            'idle_terminate_count': int(getattr(self, "_idle_terminate_count", 0)),
        }
        info['active_agent_mask'] = list(active_agent_mask)
        info['decision_step_mask'] = list(decision_step_mask)
        info['no_task_step_mask'] = list(no_task_step_mask)
        
        # [审计系统] 收集本步审计数据
        info['audit_step_info'] = self._collect_audit_step_info(commit_plans)
        if getattr(self.config, "DEBUG_PBRS_AUDIT", False):
            if 'pbrs_step_flags' in locals() and pbrs_step_flags is not None:
                info['pbrs_step_flags'] = list(pbrs_step_flags)
        
        # [P2性能统计] 在每个step末尾累计统计（无论是否终止）
        W_curr = self._get_total_W_remaining()
        deltaW = max(0.0, self._p2_W_prev - W_curr)  # 防止数值抖动造成负值
        total_active = self._get_total_active_tasks()
        
        if total_active > 0:
            self._p2_active_time += self.config.DT
            self._p2_deltaW_active += deltaW
            # 检测长时间无推进
            if deltaW < 1e-6:  # 几乎没有推进
                self._p2_zero_delta_steps += 1
            else:
                self._p2_zero_delta_steps = 0
        else:
            self._p2_idle_time += self.config.DT
        
        self._p2_W_prev = W_curr
        
        # [一致性检查] 长时间无推进警告
        if self._p2_zero_delta_steps >= 50 and total_active > 0:
            import warnings
            warnings.warn(
                f"[P2警告] 连续{self._p2_zero_delta_steps}步活跃任务未推进，"
                f"total_active={total_active}, deltaW={deltaW:.2e}",
                UserWarning
            )
            self._p2_zero_delta_steps = 0  # 重置计数，避免重复警告

        # [记录episode统计] 每步都记录，但只在episode结束时写入文件
        self._log_episode_stats(terminated, truncated)
        
        # [P2/P0新增] 将关键健康指标写入info（供审计脚本使用）
        if hasattr(self, '_last_episode_metrics'):
            info['episode_metrics'] = self._last_episode_metrics.copy()
            # 同时直接在info顶层写入这些字段（向后兼容）
            info.update(self._last_episode_metrics)
        info['rate_snapshot_used'] = use_pbrs

        # [PBRS核验] 确保Phase3与奖励使用同一步快照
        if getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
            if self._debug_rate_token_phase3 is not None and self._debug_rate_token_reward is not None:
                assert self._debug_rate_token_phase3 == self._debug_rate_token_reward, (
                    f"[Assert] rate snapshot token mismatch: phase3={self._debug_rate_token_phase3} "
                    f"reward={self._debug_rate_token_reward}"
                )

        # 清理速率快照，避免跨步污染
        self._clear_rate_snapshot()
        if getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
            assert self._rate_snapshot is None and self._rate_snapshot_token is None, "[Assert] Snapshot not cleared at step end"

        if terminated or truncated:
            # [Miss Reason分解] 在episode结束时标记失败原因
            for v in self.vehicles:
                if v.task_dag.is_finished:
                    continue  # 已完成的跳过
                
                # 如果任务已标记failed但没有fail_reason，强制设为deadline
                if v.task_dag.is_failed and not v.task_dag.fail_reason:
                    v.task_dag.fail_reason = 'deadline'
                
                # 未标记failed的任务，根据情况设置原因
                if not v.task_dag.is_failed:
                    # 检查是否有illegal action
                    if hasattr(v, 'illegal_action') and v.illegal_action:
                        v.task_dag.set_failed(reason='illegal')
                    # 检查是否有overflow（队列满）
                    elif hasattr(v, 'illegal_reason') and v.illegal_reason and 'overflow' in v.illegal_reason.lower():
                        v.task_dag.set_failed(reason='overflow')
                    # 其他未标记的保留（在_log_episode_stats中会归为unfinished或truncated）

        arrival_count = 0
        if not terminated and not truncated:
            arrival_count = self._handle_dynamic_arrivals()
        info['arrival_count'] = int(arrival_count)

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

        def _queue_head_rem(queue):
            if not queue:
                return 0.0
            head = queue[0]
            return getattr(head, "rem_bytes", getattr(head, "rem_cycles", 0.0))

        def _tx_queue_summary(txq_dict):
            return tuple(sorted((k, len(q), _queue_head_rem(q)) for k, q in txq_dict.items()))

        def _cpu_queue_summary(cpu_dict):
            return tuple(sorted((k, len(q), _queue_head_rem(q)) for k, q in cpu_dict.items()))
        
        def _to_tuple(obj):
            if hasattr(obj, "tolist"):
                return tuple(obj.tolist())
            return tuple(obj)

        strict = getattr(self.config, "CFT_CACHE_STRICT_KEY", True)
        if strict:
            dag_status = tuple(_to_tuple(v.task_dag.status) for v in self.vehicles)
            dag_exec = tuple(_to_tuple(v.task_dag.exec_locations) for v in self.vehicles)
            dag_task_loc = tuple(_to_tuple(v.task_dag.task_locations) for v in self.vehicles)
            txq_v2i_summary = _tx_queue_summary(self.txq_v2i)
            txq_v2v_summary = _tx_queue_summary(self.txq_v2v)
            veh_cpu_summary = _cpu_queue_summary(self.veh_cpu_q)
            rsu_cpu_summary = tuple(sorted(
                (rid, _cpu_queue_summary(proc_dict))
                for rid, proc_dict in self.rsu_cpu_q.items()
            ))
            active_edge_len = len(self.active_edge_keys)
            current_state_hash = hash((
                round(self.time, 3),
                rsu_queue_state,
                tuple(round(v.pos[0], 2) for v in self.vehicles),
                tuple(round(v.pos[1], 2) for v in self.vehicles),
                tuple(v.task_queue_len for v in self.vehicles),
                tuple(v.curr_target if hasattr(v, 'curr_target') else None for v in self.vehicles),
                dag_status,
                dag_exec,
                dag_task_loc,
                txq_v2i_summary,
                txq_v2v_summary,
                veh_cpu_summary,
                rsu_cpu_summary,
                active_edge_len,
            ))
        else:
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

            for i in range(num_tasks):
                if v.task_dag.exec_locations[i] is not None:
                    task_locations[i] = v.task_dag.exec_locations[i]

            for i in range(num_tasks):
                if task_locations[i] is None:
                    task_locations[i] = 'Local'

            if v.curr_subtask is not None and 0 <= v.curr_subtask < num_tasks:
                task_locations[v.curr_subtask] = v.curr_target

            # [P02修复] 使用统一队列查询方法计算等待时间
            local_wait = self._get_veh_queue_wait_time(v.id, v.cpu_freq)
            # 多RSU场景：使用所有RSU中的最小等待时间
            if len(self.rsus) > 0:
                rsu_wait_global = min([self._get_rsu_queue_wait_time(rsu.id) for rsu in self.rsus])
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
                        # [P02修复] 使用统一队列查询方法
                        cpu_fat[i] = self._get_rsu_queue_wait_time(rsu_id)
                    else:
                        # 向后兼容：使用默认RSU频率
                        node_exec_times[i] = rem_comps[i] / self.config.F_RSU
                        cpu_fat[i] = rsu_wait_global
                    channel_fat[i] = 0.0
                elif isinstance(loc, int):
                    target_veh = self._get_vehicle_by_id(loc)
                    if target_veh is None:
                        target_veh = v
                    # [P02修复] 使用统一队列查询方法
                    wait_target = self._get_veh_queue_wait_time(target_veh.id, target_veh.cpu_freq)
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
                            rsu_pos = self.config.RSU_POS  # 默认使用配置位置（向后兼容）
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
            dist = np.linalg.norm(vehicle.pos - self.config.RSU_POS)
        
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
        if len(self.vehicles) == 0:
            self._last_obs_stamp = int(self._episode_steps)
            return obs_list
        dist_matrix = self._get_dist_matrix()
        vehicle_ids = [veh.id for veh in self.vehicles]
        step_avail_l = 0.0
        step_avail_r = 0.0
        step_avail_v = 0.0
        step_neighbor_sum = 0.0
        step_best_v2v_sum = 0.0
        step_best_v2v_valid = 0
        step_v2v_beats_rsu = 0.0
        step_cost_gap_sum = 0.0
        step_cost_rsu_sum = 0.0
        step_cost_v2v_sum = 0.0
        step_cost_pair_count = 0

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
            MAX_NODES = self.config.MAX_NODES
            node_dim = 7
            padded_node_feats = np.zeros((MAX_NODES, node_dim), dtype=np.float32)
            padded_node_feats[:num_nodes, :] = node_feats

            serving_rsu_id = self._update_serving_rsu(v)
            serving_rsu = None
            if serving_rsu_id is not None and 0 <= serving_rsu_id < len(self.rsus):
                serving_rsu = self.rsus[serving_rsu_id]
            rsu_pos_for_v2i = serving_rsu.position if serving_rsu is not None else None
            if rsu_pos_for_v2i is None:
                est_v2i_rate = 0.0
            else:
                est_v2i_rate = self.channel.compute_one_rate(
                    v, rsu_pos_for_v2i, 'V2I', curr_time=self.time,
                    v2i_user_count=self._estimate_v2i_users()
                )
            # [处理器共享] 使用新的延迟估算方法
            self_wait = self._get_node_delay(v)

            self_info = np.array([
                v.vel[0] * self._inv_max_velocity, v.vel[1] * self._inv_max_velocity,
                np.clip(self_wait * self._inv_max_wait, 0, 1),
                v.cpu_freq * self._inv_max_cpu,
                np.clip(est_v2i_rate * self._inv_max_rate_v2i, 0, 1),
                v.pos[0] * self._inv_map_size, v.pos[1] * self._inv_map_size
            ], dtype=np.float32)

            task_schedulable = v.task_dag.get_action_mask()
            
            # [新设计] 环境自动选择优先级最高的任务
            selected_subtask_idx = v.task_dag.get_top_priority_task()
            if selected_subtask_idx is None:
                # 没有可调度的任务，使用无效索引-1
                selected_subtask_idx = -1

            if selected_subtask_idx >= 0 and selected_subtask_idx < v.task_dag.num_subtasks:
                task_data_size = v.task_dag.total_data[selected_subtask_idx]
                task_comp_size = v.task_dag.total_comp[selected_subtask_idx]
            else:
                task_data_size = float(np.mean(v.task_dag.total_data)) if v.task_dag.total_data.size > 0 else 0.0
                task_comp_size = float(np.mean(v.task_dag.total_comp)) if v.task_dag.total_comp.size > 0 else self.config.MEAN_COMP_LOAD

            rsu_id, rsu_rate, rsu_wait, rsu_dist, rsu_contact = self._select_best_rsu(
                v, task_comp_size, task_data_size
            )
            rsu_available = rsu_id is not None
            rsu_load_norm = np.clip(rsu_wait * self._inv_max_wait, 0, 1) if rsu_available else 0.0
            rsu_total_time = None
            if rsu_available:
                rsu_cpu = self.rsus[rsu_id].cpu_freq if (self.rsus and rsu_id < len(self.rsus)) else self.config.F_RSU
                rsu_tx_time = (task_data_size / max(rsu_rate, 1e-6)) if task_data_size > 0 else 0.0
                rsu_comp_time = task_comp_size / max(rsu_cpu, 1e-6)
                rsu_total_time = (rsu_tx_time + rsu_wait + rsu_comp_time) * 1.0
            serving_rsu_id_val = -1 if serving_rsu_id is None else int(serving_rsu_id)
            serving_rsu_onehot = np.zeros(self.config.NUM_RSU, dtype=np.float32)
            if serving_rsu_id is not None and 0 <= serving_rsu_id < self.config.NUM_RSU:
                serving_rsu_onehot[serving_rsu_id] = 1.0
            serving_rsu_info = np.zeros(5, dtype=np.float32)
            if serving_rsu is not None:
                serving_dist = serving_rsu.get_distance(v.pos)
                serving_wait = self._get_node_delay(serving_rsu)
                serving_cpu = serving_rsu.cpu_freq
                serving_speed = np.linalg.norm(v.vel)
                if serving_speed > 0.1:
                    serving_contact = max(0.0, (serving_rsu.coverage_range - serving_dist) / serving_speed)
                else:
                    serving_contact = self._max_rsu_contact_time
                serving_rsu_info = np.array([
                    serving_cpu * self._inv_max_cpu,
                    np.clip(serving_wait * self._inv_max_wait, 0, 1),
                    np.clip(serving_dist / max(self.config.RSU_RANGE, 1e-6), 0, 1),
                    np.clip(est_v2i_rate * self._inv_max_rate_v2i, 0, 1),
                    np.clip(serving_contact / max(self._max_rsu_contact_time, 1e-6), 0, 1),
                ], dtype=np.float32)

            neighbor_dim = 8
            neighbors_array = np.zeros((self.config.MAX_NEIGHBORS, neighbor_dim), dtype=np.float32)
            candidate_info = []

            # [改动B] 在候选筛选前计算 CommWait，用于 T_finish_est
            comm_wait_for_mask = self._compute_comm_wait(v.id)
            comm_wait_v2v_for_mask = comm_wait_for_mask['total_v2v']

            for j, other in enumerate(self.vehicles):
                if v.id == other.id:
                    continue
                dist = dist_matrix[v_idx, j]
                if dist > self.config.V2V_RANGE:
                    continue

                # [P02修复] 使用统一队列查询方法
                if self._is_veh_queue_full(other.id, task_comp_size):
                    continue

                # [改动B] 使用最大功率计算传输时间（最乐观估计）
                est_v2v_rate_max_power = self.channel.compute_one_rate(
                    v, other.pos, 'V2V', self.time,
                    power_dbm_override=self.config.TX_POWER_MAX_DBM
                )
                est_v2v_rate_max_power = max(est_v2v_rate_max_power, 1e-6)
                trans_time_max_power = task_data_size / est_v2v_rate_max_power if task_data_size > 0 else 0.0

                # [处理器共享] 使用新的延迟估算方法
                queue_wait_time = self._get_node_delay(other)
                comp_time = task_comp_size / max(other.cpu_freq, 1e-6)

                # [改动B] T_finish_est = CommWait + CommTx + CPUWait + CPUExec
                # 使用最大功率（最乐观）做可行性判断
                t_finish_est = comm_wait_v2v_for_mask + trans_time_max_power + queue_wait_time + comp_time

                rel_vel = other.vel - v.vel
                pos_diff = other.pos - v.pos
                pos_diff_norm = np.linalg.norm(pos_diff)
                if pos_diff_norm < 1e-6:
                    time_to_break = self._max_v2v_contact_time
                else:
                    rel_vel_proj = np.dot(rel_vel, pos_diff) / pos_diff_norm
                    if rel_vel_proj > 0.1:
                        time_to_break = (self.config.V2V_RANGE - dist) / rel_vel_proj
                    else:
                        time_to_break = self._max_v2v_contact_time

                # [改动B] 使用 T_finish_est 与 contact_time 比较
                if t_finish_est > time_to_break:
                    continue

                # 保存用于排序和显示的 rate（使用默认功率）
                est_v2v_rate = self.channel.compute_one_rate(v, other.pos, 'V2V', self.time)
                est_v2v_rate = max(est_v2v_rate, 1e-6)
                trans_time = task_data_size / est_v2v_rate if task_data_size > 0 else 0.0

                rel_pos = (other.pos - v.pos) * self._inv_v2v_range
                candidate_info.append({
                    'id': other.id,
                    'dist': dist,
                    'rel_pos': rel_pos,
                    'vel': other.vel,
                    'queue_wait': queue_wait_time,
                    'cpu_freq': other.cpu_freq,
                    'rate': est_v2v_rate,
                    'contact_time': max(time_to_break, 0.0),
                    'total_time': t_finish_est  # [改动B] 使用 T_finish_est 排序
                })

            candidate_set = self.candidate_manager.build_candidate_set(
                v, candidate_info, serving_rsu_id
            )
            candidate_ids = candidate_set["ids"]
            candidate_types = candidate_set["types"]
            target_mask_row = candidate_set["mask"].copy()
            v2v_slots = candidate_set["v2v_slots"]
            neighbor_count = sum(1 for info in v2v_slots if info is not None)

            for idx, info in enumerate(v2v_slots):
                if info is None:
                    continue
                neighbors_array[idx] = [
                    info['id'], info['rel_pos'][0], info['rel_pos'][1],
                    info['vel'][0] * self._inv_max_velocity, info['vel'][1] * self._inv_max_velocity,
                    np.clip(info['queue_wait'] * self._inv_max_wait, 0, 1),
                    info['cpu_freq'] * self._inv_max_cpu,
                    np.clip(info['rate'] * self._inv_max_rate_v2v, 0, 1)
                ]

            # [关键] 死锁兜底：如果所有目标都不可用，强制开启Local
            if not np.any(target_mask_row):
                target_mask_row[0] = True
            
            # [审计] 保存mask到vehicle对象，用于审计收集
            v._last_action_mask = target_mask_row.copy()

            resource_id_list = np.zeros(self.config.MAX_TARGETS, dtype=np.int64)
            resource_id_list[0] = 1
            resource_id_list[1] = 2
            for idx in range(self.config.MAX_NEIGHBORS):
                cid = candidate_ids[2 + idx]
                if cid >= 0:
                    resource_id_list[2 + idx] = 3 + cid

            padded_target_mask = target_mask_row.copy()

            step_avail_l += 1.0 if target_mask_row[0] else 0.0
            step_avail_r += 1.0 if target_mask_row[1] else 0.0
            if self.config.MAX_NEIGHBORS > 0:
                step_avail_v += float(np.mean(target_mask_row[2:]))
            step_neighbor_sum += float(neighbor_count)
            if neighbor_count > 0:
                best_rate = max(info['rate'] for info in v2v_slots if info is not None)
                step_best_v2v_sum += float(best_rate)
                step_best_v2v_valid += 1
            if neighbor_count > 0 and rsu_total_time is not None:
                min_v2v_time = min(info["total_time"] for info in v2v_slots if info is not None)
                if min_v2v_time < rsu_total_time:
                    step_v2v_beats_rsu += 1.0
                step_cost_gap_sum += float(min_v2v_time - rsu_total_time)
                step_cost_rsu_sum += float(rsu_total_time)
                step_cost_v2v_sum += float(min_v2v_time)
                step_cost_pair_count += 1

            self._last_candidates[v.id] = list(candidate_ids[2:])
            self._last_candidate_set[v.id] = candidate_set
            self._last_rsu_choice[v.id] = serving_rsu_id

            # [改动A] 计算通信队列等待时间（含 EDGE 挤占效应）
            comm_wait = self._compute_comm_wait(v.id)
            comm_wait_total_v2i = comm_wait['total_v2i']
            comm_wait_edge_v2i = comm_wait['edge_v2i']
            comm_wait_total_v2v = comm_wait['total_v2v']
            comm_wait_edge_v2v = comm_wait['edge_v2v']

            # 归一化 CommWait（使用 log(1+x) 压缩防止饱和）
            norm_max_comm_wait = getattr(self.config, 'NORM_MAX_COMM_WAIT', 2.0)
            comm_wait_total_v2i_norm = np.clip(np.log1p(comm_wait_total_v2i) / np.log1p(norm_max_comm_wait), 0, 1)
            comm_wait_edge_v2i_norm = np.clip(np.log1p(comm_wait_edge_v2i) / np.log1p(norm_max_comm_wait), 0, 1)
            resource_raw = np.zeros((self.config.MAX_TARGETS, self.config.RESOURCE_RAW_DIM), dtype=np.float32)
            slack_norm = val_urgency

            # Local节点特征：计算时间预估
            local_est_exec = task_comp_size / max(v.cpu_freq, 1e-6) if task_comp_size > 0 else 0.0
            local_est_comm = 0.0  # Local无传输
            local_est_wait = self_wait

            resource_raw[0] = [
                v.cpu_freq * self._inv_max_cpu,
                np.clip(self_wait * self._inv_max_wait, 0, 1),
                0.0,  # 距离为0
                0.0,  # [修复] Local无传输，Rate=0而非1
                0.0,  # 相对位置X
                0.0,  # 相对位置Y
                v.vel[0] * self._inv_max_velocity,
                v.vel[1] * self._inv_max_velocity,
                1.0,  # Node_Type = 1 (Local)
                slack_norm,
                1.0,  # Contact永久连接
                np.clip(local_est_exec / 10.0, 0, 1),  # Est_Exec_Time
                np.clip(local_est_comm / 10.0, 0, 1),  # Est_Comm_Time
                np.clip(local_est_wait / 10.0, 0, 1),  # Est_Wait_Time
            ]

            if rsu_available:
                rsu = self.rsus[rsu_id]
                rel_rsu = (rsu.position - v.pos) * self._inv_map_size
                rsu_contact_norm = np.clip(rsu_contact / max(self._max_rsu_contact_time, 1e-6), 0, 1)

                # RSU节点特征：计算时间预估
                rsu_cpu = rsu.cpu_freq if rsu else self.config.F_RSU
                rsu_est_exec = task_comp_size / max(rsu_cpu, 1e-6) if task_comp_size > 0 else 0.0
                rsu_est_comm = task_data_size / max(rsu_rate, 1e-6) if task_data_size > 0 else 0.0
                rsu_est_wait = rsu_wait

                resource_raw[1] = [
                    rsu.cpu_freq * self._inv_max_cpu,
                    np.clip(rsu_wait * self._inv_max_wait, 0, 1),
                    np.clip(rsu_dist / max(self.config.RSU_RANGE, 1e-6), 0, 1),
                    np.clip(rsu_rate * self._inv_max_rate_v2i, 0, 1),
                    rel_rsu[0],
                    rel_rsu[1],
                        0.0,  # RSU速度为0
                        0.0,  # RSU速度为0
                        2.0,  # Node_Type = 2 (RSU)
                        slack_norm,
                        rsu_contact_norm,
                        np.clip(rsu_est_exec / 10.0, 0, 1),  # Est_Exec_Time
                        np.clip(rsu_est_comm / 10.0, 0, 1),  # Est_Comm_Time
                        np.clip(rsu_est_wait / 10.0, 0, 1),  # Est_Wait_Time
                    ]

                for idx, info in enumerate(v2v_slots):
                    if info is None:
                        continue
                    contact_norm = np.clip(info['contact_time'] / max(self._max_v2v_contact_time, 1e-6), 0, 1)

                    # Neighbor节点特征：计算时间预估
                    neighbor_est_exec = task_comp_size / max(info['cpu_freq'], 1e-6) if task_comp_size > 0 else 0.0
                    neighbor_est_comm = task_data_size / max(info['rate'], 1e-6) if task_data_size > 0 else 0.0
                    neighbor_est_wait = info['queue_wait']

                    resource_raw[2 + idx] = [
                        info['cpu_freq'] * self._inv_max_cpu,
                        np.clip(info['queue_wait'] * self._inv_max_wait, 0, 1),
                        np.clip(info['dist'] * self._inv_v2v_range, 0, 1),
                        np.clip(info['rate'] * self._inv_max_rate_v2v, 0, 1),
                        info['rel_pos'][0],
                        info['rel_pos'][1],
                        info['vel'][0] * self._inv_max_velocity,
                        info['vel'][1] * self._inv_max_velocity,
                        3.0,  # Node_Type = 3 (Neighbor)
                        slack_norm,
                        contact_norm,
                        np.clip(neighbor_est_exec / 10.0, 0, 1),  # Est_Exec_Time
                        np.clip(neighbor_est_comm / 10.0, 0, 1),  # Est_Comm_Time
                        np.clip(neighbor_est_wait / 10.0, 0, 1),  # Est_Wait_Time
                    ]

                # [关键] 固定维度填充 - 适配批处理要求
                padded_adj = np.zeros((self.config.MAX_NODES, self.config.MAX_NODES), dtype=np.float32)
                padded_adj[:num_nodes, :num_nodes] = v.task_dag.adj

                padded_task_mask = np.zeros(self.config.MAX_NODES, dtype=bool)
                padded_task_mask[:num_nodes] = task_schedulable
            
                # [新增] DAG拓扑特征（用于网络特征工程）
                # L_fwd, L_bwd: [MAX_NODES], 前向/后向层级
                padded_L_fwd = np.zeros(MAX_NODES, dtype=np.int32)
                padded_L_bwd = np.zeros(MAX_NODES, dtype=np.int32)
                padded_L_fwd[:num_nodes] = v.task_dag.L_fwd
                padded_L_bwd[:num_nodes] = v.task_dag.L_bwd
            
                # data_matrix: [MAX_NODES, MAX_NODES], 边数据量
                padded_data_matrix = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
                edge_max = max(getattr(Cfg, 'MAX_EDGE_DATA', 1.0), 1.0)
                edge_norm = np.log1p(v.task_dag.data_matrix) / np.log1p(edge_max)
                padded_data_matrix[:num_nodes, :num_nodes] = np.clip(edge_norm, 0.0, 1.0)
            
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
                    # 优先从v.task_dag.exec_locations获取（Vehicle属性），其次是v.task_dag.task_locations
                    if hasattr(v, 'exec_locations') and v.task_dag.exec_locations[t_idx] is not None:
                        loc = v.task_dag.exec_locations[t_idx]
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

                # [Rank Bias用] 填充priority（用于RankBiasEncoder）
                padded_priority = np.zeros(MAX_NODES, dtype=np.float32)
                if v.task_dag.priority is not None:
                    padded_priority[:num_nodes] = v.task_dag.priority
                else:
                    # 如果priority未计算，使用均匀分布作为fallback
                    padded_priority[:num_nodes] = 0.5

                obs_list.append({
                'node_x': padded_node_feats,
                'self_info': self_info,
                'rsu_info': [rsu_load_norm],
                'serving_rsu_id': int(serving_rsu_id_val),
                'serving_rsu_onehot': serving_rsu_onehot,
                'serving_rsu_info': serving_rsu_info,
                'candidate_ids': candidate_ids.astype(np.int64),
                'candidate_types': candidate_types.astype(np.int8),
                'candidate_mask': target_mask_row.astype(np.float32),
                'adj': padded_adj,
                'neighbors': neighbors_array,
                'task_mask': padded_task_mask,
                'target_mask': padded_target_mask,  # [新设计] 简化为[2+MAX_NEIGHBORS]
                'action_mask': padded_target_mask.copy(),  # [新增] Actor专用动作掩码
                'resource_ids': resource_id_list,  # [新增] 资源节点ID列表
                'resource_raw': resource_raw,  # [新增] 资源原始物理特征
                'subtask_index': int(selected_subtask_idx),  # [新设计] 添加当前选中的任务索引
                # [新增] DAG拓扑特征
                'L_fwd': padded_L_fwd,
                'L_bwd': padded_L_bwd,
                'data_matrix': padded_data_matrix,
                'Delta': padded_Delta,
                'status': padded_status,
                'location': padded_location,
                'priority': padded_priority,  # [Rank Bias用] 节点优先级（归一化，越大越重要）
                'obs_stamp': int(self._episode_steps)
            })

        self._last_obs_stamp = int(self._episode_steps)
        num_veh = max(len(self.vehicles), 1)
        if not hasattr(self, "_episode_obs_stats"):
            self._episode_obs_stats = {
                "steps": 0,
                "avail_L_sum": 0.0,
                "avail_R_sum": 0.0,
                "avail_V_sum": 0.0,
                "neighbor_sum": 0.0,
                "best_v2v_rate_sum": 0.0,
                "best_v2v_valid_sum": 0.0,
                "v2v_beats_rsu_sum": 0.0,
                "cost_gap_sum": 0.0,
                "cost_rsu_sum": 0.0,
                "cost_v2v_sum": 0.0,
                "cost_pair_count": 0.0,
            }
        step_avail_l /= num_veh
        step_avail_r /= num_veh
        step_avail_v /= num_veh
        step_neighbor_mean = step_neighbor_sum / num_veh
        best_v2v_rate_step = (step_best_v2v_sum / step_best_v2v_valid) if step_best_v2v_valid > 0 else 0.0
        best_v2v_valid_step = step_best_v2v_valid / num_veh
        v2v_beats_rsu_step = step_v2v_beats_rsu / num_veh

        self._episode_obs_stats["steps"] += 1
        self._episode_obs_stats["avail_L_sum"] += step_avail_l
        self._episode_obs_stats["avail_R_sum"] += step_avail_r
        self._episode_obs_stats["avail_V_sum"] += step_avail_v
        self._episode_obs_stats["neighbor_sum"] += step_neighbor_mean
        self._episode_obs_stats["best_v2v_rate_sum"] += best_v2v_rate_step
        self._episode_obs_stats["best_v2v_valid_sum"] += best_v2v_valid_step
        self._episode_obs_stats["v2v_beats_rsu_sum"] += v2v_beats_rsu_step
        self._episode_obs_stats["cost_gap_sum"] += step_cost_gap_sum
        self._episode_obs_stats["cost_rsu_sum"] += step_cost_rsu_sum
        self._episode_obs_stats["cost_v2v_sum"] += step_cost_v2v_sum
        self._episode_obs_stats["cost_pair_count"] += step_cost_pair_count
        return obs_list

    def _estimate_v2i_users(self):
        """
        估算当前活跃的V2I上传用户数，用于带宽均分与噪声计算保持一致
        """
        count = 0
        for veh in self.vehicles:
            tgt = getattr(veh, 'curr_target', None)
            if tgt == 'RSU':
                count += 1
            elif isinstance(tgt, tuple) and len(tgt) == 2 and tgt[0] == 'RSU':
                count += 1
        return max(count, 1)

    def _get_upload_bytes(self, dag, subtask_id):
        if subtask_id is None or subtask_id < 0 or subtask_id >= len(getattr(dag, "total_data", [])):
            return 0.0
        rem = 0.0
        if hasattr(dag, "rem_data") and dag.rem_data is not None and subtask_id < len(dag.rem_data):
            rem = float(dag.rem_data[subtask_id])
            if rem < -1e-9:
                raise RuntimeError("rem_data negative")
            if rem < 0:
                rem = 0.0
        if rem <= 0:
            rem = float(dag.total_data[subtask_id])
        return max(rem, 0.0)

    def _get_remaining_cycles(self, dag, subtask_id):
        if subtask_id is None or subtask_id < 0 or subtask_id >= len(getattr(dag, "total_comp", [])):
            return 0.0
        rem = 0.0
        if hasattr(dag, "rem_comp") and dag.rem_comp is not None and subtask_id < len(dag.rem_comp):
            rem = float(dag.rem_comp[subtask_id])
            if rem < -1e-9:
                raise RuntimeError("rem_comp negative")
            if rem < 0:
                rem = 0.0
        if rem <= 0:
            rem = float(dag.total_comp[subtask_id])
        return max(rem, 0.0)

    def _get_reachable_f_max(self, vehicle):
        """计算可达集合的f_max：Local/serving RSU/Top V2V候选"""
        if vehicle is None:
            return self._f_max_const, {
                "f_local": 0.0,
                "f_serving_rsu": 0.0,
                "f_candidates_max": 0.0,
            }

        f_local = max(getattr(vehicle, "cpu_freq", 0.0), 1e-9)
        f_serving_rsu = 0.0
        rsu_id = getattr(vehicle, "serving_rsu_id", None)
        if rsu_id is None:
            rsu_id = self._last_rsu_choice.get(vehicle.id)
        if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
            rsu = self.rsus[rsu_id]
            if rsu.is_in_coverage(vehicle.pos):
                f_serving_rsu = max(getattr(rsu, "cpu_freq", 0.0), 0.0)

        f_candidates_max = 0.0
        candidate_set = self._last_candidate_set.get(vehicle.id)
        if candidate_set is not None:
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            for idx in range(2, len(ids)):
                if idx < len(mask) and not bool(mask[idx]):
                    continue
                cand_id = int(ids[idx])
                if cand_id < 0 or cand_id == vehicle.id:
                    continue
                cand = self._get_vehicle_by_id(cand_id)
                if cand is not None:
                    f_candidates_max = max(f_candidates_max, getattr(cand, "cpu_freq", 0.0))

        f_max = max(f_local, f_serving_rsu, f_candidates_max, 1e-9)
        return f_max, {
            "f_local": f_local,
            "f_serving_rsu": f_serving_rsu,
            "f_candidates_max": f_candidates_max,
        }

    def _compute_phi_value(self, dag, vehicle=None):
        """基于剩余DAG的关键路径长度计算潜势ϕ"""
        status = getattr(dag, "status", None)
        if status is None:
            return 0.0
        remaining_nodes = [idx for idx, s in enumerate(status) if s != 3]
        if not remaining_nodes:
            return 0.0
        rem_set = set(remaining_nodes)
        adj = getattr(dag, "adj", None)
        if adj is None:
            return 0.0
        in_deg = {u: 0 for u in rem_set}
        succ = {u: [] for u in rem_set}
        for u in rem_set:
            outs = list(np.where(adj[u] == 1)[0])
            succ[u] = [v for v in outs if v in rem_set]
        for v in rem_set:
            preds = np.where(adj[:, v] == 1)[0]
            in_deg[v] = int(np.sum([1 for p in preds if p in rem_set]))
        topo = []
        q = deque([u for u in rem_set if in_deg[u] == 0])
        while q:
            u = q.popleft()
            topo.append(u)
            for v in succ.get(u, []):
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        if len(topo) < len(rem_set):
            topo = list(rem_set)
        dp = {u: 0.0 for u in rem_set}
        for u in reversed(topo):
            cyc = max(self._get_remaining_cycles(dag, u), 0.0)
            child_vals = [dp[v] for v in succ.get(u, [])]
            dp[u] = cyc + (max(child_vals) if child_vals else 0.0)
        entry_nodes = [u for u in rem_set if all((p not in rem_set) for p in np.where(adj[:, u] == 1)[0])]
        if not entry_nodes:
            entry_nodes = topo
        cp_rem = max(dp[u] for u in entry_nodes) if entry_nodes else 0.0
        f_max, f_max_info = self._get_reachable_f_max(vehicle)
        phi = - (cp_rem / f_max) / max(self.config.T_REF, 1e-9)
        phi = float(np.clip(phi, -self.config.PHI_CLIP, 0.0))
        if getattr(self.config, "DEBUG_PBRS_AUDIT", False) and vehicle is not None:
            f_max_info["f_max"] = f_max
            f_max_info["phi"] = phi
            self._last_phi_debug[vehicle.id] = f_max_info
        return phi

    def _compute_cp_stats(self, dag):
        """计算剩余关键路径计算量与边数据量下界（仅依赖DAG状态）"""
        status = getattr(dag, "status", None)
        if status is None:
            return 0.0, 0.0
        remaining_nodes = [idx for idx, s in enumerate(status) if s != 3]
        if not remaining_nodes:
            return 0.0, 0.0
        rem_set = set(remaining_nodes)
        adj = getattr(dag, "adj", None)
        if adj is None:
            return 0.0, 0.0

        in_deg = {u: 0 for u in rem_set}
        succ = {u: [] for u in rem_set}
        for u in rem_set:
            outs = list(np.where(adj[u] == 1)[0])
            succ[u] = [v for v in outs if v in rem_set]
        for v in rem_set:
            preds = np.where(adj[:, v] == 1)[0]
            in_deg[v] = int(np.sum([1 for p in preds if p in rem_set]))

        topo = []
        q = deque([u for u in rem_set if in_deg[u] == 0])
        while q:
            u = q.popleft()
            topo.append(u)
            for v in succ.get(u, []):
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        if len(topo) < len(rem_set):
            topo = list(rem_set)

        dp_comp = {u: 0.0 for u in rem_set}
        dp_data = {u: 0.0 for u in rem_set}
        data_matrix = getattr(dag, "data_matrix", None)
        for u in reversed(topo):
            cyc = max(self._get_remaining_cycles(dag, u), 0.0)
            best_child = None
            best_comp = -float("inf")
            best_data = 0.0
            for v in succ.get(u, []):
                comp_val = dp_comp[v]
                data_val = dp_data[v]
                if comp_val > best_comp or (comp_val == best_comp and data_val > best_data):
                    best_comp = comp_val
                    best_data = data_val
                    best_child = v
            if best_child is None:
                dp_comp[u] = cyc
                dp_data[u] = 0.0
            else:
                edge_data = 0.0
                if data_matrix is not None:
                    try:
                        edge_data = float(data_matrix[u, best_child])
                    except Exception:
                        edge_data = 0.0
                dp_comp[u] = cyc + best_comp
                dp_data[u] = edge_data + best_data

        entry_nodes = [u for u in rem_set if all((p not in rem_set) for p in np.where(adj[:, u] == 1)[0])]
        if not entry_nodes:
            entry_nodes = topo
        best_entry = None
        best_comp = -float("inf")
        best_data = 0.0
        for u in entry_nodes:
            comp_val = dp_comp[u]
            data_val = dp_data[u]
            if comp_val > best_comp or (comp_val == best_comp and data_val > best_data):
                best_comp = comp_val
                best_data = data_val
                best_entry = u

        cp_rem = float(dp_comp[best_entry]) if best_entry is not None else 0.0
        cp_edge_data = float(dp_data[best_entry]) if best_entry is not None else 0.0
        return cp_rem, cp_edge_data

    def _compute_queue_lb(self, vehicle):
        """计算队列等待时间下界（本地/RSU/候选V2V取最小）"""
        waits = []
        freq_self = max(getattr(vehicle, "cpu_freq", 0.0), 1e-9)
        waits.append(self._get_veh_queue_load(vehicle.id) / freq_self)

        rsu_id = self._last_rsu_choice.get(vehicle.id)
        candidate_set = self._last_candidate_set.get(vehicle.id)
        if candidate_set is not None and len(candidate_set.get("ids", [])) > 1:
            if bool(candidate_set["mask"][1]):
                rsu_id = int(candidate_set["ids"][1])
        if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
            rsu = self.rsus[rsu_id]
            freq_r = max(getattr(rsu, "cpu_freq", 0.0), 1e-9)
            waits.append(self._get_rsu_queue_load(rsu_id) / freq_r)

        if candidate_set is not None:
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            for idx in range(2, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id >= 0:
                        cand = self._get_vehicle_by_id(cand_id)
                        if cand is not None:
                            freq_c = max(getattr(cand, "cpu_freq", 0.0), 1e-9)
                            waits.append(self._get_veh_queue_load(cand_id) / freq_c)

        if not waits:
            return 0.0
        return float(min(waits))

    def _compute_best_comm_rate(self, vehicle):
        """计算候选集合内最优通信速率（使用冻结快照）"""
        eps_rate = getattr(self.config, "EPS_RATE", 1e-9)
        best_rate = 0.0
        candidate_set = self._last_candidate_set.get(vehicle.id)

        rsu_id = self._last_rsu_choice.get(vehicle.id)
        if candidate_set is not None and len(candidate_set.get("ids", [])) > 1:
            if bool(candidate_set["mask"][1]):
                rsu_id = int(candidate_set["ids"][1])
        if rsu_id is not None and rsu_id >= 0:
            try:
                rate = self._get_rate_from_snapshot(("VEH", vehicle.id), ("RSU", int(rsu_id)), "V2I")
                best_rate = max(best_rate, rate)
            except Exception:
                pass

        if candidate_set is not None:
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            v2v_slots = candidate_set.get("v2v_slots", [])
            for idx in range(2, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id < 0:
                        continue
                    try:
                        rate = self._get_rate_from_snapshot(("VEH", vehicle.id), ("VEH", cand_id), "V2V")
                        best_rate = max(best_rate, rate)
                        continue
                    except Exception:
                        pass
                    slot_idx = idx - 2
                    if 0 <= slot_idx < len(v2v_slots) and v2v_slots[slot_idx] is not None:
                        best_rate = max(best_rate, float(v2v_slots[slot_idx].get("rate", 0.0)))

        return float(max(best_rate, eps_rate))

    def _compute_phi_value_v2(self, dag, vehicle=None):
        """PBRS_KP_V2: 基于关键路径下界 LB(s)=compute+comm+queue 的潜势ϕ"""
        if vehicle is None:
            return 0.0, {}
        cp_rem, d_cp_lb = self._compute_cp_stats(dag)
        f_max, f_max_info = self._get_reachable_f_max(vehicle)
        rate_best = self._compute_best_comm_rate(vehicle)
        comm_lb = d_cp_lb / max(rate_best, getattr(self.config, "EPS_RATE", 1e-9))
        queue_lb = self._compute_queue_lb(vehicle)
        lb = (cp_rem / max(f_max, 1e-9)) + comm_lb + queue_lb
        phi = -float(lb) / max(self.config.T_REF, 1e-9)
        phi = float(np.clip(phi, -self.config.PHI_CLIP, 0.0))

        debug = {
            "cp_rem": float(cp_rem),
            "f_max": float(f_max),
            "d_cp_lb": float(d_cp_lb),
            "rate_best": float(rate_best),
            "comm_lb": float(comm_lb),
            "queue_lb": float(queue_lb),
            "lb": float(lb),
            "phi": float(phi),
        }
        if getattr(self.config, "DEBUG_PBRS_AUDIT", False):
            f_max_info["f_max"] = f_max
            f_max_info["phi"] = phi
            self._last_phi_debug[vehicle.id] = f_max_info
        return phi, debug

    def _check_phi_monotonicity(self, total_rem_comp, phi_avg):
        """调试用：检测剩余计算下降但Phi变得更负的频率"""
        if not getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
            return
        if not hasattr(self, "_phi_debug_state"):
            self._phi_debug_state = {
                "last_rem": None,
                "last_phi": None,
                "bad_cnt": 0,
                "samples": 0,
            }
        state = self._phi_debug_state
        import random, warnings
        if random.random() > getattr(self.config, "DEBUG_PHI_MONO_PROB", 0.1):
            return
        if state["last_rem"] is not None and state["last_phi"] is not None:
            if total_rem_comp < state["last_rem"] - 1e-6 and phi_avg < state["last_phi"] - 1e-3:
                state["bad_cnt"] += 1
        state["samples"] += 1
        state["last_rem"] = total_rem_comp
        state["last_phi"] = phi_avg
        if state["bad_cnt"] > 3 and state["bad_cnt"] / max(state["samples"], 1) > 0.5:
            warnings.warn(f"[Debug] Phi monotonicity suspicion: bad_cnt={state['bad_cnt']} samples={state['samples']}", UserWarning)

    def _estimate_t_actual(self, vehicle, subtask_idx, target, cycles, power_ratio=1.0):
        """估计动作目标的执行时间（使用冻结速率与队列快照）"""
        freq_self = max(getattr(vehicle, "cpu_freq", self.config.MIN_VEHICLE_CPU_FREQ), 1e-9)
        t_local = self._get_veh_queue_wait_time(vehicle.id, freq_self) + cycles / freq_self
        if target is None or target == 'Local':
            return t_local, 0.0
        eps_rate = getattr(self.config, "EPS_RATE", 1e-9)
        dag = vehicle.task_dag
        din = self._get_upload_bytes(dag, subtask_idx)

        if self._is_rsu_location(target):
            rsu_id = self._get_rsu_id_from_location(target)
            if rsu_id is not None:
                self._assert_serving_rsu(vehicle, rsu_id, "reward_target")
            dst_node = ("RSU", rsu_id if rsu_id is not None else 0)
            rate = self._get_rate_from_snapshot(("VEH", vehicle.id), dst_node, "V2I")
            if rate is None and getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
                raise RuntimeError("[Assert] Reward path missing V2I rate in snapshot")
            rate = max(rate if rate is not None else 0.0, eps_rate)
            t_tx = din / rate if din > 0 else 0.0
            freq_r = self.config.F_RSU
            wait = 0.0
            if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                freq_r = self.rsus[rsu_id].cpu_freq
                wait = self._get_rsu_queue_wait_time(rsu_id)
            t_comp = cycles / max(freq_r, 1e-9)
            return t_tx + wait + t_comp, t_tx

        if isinstance(target, int):
            dst_node = ("VEH", target)
            rate = self._get_rate_from_snapshot(("VEH", vehicle.id), dst_node, "V2V")
            if rate is None and getattr(self.config, "DEBUG_REWARD_ASSERTS", False):
                raise RuntimeError("[Assert] Reward path missing V2V rate in snapshot")
            rate = max(rate if rate is not None else 0.0, eps_rate)
            t_tx = din / rate if din > 0 else 0.0
            tgt_veh = self._get_vehicle_by_id(target)
            freq_t = getattr(tgt_veh, "cpu_freq", self.config.MIN_VEHICLE_CPU_FREQ) if tgt_veh is not None else self.config.MIN_VEHICLE_CPU_FREQ
            wait = self._get_veh_queue_wait_time(target, freq_t) if tgt_veh is not None else 0.0
            t_comp = cycles / max(freq_t, 1e-9)
            return t_tx + wait + t_comp, t_tx

        return t_local, 0.0

    def _compute_latency_advantage(self, vehicle, ctx):
        """PBRS_KP_V2: 计算时延相对优势奖励及诊断字段"""
        subtask_idx = ctx.get("subtask")
        if subtask_idx is None or ctx.get("illegal"):
            return 0.0, {}

        cycles = float(ctx.get("cycles", 0.0))
        if cycles <= 0:
            return 0.0, {}

        target = ctx.get("target")
        t_a = float(ctx.get("t_actual", 0.0))
        if not np.isfinite(t_a):
            t_a = 0.0

        t_L, _ = self._estimate_t_actual(vehicle, subtask_idx, 'Local', cycles, power_ratio=1.0)
        t_R = None
        t_V = None

        candidate_set = self._last_candidate_set.get(vehicle.id)
        rsu_id = self._last_rsu_choice.get(vehicle.id)
        if candidate_set is not None and len(candidate_set.get("ids", [])) > 1:
            if bool(candidate_set["mask"][1]):
                rsu_id = int(candidate_set["ids"][1])
        if rsu_id is not None and rsu_id >= 0:
            try:
                t_R, _ = self._estimate_t_actual(vehicle, subtask_idx, ("RSU", int(rsu_id)), cycles, power_ratio=1.0)
            except Exception:
                t_R = None

        if candidate_set is not None:
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            v2v_slots = candidate_set.get("v2v_slots", [])
            best_t = None
            for idx in range(2, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id < 0:
                        continue
                    try:
                        t_v, _ = self._estimate_t_actual(vehicle, subtask_idx, cand_id, cycles, power_ratio=1.0)
                        if best_t is None or t_v < best_t:
                            best_t = t_v
                    except Exception:
                        slot_idx = idx - 2
                        if 0 <= slot_idx < len(v2v_slots) and v2v_slots[slot_idx] is not None:
                            t_v = float(v2v_slots[slot_idx].get("total_time", 0.0))
                            if best_t is None or t_v < best_t:
                                best_t = t_v
            t_V = best_t

        options = []
        if t_L is not None and np.isfinite(t_L):
            options.append(t_L)
        if t_R is not None and np.isfinite(t_R):
            options.append(t_R)
        if t_V is not None and np.isfinite(t_V):
            options.append(t_V)

        t_alt = None
        if options:
            for val in options:
                if np.isfinite(val) and abs(val - t_a) > 1e-9:
                    if t_alt is None or val < t_alt:
                        t_alt = val
        if t_alt is None:
            t_alt = t_a

        A_t = (t_alt - t_a) / max(self.config.T_REF, 1e-9) if np.isfinite(t_alt) and np.isfinite(t_a) else 0.0
        r_lat = float(self.config.LAT_ALPHA) * float(np.tanh(A_t))

        details = {
            "t_L": float(t_L) if t_L is not None and np.isfinite(t_L) else None,
            "t_R": float(t_R) if t_R is not None and np.isfinite(t_R) else None,
            "t_V": float(t_V) if t_V is not None and np.isfinite(t_V) else None,
            "t_a": float(t_a) if np.isfinite(t_a) else None,
            "t_alt": float(t_alt) if np.isfinite(t_alt) else None,
            "A_t": float(A_t) if np.isfinite(A_t) else None,
            "r_lat": float(r_lat),
        }
        return r_lat, details

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
        elif isinstance(pred_loc, int):
            tx_veh = self._get_vehicle_by_id(pred_loc)

        if curr_loc == 'Local':
            rx_veh = vehicle
        elif isinstance(curr_loc, int):
            rx_veh = self._get_vehicle_by_id(curr_loc)

        rate = 1e-6

        if pred_loc == 'RSU' or curr_loc == 'RSU':
            # 确定目标位置（如果是RSU）
            if self._is_rsu_location(pred_loc):
                rsu_id = self._get_rsu_id_from_location(pred_loc)
                if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                    rsu_pos = self.rsus[rsu_id].position
                else:
                    rsu_pos = self.config.RSU_POS  # 向后兼容
                target_veh = rx_veh if rx_veh else (tx_veh if tx_veh else self.vehicles[0] if len(self.vehicles) > 0 else None)
                if target_veh:
                    rate = self.channel.compute_one_rate(
                        target_veh, rsu_pos, 'V2I', self.time,
                        v2i_user_count=self._estimate_v2i_users()
                    )
                else:
                    rate = 1e6
            elif self._is_rsu_location(curr_loc):
                rsu_id = self._get_rsu_id_from_location(curr_loc)
                if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                    rsu_pos = self.rsus[rsu_id].position
                else:
                    rsu_pos = self.config.RSU_POS  # 向后兼容
                target_veh = tx_veh if tx_veh else (rx_veh if rx_veh else self.vehicles[0] if len(self.vehicles) > 0 else None)
                if target_veh:
                    rate = self.channel.compute_one_rate(
                        target_veh, rsu_pos, 'V2I', self.time,
                        v2i_user_count=self._estimate_v2i_users()
                    )
                else:
                    rate = 1e6
            else:
                # 向后兼容：使用默认RSU位置
                target_veh = rx_veh if rx_veh else tx_veh
                if target_veh:
                    rate = self.channel.compute_one_rate(
                        target_veh, self.config.RSU_POS, 'V2I', self.time,
                        v2i_user_count=self._estimate_v2i_users()
                    )
                else:
                    rate = 1e6
        else:
            # V2V通信
            if tx_veh and rx_veh:
                dist = np.linalg.norm(tx_veh.pos - rx_veh.pos)
                if dist <= self.config.V2V_RANGE:
                    rate = self.channel.compute_one_rate(tx_veh, rx_veh.pos, 'V2V', self.time)
                else:
                    rate = 1e-6
            else:
                rate = 1e-6
        final_rate = max(rate, 1e-6)
        self._comm_rate_cache[cache_key] = final_rate
        return final_rate

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
            # [P02修复] 使用统一队列查询方法
            wait_time = self._get_veh_queue_wait_time(v.id, v.cpu_freq)
            freq = v.cpu_freq
        else:
            freq = self.config.MIN_VEHICLE_CPU_FREQ
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

        eff_gain = np.tanh(self.config.EFF_SCALE * gain_ratio)

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
                    # [P02修复] 使用统一队列查询方法
                    wait_time = self._get_rsu_queue_wait_time(rsu_id)
                    freq = self.rsus[rsu_id].cpu_freq
                else:
                    wait_time = 0.0
                    freq = self.config.F_RSU
            else:
                # 单个RSU场景（向后兼容）
                if len(self.rsus) > 0:
                    # [P02修复] 使用统一队列查询方法
                    wait_time = min([self._get_rsu_queue_wait_time(rsu.id) for rsu in self.rsus])
                else:
                    wait_time = 0.0
                freq = self.config.F_RSU
        elif isinstance(target, int):
            # 其他车辆执行
            target_veh = self._get_vehicle_by_id(target)
            if target_veh is None:
                return self._calculate_local_execution_time(dag, vehicle_id)
            # [P02修复] 使用统一队列查询方法
            wait_time = self._get_veh_queue_wait_time(target_veh.id, target_veh.cpu_freq)
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
                            dist = np.linalg.norm(veh_pos - self.config.RSU_POS) if len(self.vehicles) > 0 else 500.0
                    rate = self._estimate_rate(dist, 'V2I', target)
                elif isinstance(target, int):
                    # 其他车辆执行
                    tx_pos = veh_pos
                    target_veh = self._get_vehicle_by_id(target)
                    if target_veh is None:
                        rate = 1e6
                        trans_time = input_data / max(rate, 1e-6)
                        return wait_time + comp_time + trans_time
                    rx_pos = target_veh.pos
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
            v2i_users = self._estimate_v2i_users()
            eff_bw = self.config.BW_V2I / max(v2i_users, 1)
            noise_w = self.channel._noise_power(eff_bw)
            h_bar = self.channel._path_loss(max(dist, 1.0), self.config.ALPHA_V2I)
            signal_w = self.config.dbm2watt(self.config.TX_POWER_MIN_DBM) * h_bar
            rate = eff_bw * np.log2(1 + signal_w / max(noise_w, 1e-12))
        else:
            h_bar = self.channel._path_loss(max(dist, 1.0), self.config.ALPHA_V2V)
            interference_w = self.config.dbm2watt(self.config.V2V_INTERFERENCE_DBM)
            noise_w = self.channel._noise_power(self.config.BW_V2V)
            signal_w = self.config.dbm2watt(self.config.TX_POWER_MIN_DBM) * h_bar
            rate = self.config.BW_V2V * np.log2(1 + signal_w / max(noise_w + interference_w, 1e-12))
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
                # [P02修复] 使用统一队列查询方法
                q_curr_load = self._get_veh_queue_load(vehicle_id)
                q_max_load = self.config.VEHICLE_QUEUE_CYCLES_LIMIT
            else:
                return 0.0
        elif self._is_rsu_location(target):
            # RSU执行
            if isinstance(target, tuple) and len(target) == 2:
                # 多RSU场景：使用指定的RSU队列计算量
                rsu_id = target[1]
                if 0 <= rsu_id < len(self.rsus):
                    # [P02修复] 使用统一队列查询方法
                    q_curr_load = self._get_rsu_queue_load(rsu_id)
                    q_max_load = self.config.RSU_QUEUE_CYCLES_LIMIT
                else:
                    return 0.0
            else:
                # 单个RSU场景（向后兼容）：使用所有RSU的总计算量
                # [P02修复] 使用统一队列查询方法
                q_curr_load = sum([self._get_rsu_queue_load(rsu.id) for rsu in self.rsus]) if len(self.rsus) > 0 else 0
                q_max_load = self.config.RSU_QUEUE_CYCLES_LIMIT * len(self.rsus) if len(self.rsus) > 0 else self.config.RSU_QUEUE_CYCLES_LIMIT
        elif isinstance(target, int):
            target_veh = self._get_vehicle_by_id(target)
            if target_veh is None:
                return 0.0
            # [P02修复] 使用统一队列查询方法
            q_curr_load = self._get_veh_queue_load(target_veh.id)
            q_max_load = self.config.VEHICLE_QUEUE_CYCLES_LIMIT
        else:
            return 0.0

        util_ratio = (q_curr_load + task_comp) / q_max_load
        util_ratio = np.clip(util_ratio, 0.0, 1.0)
        cong_penalty = -1.0 * (util_ratio ** self.config.CONG_GAMMA)

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
            task_comp = self.config.MEAN_COMP_LOAD

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
                    dist = np.linalg.norm(v.pos - self.config.RSU_POS)
                    in_range = (dist <= self.config.RSU_RANGE)
                    rsu_dist = dist
            
            if not in_range:
                hard_triggered = True
            else:
                # 距离预警（软约束，固定启用）
                # Distance warning (soft constraint, permanently enabled)
                safe_dist = self.config.RSU_RANGE * self.config.DIST_SAFE_FACTOR
                if rsu_dist > safe_dist:
                    dist_ratio = (rsu_dist - safe_dist) / (self.config.RSU_RANGE - safe_dist + 1e-6)
                    dist_ratio = np.clip(dist_ratio, 0.0, 1.0)
                    soft_penalty += -self.config.DIST_PENALTY_WEIGHT * (dist_ratio ** self.config.DIST_SENSITIVITY)
        
        # 2. V2V范围检查
        elif isinstance(target, int):
            target_veh = self._get_vehicle_by_id(target)
            if target_veh is None:
                hard_triggered = True
            else:
                dist = np.linalg.norm(v.pos - target_veh.pos)
                
                if dist > self.config.V2V_RANGE:
                    hard_triggered = True
                else:
                    # 距离预警（软约束，固定启用）
                    # Distance warning (soft constraint, permanently enabled)
                    safe_dist = self.config.V2V_RANGE * self.config.DIST_SAFE_FACTOR
                    if dist > safe_dist:
                        dist_ratio = (dist - safe_dist) / (self.config.V2V_RANGE - safe_dist + 1e-6)
                        dist_ratio = np.clip(dist_ratio, 0.0, 1.0)
                        soft_penalty += -self.config.DIST_PENALTY_WEIGHT * (dist_ratio ** self.config.DIST_SENSITIVITY)

        # 3. 队列溢出检查（硬约束）
        if not hard_triggered:
            q_after_load = 0.0
            q_max_load = self.config.VEHICLE_QUEUE_CYCLES_LIMIT
            if target == 'Local':
                # [P02修复] 使用统一队列查询方法
                q_after_load = self._get_veh_queue_load(v.id) + task_comp
                q_max_load = self.config.VEHICLE_QUEUE_CYCLES_LIMIT
            elif self._is_rsu_location(target):
                if isinstance(target, tuple) and len(target) == 2:
                    rsu_id = target[1]
                    if 0 <= rsu_id < len(self.rsus):
                        # [P02修复] 使用统一队列查询方法
                        q_after_load = self._get_rsu_queue_load(rsu_id) + task_comp
                    else:
                        q_after_load = task_comp
                    q_max_load = self.config.RSU_QUEUE_CYCLES_LIMIT
                else:
                    # [P02修复] 使用统一队列查询方法
                    q_after_load = (sum([self._get_rsu_queue_load(rsu.id) for rsu in self.rsus]) + task_comp) if len(self.rsus) > 0 else task_comp
                    q_max_load = self.config.RSU_QUEUE_CYCLES_LIMIT * len(self.rsus) if len(self.rsus) > 0 else self.config.RSU_QUEUE_CYCLES_LIMIT
            elif isinstance(target, int):
                target_veh = self._get_vehicle_by_id(target)
                if target_veh is not None:
                    # [P02修复] 使用统一队列查询方法
                    q_after_load = self._get_veh_queue_load(target_veh.id) + task_comp
                    q_max_load = self.config.VEHICLE_QUEUE_CYCLES_LIMIT

            if q_after_load > q_max_load:
                hard_triggered = True

        if hard_triggered:
            pass  # 已清理

        return soft_penalty, hard_triggered

    def _compute_cost_components(self, vehicle_id, target, task_idx=None, task_comp=None):
        v = self.vehicles[vehicle_id]
        dag = v.task_dag

        if task_comp is None:
            task_comp = self.config.MEAN_COMP_LOAD

        r_soft_pen, hard_triggered = self._calculate_constraint_penalty(vehicle_id, target, task_idx, task_comp)

        delay_norm = 0.0
        energy_norm = 0.0
        r_timeout = 0.0

        if task_idx is not None and 0 <= task_idx < dag.num_subtasks:
            task_comp = dag.total_comp[task_idx]
            task_data = dag.total_data[task_idx]

            max_rate = self.config.NORM_MAX_RATE_V2I
            if target == 'Local':
                # [P02修复] 使用统一队列查询方法
                queue_wait = self._get_veh_queue_wait_time(v.id, v.cpu_freq)
                cpu_freq = v.cpu_freq
                tx_time = 0.0
                max_rate = self._get_norm_rate('V2I')
            elif self._is_rsu_location(target):
                rsu_id = self._get_rsu_id_from_location(target)
                if rsu_id is not None:
                    self._assert_serving_rsu(v, rsu_id, "cost_component")
                if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                    rsu = self.rsus[rsu_id]
                    # [P02修复] 使用统一队列查询方法
                    queue_wait = self._get_rsu_queue_wait_time(rsu_id)
                    cpu_freq = rsu.cpu_freq
                    rate = self.channel.compute_one_rate(
                        v, rsu.position, 'V2I', self.time,
                        v2i_user_count=self._estimate_v2i_users()
                    )
                    rate = max(rate, 1e-6)
                    self._update_rate_norm(rate, 'V2I')
                    tx_time = task_data / rate if task_data > 0 else 0.0
                else:
                    queue_wait = 0.0
                    cpu_freq = v.cpu_freq
                    tx_time = 0.0
                max_rate = self._get_norm_rate('V2I')
            elif isinstance(target, int):
                t_veh = self._get_vehicle_by_id(target)
                if t_veh is None:
                    queue_wait = 0.0
                    cpu_freq = v.cpu_freq
                    tx_time = 0.0
                else:
                    # [P02修复] 使用统一队列查询方法
                    queue_wait = self._get_veh_queue_wait_time(t_veh.id, t_veh.cpu_freq)
                    cpu_freq = t_veh.cpu_freq
                    rate = self.channel.compute_one_rate(v, t_veh.pos, 'V2V', self.time)
                    rate = max(rate, 1e-6)
                    self._update_rate_norm(rate, 'V2V')
                    tx_time = task_data / rate if task_data > 0 else 0.0
                max_rate = self._get_norm_rate('V2V')
            else:
                queue_wait = 0.0
                cpu_freq = v.cpu_freq
                tx_time = 0.0
                max_rate = self._get_norm_rate('V2I')

            comp_time = task_comp / max(cpu_freq, 1e-6)
            max_tx_time = task_data / max(max_rate, 1e-6) if task_data > 0 else 1.0
            max_comp_time = task_comp / max(self.config.MIN_VEHICLE_CPU_FREQ, 1e-6)

            delay_norm = (tx_time / max(max_tx_time, 1e-6) +
                          queue_wait / max(self.config.NORM_MAX_WAIT_TIME, 1e-6) +
                          comp_time / max(max_comp_time, 1e-6))

            if tx_time > 0 and target != 'Local':
                tx_power_w = self.config.dbm2watt(v.tx_power_dbm)
                max_power_w = self.config.dbm2watt(self.config.TX_POWER_MAX_DBM)
                max_energy = max_power_w * max(max_tx_time, 1e-6)
                energy_norm = (tx_power_w * tx_time) / max(max_energy, 1e-6)

        # [P03修复] Deadline检查已移至step()中统一处理（Phase5后）
        # 这里仅计算r_timeout惩罚项（用于奖励塑形），不再调用set_failed
        if dag.deadline > 0 and dag.is_failed and dag.fail_reason == 'deadline':
            # 任务已被标记为deadline失败，计算超时惩罚
            elapsed = self.time - dag.start_time
            overtime_ratio = max((elapsed - dag.deadline) / dag.deadline, 0.0)
            r_timeout = -self.config.TIMEOUT_PENALTY_WEIGHT * np.tanh(self.config.TIMEOUT_STEEPNESS * overtime_ratio)

        return {
            "delay_norm": delay_norm,
            "energy_norm": energy_norm,
            "r_soft_pen": r_soft_pen,
            "r_timeout": r_timeout,
            "hard_triggered": hard_triggered,
        }

    def _clip_reward(self, reward):
        """
        [奖励函数辅助] 奖励裁剪，防止奖励爆炸

        Args:
            reward: 原始奖励值

        Returns:
            float: 裁剪后的奖励值
        """
        if reward <= self.config.REWARD_MIN:
            pass  # 已清理
        if reward >= self.config.REWARD_MAX:
            pass  # 已清理
        return np.clip(reward, self.config.REWARD_MIN, self.config.REWARD_MAX)

    def calculate_agent_reward(self, vehicle_id, target, task_idx=None, data_size=0, task_comp=None, return_components=False, cft_prev_rem=None, cft_curr_rem=None, power_ratio=None, t_tx=None):
        """
        [MAPPO奖励函数] 计算单个智能体的奖励

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

        v.last_success_bonus = 0.0
        illegal_flag = getattr(v, 'illegal_action', False)
        components = self._compute_cost_components(vehicle_id, target, task_idx, task_comp)
        hard_triggered = components.get("hard_triggered", False)

        snapshot_time = self.time
        if cft_prev_rem is None or cft_curr_rem is None:
            cft_abs = self._compute_mean_cft_pi0(snapshot_time=snapshot_time, vehicle_ids=[v.id])
            cft_prev_rem = max(cft_abs - snapshot_time, 0.0) if cft_abs is not None else 0.0
            cft_curr_rem = cft_prev_rem

        if power_ratio is None:
            power_ratio = self._power_ratio_from_dbm(getattr(v, "tx_power_dbm", getattr(Cfg, "TX_POWER_MIN_DBM", 0.0)))
        if t_tx is None:
            t_tx = 0.0
        t_tx = float(np.clip(t_tx, 0.0, self.config.DT))
        if target == 'Local':
            t_tx = 0.0
        p_max_watt = self._get_p_max_watt(target)
        dT_rem = cft_prev_rem - cft_curr_rem

        reward, parts = compute_absolute_reward(
            dT_rem,
            t_tx,
            power_ratio,
            self.config.DT,
            p_max_watt,
            self.config.REWARD_MIN,
            self.config.REWARD_MAX,
            hard_triggered=hard_triggered or illegal_flag,
            illegal_action=illegal_flag,
        )
        reward = self._clip_reward(reward)

        out_components = {
            "delay_norm": components.get("delay_norm", 0.0),
            "energy_norm": parts.get("energy_norm", 0.0),
            "r_soft_pen": components.get("r_soft_pen", 0.0),
            "r_timeout": components.get("r_timeout", 0.0),
            "hard_triggered": hard_triggered,
            "dT_eff": parts.get("dT_eff", 0.0),
            "t_tx": parts.get("t_tx", 0.0),
        }

        return (reward, out_components) if return_components else reward
    
    # ========================================================================
    # 审计系统方法
    # ========================================================================
    
    def _collect_audit_step_info(self, commit_plans):
        """
        收集本步的审计数据（12项核心指标）
        
        Args:
            commit_plans: 本步提交的action plans
            
        Returns:
            dict: 审计信息
        """
        audit_info = {}
        
        # (1) RSU mask可用性 - 从plan中统计
        rsu_available_count = 0
        for plan in commit_plans:
            # 检查RSU是否在本次决策中可用
            if plan['subtask_idx'] is not None:
                v = plan['vehicle']
                candidate_set = self._last_candidate_set.get(v.id)
                if candidate_set is not None:
                    if candidate_set["mask"][1]:
                        rsu_available_count += 1
                elif self._last_rsu_choice.get(v.id) is not None:
                    rsu_available_count += 1
        audit_info['rsu_mask_true'] = rsu_available_count
        
        # (2) V2V可选邻居数 - 从_last_candidates统计
        valid_v2v_counts = []
        for plan in commit_plans:
            if plan['subtask_idx'] is not None:
                v = plan['vehicle']
                candidate_set = self._last_candidate_set.get(v.id)
                if candidate_set is not None:
                    valid_count = int(np.sum(candidate_set["mask"][2:]))
                else:
                    candidates = self._last_candidates.get(v.id, [])
                    valid_count = sum(1 for cid in candidates if cid is not None and cid >= 0)
                valid_v2v_counts.append(valid_count)
        audit_info['valid_v2v_count'] = np.mean(valid_v2v_counts) if valid_v2v_counts else 0
        
        # (3) Illegal动作统计 - 从plan中提取
        for plan in commit_plans:
            if plan['illegal_reason'] is not None:
                v = plan['vehicle']
                target_idx = plan.get('target_idx', 0)
                
                # 判断action类型
                if target_idx == 0:
                    action_type = 'local'
                elif target_idx == 1:
                    action_type = 'rsu'
                else:
                    action_type = 'v2v'
                
                # 检查mask是否一致（critical check）
                mask_was_true = False
                if action_type == 'rsu':
                    candidate_set = self._last_candidate_set.get(v.id)
                    if candidate_set is not None:
                        mask_was_true = bool(candidate_set["mask"][1])
                    else:
                        mask_was_true = (self._last_rsu_choice.get(v.id) is not None)
                elif action_type == 'v2v':
                    candidate_set = self._last_candidate_set.get(v.id)
                    if candidate_set is not None and target_idx < len(candidate_set["mask"]):
                        mask_was_true = bool(candidate_set["mask"][target_idx])
                    else:
                        candidates = self._last_candidates.get(v.id, [])
                        neighbor_idx = target_idx - 2
                        if 0 <= neighbor_idx < len(candidates):
                            mask_was_true = (candidates[neighbor_idx] is not None and candidates[neighbor_idx] >= 0)
                else:  # local
                    mask_was_true = True  # Local永远可用
                
                audit_info['illegal_action'] = True
                audit_info['action_type'] = action_type
                audit_info['illegal_reason'] = plan['illegal_reason']
                audit_info['mask_was_true'] = mask_was_true
                break  # 记录第一个illegal即可
        
        # (4) RSU队列长度
        if self.rsus:
            rsu_queue_counts = []
            for rsu in self.rsus:
                # [P02修复] 从rsu_cpu_q计算队列长度
                proc_dict = self.rsu_cpu_q.get(rsu.id, {})
                queue_len = sum(len(q) for q in proc_dict.values())
                rsu_queue_counts.append(queue_len)
            audit_info['rsu_queue_len'] = np.mean(rsu_queue_counts) if rsu_queue_counts else 0

        return audit_info
    
    def _log_episode_stats(self, terminated, truncated):
        """
        记录episode统计信息到JSONL文件
        
        Args:
            terminated: 是否自然终止（所有任务完成）
            truncated: 是否被截断（时间限制）
        """
        if not hasattr(self, '_reward_stats'):
            return
        
        # 计算episode级统计
        episode_metrics = {}
        
        # 基本信息
        episode_metrics['episode_steps'] = self._episode_steps
        episode_metrics['terminated'] = terminated
        episode_metrics['truncated'] = truncated
        episode_metrics['terminated_reason'] = getattr(self, "_last_terminated_reason", "none")
        episode_metrics['idle_terminate_count'] = int(getattr(self, "_idle_terminate_count", 0))
        episode_metrics['seed'] = self.config.SEED if hasattr(self.config, 'SEED') else None
        episode_metrics['episode_time_seconds'] = self.time
        episode_metrics['time_limit_rate'] = 1.0 if (truncated and not terminated) else 0.0
        
        # 成功率统计
        episode_vehicle_count = len(self.vehicles)
        # [关键修复] 成功 = 完成且未超时失败
        success_count = sum([1 for v in self.vehicles 
                             if v.task_dag.is_finished and not v.task_dag.is_failed])
        episode_metrics['episode_vehicle_count'] = episode_vehicle_count
        episode_metrics['success_rate_end'] = success_count / max(episode_vehicle_count, 1)
        episode_metrics['task_success_rate'] = success_count / max(episode_vehicle_count, 1)
        episode_metrics['vehicle_success_rate'] = success_count / max(episode_vehicle_count, 1)
        
        # 子任务成功率
        # [语义说明] Subtask SR = 完成的subtask数 / 总subtask数
        # 注意：即使任务超时，完成的subtask也计入（反映实际执行进度）
        # Task SR反映deadline约束，Subtask SR反映执行完整性
        total_subtasks = 0
        completed_subtasks = 0
        for v in self.vehicles:
            total_subtasks += v.task_dag.num_subtasks
            completed_subtasks += np.sum(v.task_dag.status == 3)
        episode_metrics['total_subtasks'] = total_subtasks
        episode_metrics['subtask_success_rate'] = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
        
        # Deadline miss率
        # [关键修复] 使用is_failed标志而不是不存在的deadline_missed属性
        deadline_miss_count = sum([1 for v in self.vehicles 
                                   if v.task_dag.is_failed and v.task_dag.fail_reason == 'deadline'])
        episode_metrics['deadline_miss_rate'] = deadline_miss_count / max(episode_vehicle_count, 1)
        episode_metrics['audit_deadline_misses'] = deadline_miss_count
        
        # 决策分布
        if hasattr(self, '_decision_counts'):
            total_decisions = sum(self._decision_counts.values()) if self._decision_counts else 1
            episode_metrics['decision_frac_local'] = self._decision_counts.get('local', 0) / total_decisions
            episode_metrics['decision_frac_rsu'] = self._decision_counts.get('rsu', 0) / total_decisions
            episode_metrics['decision_frac_v2v'] = self._decision_counts.get('v2v', 0) / total_decisions
        if getattr(self, "_v2v_gain_count", 0) > 0:
            episode_metrics['v2v_gain_mean'] = self._v2v_gain_sum / self._v2v_gain_count
            episode_metrics['v2v_gain_pos_rate'] = self._v2v_gain_pos_count / self._v2v_gain_count
            episode_metrics['v2v_gain_pos_mean'] = (
                self._v2v_gain_pos_sum / self._v2v_gain_pos_count
                if self._v2v_gain_pos_count > 0 else 0.0
            )
        else:
            episode_metrics['v2v_gain_mean'] = 0.0
            episode_metrics['v2v_gain_pos_rate'] = 0.0
            episode_metrics['v2v_gain_pos_mean'] = 0.0
        
        # [P2性能统计] 服务率和空闲率
        if hasattr(self, '_p2_active_time') and hasattr(self, '_p2_idle_time'):
            total_time = self._p2_active_time + self._p2_idle_time
            if total_time > 0:
                episode_metrics['idle_fraction'] = self._p2_idle_time / total_time
                if self._p2_active_time > 0 and hasattr(self, '_p2_deltaW_active'):
                    episode_metrics['service_rate_when_active'] = self._p2_deltaW_active / self._p2_active_time
                else:
                    episode_metrics['service_rate_when_active'] = 0.0
            else:
                episode_metrics['idle_fraction'] = 0.0
                episode_metrics['service_rate_when_active'] = 0.0
        else:
            episode_metrics['idle_fraction'] = 0.0
            episode_metrics['service_rate_when_active'] = 0.0
        
        # 死锁统计
        deadlock_count = sum([1 for v in self.vehicles if hasattr(v, 'is_deadlocked') and v.is_deadlocked])
        episode_metrics['deadlock_vehicle_count'] = deadlock_count
        
        # 传输任务统计（从DAG的exec_locations统计）
        # [关键修复] 检查task_dag而不是vehicle是否有exec_locations属性
        tx_created_count = 0
        same_node_no_tx_count = 0
        for v in self.vehicles:
            if hasattr(v.task_dag, 'exec_locations'):
                for i, loc in enumerate(v.task_dag.exec_locations):
                    if loc is not None and loc != 'Local' and loc != v.id:
                        tx_created_count += 1
                    elif loc == v.id or loc == 'Local':
                        same_node_no_tx_count += 1
        episode_metrics['tx_tasks_created_count'] = tx_created_count
        episode_metrics['same_node_no_tx_count'] = same_node_no_tx_count

        # [P03新增] 详细动作统计
        if hasattr(self, '_p_target_raw'):
            total_raw = sum(self._p_target_raw.values()) or 1
            episode_metrics['p_target_raw_local'] = self._p_target_raw.get('local', 0)
            episode_metrics['p_target_raw_rsu'] = self._p_target_raw.get('rsu', 0)
            episode_metrics['p_target_raw_v2v'] = self._p_target_raw.get('v2v', 0)
            episode_metrics['p_target_raw_local_frac'] = self._p_target_raw.get('local', 0) / total_raw
            episode_metrics['p_target_raw_rsu_frac'] = self._p_target_raw.get('rsu', 0) / total_raw
            episode_metrics['p_target_raw_v2v_frac'] = self._p_target_raw.get('v2v', 0) / total_raw

        if hasattr(self, '_p_target_effective'):
            total_eff = sum(self._p_target_effective.values()) or 1
            episode_metrics['p_target_eff_local'] = self._p_target_effective.get('local', 0)
            episode_metrics['p_target_eff_rsu'] = self._p_target_effective.get('rsu', 0)
            episode_metrics['p_target_eff_v2v'] = self._p_target_effective.get('v2v', 0)
            episode_metrics['p_target_eff_local_frac'] = self._p_target_effective.get('local', 0) / total_eff
            episode_metrics['p_target_eff_rsu_frac'] = self._p_target_effective.get('rsu', 0) / total_eff
            episode_metrics['p_target_eff_v2v_frac'] = self._p_target_effective.get('v2v', 0) / total_eff

        if hasattr(self, '_fallback_reasons'):
            episode_metrics['fallback_reasons'] = dict(self._fallback_reasons)
            total_fb = sum(self._fallback_reasons.values())
            total_actions = sum(self._p_target_raw.values()) if hasattr(self, '_p_target_raw') else 1
            episode_metrics['fallback_rate'] = total_fb / max(total_actions, 1)

        if hasattr(self, "_episode_obs_stats"):
            obs_steps = max(self._episode_obs_stats.get("steps", 0), 1)
            episode_metrics['avail_L'] = self._episode_obs_stats.get("avail_L_sum", 0.0) / obs_steps
            episode_metrics['avail_R'] = self._episode_obs_stats.get("avail_R_sum", 0.0) / obs_steps
            episode_metrics['avail_V'] = self._episode_obs_stats.get("avail_V_sum", 0.0) / obs_steps
            episode_metrics['neighbor_count_mean'] = self._episode_obs_stats.get("neighbor_sum", 0.0) / obs_steps
            episode_metrics['best_v2v_rate_mean'] = self._episode_obs_stats.get("best_v2v_rate_sum", 0.0) / obs_steps
            episode_metrics['best_v2v_valid_rate'] = self._episode_obs_stats.get("best_v2v_valid_sum", 0.0) / obs_steps
            episode_metrics['v2v_beats_rsu_rate'] = self._episode_obs_stats.get("v2v_beats_rsu_sum", 0.0) / obs_steps
            cost_pair_count = self._episode_obs_stats.get("cost_pair_count", 0.0)
            if cost_pair_count > 0:
                episode_metrics['mean_cost_gap_v2v_minus_rsu'] = self._episode_obs_stats.get("cost_gap_sum", 0.0) / cost_pair_count
                episode_metrics['mean_cost_rsu'] = self._episode_obs_stats.get("cost_rsu_sum", 0.0) / cost_pair_count
                episode_metrics['mean_cost_v2v'] = self._episode_obs_stats.get("cost_v2v_sum", 0.0) / cost_pair_count
            else:
                episode_metrics['mean_cost_gap_v2v_minus_rsu'] = 0.0
                episode_metrics['mean_cost_rsu'] = 0.0
                episode_metrics['mean_cost_v2v'] = 0.0

        if self._episode_dT_eff_values:
            episode_metrics['dT_eff_mean'] = float(np.mean(self._episode_dT_eff_values))
            episode_metrics['dT_eff_p95'] = float(np.percentile(self._episode_dT_eff_values, 95))
        else:
            episode_metrics['dT_eff_mean'] = 0.0
            episode_metrics['dT_eff_p95'] = 0.0

        if self._episode_energy_norm_values:
            episode_metrics['energy_norm_mean'] = float(np.mean(self._episode_energy_norm_values))
            episode_metrics['energy_norm_p95'] = float(np.percentile(self._episode_energy_norm_values, 95))
        else:
            episode_metrics['energy_norm_mean'] = 0.0
            episode_metrics['energy_norm_p95'] = 0.0

        if self._episode_t_tx_values:
            episode_metrics['t_tx_mean'] = float(np.mean(self._episode_t_tx_values))
        else:
            episode_metrics['t_tx_mean'] = 0.0

        if self._episode_task_durations:
            episode_metrics['task_duration_mean'] = float(np.mean(self._episode_task_durations))
            episode_metrics['task_duration_p95'] = float(np.percentile(self._episode_task_durations, 95))
            episode_metrics['completed_tasks_count'] = len(self._episode_task_durations)
        else:
            episode_metrics['task_duration_mean'] = 0.0
            episode_metrics['task_duration_p95'] = 0.0
            episode_metrics['completed_tasks_count'] = 0

        if terminated or truncated:
            mean_cft_val = self._compute_mean_cft_pi0(snapshot_time=self.time)
            episode_metrics['mean_cft'] = mean_cft_val
            episode_metrics['mean_cft_rem'] = max(mean_cft_val - self.time, 0.0)
        else:
            episode_metrics['mean_cft'] = None
            episode_metrics['mean_cft_rem'] = None
        vehicle_cfts = getattr(self, "vehicle_cfts", [])
        finite_cfts = [val for val in vehicle_cfts if np.isfinite(val)]
        episode_metrics['vehicle_cft_count'] = len(finite_cfts)
        if finite_cfts:
            episode_metrics['mean_cft_est'] = float(np.mean(finite_cfts))
            episode_metrics['cft_est_valid'] = True
        else:
            episode_metrics['mean_cft_est'] = 0.0
            episode_metrics['cft_est_valid'] = False
        if self._episode_task_durations:
            episode_metrics['mean_cft_completed'] = float(np.mean(self._episode_task_durations))
        else:
            episode_metrics['mean_cft_completed'] = 0.0

        # [P03新增] delta_phi分布统计
        if hasattr(self, '_episode_delta_phi_values') and len(self._episode_delta_phi_values) > 0:
            dphi = np.array(self._episode_delta_phi_values)
            episode_metrics['delta_phi_mean'] = float(np.mean(dphi))
            episode_metrics['delta_phi_p50'] = float(np.percentile(dphi, 50))
            episode_metrics['delta_phi_p95'] = float(np.percentile(dphi, 95))
            episode_metrics['delta_phi_min'] = float(np.min(dphi))
            episode_metrics['delta_phi_max'] = float(np.max(dphi))

        # [P03新增] Clip命中率统计
        if hasattr(self, '_episode_reward_count') and self._episode_reward_count > 0:
            episode_metrics['shape_clip_hit_rate'] = self._episode_shape_clip_count / self._episode_reward_count
            episode_metrics['r_total_clip_hit_rate'] = self._episode_r_total_clip_count / self._episode_reward_count
        else:
            episode_metrics['shape_clip_hit_rate'] = 0.0
            episode_metrics['r_total_clip_hit_rate'] = 0.0

        # 非法/无任务统计
        episode_metrics['illegal_count'] = int(getattr(self, "_episode_illegal_count", 0))
        episode_metrics['no_task_count'] = int(getattr(self, "_episode_no_task_count", 0))
        episode_metrics['illegal_reasons'] = dict(getattr(self, "_episode_illegal_reasons", {}))

        # 从reward_stats提取统计信息
        metrics_dict = {}
        for name, bucket in self._reward_stats.metrics.items():
            if bucket.count > 0:
                metrics_dict[name] = {
                    'mean': bucket.sum / bucket.count,
                    'p95': bucket.get_percentile(0.95) if hasattr(bucket, 'get_percentile') else None,
                    'count': bucket.count
                }
        
        # 保存到实例变量（供train.py使用）
        self._last_episode_metrics = episode_metrics.copy()
        
        # 只在episode结束时写入JSONL文件
        if terminated or truncated:
            jsonl_path = os.environ.get('REWARD_JSONL_PATH')
            if jsonl_path:
                try:
                    import json
                    
                    # 转换numpy类型为Python原生类型
                    def convert_to_native(obj):
                        if isinstance(obj, dict):
                            return {k: convert_to_native(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [convert_to_native(v) for v in obj]
                        elif isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        else:
                            return obj
                    
                    record = {
                        'episode': getattr(self, 'episode_count', 0),
                        'metrics': convert_to_native(metrics_dict),
                        **convert_to_native(episode_metrics)
                    }
                    
                    with open(jsonl_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=True) + '\n')
                except Exception as e:
                    import warnings
                    warnings.warn(f"[JSONL写入失败] {e}", UserWarning)
    
