import gymnasium as gym
import numpy as np
import os
import csv
import random
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
from envs.rl.reward_functions import (
    compute_absolute_reward,
    compute_unified_step_reward, compute_unified_terminal_reward,
    compute_unified_pbrs, compute_phi_lb,
)
from envs.services.rsu_selector import RSUSelector
from envs.services.candidate_set_manager import CandidateSetManager
from envs.chain_proxy_sim import ChainProxySim
from envs.modules.trust import TrustManager


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
    - action_mask: 动作目标掩码 (Local/RSU/V2V)，仅对应当前选中的任务
    - subtask_index: 当前环境自动选择的任务索引 (标量)

    动作空间 (Action Space):
    - target: 卸载目标 (0=Local, 1=RSU, 2+k=Vehicle k)
    - subtask: 要调度的子任务索引
    - power: 传输功率控制

    奖励设计:
    - CFT奖励: 任务完成时间越短越好
    - 拥堵惩罚: V2V/V2I信道拥塞时产生惩罚
    """

    # Decision outcome semantics (paper-level metric contract)
    NO_TASK_REASONS = {
        "task_done",
        "no_task_dag_done",
        "no_task_dag_failed",
        "no_task_blocked",
        "no_task_assigned",
        "inflight_limit",
    }
    ILLEGAL_ACTION_REASONS = {
        "subtask_idx_out_of_range",
        "masked_subtask",
        "idx_out_of_range",
        "masked_target",
        "rsu_unavailable",
        "rsu_out_of_coverage",
        "rsu_queue_full",
        "queue_full_conflict",
        "no_candidate_cache",
        "id_mapping_fail",
        "power_invalid",
        "action_format_invalid",
    }

    def __init__(self, config=None):
        """初始化环境
        
        Args:
            config: 配置类（可选，默认使用全局Cfg）
        """
        # 使用传入的config或全局Cfg
        self.config = config if config is not None else Cfg
        
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
        # fallback速率计算仅在未传入rate_fn时使用，需要解析节点位置
        self._comm_service.set_node_resolvers(
            veh_lookup=self._get_vehicle_by_id,
            rsu_lookup=lambda rid: self.rsus[rid] if 0 <= rid < len(self.rsus) else None,
        )
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
        # [研究指标] UNIFIED 分项与区块链信誉 oracle（不启用 ChainProxySim 也可用）
        self._episode_rho_selected_values = []
        self._episode_uncertainty_selected_values = []
        self._episode_risk_penalty_values = []
        self._episode_I_total_values = []
        self._episode_I_caused_input_values = []
        self._reward_ref_ema = {
            "energy": float(max(getattr(self.config, "E_REF_UNIFIED", Cfg.P_MAX_WATT * self.config.DT), 1e-12)),
            "interf": float(max(getattr(self.config, "I_REF_MIN_UNIFIED", 1e-8), 1e-12)),
            "risk": float(max(getattr(self.config, "RISK_REF_UNIFIED_INIT", 0.25), 1e-6)),
        }
        self._last_obs_stamp = None
        # [P2性能统计] 运行期累积器
        self._p2_active_time = 0.0
        self._p2_idle_time = 0.0
        self._p2_deltaW_active = 0.0
        self._p2_zero_delta_steps = 0
        self._p2_comm_only_steps = 0
        self._p2_B_prev = 0.0
        
        # [审计系统] 12项核心指标收集
        self._audit_step_info = {}
        # [Deadline检查计数] 用于诊断是否触发deadline判定
        self._audit_deadline_checks = 0
        self._audit_deadline_misses = 0
        # [审计] 奖励方案生效与t_est对比
        self._audit_results_dir = os.environ.get("AUDIT_RESULTS_DIR")
        self._audit_scheme_activation_written = False
        self._audit_subtask_est = {}
        self._audit_t_est_records = []
        self._audit_per_decision_rewards = []  # [审计] per-decision奖励分项记录
        self._last_commit_plans = []  # [审计] 保存commit_plans供per-decision审计使用
        self._audit_t_est_path = os.environ.get("AUDIT_T_EST_REAL_PATH")
        self._audit_run_id = os.environ.get("RUN_ID")
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
        self._episode_hard_trigger_count = 0
        self._episode_illegal_reasons = {}
        self._episode_no_task_reasons = {}
        self._unified_nonfinite_count = 0
        self._unified_consistency_mismatch_count = 0
        self._unified_illegal_trigger_count = 0
        self._episode_candidate_stats = {"reachable": [], "dropped": []}
        self._last_candidate_step_stats = {}
        self._episode_not_in_candidate_fallback_cnt = 0
        self._episode_illegal_by_connectivity_cnt = 0
        self._episode_domain_params = {}

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
        # V2V 零干扰参考速率预计算常量（obs col3 用 R_ref 代替含干扰估计的 rate）
        self._v2v_ref_p_w = self.config.P_MAX_WATT  # P_ref = P_MAX
        self._v2v_ref_bw = self.config.V2V_BW_PER_RB
        self._v2v_ref_noise_w = self.channel.noise_psd_w_hz * self._v2v_ref_bw
        # log 压缩归一：R_ref_max = R_ref(d=1m)，log_max = log(1 + R_ref_max)
        g_d1 = self.channel.beta0 * (1.0 ** (-self.config.PL_BETA_V2V))
        snr_max = self._v2v_ref_p_w * g_d1 / max(self._v2v_ref_noise_w, 1e-30)
        r_ref_max = self._v2v_ref_bw * np.log2(1.0 + snr_max)
        self._v2v_ref_log_max = float(np.log(1.0 + r_ref_max))  # 用于 col3 归一
        self._max_rsu_contact_time = self.config.RSU_RANGE / max(self.config.VEL_MIN, 1e-6)
        self._max_v2v_contact_time = self.config.V2V_RANGE / max(self.config.VEL_MIN, 1e-6)
        # 动态车辆统计：记录整个episode出现过的车辆ID
        self._vehicles_seen = set()
        # 奖励/通信快照
        self._rate_snapshot = None
        self._rate_snapshot_step = -1
        self._rate_snapshot_token = None
        self._rate_snapshot_prev = None
        self._rate_snapshot_prev_step = -1
        self._rate_prev_cache_v2i = np.zeros(self.config.NUM_VEHICLES, dtype=np.float32)
        self._rate_prev_cache_v2v = np.zeros(self.config.NUM_VEHICLES, dtype=np.float32)
        self._f_max_const = max(self.config.MAX_VEHICLE_CPU_FREQ, getattr(self.config, "F_RSU", 0.0))
        # Chain proxy (settlement risk layer)
        self.chain_proxy = None
        self.chain_state_dict = {
            "p50_confirm": 0.0,
            "p95_confirm": 0.0,
            "p_fail": 0.0,
            "mempool_len": 0.0,
            "rho": 0.0,
        }
        self.chain_state_vec = np.zeros(self.config.CHAIN_OBS_DIM, dtype=np.float32)
        self._chain_tx_total = 0
        self._chain_risk_cost_total = 0.0
        self._chain_p95_sum = 0.0
        self._chain_pfail_sum = 0.0
        self._chain_steps = 0
        
        # =====================================================================
        # [Gymnasium接口] 定义动作空间和观测空间
        # =====================================================================
        # 动作空间：Tuple of Dict (每个车辆一个动作)
        # 每个车辆的动作:
        # - subtask: 子任务索引（Stage 1/2 接口预留，环境暂不使用）
        # - target: 0=Local, 1=RSU, 2...(2+MAX_NEIGHBORS-1)=V2V邻居
        # - power: 连续rho∈[0,1]（Beta输出），与环境解析保持一致
        single_agent_action_space = gym.spaces.Dict({
            "subtask": gym.spaces.Discrete(self.config.MAX_NODES),
            "target": gym.spaces.Discrete(self.config.MAX_TARGETS),
            "power": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })
        self.action_space = gym.spaces.Tuple([single_agent_action_space] * self.config.NUM_VEHICLES)
        
        # 观测空间：Dict空间（具体维度在reset后确定）
        # 这里定义一个占位符，实际维度在reset()后根据DAG大小动态确定
        max_loc_id = max(self.config.NUM_VEHICLES + 2, 3)
        self.observation_space = gym.spaces.Dict({
            'node_x': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.MAX_NODES, 7), dtype=np.float32),
            'self_info': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            'rsu_info': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'serving_rsu_onehot': gym.spaces.Box(low=0.0, high=1.0, shape=(self.config.NUM_RSU,), dtype=np.float32),
            'candidate_ids': gym.spaces.Box(low=-1, high=max(self.config.NUM_VEHICLES, 1), shape=(self.config.MAX_TARGETS,), dtype=np.int64),
            'candidate_types': gym.spaces.Box(low=0, high=3, shape=(self.config.MAX_TARGETS,), dtype=np.int8),
            'adj': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_NODES, self.config.MAX_NODES), dtype=np.float32),
            'neighbors': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.MAX_NEIGHBORS, 8), dtype=np.float32),
            'subtask_mask': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_NODES,), dtype=np.float32),
            'node_valid_mask': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_NODES,), dtype=np.float32),
            'task_mask': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_NODES,), dtype=np.float32),
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_TARGETS,), dtype=np.float32),
            'rate_prev': gym.spaces.Box(low=0.0, high=1.0, shape=(self.config.MAX_TARGETS,), dtype=np.float32),
            'resource_ids': gym.spaces.Box(low=0, high=max_loc_id, shape=(self.config.MAX_TARGETS,), dtype=np.int64),
            'resource_raw': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.config.MAX_TARGETS, self.config.RESOURCE_RAW_DIM),
                dtype=np.float32
            ),
            'subtask_index': gym.spaces.Box(low=-1, high=self.config.MAX_NODES - 1, shape=(), dtype=np.int64),
            'L_fwd': gym.spaces.Box(low=0, high=self.config.MAX_NODES, shape=(self.config.MAX_NODES,), dtype=np.int32),
            'L_bwd': gym.spaces.Box(low=0, high=self.config.MAX_NODES, shape=(self.config.MAX_NODES,), dtype=np.int32),
            'data_matrix': gym.spaces.Box(low=0, high=1, shape=(self.config.MAX_NODES, self.config.MAX_NODES), dtype=np.float32),
            'Delta': gym.spaces.Box(low=0, high=self.config.MAX_NODES, shape=(self.config.MAX_NODES, self.config.MAX_NODES), dtype=np.int32),
            'location': gym.spaces.Box(low=0, high=max_loc_id, shape=(self.config.MAX_NODES,), dtype=np.int32),
            'obs_stamp': gym.spaces.Box(low=0, high=self.config.MAX_STEPS, shape=(), dtype=np.int64),
            'global_state': gym.spaces.Box(low=0.0, high=1.0, shape=(30,), dtype=np.float32),
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

    def _v2v_ref_rate(self, dist):
        """V2V 零干扰参考速率 R_ref = B_RB * log2(1 + P_ref*G(d)/(N0*B_RB))"""
        d = max(float(dist), 1.0)
        g_d = self.channel.beta0 * (d ** (-self.config.PL_BETA_V2V))
        snr = self._v2v_ref_p_w * g_d / max(self._v2v_ref_noise_w, 1e-30)
        return self._v2v_ref_bw * np.log2(1.0 + snr)

    def _v2v_ref_rate_norm(self, dist):
        """log 压缩归一的 V2V 参考速率：log(1+R_ref)/log(1+R_ref_max) ∈ [0,1]"""
        r = self._v2v_ref_rate(dist)
        return float(np.log(1.0 + r) / max(self._v2v_ref_log_max, 1e-30))

    def _rate_key(self, src_node, dst_node, link_type):
        """统一的速率key，确保phase3和reward使用同一标识"""
        return (
            link_type,
            src_node[0] if src_node else None,
            src_node[1] if src_node else None,
            dst_node[0] if dst_node else None,
            dst_node[1] if dst_node else None,
        )

    def _compute_pair_rate(self, src_node, dst_node, link_type, power_dbm=None,
                           active_v2i_count=None, active_v2v_vehicles=None):
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
            v2i_count = active_v2i_count
            if v2i_count is None and link_type == "V2I":
                v2i_count = self._estimate_v2i_users()
            active_v2v_list = None
            if link_type == "V2V" and active_v2v_vehicles is not None:
                active_v2v_list = [v for v in active_v2v_vehicles if v is not None and v.id != src_veh.id]
            return self.channel.compute_one_rate(
                src_veh,
                dst_pos,
                link_type,
                self.time,
                power_dbm_override=power_dbm,
                v2i_user_count=v2i_count if link_type == "V2I" else None,
                active_tx_vehicles=active_v2v_list if link_type == "V2V" else None,
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
        v2i_count = active_v2i_count if active_v2i_count is not None else self._estimate_v2i_users()
        return self.channel.compute_one_rate(
            rsu_proxy,
            dst_pos,
            "V2I",
            self.time,
            power_dbm_override=power_dbm,
            v2i_user_count=v2i_count,
        )

    def _capture_rate_snapshot(self, commit_plans):
        """在SNAPSHOT_PRE冻结本步速率，用于phase3与奖励。
        只读 env 状态（队列 + commit_plans），不写任何 env 字段。"""
        rsu_pos_default = self.rsus[0].position if len(self.rsus) > 0 else self.config.RSU_POS
        rsu_pos_map = {rsu.id: rsu.position for rsu in self.rsus}
        eps_rate = getattr(self.config, "EPS_RATE", 1e-9)

        # ── 1) 从队列 + commit_plans 收集显式链路列表 ──
        v2v_links = []       # [{sender_id, tx_pos, rx_pos, power_w, tx_kind}]
        v2i_links = []       # [{sender_id, tx_pos, rsu_id, power_w, tx_kind}]
        v2v_target_map = {}  # sender_vid -> recv_vid （用于 link_rates key）
        v2i_rsu_map = {}     # sender_vid -> rsu_id
        seen_v2v = set()
        seen_v2i = set()

        def _add_v2v(sid, rid, power_dbm, tx_kind="INPUT"):
            if sid in seen_v2v:
                return
            seen_v2v.add(sid)
            tx = self._get_vehicle_by_id(sid)
            rx = self._get_vehicle_by_id(rid)
            if tx is None or rx is None:
                return
            v2v_target_map[sid] = rid
            v2v_links.append({
                'sender_id': sid,
                'tx_pos': np.array(tx.pos, dtype=float),
                'rx_pos': np.array(rx.pos, dtype=float),
                'power_w': self.config.dbm2watt(power_dbm if power_dbm is not None else tx.tx_power_dbm),
                'tx_kind': str(tx_kind).upper(),
            })

        def _add_v2i(sid, rsu_id, power_dbm, tx_kind="INPUT"):
            if sid in seen_v2i:
                return
            seen_v2i.add(sid)
            tx = self._get_vehicle_by_id(sid)
            if tx is None:
                return
            v2i_rsu_map[sid] = rsu_id
            v2i_links.append({
                'sender_id': sid,
                'tx_pos': np.array(tx.pos, dtype=float),
                'rsu_id': rsu_id,
                'power_w': self.config.dbm2watt(power_dbm if power_dbm is not None else tx.tx_power_dbm),
                'tx_kind': str(tx_kind).upper(),
            })

        # a) 现有队列
        for tx_node, queue in self.txq_v2v.items():
            if tx_node[0] == "VEH" and queue:
                sid = int(tx_node[1])
                for job in queue:
                    if job.dst_node[0] == "VEH":
                        _add_v2v(sid, int(job.dst_node[1]),
                                 getattr(job, "tx_power_dbm", None),
                                 getattr(job, "kind", "INPUT"))
                        break
        for tx_node, queue in self.txq_v2i.items():
            if tx_node[0] == "VEH" and queue:
                sid = int(tx_node[1])
                for job in queue:
                    if job.dst_node[0] == "RSU":
                        _add_v2i(sid, int(job.dst_node[1]),
                                 getattr(job, "tx_power_dbm", None),
                                 getattr(job, "kind", "INPUT"))
                        break

        # b) 本步 commit_plans
        for plan in commit_plans:
            tgt = plan.get("planned_target")
            if tgt is None or tgt == "Local":
                continue
            vid = int(plan["vehicle_id"])
            pdbm = plan.get("power_dbm")
            if isinstance(tgt, int):
                _add_v2v(vid, int(tgt), pdbm, "INPUT")
            elif isinstance(tgt, tuple) and tgt[0] == "RSU":
                _add_v2i(vid, int(tgt[1]), pdbm, "INPUT")

        # ── 2) 调用新的显式接口计算速率（不修改任何 env 状态）──
        v2v_rates = self.channel.compute_v2v_rb_sinr(v2v_links)
        v2i_rates = self.channel.compute_v2i_rates(v2i_links, rsu_pos_map, rsu_pos_default)

        # ── 3) 写入 link_rates ──
        link_rates = {}
        for sid, rid in v2v_target_map.items():
            if sid in v2v_rates:
                key = self._rate_key(("VEH", sid), ("VEH", rid), "V2V")
                link_rates[key] = max(v2v_rates[sid], eps_rate)
        for sid, rsu_id in v2i_rsu_map.items():
            if sid in v2i_rates:
                key = self._rate_key(("VEH", sid), ("RSU", rsu_id), "V2I")
                link_rates[key] = max(v2i_rates[sid], eps_rate)

        active_v2i_count = len(seen_v2i)
        active_v2v_vehicles = [self._get_vehicle_by_id(vid) for vid in seen_v2v]

        def _add_pair(src_node, dst_node, link_type, power_dbm=None):
            key = self._rate_key(src_node, dst_node, link_type)
            if key in link_rates:
                return
            rate = self._compute_pair_rate(
                src_node, dst_node, link_type, power_dbm,
                active_v2i_count=active_v2i_count,
                active_v2v_vehicles=active_v2v_vehicles,
            )
            link_rates[key] = max(rate, eps_rate)

        # 4) 队列中其他 job（多目标情况的 fallback）
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
            types = candidate_set.get("types", [])
            # [BugFix] 统一按types判断实体类型，覆盖所有RSU/V2V slot
            for idx in range(1, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id < 0:
                        continue
                    cand_type = int(types[idx]) if idx < len(types) else 0
                    if cand_type == 2:  # RSU → V2I pair
                        _add_pair(("VEH", v.id), ("RSU", cand_id), "V2I", tx_power_dbm)
                    elif cand_type == 3:  # V2V pair
                        _add_pair(("VEH", v.id), ("VEH", cand_id), "V2V", tx_power_dbm)

        # Handover freeze: 冻结期内 V2I 速率=0
        frozen_vids = set()
        for v in self.vehicles:
            freeze = getattr(v, '_ho_freeze_remain', 0)
            if freeze > 0:
                frozen_vids.add(v.id)
                v._ho_freeze_remain = freeze - 1
        if frozen_vids:
            # key = (link_type, src_type, src_id, dst_type, dst_id)
            keys_to_zero = [
                k for k in link_rates
                if k[0] == 'V2I' and k[1] == 'VEH' and k[2] in frozen_vids
            ]
            for k in keys_to_zero:
                link_rates[k] = 0.0

        # 合并 v2v + v2i 的原始速率（兼容旧字段 raw_rates）
        _raw_rates = {}
        _raw_rates.update(v2v_rates)
        _raw_rates.update(v2i_rates)
        self._rate_snapshot = {
            "step": self.steps,
            "raw_rates": _raw_rates,
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

    def _classify_decision_state(self, illegal_reason, subtask_idx):
        if illegal_reason in self.NO_TASK_REASONS:
            return "no_task_available"
        if illegal_reason in self.ILLEGAL_ACTION_REASONS:
            return "illegal_action"
        if subtask_idx is None:
            return "no_task_available"
        return "valid_action"

    def _annotate_plan_decision_state(self, plan):
        state = self._classify_decision_state(plan.get("illegal_reason"), plan.get("subtask_idx"))
        plan["decision_state"] = state
        plan["is_no_task_available"] = (state == "no_task_available")
        plan["is_illegal_action"] = (state == "illegal_action")
        plan["is_valid_action"] = (state == "valid_action")
        return plan

    def _plan_actions_snapshot(self, actions):
        plans = []
        max_schedule = max(1, int(getattr(self.config, "MAX_SCHEDULE_PER_STEP", 1)))
        inflight_limit = int(getattr(self.config, "MAX_INFLIGHT_SUBTASKS_PER_VEHICLE", 0))
        
        # [P0修复] 随机打乱车辆处理顺序，消除queue_full检查的ID偏置
        # 修复前：按vehicle_id顺序处理，低ID优先检查队列，高ID更易遇到队列满
        # 修复后：可复现随机顺序，消除ID与queue_full的相关性
        import random
        vehicle_indices = list(range(len(self.vehicles)))
        shuffle_rng = random.Random(self.episode_count * 10000 + self.steps)
        shuffle_rng.shuffle(vehicle_indices)
        
        for i in vehicle_indices:
            v = self.vehicles[i]
            schedule_limit = max_schedule
            if v.task_dag.is_finished or v.task_dag.is_failed:
                plan = {
                    "vehicle": v,
                    "vehicle_id": v.id,
                    "index": i,
                    "requested_subtask_idx": None,
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
                    "power_invalid": False,
                    "power_ratio": None,
                    "power_dbm": None,
                }
                plans.append(plan)
                continue
            plan = {
                "vehicle": v,
                "vehicle_id": v.id,
                "index": i,
                "requested_subtask_idx": None,
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
                "power_invalid": False,
                "power_ratio": None,
                "power_dbm": None,
            }
            # 防御性处理：如果actions长度小于车辆数，缺失的动作默认回退到Local
            if i >= len(actions):
                act = {'subtask': 0, 'target': 0, 'power': 1.0}
            else:
                act = actions[i]
            if act is None:
                plans.append(plan)
                continue

            # [改动C] 动作接口统一为 dict: {"target": int, "power": float in [0,1]}
            # 移除离散 power_level 分支，Agent 使用 Beta 分布输出连续功率 rho∈[0,1]
            if isinstance(act, dict):
                try:
                    requested_subtask_idx = int(act.get("subtask", -1))
                except Exception:
                    requested_subtask_idx = -1
                    if plan["illegal_reason"] is None:
                        plan["illegal_reason"] = "action_format_invalid"
                try:
                    target_idx = int(act.get("target", 0))
                except Exception:
                    target_idx = 0
                    plan["illegal_reason"] = "action_format_invalid"
                raw_power = act.get("power", 1.0)
            else:
                # 兼容数组格式：[target_idx, power_ratio]，power 直接为连续值
                act_array = np.asarray(act).flatten()
                requested_subtask_idx = -1
                try:
                    target_idx = int(act_array[0]) if len(act_array) > 0 else 0
                except Exception:
                    target_idx = 0
                    plan["illegal_reason"] = "action_format_invalid"
                raw_power = act_array[1] if len(act_array) > 1 else 1.0

            try:
                raw_power_f = float(raw_power)
            except Exception:
                raw_power_f = float("nan")
            plan["power_invalid"] = (not np.isfinite(raw_power_f)) or (raw_power_f < 0.0) or (raw_power_f > 1.0)
            p_norm = float(np.clip(np.nan_to_num(raw_power_f, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
            
            plan["requested_subtask_idx"] = requested_subtask_idx
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

            # Stage 3：环境正式使用 RL 给出的 subtask（严格校验，不再静态 top-priority 兜底）
            subtask_idx = None
            extra_subtasks = []
            schedulable_mask = np.asarray(dag.get_action_mask(), dtype=bool)
            if not np.any(schedulable_mask):
                # 无可调度任务：no-task 语义（非非法动作）
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
            if plan.get("illegal_reason") is not None:
                # 动作格式/功率等已非法，直接拦截，不进行本地回退
                plans.append(plan)
                continue
            if requested_subtask_idx is None:
                requested_subtask_idx = -1
            if requested_subtask_idx < 0 or requested_subtask_idx >= dag.num_subtasks:
                plan["illegal_reason"] = "subtask_idx_out_of_range"
                plans.append(plan)
                continue
            if requested_subtask_idx >= len(schedulable_mask) or (not bool(schedulable_mask[requested_subtask_idx])):
                plan["illegal_reason"] = "masked_subtask"
                plans.append(plan)
                continue
            subtask_idx = int(requested_subtask_idx)

            task_comp = v.task_dag.total_comp[subtask_idx] if subtask_idx < len(v.task_dag.total_comp) else self.config.MEAN_COMP_LOAD
            task_data = v.task_dag.total_data[subtask_idx] if subtask_idx < len(v.task_dag.total_data) else 0.0
            plan["subtask_idx"] = subtask_idx
            plan["extra_subtask_indices"] = list(extra_subtasks)
            plan["task_comp"] = task_comp
            plan["task_data"] = task_data

            # 功率映射：P = Pmin * (Pmax/Pmin)^a_power (对数域均匀)
            p_min_w = Cfg.dbm2watt(self.config.TX_POWER_MIN_DBM)
            p_max_w = Cfg.dbm2watt(self.config.TX_POWER_MAX_DBM)
            p_watt = p_min_w * (p_max_w / max(p_min_w, 1e-12)) ** float(p_norm)
            raw_power = Cfg.watt2dbm(p_watt)
            plan["power_ratio"] = p_norm
            plan["power_dbm"] = np.clip(raw_power, self.config.TX_POWER_MIN_DBM, self.config.TX_POWER_MAX_DBM)
            plan["power_watt"] = float(p_watt)

            desired_target = 'Local'
            desired_kind = "local"
            if plan.get("power_invalid", False) and plan.get("illegal_reason") is None:
                plan["illegal_reason"] = "power_invalid"
            
            # 计算RSU和V2V的action索引边界
            enable_rsu_selection = getattr(self.config, 'ENABLE_RSU_SELECTION', False)
            num_rsu = len(self.rsus)
            rsu_start_idx = 1  # RSU选项从index 1开始
            rsu_end_idx = (1 + num_rsu) if enable_rsu_selection else 2  # RSU选项结束索引(不含)
            v2v_start_idx = rsu_end_idx  # V2V选项起始索引

            if plan.get("illegal_reason") is not None:
                desired_target = 'Local'
                desired_kind = "local"
            elif target_idx >= self.config.MAX_TARGETS:
                plan["illegal_reason"] = "idx_out_of_range"
            elif target_idx == 0:
                desired_target = 'Local'
                desired_kind = "local"
            elif rsu_start_idx <= target_idx < rsu_end_idx:
                # RSU动作：target_idx映射到具体RSU_id
                candidate_set = self._last_candidate_set.get(v.id)
                candidate_mask = candidate_set["mask"] if candidate_set is not None else None
                
                if enable_rsu_selection:
                    # 新模式：agent直接选择RSU_id
                    selected_rsu_id = target_idx - rsu_start_idx  # target_idx=1 -> RSU_0
                else:
                    # 旧模式：env自动选择serving RSU
                    selected_rsu_id = getattr(v, "serving_rsu_id", None)
                    if selected_rsu_id is None:
                        selected_rsu_id = self._update_serving_rsu(v)
                
                # 检查mask
                if candidate_mask is not None and (target_idx >= len(candidate_mask) or not candidate_mask[target_idx]):
                    plan["illegal_reason"] = "masked_target"
                    desired_target = 'Local'
                    desired_kind = "local"
                elif selected_rsu_id is None or not (0 <= selected_rsu_id < num_rsu):
                    plan["illegal_reason"] = "rsu_unavailable"
                    desired_target = 'Local'
                    desired_kind = "local"
                else:
                    rsu = self.rsus[selected_rsu_id]
                    if rsu is None:
                        plan["illegal_reason"] = "rsu_unavailable"
                        desired_target = 'Local'
                        desired_kind = "local"
                    elif not rsu.is_in_coverage(v.pos):
                        plan["illegal_reason"] = "rsu_out_of_coverage"
                        desired_target = 'Local'
                        desired_kind = "local"
                    elif self._is_rsu_queue_full(selected_rsu_id, task_comp):
                        plan["illegal_reason"] = "rsu_queue_full"
                        desired_target = 'Local'
                        desired_kind = "local"
                    else:
                        desired_target = ('RSU', selected_rsu_id)
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

            # [P0修复] 使用随机顺序处理冲突，消除vehicle_id偏置
            import random
            shuffle_rng = random.Random(self.episode_count * 10000 + self.steps + rsu_id)
            shuffled_reqs = list(reqs)
            shuffle_rng.shuffle(shuffled_reqs)
            for plan in shuffled_reqs:
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

            # [P0修复] 使用随机顺序处理V2V冲突，消除vehicle_id偏置
            import random
            shuffle_rng = random.Random(self.episode_count * 10000 + self.steps + tgt_id)
            shuffled_reqs = list(reqs)
            shuffle_rng.shuffle(shuffled_reqs)
            for plan in shuffled_reqs:
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

        for plan in plans:
            self._annotate_plan_decision_state(plan)

        return plans

    def _update_reward_ref_ema(self, key, p95_value):
        alpha = float(np.clip(getattr(self.config, "REWARD_REF_EMA_ALPHA", 0.05), 1e-3, 1.0))
        if key == "energy":
            floor = float(max(getattr(self.config, "REWARD_REF_ENERGY_MIN", 1e-8), 1e-12))
        elif key == "interf":
            floor = float(max(getattr(self.config, "I_REF_MIN_UNIFIED", 1e-8), 1e-12))
        else:
            floor = float(max(getattr(self.config, "REWARD_REF_RISK_MIN", 0.05), 1e-6))
        cap = float(max(getattr(self.config, "REWARD_REF_EMA_CAP", 1e3), floor))
        curr = float(np.clip(np.nan_to_num(p95_value, nan=0.0, posinf=cap, neginf=0.0), floor, cap))
        warmup_eps = int(max(getattr(self.config, "REWARD_REF_WARMUP_EPISODES", 0), 0))
        freeze_after = bool(getattr(self.config, "REWARD_REF_FREEZE_AFTER_WARMUP", True))
        episode_idx = int(getattr(self, "episode_count", 0))
        if freeze_after and warmup_eps > 0 and episode_idx > warmup_eps:
            frozen = float(self._reward_ref_ema.get(key, curr))
            frozen = float(np.clip(np.nan_to_num(frozen, nan=curr, posinf=cap, neginf=floor), floor, cap))
            self._reward_ref_ema[key] = frozen
            return frozen
        prev = float(self._reward_ref_ema.get(key, curr))
        new_val = (1.0 - alpha) * prev + alpha * curr
        self._reward_ref_ema[key] = float(np.clip(new_val, floor, cap))
        return self._reward_ref_ema[key]

    def _init_rsus(self):
        """
        初始化RSU实体列表（道路模型：等间距双侧交替部署）

        部署规则：
        - RSU 均匀分布在 [0, MAP_SIZE] 上，间距 d = MAP_SIZE / NUM_RSU
        - 偶数编号 RSU（0,2,...）位于道路右侧（+Y）：y =  road_width + RSU_Y_DIST
        - 奇数编号 RSU（1,3,...）位于道路左侧（-Y）：y = -RSU_Y_DIST
        - 双侧部署减少单侧遮挡，x 方向水平覆盖 ≈ d/2，相邻 RSU 仅有少量边界重叠
        """
        num_rsu = getattr(Cfg, 'NUM_RSU', 3)
        self.rsus = []

        road_width = getattr(Cfg, 'NUM_LANES', 3) * getattr(Cfg, 'LANE_WIDTH', 3.5)
        rsu_y_dist = getattr(Cfg, 'RSU_Y_DIST', 10.0)
        y_right = road_width + rsu_y_dist   # 右侧（上方），例如 +17m
        y_left  = -rsu_y_dist               # 左侧（下方），例如 -10m

        map_size = self.config.MAP_SIZE
        d_inter = map_size / num_rsu        # 等间距

        # 全覆盖校验：用两侧中较大的 h 做保守估计
        h_max = max(abs(y_right), abs(y_left))
        min_range = np.sqrt((d_inter / 2) ** 2 + h_max ** 2)
        rsu_range = max(self.config.RSU_RANGE, min_range * 1.02)  # 留2%裕量
        if rsu_range > self.config.RSU_RANGE:
            self.config.RSU_RANGE = rsu_range

        for i in range(num_rsu):
            x_pos = (i * d_inter) + (d_inter / 2)
            y_pos = y_right if (i % 2 == 0) else y_left   # 偶数右侧，奇数左侧
            pos = np.array([x_pos, y_pos])

            rsu = RSU(
                rsu_id=i,
                position=pos,
                cpu_freq=self.config.F_RSU,
                num_processors=getattr(Cfg, 'RSU_NUM_PROCESSORS', 6),
                queue_limit=100,
                coverage_range=rsu_range,
            )
            self.rsus.append(rsu)

        if self.rsus:
            mid = len(self.rsus) // 2
            self.config.RSU_POS = self.rsus[mid].position.copy()

    def _get_nearest_rsu(self, position):
        """获取距离指定位置最近的RSU（委托给RSUSelector）"""
        return self.rsu_selector.get_nearest_rsu(position)

    def _get_all_rsus_in_range(self, position):
        """获取覆盖范围内所有RSU（委托给RSUSelector）"""
        return self.rsu_selector.get_all_rsus_in_range(position)

    def _update_serving_rsu(self, vehicle):
        """
        RSU 切换判据: 滞回 + 最小驻留 + 冻结
        
        Q(i,k) = P_ref * G(d_{i,k})  (参考信号质量)
        切换条件: Q_new > Q_curr + HO_HYST_DB 且已驻留 >= MIN_RSU_STAY_STEPS
        切换后: HO_FREEZE_STEPS 步内 V2I 速率=0
        """
        candidates = self._get_all_rsus_in_range(vehicle.pos)
        if not candidates:
            vehicle.serving_rsu_id = None
            return None

        # 初始化 handover 状态
        if not hasattr(vehicle, '_ho_stay_steps'):
            vehicle._ho_stay_steps = 0
        if not hasattr(vehicle, '_ho_freeze_remain'):
            vehicle._ho_freeze_remain = 0

        # 计算参考信号质量 Q(i,k) = G(d_{i,k}) （用路径损耗增益即可排序）
        p_ref_w = Cfg.dbm2watt(self.config.TX_POWER_UP_DBM)
        q_scores = {}
        for rsu in candidates:
            dist = max(np.linalg.norm(vehicle.pos - rsu.position), 1.0)
            g = self.channel._path_loss(dist, Cfg.PL_BETA_V2I)
            q_scores[rsu.id] = p_ref_w * g

        curr_rsu_id = getattr(vehicle, "serving_rsu_id", None)
        best_new_id = max(q_scores, key=q_scores.get)
        hyst_linear = 10 ** (self.config.HO_HYST_DB / 10.0)

        if curr_rsu_id is not None and curr_rsu_id in q_scores:
            # 当前 RSU 仍在覆盖内
            vehicle._ho_stay_steps += 1
            q_curr = q_scores[curr_rsu_id]
            q_best = q_scores[best_new_id]

            need_switch = (
                best_new_id != curr_rsu_id
                and q_best > q_curr * hyst_linear
                and vehicle._ho_stay_steps >= self.config.MIN_RSU_STAY_STEPS
            )
            if need_switch:
                vehicle.serving_rsu_id = best_new_id
                vehicle._ho_stay_steps = 0
                vehicle._ho_freeze_remain = self.config.HO_FREEZE_STEPS
                # 记录切换事件
                if not hasattr(self, '_ho_events'):
                    self._ho_events = []
                self._ho_events.append({
                    'step': self.steps, 'veh': vehicle.id,
                    'from': curr_rsu_id, 'to': best_new_id
                })
                return best_new_id
            return curr_rsu_id
        else:
            # 首次连接或当前 RSU 脱离覆盖 => 直接选最佳
            vehicle.serving_rsu_id = best_new_id
            vehicle._ho_stay_steps = 0
            if curr_rsu_id is not None:
                # 被迫切换也冻结
                vehicle._ho_freeze_remain = self.config.HO_FREEZE_STEPS
            return best_new_id

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
        """强制校验RSU目标必须等于serving_rsu_id（仅在旧模式下）"""
        # [RSU选择] 新模式下跳过此断言，因为agent可以选择任意覆盖范围内的RSU
        if getattr(self.config, 'ENABLE_RSU_SELECTION', False):
            return
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
        
        # [P0修复] 通信阶段时间，用于contact time约束
        t_comm_phase = comm_wait_v2i + tx_time

        if speed > 0.1:
            contact_time = max(0.0, (rsu.coverage_range - dist) / speed)
        else:
            contact_time = self._max_rsu_contact_time

        # [R3修复] 不再用预测接触时间进行硬剔除；保留用于特征/软排序

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

    def _get_total_B_remaining(self):
        """
        [P2辅助函数] 计算通信队列中的总剩余数据量（bytes/bits）

        Returns:
            float: 总剩余传输量
        """
        total_B = 0.0
        for queue in self.txq_v2i.values():
            for job in queue:
                total_B += float(getattr(job, "rem_bytes", 0.0))
        for queue in self.txq_v2v.values():
            for job in queue:
                total_B += float(getattr(job, "rem_bytes", 0.0))
        return total_B
    
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
        # [场景修复] 使用可配置的生成范围，确保覆盖所有RSU
        x_min = getattr(self.config, 'VEHICLE_SPAWN_X_MIN', 0.0) * self.config.MAP_SIZE
        x_max = getattr(self.config, 'VEHICLE_SPAWN_X_MAX', 0.8) * self.config.MAP_SIZE
        x_pos = np.random.uniform(x_min, x_max)
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

    def _assign_new_dag_to_vehicle(self, vehicle) -> None:
        """
        为已完成/失败 DAG 的车辆就地分配新任务。

        与 _respawn_vehicle 的区别：
        - 保留车辆当前位置、速度、cpu_freq（车辆不消失，只换任务）
        - 仅清除任务相关队列，不重建车辆对象
        用于 TASK_RESPAWN_ON_COMPLETION=True 的静态车辆场景。
        """
        vid = vehicle.id
        # 清除任务相关队列
        if vid in self.veh_cpu_q:
            self.veh_cpu_q[vid].clear()
        self.txq_v2i.pop(("VEH", vid), None)
        self.txq_v2v.pop(("VEH", vid), None)
        self._last_candidates.pop(vid, None)
        self._last_candidate_set.pop(vid, None)
        self._last_rsu_choice.pop(vid, None)

        # 重置车辆任务状态（保留 pos / vel / cpu_freq）
        if hasattr(vehicle, 'capacity_tracker'):
            vehicle.capacity_tracker.clear()
        vehicle.task_queue_len = 0
        vehicle.last_scheduled_subtask = -1
        vehicle.last_action_step = -1
        vehicle.last_action_target = 'Local'
        vehicle.subtask_reward_buffer = 0.0

        # 分配新 DAG（继承原车辆 CPU 频率，保持异构性）
        old_id = vehicle.task_dag.id if vehicle.task_dag is not None else 0
        adj, prof, data, ddl, extra = self.dag_gen.generate_from_config(veh_f=vehicle.cpu_freq)
        vehicle.task_dag = DAGTask(old_id + 1, adj, prof, data, ddl)
        vehicle.task_dag.deadline_gamma = extra.get("deadline_gamma")
        vehicle.task_dag.critical_path_cycles = extra.get("critical_path_cycles")
        vehicle.task_dag.deadline_base_time = extra.get("deadline_base_time")
        vehicle.task_dag.deadline_slack = extra.get("deadline_slack")
        vehicle.task_dag.start_time = self.time

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

    def _apply_domain_randomization(self):
        """按episode采样环境参数，提升训练鲁棒性（可开关）"""
        def _sample_uniform(min_val, max_val, fallback):
            lo = float(min_val) if min_val is not None else float(fallback)
            hi = float(max_val) if max_val is not None else float(fallback)
            if lo > hi:
                lo, hi = hi, lo
            if abs(hi - lo) < 1e-9:
                return float(lo)
            return float(np.random.uniform(lo, hi))

        params = {}
        v2v_range = _sample_uniform(
            getattr(self.config, "DR_V2V_RANGE_MIN", None),
            getattr(self.config, "DR_V2V_RANGE_MAX", None),
            self.config.V2V_RANGE,
        )
        rsu_range = _sample_uniform(
            getattr(self.config, "DR_RSU_RANGE_MIN", None),
            getattr(self.config, "DR_RSU_RANGE_MAX", None),
            self.config.RSU_RANGE,
        )
        dag_fat = _sample_uniform(
            getattr(self.config, "DR_DAG_FAT_MIN", None),
            getattr(self.config, "DR_DAG_FAT_MAX", None),
            self.config.DAG_FAT,
        )
        dag_density = _sample_uniform(
            getattr(self.config, "DR_DAG_DENSITY_MIN", None),
            getattr(self.config, "DR_DAG_DENSITY_MAX", None),
            self.config.DAG_DENSITY,
        )

        self.config.V2V_RANGE = v2v_range
        self.config.RSU_RANGE = rsu_range
        self.config.DAG_FAT = dag_fat
        self.config.DAG_DENSITY = dag_density
        if hasattr(self, "dag_gen"):
            self.dag_gen.fat = dag_fat
            self.dag_gen.density = dag_density

        # 更新依赖的归一化常数/接触时间
        self._inv_v2v_range = 1.0 / max(self.config.V2V_RANGE, 1e-6)
        self._max_rsu_contact_time = self.config.RSU_RANGE / max(self.config.VEL_MIN, 1e-6)
        self._max_v2v_contact_time = self.config.V2V_RANGE / max(self.config.VEL_MIN, 1e-6)

        params["dr_v2v_range"] = v2v_range
        params["dr_rsu_range"] = rsu_range
        params["dr_dag_fat"] = dag_fat
        params["dr_dag_density"] = dag_density
        self._episode_domain_params = params
    
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
        if hasattr(vehicle.task_dag, 'exec_locations'):
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
            random.seed(seed)  # [P0-2修复] 同时设置random模块

        if hasattr(self, "_reward_stats"):
            self._reward_stats.reset()

        self.vehicles = []
        self.time = 0.0
        self.steps = 0
        self._episode_steps = 0
        self._episode_id = getattr(self, "_episode_id", 0) + 1
        self.episode_count = self._episode_id
        # [P2性能统计] Episode级统计清零
        self._p2_active_time = 0.0
        self._p2_idle_time = 0.0
        self._p2_deltaW_active = 0.0
        self._p2_zero_delta_steps = 0
        self._p2_comm_only_steps = 0
        self._clear_rate_snapshot()
        self._rate_snapshot_prev = None
        self._rate_snapshot_prev_step = -1
        self._rate_prev_cache_v2i = np.zeros(self.config.NUM_VEHICLES, dtype=np.float32)
        self._rate_prev_cache_v2v = np.zeros(self.config.NUM_VEHICLES, dtype=np.float32)
        self._episode_dT_eff_values = []
        self._episode_energy_norm_values = []
        self._episode_t_tx_values = []
        self._episode_task_durations = []
        # 任务重生场景：本 episode 内成功/完成/超时失败 DAG 计数（用于 T_SR、deadline_miss_rate）
        self._episode_task_success_count = 0
        self._episode_task_completion_count = 0
        self._episode_task_deadline_fail_count = 0
        self._episode_rho_selected_values = []
        self._episode_uncertainty_selected_values = []
        self._episode_risk_penalty_values = []
        self._episode_I_total_values = []
        self._episode_I_caused_input_values = []
        self._pbrs_debug_records = []
        self._last_phi_debug = {}
        self._episode_illegal_count = 0
        self._episode_no_task_count = 0
        self._episode_hard_trigger_count = 0
        self._episode_illegal_reasons = {}
        self._episode_no_task_reasons = {}
        # Reset UNIFIED per-episode counters; these metrics should not accumulate across episodes.
        self._unified_nonfinite_count = 0
        self._unified_consistency_mismatch_count = 0
        self._unified_illegal_trigger_count = 0
        self._episode_candidate_stats = {"reachable": [], "dropped": []}
        self._last_candidate_step_stats = {}
        self._episode_not_in_candidate_fallback_cnt = 0
        self._episode_illegal_by_connectivity_cnt = 0
        self._episode_domain_params = {}
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
        self._audit_subtask_est = {}
        self._audit_t_est_records = []
        self._audit_per_decision_rewards = []  # [审计] per-decision奖励分项记录
        self._last_commit_plans = []  # [审计] 保存commit_plans供per-decision审计使用
        self._audit_results_dir = os.environ.get("AUDIT_RESULTS_DIR", self._audit_results_dir)
        self._audit_t_est_path = os.environ.get("AUDIT_T_EST_REAL_PATH", self._audit_t_est_path)
        self._audit_run_id = os.environ.get("RUN_ID", self._audit_run_id)
        # Chain proxy reset
        if getattr(self.config, "CHAIN_ENABLED", False):
            self.chain_proxy = ChainProxySim(self.config, seed=seed)
            self.chain_proxy.reset()
        else:
            self.chain_proxy = None
        self.chain_state_dict = {
            "p50_confirm": 0.0,
            "p95_confirm": 0.0,
            "p_fail": 0.0,
            "mempool_len": 0.0,
            "rho": 0.0,
        }
        self.chain_state_vec = np.zeros(self.config.CHAIN_OBS_DIM, dtype=np.float32)
        self._chain_tx_total = 0
        self._chain_risk_cost_total = 0.0
        self._chain_p95_sum = 0.0
        self._chain_pfail_sum = 0.0
        self._chain_steps = 0
        # Handover 事件记录
        self._ho_events = []
        # Trust / 信誉外生过程
        if not hasattr(self, '_trust_mgr'):
            rng = np.random.default_rng(seed)
            self._trust_mgr = TrustManager(rng=rng)
        if getattr(self.config, 'TRUST_ENABLED', True):
            # 构建所有远端节点 key：RSU + 所有车辆
            remote_keys = []
            for rsu in self.rsus:
                remote_keys.append(('RSU', rsu.id))
            for vid in range(self.config.NUM_VEHICLES):
                remote_keys.append(('VEH', vid))
            self._trust_mgr.reset(remote_keys)
        self._edge_rate_recompute_counts = []
        self._edge_rate_delta_records = []
        self._edge_rate_audit_step = None

        # [Domain Randomization] 只在reset时采样，不影响单步推进
        if getattr(self.config, "DOMAIN_RANDOMIZATION", False):
            self._apply_domain_randomization()
        
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
        self._p2_B_prev = self._get_total_B_remaining()
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
        self._episode_no_task_reasons = {}

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
                        v.illegal_action = False
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
                            v.illegal_action = False
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
                        parent_task_id=None,  # INPUT无parent
                        dag_uid=id(v.task_dag),
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
                            parent_task_id=parent_id,
                            dag_uid=id(v.task_dag),
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
        active_v2i_senders = set()
        for tx_node, queue in self.txq_v2i.items():
            if tx_node[0] == "VEH" and queue:
                active_v2i_senders.add(int(tx_node[1]))
        active_v2v_senders = set()
        for tx_node, queue in self.txq_v2v.items():
            if tx_node[0] == "VEH" and queue:
                active_v2v_senders.add(int(tx_node[1]))
        active_v2i_count = len(active_v2i_senders)
        active_v2v_vehicles = [self._get_vehicle_by_id(vid) for vid in active_v2v_senders]
        audit_edge = bool(getattr(self.config, "EDGE_RATE_RECOMPUTE_AUDIT", False))
        edge_recompute_cnt = 0
        edge_rate_deltas = []
        for q_dict, link_type in ((self.txq_v2i, "V2I"), (self.txq_v2v, "V2V")):
            for tx_node, queue in q_dict.items():
                for job in queue:
                    key = self._rate_key(job.src_node, job.dst_node, link_type)
                    is_edge = getattr(job, "kind", None) == "EDGE"
                    needs_recompute = key not in self._rate_snapshot["links"] or is_edge
                    if needs_recompute:
                        old_rate = self._rate_snapshot["links"].get(key)
                        rate = self._compute_pair_rate(
                            job.src_node,
                            job.dst_node,
                            link_type,
                            getattr(job, "tx_power_dbm", None),
                            active_v2i_count=active_v2i_count,
                            active_v2v_vehicles=active_v2v_vehicles,
                        )
                        rate = max(rate, getattr(self.config, "EPS_RATE", 1e-9))
                        self._rate_snapshot["links"][key] = rate
                        if audit_edge and is_edge:
                            edge_recompute_cnt += 1
                            if old_rate is not None:
                                delta = abs(rate - old_rate) / max(old_rate, getattr(self.config, "EPS_RATE", 1e-9))
                                edge_rate_deltas.append(float(delta))
                            else:
                                edge_rate_deltas.append(0.0)
        if audit_edge:
            if not hasattr(self, "_edge_rate_recompute_counts"):
                self._edge_rate_recompute_counts = []
            if not hasattr(self, "_edge_rate_delta_records"):
                self._edge_rate_delta_records = []
            self._edge_rate_recompute_counts.append(int(edge_recompute_cnt))
            if edge_rate_deltas:
                self._edge_rate_delta_records.extend(edge_rate_deltas)
                edge_delta_mean = float(np.mean(edge_rate_deltas))
                edge_delta_p95 = float(np.percentile(edge_rate_deltas, 95))
            else:
                edge_delta_mean = 0.0
                edge_delta_p95 = 0.0
            self._edge_rate_audit_step = {
                "edge_rate_recompute_cnt": int(edge_recompute_cnt),
                "edge_rate_delta_mean": edge_delta_mean,
                "edge_rate_delta_p95": edge_delta_p95,
            }
        else:
            self._edge_rate_audit_step = None
        comm_result = self._comm_service.step(
            self.txq_v2i,
            self.txq_v2v,
            self.config.DT,
            self.time,
            rate_fn=lambda job, tx_node: self._compute_job_rate(job, tx_node),
        )

        if self._rate_prev_cache_v2i is None or len(self._rate_prev_cache_v2i) != self.config.NUM_VEHICLES:
            self._rate_prev_cache_v2i = np.zeros(self.config.NUM_VEHICLES, dtype=np.float32)
        else:
            self._rate_prev_cache_v2i.fill(0.0)
        if self._rate_prev_cache_v2v is None or len(self._rate_prev_cache_v2v) != self.config.NUM_VEHICLES:
            self._rate_prev_cache_v2v = np.zeros(self.config.NUM_VEHICLES, dtype=np.float32)
        else:
            self._rate_prev_cache_v2v.fill(0.0)
        v2i_bytes = {}
        v2i_time = {}
        v2v_bytes = {}
        v2v_time = {}

        def _accumulate(job, is_v2i):
            if job is None:
                return
            if job.step_time_used is None or job.step_time_used <= 0:
                return
            if job.step_bytes_sent is None or job.step_bytes_sent <= 0:
                return
            if job.src_node is None or job.src_node[0] != "VEH":
                return
            vid = int(job.src_node[1])
            if is_v2i:
                v2i_bytes[vid] = v2i_bytes.get(vid, 0.0) + float(job.step_bytes_sent)
                v2i_time[vid] = v2i_time.get(vid, 0.0) + float(job.step_time_used)
            else:
                v2v_bytes[vid] = v2v_bytes.get(vid, 0.0) + float(job.step_bytes_sent)
                v2v_time[vid] = v2v_time.get(vid, 0.0) + float(job.step_time_used)

        for q_dict, is_v2i in ((self.txq_v2i, True), (self.txq_v2v, False)):
            for queue in q_dict.values():
                for job in queue:
                    _accumulate(job, is_v2i)
        for job in comm_result.completed_jobs:
            link_type = getattr(job, "link_type", None)
            if link_type == "V2I":
                _accumulate(job, True)
            elif link_type == "V2V":
                _accumulate(job, False)

        for vid, total_time in v2i_time.items():
            if total_time <= 0:
                continue
            if 0 <= vid < len(self._rate_prev_cache_v2i):
                self._rate_prev_cache_v2i[vid] = float(v2i_bytes.get(vid, 0.0) / total_time)
        for vid, total_time in v2v_time.items():
            if total_time <= 0:
                continue
            if 0 <= vid < len(self._rate_prev_cache_v2v):
                self._rate_prev_cache_v2v[vid] = float(v2v_bytes.get(vid, 0.0) / total_time)
        
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
            if getattr(job, "dag_uid", None) is not None and getattr(v, "task_dag", None) is not None:
                if job.dag_uid != id(v.task_dag):
                    return
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
        trust_enabled = getattr(self.config, 'TRUST_ENABLED', True) and hasattr(self, '_trust_mgr')
        for job in cpu_result.completed_jobs:
            v = self._get_vehicle_by_id(job.owner_vehicle_id)
            if v is not None:
                dag = v.task_dag
                if dag.is_finished or dag.is_failed:
                    continue
                if getattr(job, "dag_uid", None) is not None and job.dag_uid != id(dag):
                    continue
                if job.subtask_id is None or int(job.subtask_id) < 0 or int(job.subtask_id) >= int(getattr(dag, "num_subtasks", 0)):
                    # 动态车辆重生/队列异步完成时的陈旧作业保护，避免索引当前新DAG越界。
                    continue

                # 信誉采样：远端子任务完成前检查可靠性
                if trust_enabled:
                    exec_loc = dag.exec_locations[job.subtask_id] if job.subtask_id < len(dag.exec_locations) else None
                    is_remote = (exec_loc is not None and exec_loc != 'Local')
                    if is_remote:
                        if isinstance(exec_loc, tuple) and exec_loc[0] == 'RSU':
                            node_key = ('RSU', exec_loc[1])
                        elif isinstance(exec_loc, int):
                            node_key = ('VEH', exec_loc)
                        else:
                            node_key = None
                        if node_key is not None:
                            success = self._trust_mgr.sample_outcome(node_key)
                            delay_override = None
                            if getattr(self.config, "CHAIN_ENABLED", False) and getattr(self.config, "CHAIN_TRUST_DELAY_COUPLED", False):
                                try:
                                    p95 = float(self.chain_state_dict.get("p95_confirm", 0.0))
                                except Exception:
                                    p95 = 0.0
                                base = int(getattr(self.config, "CHAIN_TRUST_DELAY_BASE_STEPS", 0))
                                dmin = int(getattr(self.config, "CHAIN_TRUST_DELAY_MIN_STEPS", 0))
                                dmax = int(getattr(self.config, "CHAIN_TRUST_DELAY_MAX_STEPS", 50))
                                dt = float(getattr(self.config, "DT", 0.1))
                                est = int(round(p95 / max(dt, 1e-9))) + base
                                if est < dmin:
                                    est = dmin
                                if est > dmax:
                                    est = dmax
                                delay_override = est
                            self._trust_mgr.submit_evidence(self.steps, node_key, success, delay_steps=delay_override)
                            if not success:
                                # 失败：子任务回到 READY 重试（不返还耗时/能耗）
                                dag.status[job.subtask_id] = 1  # READY
                                dag.exec_locations[job.subtask_id] = None
                                if hasattr(dag, 'task_locations'):
                                    dag.task_locations[job.subtask_id] = None
                                self._trust_mgr.retry_events.append(
                                    (self.steps, v.id, node_key, job.subtask_id)
                                )
                                continue  # 跳过 on_compute_done

                self._audit_on_compute_done(job, self.time)
                self._dag_handler.on_compute_done(
                    job, v, self.time, 
                    veh_cpu_q=self.veh_cpu_q,
                    rsu_cpu_q=self.rsu_cpu_q,
                    rsus=self.rsus
                )
        # 处理到期的信誉延迟证据
        if trust_enabled:
            self._trust_mgr.process_pending(self.steps)
    
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
        prev_completed_counts = {
            v.id: int(np.sum(v.task_dag.status == 3)) for v in self.vehicles
        }
        prev_energy_input_cost = {
            v.id: float(self.E_tx_input_cost.get(v.id, 0.0))
            for v in self.vehicles
        }

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

        scheme = getattr(self.config, "REWARD_SCHEME", "LEGACY_CFT")
        use_pbrs = scheme != "LEGACY_CFT"
        unified_enable_pbrs = bool(getattr(self.config, "ENABLE_PBRS", False)) and (
            float(getattr(self.config, "PBRS_BETA", 0.0)) > 0.0
        )
        pbrs_phi_mode = getattr(self.config, "PBRS_PHI_MODE", "STATE_ONLY")
        phi_prev_cache = {}
        phi_prev_debug_cache = {}
        if use_pbrs and scheme != "UNIFIED" and pbrs_phi_mode == "STATE_ONLY":
            if getattr(self.config, "PBRS_PHI_ACTION_INVARIANT_CHECK", False):
                self._assert_phi_action_invariant(scheme)
            phi_prev_cache, phi_prev_debug_cache = self._compute_phi_state_only_batch(scheme)

        # =====================================================================
        # [新FIFO队列系统] Phase 1-5 推进
        # =====================================================================
        
        # 解析动作并生成计划
        plans = self._plan_actions_snapshot(actions)
        num_agents = len(self.vehicles)
        active_agent_mask = [0] * num_agents
        decision_step_mask = [False] * num_agents
        no_task_step_mask = [False] * num_agents
        step_no_task_count = 0
        step_illegal_action_count = 0
        for plan in plans:
            idx = plan.get("index")
            if idx is None or idx < 0 or idx >= num_agents:
                continue
            has_subtask = plan.get("subtask_idx") is not None
            has_valid_action = plan.get("illegal_reason") is None
            is_decision = bool(has_subtask)
            is_active = bool(has_subtask and has_valid_action)
            decision_step_mask[idx] = bool(is_decision)
            active_agent_mask[idx] = 1 if is_active else 0
            no_task_step_mask[idx] = bool(plan.get("is_no_task_available", False))
            reason = plan.get("illegal_reason")
            if plan.get("is_no_task_available", False):
                step_no_task_count += 1
                if reason:
                    self._episode_no_task_reasons[reason] = self._episode_no_task_reasons.get(reason, 0) + 1
            elif plan.get("is_illegal_action", False):
                step_illegal_action_count += 1
                if reason:
                    self._episode_illegal_reasons[reason] = self._episode_illegal_reasons.get(reason, 0) + 1
        self._episode_no_task_count += int(step_no_task_count)
        self._episode_illegal_count += int(step_illegal_action_count)
        commit_plans = [p for p in plans if p["subtask_idx"] is not None]
        not_in_candidate_reasons = {
            "masked_target",
            "no_candidate_cache",
            "id_mapping_fail",
            "idx_out_of_range",
        }
        connectivity_reasons = {
            "rsu_unavailable",
            "rsu_out_of_coverage",
            "no_candidate_cache",
        }
        step_not_in_candidate_fallback_cnt = 0
        step_illegal_by_connectivity_cnt = 0
        for plan in plans:
            reason = plan.get("illegal_reason")
            if reason in not_in_candidate_reasons:
                step_not_in_candidate_fallback_cnt += 1
            if reason in connectivity_reasons:
                step_illegal_by_connectivity_cnt += 1
        self._episode_not_in_candidate_fallback_cnt += step_not_in_candidate_fallback_cnt
        self._episode_illegal_by_connectivity_cnt += step_illegal_by_connectivity_cnt
        # [Chain] 交易产生规则（基于最终planned_target与非法回退）
        tx_flags = {}
        tx_arrivals_step = 0
        for plan in plans:
            vid = plan.get("vehicle_id")
            if vid is None:
                continue
            tx_i = 0
            if plan.get("subtask_idx") is None or plan.get("illegal_reason") is not None:
                tx_i = 0
            else:
                tgt = plan.get("planned_target")
                if tgt == 'Local' or tgt is None:
                    tx_i = 0
                elif isinstance(tgt, tuple) and tgt[0] == "RSU":
                    tx_i = 1 if getattr(self.config, "CHAIN_CHARGE_RSU", True) else 0
                elif isinstance(tgt, int):
                    tx_i = 1
            tx_flags[int(vid)] = tx_i
            tx_arrivals_step += tx_i
        batch_k = int(getattr(self.config, "CHAIN_BATCH_K", 1))
        if batch_k <= 0:
            batch_k = 1
        tx_arrivals_step = int(tx_arrivals_step // batch_k)
        # [P0修复] 消除commit顺序效应
        # 修复前：按vehicle_id排序，低ID车辆优先入队，高ID更易因队列满失败
        # 修复后：使用可复现的随机打乱，消除ID与入队顺序的相关性
        import random
        shuffle_rng = random.Random(self.episode_count * 10000 + self.steps)
        shuffle_rng.shuffle(commit_plans)
        self._last_commit_plans = commit_plans  # [审计] 保存供per-decision审计使用
        
        # 更新功率（在Phase1之前）
        for plan in commit_plans:
            v = plan["vehicle"]
            if plan["power_dbm"] is not None:
                v.tx_power_dbm = plan["power_dbm"]
                step_power_ratio[v.id] = plan["power_ratio"] if plan["power_ratio"] is not None else step_power_ratio.get(v.id, 0.0)
            
            v.illegal_action = bool(plan.get("is_illegal_action", False))
            v.illegal_reason = plan["illegal_reason"] if v.illegal_action else None

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
            power_ratio_for_est = None
            if subtask_idx is not None:
                freq_self = max(getattr(v, "cpu_freq", self.config.MIN_VEHICLE_CPU_FREQ), 1e-9)
                t_local = self._get_veh_queue_wait_time(v.id, freq_self) + cycles / freq_self
                power_ratio_for_est = plan.get("power_ratio") if plan else None
                if power_ratio_for_est is None:
                    power_ratio_for_est = 0.0
                t_actual, t_tx = self._estimate_t_actual(
                    v,
                    subtask_idx,
                    target,
                    cycles,
                    power_ratio_for_est
                )
            if scheme == "UNIFIED":
                if unified_enable_pbrs:
                    # UNIFIED PBRS: phi = -(eps + LB/Td)^q
                    LB_prev = self._compute_lb_snapshot(v)
                    Td_u = dag.deadline if dag.deadline > 0 else (self.config.MAX_STEPS * self.config.DT)
                    phi_prev = compute_phi_lb(LB_prev, Td_u)
                    phi_debug = {"lb": LB_prev, "Td": Td_u}
                else:
                    phi_prev = 0.0
                    phi_debug = {}
            elif use_pbrs and pbrs_phi_mode == "STATE_ONLY":
                if scheme == "PBRS_KP_V2":
                    phi_prev = phi_prev_cache.get(v.id, 0.0)
                    phi_debug = phi_prev_debug_cache.get(v.id, {})
                else:
                    phi_prev = phi_prev_cache.get(v.id, 0.0)
                    phi_debug = {}
            elif scheme == "PBRS_KP_V2":
                phi_prev, phi_debug = self._compute_phi_value_v2(dag, vehicle=v)
            else:
                phi_prev = self._compute_phi_value(dag, vehicle=v)
                phi_debug = {}
            power_ratio_raw = plan.get("power_ratio") if plan else None
            if power_ratio_raw is None:
                power_ratio_raw = 0.0
            power_ratio = float(np.clip(power_ratio_raw, 0.0, 1.0))
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
                "illegal_action": bool((plan and plan.get("is_illegal_action", False)) or getattr(v, "illegal_action", False)),
                "no_task_available": bool(plan.get("is_no_task_available", False)) if plan else False,
                "illegal": bool((plan and plan.get("is_illegal_action", False)) or getattr(v, "illegal_action", False)),
                "illegal_reason": plan.get("illegal_reason") if plan else None,  # [Stage 1] 传播原因
                "power_ratio": power_ratio,
            }
            if subtask_idx is not None:
                key = (int(v.id), int(subtask_idx))
                if key not in self._audit_subtask_est:
                    self._audit_subtask_est[key] = {
                        "episode": int(getattr(self, "episode_count", 0)),
                        "vehicle_id": int(v.id),
                        "subtask_id": int(subtask_idx),
                        "decision_time": float(snapshot_time),
                        "t_est": float(t_actual) if np.isfinite(t_actual) else 0.0,
                        "action_type": self._audit_action_type(target),
                    }
            if not np.isfinite(reward_cache[v.id]["phi_prev"]):
                reward_cache[v.id]["phi_prev"] = 0.0
            t_local = float(np.nan_to_num(t_local, nan=0.0, posinf=0.0, neginf=0.0))
            t_actual = float(np.nan_to_num(t_actual, nan=0.0, posinf=0.0, neginf=0.0))
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
        # [Chain] 结算风险层状态更新（不影响物理推进）
        # 推荐时点：Phase1之后、Phase3之前
        # =====================================================================
        if getattr(self.config, "CHAIN_ENABLED", False) and self.chain_proxy is not None:
            self.chain_state_dict = self.chain_proxy.step(tx_arrivals_step)
        else:
            self.chain_state_dict = {
                "p50_confirm": 0.0,
                "p95_confirm": 0.0,
                "p_fail": 0.0,
                "mempool_len": 0.0,
                "rho": 0.0,
            }
        chain_vals = [
            self.chain_state_dict.get("p50_confirm", 0.0),
            self.chain_state_dict.get("p95_confirm", 0.0),
            self.chain_state_dict.get("p_fail", 0.0),
            self.chain_state_dict.get("mempool_len", 0.0),
            self.chain_state_dict.get("rho", 0.0),
        ]
        self.chain_state_vec = np.zeros(self.config.CHAIN_OBS_DIM, dtype=np.float32)
        for i in range(min(len(chain_vals), self.chain_state_vec.shape[0])):
            self.chain_state_vec[i] = float(chain_vals[i])
        self._chain_tx_total += int(tx_arrivals_step)
        self._chain_p95_sum += float(self.chain_state_dict.get("p95_confirm", 0.0))
        self._chain_pfail_sum += float(self.chain_state_dict.get("p_fail", 0.0))
        self._chain_steps += 1
        
        # =====================================================================
        # [P1修复] Post-Commit Snapshot: 重新估计t_actual
        # 目的: 让奖励依赖"联合动作后"的队列状态，而非个人快照
        # 修复前: t_actual在决策前估计，不包含并发入队效应
        # 修复后: t_actual在决策提交后重新估计，包含所有并发入队
        # =====================================================================
        if getattr(self.config, 'USE_POST_COMMIT_SNAPSHOT', True):
            for v in self.vehicles:
                ctx = reward_cache.get(v.id)
                if ctx is None or ctx.get("subtask") is None:
                    continue
                subtask_idx = ctx["subtask"]
                target = ctx["target"]
                cycles = ctx["cycles"]
                power_ratio = ctx.get("power_ratio", 0.0)
                if power_ratio is None:
                    power_ratio = 0.0
                
                # 重新估计t_actual（此时队列已包含所有并发入队）
                t_actual_post, t_tx_post = self._estimate_t_actual(
                    v, subtask_idx, target, cycles, power_ratio
                )
                
                # 更新reward_cache
                reward_cache[v.id]["t_actual"] = t_actual_post
                reward_cache[v.id]["t_tx"] = t_tx_post
                reward_cache[v.id]["post_commit_updated"] = True

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

        step_completed_counts = {}
        step_energy_cost = {}
        step_energy_input_cost = {}
        for v in self.vehicles:
            prev_completed = prev_completed_counts.get(v.id, 0)
            curr_completed = int(np.sum(v.task_dag.status == 3))
            step_completed_counts[v.id] = max(curr_completed - prev_completed, 0)
            prev_energy = prev_energy_input_cost.get(v.id, 0.0)
            curr_energy = float(self.E_tx_input_cost.get(v.id, 0.0))
            step_energy_cost[v.id] = max(curr_energy - prev_energy, 0.0)
            step_energy_input_cost[v.id] = step_energy_cost[v.id]

        phi_next_cache = {}
        phi_next_debug_cache = {}
        if use_pbrs and scheme != "UNIFIED" and pbrs_phi_mode == "STATE_ONLY":
            phi_next_cache, phi_next_debug_cache = self._compute_phi_state_only_batch(scheme)

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
        risk_cost_sum = 0.0
        
        if scheme == "UNIFIED":
            # ============ 统一奖励方案 ============
            rewards = []
            step_unified_nonfinite_count = 0
            step_unified_consistency_mismatch_count = 0
            step_unified_illegal_trigger_count = 0
            v2v_stats = getattr(self.channel, 'last_v2v_stats', {})
            v2i_stats = getattr(self.channel, 'last_v2i_stats', {})
            v2v_i_caused_input_dict = v2v_stats.get('i_caused_input', {})
            v2i_i_caused_input_dict = v2i_stats.get('i_caused_input', {})
            v2v_i_total_dict = v2v_stats.get('i_total', {})
            v2i_i_total_dict = v2i_stats.get('i_total', {})

            energy_vals = [float(max(step_energy_input_cost.get(v.id, 0.0), 0.0)) for v in self.vehicles]
            interf_vals = [
                float(max(
                    float(v2v_i_caused_input_dict.get(v.id, 0.0)) + float(v2i_i_caused_input_dict.get(v.id, 0.0)),
                    0.0,
                ))
                for v in self.vehicles
            ]
            energy_ref = self._update_reward_ref_ema(
                "energy",
                float(np.percentile(energy_vals, 95)) if energy_vals else 0.0,
            )
            interf_ref = self._update_reward_ref_ema(
                "interf",
                float(np.percentile(interf_vals, 95)) if interf_vals else 0.0,
            )

            trust_enabled = bool(getattr(self.config, "TRUST_ENABLED", False))
            trust_snapshot = {}
            risk_vals = []
            for v in self.vehicles:
                ctx = reward_cache.get(v.id, {})
                target = ctx.get("target", "Local")
                is_remote = not (target is None or target == "Local")
                rho_target = 1.0
                uncertainty_target = 0.0
                if is_remote and trust_enabled and hasattr(self, "_trust_mgr"):
                    node_key = None
                    if isinstance(target, tuple) and len(target) > 1 and target[0] == "RSU":
                        node_key = ("RSU", int(target[1]))
                    elif isinstance(target, int) and not isinstance(target, bool):
                        node_key = ("VEH", int(target))
                    if node_key is not None:
                        rho_target, uncertainty_target = self._trust_mgr.get_reputation(node_key)
                trust_snapshot[v.id] = (is_remote, float(rho_target), float(uncertainty_target), target)
                if is_remote:
                    risk_vals.append(float(max(1.0 - float(rho_target), 0.0)))
            risk_ref = self._update_reward_ref_ema(
                "risk",
                float(np.percentile(risk_vals, 95)) if risk_vals else 0.0,
            )
            dt_total = float(self.config.DT)
            dt_idle = float(max(getattr(self.config, "DT_IDLE", 0.0), 0.0))
            dt_used = float(max(dt_total, dt_idle))
            progress_mode = str(getattr(self.config, "PROGRESS_REWARD_MODE", "DELTA_CFT_ABS")).upper()
            progress_ref = float(max(getattr(self.config, "PROGRESS_REF_SECONDS", max(dt_total, 1e-3)), 1e-6))
            w_progress = float(getattr(self.config, "W_PROGRESS", 0.0))
            for i, v in enumerate(self.vehicles):
                dag = v.task_dag
                ctx = reward_cache.get(v.id, {})
                Td = dag.deadline if dag.deadline > 0 else (self.config.MAX_STEPS * self.config.DT)

                # 每步分量
                illegal = bool(ctx.get("illegal_action", ctx.get("illegal", False)))
                if illegal:
                    step_unified_illegal_trigger_count += 1
                energy_step = step_energy_input_cost.get(v.id, 0.0)
                # 可控干扰口径：仅统计 INPUT_TX 造成的干扰（EDGE 仅作为背景负载）
                i_caused_input = float(v2v_i_caused_input_dict.get(v.id, 0.0)) + float(
                    v2i_i_caused_input_dict.get(v.id, 0.0)
                )
                # 全量干扰口径：用于场景统计（包含 INPUT/EDGE）
                i_total_all = float(v2v_i_total_dict.get(v.id, 0.0)) + float(
                    v2i_i_total_dict.get(v.id, 0.0)
                )
                is_remote, rho_target, uncertainty_target, target = trust_snapshot.get(
                    v.id,
                    (False, 1.0, 0.0, ctx.get("target", "Local")),
                )

                # [Chain] 结算风险层：只做指标统计（不改UNIFIED reward结构）
                # 口径：每次产生交易(tx_flag=1)时，累计 deposit * (alpha_D*p95 + alpha_F*p_fail)
                if getattr(self.config, "CHAIN_ENABLED", False):
                    alpha_d = float(getattr(self.config, "CHAIN_RISK_WEIGHT_DEPOSIT", 0.0))
                    alpha_f = float(getattr(self.config, "CHAIN_RISK_WEIGHT_FAIL", 0.0))
                    tx_flag = int(tx_flags.get(v.id, 0))
                    if tx_flag and (alpha_d > 0.0 or alpha_f > 0.0):
                        proxy_size = float(ctx.get("cycles", 0.0))
                        deposit = float(self.config.CHAIN_DEPOSIT_BASE) + float(self.config.CHAIN_DEPOSIT_SCALE) * proxy_size
                        p95 = float(self.chain_state_dict.get("p95_confirm", 0.0))
                        p_fail = float(self.chain_state_dict.get("p_fail", 0.0))
                        lock_cost = alpha_d * deposit * p95
                        fail_cost = alpha_f * deposit * p_fail
                        risk_cost_sum += (lock_cost + fail_cost)

                cft_prev_v = vehicle_cfts_prev[i] if i < len(vehicle_cfts_prev) else np.nan
                cft_curr_v = vehicle_cfts[i] if i < len(vehicle_cfts) else np.nan
                prev_rem_v = 0.0
                curr_rem_v = 0.0
                delta_cft_rem_v = 0.0
                delta_cft_abs_true = 0.0  # 真实绝对CFT差（恒等于cft_prev-cft_curr，与mode无关）
                delta_prog = 0.0
                if np.isfinite(cft_prev_v) and np.isfinite(cft_curr_v):
                    prev_rem_v = max(float(cft_prev_v) - t_prev, 0.0)
                    curr_rem_v = max(float(cft_curr_v) - t_curr, 0.0)
                    delta_cft_rem_v = prev_rem_v - curr_rem_v
                    delta_cft_abs_true = float(cft_prev_v) - float(cft_curr_v)
                    if progress_mode == "DELTA_SLACK":
                        slack_prev = float(Td) - prev_rem_v
                        slack_curr = float(Td) - curr_rem_v
                        delta_prog = slack_curr - slack_prev
                    else:
                        delta_prog = delta_cft_abs_true
                r_prog = w_progress * float(np.clip(delta_prog / progress_ref, -1.0, 1.0))
                if not np.isfinite(r_prog):
                    r_prog = 0.0

                step_dt = dt_used
                r_step, step_info = compute_unified_step_reward(
                    dt=step_dt, Td=Td,
                    E_tx=energy_step, I_caused=i_caused_input,
                    illegal=illegal,
                    is_remote=is_remote, rho_target=rho_target,
                    E_ref=energy_ref, I_ref=interf_ref, risk_ref=risk_ref,
                    r_prog=r_prog,
                )

                # 终局
                r_term = 0.0
                finished_prev = ctx.get("finished_prev", False)
                failed_prev = ctx.get("failed_prev", False)
                if (not finished_prev) and dag.is_finished and not dag.is_failed:
                    Tf = dag.completion_time if dag.completion_time is not None else (self.time - dag.start_time)
                    r_term, _ = compute_unified_terminal_reward(True, Tf, Td)
                elif (not failed_prev) and dag.is_failed:
                    Tf = self.time - dag.start_time
                    r_term, _ = compute_unified_terminal_reward(False, Tf, Td)

                # PBRS
                r_pbrs = 0.0
                use_pbrs_u = unified_enable_pbrs
                if use_pbrs_u:
                    phi_prev = ctx.get("phi_prev", 0.0)
                    # 计算 phi_next（基于下界快照）
                    LB_next = self._compute_lb_snapshot(v)
                    phi_next = compute_phi_lb(LB_next, Td)
                    terminated = dag.is_finished or dag.is_failed
                    r_pbrs = compute_unified_pbrs(phi_prev, phi_next, terminated=terminated)

                r_total_raw = float(r_step + r_term + r_pbrs)
                if use_pbrs_u:
                    recomposed = float(r_step) + float(r_term) + float(r_pbrs)
                    if (not np.isfinite(recomposed)) or (not np.isfinite(r_total_raw)) or (abs(recomposed - r_total_raw) > 1e-6):
                        step_unified_consistency_mismatch_count += 1
                r_total = self._clip_reward(r_total_raw)
                for rv in (r_step, r_term, r_pbrs, r_total):
                    if not np.isfinite(rv):
                        step_unified_nonfinite_count += 1
                rewards.append(r_total)

                # 累积统计
                if hasattr(self, "_reward_stats"):
                    self._reward_stats.add_metric("reward", r_total)
                    self._reward_stats.add_metric("r_step", r_step)
                    self._reward_stats.add_metric("r_term", r_term)
                    self._reward_stats.add_metric("r_pbrs", r_pbrs)
                    # UNIFIED step components (mean + abs for dominance checks)
                    for k in ("r_time", "r_prog", "r_energy", "r_interf", "r_risk", "r_illegal"):
                        val = float(step_info.get(k, 0.0))
                        self._reward_stats.add_metric(k, val)
                        self._reward_stats.add_metric(f"{k}_abs", abs(val))
                    self._reward_stats.add_metric("r_term_abs", abs(float(r_term)))
                    self._reward_stats.add_metric("r_pbrs_abs", abs(float(r_pbrs)))
                    self._reward_stats.add_metric("r_step_abs", abs(float(r_step)))
                    self._reward_stats.add_metric("dt_used", step_dt)
                    self._reward_stats.add_metric("implied_dt", step_info.get("dt_used", step_dt))
                    self._reward_stats.add_metric("delta_cft", delta_cft_rem_v)
                    self._reward_stats.add_metric("delta_cft_abs", delta_cft_abs_true)  # 真实绝对差，恒≈-DT（队列不变时）
                    self._reward_stats.add_metric("delta_cft_rem", delta_cft_rem_v)     # 剩余差，恒≈+DT
                    self._reward_stats.add_metric("delta_cft_prog", delta_prog)         # 用于r_prog的mode相关信号
                    self._reward_stats.add_metric("cft_prev_rem", prev_rem_v)
                    self._reward_stats.add_metric("cft_curr_rem", curr_rem_v)
                    self._reward_stats.add_metric("dT_eff", delta_cft_rem_v - step_dt)
                    self._reward_stats.add_metric("energy_norm", step_info.get("energy_norm", 0.0))
                    # Interference externality (per decision-step)
                    self._reward_stats.add_metric("I_total", float(i_total_all))
                    self._reward_stats.add_metric("I_total_abs", abs(float(i_total_all)))
                # Reputation oracle stats (only meaningful for remote decisions)
                    if is_remote:
                        self._reward_stats.add_metric("rho_selected", float(rho_target))
                        self._reward_stats.add_metric("uncertainty_selected", float(max(uncertainty_target, 0.0)))
                        risk_penalty = -float(step_info.get("r_risk", 0.0))
                        self._reward_stats.add_metric("risk_penalty", float(max(risk_penalty, 0.0)))
                self._episode_energy_norm_values.append(step_info.get("energy_norm", 0.0))
                self._episode_t_tx_values.append(float(ctx.get("t_tx", 0.0)))
                self._episode_I_total_values.append(float(i_total_all))
                self._episode_I_caused_input_values.append(float(i_caused_input))
                if is_remote:
                    self._episode_rho_selected_values.append(float(rho_target))
                    self._episode_uncertainty_selected_values.append(float(max(uncertainty_target, 0.0)))
                    self._episode_risk_penalty_values.append(float(max(-float(step_info.get("r_risk", 0.0)), 0.0)))
            self._unified_nonfinite_count += int(step_unified_nonfinite_count)
            self._unified_consistency_mismatch_count += int(step_unified_consistency_mismatch_count)
            self._unified_illegal_trigger_count += int(step_unified_illegal_trigger_count)
            self._episode_hard_trigger_count += int(step_unified_illegal_trigger_count)
            if hasattr(self, "_reward_stats"):
                if step_unified_nonfinite_count > 0:
                    self._reward_stats.add_counter("unified_nonfinite", int(step_unified_nonfinite_count))
                if step_unified_consistency_mismatch_count > 0:
                    self._reward_stats.add_counter("unified_consistency_mismatch", int(step_unified_consistency_mismatch_count))
                if step_unified_illegal_trigger_count > 0:
                    self._reward_stats.add_counter("illegal_trigger_unified", int(step_unified_illegal_trigger_count))

        elif scheme == "LEGACY_CFT":
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
                ctx_prev = reward_cache.get(v.id, {})
                r_progress = float(step_completed_counts.get(v.id, 0)) * self.config.SUBTASK_SUCCESS_BONUS
                r_term = 0.0
                if (not ctx_prev.get("finished_prev")) and dag.is_finished:
                    r_term += self.config.TERMINAL_BONUS_SUCC
                if (not ctx_prev.get("failed_prev")) and dag.is_failed:
                    r_term += self.config.TERMINAL_PENALTY_FAIL
                r_timeout = 0.0
                if dag.deadline > 0 and dag.is_failed and dag.fail_reason == 'deadline':
                    elapsed = self.time - dag.start_time
                    overtime_ratio = max((elapsed - dag.deadline) / dag.deadline, 0.0)
                    r_timeout = -self.config.TIMEOUT_PENALTY_WEIGHT * np.tanh(
                        self.config.TIMEOUT_STEEPNESS * overtime_ratio
                    )
                energy_step = step_energy_cost.get(v.id, 0.0)
                p_circuit = float(getattr(self.config, "P_CIRCUIT_WATT", 0.0))
                e_max = max((p_max_watt + p_circuit) * float(self.config.DT), 1e-12)
                energy_norm_real = float(np.clip(energy_step / e_max, 0.0, 1.0))
                r_energy = -self.config.DELTA_CFT_ENERGY_WEIGHT * energy_norm_real

                reward_parts = None
                r_shape = 0.0
                if getattr(v, 'illegal_action', False):
                    # 非法动作吃满统一惩罚，再做等比例缩放（不再硬裁剪）
                    r = -float(getattr(self.config, "W_ILLEGAL", abs(self.config.REWARD_MIN)))
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
                    shape_reward, reward_parts = compute_absolute_reward(
                        dT_rem_v, t_tx, power_ratio, self.config.DT, p_max_watt,
                        self.config.REWARD_MIN, self.config.REWARD_MAX, hard_triggered=hard_triggered, illegal_action=False
                    )
                    if hard_triggered:
                        self._episode_hard_trigger_count += 1
                        energy_norm_real = 0.0
                        r_energy = 0.0
                    reward_parts["energy_norm"] = energy_norm_real
                    r_shape = shape_reward
                    base_reward = r_progress + r_term + r_timeout + r_energy
                    r = self._clip_reward(base_reward + r_shape)
                if hasattr(v, 'subtask_reward_buffer'):
                    v.subtask_reward_buffer = 0.0

                self._episode_dT_eff_values.append(dT_eff)
                self._episode_energy_norm_values.append(reward_parts.get("energy_norm", 0.0) if reward_parts else 0.0)
                self._episode_t_tx_values.append(step_tx_time.get(v.id, 0.0))

                # [Chain] 风险成本（仅影响奖励，不影响物理推进）
                r_chain = 0.0
                if getattr(self.config, "CHAIN_ENABLED", False):
                    alpha_d = float(getattr(self.config, "CHAIN_RISK_WEIGHT_DEPOSIT", 0.0))
                    alpha_f = float(getattr(self.config, "CHAIN_RISK_WEIGHT_FAIL", 0.0))
                    tx_flag = int(tx_flags.get(v.id, 0))
                    if tx_flag and (alpha_d > 0.0 or alpha_f > 0.0):
                        proxy_size = float(task_comp if task_comp is not None else 0.0)
                        deposit = float(self.config.CHAIN_DEPOSIT_BASE) + float(self.config.CHAIN_DEPOSIT_SCALE) * proxy_size
                        p95 = float(self.chain_state_dict.get("p95_confirm", 0.0))
                        p_fail = float(self.chain_state_dict.get("p_fail", 0.0))
                        lock_cost = alpha_d * deposit * p95
                        fail_cost = alpha_f * deposit * p_fail
                        r_chain = -(lock_cost + fail_cost)
                        risk_cost_sum += (lock_cost + fail_cost)
                r = float(r + r_chain)
                rewards.append(r)
                if hasattr(self, "_reward_stats") and reward_parts is not None:
                    self._reward_stats.add_metric("reward", r)
                    self._reward_stats.add_metric("dT_clipped", reward_parts.get("dT", 0.0))
                    self._reward_stats.add_metric("energy_norm", reward_parts.get("energy_norm", 0.0))
        else:
            assert self._rate_snapshot is None or self._rate_snapshot.get("step", -1) == self.steps
            phi_list = []
            rem_list = []
            pbrs_step_flags = [] if getattr(self.config, "DEBUG_PBRS_AUDIT", False) else None
            for v in self.vehicles:
                dag = v.task_dag
                if scheme == "PBRS_KP_V2":
                    default_phi_prev, _ = self._compute_phi_value_v2(dag, vehicle=v)
                else:
                    default_phi_prev = self._compute_phi_value(dag, vehicle=v)
                if use_pbrs and pbrs_phi_mode == "STATE_ONLY":
                    default_phi_prev = phi_prev_cache.get(v.id, default_phi_prev)
                ctx = reward_cache.get(v.id, {
                    "phi_prev": default_phi_prev,
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
                if use_pbrs and pbrs_phi_mode == "STATE_ONLY":
                    if scheme == "PBRS_KP_V2":
                        phi_next = phi_next_cache.get(v.id, 0.0)
                        phi_debug = phi_next_debug_cache.get(v.id, {})
                    else:
                        phi_next = phi_next_cache.get(v.id, 0.0)
                        phi_debug = {}
                elif scheme == "PBRS_KP_V2":
                    phi_next, phi_debug = self._compute_phi_value_v2(dag, vehicle=v)
                else:
                    phi_next = self._compute_phi_value(dag, vehicle=v)
                    phi_debug = {}
                phi_next = float(np.nan_to_num(phi_next, nan=0.0, posinf=0.0, neginf=0.0))
                phi_list.append(phi_next)
                try:
                    rem_list.append(float(np.sum(dag.rem_comp[dag.status != 3])))
                except Exception:
                    pass

                delta_t = ctx.get("t_local", 0.0) - ctx.get("t_actual", 0.0)
                if not np.isfinite(delta_t):
                    delta_t = 0.0
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
                r_progress = float(step_completed_counts.get(v.id, 0)) * self.config.SUBTASK_SUCCESS_BONUS
                energy_step = step_energy_cost.get(v.id, 0.0)
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
                    p_ratio_raw = ctx.get("power_ratio", 0.0)
                    if p_ratio_raw is None:
                        p_ratio_raw = 0.0
                    p_ratio = float(np.clip(p_ratio_raw, 0.0, 1.0))
                    e_tx = float(energy_step)
                    # [P0修复] 使用标定后的能耗惩罚
                    # 原问题：E_REF=1.0太大，导致r_energy量级<0.001，完全失效
                    # 修复：E_REF_CAL基于物理估算(P_max×typical_tx_time≈0.01J)
                    E_REF_CAL = getattr(self.config, 'E_REF_CALIBRATED', 0.01)
                    W_E = getattr(self.config, 'ENERGY_WEIGHT_CALIBRATED', 0.4)
                    r_energy = -W_E * float(np.tanh(e_tx / max(E_REF_CAL, 1e-9)))
                    if not ctx.get("illegal"):
                        # 功率正则（二次惩罚高功率）
                        W_P = getattr(self.config, 'POWER_WEIGHT_CALIBRATED', 0.15)
                        r_power = -W_P * float(p_ratio ** 2)
                    if dag.deadline > 0 and dag.is_failed and dag.fail_reason == 'deadline':
                        elapsed = self.time - dag.start_time
                        overtime_ratio = max((elapsed - dag.deadline) / dag.deadline, 0.0)
                        r_timeout = -self.config.TIMEOUT_L1 * np.tanh(self.config.TIMEOUT_K * overtime_ratio)
                        r_timeout += -self.config.TIMEOUT_L2 * float(max(overtime_ratio - self.config.TIMEOUT_O0, 0.0) ** 2)
                    base_reward = r_progress + r_term + r_timeout + r_energy + r_power + r_illegal
                    r_total = base_reward + r_lat + r_shape
                else:
                    if getattr(self.config, "ENERGY_LAMBDA_PBRS", 0.0) > 0.0:
                        e_tx = float(energy_step)
                        e_norm = e_tx / max(self.config.E_REF, 1e-9)
                        r_energy = -self.config.ENERGY_LAMBDA_PBRS * float(np.clip(e_norm, 0.0, self.config.E_CLIP))
                    if dag.deadline > 0 and dag.is_failed and dag.fail_reason == 'deadline':
                        elapsed = self.time - dag.start_time
                        overtime_ratio = max((elapsed - dag.deadline) / dag.deadline, 0.0)
                        r_timeout = -self.config.TIMEOUT_PENALTY_WEIGHT * np.tanh(
                            self.config.TIMEOUT_STEEPNESS * overtime_ratio
                        )
                    base_reward = r_progress + r_term + r_timeout + r_energy + r_illegal
                    r_total = base_reward + r_base + r_shape

                # [Chain] 风险成本（仅影响奖励，不影响物理推进）
                r_chain = 0.0
                if getattr(self.config, "CHAIN_ENABLED", False):
                    alpha_d = float(getattr(self.config, "CHAIN_RISK_WEIGHT_DEPOSIT", 0.0))
                    alpha_f = float(getattr(self.config, "CHAIN_RISK_WEIGHT_FAIL", 0.0))
                    tx_flag = int(tx_flags.get(v.id, 0))
                    if tx_flag and (alpha_d > 0.0 or alpha_f > 0.0):
                        proxy_size = float(ctx.get("cycles", 0.0))
                        deposit = float(self.config.CHAIN_DEPOSIT_BASE) + float(self.config.CHAIN_DEPOSIT_SCALE) * proxy_size
                        p95 = float(self.chain_state_dict.get("p95_confirm", 0.0))
                        p_fail = float(self.chain_state_dict.get("p_fail", 0.0))
                        lock_cost = alpha_d * deposit * p95
                        fail_cost = alpha_f * deposit * p_fail
                        r_chain = -(lock_cost + fail_cost)
                        risk_cost_sum += (lock_cost + fail_cost)
                r_total = float(r_total + r_chain)
                r_total = float(np.clip(r_total, -self.config.R_CLIP, self.config.R_CLIP))
                r_total = float(np.nan_to_num(r_total, nan=0.0))
                r_shape = float(np.nan_to_num(r_shape, nan=0.0))

                # [审计] per-decision奖励分项记录
                _is_decision = ctx.get("subtask") is not None
                if getattr(self.config, "AUDIT_PER_DECISION_REWARD", False) and _is_decision:
                    tgt = ctx.get("target")
                    action_type = "Local"
                    rsu_id = -1
                    if isinstance(tgt, tuple):
                        if tgt[0] == "RSU":
                            action_type = "RSU"
                            rsu_id = int(tgt[1]) if len(tgt) > 1 else -1
                    elif isinstance(tgt, int):
                        # V2V目标是int（邻居车辆ID）
                        action_type = "V2V"
                    elif tgt == 'Local':
                        action_type = "Local"
                    # 获取并发统计
                    n_rsu_total = sum(1 for p in self._last_commit_plans if p.get("planned_kind") == "rsu")
                    n_rsu_id = sum(1 for p in self._last_commit_plans 
                                   if p.get("planned_kind") == "rsu" and 
                                   isinstance(p.get("planned_target"), tuple) and 
                                   len(p.get("planned_target", ())) > 1 and
                                   p.get("planned_target", (None, -1))[1] == rsu_id) if rsu_id >= 0 else 0
                    # 获取队列负载状态（审计用）
                    r_queue = 0.0
                    rsu_load_ratio = 0.0
                    if rsu_id >= 0:
                        rsu_load = self._get_rsu_queue_load(rsu_id)
                        rsu_limit = getattr(self.config, 'RSU_QUEUE_CYCLES_LIMIT', 1e12)
                        rsu_load_ratio = float(rsu_load / max(rsu_limit, 1e-9))
                        load_weight = getattr(self.config, 'RSU_LOAD_WEIGHT', 0.3)
                        r_queue = -load_weight * rsu_load_ratio
                    # 获取t_est和est_error
                    t_est = ctx.get("t_actual", 0.0)
                    est_error = ctx.get("est_error", 0.0)  # 如果有的话
                    
                    self._audit_per_decision_rewards.append({
                        "episode": self.episode_count,
                        "step": self.steps,
                        "veh_id": v.id,
                        "subtask_idx": int(ctx.get("subtask", -1)) if ctx.get("subtask") is not None else -1,
                        "action_type": action_type,
                        "rsu_id": rsu_id,
                        "n_rsu_total": n_rsu_total,
                        "n_rsu_id": n_rsu_id,
                        "t_est": float(t_est),
                        "r_lat": float(r_lat) if 'r_lat' in dir() else 0.0,
                        "r_shape": float(r_shape),
                        "r_queue": float(r_queue),
                        "r_energy": float(r_energy),
                        "r_power": float(r_power),
                        "r_timeout": float(r_timeout),
                        "r_term": float(r_term),
                        "r_illegal": float(r_illegal),
                        "r_total": float(r_total),
                        "power_ratio": float(ctx.get("power_ratio", 0.0)),
                        "t_tx": float(ctx.get("t_tx", 0.0)),  # 从ctx获取t_tx
                        "e_tx": float(e_tx),  # 使用局部计算的e_tx
                    })

                # PBRS诊断（非法/无任务统计已在plan阶段统一计数，避免口径重复）

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
                    is_no_task_step = bool(ctx.get("no_task_available", False))
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
                        _add_metric("r_energy", r_energy)
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
                e_max = max((p_max_watt + p_circuit) * float(self.config.DT), 1e-12)
                energy_norm = float(np.clip(energy_step / e_max, 0.0, 1.0))

                if hasattr(self, '_reward_stats'):
                    self._reward_stats.add_metric("power_ratio", p_ratio)

                self._episode_dT_eff_values.append(dT_eff)
                self._episode_energy_norm_values.append(energy_norm)
                self._episode_t_tx_values.append(ctx.get("t_tx", 0.0))
            if phi_list:
                phi_avg = float(np.mean(phi_list))
                total_rem = float(np.sum(rem_list)) if rem_list else 0.0

        # [Chain] 累计风险成本（episode级）
        self._chain_risk_cost_total += float(risk_cost_sum)

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
        info["tx_arrivals_step"] = int(tx_arrivals_step)
        info["chain_state"] = dict(self.chain_state_dict)
        info["risk_cost_sum"] = float(risk_cost_sum)
        info["risk_cost_mean"] = float(risk_cost_sum / max(len(self.vehicles), 1))
        cand_stats = getattr(self, "_last_candidate_step_stats", {}) or {}
        info["candidate_reachable_cnt_mean"] = float(cand_stats.get("candidate_reachable_cnt_mean", 0.0))
        info["candidate_reachable_cnt_p95"] = float(cand_stats.get("candidate_reachable_cnt_p95", 0.0))
        info["candidate_dropped_cnt_mean"] = float(cand_stats.get("candidate_dropped_cnt_mean", 0.0))
        info["candidate_dropped_cnt_p95"] = float(cand_stats.get("candidate_dropped_cnt_p95", 0.0))
        info["feasible_cnt_v2v_mean"] = float(cand_stats.get("feasible_cnt_v2v_mean", 0.0))
        info["padded_cnt_v2v_mean"] = float(cand_stats.get("padded_cnt_v2v_mean", 0.0))
        info["masked_cnt_total_mean"] = float(cand_stats.get("masked_cnt_total_mean", 0.0))
        info["not_in_candidate_fallback_cnt"] = int(step_not_in_candidate_fallback_cnt)
        info["illegal_by_connectivity_cnt"] = int(step_illegal_by_connectivity_cnt)
        if getattr(self.config, "EDGE_RATE_RECOMPUTE_AUDIT", False):
            edge_audit = getattr(self, "_edge_rate_audit_step", None) or {}
            info["edge_rate_recompute_cnt"] = int(edge_audit.get("edge_rate_recompute_cnt", 0))
            info["edge_rate_delta_mean"] = float(edge_audit.get("edge_rate_delta_mean", 0.0))
            info["edge_rate_delta_p95"] = float(edge_audit.get("edge_rate_delta_p95", 0.0))
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
        B_curr = self._get_total_B_remaining()
        deltaW = max(0.0, self._p2_W_prev - W_curr)  # 防止数值抖动造成负值
        deltaB = max(0.0, self._p2_B_prev - B_curr)
        total_active = self._get_total_active_tasks()
        
        if total_active > 0:
            self._p2_active_time += self.config.DT
            self._p2_deltaW_active += deltaW
            # 检测长时间无推进
            if deltaW < 1e-6 and deltaB < 1e-6:  # 计算与通信均几乎没有推进
                self._p2_zero_delta_steps += 1
            else:
                self._p2_zero_delta_steps = 0
            if deltaW < 1e-6 and deltaB >= 1e-6:
                self._p2_comm_only_steps += 1
        else:
            self._p2_idle_time += self.config.DT
        
        self._p2_W_prev = W_curr
        self._p2_B_prev = B_curr
        
        # [一致性检查] 长时间无推进警告
        if self._p2_zero_delta_steps >= 50 and total_active > 0:
            import warnings
            warnings.warn(
                f"[P2警告] 连续{self._p2_zero_delta_steps}步活跃任务计算与通信均未推进，"
                f"total_active={total_active}, W_curr={W_curr:.2e}, B_curr={B_curr:.2e}",
                UserWarning
            )
            self._p2_zero_delta_steps = 0  # 重置计数，避免重复警告

        # 任务重生场景：截断/终止时把本步已完成(未respawn)的 DAG 计入 episode 成功/完成/超时数，保证 T_SR、deadline_miss 含最后一步
        if (terminated or truncated) and getattr(self.config, 'TASK_RESPAWN_ON_COMPLETION', False):
            for v in self.vehicles:
                if v.task_dag.is_finished or v.task_dag.is_failed:
                    if v.task_dag.is_finished and not v.task_dag.is_failed:
                        self._episode_task_success_count = getattr(self, '_episode_task_success_count', 0) + 1
                    if v.task_dag.is_failed and getattr(v.task_dag, 'fail_reason', None) == 'deadline':
                        self._episode_task_deadline_fail_count = getattr(self, '_episode_task_deadline_fail_count', 0) + 1
                    self._episode_task_completion_count = getattr(self, '_episode_task_completion_count', 0) + 1
        # [记录episode统计] 每步都记录，但只在episode结束时写入文件
        self._log_episode_stats(terminated, truncated)
        
        # [P2/P0新增] 将关键健康指标写入info（供审计脚本使用）
        if hasattr(self, '_last_episode_metrics'):
            info['episode_metrics'] = self._last_episode_metrics.copy()
            # 同时直接在info顶层写入这些字段（向后兼容）
            info.update(self._last_episode_metrics)
        info['rate_snapshot_used'] = use_pbrs

        # =========== E. 统一指标输出 ===========
        # 四段分解 + 干扰 + 公平性（每步累积，终局时输出汇总）
        v2v_stats = getattr(self.channel, 'last_v2v_stats', {})
        v2i_stats = getattr(self.channel, 'last_v2i_stats', {})
        sinr_vals = v2v_stats.get('sinr_values', [])
        rb_occ = v2v_stats.get('rb_occupancy', np.zeros(getattr(self.channel, 'num_rb', 4), dtype=int))
        i_caused_dict = v2v_stats.get('i_caused', {})
        i_total_dict = v2v_stats.get('i_total', {})
        i_caused_input_dict = v2v_stats.get('i_caused_input', {})
        i_total_input_dict = v2v_stats.get('i_total_input', {})
        v2i_sinr_vals = v2i_stats.get('sinr_values', [])
        v2i_i_caused_dict = v2i_stats.get('i_caused', {})
        v2i_i_total_dict = v2i_stats.get('i_total', {})
        v2i_i_caused_input_dict = v2i_stats.get('i_caused_input', {})
        v2i_i_total_input_dict = v2i_stats.get('i_total_input', {})

        i_total_all_vals = list(i_total_dict.values()) + list(v2i_i_total_dict.values())
        i_caused_input_all_vals = list(i_caused_input_dict.values()) + list(v2i_i_caused_input_dict.values())

        # 累积 per-step 干扰统计
        if not hasattr(self, '_ep_sinr_all'):
            self._ep_sinr_all = []
            self._ep_i_caused_all = []
            self._ep_i_total_all = []
            self._ep_rb_occ_all = []
            self._ep_power_watt_all = []
            self._ep_e_tx_all = []
            self._ep_finish_times = []

        if sinr_vals:
            self._ep_sinr_all.extend(sinr_vals)
        if v2i_sinr_vals:
            self._ep_sinr_all.extend(v2i_sinr_vals)
        self._ep_i_caused_all.extend(i_caused_input_all_vals)
        self._ep_i_total_all.extend(i_total_all_vals)
        if np.any(rb_occ > 0):
            self._ep_rb_occ_all.append(rb_occ.copy())
        for vid, e in step_energy_input_cost.items():
            if e > 0:
                self._ep_e_tx_all.append(float(e))
        for pid, pw in step_power_ratio.items():
            if pw > 0:
                p_w = Cfg.dbm2watt(Cfg.TX_POWER_MIN_DBM) * (
                    Cfg.dbm2watt(Cfg.TX_POWER_MAX_DBM) / max(Cfg.dbm2watt(Cfg.TX_POWER_MIN_DBM), 1e-12)
                ) ** float(pw)
                self._ep_power_watt_all.append(float(p_w))

        # per-step 干扰快照写入 info
        info['v2v_sinr_p50'] = float(np.median(sinr_vals)) if sinr_vals else 0.0
        info['v2v_sinr_p05'] = float(np.percentile(sinr_vals, 5)) if len(sinr_vals) >= 2 else 0.0
        info['v2v_rb_concurrency'] = float(np.mean(rb_occ[rb_occ > 0])) if np.any(rb_occ > 0) else 0.0
        info['v2v_i_caused_mean'] = float(np.mean(list(i_caused_input_dict.values()))) if i_caused_input_dict else 0.0
        info['v2v_i_total_mean'] = float(np.mean(list(i_total_dict.values()))) if i_total_dict else 0.0
        info['v2v_i_total_p95'] = float(np.percentile(list(i_total_dict.values()), 95)) if i_total_dict else 0.0
        info['v2v_i_caused_input_mean'] = float(np.mean(list(i_caused_input_dict.values()))) if i_caused_input_dict else 0.0
        info['v2i_i_total_mean'] = float(np.mean(list(v2i_i_total_dict.values()))) if v2i_i_total_dict else 0.0
        info['v2i_i_total_p95'] = float(np.percentile(list(v2i_i_total_dict.values()), 95)) if v2i_i_total_dict else 0.0
        info['v2i_i_caused_input_mean'] = float(np.mean(list(v2i_i_caused_input_dict.values()))) if v2i_i_caused_input_dict else 0.0
        info['interf_i_total_all_mean'] = float(np.mean(i_total_all_vals)) if i_total_all_vals else 0.0
        info['interf_i_total_all_p95'] = float(np.percentile(i_total_all_vals, 95)) if i_total_all_vals else 0.0
        info['interf_i_caused_input_mean'] = float(np.mean(i_caused_input_all_vals)) if i_caused_input_all_vals else 0.0

        # Handover 事件数
        info['ho_event_count'] = len(getattr(self, '_ho_events', []))

        # Trust 统计
        if hasattr(self, '_trust_mgr'):
            trust_stats = self._trust_mgr.get_stats()
            info['trust_attempts'] = trust_stats['trust_attempts']
            info['trust_failures'] = trust_stats['trust_failures']
            info['trust_failure_rate'] = trust_stats['trust_failure_rate']
            info['trust_retry_count'] = trust_stats['trust_retry_count']
            info['malicious_count'] = trust_stats.get('malicious_count', 0)

        # 终局汇总指标
        if terminated or truncated:
            # 收集所有车辆 finish time
            finish_times = []
            for v in self.vehicles:
                dag = v.task_dag
                if dag.is_finished and dag.completion_time is not None:
                    finish_times.append(dag.completion_time)
            success_count = sum(1 for v in self.vehicles if v.task_dag.is_finished and not v.task_dag.is_failed)
            total_count = len(self.vehicles)
            # 任务重生场景：用本 episode 累计成功完成数/完成数，否则用截断瞬间的当前 DAG 成功数
            if getattr(self.config, 'TASK_RESPAWN_ON_COMPLETION', False):
                comp = max(getattr(self, '_episode_task_completion_count', 0), 1)
                info['ep_success_rate'] = float(getattr(self, '_episode_task_success_count', 0) / comp)
                info['ep_success_count'] = int(getattr(self, '_episode_task_success_count', 0))
            else:
                info['ep_success_rate'] = float(success_count / max(total_count, 1))
                info['ep_success_count'] = int(success_count)

            if finish_times:
                ft = np.array(finish_times)
                deadlines = np.array([v.task_dag.deadline for v in self.vehicles
                                     if v.task_dag.is_finished and v.task_dag.completion_time is not None])
                delta_t = ft - deadlines if len(deadlines) == len(ft) else ft
                info['ep_T_finish_mean'] = float(np.mean(ft))
                info['ep_T_finish_p50'] = float(np.median(ft))
                info['ep_makespan'] = float(np.max(ft))
                info['ep_delta_T_p50'] = float(np.median(delta_t))
                info['ep_delta_T_p95'] = float(np.percentile(delta_t, 95))
            else:
                info['ep_T_finish_mean'] = 0.0
                info['ep_T_finish_p50'] = 0.0
                info['ep_makespan'] = 0.0
                info['ep_delta_T_p50'] = 0.0
                info['ep_delta_T_p95'] = 0.0

            # 干扰汇总
            sinr_all = self._ep_sinr_all
            if sinr_all:
                sa = np.array(sinr_all)
                info['ep_sinr_p50'] = float(np.median(sa))
                info['ep_sinr_p05'] = float(np.percentile(sa, 5))
                info['ep_sinr_p95'] = float(np.percentile(sa, 95))
            else:
                info['ep_sinr_p50'] = info['ep_sinr_p05'] = info['ep_sinr_p95'] = 0.0
            if self._ep_rb_occ_all:
                rb_stack = np.stack(self._ep_rb_occ_all)
                info['ep_rb_concurrency_mean'] = float(np.mean(rb_stack[rb_stack > 0])) if np.any(rb_stack > 0) else 0.0
            else:
                info['ep_rb_concurrency_mean'] = 0.0
            if self._ep_i_caused_all:
                info['ep_i_caused_mean'] = float(np.mean(self._ep_i_caused_all))
            else:
                info['ep_i_caused_mean'] = 0.0

            # 功率与能耗
            if self._ep_e_tx_all:
                info['ep_E_tx_total'] = float(np.sum(self._ep_e_tx_all))
                info['ep_E_tx_mean'] = float(np.mean(self._ep_e_tx_all))
            else:
                info['ep_E_tx_total'] = info['ep_E_tx_mean'] = 0.0
            if self._ep_power_watt_all:
                info['ep_power_mean_W'] = float(np.mean(self._ep_power_watt_all))
            else:
                info['ep_power_mean_W'] = 0.0

            # 公平性: Jain's fairness index + worst 10%
            if finish_times and len(finish_times) >= 2:
                ft_arr = np.array(finish_times)
                jain = float((np.sum(ft_arr) ** 2) / (len(ft_arr) * np.sum(ft_arr ** 2)))
                worst_k = max(1, int(len(ft_arr) * 0.1))
                worst10 = np.sort(ft_arr)[-worst_k:]
                info['ep_jain_fairness'] = jain
                info['ep_worst10_mean'] = float(np.mean(worst10))
            else:
                info['ep_jain_fairness'] = 1.0
                info['ep_worst10_mean'] = 0.0

            # 四段分解（逐车辆取均值）
            T_tx_svc_list, T_tx_wait_list, T_cpu_svc_list, T_cpu_wait_list = [], [], [], []
            for v in self.vehicles:
                vid = v.id
                # 通信服务时间 = 实际传输字节/速率的累积
                tx_svc = step_tx_time.get(vid, 0.0)
                # 通信等待时间
                cw = self._compute_comm_wait(vid) if hasattr(self, '_compute_comm_wait') else {'total_v2i': 0, 'total_v2v': 0}
                tx_wait = cw.get('total_v2i', 0.0) + cw.get('total_v2v', 0.0)
                # CPU 服务 = 完成 cycles / freq
                cpu_svc = self.CPU_cycles_local.get(vid, 0.0) / max(v.cpu_freq, 1e-9)
                # CPU 等待 = queue_load / freq
                cpu_wait = self._get_veh_queue_load(vid) / max(v.cpu_freq, 1e-9)
                T_tx_svc_list.append(tx_svc)
                T_tx_wait_list.append(tx_wait)
                T_cpu_svc_list.append(cpu_svc)
                T_cpu_wait_list.append(cpu_wait)
            info['ep_T_tx_svc_mean'] = float(np.mean(T_tx_svc_list)) if T_tx_svc_list else 0.0
            info['ep_T_tx_wait_mean'] = float(np.mean(T_tx_wait_list)) if T_tx_wait_list else 0.0
            info['ep_T_cpu_svc_mean'] = float(np.mean(T_cpu_svc_list)) if T_cpu_svc_list else 0.0
            info['ep_T_cpu_wait_mean'] = float(np.mean(T_cpu_wait_list)) if T_cpu_wait_list else 0.0

            # 重置 episode 累积
            self._ep_sinr_all = []
            self._ep_i_caused_all = []
            self._ep_i_total_all = []
            self._ep_rb_occ_all = []
            self._ep_power_watt_all = []
            self._ep_e_tx_all = []

        if self._rate_snapshot is not None and self._rate_snapshot_step == self.steps:
            self._rate_snapshot_prev = self._rate_snapshot
            self._rate_snapshot_prev_step = self._rate_snapshot_step
        else:
            self._rate_snapshot_prev = None
            self._rate_snapshot_prev_step = -1

        # 清理速率快照，避免跨步污染
        self._clear_rate_snapshot()

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

        boundary_respawn_count = 0
        if not terminated and not truncated:
            exited_ids = []
            map_size = float(self.config.MAP_SIZE)
            for v in self.vehicles:
                x_pos = float(v.pos[0])
                if 0.0 <= x_pos <= map_size:
                    continue
                if self._vehicle_has_active_jobs(v.id):
                    # 车辆仍有在途任务时，不直接重生，先回退到边界避免通信队列长期停滞。
                    v.pos[0] = float(np.clip(x_pos, 0.0, map_size))
                else:
                    exited_ids.append(v.id)
            for vehicle_id in exited_ids:
                self._respawn_vehicle(vehicle_id)
                boundary_respawn_count += 1

        arrival_count = 0
        if not terminated and not truncated:
            arrival_count = self._handle_dynamic_arrivals()
        info['arrival_count'] = int(arrival_count)
        info['boundary_respawn_count'] = int(boundary_respawn_count)

        # 任务持续生成：DAG 完成/失败后立即为该车辆分配新任务（静态车辆场景）
        task_respawn_count = 0
        if not terminated and not truncated and getattr(self.config, 'TASK_RESPAWN_ON_COMPLETION', False):
            for v in self.vehicles:
                if (v.task_dag.is_finished or v.task_dag.is_failed):
                    if not self._vehicle_has_active_jobs(v.id):
                        if v.task_dag.is_finished and not v.task_dag.is_failed:
                            self._episode_task_success_count = getattr(self, '_episode_task_success_count', 0) + 1
                        if v.task_dag.is_failed and getattr(v.task_dag, 'fail_reason', None) == 'deadline':
                            self._episode_task_deadline_fail_count = getattr(self, '_episode_task_deadline_fail_count', 0) + 1
                        self._episode_task_completion_count = getattr(self, '_episode_task_completion_count', 0) + 1
                        self._assign_new_dag_to_vehicle(v)
                        task_respawn_count += 1
        info['task_respawn_count'] = int(task_respawn_count)

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
        - action_mask: 合法动作 (位置有效、队列未满)

        维度约束:
        - neighbors使用固定维度填充 (MAX_NEIGHBORS, 8)
        - 满足Gymnasium批处理要求
        """
        def _nan_clip(arr, low=None, high=None, dtype=None):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if low is not None or high is not None:
                lo = low if low is not None else -np.inf
                hi = high if high is not None else np.inf
                arr = np.clip(arr, lo, hi)
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            return arr

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
        step_candidate_reachable = []
        step_candidate_dropped = []
        step_feasible_v2v = []
        step_padded_v2v = []
        step_masked_total = []

        # --- CTDE global_state（集中训练共享）---
        # 维度布局（共30维，固定对应 CTDE_GLOBAL_DIM=30）：
        #   [0:3]  per-RSU 计算队列等待时间（归一化）
        #   [3:6]  per-RSU 关联车辆数占总车辆比
        #   [6]    本episode累计Local决策占比
        #   [7]    本episode累计RSU决策占比
        #   [8]    本episode累计V2V决策占比
        #   [9]    当前步归一化时间进度
        #   [10]   全局平均RSU等待时间（归一化）
        #   [11-29] 预留为0
        _gs = np.zeros(30, dtype=np.float32)
        _n_rsu = len(self.rsus)
        for _k, _rsu in enumerate(self.rsus[:3]):
            _gs[_k] = float(np.clip(
                self._get_rsu_queue_wait_time(_rsu.id) * self._inv_max_wait, 0.0, 1.0
            ))
        _n_veh = max(len(self.vehicles), 1)
        _veh_per_rsu = [0] * 3
        for _v in self.vehicles:
            _sid = getattr(_v, 'serving_rsu_id', None)
            if _sid is not None and 0 <= _sid < 3:
                _veh_per_rsu[_sid] += 1
        for _k in range(3):
            _gs[3 + _k] = float(_veh_per_rsu[_k]) / _n_veh
        _dec_total = max(sum(getattr(self, '_decision_counts', {k: 0 for k in ('local', 'rsu', 'v2v')}).values()), 1)
        _edc = getattr(self, '_decision_counts', {'local': 0, 'rsu': 0, 'v2v': 0})
        _gs[6] = float(_edc.get('local', 0)) / _dec_total
        _gs[7] = float(_edc.get('rsu', 0)) / _dec_total
        _gs[8] = float(_edc.get('v2v', 0)) / _dec_total
        _gs[9] = float(np.clip(self._episode_steps / max(self.config.MAX_STEPS, 1), 0.0, 1.0))
        if _n_rsu > 0:
            _gs[10] = float(np.clip(
                np.mean([self._get_rsu_queue_wait_time(r.id) * self._inv_max_wait for r in self.rsus]),
                0.0, 1.0
            ))
        global_state_vec = _gs
        # --- end CTDE global_state ---

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
            
            # [RSU选择] 获取所有覆盖范围内的RSU（用于ENABLE_RSU_SELECTION模式）
            rsus_in_range_list = self._get_all_rsus_in_range(v.pos)
            rsus_in_range_ids = [rsu.id for rsu in rsus_in_range_list] if rsus_in_range_list else []
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

            prev_v2i_rate = 0.0
            if self._rate_prev_cache_v2i is not None and 0 <= v.id < len(self._rate_prev_cache_v2i):
                prev_v2i_rate = float(self._rate_prev_cache_v2i[v.id])
            prev_v2i_rate_norm = np.clip(prev_v2i_rate * self._inv_max_rate_v2i, 0, 1)

            self_info = np.array([
                v.vel[0] * self._inv_max_velocity, v.vel[1] * self._inv_max_velocity,
                np.clip(self_wait * self._inv_max_wait, 0, 1),
                v.cpu_freq * self._inv_max_cpu,
                np.clip(est_v2i_rate * self._inv_max_rate_v2i, 0, 1),
                prev_v2i_rate_norm,
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
            serving_rsu_onehot = np.zeros(self.config.NUM_RSU, dtype=np.float32)
            if serving_rsu_id is not None and 0 <= serving_rsu_id < self.config.NUM_RSU:
                serving_rsu_onehot[serving_rsu_id] = 1.0
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

                # 使用当前发射功率估计V2V速率，避免对候选过度乐观
                est_v2v_rate = self.channel.compute_one_rate(
                    v, other.pos, 'V2V', self.time
                )
                est_v2v_rate = max(est_v2v_rate, 1e-6)
                trans_time = task_data_size / est_v2v_rate if task_data_size > 0 else 0.0

                # [处理器共享] 使用新的延迟估算方法
                queue_wait_time = self._get_node_delay(other)
                comp_time = task_comp_size / max(other.cpu_freq, 1e-6)

                # T_finish_est = CommWait + CommTx + CPUWait + CPUExec
                # 使用与候选排序一致的估计，减少V2V过选偏差
                t_finish_est = comm_wait_v2v_for_mask + trans_time + queue_wait_time + comp_time
                
                # [P0修复] 通信阶段时间，用于contact time约束
                # 物理语义：contact time仅约束数据传输阶段，计算在目标节点本地执行
                t_comm_phase = comm_wait_v2v_for_mask + trans_time

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

                # [R3修复] 不再用预测接触时间进行硬剔除；超出则作为软惩罚
                contact_slack = time_to_break - t_comm_phase
                if contact_slack < 0:
                    t_finish_est += abs(contact_slack)

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
                v, candidate_info, serving_rsu_id,
                rsus_in_range=rsus_in_range_ids  # [RSU选择] 传递覆盖范围内的RSU列表
            )
            candidate_ids = candidate_set["ids"]
            candidate_types = candidate_set["types"]
            target_mask_row = candidate_set["mask"].copy()
            v2v_slots = candidate_set["v2v_slots"]
            rsu_start_idx = candidate_set.get("rsu_start_idx", 1)
            rsu_end_idx = candidate_set.get("rsu_end_idx", 2)
            v2v_start_idx = candidate_set.get("v2v_start_idx", rsu_end_idx)
            neighbor_count = sum(1 for info in v2v_slots if info is not None)
            step_candidate_reachable.append(int(candidate_set.get("reachable_cnt", 0)))
            step_candidate_dropped.append(int(candidate_set.get("dropped_cnt", 0)))
            step_feasible_v2v.append(int(candidate_set.get("feasible_cnt_v2v", 0)))
            step_padded_v2v.append(int(candidate_set.get("padded_cnt_v2v", 0)))
            step_masked_total.append(int(candidate_set.get("masked_cnt_total", 0)))

            for idx, info in enumerate(v2v_slots):
                if info is None:
                    continue
                neighbors_array[idx] = [
                    info['id'], info['rel_pos'][0], info['rel_pos'][1],
                    info['vel'][0] * self._inv_max_velocity, info['vel'][1] * self._inv_max_velocity,
                    np.clip(info['queue_wait'] * self._inv_max_wait, 0, 1),
                    info['cpu_freq'] * self._inv_max_cpu,
                    np.clip(self._v2v_ref_rate_norm(info['dist']), 0, 1)
                ]

            # [关键] 死锁兜底：如果所有目标都不可用，强制开启Local
            if not np.any(target_mask_row):
                target_mask_row[0] = True
            
            # [审计] 保存mask到vehicle对象，用于审计收集
            v._last_action_mask = target_mask_row.copy()

            enable_rsu_selection = getattr(self.config, "ENABLE_RSU_SELECTION", False)
            num_rsu = len(self.rsus)

            resource_id_list = np.zeros(self.config.MAX_TARGETS, dtype=np.int64)
            resource_id_list[0] = 1
            for idx in range(rsu_start_idx, min(rsu_end_idx, self.config.MAX_TARGETS)):
                if idx < len(candidate_types) and int(candidate_types[idx]) == 2:
                    if idx < len(candidate_ids) and int(candidate_ids[idx]) >= 0 and bool(target_mask_row[idx]):
                        resource_id_list[idx] = 2

            for idx in range(self.config.MAX_NEIGHBORS):
                target_idx = v2v_start_idx + idx
                if target_idx >= self.config.MAX_TARGETS:
                    break
                cid = int(candidate_ids[target_idx])
                if cid >= 0:
                    resource_id_list[target_idx] = 3 + cid

            padded_target_mask = target_mask_row.copy()

            step_avail_l += 1.0 if target_mask_row[0] else 0.0
            rsu_mask = target_mask_row[rsu_start_idx:rsu_end_idx]
            step_avail_r += 1.0 if np.any(rsu_mask) else 0.0
            if self.config.MAX_NEIGHBORS > 0 and v2v_start_idx < len(target_mask_row):
                step_avail_v += float(np.mean(target_mask_row[v2v_start_idx:]))
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

            # [BugFix] 只缓存V2V候选ID，用types过滤而非假设index>=2都是V2V
            self._last_candidates[v.id] = [
                int(candidate_ids[i]) for i in range(len(candidate_ids))
                if i < len(candidate_types) and int(candidate_types[i]) == 3
            ]
            self._last_candidate_set[v.id] = candidate_set
            self._last_rsu_choice[v.id] = serving_rsu_id

            # [改动A] 计算通信队列等待时间（含 EDGE 挤占效应）
            comm_wait = self._compute_comm_wait(v.id)
            comm_wait_total_v2i = comm_wait['total_v2i']
            comm_wait_total_v2v = comm_wait['total_v2v']

            # 归一化 CommWait（使用 log(1+x) 压缩防止饱和）
            norm_max_comm_wait = getattr(self.config, 'NORM_MAX_COMM_WAIT', 2.0)
            comm_wait_total_v2i_norm = np.clip(np.log1p(comm_wait_total_v2i) / np.log1p(norm_max_comm_wait), 0, 1)
            comm_wait_total_v2v_norm = np.clip(np.log1p(comm_wait_total_v2v) / np.log1p(norm_max_comm_wait), 0, 1)
            resource_raw = np.zeros((self.config.MAX_TARGETS, self.config.RESOURCE_RAW_DIM), dtype=np.float32)

            # --- 物理约束特征 + 信誉特征布局 ---
            # col 0-8: cpu/queue/dist/rate/rel_xy/vel_xy/node_type  (物理)
            # col 9 : slack_norm  (硬约束: 任务松弛度)
            # col10 : contact_norm (硬约束: 链路剩余接触时间)
            # col11 : t_comp_lb   (快照下界: 计算时间)
            # col12 : hat_rho     (信誉估计)
            # col13 : uncertainty (信誉不确定度)
            trust_enabled = getattr(self.config, 'TRUST_ENABLED', False)
            slack_norm = val_urgency

            # --- 下界快照 (D) ---
            # t_comp_lb = w_v/f_m + cpu_backlog/f_m
            local_backlog = self._get_veh_queue_load(v.id)  # cycles
            local_comp_lb = (task_comp_size + local_backlog) / max(v.cpu_freq, 1e-6)
            local_tx_lb = 0.0  # Local 无传输

            # comm_wait 下界 = tx_backlog_bytes / service_rate_lb
            txq_bytes_v2i = sum(j.rem_bytes for j in self.txq_v2i.get(("VEH", v.id), []))
            txq_bytes_v2v = sum(j.rem_bytes for j in self.txq_v2v.get(("VEH", v.id), []))
            # service_rate_lb 使用 rate_prev 或 eps
            rate_prev_v2i = float(self._rate_prev_cache_v2i[v.id]) if v.id < len(self._rate_prev_cache_v2i) else 0.0
            rate_prev_v2v = float(self._rate_prev_cache_v2v[v.id]) if v.id < len(self._rate_prev_cache_v2v) else 0.0
            eps_rate = getattr(self.config, 'EPS_RATE', 1e-9)
            tx_wait_lb_v2i = txq_bytes_v2i * 8.0 / max(rate_prev_v2i, eps_rate)
            tx_wait_lb_v2v = txq_bytes_v2v * 8.0 / max(rate_prev_v2v, eps_rate)

            resource_raw[0] = [
                v.cpu_freq * self._inv_max_cpu,             # 0 cpu
                np.clip(self_wait * self._inv_max_wait, 0, 1),  # 1 queue_wait
                0.0,                                        # 2 distance
                0.0,                                        # 3 rate (Local=0)
                0.0,                                        # 4 rel_x
                0.0,                                        # 5 rel_y
                v.vel[0] * self._inv_max_velocity,          # 6 vel_x
                v.vel[1] * self._inv_max_velocity,          # 7 vel_y
                1.0,                                        # 8 node_type=Local
                slack_norm,                                 # 9 slack_norm (任务松弛度)
                1.0,                                        #10 contact_norm (Local永久连接)
                np.clip(local_comp_lb / 10.0, 0, 1),       #11 t_comp_lb
                1.0,                                        #12 hat_rho (Local始终可信)
                0.0,                                        #13 uncertainty (Local=0)
            ]

            v2i_user_count = self._estimate_v2i_users()
            if enable_rsu_selection:
                speed = np.linalg.norm(v.vel)
                for rsu_id_iter in range(num_rsu):
                    idx = rsu_start_idx + rsu_id_iter
                    if idx >= self.config.MAX_TARGETS:
                        break
                    if rsu_id_iter not in rsus_in_range_ids:
                        continue
                    rsu = self.rsus[rsu_id_iter]
                    rsu_dist_iter = rsu.get_distance(v.pos)
                    rsu_rate_iter = self.channel.compute_one_rate(
                        v, rsu.position, 'V2I', self.time,
                        v2i_user_count=v2i_user_count
                    )
                    rsu_rate_iter = max(rsu_rate_iter, 1e-6)
                    rsu_wait_iter = self._get_node_delay(rsu)
                    rsu_cpu_iter = rsu.cpu_freq
                    if speed > 0.1:
                        contact_time = max(0.0, (rsu.coverage_range - rsu_dist_iter) / speed)
                    else:
                        contact_time = self._max_rsu_contact_time
                    rsu_contact_norm = np.clip(contact_time / max(self._max_rsu_contact_time, 1e-6), 0, 1)
                    rel_rsu = (rsu.position - v.pos) * self._inv_map_size
                    # 下界快照
                    rsu_backlog = self._get_rsu_queue_load(rsu_id_iter)
                    rsu_comp_lb = (task_comp_size + rsu_backlog) / max(rsu_cpu_iter, 1e-6)
                    rsu_tx_lb = task_data_size * 8.0 / max(rsu_rate_iter, eps_rate) if task_data_size > 0 else 0.0
                    # 信誉特征
                    if trust_enabled and hasattr(self, '_trust_mgr'):
                        _rho, _unc = self._trust_mgr.get_reputation(('RSU', rsu_id_iter))
                    else:
                        _rho, _unc = 1.0, 0.5
                    resource_raw[idx] = [
                        rsu_cpu_iter * self._inv_max_cpu,             # 0
                        np.clip(rsu_wait_iter * self._inv_max_wait, 0, 1),  # 1
                        np.clip(rsu_dist_iter / max(self.config.RSU_RANGE, 1e-6), 0, 1),  # 2
                        np.clip(rsu_rate_iter * self._inv_max_rate_v2i, 0, 1),  # 3
                        rel_rsu[0],                                   # 4
                        rel_rsu[1],                                   # 5
                        0.0,                                          # 6
                        0.0,                                          # 7
                        2.0,                                          # 8 RSU
                        slack_norm,                                   # 9 slack_norm
                        rsu_contact_norm,                             #10 contact_norm
                        np.clip(rsu_comp_lb / 10.0, 0, 1),           #11 t_comp_lb
                        float(np.clip(_rho, 0, 1)),                   #12 hat_rho
                        float(np.clip(_unc, 0, 1)),                   #13 uncertainty
                    ]
            elif rsu_available:
                rsu = self.rsus[rsu_id]
                rel_rsu = (rsu.position - v.pos) * self._inv_map_size
                rsu_contact_norm = np.clip(rsu_contact / max(self._max_rsu_contact_time, 1e-6), 0, 1)
                rsu_cpu = rsu.cpu_freq if rsu else self.config.F_RSU
                rsu_backlog_s = self._get_rsu_queue_load(rsu_id)
                rsu_comp_lb_s = (task_comp_size + rsu_backlog_s) / max(rsu_cpu, 1e-6)
                rsu_tx_lb_s = task_data_size * 8.0 / max(rsu_rate, eps_rate) if task_data_size > 0 else 0.0
                if trust_enabled and hasattr(self, '_trust_mgr'):
                    _rho_s, _unc_s = self._trust_mgr.get_reputation(('RSU', rsu_id))
                else:
                    _rho_s, _unc_s = 1.0, 0.5
                resource_raw[1] = [
                    rsu.cpu_freq * self._inv_max_cpu,              # 0
                    np.clip(rsu_wait * self._inv_max_wait, 0, 1),  # 1
                    np.clip(rsu_dist / max(self.config.RSU_RANGE, 1e-6), 0, 1),  # 2
                    np.clip(rsu_rate * self._inv_max_rate_v2i, 0, 1),  # 3
                    rel_rsu[0],                                    # 4
                    rel_rsu[1],                                    # 5
                    0.0,                                           # 6
                    0.0,                                           # 7
                    2.0,                                           # 8 RSU
                    slack_norm,                                    # 9 slack_norm
                    rsu_contact_norm,                              #10 contact_norm
                    np.clip(rsu_comp_lb_s / 10.0, 0, 1),          #11 t_comp_lb
                    float(np.clip(_rho_s, 0, 1)),                  #12 hat_rho
                    float(np.clip(_unc_s, 0, 1)),                  #13 uncertainty
                ]

            for idx, info in enumerate(v2v_slots):
                if info is None:
                    continue
                target_idx = v2v_start_idx + idx
                if target_idx >= self.config.MAX_TARGETS:
                    break
                # 下界快照 + 信誉
                nbr_id = info.get('id', -1)
                nbr_backlog = self._get_veh_queue_load(int(nbr_id)) if nbr_id >= 0 else 0.0
                nbr_comp_lb = (task_comp_size + nbr_backlog) / max(info['cpu_freq'], 1e-6)
                contact_norm_v = np.clip(info['contact_time'] / max(self._max_v2v_contact_time, 1e-6), 0, 1)
                if trust_enabled and hasattr(self, '_trust_mgr') and nbr_id >= 0:
                    _rho_v, _unc_v = self._trust_mgr.get_reputation(('VEH', int(nbr_id)))
                else:
                    _rho_v, _unc_v = 1.0, 0.5
                # col3: log 压缩归一的零干扰参考速率（与动作无关；训练依赖 rate_prev 学习真实服务效果）
                resource_raw[target_idx] = [
                    info['cpu_freq'] * self._inv_max_cpu,            # 0
                    np.clip(info['queue_wait'] * self._inv_max_wait, 0, 1),  # 1
                    np.clip(info['dist'] * self._inv_v2v_range, 0, 1),  # 2
                    np.clip(self._v2v_ref_rate_norm(info['dist']), 0, 1),  # 3 R_ref (log-norm)
                    info['rel_pos'][0],                              # 4
                    info['rel_pos'][1],                              # 5
                    info['vel'][0] * self._inv_max_velocity,         # 6
                    info['vel'][1] * self._inv_max_velocity,         # 7
                    3.0,                                             # 8 V2V
                    slack_norm,                                      # 9 slack_norm
                    contact_norm_v,                                  #10 contact_norm
                    np.clip(nbr_comp_lb / 10.0, 0, 1),              #11 t_comp_lb
                    float(np.clip(_rho_v, 0, 1)),                    #12 hat_rho
                    float(np.clip(_unc_v, 0, 1)),                    #13 uncertainty
                ]

            rate_prev = np.zeros(self.config.MAX_TARGETS, dtype=np.float32)
            snap_prev = getattr(self, "_rate_snapshot_prev", None)
            links_prev = snap_prev.get("links", {}) if isinstance(snap_prev, dict) else {}
            if links_prev:
                src_node = ("VEH", v.id)
                for idx in range(self.config.MAX_TARGETS):
                    typ = int(candidate_types[idx]) if idx < len(candidate_types) else 0
                    cid = int(candidate_ids[idx]) if idx < len(candidate_ids) else -1
                    if typ == 2 and cid >= 0:
                        key = self._rate_key(src_node, ("RSU", cid), "V2I")
                        rate = links_prev.get(key)
                        if rate is not None and np.isfinite(rate):
                            rate_prev[idx] = float(np.clip(rate * self._inv_max_rate_v2i, 0.0, 1.0))
                    elif typ == 3 and cid >= 0:
                        key = self._rate_key(src_node, ("VEH", cid), "V2V")
                        rate = links_prev.get(key)
                        if rate is not None and np.isfinite(rate):
                            rate_prev[idx] = float(np.clip(rate * self._inv_max_rate_v2v, 0.0, 1.0))

            # [关键] 固定维度填充 - 适配批处理要求
            padded_adj = np.zeros((self.config.MAX_NODES, self.config.MAX_NODES), dtype=np.float32)
            padded_adj[:num_nodes, :num_nodes] = v.task_dag.adj

            padded_task_mask = np.zeros(self.config.MAX_NODES, dtype=bool)
            padded_task_mask[:num_nodes] = task_schedulable
            padded_node_valid_mask = np.zeros(self.config.MAX_NODES, dtype=bool)
            padded_node_valid_mask[:num_nodes] = True

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

            # location: [MAX_NODES], 任务执行位置编码
            # 0: Unscheduled, 1: Local, 2: RSU, 3+: Neighbor vehicle ID
            padded_location = np.zeros(MAX_NODES, dtype=np.int32)
            for t_idx in range(num_nodes):
                # 优先从v.task_dag.exec_locations获取（Vehicle属性），其次是v.task_dag.task_locations
                if hasattr(v.task_dag, 'exec_locations') and v.task_dag.exec_locations[t_idx] is not None:
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

            # 数值稳定性处理：确保归一化特征有限且落在合理范围
            padded_node_feats = _nan_clip(padded_node_feats, 0.0, 1.0, dtype=np.float32)
            padded_adj = _nan_clip(padded_adj, 0.0, 1.0, dtype=np.float32)
            padded_data_matrix = _nan_clip(padded_data_matrix, 0.0, 1.0, dtype=np.float32)

            self_info = _nan_clip(self_info, dtype=np.float32)
            if self_info.shape[0] >= 2:
                self_info[:2] = np.clip(self_info[:2], -1.0, 1.0)
            if self_info.shape[0] > 2:
                self_info[2:] = np.clip(self_info[2:], 0.0, 1.0)

            rsu_info = _nan_clip(np.array([rsu_load_norm], dtype=np.float32), 0.0, 1.0, dtype=np.float32)
            serving_rsu_onehot = _nan_clip(serving_rsu_onehot, 0.0, 1.0, dtype=np.float32)

            neighbors_array = _nan_clip(neighbors_array, dtype=np.float32)
            if neighbors_array.shape[1] >= 8:
                neighbors_array[:, 1:3] = np.clip(neighbors_array[:, 1:3], -1.0, 1.0)
                neighbors_array[:, 3:5] = np.clip(neighbors_array[:, 3:5], -1.0, 1.0)
                neighbors_array[:, 5:8] = np.clip(neighbors_array[:, 5:8], 0.0, 1.0)

            resource_raw = _nan_clip(resource_raw, dtype=np.float32)
            for col in (0, 1, 2, 3, 9, 10, 11, 12, 13):
                if col < resource_raw.shape[1]:
                    resource_raw[:, col] = np.clip(resource_raw[:, col], 0.0, 1.0)
            for col in (4, 5, 6, 7):
                if col < resource_raw.shape[1]:
                    resource_raw[:, col] = np.clip(resource_raw[:, col], -1.0, 1.0)

            rate_prev = _nan_clip(rate_prev, 0.0, 1.0, dtype=np.float32)

            task_mask_obs = padded_task_mask.astype(np.float32)
            subtask_mask_obs = padded_task_mask.astype(np.float32)
            node_valid_mask_obs = padded_node_valid_mask.astype(np.float32)
            action_mask_obs = padded_target_mask.astype(np.float32)
            subtask_index_obs = np.array(int(selected_subtask_idx), dtype=np.int64)
            obs_stamp_obs = np.array(int(self._episode_steps), dtype=np.int64)
            obs_list.append({
            'node_x': padded_node_feats,
            'self_info': self_info,
            'rsu_info': rsu_info,
            'serving_rsu_onehot': serving_rsu_onehot,
            'candidate_ids': candidate_ids.astype(np.int64),
            'candidate_types': candidate_types.astype(np.int8),
            'adj': padded_adj,
            'neighbors': neighbors_array,
            'subtask_mask': subtask_mask_obs,  # READY且未分配（供SubtaskHead采样）
            'node_valid_mask': node_valid_mask_obs,  # 仅表示padding有效节点
            'task_mask': task_mask_obs,
            'action_mask': action_mask_obs,  # [新增] Actor专用动作掩码
            'rate_prev': rate_prev,
            'resource_ids': resource_id_list,  # [新增] 资源节点ID列表
            'resource_raw': resource_raw,  # [新增] 资源原始物理特征
            'subtask_index': subtask_index_obs,  # [新设计] 添加当前选中的任务索引
            # [新增] DAG拓扑特征
            'L_fwd': padded_L_fwd,
            'L_bwd': padded_L_bwd,
            'data_matrix': padded_data_matrix,
            'Delta': padded_Delta,
            'location': padded_location,
            'obs_stamp': obs_stamp_obs,
            'global_state': global_state_vec,
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

        if not hasattr(self, "_episode_candidate_stats"):
            self._episode_candidate_stats = {"reachable": [], "dropped": []}
        if step_candidate_reachable:
            step_reach_mean = float(np.mean(step_candidate_reachable))
            step_drop_mean = float(np.mean(step_candidate_dropped)) if step_candidate_dropped else 0.0
            step_reach_p95 = float(np.percentile(step_candidate_reachable, 95))
            step_drop_p95 = float(np.percentile(step_candidate_dropped, 95)) if step_candidate_dropped else 0.0
            self._episode_candidate_stats["reachable"].extend(step_candidate_reachable)
            self._episode_candidate_stats["dropped"].extend(step_candidate_dropped)
        else:
            step_reach_mean = 0.0
            step_drop_mean = 0.0
            step_reach_p95 = 0.0
            step_drop_p95 = 0.0
        self._last_candidate_step_stats = {
            "candidate_reachable_cnt_mean": step_reach_mean,
            "candidate_reachable_cnt_p95": step_reach_p95,
            "candidate_dropped_cnt_mean": step_drop_mean,
            "candidate_dropped_cnt_p95": step_drop_p95,
            "feasible_cnt_v2v_mean": float(np.mean(step_feasible_v2v)) if step_feasible_v2v else 0.0,
            "padded_cnt_v2v_mean": float(np.mean(step_padded_v2v)) if step_padded_v2v else 0.0,
            "masked_cnt_total_mean": float(np.mean(step_masked_total)) if step_masked_total else 0.0,
        }

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
        senders = 0
        for tx_node, queue in self.txq_v2i.items():
            if tx_node[0] == "VEH" and queue:
                senders += 1
        if senders == 0:
            for veh in self.vehicles:
                tgt = getattr(veh, 'curr_target', None)
                if tgt == 'RSU':
                    senders += 1
                elif isinstance(tgt, tuple) and len(tgt) == 2 and tgt[0] == 'RSU':
                    senders += 1
        return max(senders, 1)

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
            types = candidate_set.get("types", [])
            # [BugFix] 按types区分RSU/V2V，不再假设index>=2都是V2V
            for idx in range(1, len(ids)):
                if idx < len(mask) and not bool(mask[idx]):
                    continue
                cand_id = int(ids[idx])
                if cand_id < 0:
                    continue
                cand_type = int(types[idx]) if idx < len(types) else 0
                if cand_type == 2:  # RSU → 取RSU cpu_freq
                    if 0 <= cand_id < len(self.rsus):
                        f_candidates_max = max(f_candidates_max, getattr(self.rsus[cand_id], "cpu_freq", 0.0))
                elif cand_type == 3:  # V2V → 取vehicle cpu_freq
                    if cand_id == vehicle.id:
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

    def _get_active_tx_sets_from_queues(self):
        """基于当前队列状态收集active集合（不含本步新动作）"""
        active_v2i_senders = set()
        active_v2v_senders = set()
        for tx_node, queue in self.txq_v2i.items():
            if tx_node[0] == "VEH" and queue:
                active_v2i_senders.add(int(tx_node[1]))
        for tx_node, queue in self.txq_v2v.items():
            if tx_node[0] == "VEH" and queue:
                active_v2v_senders.add(int(tx_node[1]))
        active_v2i_count = len(active_v2i_senders)
        active_v2v_vehicles = [self._get_vehicle_by_id(vid) for vid in active_v2v_senders]
        active_v2v_vehicles = [v for v in active_v2v_vehicles if v is not None]
        return active_v2i_count, active_v2v_vehicles

    def _get_state_only_v2v_candidates(self, vehicle):
        """仅基于当前状态收集V2V候选（不依赖action/candidate_set缓存）"""
        candidates = []
        if vehicle is None:
            return candidates
        for other in self.vehicles:
            if other.id == vehicle.id:
                continue
            dist = np.linalg.norm(other.pos - vehicle.pos)
            if dist <= self.config.V2V_RANGE:
                candidates.append(other)
        return candidates

    def _get_reachable_f_max_state_only(self, vehicle):
        """state-only版本：Local/serving RSU/V2V邻车的f_max"""
        if vehicle is None:
            return self._f_max_const, {
                "f_local": 0.0,
                "f_serving_rsu": 0.0,
                "f_candidates_max": 0.0,
            }
        f_local = max(getattr(vehicle, "cpu_freq", 0.0), 1e-9)
        rsu, rsu_id = self._get_serving_rsu(vehicle)
        f_serving_rsu = 0.0
        if rsu is not None and rsu_id is not None:
            if rsu.is_in_coverage(vehicle.pos):
                f_serving_rsu = max(getattr(rsu, "cpu_freq", 0.0), 0.0)
        f_candidates_max = 0.0
        for cand in self._get_state_only_v2v_candidates(vehicle):
            f_candidates_max = max(f_candidates_max, getattr(cand, "cpu_freq", 0.0))
        f_max = max(f_local, f_serving_rsu, f_candidates_max, 1e-9)
        return f_max, {
            "f_local": f_local,
            "f_serving_rsu": f_serving_rsu,
            "f_candidates_max": f_candidates_max,
        }

    def _compute_queue_lb_state_only(self, vehicle, candidates, rsu):
        """state-only版本：队列等待下界"""
        waits = []
        freq_self = max(getattr(vehicle, "cpu_freq", 0.0), 1e-9)
        waits.append(self._get_veh_queue_load(vehicle.id) / freq_self)
        if rsu is not None:
            freq_r = max(getattr(rsu, "cpu_freq", 0.0), 1e-9)
            waits.append(self._get_rsu_queue_load(rsu.id) / freq_r)
        for cand in candidates:
            freq_c = max(getattr(cand, "cpu_freq", 0.0), 1e-9)
            waits.append(self._get_veh_queue_load(cand.id) / freq_c)
        if not waits:
            return 0.0
        return float(min(waits))

    def _compute_best_comm_rate_state_only(self, vehicle, candidates, rsu, active_v2i_count, active_v2v_vehicles):
        """state-only版本：基于当前active集合估计最优通信速率"""
        eps_rate = getattr(self.config, "EPS_RATE", 1e-9)
        fixed_power = getattr(self.config, "PBRS_PHI_POWER_DBM", self.config.TX_POWER_MAX_DBM)
        best_rate = 0.0
        if rsu is not None:
            rate = self._compute_pair_rate(
                ("VEH", vehicle.id),
                ("RSU", int(rsu.id)),
                "V2I",
                fixed_power,
                active_v2i_count=active_v2i_count,
                active_v2v_vehicles=active_v2v_vehicles,
            )
            best_rate = max(best_rate, rate)
        for cand in candidates:
            rate = self._compute_pair_rate(
                ("VEH", vehicle.id),
                ("VEH", int(cand.id)),
                "V2V",
                fixed_power,
                active_v2i_count=active_v2i_count,
                active_v2v_vehicles=active_v2v_vehicles,
            )
            best_rate = max(best_rate, rate)
        return float(max(best_rate, eps_rate))

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

    def _compute_phi_value_state_only(self, dag, vehicle=None):
        """state-only版本：避免action依赖"""
        if vehicle is None:
            return 0.0
        status = getattr(dag, "status", None)
        if status is None:
            return 0.0
        remaining_nodes = [idx for idx, s in enumerate(status) if s != 3]
        if not remaining_nodes or getattr(dag, "is_finished", False):
            return 0.0
        cp_rem, _ = self._compute_cp_stats(dag)
        f_max, f_max_info = self._get_reachable_f_max_state_only(vehicle)
        phi = -(cp_rem / max(f_max, 1e-9)) / max(self.config.T_REF, 1e-9)
        phi = float(np.clip(phi, -self.config.PHI_CLIP, 0.0))
        if getattr(self.config, "DEBUG_PBRS_AUDIT", False):
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

        # [通用化] 从candidate_set动态获取所有可用RSU
        candidate_set = self._last_candidate_set.get(vehicle.id)
        rsu_found = False
        if candidate_set is not None:
            types = candidate_set.get("types", [])
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            for idx in range(len(types)):
                if int(types[idx]) == 2 and idx < len(mask) and bool(mask[idx]):
                    rid = int(ids[idx])
                    if 0 <= rid < len(self.rsus):
                        rsu = self.rsus[rid]
                        freq_r = max(getattr(rsu, "cpu_freq", 0.0), 1e-9)
                        waits.append(self._get_rsu_queue_load(rid) / freq_r)
                        rsu_found = True
        if not rsu_found:
            rsu_id = self._last_rsu_choice.get(vehicle.id)
            if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                rsu = self.rsus[rsu_id]
                freq_r = max(getattr(rsu, "cpu_freq", 0.0), 1e-9)
                waits.append(self._get_rsu_queue_load(rsu_id) / freq_r)

        if candidate_set is not None:
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            types = candidate_set.get("types", [])
            # [BugFix] 按types区分RSU/V2V，不再假设index>=2都是V2V
            for idx in range(2, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id < 0:
                        continue
                    cand_type = int(types[idx]) if idx < len(types) else 0
                    if cand_type == 2:  # RSU → 已在上方rsu_id分支处理，跳过避免重复
                        if 0 <= cand_id < len(self.rsus):
                            rsu_obj = self.rsus[cand_id]
                            freq_r = max(getattr(rsu_obj, "cpu_freq", 0.0), 1e-9)
                            waits.append(self._get_rsu_queue_load(cand_id) / freq_r)
                    elif cand_type == 3:  # V2V
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

        # [通用化] 从candidate_set动态获取所有可用RSU的V2I速率
        rsu_checked = False
        if candidate_set is not None:
            types_arr = candidate_set.get("types", [])
            ids_arr = candidate_set.get("ids", [])
            mask_arr = candidate_set.get("mask", [])
            for idx in range(len(types_arr)):
                if int(types_arr[idx]) == 2 and idx < len(mask_arr) and bool(mask_arr[idx]):
                    rid = int(ids_arr[idx])
                    if rid >= 0:
                        try:
                            rate = self._get_rate_from_snapshot(("VEH", vehicle.id), ("RSU", rid), "V2I")
                            best_rate = max(best_rate, rate)
                            rsu_checked = True
                        except Exception:
                            pass
        if not rsu_checked:
            rsu_id = self._last_rsu_choice.get(vehicle.id)
            if rsu_id is not None and rsu_id >= 0:
                try:
                    rate = self._get_rate_from_snapshot(("VEH", vehicle.id), ("RSU", int(rsu_id)), "V2I")
                    best_rate = max(best_rate, rate)
                except Exception:
                    pass

        if candidate_set is not None:
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            types = candidate_set.get("types", [])
            v2v_slots = candidate_set.get("v2v_slots", [])
            v2v_start = int(candidate_set.get("v2v_start_idx", 2))
            # [BugFix] 按types区分RSU/V2V link类型
            for idx in range(2, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id < 0:
                        continue
                    cand_type = int(types[idx]) if idx < len(types) else 0
                    if cand_type == 2:  # RSU → V2I link
                        try:
                            rate = self._get_rate_from_snapshot(("VEH", vehicle.id), ("RSU", cand_id), "V2I")
                            best_rate = max(best_rate, rate)
                        except Exception:
                            pass
                    elif cand_type == 3:  # V2V link
                        try:
                            rate = self._get_rate_from_snapshot(("VEH", vehicle.id), ("VEH", cand_id), "V2V")
                            best_rate = max(best_rate, rate)
                            continue
                        except Exception:
                            pass
                        slot_idx = idx - v2v_start
                        if 0 <= slot_idx < len(v2v_slots) and v2v_slots[slot_idx] is not None:
                            best_rate = max(best_rate, float(v2v_slots[slot_idx].get("rate", 0.0)))

        return float(max(best_rate, eps_rate))

    def _compute_phi_value_v2(self, dag, vehicle=None):
        """PBRS_KP_V2: 基于关键路径下界 LB(s)=compute+comm+queue 的潜势ϕ"""
        if vehicle is None:
            return 0.0, {}
        status = getattr(dag, "status", None)
        if status is None:
            return 0.0, {}
        remaining_nodes = [idx for idx, s in enumerate(status) if s != 3]
        if not remaining_nodes or getattr(dag, "is_finished", False):
            debug = {
                "cp_rem": 0.0,
                "f_max": 0.0,
                "d_cp_lb": 0.0,
                "rate_best": 0.0,
                "comm_lb": 0.0,
                "queue_lb": 0.0,
                "lb": 0.0,
                "phi": 0.0,
            }
            return 0.0, debug
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

    def _compute_lb_snapshot(self, vehicle):
        """
        计算关键路径下界 LB(s)（仅用快照量，不依赖未来）。
        LB = cp_rem_cycles / f_max + cp_data / rate_best + queue_lb
        """
        dag = vehicle.task_dag
        status = getattr(dag, "status", None)
        if status is None or dag.is_finished:
            return 0.0
        remaining = [idx for idx, s in enumerate(status) if s != 3]
        if not remaining:
            return 0.0
        cp_rem, d_cp_lb = self._compute_cp_stats(dag)
        f_max, _ = self._get_reachable_f_max(vehicle)
        rate_best = self._compute_best_comm_rate(vehicle)
        queue_lb = self._compute_queue_lb(vehicle)
        lb = (cp_rem / max(f_max, 1e-9)) + (d_cp_lb / max(rate_best, getattr(self.config, "EPS_RATE", 1e-9))) + queue_lb
        return max(lb, 0.0)

    def _compute_phi_value_v2_state_only(self, dag, vehicle=None):
        """state-only版本：PBRS_KP_V2势函数"""
        if vehicle is None:
            return 0.0, {}
        status = getattr(dag, "status", None)
        if status is None:
            return 0.0, {}
        remaining_nodes = [idx for idx, s in enumerate(status) if s != 3]
        if not remaining_nodes or getattr(dag, "is_finished", False):
            debug = {
                "cp_rem": 0.0,
                "f_max": 0.0,
                "d_cp_lb": 0.0,
                "rate_best": 0.0,
                "comm_lb": 0.0,
                "queue_lb": 0.0,
                "lb": 0.0,
                "phi": 0.0,
            }
            return 0.0, debug
        cp_rem, d_cp_lb = self._compute_cp_stats(dag)
        f_max, f_max_info = self._get_reachable_f_max_state_only(vehicle)
        active_v2i_count, active_v2v_vehicles = self._get_active_tx_sets_from_queues()
        rsu, _ = self._get_serving_rsu(vehicle)
        candidates = self._get_state_only_v2v_candidates(vehicle)
        rate_best = self._compute_best_comm_rate_state_only(
            vehicle, candidates, rsu, active_v2i_count, active_v2v_vehicles
        )
        comm_lb = d_cp_lb / max(rate_best, getattr(self.config, "EPS_RATE", 1e-9))
        queue_lb = self._compute_queue_lb_state_only(vehicle, candidates, rsu)
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

    def _compute_phi_state_only_batch(self, scheme):
        """批量计算state-only Phi（用于PBRS时序与审计）"""
        phi_cache = {}
        debug_cache = {}
        for v in self.vehicles:
            dag = v.task_dag
            if scheme == "PBRS_KP_V2":
                phi, debug = self._compute_phi_value_v2_state_only(dag, vehicle=v)
            else:
                phi = self._compute_phi_value_state_only(dag, vehicle=v)
                debug = {}
            phi_cache[v.id] = phi
            debug_cache[v.id] = debug
        return phi_cache, debug_cache

    def _assert_phi_action_invariant(self, scheme):
        """检查Phi与action无关性（仅用于审计）"""
        tol = 1e-6
        phi_ref, _ = self._compute_phi_state_only_batch(scheme)
        actions_a = self.action_space.sample()
        _ = self._plan_actions_snapshot(actions_a)
        phi_a, _ = self._compute_phi_state_only_batch(scheme)
        actions_b = self.action_space.sample()
        _ = self._plan_actions_snapshot(actions_b)
        phi_b, _ = self._compute_phi_state_only_batch(scheme)
        for vid, phi_val in phi_ref.items():
            phi_a_val = phi_a.get(vid, 0.0)
            phi_b_val = phi_b.get(vid, 0.0)
            if not (np.isfinite(phi_val) and np.isfinite(phi_a_val) and np.isfinite(phi_b_val)):
                raise AssertionError(f"[PBRS] Phi not finite in action-invariant check veh={vid}")
            if abs(phi_val - phi_a_val) > tol or abs(phi_val - phi_b_val) > tol:
                raise AssertionError(
                    f"[PBRS] Phi depends on action veh={vid}: "
                    f"phi_ref={phi_val:.6f}, phi_a={phi_a_val:.6f}, phi_b={phi_b_val:.6f}"
                )

    def _estimate_t_actual(self, vehicle, subtask_idx, target, cycles, power_ratio=1.0):
        """
        估计动作目标的执行时间（使用冻结速率与队列快照）
        
        [V6修复] 加入通信队列等待时间 comm_wait，使 t_est 正确反映并发拥塞
        可通过 USE_COMM_WAIT_IN_EST=False 回退旧口径
        
        完整估计: t_est = comm_wait + t_tx + cpu_wait + t_comp
        """
        freq_self = max(getattr(vehicle, "cpu_freq", self.config.MIN_VEHICLE_CPU_FREQ), 1e-9)
        t_local = self._get_veh_queue_wait_time(vehicle.id, freq_self) + cycles / freq_self
        if target is None or target == 'Local':
            return t_local, 0.0
        eps_rate = getattr(self.config, "EPS_RATE", 1e-9)
        dag = vehicle.task_dag
        din = self._get_upload_bytes(dag, subtask_idx)
        
        # [V6修复] 获取通信队列等待时间（可配置开关）
        use_comm_wait = getattr(self.config, 'USE_COMM_WAIT_IN_EST', True)
        comm_wait_dict = self._compute_comm_wait(vehicle.id) if use_comm_wait else {'total_v2i': 0.0, 'total_v2v': 0.0}

        if self._is_rsu_location(target):
            rsu_id = self._get_rsu_id_from_location(target)
            if rsu_id is not None:
                self._assert_serving_rsu(vehicle, rsu_id, "reward_target")
            dst_node = ("RSU", rsu_id if rsu_id is not None else 0)
            rate = self._get_rate_from_snapshot(("VEH", vehicle.id), dst_node, "V2I")
            rate = max(rate if rate is not None else 0.0, eps_rate)
            t_tx = din / rate if din > 0 else 0.0
            tx_timeout = float(getattr(self.config, "TX_TIMEOUT_SECONDS", 2.0))
            if tx_timeout > 0 and t_tx > tx_timeout:
                t_tx = tx_timeout
            freq_r = self.config.F_RSU
            cpu_wait = 0.0
            if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                freq_r = self.rsus[rsu_id].cpu_freq
                cpu_wait = self._get_rsu_queue_wait_time(rsu_id)
            t_comp = cycles / max(freq_r, 1e-9)
            comm_wait = comm_wait_dict['total_v2i']
            return comm_wait + t_tx + cpu_wait + t_comp, t_tx

        if isinstance(target, int):
            dst_node = ("VEH", target)
            rate = self._get_rate_from_snapshot(("VEH", vehicle.id), dst_node, "V2V")
            rate = max(rate if rate is not None else 0.0, eps_rate)
            t_tx = din / rate if din > 0 else 0.0
            tx_timeout = float(getattr(self.config, "TX_TIMEOUT_SECONDS", 2.0))
            if tx_timeout > 0 and t_tx > tx_timeout:
                t_tx = tx_timeout
            tgt_veh = self._get_vehicle_by_id(target)
            freq_t = getattr(tgt_veh, "cpu_freq", self.config.MIN_VEHICLE_CPU_FREQ) if tgt_veh is not None else self.config.MIN_VEHICLE_CPU_FREQ
            cpu_wait = self._get_veh_queue_wait_time(target, freq_t) if tgt_veh is not None else 0.0
            t_comp = cycles / max(freq_t, 1e-9)
            comm_wait = comm_wait_dict['total_v2v']
            return comm_wait + t_tx + cpu_wait + t_comp, t_tx

        return t_local, 0.0

    def _compute_slack_based_time_reward(self, vehicle, ctx):
        """
        [P0修复] Slack-Based绝对效用时间奖励
        
        原r_lat使用相对优势tanh((t_alt-t_a)/T_REF)导致RSU获得系统性正奖励。
        新设计基于deadline slack，直接与优化目标对齐。
        
        slack = deadline_remaining - t_finish_est
        r_time = W_S * tanh(slack / S0)
        
        优点：
        - 不比较不同动作，消除系统性偏置
        - 直接反映任务完成与deadline的关系
        - 正slack=有余量→正奖励，负slack=可能超时→负惩罚
        """
        subtask_idx = ctx.get("subtask")
        if subtask_idx is None or ctx.get("illegal"):
            return 0.0, {}
        
        t_a = float(ctx.get("t_actual", 0.0))
        if not np.isfinite(t_a):
            t_a = 0.0
        
        dag = vehicle.task_dag
        if dag is None:
            return 0.0, {}
        
        # 计算deadline slack
        elapsed = self.time - dag.start_time
        deadline_rem = dag.deadline - elapsed
        slack = deadline_rem - t_a
        
        # 参数（可配置）
        S0 = getattr(self.config, 'SLACK_S0', 3.0)  # 使tanh饱和合理的标定值
        W_S = getattr(self.config, 'SLACK_WEIGHT', 1.0)
        
        # 计算slack-based奖励
        r_time_base = W_S * float(np.tanh(slack / max(S0, 1e-9)))
        
        # [P0修复v2] 加入队列负载惩罚，解决"并发越高r_time越高"的问题
        # 原因：t_a使用_get_rsu_queue_wait_time(返回最小处理器等待)，不随并发增长
        # 修复：使用_get_rsu_queue_load(返回总负载)计算队列惩罚
        r_queue_penalty = 0.0
        target = ctx.get("target")
        if self._is_rsu_location(target):
            rsu_id = self._get_rsu_id_from_location(target)
            if rsu_id is not None and 0 <= rsu_id < len(self.rsus):
                rsu_load = self._get_rsu_queue_load(rsu_id)
                rsu_limit = getattr(self.config, 'RSU_QUEUE_CYCLES_LIMIT', 1e12)
                load_ratio = float(rsu_load / max(rsu_limit, 1e-9))
                # 队列负载惩罚权重（可配置）
                # [P0v3] 增大权重从0.5到1.5，并降低敏感度使惩罚更线性
                W_Q = getattr(self.config, 'TIME_QUEUE_PENALTY_WEIGHT', 1.5)
                r_queue_penalty = -W_Q * float(np.tanh(load_ratio * 3.0))  # 3.0使惩罚更线性
        
        r_time = r_time_base + r_queue_penalty
        
        # 为了兼容原有代码，同时计算旧的t_L/t_R/t_V用于调试
        cycles = float(ctx.get("cycles", 0.0))
        t_L, _ = self._estimate_t_actual(vehicle, subtask_idx, 'Local', cycles, power_ratio=1.0) if cycles > 0 else (None, 0)
        
        details = {
            "slack": float(slack),
            "deadline_rem": float(deadline_rem),
            "t_a": float(t_a),
            "t_L": float(t_L) if t_L is not None and np.isfinite(t_L) else None,
            "S0": float(S0),
            "r_time_base": float(r_time_base),
            "r_queue_penalty": float(r_queue_penalty),
            "r_time": float(r_time),
            # 兼容旧字段
            "r_lat": float(r_time),
        }
        return r_time, details

    def _compute_latency_advantage(self, vehicle, ctx):
        """
        PBRS_KP_V2: 计算时延奖励
        
        [P0修复] 默认使用slack-based绝对效用替代相对优势
        可通过config.USE_RELATIVE_ADVANTAGE=True切换回原方案（用于对比实验）
        """
        # 新方案：slack-based绝对效用
        if not getattr(self.config, 'USE_RELATIVE_ADVANTAGE', False):
            return self._compute_slack_based_time_reward(vehicle, ctx)
        
        # ===== 以下为原有相对优势方案（保留用于对比） =====
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
        # [通用化] 从candidate_set动态获取所有可用RSU的最优时延
        rsu_found = False
        if candidate_set is not None:
            types_arr = candidate_set.get("types", [])
            ids_arr = candidate_set.get("ids", [])
            mask_arr = candidate_set.get("mask", [])
            for idx in range(len(types_arr)):
                if int(types_arr[idx]) == 2 and idx < len(mask_arr) and bool(mask_arr[idx]):
                    rid = int(ids_arr[idx])
                    if rid >= 0:
                        try:
                            t_r, _ = self._estimate_t_actual(vehicle, subtask_idx, ("RSU", rid), cycles, power_ratio=1.0)
                            if t_R is None or t_r < t_R:
                                t_R = t_r
                            rsu_found = True
                        except Exception:
                            pass
        if not rsu_found:
            rsu_id = self._last_rsu_choice.get(vehicle.id)
            if rsu_id is not None and rsu_id >= 0:
                try:
                    t_R, _ = self._estimate_t_actual(vehicle, subtask_idx, ("RSU", int(rsu_id)), cycles, power_ratio=1.0)
                except Exception:
                    t_R = None

        if candidate_set is not None:
            ids = candidate_set.get("ids", [])
            mask = candidate_set.get("mask", [])
            types = candidate_set.get("types", [])
            v2v_slots = candidate_set.get("v2v_slots", [])
            v2v_start = int(candidate_set.get("v2v_start_idx", 2))
            best_t = None
            # [BugFix] 按types区分RSU/V2V，不再假设index>=2都是V2V
            for idx in range(2, len(ids)):
                if idx < len(mask) and bool(mask[idx]):
                    cand_id = int(ids[idx])
                    if cand_id < 0:
                        continue
                    cand_type = int(types[idx]) if idx < len(types) else 0
                    if cand_type == 2:  # RSU → 估算RSU时延
                        try:
                            t_r, _ = self._estimate_t_actual(vehicle, subtask_idx, ("RSU", cand_id), cycles, power_ratio=1.0)
                            if t_R is None or t_r < t_R:
                                t_R = t_r
                        except Exception:
                            pass
                    elif cand_type == 3:  # V2V
                        try:
                            t_v, _ = self._estimate_t_actual(vehicle, subtask_idx, cand_id, cycles, power_ratio=1.0)
                            if best_t is None or t_v < best_t:
                                best_t = t_v
                        except Exception:
                            slot_idx = idx - v2v_start
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

    def _audit_action_type(self, target):
        if target is None or target == "Local":
            return "Local"
        if isinstance(target, tuple) and len(target) > 0 and target[0] == "RSU":
            return "RSU"
        if isinstance(target, int):
            return "V2V"
        return "Other"

    def _audit_energy_lambda_effective(self, scheme):
        if scheme == "PBRS_KP_V2":
            return ("ENERGY_LAMBDA", float(getattr(self.config, "ENERGY_LAMBDA", 0.0)))
        if scheme == "PBRS_KP":
            return ("ENERGY_LAMBDA_PBRS", float(getattr(self.config, "ENERGY_LAMBDA_PBRS", 0.0)))
        return ("DELTA_CFT_ENERGY_WEIGHT", float(getattr(self.config, "DELTA_CFT_ENERGY_WEIGHT", 0.0)))

    def _audit_append_t_est_record(self, record):
        self._audit_t_est_records.append(record)
        if not self._audit_t_est_path:
            return
        try:
            os.makedirs(os.path.dirname(self._audit_t_est_path), exist_ok=True)
            import json
            with open(self._audit_t_est_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def _audit_write_scheme_activation(self, row):
        if self._audit_scheme_activation_written:
            return
        if not self._audit_results_dir:
            return
        os.makedirs(self._audit_results_dir, exist_ok=True)
        path = os.path.join(self._audit_results_dir, "scheme_activation_check.csv")
        header = [
            "run_id",
            "episode",
            "scheme",
            "passed",
            "reason",
            "decision_offload_frac",
            "r_base_nonzero",
            "r_shape_nonzero",
            "r_lat_nonzero",
            "r_lat_abs_mean",
            "comm_lb_nonzero",
            "queue_lb_nonzero",
            "r_energy_nonzero",
            "r_power_nonzero",
            "energy_lambda_effective_name",
            "energy_lambda_effective_value",
            "unified_nonfinite_count",
            "unified_consistency_mismatch_count",
            "unified_illegal_count",
        ]
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        self._audit_scheme_activation_written = True

    def _audit_check_reward_scheme(self, metrics_dict, episode_metrics):
        if not self._audit_results_dir:
            return
        scheme = getattr(self.config, "REWARD_SCHEME", "LEGACY_CFT")

        def _stat(name, field, default=0.0):
            return metrics_dict.get(name, {}).get(field, default)

        r_base_nz = int(_stat("r_base", "nonzero_count", 0))
        r_shape_nz = int(_stat("r_shape", "nonzero_count", 0))
        r_lat_nz = int(_stat("r_lat", "nonzero_count", 0))
        r_lat_abs_mean = float(_stat("r_lat", "abs_mean", 0.0))
        comm_lb_count = int(_stat("comm_lb", "count", 0))
        queue_lb_count = int(_stat("queue_lb", "count", 0))
        comm_lb_nz = int(_stat("comm_lb", "nonzero_count", 0))
        queue_lb_nz = int(_stat("queue_lb", "nonzero_count", 0))
        r_energy_nz = int(_stat("r_energy", "nonzero_count", 0))
        r_power_nz = int(_stat("r_power", "nonzero_count", 0))
        decision_offload_frac = float(
            episode_metrics.get("decision_frac_rsu", 0.0) + episode_metrics.get("decision_frac_v2v", 0.0)
        )

        energy_name, energy_val = self._audit_energy_lambda_effective(scheme)

        errors = []
        unified_nonfinite_count = int(episode_metrics.get("unified_nonfinite_count", 0))
        unified_consistency_mismatch_count = int(episode_metrics.get("unified_consistency_mismatch_count", 0))
        unified_illegal_count = int(
            episode_metrics.get(
                "unified_illegal_trigger_count",
                episode_metrics.get("illegal_count", 0),
            )
        )

        if scheme == "PBRS_KP":
            if max(r_base_nz, r_shape_nz) <= 0:
                errors.append("PBRS_KP missing r_base/r_shape nonzero")
            # NOTE: r_energy=0 is correct - only root uploads incur energy
            # if energy_val > 0.0 and decision_offload_frac > 0.0 and r_energy_nz <= 0:
            #     errors.append("PBRS_KP energy_lambda active but r_energy all zero")
            pass  # placeholder
        elif scheme == "PBRS_KP_V2":
            if r_lat_nz <= 0 or r_lat_abs_mean <= 1e-6:
                errors.append("PBRS_KP_V2 r_lat not active")
            if comm_lb_count <= 0 or queue_lb_count <= 0:
                errors.append("PBRS_KP_V2 comm_lb/queue_lb missing")
            # NOTE: r_energy=0 is correct - only root uploads incur energy
            # if energy_val > 0.0 and decision_offload_frac > 0.0 and r_energy_nz <= 0:
            #     errors.append("PBRS_KP_V2 energy_lambda active but r_energy all zero")
            pass  # placeholder
            if float(getattr(self.config, "POWER_LAMBDA", 0.0)) > 0.0 and decision_offload_frac > 0.0 and r_power_nz <= 0:
                errors.append("PBRS_KP_V2 power_lambda active but r_power all zero")
        elif scheme == "UNIFIED":
            for metric_name in ("reward", "r_step", "r_term", "r_pbrs"):
                stat = metrics_dict.get(metric_name, {})
                for field in ("mean", "min", "max", "abs_mean", "p95"):
                    val = stat.get(field, 0.0)
                    if not np.isfinite(val):
                        errors.append(f"UNIFIED {metric_name}.{field} non-finite")
                        break
            if unified_nonfinite_count > 0:
                errors.append(f"UNIFIED non-finite components={unified_nonfinite_count}")
            if float(getattr(self.config, "PBRS_BETA", 0.0)) > 0.0 and unified_consistency_mismatch_count > 0:
                errors.append(f"UNIFIED pbrs consistency mismatch={unified_consistency_mismatch_count}")
        else:
            if int(_stat("dT_clipped", "nonzero_count", 0)) <= 0:
                errors.append("LEGACY_CFT missing dT_clipped nonzero")

        row = {
            "run_id": self._audit_run_id or "",
            "episode": int(getattr(self, "episode_count", 0)),
            "scheme": scheme,
            "passed": "no" if errors else "yes",
            "reason": ";".join(errors) if errors else "",
            "decision_offload_frac": decision_offload_frac,
            "r_base_nonzero": r_base_nz,
            "r_shape_nonzero": r_shape_nz,
            "r_lat_nonzero": r_lat_nz,
            "r_lat_abs_mean": r_lat_abs_mean,
            "comm_lb_nonzero": comm_lb_nz,
            "queue_lb_nonzero": queue_lb_nz,
            "r_energy_nonzero": r_energy_nz,
            "r_power_nonzero": r_power_nz,
            "energy_lambda_effective_name": energy_name,
            "energy_lambda_effective_value": energy_val,
            "unified_nonfinite_count": unified_nonfinite_count,
            "unified_consistency_mismatch_count": unified_consistency_mismatch_count,
            "unified_illegal_count": unified_illegal_count,
        }
        self._audit_write_scheme_activation(row)
        if errors:
            # Changed: Log warning instead of raising exception to avoid training interruption
            # Use AUDIT_FATAL_ON_SCHEME_ERROR=True to restore old behavior for debugging
            if getattr(self.config, "AUDIT_FATAL_ON_SCHEME_ERROR", False):
                raise RuntimeError(f"[RewardSchemeAudit] {scheme} failed: {row['reason']}")
            else:
                import warnings
                warnings.warn(f"[RewardSchemeAudit] {scheme} warning: {row['reason']}", stacklevel=2)

    def _audit_on_compute_done(self, job, time_now):
        key = (int(job.owner_vehicle_id), int(job.subtask_id))
        if key not in self._audit_subtask_est:
            return
        record = self._audit_subtask_est.pop(key)
        finish_time = job.finish_time if job.finish_time is not None else time_now
        t_real = float(max(finish_time - record.get("decision_time", time_now), 0.0))
        t_est = float(record.get("t_est", 0.0))
        out = {
            "episode": record.get("episode", int(getattr(self, "episode_count", 0))),
            "vehicle_id": record.get("vehicle_id", int(job.owner_vehicle_id)),
            "subtask_id": record.get("subtask_id", int(job.subtask_id)),
            "action_type": record.get("action_type", "Unknown"),
            "decision_time": record.get("decision_time", time_now),
            "finish_time": float(finish_time),
            "t_actual_est": t_est,
            "t_actual_real": t_real,
            "est_error": float(t_real - t_est),
        }
        self._audit_append_t_est_record(out)

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
        [奖励函数辅助] 奖励缩放（兼容旧调用名）

        Args:
            reward: 原始奖励值

        Returns:
            float: 缩放后的奖励值
        """
        reward = float(reward)
        if not np.isfinite(reward):
            return reward
        reward_scale = float(getattr(self.config, "REWARD_SCALE", 1.0))
        if abs(reward_scale) < 1e-12:
            return reward
        return reward / reward_scale

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
                    # [通用化] 按types判断是否有任一RSU可用
                    types_arr = candidate_set.get("types", [])
                    mask_arr = candidate_set.get("mask", [])
                    has_rsu = any(int(types_arr[j]) == 2 and bool(mask_arr[j])
                                 for j in range(len(types_arr)) if j < len(mask_arr))
                    if has_rsu:
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
                
                # 判断action类型（支持RSU选择模式）
                enable_rsu_selection = getattr(self.config, 'ENABLE_RSU_SELECTION', False)
                num_rsu = len(self.rsus)
                rsu_end_idx = (1 + num_rsu) if enable_rsu_selection else 2
                
                if target_idx == 0:
                    action_type = 'local'
                elif 1 <= target_idx < rsu_end_idx:
                    action_type = 'rsu'
                else:
                    action_type = 'v2v'
                
                # [通用化] 直接用target_idx查mask，不假设RSU在特定位置
                mask_was_true = False
                candidate_set = self._last_candidate_set.get(v.id)
                if action_type == 'local':
                    mask_was_true = True  # Local永远可用
                elif candidate_set is not None and target_idx < len(candidate_set["mask"]):
                    mask_was_true = bool(candidate_set["mask"][target_idx])
                elif action_type == 'rsu':
                    mask_was_true = (self._last_rsu_choice.get(v.id) is not None)
                elif action_type == 'v2v':
                    candidates = self._last_candidates.get(v.id, [])
                    v2v_start = int(candidate_set.get("v2v_start_idx", 2)) if candidate_set else 2
                    neighbor_idx = target_idx - v2v_start
                    if 0 <= neighbor_idx < len(candidates):
                        mask_was_true = (candidates[neighbor_idx] is not None and candidates[neighbor_idx] >= 0)
                
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
        success_count = sum([1 for v in self.vehicles 
                             if v.task_dag.is_finished and not v.task_dag.is_failed])
        episode_metrics['episode_vehicle_count'] = episode_vehicle_count
        # success_rate_end = 截断/终止瞬间「当前 DAG 已成功」的车辆占比（快照，非累计）
        episode_metrics['success_rate_end'] = success_count / max(episode_vehicle_count, 1)
        # 任务重生场景：T_SR / V_SR = 本 episode 内成功完成 DAG 数 / 本 episode 内完成 DAG 总数（累计）
        if getattr(self.config, 'TASK_RESPAWN_ON_COMPLETION', False) and getattr(self, '_episode_task_completion_count', 0) > 0:
            comp = max(getattr(self, '_episode_task_completion_count', 0), 1)
            episode_metrics['task_success_count'] = int(getattr(self, '_episode_task_success_count', 0))
            episode_metrics['task_completion_count'] = int(getattr(self, '_episode_task_completion_count', 0))
            episode_metrics['task_success_rate'] = getattr(self, '_episode_task_success_count', 0) / comp
            episode_metrics['vehicle_success_rate'] = episode_metrics['task_success_rate']
        else:
            episode_metrics['task_success_rate'] = success_count / max(episode_vehicle_count, 1)
            episode_metrics['vehicle_success_rate'] = episode_metrics['task_success_rate']
        
        # 子任务成功率：当前所有车辆当前 DAG 的「已完成子任务数/子任务总数」（快照）
        total_subtasks = 0
        completed_subtasks = 0
        for v in self.vehicles:
            total_subtasks += v.task_dag.num_subtasks
            completed_subtasks += np.sum(v.task_dag.status == 3)
        episode_metrics['total_subtasks'] = total_subtasks
        episode_metrics['subtask_success_rate'] = (completed_subtasks / total_subtasks) if total_subtasks > 0 else 0.0
        
        # Deadline miss 率
        # 非重生：截断瞬间当前 DAG 因 deadline 失败车辆数/车辆数
        # 任务重生：本 episode 内因 deadline 失败的 DAG 数 / 本 episode 内完成的 DAG 总数
        if getattr(self.config, 'TASK_RESPAWN_ON_COMPLETION', False) and getattr(self, '_episode_task_completion_count', 0) > 0:
            comp = max(getattr(self, '_episode_task_completion_count', 0), 1)
            episode_metrics['deadline_miss_rate'] = getattr(self, '_episode_task_deadline_fail_count', 0) / comp
            episode_metrics['audit_deadline_misses'] = int(getattr(self, '_episode_task_deadline_fail_count', 0))
        else:
            deadline_miss_count = sum([1 for v in self.vehicles 
                                       if v.task_dag.is_failed and getattr(v.task_dag, 'fail_reason', None) == 'deadline'])
            episode_metrics['deadline_miss_rate'] = deadline_miss_count / max(episode_vehicle_count, 1)
            episode_metrics['audit_deadline_misses'] = deadline_miss_count
        
        # 决策分布
        total_decisions = 0
        if hasattr(self, '_decision_counts'):
            total_decisions = int(sum(self._decision_counts.values())) if self._decision_counts else 0
            den = max(total_decisions, 1)
            episode_metrics['decision_frac_local'] = self._decision_counts.get('local', 0) / den
            episode_metrics['decision_frac_rsu'] = self._decision_counts.get('rsu', 0) / den
            episode_metrics['decision_frac_v2v'] = self._decision_counts.get('v2v', 0) / den
        else:
            den = 1

        # Constraint/event rates: keep metric semantics explicit.
        illegal_count_legacy = int(getattr(self, "_episode_illegal_count", 0))
        no_task_count = int(getattr(self, "_episode_no_task_count", 0))
        unified_illegal_count = int(getattr(self, "_unified_illegal_trigger_count", 0))
        illegal_count_effective = illegal_count_legacy
        hard_trigger_count = int(getattr(self, "_episode_hard_trigger_count", 0))
        if hard_trigger_count <= 0 and getattr(self.config, "REWARD_SCHEME", "LEGACY_CFT") == "UNIFIED":
            hard_trigger_count = unified_illegal_count
        # Use agent-step opportunities as denominator to match unified trigger counting semantics.
        rate_den = max(int(episode_vehicle_count) * max(int(self._episode_steps), 1), 1)

        episode_metrics['illegal_count_effective'] = illegal_count_effective
        episode_metrics['hard_trigger_count'] = hard_trigger_count
        episode_metrics['illegal_action_rate'] = illegal_count_effective / rate_den
        episode_metrics['hard_trigger_rate'] = hard_trigger_count / rate_den
        episode_metrics['no_task_rate'] = no_task_count / rate_den
        episode_metrics['on_task_rate'] = max(0.0, 1.0 - episode_metrics['no_task_rate'])
        # Alias for downstream stats scripts that explicitly look for "has_task_available".
        episode_metrics['has_task_available_rate'] = episode_metrics['on_task_rate']
        episode_metrics['unified_illegal_trigger_rate'] = unified_illegal_count / rate_den
        if self._episode_illegal_reasons:
            top_reason, top_count = max(self._episode_illegal_reasons.items(), key=lambda kv: kv[1])
            episode_metrics["top_illegal_reason"] = str(top_reason)
            episode_metrics["top_illegal_reason_count"] = int(top_count)
        else:
            episode_metrics["top_illegal_reason"] = ""
            episode_metrics["top_illegal_reason_count"] = 0
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
        if hasattr(self, "_episode_candidate_stats"):
            reach = self._episode_candidate_stats.get("reachable", [])
            drop = self._episode_candidate_stats.get("dropped", [])
            episode_metrics["candidate_reachable_cnt_mean"] = float(np.mean(reach)) if reach else 0.0
            episode_metrics["candidate_reachable_cnt_p95"] = float(np.percentile(reach, 95)) if reach else 0.0
            episode_metrics["candidate_dropped_cnt_mean"] = float(np.mean(drop)) if drop else 0.0
            episode_metrics["candidate_dropped_cnt_p95"] = float(np.percentile(drop, 95)) if drop else 0.0
        episode_metrics["not_in_candidate_fallback_cnt"] = int(getattr(self, "_episode_not_in_candidate_fallback_cnt", 0))
        episode_metrics["illegal_by_connectivity_cnt"] = int(getattr(self, "_episode_illegal_by_connectivity_cnt", 0))
        if getattr(self, "_episode_domain_params", None):
            episode_metrics.update(self._episode_domain_params)

        # [Chain] Episode级风险统计
        episode_metrics['chain_tx_total'] = int(getattr(self, "_chain_tx_total", 0))
        episode_metrics['chain_risk_cost_total'] = float(getattr(self, "_chain_risk_cost_total", 0.0))
        chain_steps = int(getattr(self, "_chain_steps", 0))
        if chain_steps > 0:
            episode_metrics['chain_p95_mean'] = float(getattr(self, "_chain_p95_sum", 0.0)) / chain_steps
            episode_metrics['chain_pfail_mean'] = float(getattr(self, "_chain_pfail_sum", 0.0)) / chain_steps
        else:
            episode_metrics['chain_p95_mean'] = 0.0
            episode_metrics['chain_pfail_mean'] = 0.0
        if hasattr(self, "_trust_mgr"):
            trust_stats = self._trust_mgr.get_stats()
            episode_metrics["trust_attempts"] = int(trust_stats.get("trust_attempts", 0))
            episode_metrics["trust_failures"] = int(trust_stats.get("trust_failures", 0))
            episode_metrics["trust_failure_rate"] = float(trust_stats.get("trust_failure_rate", 0.0))
            episode_metrics["trust_retry_count"] = int(trust_stats.get("trust_retry_count", 0))
            episode_metrics["malicious_count"] = int(trust_stats.get("malicious_count", 0))
        if getattr(self.config, "EDGE_RATE_RECOMPUTE_AUDIT", False):
            counts = getattr(self, "_edge_rate_recompute_counts", [])
            deltas = getattr(self, "_edge_rate_delta_records", [])
            episode_metrics['edge_rate_recompute_cnt_mean'] = float(np.mean(counts)) if counts else 0.0
            if deltas:
                episode_metrics['edge_rate_delta_mean'] = float(np.mean(deltas))
                episode_metrics['edge_rate_delta_p95'] = float(np.percentile(deltas, 95))
            else:
                episode_metrics['edge_rate_delta_mean'] = 0.0
                episode_metrics['edge_rate_delta_p95'] = 0.0

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
            tx_cap = float(getattr(self.config, "TX_TIMEOUT_SECONDS", 2.0))
            capped = np.clip(np.asarray(self._episode_t_tx_values, dtype=np.float64), 0.0, max(tx_cap, 1e-6))
            episode_metrics['t_tx_mean'] = float(np.mean(capped))
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

        # Deadline / CP meta (helps reward scale reasoning)
        try:
            ddl_list = [float(v.task_dag.deadline) for v in self.vehicles if getattr(v, "task_dag", None) is not None]
            ddl_list = [x for x in ddl_list if np.isfinite(x) and x > 0]
            episode_metrics["deadline_seconds_mean"] = float(np.mean(ddl_list)) if ddl_list else 0.0
        except Exception:
            episode_metrics["deadline_seconds_mean"] = 0.0
        try:
            gamma_list = [float(getattr(v.task_dag, "deadline_gamma", 0.0) or 0.0) for v in self.vehicles if getattr(v, "task_dag", None) is not None]
            gamma_list = [x for x in gamma_list if np.isfinite(x) and x > 0]
            episode_metrics["deadline_gamma_mean"] = float(np.mean(gamma_list)) if gamma_list else 0.0
        except Exception:
            episode_metrics["deadline_gamma_mean"] = 0.0
        try:
            cp_list = [float(getattr(v.task_dag, "critical_path_cycles", 0.0) or 0.0) for v in self.vehicles if getattr(v, "task_dag", None) is not None]
            cp_list = [x for x in cp_list if np.isfinite(x) and x > 0]
            episode_metrics["critical_path_cycles_mean"] = float(np.mean(cp_list)) if cp_list else 0.0
        except Exception:
            episode_metrics["critical_path_cycles_mean"] = 0.0

        # Interference / trust oracle (episode aggregates)
        if self._episode_I_total_values:
            arr_total = np.array(self._episode_I_total_values, dtype=np.float32)
            episode_metrics["I_total_mean"] = float(np.mean(arr_total))
            episode_metrics["I_total_p50"] = float(np.percentile(arr_total, 50))
            episode_metrics["I_total_p95"] = float(np.percentile(arr_total, 95))
        else:
            episode_metrics["I_total_mean"] = 0.0
            episode_metrics["I_total_p50"] = 0.0
            episode_metrics["I_total_p95"] = 0.0

        if self._episode_I_caused_input_values:
            arr_caused = np.array(self._episode_I_caused_input_values, dtype=np.float32)
            episode_metrics["I_caused_mean"] = float(np.mean(arr_caused))
            episode_metrics["I_caused_p95"] = float(np.percentile(arr_caused, 95))
        else:
            episode_metrics["I_caused_mean"] = 0.0
            episode_metrics["I_caused_p95"] = 0.0
        if self._episode_rho_selected_values:
            arr = np.array(self._episode_rho_selected_values, dtype=np.float32)
            episode_metrics["rho_selected_mean"] = float(np.mean(arr))
            episode_metrics["rho_selected_p10"] = float(np.percentile(arr, 10))
            episode_metrics["rho_selected_p50"] = float(np.percentile(arr, 50))
            episode_metrics["rho_selected_p95"] = float(np.percentile(arr, 95))
            episode_metrics["rho_selected_lt_0p6_rate"] = float(np.mean(arr < 0.6))
            episode_metrics["rho_selected_lt_0p7_rate"] = float(np.mean(arr < 0.7))
        else:
            episode_metrics["rho_selected_mean"] = 0.0
            episode_metrics["rho_selected_p10"] = 0.0
            episode_metrics["rho_selected_p50"] = 0.0
            episode_metrics["rho_selected_p95"] = 0.0
            episode_metrics["rho_selected_lt_0p6_rate"] = 0.0
            episode_metrics["rho_selected_lt_0p7_rate"] = 0.0
        if self._episode_uncertainty_selected_values:
            arr = np.array(self._episode_uncertainty_selected_values, dtype=np.float32)
            episode_metrics["uncertainty_selected_mean"] = float(np.mean(arr))
            episode_metrics["uncertainty_selected_p90"] = float(np.percentile(arr, 90))
        else:
            episode_metrics["uncertainty_selected_mean"] = 0.0
            episode_metrics["uncertainty_selected_p90"] = 0.0
        if self._episode_risk_penalty_values:
            arr = np.array(self._episode_risk_penalty_values, dtype=np.float32)
            episode_metrics["risk_penalty_mean"] = float(np.mean(arr))
        else:
            episode_metrics["risk_penalty_mean"] = 0.0

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
        episode_metrics['no_task_reasons'] = dict(getattr(self, "_episode_no_task_reasons", {}))
        episode_metrics['unified_nonfinite_count'] = int(getattr(self, "_unified_nonfinite_count", 0))
        episode_metrics['unified_consistency_mismatch_count'] = int(getattr(self, "_unified_consistency_mismatch_count", 0))
        episode_metrics['unified_illegal_trigger_count'] = int(getattr(self, "_unified_illegal_trigger_count", 0))

        # 从reward_stats提取统计信息
        metrics_dict = {}
        for name, bucket in self._reward_stats.metrics.items():
            metrics_dict[name] = {
                "mean": (bucket.sum / bucket.count) if bucket.count > 0 else 0.0,
                "min": bucket.min if bucket.count > 0 else 0.0,
                "max": bucket.max if bucket.count > 0 else 0.0,
                "abs_mean": (bucket.abs_sum / bucket.count) if bucket.count > 0 else 0.0,
                "p95": bucket.p95(),
                "count": bucket.count,
                "nonzero_count": bucket.nonzero_count,
            }

        scheme = getattr(self.config, "REWARD_SCHEME", "LEGACY_CFT")
        if scheme == "PBRS_KP":
            required_metrics = [
                "r_base", "r_shape", "r_term", "r_illegal", "r_timeout", "r_energy", "r_total",
            ]
        elif scheme == "PBRS_KP_V2":
            required_metrics = [
                "r_lat", "comm_lb", "queue_lb", "lb", "r_shape", "r_timeout", "r_energy", "r_power", "r_total",
            ]
        elif scheme == "UNIFIED":
            required_metrics = [
                "reward", "r_step", "r_term", "r_pbrs",
            ]
        else:
            required_metrics = [
                "reward", "dT_clipped", "energy_norm",
            ]
        for name in required_metrics:
            if name not in metrics_dict:
                metrics_dict[name] = {
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "abs_mean": 0.0,
                    "p95": 0.0,
                    "count": 0,
                    "nonzero_count": 0,
                }

        # 补齐训练端常用时间/进度字段，避免env_stats存在但字段缺失时回退失败。
        def _metric_mean(name, default=0.0):
            stat = metrics_dict.get(name)
            if not isinstance(stat, dict):
                return default
            val = stat.get("mean")
            return float(val) if val is not None else default

        def _metric_p95(name, default=0.0):
            stat = metrics_dict.get(name)
            if not isinstance(stat, dict):
                return default
            val = stat.get("p95")
            return float(val) if val is not None else default

        episode_metrics["dT_mean"] = _metric_mean("delta_cft", 0.0)
        episode_metrics["cft_prev_rem_mean"] = _metric_mean("cft_prev_rem", 0.0)
        episode_metrics["cft_curr_rem_mean"] = _metric_mean("cft_curr_rem", 0.0)
        episode_metrics["dt_used_mean"] = _metric_mean("dt_used", 0.0)
        episode_metrics["implied_dt_mean"] = _metric_mean("implied_dt", episode_metrics["dt_used_mean"])
        episode_metrics["dCFT_abs_mean"] = _metric_mean("delta_cft_abs", 0.0)   # 真实绝对CFT差≈-DT
        episode_metrics["dCFT_abs_p95"] = _metric_p95("delta_cft_abs", 0.0)
        episode_metrics["dCFT_rem_mean"] = _metric_mean("delta_cft_rem", 0.0)   # 剩余CFT差≈+DT
        episode_metrics["dCFT_rem_p95"] = _metric_p95("delta_cft_rem", 0.0)
        episode_metrics["dCFT_prog_mean"] = _metric_mean("delta_cft_prog", 0.0) # r_prog使用的信号
        episode_metrics["dCFT_prog_p95"] = _metric_p95("delta_cft_prog", 0.0)
        episode_metrics["r_prog_mean"] = _metric_mean("r_prog", 0.0)
        episode_metrics["reward_step_p95"] = _metric_p95("r_step", _metric_p95("reward_step", 0.0))
        # dT_eff 优先沿用episode聚合，缺失时回退到reward_stats
        if (not np.isfinite(episode_metrics.get("dT_eff_mean", np.nan))) or (
            abs(float(episode_metrics.get("dT_eff_mean", 0.0))) < 1e-12
            and metrics_dict.get("dT_eff", {}).get("count", 0) > 0
        ):
            episode_metrics["dT_eff_mean"] = _metric_mean("dT_eff", 0.0)
        if (not np.isfinite(episode_metrics.get("dT_eff_p95", np.nan))) or (
            abs(float(episode_metrics.get("dT_eff_p95", 0.0))) < 1e-12
            and metrics_dict.get("dT_eff", {}).get("count", 0) > 0
        ):
            episode_metrics["dT_eff_p95"] = _metric_p95("dT_eff", 0.0)

        # 保存到实例变量（供train.py使用）
        self._last_episode_metrics = episode_metrics.copy()
        
        # 只在episode结束时写入JSONL文件
        if terminated or truncated:
            self._audit_check_reward_scheme(metrics_dict, episode_metrics)
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
    
