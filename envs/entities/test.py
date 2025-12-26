import gymnasium as gym
import numpy as np
from configs.config import SystemConfig as Cfg
from envs.modules.channel import ChannelModel
from envs.entities.vehicle import Vehicle
from envs.entities.task_dag import DAGTask
from utils.dag_generator import DAGGenerator


class VecOffloadingEnv(gym.Env):
    """
    [VecOffloadingEnv] - Production Ready
    已修复：队列重复计数、溢出释放错误、死循环、缓存失效逻辑、功率初始化。
    """

    def __init__(self):
        self.channel = ChannelModel()
        self.dag_gen = DAGGenerator()
        self.vehicles = []
        self.time = 0.0

        # 队列状态追踪
        self.rsu_queue_curr = 0
        self.last_global_cft = 0.0
        self.steps = 0  # 步数追踪

        # 缓存系统
        self._comm_rate_cache = {}
        self._cache_time_step = -1.0
        self._cft_cache = None
        self._cft_cache_time = 0.0
        self._cft_cache_valid = False

    def reset(self, seed=None, options=None):
        if seed:
            np.random.seed(seed)

        self.vehicles = []
        self.time = 0.0
        self.steps = 0
        self.rsu_queue_curr = 0

        # 清空缓存
        self._comm_rate_cache = {}
        self._cache_time_step = -1.0
        self._cft_cache = None
        self._cft_cache_valid = False

        for i in range(Cfg.NUM_VEHICLES):
            pos = np.random.rand(2) * Cfg.MAP_SIZE
            v = Vehicle(i, pos)
            v.cpu_freq = np.random.uniform(Cfg.MIN_VEHICLE_CPU_FREQ, Cfg.MAX_VEHICLE_CPU_FREQ)

            # [修复 #4] 初始化功率，防止首次 observation 报错
            v.tx_power_dbm = Cfg.TX_POWER_DEFAULT_DBM if hasattr(Cfg, 'TX_POWER_DEFAULT_DBM') else Cfg.TX_POWER_MIN_DBM

            # DAG 生成
            n_node = np.random.randint(Cfg.MIN_NODES, Cfg.MAX_NODES + 1)
            adj, prof, data, ddl = self.dag_gen.generate(n_node, veh_f=v.cpu_freq)
            v.task_dag = DAGTask(0, adj, prof, data, ddl)
            v.task_dag.start_time = 0.0
            v.task_queue_len = 0
            v.vel = (np.random.rand(2) - 0.5) * 2 * Cfg.MAX_VELOCITY

            # [新增] 状态追踪，防止重复入队
            # 记录上一次处理的子任务ID，用于判断是否是"新任务"
            v.last_scheduled_subtask = -1

            self.vehicles.append(v)

        self.last_global_cft = self._calculate_global_cft_critical_path()
        return self._get_obs(), {}

    def step(self, actions):
        self.steps += 1

        # =========================================================================
        # [修复 #3] 缓存清理 - 使用阈值防止浮点误差导致失效
        # =========================================================================
        if abs(self.time - self._cache_time_step) > 1e-6:
            self._comm_rate_cache.clear()
            self._cache_time_step = self.time

        self._cft_cache = None
        self._cft_cache_valid = False

        # =========================================================================
        # 1. 动作执行 (仅在任务切换时处理队列)
        # =========================================================================
        for i, v in enumerate(self.vehicles):
            act = actions[i]
            if not act: continue

            # 解析动作
            target_idx = int(act['target'])
            subtask_idx = int(act['subtask'])

            # 目标解析
            if target_idx == 0:
                target = 'Local'
            elif target_idx == 1:
                target = 'RSU'
            else:
                neighbor_id = target_idx - 2
                target = neighbor_id if 0 <= neighbor_id < len(self.vehicles) else 'Local'

            # 功率控制
            p_norm = np.clip(act.get('power', 1.0), 0.0, 1.0)
            v.tx_power_dbm = Cfg.TX_POWER_MIN_DBM + p_norm * (Cfg.TX_POWER_MAX_DBM - Cfg.TX_POWER_MIN_DBM)

            # [修复 #1] 核心队列逻辑：仅当这是一个新的调度决策时，才进行溢出检查和入队
            # 强化学习每个 step 都会输出动作，但我们不能每个 step 都把同一个任务重复加入队列
            if subtask_idx != v.last_scheduled_subtask:

                actual_target = target

                # 溢出回退检查 (Overflow Check)
                if target == 'RSU':
                    if self.rsu_queue_curr >= Cfg.RSU_QUEUE_LIMIT:
                        actual_target = 'Local'
                elif isinstance(target, int):
                    t_veh = self.vehicles[target]
                    if t_veh.task_queue_len >= Cfg.VEHICLE_QUEUE_LIMIT:
                        actual_target = 'Local'

                # 应用新的调度
                v.curr_target = actual_target
                v.curr_subtask = subtask_idx
                v.task_dag.assign_task(subtask_idx, actual_target)
                v.last_scheduled_subtask = subtask_idx  # 标记已处理

                # 仅在此刻增加队列计数
                if actual_target == 'RSU':
                    self.rsu_queue_curr += 1
                elif isinstance(actual_target, int):
                    self.vehicles[actual_target].task_queue_len += 1
                elif actual_target == 'Local':
                    v.task_queue_len += 1

            else:
                # 如果是正在执行的任务，保持原有的 target，不要因为 Agent 的随机探索而改变
                # 除非我们支持"任务迁移"(Task Migration)，否则一旦开始执行就锁定位置
                pass

        # 物理硬截断保险 (依然保留，防止数值漂移)
        for v_check in self.vehicles:
            v_check.task_queue_len = min(v_check.task_queue_len, Cfg.VEHICLE_QUEUE_LIMIT)
        self.rsu_queue_curr = min(self.rsu_queue_curr, Cfg.RSU_QUEUE_LIMIT)

        # =========================================================================
        # 2. 物理演化 & 任务进度
        # =========================================================================
        rates_v2i = self.channel.compute_rates(self.vehicles, Cfg.RSU_POS)

        for v in self.vehicles:
            c_spd = 0.0
            comp_spd = 0.0
            tgt = v.curr_target

            if tgt == 'Local':
                comp_spd = v.cpu_freq
                c_spd = 1e12
            elif tgt == 'RSU':
                comp_spd = Cfg.F_RSU
                c_spd = rates_v2i.get(v.id, 1e-6)
            elif isinstance(tgt, int):
                target_veh = self.vehicles[tgt]
                comp_spd = target_veh.cpu_freq
                c_spd = self.channel.compute_one_rate(v, target_veh.pos, 'V2V', self.time)

            c_spd = max(c_spd, 1e-6)

            # 更新任务进度
            task_finished = False
            if v.curr_subtask is not None:
                task_finished = v.task_dag.step_progress(v.curr_subtask, comp_spd, c_spd, Cfg.DT)

                if task_finished:
                    # [修复 #1] 释放队列：释放的是 v.curr_target，即任务实际所在的位置
                    # 由于我们上面锁定了 curr_target，这里释放的一定是正确的队列
                    if tgt == 'RSU':
                        self.rsu_queue_curr = max(0, self.rsu_queue_curr - 1)
                    elif isinstance(tgt, int):
                        target_veh = self.vehicles[tgt]
                        target_veh.task_queue_len = max(0, target_veh.task_queue_len - 1)
                    elif tgt == 'Local':
                        v.task_queue_len = max(0, v.task_queue_len - 1)

                    v.curr_subtask = None
                    # 注意：v.last_scheduled_subtask 不需要重置，
                    # 因为下一个任务的 ID 一定不同，会触发新的入队逻辑

            v.update_pos(Cfg.DT, Cfg.MAP_SIZE)

        self.time += Cfg.DT

        # =========================================================================
        # 3. 奖励与终止
        # =========================================================================
        current_global_cft = self._calculate_global_cft_critical_path()
        cft_diff = self.last_global_cft - current_global_cft

        # 奖励计算 (可根据建议 #10 优化分配机制，这里暂时保持 Cooperative)
        global_reward = cft_diff * Cfg.REWARD_SCALE
        avg_reward = global_reward / len(self.vehicles)
        rewards = [avg_reward] * len(self.vehicles)

        self.last_global_cft = current_global_cft

        # [修复 #2] 终止条件：任务全部完成 OR 超时
        all_finished = all(v.task_dag.is_finished for v in self.vehicles)
        # 假设 Config 中有 MAX_STEPS，通常设为 100-200
        time_limit_reached = self.steps >= Cfg.MAX_STEPS

        done = all_finished or time_limit_reached

        # Info 中可以返回是否是因为超时结束
        info = {'timeout': time_limit_reached} if time_limit_reached else {}

        return self._get_obs(), rewards, done, info

    # _get_inter_task_comm_rate 和 _calculate_global_cft_critical_path
    # 保持之前确认的"物理完备版"逻辑不变，直接复用即可。
    # ...