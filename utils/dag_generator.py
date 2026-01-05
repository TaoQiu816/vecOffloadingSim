import numpy as np
import os
import networkx as nx
from configs.config import SystemConfig as Cfg

# 导入daggen库（如果可用）
try:
    import daggen
except ImportError:
    daggen = None
    if os.environ.get("DAGGEN_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
        print("[Warning] 'daggen' library not found. Using simple fallback generator.")


class DAGGenerator:
    """
    DAG 任务生成器
    
    功能：
    - 根据配置参数生成随机DAG任务
    - 计算相对截止时间（Deadline）
    - 确保生成的DAG结构合理（至少有一个入口和出口节点）
    
    核心原则：
    - 只负责"生"任务，不负责"做"任务（调度逻辑在环境层）
    - Deadline计算基于理想本地执行时间，可通过γ控制紧/松程度
    """
    
    def __init__(self):
        """
        初始化DAG生成器
        
        从SystemConfig中读取参数：
        - 计算量范围（MIN_COMP, MAX_COMP）
        - 数据量范围（MIN_DATA, MAX_DATA）
        - 边数据量范围（MIN_EDGE_DATA, MAX_EDGE_DATA）
        - DAG拓扑参数（DAG_FAT, DAG_DENSITY, DAG_REGULAR, DAG_CCR）
        - Deadline因子范围（DEADLINE_TIGHTENING_MIN, DEADLINE_TIGHTENING_MAX）
        """
        # 任务属性范围
        self.comp_range = (Cfg.MIN_COMP, Cfg.MAX_COMP)  # 计算量范围 (Cycles)
        self.input_range = (Cfg.MIN_DATA, Cfg.MAX_DATA)  # 节点输入数据量 (Bits)
        
        # 任务间传输数据量范围（如果未定义则回退到节点输入数据量）
        min_edge = getattr(Cfg, 'MIN_EDGE_DATA', Cfg.MIN_DATA)
        max_edge = getattr(Cfg, 'MAX_EDGE_DATA', Cfg.MAX_DATA)
        self.edge_data_range = (min_edge, max_edge)

        # DAG拓扑参数
        self.fat = getattr(Cfg, 'DAG_FAT', 0.5)
        self.density = getattr(Cfg, 'DAG_DENSITY', 0.5)
        self.regular = getattr(Cfg, 'DAG_REGULAR', 0.5)
        self.ccr = getattr(Cfg, 'DAG_CCR', 0.5)

    def generate(self, num_nodes, veh_f=None):
        """
        生成单个DAG任务实例
        
        Args:
            num_nodes: 节点数
            veh_f: 车辆CPU频率(Hz)，用于Deadline计算。如果为None则使用配置的最小值
        
        Returns:
            adj_matrix: 邻接矩阵
            profiles: 节点属性列表
            data_matrix: 边数据传输矩阵
            deadline: 相对截止时间(秒)
        """
        if num_nodes <= 0:
            return None, [], None, 0

        # 初始化矩阵
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        data_matrix = np.zeros((num_nodes, num_nodes))  # Bits

        use_fallback = True

        # --- 1. 拓扑生成 (daggen) ---
        if daggen is not None:
            try:
                seed = np.random.randint(0, 100000)
                dag = daggen.DAG(seed=seed, n=num_nodes,
                                 fat=self.fat, density=self.density,
                                 regular=self.regular, ccr=self.ccr)

                _, raw_edges = dag.task_n_edge_dicts()

                for edge in raw_edges:
                    u = edge.get('u', edge.get('src'))
                    v = edge.get('v', edge.get('dst'))

                    if u is not None and v is not None:
                        # 索引修正
                        if u >= num_nodes or v >= num_nodes:
                            u_idx, v_idx = u - 1, v - 1
                        else:
                            u_idx, v_idx = u, v

                        if 0 <= u_idx < num_nodes and 0 <= v_idx < num_nodes:
                            if u_idx != v_idx:
                                adj_matrix[u_idx, v_idx] = 1
                                data_matrix[u_idx, v_idx] = np.random.uniform(*self.edge_data_range)
                                use_fallback = False
            except Exception as e:
                use_fallback = True

        # Fallback生成策略（当daggen不可用时）
        if use_fallback:
            # 根据CCR调整边密度：CCR越高，通信越多，需要更多边
            ccr_factor = np.clip(self.ccr, 0.2, 2.0)
            edge_prob = min(0.7, 0.2 + 0.3 * ccr_factor)
            
            for i in range(num_nodes - 1):
                adj_matrix[i, i + 1] = 1
                edge_data = np.random.uniform(*self.edge_data_range) * ccr_factor
                data_matrix[i, i + 1] = edge_data

                if i + 2 < num_nodes and np.random.rand() < edge_prob:
                    target = np.random.randint(i + 2, num_nodes)
                    adj_matrix[i, target] = 1
                    edge_data = np.random.uniform(*self.edge_data_range) * ccr_factor
                    data_matrix[i, target] = edge_data

        # 生成节点属性
        profiles = []
        in_degrees = np.sum(adj_matrix, axis=0)

        for i in range(num_nodes):
            comp = np.random.uniform(*self.comp_range)
            # 只有入度为0的节点（入口节点）才有输入数据
            is_entry = (in_degrees[i] == 0)
            inp_d = np.random.uniform(*self.input_range) if is_entry else 0.0

            profiles.append({
                'comp': comp,       # 计算量 (Cycles)
                'input_data': inp_d # 输入数据量 (Bits)
            })

        # 计算相对截止时间
        deadline, gamma, critical_path_cycles, base_time = self._calc_deadline(num_nodes, adj_matrix, profiles, veh_f)

        extras = {
            "deadline_gamma": gamma,
            "critical_path_cycles": critical_path_cycles,
            "deadline_base_time": base_time,
            "deadline_seconds": deadline,
            "deadline_slack": getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0),
        }

        return adj_matrix, profiles, data_matrix, deadline, extras

    def _calc_deadline(self, n, adj, profiles, f_base=None):
        """
        计算相对截止时间（支持3种模式）
        
        模式1 (TOTAL_MEDIAN): deadline = γ × (total_comp / f_median)
          - total_comp: 所有子任务计算量之和
          - f_median: (F_MIN + F_MAX) / 2 平均算力
          - γ: 松紧因子 [DEADLINE_TIGHTENING_MIN, DEADLINE_TIGHTENING_MAX]
        
        模式2 (TOTAL_LOCAL): deadline = γ × (total_comp / f_local)
          - f_local: 任务所属车辆的实际CPU频率
          - 考虑车辆本地队列时间（队列在仿真执行时自动计入）
        
        模式3 (FIXED_RANGE): deadline ∈ [DEADLINE_FIXED_MIN, DEADLINE_FIXED_MAX]
          - 直接从固定范围随机，与计算量无关
        
        Args:
            n: 节点数
            adj: 邻接矩阵
            profiles: 节点属性列表
            f_base: 本地CPU频率(Hz)，仅模式2使用
        
        Returns:
            tuple: (deadline_seconds, gamma, total_cycles, base_time)
        """
        # 计算总计算量
        comp_arr = np.array([p['comp'] for p in profiles], dtype=float)
        total_cycles = np.sum(comp_arr)
        
        if total_cycles <= 0 or not np.isfinite(total_cycles):
            total_cycles = Cfg.MEAN_COMP_LOAD * n
        
        # 获取deadline模式
        mode = getattr(Cfg, 'DEADLINE_MODE', 'TOTAL_MEDIAN')
        
        if mode == 'FIXED_RANGE':
            # 模式3: 固定范围直接随机
            d_min = getattr(Cfg, 'DEADLINE_FIXED_MIN', 2.0)
            d_max = getattr(Cfg, 'DEADLINE_FIXED_MAX', 5.0)
            deadline = np.random.uniform(d_min, d_max)
            gamma = deadline / (total_cycles / Cfg.MIN_VEHICLE_CPU_FREQ)  # 反推gamma（仅供记录）
            base_time = total_cycles / Cfg.MIN_VEHICLE_CPU_FREQ
            
        elif mode == 'TOTAL_LOCAL':
            # 模式2: 使用本地CPU频率
            if f_base is None or f_base <= 0:
                f_base = (Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ) / 2.0
            base_time = total_cycles / f_base
            
            gamma_min = getattr(Cfg, 'DEADLINE_TIGHTENING_MIN', 4.0)
            gamma_max = getattr(Cfg, 'DEADLINE_TIGHTENING_MAX', 7.0)
            gamma_min = max(0.1, gamma_min)
            gamma_max = max(gamma_min, gamma_max)
            gamma = np.random.uniform(gamma_min, gamma_max)
            
            slack = max(0.0, getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0))
            deadline = gamma * base_time + slack
            
        else:  # 'TOTAL_MEDIAN' (默认)
            # 模式1: 使用平均算力（中位数）
            f_median = (Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ) / 2.0
            base_time = total_cycles / f_median
            
            gamma_min = getattr(Cfg, 'DEADLINE_TIGHTENING_MIN', 4.0)
            gamma_max = getattr(Cfg, 'DEADLINE_TIGHTENING_MAX', 7.0)
            gamma_min = max(0.1, gamma_min)
            gamma_max = max(gamma_min, gamma_max)
            gamma = np.random.uniform(gamma_min, gamma_max)
            
            slack = max(0.0, getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0))
            deadline = gamma * base_time + slack
        
        # 安全检查
        if not np.isfinite(base_time) or base_time <= 0:
            base_time = Cfg.MIN_COMP / Cfg.MIN_VEHICLE_CPU_FREQ
        
        if not np.isfinite(deadline) or deadline <= 0:
            deadline = 0.1
        else:
            deadline = max(deadline, 0.1)

        return float(deadline), float(gamma), float(total_cycles), float(base_time)

    @staticmethod
    def _critical_path_cycles(adj, comp_arr):
        n = len(comp_arr)
        if n == 0:
            return 0.0
        indeg = np.sum(adj, axis=0)
        order = []
        queue = [i for i in range(n) if indeg[i] == 0]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in np.where(adj[u] == 1)[0]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(int(v))
        if len(order) != n:
            return float(np.sum(comp_arr))
        dp = np.zeros(n, dtype=float)
        for u in order:
            preds = np.where(adj[:, u] == 1)[0]
            if len(preds) == 0:
                dp[u] = comp_arr[u]
            else:
                dp[u] = comp_arr[u] + np.max(dp[preds])
        return float(np.max(dp))
