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
        计算相对截止时间（关键路径版本）
        
        公式: T_deadline = γ × (critical_path_cycles / f_local) + slack_seconds
        - critical_path_cycles: DAG关键路径计算量 (Cycles)
        - f_local: 本地CPU频率 (Hz)
        - γ: 截止时间因子（松紧因子），范围 [gamma_min, gamma_max]
        - slack_seconds: 固定松弛时间
        
        设计原则:
        - 基于关键路径执行时间（不考虑排队）
        - γ < 1.0 强制卸载，γ > 1.0 放宽Deadline
        
        Args:
            n: 节点数
            adj: 邻接矩阵
            profiles: 节点属性列表
            f_base: 本地CPU频率(Hz)，None则使用配置最小值
        
        Returns:
            tuple: (deadline_seconds, gamma, critical_path_cycles, base_time)
        """
        # 获取本地算力
        if f_base is None or f_base <= 0:
            f_base = Cfg.MIN_VEHICLE_CPU_FREQ

        # 计算关键路径计算量
        comp_arr = np.array([p['comp'] for p in profiles], dtype=float)
        critical_path = self._critical_path_cycles(adj, comp_arr)
        if critical_path <= 0 or not np.isfinite(critical_path):
            critical_path = max(Cfg.MIN_COMP, np.sum(comp_arr))
        
        # 基准时间 = 关键路径 / 本地CPU
        base_time = critical_path / f_base
        if not np.isfinite(base_time) or base_time <= 0:
            base_time = Cfg.MIN_COMP / max(f_base, Cfg.MIN_VEHICLE_CPU_FREQ)

        # 获取截止时间因子范围
        gamma_min = getattr(Cfg, 'DEADLINE_TIGHTENING_MIN', 0.70)
        gamma_max = getattr(Cfg, 'DEADLINE_TIGHTENING_MAX', 0.80)
        # [修复] 移除0.99上限约束，允许gamma>1.0以放宽deadline
        gamma_min = max(0.1, gamma_min)
        gamma_max = max(gamma_min, gamma_max)
        gamma = np.random.uniform(gamma_min, gamma_max)
        slack = max(0.0, getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0))

        # 计算截止时间
        deadline = gamma * base_time + slack
        
        # 安全检查：确保deadline是有效的正数
        if not np.isfinite(deadline) or deadline <= 0:
            deadline = 0.1 + slack
        else:
            deadline = max(deadline, 0.1)

        return float(deadline), float(gamma), float(critical_path), float(base_time)

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
