import numpy as np
import networkx as nx
from configs.config import SystemConfig as Cfg

# 尝试导入 daggen，增加代码的鲁棒性
try:
    import daggen
except ImportError:
    daggen = None
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
    - Deadline计算基于理想本地执行时间，确保本地计算无法满足
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
        # --- 1. 任务属性范围 ---
        # 计算量范围 (Cycles)
        self.comp_range = (Cfg.MIN_COMP, Cfg.MAX_COMP)

        # [关键修改] 区分 Edge Data 和 Node Input Data
        # 节点输入数据量 (Node Input)
        self.input_range = (Cfg.MIN_DATA, Cfg.MAX_DATA)

        # 任务间传输数据量 (Edge Data) -> 对应 Config 中的 EDGE_DATA
        # 如果 Config 里没定义这两个变量，回退到 DATA
        min_edge = getattr(Cfg, 'MIN_EDGE_DATA', Cfg.MIN_DATA)
        max_edge = getattr(Cfg, 'MAX_EDGE_DATA', Cfg.MAX_DATA)
        self.edge_data_range = (min_edge, max_edge)

        # --- 2. DAG 拓扑参数 ---
        self.fat = getattr(Cfg, 'DAG_FAT', 0.5)
        self.density = getattr(Cfg, 'DAG_DENSITY', 0.5)
        self.regular = getattr(Cfg, 'DAG_REGULAR', 0.5)
        self.ccr = getattr(Cfg, 'DAG_CCR', 0.5)

    # --- [修改部分] ---
    # 为 generate 增加 veh_f 参数，默认值为 None。如果为 None 则使用 Cfg 中的全局基准频率
    def generate(self, num_nodes, veh_f=None):
        """
        生成单个 DAG 任务实例
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
                                # [修改] 使用 edge_data_range
                                data_matrix[u_idx, v_idx] = np.random.uniform(*self.edge_data_range)
                                use_fallback = False
            except Exception as e:
                use_fallback = True

        # --- 2. Fallback 生成策略 ---
        if use_fallback:
            base_edges = num_nodes - 1  # 最小边数（链式结构）
            max_extra = num_nodes * (num_nodes - 1) // 2 - base_edges  # 最大额外边数
            
            # [关键修改] 根据 CCR 调整边密度
            # CCR 越高，通信越多，需要更多的边来传输数据
            ccr_factor = np.clip(self.ccr, 0.2, 2.0) / 1.0  # 归一化到 [0.2, 2.0] 范围
            edge_prob = min(0.7, 0.2 + 0.3 * ccr_factor)  # 边密度随 CCR 增加
            
            for i in range(num_nodes - 1):
                adj_matrix[i, i + 1] = 1
                edge_data = np.random.uniform(*self.edge_data_range) * ccr_factor
                data_matrix[i, i + 1] = edge_data

                if i + 2 < num_nodes and np.random.rand() < edge_prob:
                    target = np.random.randint(i + 2, num_nodes)
                    adj_matrix[i, target] = 1
                    edge_data = np.random.uniform(*self.edge_data_range) * ccr_factor
                    data_matrix[i, target] = edge_data

        # --- 3. 生成节点属性 (Profiles) ---
        profiles = []
        in_degrees = np.sum(adj_matrix, axis=0)

        for i in range(num_nodes):
            comp = np.random.uniform(*self.comp_range)
            # 只有入度为 0 的节点才有输入数据
            is_entry = (in_degrees[i] == 0)
            inp_d = np.random.uniform(*self.input_range) if is_entry else 0.0

            profiles.append({
                'comp': comp,       # Cycles
                'input_data': inp_d # Bits (Initial Input)
            })

        # --- 4. 计算 Deadline ---
        deadline = self._calc_deadline(num_nodes, adj_matrix, profiles, veh_f)

        return adj_matrix, profiles, data_matrix, deadline

    def _calc_deadline(self, n, adj, profiles, f_base=None):
        """
        [Deadline计算 - 相对截止时间]
        
        计算公式: T_deadline = γ × (W_total / f_local)
        - W_total: DAG所有子任务的计算量之和 (Cycles)
        - f_local: 车辆本地CPU频率 (Hz)
        - γ: 截止时间因子（紧缩因子），可调整范围 [gamma_min, gamma_max]
        
        设计原则:
        - "Deadline是铁律，只看硬件能力，不听任何借口（不看排队）"
        - 明确排除本地排队时延，确保"本地计算必死"
        - γ < 1.0 确保本地执行时间必然大于Deadline，强迫卸载
        
        注意：此函数只负责"生"任务，不负责"做"任务（调度逻辑在环境层）

        Args:
            n: 节点数
            adj: 邻接矩阵
            profiles: 节点属性列表 [{'comp': cycles, 'input_data': bits}, ...]
            f_base: 车辆本地CPU频率 (Hz)，如果为None则使用Config基准

        Returns:
            float: Deadline时间 (秒)，保证 >= 0.1，不会是NaN或负数
        """
        # 1. 获取本地算力（如果未提供则使用最小值）
        if f_base is None or f_base <= 0:
            f_base = Cfg.MIN_VEHICLE_CPU_FREQ
        
        # 2. 计算总计算量（所有子任务的计算量之和）
        total_comp = sum(p['comp'] for p in profiles)
        
        # 安全检查：确保计算量有效
        if total_comp <= 0 or not np.isfinite(total_comp):
            # 如果计算量无效，使用最小值
            total_comp = Cfg.MIN_COMP
        
        # 3. 计算理想本地执行时间（不考虑排队）
        t_local_ideal = total_comp / f_base
        
        # 安全检查：确保执行时间有效
        if not np.isfinite(t_local_ideal) or t_local_ideal <= 0:
            t_local_ideal = Cfg.MIN_COMP / Cfg.MIN_VEHICLE_CPU_FREQ

        # 4. 获取截止时间因子（紧缩因子）
        gamma_min = getattr(Cfg, 'DEADLINE_TIGHTENING_MIN', 0.70)
        gamma_max = getattr(Cfg, 'DEADLINE_TIGHTENING_MAX', 0.80)
        
        # 确保gamma范围合理
        gamma_min = max(0.1, min(gamma_min, 0.99))  # 限制在 [0.1, 0.99]
        gamma_max = max(gamma_min, min(gamma_max, 0.99))  # gamma_max >= gamma_min
        
        gamma = np.random.uniform(gamma_min, gamma_max)

        # 5. 计算相对截止时间
        deadline = gamma * t_local_ideal
        
        # 6. 最终安全检查：确保deadline是有效的正数
        if not np.isfinite(deadline) or deadline <= 0:
            # 如果计算出错，返回最小值
            deadline = 0.1
        else:
            # 确保deadline不小于最小值
            deadline = max(deadline, 0.1)

        return float(deadline)