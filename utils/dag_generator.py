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
                # daggen使用位置参数: DAG(seed, n, fat, density, regular, ccr)
                dag = daggen.DAG(seed, num_nodes, self.fat, self.density,
                                 self.regular, self.ccr)

                _, raw_edges = dag.task_n_edge_dicts()

                for edge in raw_edges:
                    # daggen边字段是 'source' 和 'target'，索引从1开始
                    u = edge.get('source', edge.get('u', edge.get('src')))
                    v = edge.get('target', edge.get('v', edge.get('dst')))

                    if u is not None and v is not None:
                        # daggen索引从1开始，转为0-based
                        u_idx, v_idx = u - 1, v - 1

                        if 0 <= u_idx < num_nodes and 0 <= v_idx < num_nodes:
                            if u_idx != v_idx:
                                adj_matrix[u_idx, v_idx] = 1
                                data_matrix[u_idx, v_idx] = np.random.uniform(*self.edge_data_range)
                                use_fallback = False
            except Exception as e:
                use_fallback = True

        # Fallback生成策略（当daggen不可用时）- 生成有并行宽度的DAG
        if use_fallback:
            adj_matrix, data_matrix = self._generate_layered_dag(num_nodes)

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
        计算相对截止时间（基于关键路径）
        
        核心公式:
            T_base = CP_total / f_ref           # 关键路径基准时间
            deadline_raw = gamma * T_base + slack
            LB0 = CP_total / f_max              # 物理下界
            deadline = max(deadline_raw, (1+eps) * LB0)  # 保证可行性
        
        模式说明:
        - CRITICAL_PATH (推荐): 使用关键路径计算量，反映DAG结构特性
        - TOTAL_MEDIAN: 使用总计算量（向后兼容）
        - TOTAL_LOCAL: 使用本地频率（向后兼容）
        - FIXED_RANGE: 固定范围随机
        
        Args:
            n: 节点数
            adj: 邻接矩阵
            profiles: 节点属性列表
            f_base: 本地CPU频率(Hz)
        
        Returns:
            tuple: (deadline_seconds, gamma, critical_path_cycles, base_time)
        """
        comp_arr = np.array([p['comp'] for p in profiles], dtype=float)
        total_cycles = np.sum(comp_arr)
        
        # 计算关键路径长度（最长路径计算量和）
        cp_cycles = self._critical_path_cycles(adj, comp_arr)
        if cp_cycles <= 0 or not np.isfinite(cp_cycles):
            cp_cycles = total_cycles  # fallback
        
        if total_cycles <= 0 or not np.isfinite(total_cycles):
            total_cycles = Cfg.MEAN_COMP_LOAD * n
            cp_cycles = total_cycles
        
        # 系统算力参数
        f_median = (Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ) / 2.0
        f_max = max(Cfg.MAX_VEHICLE_CPU_FREQ, getattr(Cfg, 'F_RSU', Cfg.MAX_VEHICLE_CPU_FREQ))
        
        # 物理下界：即使最优调度也无法突破
        LB0 = cp_cycles / f_max
        
        # 获取deadline模式
        mode = getattr(Cfg, 'DEADLINE_MODE', 'TOTAL_MEDIAN')
        
        # gamma范围
        gamma_min = getattr(Cfg, 'DEADLINE_TIGHTENING_MIN', 1.3)
        gamma_max = getattr(Cfg, 'DEADLINE_TIGHTENING_MAX', 2.0)
        gamma_min = max(0.1, gamma_min)
        gamma_max = max(gamma_min, gamma_max)
        gamma = np.random.uniform(gamma_min, gamma_max)
        
        slack = max(0.0, getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0))
        eps = getattr(Cfg, 'DEADLINE_LB_EPS', 0.05)  # 下界裕量
        
        if mode == 'FIXED_RANGE':
            # 模式: 固定范围直接随机
            d_min = getattr(Cfg, 'DEADLINE_FIXED_MIN', 2.0)
            d_max = getattr(Cfg, 'DEADLINE_FIXED_MAX', 5.0)
            deadline_raw = np.random.uniform(d_min, d_max)
            base_time = cp_cycles / f_median
            gamma = deadline_raw / max(base_time, 1e-9)  # 反推gamma
            
        elif mode == 'TOTAL_LOCAL':
            # 模式: 使用本地CPU频率 + 总计算量
            if f_base is None or f_base <= 0:
                f_base = f_median
            base_time = total_cycles / f_base
            deadline_raw = gamma * base_time + slack
            
        elif mode == 'CRITICAL_PATH':
            # 模式: 使用关键路径（推荐）
            base_time = cp_cycles / f_median
            deadline_raw = gamma * base_time + slack
            
        else:  # 'TOTAL_MEDIAN' (默认，向后兼容)
            # 模式: 使用总计算量 + 平均算力
            base_time = total_cycles / f_median
            deadline_raw = gamma * base_time + slack
        
        # 物理下界保险：确保deadline至少比LB0大(1+eps)倍
        deadline = max(deadline_raw, (1.0 + eps) * LB0)
        
        # 安全检查
        if not np.isfinite(base_time) or base_time <= 0:
            base_time = Cfg.MIN_COMP / Cfg.MIN_VEHICLE_CPU_FREQ
        
        if not np.isfinite(deadline) or deadline <= 0:
            deadline = max(0.1, (1.0 + eps) * LB0)
        
        return float(deadline), float(gamma), float(cp_cycles), float(base_time)

    def _generate_layered_dag(self, num_nodes):
        """
        生成有并行宽度的层级DAG（fallback方法）
        
        结构设计:
        - 入口层: 1个入口节点
        - 中间层: 多层，每层2-3个并行节点
        - 出口层: 1个出口节点
        
        连接规则:
        - 每层节点连接到下一层的部分或全部节点
        - 确保DAG连通且无环
        
        Args:
            num_nodes: 节点总数
            
        Returns:
            adj_matrix: 邻接矩阵
            data_matrix: 边数据矩阵
        """
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        data_matrix = np.zeros((num_nodes, num_nodes))
        
        if num_nodes <= 2:
            # 极小DAG，直接链式
            if num_nodes == 2:
                adj_matrix[0, 1] = 1
                data_matrix[0, 1] = np.random.uniform(*self.edge_data_range)
            return adj_matrix, data_matrix
        
        # 计算层级结构
        # 入口层1个节点，出口层1个节点，中间层分配剩余节点
        middle_nodes = num_nodes - 2
        
        # 根据fat参数确定平均每层宽度
        # fat=0.5 -> 宽度约2-3
        avg_width = max(2, int(2 + 2 * self.fat))
        
        # 分配中间层
        layers = [[0]]  # 入口层
        node_idx = 1
        
        while node_idx < num_nodes - 1:
            # 当前层宽度
            remaining = num_nodes - 1 - node_idx
            width = min(remaining, avg_width)
            width = max(1, int(width * np.random.uniform(0.7, 1.3)))  # 添加随机性
            width = min(width, remaining)
            
            layer = list(range(node_idx, node_idx + width))
            layers.append(layer)
            node_idx += width
        
        # 出口层
        layers.append([num_nodes - 1])
        
        # 建立层间连接
        density_factor = max(0.3, min(0.8, 0.4 + 0.3 * self.density))
        
        for l in range(len(layers) - 1):
            curr_layer = layers[l]
            next_layer = layers[l + 1]
            
            # 确保每个当前层节点至少有一条出边
            for u in curr_layer:
                # 选择目标节点数量
                num_targets = max(1, int(len(next_layer) * density_factor))
                num_targets = min(num_targets, len(next_layer))
                
                targets = np.random.choice(next_layer, size=num_targets, replace=False)
                for v in targets:
                    adj_matrix[u, v] = 1
                    data_matrix[u, v] = np.random.uniform(*self.edge_data_range)
            
            # 确保每个下一层节点至少有一条入边
            for v in next_layer:
                if np.sum(adj_matrix[:, v]) == 0:
                    u = np.random.choice(curr_layer)
                    adj_matrix[u, v] = 1
                    data_matrix[u, v] = np.random.uniform(*self.edge_data_range)
        
        # 可选：添加跨层边（增加复杂度）
        if self.density > 0.3 and len(layers) > 3:
            num_skip_edges = int(num_nodes * self.density * 0.3)
            for _ in range(num_skip_edges):
                # 选择跨层连接
                src_layer_idx = np.random.randint(0, len(layers) - 2)
                dst_layer_idx = np.random.randint(src_layer_idx + 2, len(layers))
                
                u = np.random.choice(layers[src_layer_idx])
                v = np.random.choice(layers[dst_layer_idx])
                
                if adj_matrix[u, v] == 0:
                    adj_matrix[u, v] = 1
                    data_matrix[u, v] = np.random.uniform(*self.edge_data_range)
        
        return adj_matrix, data_matrix

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
