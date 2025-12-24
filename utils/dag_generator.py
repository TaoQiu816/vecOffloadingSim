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
    def __init__(self):
        """
        [DAG 任务生成器]

        修改点:
        1. 不再通过 kwargs 传参，而是直接读取 Cfg 中的物理约束范围。
        2. 确保生成的计算量(Comp)和数据量(Data)与环境的归一化基准一致。
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
            for i in range(num_nodes - 1):
                adj_matrix[i, i + 1] = 1
                data_matrix[i, i + 1] = np.random.uniform(*self.edge_data_range)

                if i + 2 < num_nodes and np.random.rand() > 0.7:
                    target = np.random.randint(i + 2, num_nodes)
                    adj_matrix[i, target] = 1
                    data_matrix[i, target] = np.random.uniform(*self.edge_data_range)

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
        基于车辆个体算力计算 Deadline。
        """
        G = nx.DiGraph()

        # 使用传入的车辆频率，如果没有则使用全局基准
        exec_f = f_base if f_base is not None else getattr(Cfg, 'F_VEHICLE', 1e9)

        # 构建图: 节点权重 = 本地计算时间
        # 注意: 计算 Deadline 的基准通常是 "本地串行执行时间"，所以不考虑传输时间
        for i in range(n):
            t_comp = profiles[i]['comp'] / exec_f
            G.add_node(i, weight=t_comp)

        # 添加边
        rows, cols = np.nonzero(adj)
        for u, v in zip(rows, cols):
            G.add_edge(u, v)

        try:
            # 寻找关键路径 (本地计算时间最长的路径)
            path = nx.dag_longest_path(G, weight='weight')
            critical_time = sum(G.nodes[i]['weight'] for i in path)

            # [关键修改] 使用 Config 中的范围因子 [0.8, 1.5]
            # 0.8 (Hard): 逼迫卸载; 1.5 (Easy): 允许本地
            f_min = getattr(Cfg, 'DEADLINE_FACTOR_MIN', 0.8)
            f_max = getattr(Cfg, 'DEADLINE_FACTOR_MAX', 1.5)
            factor = np.random.uniform(f_min, f_max)

            # 加上极小值防止为 0
            return max(critical_time * factor, 0.1)
        except Exception:
            # 异常兜底
            return 1.0