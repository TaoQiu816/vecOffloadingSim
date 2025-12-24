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
        # --- 1. 任务属性范围 (直接对接 Config) ---
        # 确保生成的任务在环境设计的物理能力范围内
        self.comp_range = (Cfg.MIN_COMP, Cfg.MAX_COMP)
        self.data_range = (Cfg.MIN_DATA, Cfg.MAX_DATA)

        # 入口数据量 (通常设定为与内部传输数据量相似或略大)
        self.input_range = (Cfg.MIN_DATA, Cfg.MAX_DATA)

        # --- 2. DAG 拓扑参数 ---
        # 如果 Config 中没有定义这些具体的拓扑参数，使用默认值
        self.fat = getattr(Cfg, 'DAG_FAT', 0.5)  # 宽度: 决定并行的最大宽度
        self.density = getattr(Cfg, 'DAG_DENSITY', 0.5)  # 密度: 决定边的数量
        self.regular = getattr(Cfg, 'DAG_REGULAR', 0.5)  # 规则度: 决定层级结构的整齐程度
        self.ccr = getattr(Cfg, 'DAG_CCR', 0.5)  # 通信计算比 (Communication-to-Computation Ratio)

    # --- [修改部分] ---
    # 为 generate 增加 veh_f 参数，默认值为 None。如果为 None 则使用 Cfg 中的全局基准频率
    def generate(self, num_nodes, veh_f=None):
        """
        生成单个 DAG 任务实例

        Args:
            num_nodes (int): 任务节点数量
            veh_f (float, optional): 当前车辆的实际 CPU 频率。若传入，则以该频率计算 Deadline。

        Returns:
            adj_matrix (NxN int): 邻接矩阵 (0/1)
            profiles (List[Dict]): 节点属性列表 [{'comp': ..., 'input_data': ...}]
            data_matrix (NxN float): 边上的数据传输量 (Bits)
            deadline (float): 任务截止时间
        """
        if num_nodes <= 0:
            return None, [], None, 0

        # 初始化矩阵
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        data_matrix = np.zeros((num_nodes, num_nodes))  # Bits

        use_fallback = True

        # --- 1. 拓扑生成 (优先使用 daggen) ---
        if daggen is not None:
            try:
                # 随机种子确保多样性
                seed = np.random.randint(0, 100000)
                dag = daggen.DAG(seed=seed, n=num_nodes,
                                 fat=self.fat, density=self.density,
                                 regular=self.regular, ccr=self.ccr)

                # 获取边列表
                _, raw_edges = dag.task_n_edge_dicts()

                for edge in raw_edges:
                    # 兼容不同版本 daggen 的键名
                    u = edge.get('u', edge.get('src'))
                    v = edge.get('v', edge.get('dst'))

                    # 索引修正: daggen 有时输出 1-based索引，需检查并修正
                    if u is not None and v is not None:
                        # 简单的边界检查与修正
                        if u >= num_nodes or v >= num_nodes:
                            u_idx, v_idx = u - 1, v - 1
                        else:
                            u_idx, v_idx = u, v

                        # 填充矩阵
                        if 0 <= u_idx < num_nodes and 0 <= v_idx < num_nodes:
                            if u_idx != v_idx:  # 避免自环
                                adj_matrix[u_idx, v_idx] = 1
                                # 边上的数据量 (传输依赖)
                                data_matrix[u_idx, v_idx] = np.random.uniform(*self.data_range)
                                use_fallback = False
            except Exception as e:
                use_fallback = True

        # --- 2. Fallback 生成策略 (若 daggen 失败) ---
        if use_fallback:
            for i in range(num_nodes - 1):
                adj_matrix[i, i + 1] = 1
                data_matrix[i, i + 1] = np.random.uniform(*self.data_range)

                if i + 2 < num_nodes and np.random.rand() > 0.7:
                    target = np.random.randint(i + 2, num_nodes)
                    adj_matrix[i, target] = 1
                    data_matrix[i, target] = np.random.uniform(*self.data_range)

        # --- 3. 生成节点属性 (Profiles) ---
        profiles = []
        in_degrees = np.sum(adj_matrix, axis=0)

        for i in range(num_nodes):
            comp = np.random.uniform(*self.comp_range)
            is_entry = (in_degrees[i] == 0)
            inp_d = np.random.uniform(*self.input_range) if is_entry else 0.0

            profiles.append({
                'comp': comp,  # Cycles
                'input_data': inp_d  # Bits
            })

        # --- 4. 计算 Deadline ---
        # --- [修改部分] 将 veh_f 传给 _calc_deadline 方法 ---
        deadline = self._calc_deadline(num_nodes, adj_matrix, profiles, veh_f)

        return adj_matrix, profiles, data_matrix, deadline

    # --- [修改部分] 增加 f_base 参数用于接收特定的车辆频率 ---
    def _calc_deadline(self, n, adj, profiles, f_base=None):
        """
        基于特定或基准执行能力计算 Deadline。

        逻辑:
        Deadline = (本地串行执行的最长路径时间) * 紧迫因子
        紧迫因子 < 1.0 意味着如果不卸载或并行处理，任务很可能超时。
        """
        G = nx.DiGraph()

        # [新增] 频率选择：优先用传入的车载频率，没有则用全局基准
        exec_f = f_base if f_base is not None else getattr(Cfg, 'F_VEHICLE', 1e9)

        # 构建图: 节点权重 = 在当前频率下的计算时间
        for i in range(n):
            t_comp = profiles[i]['comp'] / exec_f
            G.add_node(i, weight=t_comp)

        # 添加边
        rows, cols = np.nonzero(adj)
        for u, v in zip(rows, cols):
            G.add_edge(u, v)

        try:
            # 寻找带权重的最长路径 (Critical Path)
            #
            path = nx.dag_longest_path(G, weight='weight')
            critical_time = sum(G.nodes[i]['weight'] for i in path)

            # 使用 Config 中的因子，默认 0.8 表示需要节省 20% 的时间
            factor = getattr(Cfg, 'DEADLINE_FACTOR', 0.8)

            # 加上一个极小值防止 deadline 为 0
            return max(critical_time * factor, 0.1)
        except Exception:
            # 防止成环导致的异常
            return 1.0