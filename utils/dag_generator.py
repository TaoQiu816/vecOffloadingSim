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

    @staticmethod
    def _mix_enabled() -> bool:
        return bool(getattr(Cfg, "TASK_CLASS_MIX_ENABLE", False))

    @staticmethod
    def _pick_task_class() -> str:
        p_b = float(np.clip(getattr(Cfg, "TASK_CLASS_B_PROB", 0.2), 0.0, 1.0))
        return "B" if np.random.rand() < p_b else "A"

    @staticmethod
    def _sample_totals_by_class(task_class: str) -> tuple[float, float]:
        if task_class == "B":
            d_min = float(getattr(Cfg, "TASK_B_TOTAL_DATA_MIN", 1.0e7))
            d_max = float(getattr(Cfg, "TASK_B_TOTAL_DATA_MAX", 4.0e7))
            c_min = float(getattr(Cfg, "TASK_B_TOTAL_COMP_MIN", 1.0e9))
            c_max = float(getattr(Cfg, "TASK_B_TOTAL_COMP_MAX", 3.0e9))
        else:
            d_min = float(getattr(Cfg, "TASK_A_TOTAL_DATA_MIN", 1.0e5))
            d_max = float(getattr(Cfg, "TASK_A_TOTAL_DATA_MAX", 1.0e7))
            c_min = float(getattr(Cfg, "TASK_A_TOTAL_COMP_MIN", 1.0e8))
            c_max = float(getattr(Cfg, "TASK_A_TOTAL_COMP_MAX", 1.0e9))

        d_min, d_max = max(d_min, 1.0), max(d_max, d_min)
        c_min, c_max = max(c_min, 1.0), max(c_max, c_min)
        total_data = float(np.random.uniform(d_min, d_max))
        total_comp = float(np.random.uniform(c_min, c_max))
        return total_data, total_comp

    @staticmethod
    def _split_total(total: float, n: int, jitter: float) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=float)
        jit = float(np.clip(jitter, 0.0, 0.99))
        lo = max(1e-3, 1.0 - jit)
        hi = 1.0 + jit
        w = np.random.uniform(lo, hi, size=n).astype(float)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            return np.full(n, float(total) / float(n), dtype=float)
        return (float(total) * w / w_sum).astype(float)

    @staticmethod
    def _normalize_weights(weights: dict, keys: list[str]) -> np.ndarray:
        vals = np.array([max(float(weights.get(k, 0.0)), 0.0) for k in keys], dtype=float)
        total = float(np.sum(vals))
        if total <= 0.0:
            return np.full(len(keys), 1.0 / max(len(keys), 1), dtype=float)
        return vals / total

    @staticmethod
    def _get_workload_specs() -> dict:
        return getattr(Cfg, "WORKLOAD_PROFILE_SPECS", {})

    def _pick_workload_profile(self, sampling_profile=None) -> str | None:
        specs = self._get_workload_specs()
        if not specs:
            return None
        if sampling_profile is not None:
            profile_name = sampling_profile.get("workload_profile")
            if profile_name in specs:
                return str(profile_name)
        return None

    def _resolve_workload_spec(self, workload_profile: str | None, sampling_profile=None) -> dict | None:
        specs = self._get_workload_specs()
        if workload_profile is None:
            return None
        spec = specs.get(str(workload_profile))
        if spec is None:
            return None
        return dict(spec)

    def _sample_num_nodes(self, sampling_profile=None) -> tuple[int, str | None]:
        workload_profile = self._pick_workload_profile(sampling_profile=sampling_profile)
        spec = self._resolve_workload_spec(workload_profile, sampling_profile=sampling_profile)
        if spec is not None and "node_range" in spec:
            lo, hi = spec["node_range"]
            lo = max(int(lo), 1)
            hi = max(int(hi), lo)
            return int(np.random.randint(lo, hi + 1)), workload_profile
        return int(np.random.randint(Cfg.MIN_NODES, Cfg.MAX_NODES + 1)), workload_profile

    def generate(self, num_nodes, veh_f=None, workload_profile=None, sampling_profile=None):
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

        workload_spec = self._resolve_workload_spec(workload_profile, sampling_profile=sampling_profile)
        edge_range = self.edge_data_range
        if workload_spec is not None and "edge_data" in workload_spec:
            edge_range = tuple(workload_spec["edge_data"])

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
                                data_matrix[u_idx, v_idx] = np.random.uniform(*edge_range)
                                use_fallback = False
            except Exception as e:
                use_fallback = True

        # Fallback生成策略（当daggen不可用时）- 生成有并行宽度的DAG
        if use_fallback:
            adj_matrix, data_matrix = self._generate_layered_dag(num_nodes, edge_range=edge_range)

        # 生成节点属性
        profiles = []
        in_degrees = np.sum(adj_matrix, axis=0)
        entry_nodes = np.where(in_degrees == 0)[0]
        if len(entry_nodes) == 0:
            entry_nodes = np.array([0], dtype=int)

        task_class = None
        if workload_spec is not None:
            total_data_bits = float(np.random.uniform(*workload_spec["total_data"]))
            total_comp_cycles = float(np.random.uniform(*workload_spec["total_comp"]))
            comp_jit = float(getattr(Cfg, "TASK_COMP_SPLIT_JITTER", 0.6))
            data_jit = float(getattr(Cfg, "TASK_DATA_SPLIT_JITTER", 0.8))
            comp_arr = self._split_total(total_comp_cycles, num_nodes, comp_jit)
            entry_data_arr = self._split_total(total_data_bits, len(entry_nodes), data_jit)

            entry_data_map = {int(entry_nodes[k]): float(entry_data_arr[k]) for k in range(len(entry_nodes))}
            for i in range(num_nodes):
                profiles.append({
                    'comp': float(comp_arr[i]),
                    'input_data': float(entry_data_map.get(int(i), 0.0))
                })
            task_class = str(workload_profile).upper()
        elif self._mix_enabled():
            task_class = self._pick_task_class()
            total_data_bits, total_comp_cycles = self._sample_totals_by_class(task_class)
            comp_jit = float(getattr(Cfg, "TASK_COMP_SPLIT_JITTER", 0.6))
            data_jit = float(getattr(Cfg, "TASK_DATA_SPLIT_JITTER", 0.8))
            comp_arr = self._split_total(total_comp_cycles, num_nodes, comp_jit)
            entry_data_arr = self._split_total(total_data_bits, len(entry_nodes), data_jit)

            entry_data_map = {int(entry_nodes[k]): float(entry_data_arr[k]) for k in range(len(entry_nodes))}
            for i in range(num_nodes):
                profiles.append({
                    'comp': float(comp_arr[i]),       # 计算量 (Cycles)
                    'input_data': float(entry_data_map.get(int(i), 0.0))  # 输入数据量 (Bits)
                })
        else:
            for i in range(num_nodes):
                comp = np.random.uniform(*self.comp_range)
                is_entry = (in_degrees[i] == 0)  # 只有入口节点有输入数据
                inp_d = np.random.uniform(*self.input_range) if is_entry else 0.0
                profiles.append({
                    'comp': comp,       # 计算量 (Cycles)
                    'input_data': inp_d # 输入数据量 (Bits)
                })

        # 计算相对截止时间
        deadline, gamma, critical_path_cycles, base_time = self._calc_deadline(
            num_nodes, adj_matrix, profiles, veh_f, task_class=task_class, workload_profile=workload_profile,
            sampling_profile=sampling_profile,
        )

        extras = {
            "deadline_gamma": gamma,
            "critical_path_cycles": critical_path_cycles,
            "deadline_base_time": base_time,
            "deadline_seconds": deadline,
            "deadline_slack": getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0),
            "task_class": task_class or "LEGACY",
        }

        return adj_matrix, profiles, data_matrix, deadline, extras

    def generate_from_config(self, veh_f=None, sampling_profile=None):
        """
        根据配置选择DAG来源

        支持:
        - synthetic_small: 按MIN_NODES~MAX_NODES随机
        - synthetic_large: 从DAG_LARGE_NODE_OPTIONS选择
        - workflow_json: 读取本地JSON文件
        """
        source = getattr(Cfg, "DAG_SOURCE", "synthetic_small")
        if source == "synthetic_small":
            num_nodes, workload_profile = self._sample_num_nodes(sampling_profile=sampling_profile)
            return self.generate(num_nodes, veh_f=veh_f, workload_profile=workload_profile, sampling_profile=sampling_profile)
        if source == "synthetic_large":
            options = getattr(Cfg, "DAG_LARGE_NODE_OPTIONS", [20, 50, 100])
            if not options:
                raise ValueError("DAG_LARGE_NODE_OPTIONS is empty")
            num_nodes = int(np.random.choice(options))
            if num_nodes > Cfg.MAX_NODES:
                raise ValueError(
                    f"synthetic_large nodes={num_nodes} exceeds MAX_NODES={Cfg.MAX_NODES}"
                )
            return self.generate(num_nodes, veh_f=veh_f)
        if source == "workflow_json":
            path = getattr(Cfg, "WORKFLOW_JSON_PATH", None)
            if not path:
                raise ValueError("WORKFLOW_JSON_PATH is required for workflow_json")
            return self._load_workflow_json(path, veh_f=veh_f)
        raise ValueError(f"Unknown DAG_SOURCE: {source}")

    def _load_workflow_json(self, path, veh_f=None):
        import json

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        def _bytes_to_bits(val):
            return float(val) * 8.0

        nodes = payload.get("nodes")
        node_comp = payload.get("node_comp_cycles")
        node_input_bits = payload.get("node_input_bits")
        node_input_bytes = payload.get("node_input_bytes")
        if nodes is not None:
            num_nodes = len(nodes)
            comp_arr = [float(n.get("comp_cycles", 0.0)) for n in nodes]
            input_arr = []
            for n in nodes:
                if "input_data_bits" in n:
                    input_arr.append(float(n.get("input_data_bits", 0.0)))
                else:
                    # workflow_json的*_bytes字段统一转换为bit
                    input_arr.append(_bytes_to_bits(n.get("input_data_bytes", 0.0)))
        elif node_comp is not None:
            num_nodes = len(node_comp)
            comp_arr = [float(x) for x in node_comp]
            if node_input_bits is not None and len(node_input_bits) == num_nodes:
                input_arr = [float(x) for x in node_input_bits]
            elif node_input_bytes is not None and len(node_input_bytes) == num_nodes:
                input_arr = [_bytes_to_bits(x) for x in node_input_bytes]
            else:
                input_arr = [0.0] * num_nodes
        else:
            raise ValueError("workflow_json missing nodes or node_comp_cycles")

        if num_nodes <= 0:
            raise ValueError("workflow_json has no nodes")
        if num_nodes > Cfg.MAX_NODES:
            raise ValueError(
                f"workflow_json nodes={num_nodes} exceeds MAX_NODES={Cfg.MAX_NODES}"
            )

        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        data_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        edges = payload.get("edges", [])
        for edge in edges:
            if isinstance(edge, dict):
                u = edge.get("src")
                v = edge.get("dst")
                if "data_bits" in edge:
                    data_bits = edge.get("data_bits", 0.0)
                else:
                    # workflow_json的*_bytes字段统一转换为bit
                    data_bits = _bytes_to_bits(edge.get("data_bytes", 0.0))
            else:
                u = edge[0] if len(edge) > 0 else None
                v = edge[1] if len(edge) > 1 else None
                data_bits = edge[2] if len(edge) > 2 else 0.0
            if u is None or v is None:
                continue
            u_idx = int(u)
            v_idx = int(v)
            if u_idx == v_idx:
                continue
            if not (0 <= u_idx < num_nodes and 0 <= v_idx < num_nodes):
                raise ValueError(f"workflow_json edge out of range: {u_idx}->{v_idx}")
            adj_matrix[u_idx, v_idx] = 1
            data_matrix[u_idx, v_idx] = float(data_bits)

        # DAG校验
        graph = nx.DiGraph(adj_matrix)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("workflow_json edges contain cycle")

        in_degrees = np.sum(adj_matrix, axis=0)
        profiles = []
        for i in range(num_nodes):
            comp = comp_arr[i] if i < len(comp_arr) else float(np.random.uniform(*self.comp_range))
            is_entry = (in_degrees[i] == 0)
            inp_d = input_arr[i] if is_entry else 0.0
            profiles.append({
                "comp": comp,
                "input_data": inp_d,
            })

        deadline, gamma, critical_path_cycles, base_time = self._calc_deadline(
            num_nodes, adj_matrix, profiles, veh_f
        )
        extras = {
            "deadline_gamma": gamma,
            "critical_path_cycles": critical_path_cycles,
            "deadline_base_time": base_time,
            "deadline_seconds": deadline,
            "deadline_slack": getattr(Cfg, "DEADLINE_SLACK_SECONDS", 0.0),
            "workflow_path": path,
        }
        return adj_matrix, profiles, data_matrix, deadline, extras

    def _calc_deadline(self, n, adj, profiles, f_base=None, task_class=None, workload_profile=None, sampling_profile=None):
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
        
        # 计算关键路径长度（最长路径计算量和）与关键路径深度（节点数）
        cp_cycles = self._critical_path_cycles(adj, comp_arr)
        cp_depth = self._critical_path_depth(adj)
        if cp_cycles <= 0 or not np.isfinite(cp_cycles):
            cp_cycles = total_cycles  # fallback
        if cp_depth <= 0 or not np.isfinite(cp_depth):
            cp_depth = float(max(1, n))
        
        if total_cycles <= 0 or not np.isfinite(total_cycles):
            total_cycles = Cfg.MEAN_COMP_LOAD * n
            cp_cycles = total_cycles
        
        # 系统算力参数
        f_median = (Cfg.MIN_VEHICLE_CPU_FREQ + Cfg.MAX_VEHICLE_CPU_FREQ) / 2.0
        f_max = max(Cfg.MAX_VEHICLE_CPU_FREQ, getattr(Cfg, 'F_RSU', Cfg.MAX_VEHICLE_CPU_FREQ))
        
        # 物理下界：即使最优调度也无法突破
        LB0 = cp_cycles / f_max
        # Step量化保护：DT离散仿真下，至少给关键路径深度对应的步数预算
        dt = float(max(getattr(Cfg, "DT", 0.1), 1e-6))
        delta = float(max(getattr(Cfg, "DEADLINE_STEP_GUARD_DELTA", 1.0), 0.0))
        LB_step = (float(cp_depth) + delta) * dt
        LB_star = max(LB0, LB_step)
        
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
        
        workload_spec = self._resolve_workload_spec(workload_profile, sampling_profile=sampling_profile)

        if workload_spec is not None:
            alpha_min, alpha_max = workload_spec.get("deadline_alpha", (1.0, 1.0))
            alpha_min = max(1.0, float(alpha_min))
            alpha_max = max(alpha_min, float(alpha_max))
            alpha = float(np.random.uniform(alpha_min, alpha_max))
            slack = max(float(workload_spec.get("deadline_slack", slack)), 0.0)
            base_time = LB_star
            deadline_raw = alpha * LB_star + slack
            gamma = deadline_raw / max(base_time, 1e-9)

        elif self._mix_enabled() and task_class in ("A", "B"):
            if task_class == "B":
                d_min = float(getattr(Cfg, "TASK_B_DEADLINE_MIN", 1.0))
                d_max = float(getattr(Cfg, "TASK_B_DEADLINE_MAX", 2.5))
            else:
                d_min = float(getattr(Cfg, "TASK_A_DEADLINE_MIN", 0.4))
                d_max = float(getattr(Cfg, "TASK_A_DEADLINE_MAX", 1.0))
            d_min = max(0.05, d_min)
            d_max = max(d_min, d_max)
            deadline_raw = float(np.random.uniform(d_min, d_max))
            base_time = LB_star
            gamma = deadline_raw / max(base_time, 1e-9)

        elif mode == 'FIXED_RANGE':
            # 模式: 固定范围直接随机
            d_min = getattr(Cfg, 'DEADLINE_FIXED_MIN', 2.0)
            d_max = getattr(Cfg, 'DEADLINE_FIXED_MAX', 5.0)
            deadline_raw = np.random.uniform(d_min, d_max)
            base_time = cp_cycles / f_median
            gamma = deadline_raw / max(base_time, 1e-9)  # 反推gamma
        elif mode == 'LB_ALPHA':
            # 模式: 物理下界驱动（CMDP/可行域校准用）
            # deadline = alpha * Tmin + slack, Tmin = LB0 = cp_cycles / f_max
            alpha_min = getattr(Cfg, 'DEADLINE_ALPHA_MIN', 1.4)
            alpha_max = getattr(Cfg, 'DEADLINE_ALPHA_MAX', 2.0)
            alpha_min = max(1.0, alpha_min)
            alpha_max = max(alpha_min, alpha_max)
            alpha = np.random.uniform(alpha_min, alpha_max)
            base_time = LB_star
            deadline_raw = alpha * LB0 + slack
            gamma = deadline_raw / max(base_time, 1e-9)
            
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
        deadline = max(deadline_raw, (1.0 + eps) * LB_star)
        
        # 安全检查
        if not np.isfinite(base_time) or base_time <= 0:
            base_time = Cfg.MIN_COMP / Cfg.MIN_VEHICLE_CPU_FREQ
        
        if not np.isfinite(deadline) or deadline <= 0:
            deadline = max(0.1, (1.0 + eps) * LB_star)
        
        return float(deadline), float(gamma), float(cp_cycles), float(base_time)

    def _generate_layered_dag(self, num_nodes, edge_range=None):
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
        edge_range = tuple(edge_range) if edge_range is not None else self.edge_data_range
        
        if num_nodes <= 2:
            # 极小DAG，直接链式
            if num_nodes == 2:
                adj_matrix[0, 1] = 1
                data_matrix[0, 1] = np.random.uniform(*edge_range)
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
                    data_matrix[u, v] = np.random.uniform(*edge_range)
            
            # 确保每个下一层节点至少有一条入边
            for v in next_layer:
                if np.sum(adj_matrix[:, v]) == 0:
                    u = np.random.choice(curr_layer)
                    adj_matrix[u, v] = 1
                    data_matrix[u, v] = np.random.uniform(*edge_range)
        
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
                    data_matrix[u, v] = np.random.uniform(*edge_range)
        
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

    @staticmethod
    def _critical_path_depth(adj):
        n = int(adj.shape[0]) if adj is not None else 0
        if n <= 0:
            return 0.0
        indeg = np.sum(adj, axis=0).astype(int)
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
            return float(n)
        dp = np.ones(n, dtype=float)
        for u in order:
            for v in np.where(adj[u] == 1)[0]:
                dp[v] = max(dp[v], dp[u] + 1.0)
        return float(np.max(dp))
