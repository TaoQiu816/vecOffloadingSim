import torch
import numpy as np
from torch_geometric.data import Data, HeteroData
from configs.config import SystemConfig as Cfg


class GraphBuilder:
    """
    [图数据构建器]
    负责将 VecOffloadingEnv 中的实体对象 (Vehicle, DAGTask) 转换为
    PyTorch Geometric (PyG) 可用的图数据格式。

    修改记录:
    - [关键修复] get_dag_graph 输出维度修正为 7 (增加 out_degree)。
    """

    @staticmethod
    def get_dag_graph(dag_task, current_time, target_mask=None):
        """
        构建单个任务的 DAG 图数据

        Args:
            dag_task: DAGTask 实体
            current_time: 当前仿真时间
            target_mask (np.array, optional): [Num_Subtasks, Num_Targets] 物理约束掩码
                                            来自 env.step() 返回的 observation
        """
        # A. 计算时间特征
        elapsed = current_time - dag_task.start_time
        t_rem = dag_task.deadline - elapsed
        t_total = dag_task.deadline if dag_task.deadline > 0 else 1.0

        num_nodes = dag_task.num_subtasks

        # 特征构建 (归一化)
        feat_comp = dag_task.rem_comp / Cfg.NORM_MAX_COMP
        feat_data = dag_task.rem_data / Cfg.NORM_MAX_DATA
        feat_status = dag_task.status / 3.0  # 状态归一化 (0~3)
        feat_in_degree = dag_task.in_degree / 5.0  # 入度归一化 (假设 max=5)

        # [关键新增] 出度特征 (与 Env 和 Config 对齐)
        # 如果 DAGTask 中还没计算 out_degree，这里临时算一下，防止报错
        if hasattr(dag_task, 'out_degree'):
            feat_out_degree = dag_task.out_degree / 5.0
        else:
            feat_out_degree = np.sum(dag_task.adj, axis=1) / 5.0

        # Deadline 相关
        feat_trem = np.clip(t_rem / 10.0, -1.0, 1.0)  # 数值
        feat_urgency = np.clip(t_rem / t_total, 0.0, 1.0) if t_rem > 0 else 0.0  # 比例

        # 拼接特征矩阵 X: [N, 7]
        # 必须与 TrainConfig.TASK_INPUT_DIM = 7 严格一致
        features = np.stack([
            feat_comp,  # 1. 计算量
            feat_data,  # 2. 数据量
            feat_status,  # 3. 状态
            feat_in_degree,  # 4. 入度
            feat_out_degree,  # 5. 出度 [新增]
            np.full(num_nodes, feat_trem),  # 6. 剩余时间
            np.full(num_nodes, feat_urgency)  # 7. 紧迫度
        ], axis=1)

        x = torch.tensor(features, dtype=torch.float32)

        # 构建边
        # 注意: 这里的 adj 是 numpy array
        src, dst = np.nonzero(dag_task.adj)
        # 显式使用 np.array 包装列表，防止 PyTorch 版本兼容性警告
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

        # Task Mask (逻辑约束: 哪些任务 Ready?)
        raw_mask = np.array(dag_task.get_action_mask(), dtype=bool)
        schedulable_mask = torch.as_tensor(raw_mask, dtype=torch.bool)

        data = Data(x=x, edge_index=edge_index)
        data.mask = schedulable_mask

        # [关键] 存储 Target Mask (物理约束: 哪些节点可达?)
        if target_mask is not None:
            # 确保维度匹配: 如果是 1D (整图 Mask)，扩展为 [N, Targets]
            tm = torch.as_tensor(target_mask, dtype=torch.bool)
            if tm.dim() == 1:
                tm = tm.unsqueeze(0).expand(num_nodes, -1)
            data.target_mask = tm
        else:
            # 如果没传，不设置该属性，Policy 中会有 None check
            pass

        # 记录 target_idx (用于 Decoder 知道当前决策哪个子任务)
        # 策略: 选择第一个 Ready 的任务作为本次决策的 Anchor
        ready_indices = data.mask.nonzero()
        data.target_idx = ready_indices[0].item() if len(ready_indices) > 0 else 0

        return data

    @staticmethod
    def _estimate_rate(dist, link_type='V2I'):
        """
        [辅助方法] 快速估算速率用于生成特征 (Observation Only)
        """
        dist = max(dist, 1.0)

        # 读取 Config 参数
        if link_type == 'V2I':
            B = Cfg.BW_V2I
            P_tx = Cfg.TX_POWER_MAX_DBM
            alpha = Cfg.ALPHA_V2I
        else:
            B = Cfg.BW_V2V
            P_tx = 20.0
            alpha = Cfg.ALPHA_V2V

        noise_w = Cfg.dbm2watt(Cfg.NOISE_POWER_DBM)
        p_tx_w = Cfg.dbm2watt(P_tx)

        # Path Loss
        pl = (dist ** (-alpha)) * (10 ** (getattr(Cfg, 'BETA_0_DB', -30) / 10.0))

        # SNR
        snr = (p_tx_w * pl) / noise_w
        rate = B * np.log2(1 + snr)

        return rate

    @staticmethod
    def get_topology_graph(vehicles, rsu_queue_len):
        """
        构建车辆拓扑异构图
        修改点:
        1. 节点特征使用车辆个体真实 cpu_freq。
        2. V2V 边特征增加 [相对算力比]，维度变为 2。
        """
        data = HeteroData()

        # ==========================================
        # 1. 构建节点特征 (Node Features)
        # ==========================================
        v_feats = []
        v_coords = []
        v_cpus = []  # 记录所有车的算力用于后续计算边特征

        for v in vehicles:
            n_pos = v.pos / Cfg.MAP_SIZE
            n_vel = v.vel / Cfg.MAX_VELOCITY

            # [异构算力修复] 基于个体频率计算排队负载
            wait_time = (v.task_queue_len * Cfg.MEAN_COMP_LOAD) / v.cpu_freq
            n_load = np.clip(wait_time / Cfg.NORM_MAX_WAIT_TIME, 0, 1)
            n_cpu = v.cpu_freq / Cfg.NORM_MAX_CPU

            # V2I 特征
            dist_rsu = np.linalg.norm(v.pos - Cfg.RSU_POS)
            est_v2i_rate = GraphBuilder._estimate_rate(dist_rsu, 'V2I')
            n_v2i_rate = np.clip(est_v2i_rate / Cfg.NORM_MAX_RATE_V2I, 0, 1)

            v_feats.append([
                n_pos[0], n_pos[1], n_vel[0], n_vel[1],
                n_load, n_cpu, n_v2i_rate
            ])
            v_coords.append(v.pos)
            v_cpus.append(v.cpu_freq)

        data['vehicle'].x = torch.tensor(v_feats, dtype=torch.float32)
        v_coords = np.array(v_coords)
        v_cpus = np.array(v_cpus)

        # RSU 特征 (固定算力 Cfg.F_RSU)
        r_pos = Cfg.RSU_POS / Cfg.MAP_SIZE
        r_wait = (rsu_queue_len * Cfg.MEAN_COMP_LOAD) / Cfg.F_RSU
        r_load = np.clip(r_wait / Cfg.NORM_MAX_WAIT_TIME, 0, 1)
        r_cpu = Cfg.F_RSU / Cfg.NORM_MAX_CPU
        r_feat = [[r_pos[0], r_pos[1], 0.0, 0.0, r_load, r_cpu, 1.0]]
        data['rsu'].x = torch.tensor(r_feat, dtype=torch.float32)

        # ==========================================
        # 2. 构建通信边 (Edge Features 增强)
        # ==========================================

        # --- A. V2V Edges ---
        dists_v2v = np.linalg.norm(v_coords[:, None, :] - v_coords[None, :, :], axis=-1)
        mask_v2v = (dists_v2v <= Cfg.V2V_RANGE) & (dists_v2v > 0)
        src_v, dst_v = np.nonzero(mask_v2v)

        if len(src_v) > 0:
            data['vehicle', 'v2v', 'vehicle'].edge_index = torch.tensor(np.array([src_v, dst_v]), dtype=torch.long)

            # 边属性构建: [归一化速率, 相对算力比]
            edge_attr_list = []
            for s, d in zip(src_v, dst_v):
                # 1. 速率特征
                rate = GraphBuilder._estimate_rate(dists_v2v[s, d], 'V2V')
                n_rate = np.clip(rate / Cfg.NORM_MAX_RATE_V2V, 0, 1)

                # 2. 相对算力比特征 (对方算力 / 自己算力)
                # 映射到 [0, 5] 并归一化，诱导智能体流向高算力节点
                cpu_ratio = v_cpus[d] / v_cpus[s]
                n_cpu_ratio = np.clip(cpu_ratio, 0.0, 5.0) / 5.0

                edge_attr_list.append([n_rate, n_cpu_ratio])

            # Edge Attr Dim 现在是 [E, 2]
            data['vehicle', 'v2v', 'vehicle'].edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            data['vehicle', 'v2v', 'vehicle'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['vehicle', 'v2v', 'vehicle'].edge_attr = torch.empty((0, 2), dtype=torch.float32)

        # --- B. V2I Edges ---
        dists_v2i = np.linalg.norm(v_coords - Cfg.RSU_POS, axis=1)
        mask_v2i = dists_v2i <= Cfg.RSU_RANGE
        valid_v_indices = np.nonzero(mask_v2i)[0]

        if len(valid_v_indices) > 0:
            src_v2i = valid_v_indices
            dst_v2i = np.zeros_like(src_v2i)

            # V2I 边特征也可以增加相对算力比 [n_rate, RSU算力/车算力]
            v2i_attr_list = []
            for idx in valid_v_indices:
                rate = GraphBuilder._estimate_rate(dists_v2i[idx], 'V2I')
                n_rate = np.clip(rate / Cfg.NORM_MAX_RATE_V2I, 0, 1)
                cpu_ratio = Cfg.F_RSU / v_cpus[idx]
                n_cpu_ratio = np.clip(cpu_ratio, 0.0, 10.0) / 10.0  # RSU优势通常很大
                v2i_attr_list.append([n_rate, n_cpu_ratio])

                # V2I 正向边
                data['vehicle', 'v2i', 'rsu'].edge_index = torch.tensor(np.array([src_v2i, dst_v2i]), dtype=torch.long)
                data['vehicle', 'v2i', 'rsu'].edge_attr = torch.tensor(v2i_attr_list, dtype=torch.float32)

                # --- [关键修复] I2V 反向边 ---
                # 必须为反向边也填充 edge_attr，保持维度 [E, 2]，否则 GNN 可能会崩溃
                data['rsu', 'i2v', 'vehicle'].edge_index = torch.tensor(np.array([dst_v2i, src_v2i]), dtype=torch.long)

                # 直接复用 v2i 的属性即可满足维度要求 (下行速率通常与上行相关，算力比取倒数或保持占位均可)
                # 这里简单复用以保证形状匹配
                data['rsu', 'i2v', 'vehicle'].edge_attr = torch.tensor(v2i_attr_list, dtype=torch.float32)

            else:
                # 空数据处理：必须同时处理 V2I 和 I2V
                empty_idx = torch.empty((2, 0), dtype=torch.long)
                empty_attr = torch.empty((0, 2), dtype=torch.float32)

                data['vehicle', 'v2i', 'rsu'].edge_index = empty_idx
                data['vehicle', 'v2i', 'rsu'].edge_attr = empty_attr

                data['rsu', 'i2v', 'vehicle'].edge_index = empty_idx
                data['rsu', 'i2v', 'vehicle'].edge_attr = empty_attr

            return data