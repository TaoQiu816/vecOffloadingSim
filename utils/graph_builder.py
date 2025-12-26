import torch
import numpy as np
from torch_geometric.data import Data, HeteroData
from configs.config import SystemConfig as Cfg


class GraphBuilder:
    """
    [图数据构建器] - Final Fixed Version
    1. 修正 RSU 特征为 1 维 (对齐 TrainConfig)。
    2. 修复 V2I 边构建时的缩进逻辑错误。
    3. 确保计算负载使用 MEAN_COMP_LOAD 以对齐物理量纲。
    """

    @staticmethod
    def get_dag_graph(dag_task, current_time, target_mask=None):
        """
        构建单个任务的 DAG 图数据
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
        feat_in_degree = dag_task.in_degree / 5.0  # 入度归一化

        # [关键新增] 出度特征 (与 Env 和 Config 对齐)
        if hasattr(dag_task, 'out_degree'):
            feat_out_degree = dag_task.out_degree / 5.0
        else:
            feat_out_degree = np.sum(dag_task.adj, axis=1) / 5.0

        # Deadline 相关
        feat_trem = np.clip(t_rem / 10.0, -1.0, 1.0)  # 数值
        feat_urgency = np.clip(t_rem / t_total, 0.0, 1.0) if t_rem > 0 else 0.0  # 比例

        # 拼接特征矩阵 X: [N, 7]
        features = np.stack([
            feat_comp,  # 1. 计算量
            feat_data,  # 2. 数据量
            feat_status,  # 3. 状态
            feat_in_degree,  # 4. 入度
            feat_out_degree,  # 5. 出度
            np.full(num_nodes, feat_trem),  # 6. 剩余时间
            np.full(num_nodes, feat_urgency)  # 7. 紧迫度
        ], axis=1)

        x = torch.tensor(features, dtype=torch.float32)

        # 构建边
        src, dst = np.nonzero(dag_task.adj)
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

        # Task Mask
        raw_mask = np.array(dag_task.get_action_mask(), dtype=bool)
        schedulable_mask = torch.as_tensor(raw_mask, dtype=torch.bool)

        data = Data(x=x, edge_index=edge_index)
        data.mask = schedulable_mask

        # 存储 Target Mask
        if target_mask is not None:
            tm = torch.as_tensor(target_mask, dtype=torch.bool)
            if tm.dim() == 1:
                tm = tm.unsqueeze(0).expand(num_nodes, -1)
            data.target_mask = tm

        # 记录 target_idx
        ready_indices = data.mask.nonzero()
        data.target_idx = ready_indices[0].item() if len(ready_indices) > 0 else 0

        return data

    @staticmethod
    def _estimate_rate(dist, link_type='V2I'):
        """
        [辅助方法] 快速估算速率用于生成特征
        """
        dist = max(dist, 1.0)
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
        pl = (dist ** (-alpha)) * (10 ** (getattr(Cfg, 'BETA_0_DB', -30) / 10.0))
        snr = (p_tx_w * pl) / noise_w
        rate = B * np.log2(1 + snr)
        return rate

    @staticmethod
    def get_topology_graph(vehicles, rsu_queue_len):
        """
        构建车辆拓扑异构图
        """
        data = HeteroData()

        if len(vehicles) == 0:
            data['vehicle'].x = torch.empty((0, 7), dtype=torch.float32)
            r_wait = (rsu_queue_len * Cfg.MEAN_COMP_LOAD) / Cfg.F_RSU
            r_load = np.clip(r_wait / Cfg.NORM_MAX_WAIT_TIME, 0, 1)
            data['rsu'].x = torch.tensor([[r_load]], dtype=torch.float32)
            empty_idx = torch.empty((2, 0), dtype=torch.long)
            empty_attr = torch.empty((0, 2), dtype=torch.float32)
            data['vehicle', 'v2v', 'vehicle'].edge_index = empty_idx
            data['vehicle', 'v2v', 'vehicle'].edge_attr = empty_attr
            data['vehicle', 'v2i', 'rsu'].edge_index = empty_idx
            data['vehicle', 'v2i', 'rsu'].edge_attr = empty_attr
            data['rsu', 'i2v', 'vehicle'].edge_index = empty_idx
            data['rsu', 'i2v', 'vehicle'].edge_attr = empty_attr
            return data

        # ==========================================
        # 1. 构建节点特征 (Node Features)
        # ==========================================
        v_feats = []
        v_coords = []
        v_cpus = []

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

        # [关键修复 A] RSU 特征维度必须为 1 (与 TrainConfig.RSU_INPUT_DIM = 1 一致)
        # 移除位置、速度、CPU 等冗余项，仅保留归一化负载
        r_wait = (rsu_queue_len * Cfg.MEAN_COMP_LOAD) / Cfg.F_RSU
        r_load = np.clip(r_wait / Cfg.NORM_MAX_WAIT_TIME, 0, 1)

        # 维度变为 [1, 1]，防止卷积层 mat1/mat2 乘法报错
        data['rsu'].x = torch.tensor([[r_load]], dtype=torch.float32)

        # ==========================================
        # 2. 构建通信边 (Edge Features)
        # ==========================================

        # --- A. V2V Edges ---
        dists_v2v = np.linalg.norm(v_coords[:, None, :] - v_coords[None, :, :], axis=-1)
        mask_v2v = (dists_v2v <= Cfg.V2V_RANGE) & (dists_v2v > 0)
        src_v, dst_v = np.nonzero(mask_v2v)

        if len(src_v) > 0:
            data['vehicle', 'v2v', 'vehicle'].edge_index = torch.tensor(np.array([src_v, dst_v]), dtype=torch.long)
            edge_attr_list = []
            for s, d in zip(src_v, dst_v):
                rate = GraphBuilder._estimate_rate(dists_v2v[s, d], 'V2V')
                n_rate = np.clip(rate / Cfg.NORM_MAX_RATE_V2V, 0, 1)
                cpu_ratio = v_cpus[d] / v_cpus[s]
                n_cpu_ratio = np.clip(cpu_ratio, 0.0, 5.0) / 5.0
                edge_attr_list.append([n_rate, n_cpu_ratio])
            data['vehicle', 'v2v', 'vehicle'].edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            data['vehicle', 'v2v', 'vehicle'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['vehicle', 'v2v', 'vehicle'].edge_attr = torch.empty((0, 2), dtype=torch.float32)

        # --- B. V2I / I2V Edges (修正缩进与逻辑错误) ---
        dists_v2i = np.linalg.norm(v_coords - Cfg.RSU_POS, axis=1)
        mask_v2i = dists_v2i <= Cfg.RSU_RANGE
        valid_v_indices = np.nonzero(mask_v2i)[0]

        # [关键修复 B] 修正 if 逻辑：先收集完整数据，再统一转 Tensor 并赋值
        if len(valid_v_indices) > 0:
            src_v2i = valid_v_indices
            dst_v2i = np.zeros_like(src_v2i)  # RSU 索引始终为 0

            v2i_attr_list = []
            for idx in valid_v_indices:
                rate = GraphBuilder._estimate_rate(dists_v2i[idx], 'V2I')
                n_rate = np.clip(rate / Cfg.NORM_MAX_RATE_V2I, 0, 1)
                cpu_ratio = Cfg.F_RSU / v_cpus[idx]
                n_cpu_ratio = np.clip(cpu_ratio, 0.0, 10.0) / 10.0
                v2i_attr_list.append([n_rate, n_cpu_ratio])

            # [关键修复] 将 Tensor 转换和赋值移出 for 循环
            edge_attr_tensor = torch.tensor(v2i_attr_list, dtype=torch.float32)

            # V2I (Vehicle -> RSU)
            data['vehicle', 'v2i', 'rsu'].edge_index = torch.tensor(np.array([src_v2i, dst_v2i]), dtype=torch.long)
            data['vehicle', 'v2i', 'rsu'].edge_attr = edge_attr_tensor

            # [关键修复 C] I2V (RSU -> Vehicle) 反向边构建
            data['rsu', 'i2v', 'vehicle'].edge_index = torch.tensor(np.array([dst_v2i, src_v2i]), dtype=torch.long)
            data['rsu', 'i2v', 'vehicle'].edge_attr = edge_attr_tensor

        else:
            # [关键修复] else 块现在正确地对应 if len(valid_v_indices) > 0
            empty_idx = torch.empty((2, 0), dtype=torch.long)
            empty_attr = torch.empty((0, 2), dtype=torch.float32)

            data['vehicle', 'v2i', 'rsu'].edge_index = empty_idx
            data['vehicle', 'v2i', 'rsu'].edge_attr = empty_attr
            data['rsu', 'i2v', 'vehicle'].edge_index = empty_idx
            data['rsu', 'i2v', 'vehicle'].edge_attr = empty_attr

        return data