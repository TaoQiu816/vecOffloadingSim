import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.encoders import DAGTaskEncoder, HeteroTopologyEncoder


# # =========================================================================
# # 1. 高级图编码器 (Encoders)
# # [说明] 这里使用上一轮修复后的"Robust"版本，支持边特征和动态层数
# # =========================================================================
#
# class DAGTaskEncoder(nn.Module):
#     """
#     [DAG 任务编码器 - 增强版]
#     使用 ModuleList 支持动态层数，确保 Critic 能提取深层特征。
#     """
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=2):
#         super(DAGTaskEncoder, self).__init__()
#
#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.num_layers = num_layers
#
#         # Layer 1
#         self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True))
#         self.norms.append(LayerNorm(hidden_dim * heads))
#
#         # Middle Layers
#         mid_dim = hidden_dim * heads
#         for _ in range(num_layers - 2):
#             self.layers.append(GATv2Conv(mid_dim, hidden_dim, heads=heads, concat=True))
#             self.norms.append(LayerNorm(mid_dim))
#
#         # Last Layer
#         self.layers.append(GATv2Conv(mid_dim, output_dim, heads=1, concat=False))
#         self.last_norm = LayerNorm(output_dim)
#
#     def forward(self, x, edge_index, batch=None):
#         for i in range(self.num_layers - 1):
#             x = self.layers[i](x, edge_index)
#             x = self.norms[i](x)
#             x = F.elu(x)
#             x = F.dropout(x, p=0.2, training=self.training)
#
#         node_emb = self.layers[-1](x, edge_index)
#
#         # Global Pooling
#         if batch is not None:
#             graph_emb = global_mean_pool(node_emb, batch)
#         else:
#             graph_emb = node_emb.mean(dim=0, keepdim=True)
#
#         return node_emb, graph_emb
#
#
# class HeteroTopologyEncoder(nn.Module):
#     """
#     [异构网络拓扑编码器 - 完整修复版]
#
#     关键修正:
#     1. [新增] 接收 edge_dim 参数，启用边特征 (带宽/算力比)。
#     2. [新增] Fallback 线性层，防止无边情况下维度不匹配。
#     3. [修正] V2I/I2V 禁止自环，V2V 允许自环。
#     """
#
#     def __init__(self, vehicle_dim, rsu_dim, hidden_dim, output_dim, heads=2, num_layers=2):
#         super(HeteroTopologyEncoder, self).__init__()
#
#         # 引用 Config 中的边特征维度 (通常为 2)
#         edge_dim = TrainConfig.EDGE_INPUT_DIM
#         self.num_layers = num_layers
#
#         # [新增] 兜底投影层 (处理孤立节点)
#         self.veh_fallback = nn.Linear(vehicle_dim, output_dim)
#         self.rsu_fallback = nn.Linear(rsu_dim, output_dim)
#
#         self.layers = nn.ModuleList()
#
#         # --- Layer 1 ---
#         self.layers.append(HeteroConv({
#             ('vehicle', 'v2v', 'vehicle'): GATv2Conv(
#                 vehicle_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=True
#             ),
#             ('vehicle', 'v2i', 'rsu'): GATv2Conv(
#                 (vehicle_dim, rsu_dim), hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False
#             ),
#             ('rsu', 'i2v', 'vehicle'): GATv2Conv(
#                 (rsu_dim, vehicle_dim), hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False
#             ),
#         }, aggr='sum'))
#
#         # --- Middle & Last Layers ---
#         mid_dim = hidden_dim * heads
#         for i in range(1, num_layers):
#             is_last = (i == num_layers - 1)
#             out_d = output_dim if is_last else hidden_dim
#             n_heads = 1 if is_last else heads
#             do_concat = not is_last
#
#             self.layers.append(HeteroConv({
#                 ('vehicle', 'v2v', 'vehicle'): GATv2Conv(
#                     mid_dim, out_d, heads=n_heads, edge_dim=edge_dim, concat=do_concat, add_self_loops=True
#                 ),
#                 ('vehicle', 'v2i', 'rsu'): GATv2Conv(
#                     (mid_dim, mid_dim), out_d, heads=n_heads, edge_dim=edge_dim, concat=do_concat, add_self_loops=False
#                 ),
#                 ('rsu', 'i2v', 'vehicle'): GATv2Conv(
#                     (mid_dim, mid_dim), out_d, heads=n_heads, edge_dim=edge_dim, concat=do_concat, add_self_loops=False
#                 ),
#             }, aggr='sum'))
#
#         self.ln_veh = LayerNorm(output_dim)
#         self.ln_rsu = LayerNorm(output_dim)
#
#     def forward(self, x_dict, edge_index_dict, edge_attr_dict):
#         # [关键] 鲁棒性检查: 如果没有边，直接使用线性投影
#         has_edges = len(edge_index_dict) > 0 and any(e.size(1) > 0 for e in edge_index_dict.values())
#
#         if not has_edges:
#             out = {}
#             if 'vehicle' in x_dict: out['vehicle'] = self.veh_fallback(x_dict['vehicle'])
#             if 'rsu' in x_dict: out['rsu'] = self.rsu_fallback(x_dict['rsu'])
#             return out
#
#         x = x_dict
#         for i, conv in enumerate(self.layers):
#             # 必须传入 edge_attr_dict
#             x = conv(x, edge_index_dict, edge_attr_dict=edge_attr_dict)
#
#             if i < self.num_layers - 1:
#                 x = {key: F.elu(val) for key, val in x.items()}
#                 # x = {key: F.dropout(val, p=0.2, training=self.training) for key, val in x.items()}
#
#         if 'vehicle' in x: x['vehicle'] = self.ln_veh(x['vehicle'])
#         if 'rsu' in x: x['rsu'] = self.ln_rsu(x['rsu'])
#
#         return x
#

# =========================================================================
# 2. 策略网络 (Actor & Critic)
# =========================================================================

class TransformerHybridActor(nn.Module):
    """
    [Actor]
    融合 DAG 特征与网络拓扑特征，输出卸载决策和功率控制。
    """

    def __init__(self, task_feat_dim, veh_feat_dim, rsu_feat_dim,
                 embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerHybridActor, self).__init__()

        self.embed_dim = embed_dim
        # Encoders (使用上面的增强版)
        self.task_encoder = DAGTaskEncoder(task_feat_dim, embed_dim, embed_dim, num_layers=num_layers)
        self.topo_encoder = HeteroTopologyEncoder(veh_feat_dim, rsu_feat_dim, embed_dim, embed_dim,
                                                  num_layers=num_layers)

        # Power Head (功率控制)
        # 输入: Task Global (Embed) + Self State (Embed)
        self.power_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出归一化功率 [0, 1]
        )

    def forward(self, dag_batch, topo_batch, candidates_mask=None):
        batch_size = dag_batch.num_graphs

        # 1. Encoding
        # Task: [Total_Subtasks, Dim]
        task_node_emb, task_global_emb = self.task_encoder(dag_batch.x, dag_batch.edge_index, dag_batch.batch)

        # Topology: {'vehicle': [Total_Veh, Dim], 'rsu': ...}
        # [关键] 传入 edge_attr_dict 以利用边特征
        topo_node_dict = self.topo_encoder(
            topo_batch.x_dict,
            topo_batch.edge_index_dict,
            topo_batch.edge_attr_dict
        )
        veh_emb = topo_node_dict['vehicle']
        rsu_emb = topo_node_dict['rsu']

        # 2. Batch Loop (处理变长图 Masking)
        logits_list = []
        power_list = []

        dag_ptr = dag_batch.ptr
        veh_ptr = topo_batch['vehicle'].ptr
        rsu_ptr = topo_batch['rsu'].ptr

        for i in range(batch_size):
            # --- A. 任务侧 ---
            t_start, t_end = dag_ptr[i], dag_ptr[i + 1]
            curr_task_emb = task_node_emb[t_start:t_end]  # [Num_Sub, Dim]

            # 获取 Mask
            if hasattr(dag_batch, 'mask'):
                curr_task_mask = dag_batch.mask[t_start:t_end]
            else:
                curr_task_mask = torch.ones(curr_task_emb.size(0), dtype=torch.bool, device=curr_task_emb.device)

            # 获取物理可达 Mask (Target Mask)
            if hasattr(dag_batch, 'target_mask'):
                tm = dag_batch.target_mask[t_start:t_end]
                # 确保维度是 [Num_Sub, Num_Targets]
                curr_target_mask = tm if tm.dim() > 1 else tm.unsqueeze(0).expand(curr_task_emb.size(0), -1)
            else:
                curr_target_mask = None

            # --- B. 拓扑侧 ---
            v_start, v_end = veh_ptr[i], veh_ptr[i + 1]
            r_start, r_end = rsu_ptr[i], rsu_ptr[i + 1]
            curr_veh_emb = veh_emb[v_start:v_end]
            curr_rsu_emb = rsu_emb[r_start:r_end]

            # 拼接目标集合: [RSU(1) + Vehicles(N)]
            # 注意: 这里 vehicle 包含 Self 和 Neighbors
            targets_emb = torch.cat([curr_rsu_emb, curr_veh_emb], dim=0)

            # --- C. Attention 匹配 (Pointer Network) ---
            # Q: Task, K: Targets
            # Scores: [Num_Sub, Num_Targets]
            scores = torch.matmul(curr_task_emb, targets_emb.t()) / (self.embed_dim ** 0.5)

            # Mask 1: 屏蔽不可调度的任务 (Row Mask)
            scores = scores.masked_fill(~curr_task_mask.unsqueeze(1), -1e9)

            # Mask 2: 屏蔽物理不可达的目标 (Col Mask)
            if curr_target_mask is not None:
                # 安全切片，防止 target_mask 维度与 targets_emb 不一致
                min_tgt = min(scores.size(1), curr_target_mask.size(1))
                scores[:, :min_tgt] = scores[:, :min_tgt].masked_fill(~curr_target_mask[:, :min_tgt], -1e9)

            logits_list.append(scores.flatten())

            # --- D. 功率控制 ---
            curr_task_global = task_global_emb[i:i + 1]

            # [优化] 获取"自身"的 Embedding
            # 在 data_utils 构建的 local_topo 中，Vehicles 的第 0 个通常是 Self
            # 使用 curr_veh_emb[0:1] 比 mean 更精准地代表 Agent 自己的状态
            self_veh_feat = curr_veh_emb[0:1]

            p_feat = torch.cat([curr_task_global, self_veh_feat], dim=1)
            power_list.append(self.power_head(p_feat))

        # 3. Output Assembly
        # Pad logits to same length for batch processing
        padded_logits = nn.utils.rnn.pad_sequence(logits_list, batch_first=True, padding_value=-1e16)
        power_tensor = torch.cat(power_list, dim=0)

        return padded_logits, None, power_tensor


class TransformerHybridCritic(nn.Module):
    """
    [Critic]
    评估状态价值 V(s)。
    """

    def __init__(self, task_feat_dim, veh_feat_dim, rsu_feat_dim,
                 embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerHybridCritic, self).__init__()

        self.task_encoder = DAGTaskEncoder(task_feat_dim, embed_dim, embed_dim, num_layers=num_layers)
        self.topo_encoder = HeteroTopologyEncoder(veh_feat_dim, rsu_feat_dim, embed_dim, embed_dim,
                                                  num_layers=num_layers)

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, dag_batch, topo_batch):
        # 1. DAG Global Feature
        _, task_global_emb = self.task_encoder(dag_batch.x, dag_batch.edge_index, dag_batch.batch)

        # 2. Topology Global Feature
        # [关键] 传入 edge_attr_dict
        topo_node_dict = self.topo_encoder(
            topo_batch.x_dict,
            topo_batch.edge_index_dict,
            topo_batch.edge_attr_dict
        )

        veh_emb = topo_node_dict['vehicle']
        rsu_emb = topo_node_dict['rsu']

        # 聚合所有车辆信息 (Global Mean Pool)
        if hasattr(topo_batch['vehicle'], 'batch'):
            veh_global = global_mean_pool(veh_emb, topo_batch['vehicle'].batch)
        else:
            veh_global = veh_emb.mean(dim=0, keepdim=True)

        # 聚合 RSU 信息
        if hasattr(topo_batch['rsu'], 'batch') and topo_batch['rsu'].batch is not None:
            rsu_global = global_mean_pool(rsu_emb, topo_batch['rsu'].batch)
        else:
            # 如果 RSU batch 属性丢失 (常发生于单图 Batch)，取平均
            rsu_mean = torch.mean(rsu_emb, dim=0, keepdim=True)
            rsu_global = rsu_mean.expand(veh_global.size(0), -1)

        # 融合网络上下文
        net_context = veh_global + rsu_global

        # 3. Value Prediction
        cat_feat = torch.cat([task_global_emb, net_context], dim=1)
        value = self.value_head(cat_feat)

        return value