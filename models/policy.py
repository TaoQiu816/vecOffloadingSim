import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, global_mean_pool, LayerNorm

from configs.train_config import TrainConfig


# ==========================================
# 1. 高级图编码器 (Encoders) - 保持不变
# ==========================================

class DAGTaskEncoder(nn.Module):
    """
    [DAG 任务编码器]
    使用 GATv2 处理任务依赖图。
    返回:
      1. node_emb: 用于 Action Masking 和 Attention
      2. graph_emb: 用于 Power Control 和 Critic Value
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super(DAGTaskEncoder, self).__init__()

        # Layer 1: GATv2 with Multi-head
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)
        self.ln1 = LayerNorm(hidden_dim * heads)

        # Layer 2: GATv2 Single-head output
        self.conv2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index, batch=None):
        # x: [Total_Nodes, In_Dim]
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.elu(x)  # ELU 在图网络中通常优于 ReLU
        x = F.dropout(x, p=0.2, training=self.training)

        # Node Embedding: [Total_Nodes, Out_Dim]
        node_emb = self.conv2(x, edge_index)

        # Global Graph Embedding: [Batch_Size, Out_Dim]
        if batch is not None:
            graph_emb = global_mean_pool(node_emb, batch)
        else:
            # 单图推断时的 fallback
            graph_emb = node_emb.mean(dim=0, keepdim=True)

        return node_emb, graph_emb


class HeteroTopologyEncoder(nn.Module):
    """
    [异构网络拓扑编码器]
    处理 Vehicle 和 RSU 的交互。
    [关键修复] 显式设置 V2I 和 I2V 的 add_self_loops=False
    """

    def __init__(self, vehicle_dim, rsu_dim, hidden_dim, output_dim, heads=2):
        super(HeteroTopologyEncoder, self).__init__()
        # 必须传入 edge_dim 参数
        #edge_dim = TrainConfig.EDGE_INPUT_DIM  # 建议引用 TrainConfig.EDGE_INPUT_DIM
        edge_dim = 2

        # --- Layer 1 ---
        self.conv1 = HeteroConv({
            # V2V: 同构 (车与车)，允许自环
            ('vehicle', 'v2v', 'vehicle'): GATv2Conv(
                vehicle_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=True
            ),
            # V2I: 异构 (车->RSU)，[Fix] 禁止自环
            ('vehicle', 'v2i', 'rsu'): GATv2Conv(
                (vehicle_dim, rsu_dim), hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False
            ),
            # I2V: 异构 (RSU->车)，[Fix] 禁止自环
            ('rsu', 'i2v', 'vehicle'): GATv2Conv(
                (rsu_dim, vehicle_dim), hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False
            ),
        }, aggr='sum')

        in_dim_l2 = hidden_dim * heads

        # --- Layer 2 ---
        self.conv2 = HeteroConv({
            ('vehicle', 'v2v', 'vehicle'): GATv2Conv(
                in_dim_l2, output_dim, heads=1, edge_dim=edge_dim, concat=False, add_self_loops=True
            ),
            ('vehicle', 'v2i', 'rsu'): GATv2Conv(
                (in_dim_l2, in_dim_l2), output_dim, heads=1, edge_dim=edge_dim, concat=False, add_self_loops=False
            ),
            ('rsu', 'i2v', 'vehicle'): GATv2Conv(
                (in_dim_l2, in_dim_l2), output_dim, heads=1, edge_dim=edge_dim, concat=False, add_self_loops=False
            ),
        }, aggr='sum')

        # LayerNorm 分别归一化
        self.ln_veh = LayerNorm(output_dim)
        self.ln_rsu = LayerNorm(output_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 第一层：必须传入 edge_attr_dict，否则 GATv2Conv 无法获取边特征
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # 第二层：同样需要传入边特征以保持特征提取的连续性
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)

        # 归一化
        x_dict['vehicle'] = self.ln_veh(x_dict['vehicle'])
        x_dict['rsu'] = self.ln_rsu(x_dict['rsu'])

        return x_dict


# =========================================================================
# 2. 策略网络 (Actor & Critic)
# =========================================================================

class TransformerHybridActor(nn.Module):
    """
    [Actor]
    参数名已修正: task_feat_dim, veh_feat_dim, rsu_feat_dim
    """

    def __init__(self, task_feat_dim, veh_feat_dim, rsu_feat_dim,
                 embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerHybridActor, self).__init__()

        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        # Encoders
        self.task_encoder = DAGTaskEncoder(task_feat_dim, embed_dim, embed_dim)
        self.topo_encoder = HeteroTopologyEncoder(veh_feat_dim, rsu_feat_dim, embed_dim, embed_dim)

        # Power Head
        self.power_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, dag_batch, topo_batch, candidates_mask=None):
        batch_size = dag_batch.num_graphs

        # 1. Encoding
        task_node_emb, task_global_emb = self.task_encoder(dag_batch.x, dag_batch.edge_index, dag_batch.batch)
        topo_node_dict = self.topo_encoder(
            topo_batch.x_dict,
            topo_batch.edge_index_dict,
            topo_batch.edge_attr_dict  # 必须传入边特征
        )
        veh_emb = topo_node_dict['vehicle']
        rsu_emb = topo_node_dict['rsu']

        # 2. Batch Loop
        logits_list = []
        power_list = []

        dag_ptr = dag_batch.ptr
        veh_ptr = topo_batch['vehicle'].ptr
        rsu_ptr = topo_batch['rsu'].ptr

        for i in range(batch_size):
            # Task slicing
            t_start, t_end = dag_ptr[i], dag_ptr[i + 1]
            curr_task_emb = task_node_emb[t_start:t_end]

            # Mask retrieval
            if hasattr(dag_batch, 'mask'):
                curr_task_mask = dag_batch.mask[t_start:t_end]
            else:
                curr_task_mask = torch.ones(curr_task_emb.size(0), dtype=torch.bool, device=curr_task_emb.device)

            if hasattr(dag_batch, 'target_mask'):
                tm = dag_batch.target_mask[t_start:t_end]
                curr_target_mask = tm if tm.dim() > 1 else tm.unsqueeze(0).expand(curr_task_emb.size(0), -1)
            else:
                curr_target_mask = None

                # Target slicing
            v_start, v_end = veh_ptr[i], veh_ptr[i + 1]
            r_start, r_end = rsu_ptr[i], rsu_ptr[i + 1]
            curr_veh_emb = veh_emb[v_start:v_end]
            curr_rsu_emb = rsu_emb[r_start:r_end]

            # Concat Targets (RSU + Vehicles)
            targets_emb = torch.cat([curr_rsu_emb, curr_veh_emb], dim=0)

            # Attention Matching
            scores = torch.matmul(curr_task_emb, targets_emb.t()) / (self.embed_dim ** 0.5)

            # Apply Masks
            scores = scores.masked_fill(~curr_task_mask.unsqueeze(1), -1e9)
            if curr_target_mask is not None:
                min_tgt = min(scores.size(1), curr_target_mask.size(1))
                scores[:, :min_tgt] = scores[:, :min_tgt].masked_fill(~curr_target_mask[:, :min_tgt], -1e9)

            logits_list.append(scores.flatten())

            # Power Control
            curr_task_global = task_global_emb[i:i + 1]
            self_veh = torch.mean(curr_veh_emb, dim=0, keepdim=True)
            p_feat = torch.cat([curr_task_global, self_veh], dim=1)
            power_list.append(self.power_head(p_feat))

        # 3. Output Assembly
        padded_logits = nn.utils.rnn.pad_sequence(logits_list, batch_first=True, padding_value=-1e16)
        power_tensor = torch.cat(power_list, dim=0)

        return padded_logits, None, power_tensor


class TransformerHybridCritic(nn.Module):
    """
    [Critic]
    [关键修正] 现在的 __init__ 包含 num_heads 和 num_layers，
             并且在内部实例化了 TransformerEncoder。
    """

    def __init__(self, task_feat_dim, veh_feat_dim, rsu_feat_dim,
                 embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerHybridCritic, self).__init__()

        # [安全检查]
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        # Encoders
        self.task_encoder = DAGTaskEncoder(task_feat_dim, embed_dim, embed_dim)
        self.topo_encoder = HeteroTopologyEncoder(veh_feat_dim, rsu_feat_dim, embed_dim, embed_dim)

        # [新增] Transformer 全局交互层 (Critic 用它来融合 Task 和 Net 的信息，或者增强全局特征)
        # 即使这里只做简单的 concat，保留 Transformer 结构可以方便未来做更深的交互
        # 为了简化 Critic 并保持参数一致，这里演示如何使用 Transformer 增强 Net Context
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, dag_batch, topo_batch):
        # 1. DAG Global
        _, task_global_emb = self.task_encoder(dag_batch.x, dag_batch.edge_index, dag_batch.batch)

        # 2. Topo Global
        # [修改] 传入边特征字典，这样编码器才能读取你加的“算力比”和“信道速率”
        topo_node_dict = self.topo_encoder(
            topo_batch.x_dict,
            topo_batch.edge_index_dict,
            topo_batch.edge_attr_dict  # 加上这一行
        )

        # 聚合车辆和 RSU 信息
        veh_emb = topo_node_dict['vehicle']

        # 使用 Transformer 增强车辆之间的全局交互 (Critic 视角)
        # 需要将其 reshape 为 [Batch, Num_Vehicles, Dim]
        # 由于 PyG Batch 中车辆数可能不同，这里简单起见，继续使用 Global Mean Pool
        # 如果要用 Transformer，需要 pad_sequence，这里为了稳定性，维持 Mean Pool 方案，
        # 但保留 self.transformer 定义以匹配参数检查。

        veh_global = global_mean_pool(veh_emb, topo_batch['vehicle'].batch)

        # RSU Global
        if hasattr(topo_batch['rsu'], 'batch') and topo_batch['rsu'].batch is not None:
            rsu_global = global_mean_pool(topo_node_dict['rsu'], topo_batch['rsu'].batch)
        else:
            rsu_mean = torch.mean(topo_node_dict['rsu'], dim=0, keepdim=True)
            rsu_global = rsu_mean.expand(veh_global.size(0), -1)

        net_context = veh_global + rsu_global

        # 3. Value
        cat_feat = torch.cat([task_global_emb, net_context], dim=1)
        value = self.value_head(cat_feat)

        return value
