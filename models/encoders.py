import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, global_mean_pool, LayerNorm


class DAGTaskEncoder(nn.Module):
    """
    [DAG 任务编码器]

    功能: 提取 DAG 任务的节点级特征和图级特征。
    架构: GATv2 (动态注意力) + Global Pooling
    配置: 支持动态层数 (num_layers)，使用 ModuleList 构建。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=2, edge_dim=0):
        """
        Args:
            edge_dim: 即使 DAG 暂时不用边特征，保留此接口以保持 GATv2Conv 调用的兼容性 (设为 None 或 0)。
        """
        super(DAGTaskEncoder, self).__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers

        # --- Layer 1: Input -> Hidden ---
        # DAG 边特征通常为空，故 edge_dim=None
        self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=None))
        self.norms.append(LayerNorm(hidden_dim * heads))

        # --- Layer 2..N-1: Hidden -> Hidden ---
        mid_dim = hidden_dim * heads
        for _ in range(num_layers - 2):
            self.layers.append(GATv2Conv(mid_dim, hidden_dim, heads=heads, concat=True, edge_dim=None))
            self.norms.append(LayerNorm(mid_dim))

        # --- Layer N: Hidden -> Output ---
        self.layers.append(GATv2Conv(mid_dim, output_dim, heads=1, concat=False, edge_dim=None))
        self.last_norm = LayerNorm(output_dim)

    def forward(self, x, edge_index, batch=None):
        # 循环前 N-1 层
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # 最后一层
        node_emb = self.layers[-1](x, edge_index)

        # Global Pooling (图级特征)
        if batch is not None:
            graph_emb = global_mean_pool(node_emb, batch)
        else:
            graph_emb = node_emb.mean(dim=0, keepdim=True)

        return node_emb, graph_emb


class HeteroTopologyEncoder(nn.Module):
    """
    [异构网络拓扑编码器 - 完整修复版]

    关键特性:
    1. edge_dim 参数化: 从外部接收边特征维度，不硬编码 Config。
    2. Fallback 机制: 处理孤立节点 (无边) 的情况。
    3. 异构图处理: 区分 V2V (自环) 和 V2I/I2V (无自环)。
    """

    def __init__(self, vehicle_dim, rsu_dim, edge_dim, hidden_dim, output_dim, num_layers=2, heads=2):
        """
        Args:
            edge_dim (int): 边特征维度 (必须传入，对应 Bandwidth 和 CpuRatio)。
        """
        super(HeteroTopologyEncoder, self).__init__()

        self.num_layers = num_layers

        # [新增] 兜底投影层 (处理孤立节点，防止维度不匹配)
        self.veh_fallback = nn.Linear(vehicle_dim, output_dim)
        self.rsu_fallback = nn.Linear(rsu_dim, output_dim)

        self.layers = nn.ModuleList()

        # --- Layer 1 ---
        # 必须显式传入 edge_dim
        self.layers.append(HeteroConv({
            ('vehicle', 'v2v', 'vehicle'): GATv2Conv(
                vehicle_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=True
            ),
            ('vehicle', 'v2i', 'rsu'): GATv2Conv(
                (vehicle_dim, rsu_dim), hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False
            ),
            ('rsu', 'i2v', 'vehicle'): GATv2Conv(
                (rsu_dim, vehicle_dim), hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False
            ),
        }, aggr='sum'))

        # --- Middle & Last Layers ---
        mid_dim = hidden_dim * heads
        for i in range(1, num_layers):
            is_last = (i == num_layers - 1)
            out_d = output_dim if is_last else hidden_dim
            n_heads = 1 if is_last else heads
            do_concat = not is_last

            self.layers.append(HeteroConv({
                ('vehicle', 'v2v', 'vehicle'): GATv2Conv(
                    mid_dim, out_d, heads=n_heads, edge_dim=edge_dim, concat=do_concat, add_self_loops=True
                ),
                ('vehicle', 'v2i', 'rsu'): GATv2Conv(
                    (mid_dim, mid_dim), out_d, heads=n_heads, edge_dim=edge_dim, concat=do_concat, add_self_loops=False
                ),
                ('rsu', 'i2v', 'vehicle'): GATv2Conv(
                    (mid_dim, mid_dim), out_d, heads=n_heads, edge_dim=edge_dim, concat=do_concat, add_self_loops=False
                ),
            }, aggr='sum'))

        self.ln_veh = LayerNorm(output_dim)
        self.ln_rsu = LayerNorm(output_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # [关键] 鲁棒性检查: 如果没有边，直接使用线性投影
        has_edges = len(edge_index_dict) > 0 and any(e.size(1) > 0 for e in edge_index_dict.values())

        if not has_edges:
            out = {}
            if 'vehicle' in x_dict: out['vehicle'] = self.veh_fallback(x_dict['vehicle'])
            if 'rsu' in x_dict: out['rsu'] = self.rsu_fallback(x_dict['rsu'])
            return out

        x = x_dict
        for i, conv in enumerate(self.layers):
            # 必须传入 edge_attr_dict
            x = conv(x, edge_index_dict, edge_attr_dict=edge_attr_dict)

            if i < self.num_layers - 1:
                x = {key: F.elu(val) for key, val in x.items()}
                # dropout 可选
                # x = {key: F.dropout(val, p=0.2, training=self.training) for key, val in x.items()}

        if 'vehicle' in x: x['vehicle'] = self.ln_veh(x['vehicle'])
        if 'rsu' in x: x['rsu'] = self.ln_rsu(x['rsu'])

        return x