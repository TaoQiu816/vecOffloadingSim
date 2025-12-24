import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, global_mean_pool, LayerNorm


class DAGTaskEncoder(nn.Module):
    """
    [DAG 任务编码器]

    功能:
    1. 提取每个子任务的节点特征 (用于动作 Mask 和 Transformer Query)。
    2. 提取整个 DAG 的全局特征 (用于 Critic 价值评估和功率控制)。

    架构: GATv2 (动态注意力) + Global Pooling
    理由: GATv2 比普通 GAT 更强，能处理"对于不同查询节点，邻居的重要性是动态变化"的情况。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super(DAGTaskEncoder, self).__init__()

        # 第一层 GATv2: 聚合一阶邻居 (前驱/后继)
        # concat=True -> 输出维度: hidden_dim * heads
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True)

        # LayerNorm: 防止层数加深后梯度消失/爆炸，加速收敛
        self.ln1 = LayerNorm(hidden_dim * heads)

        # 第二层 GATv2: 聚合二阶邻居
        # concat=False -> 输出维度: output_dim
        self.conv2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: 节点特征矩阵 [Total_Nodes, Input_Dim]
            edge_index: 边索引 [2, Num_Edges]
            batch: (可选) PyG batch 向量，指示每个节点属于哪个图 [Total_Nodes]

        Returns:
            node_emb: [Total_Nodes, Output_Dim] (节点级特征)
            graph_emb: [Batch_Size, Output_Dim] (图级特征)
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.elu(x)  # ELU 通常比 ReLU 在 GNN 中表现更好
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 2
        node_emb = self.conv2(x, edge_index)

        # Global Pooling (聚合图级特征)
        if batch is not None:
            # 使用 Mean Pooling 获取图的整体语义
            graph_emb = global_mean_pool(node_emb, batch)
        else:
            # 如果没有 batch (单图推断)，直接取平均
            graph_emb = node_emb.mean(dim=0, keepdim=True)

        return node_emb, graph_emb


class HeteroTopologyEncoder(nn.Module):
    """
    [异构网络拓扑编码器]

    功能: 处理 V2X 网络中的异构节点 (Vehicle, RSU) 和异构边 (V2V, V2I, I2V)。
    架构: HeteroConv (PyG 异构图卷积)

    输入数据说明:
    由于 Vehicle 和 RSU 的特征维度不同 (self_info=6, rsu_info=1)，
    我们需要 HeteroConv 来分别处理它们的投影。
    """

    def __init__(self, vehicle_dim, rsu_dim, hidden_dim, output_dim, heads=2):
        super(HeteroTopologyEncoder, self).__init__()

        # --- 第一层异构卷积 ---
        # 针对每种边类型定义不同的 GAT 算子
        self.conv1 = HeteroConv({

            # 1. V2V (Vehicle-to-Vehicle): 同构交互
            # 车辆间交换位置、负载、速度信息
            ('vehicle', 'v2v', 'vehicle'): GATv2Conv(vehicle_dim, hidden_dim,
                                                     heads=heads, concat=True, add_self_loops=True),

            # 2. V2I (Vehicle-to-RSU): 上报信息
            # 车辆将自身状态传给 RSU (RSU 聚合周围车辆信息)
            # (-1, -1) 表示 Lazy Init，PyG 会根据输入自动推断源/目标维度
            ('vehicle', 'v2i', 'rsu'): GATv2Conv((vehicle_dim, rsu_dim), hidden_dim,
                                                 heads=heads, concat=True, add_self_loops=False),

            # 3. I2V (RSU-to-Vehicle): 广播/反馈
            # RSU 将全局拥堵信息反馈给车辆
            ('rsu', 'i2v', 'vehicle'): GATv2Conv((rsu_dim, vehicle_dim), hidden_dim,
                                                 heads=heads, concat=True, add_self_loops=False),

        }, aggr='sum')  # Sum 聚合能模拟无线信道的"叠加干扰"特性

        # 中间层维度 (concat=True 后)
        in_dim_l2 = hidden_dim * heads

        # --- 第二层异构卷积 ---
        self.conv2 = HeteroConv({
            ('vehicle', 'v2v', 'vehicle'): GATv2Conv(in_dim_l2, output_dim, heads=1, concat=False),
            ('vehicle', 'v2i', 'rsu'): GATv2Conv(in_dim_l2, output_dim, heads=1, concat=False),
            ('rsu', 'i2v', 'vehicle'): GATv2Conv(in_dim_l2, output_dim, heads=1, concat=False),
        }, aggr='sum')

        # 归一化层 (针对不同类型的节点分别归一化)
        self.ln_veh = LayerNorm(output_dim)
        self.ln_rsu = LayerNorm(output_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict (dict): {'vehicle': [N_v, 6], 'rsu': [N_r, 1]}
            edge_index_dict (dict): {('vehicle','v2v','vehicle'): [2, E], ...}

        Returns:
            out_dict (dict): {'vehicle': [N_v, Out_Dim], 'rsu': [N_r, Out_Dim]}
            这些 Embedding 将被传入 Transformer 作为 Key/Value
        """
        # [鲁棒性检查] 防止某些 Episode 中没有 V2V 边导致 Crash
        if len(edge_index_dict) == 0:
            # 如果没有边，直接做一个线性投影返回 (Skip Connection)
            # 注意: 这里简化处理，实际工程中最好保留投影层
            return x_dict

            # Layer 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}

        # Layer 2
        x_dict = self.conv2(x_dict, edge_index_dict)

        # Layer Norm (分类处理)
        if 'vehicle' in x_dict:
            x_dict['vehicle'] = self.ln_veh(x_dict['vehicle'])
        if 'rsu' in x_dict:
            x_dict['rsu'] = self.ln_rsu(x_dict['rsu'])

        return x_dict