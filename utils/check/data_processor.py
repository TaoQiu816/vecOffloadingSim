import torch
import numpy as np
from torch_geometric.data import Data, Batch


def process_obs_to_tensor(obs_list, device='cpu'):
    """
    将环境返回的 obs_list (List[Dict]) 转换为 PyTorch Tensors。

    输入：
    - obs_list: 每个车辆的任务图信息，包括节点特征、邻接矩阵、邻居信息等。
    - device: 目标设备（默认为 'cpu'）。

    返回：
    - dag_batch: PyTorch Geometric Batch 对象，包含所有车辆的 DAG 图数据。
    - neigh_tensor: 邻居特征矩阵，经过填充后的邻居信息。
    - mask_tensor: 邻居掩码矩阵，标记有效邻居。
    """
    batch_dag_data = []  # 存放 DAG 图数据 (给 GAT)

    # 存放邻居序列 (给 Transformer)
    MAX_NEIGH = 10  # 假设最大邻居数为 10 (不足补0，超过截断)
    batch_neigh_feats = []  # 存放邻居特征
    batch_neigh_mask = []  # 存放邻居掩码

    # 存放自我特征的占位符
    batch_self_feats = []

    for obs in obs_list:
        # 1. 处理 DAG (构建 PyG Data 对象)
        x = torch.tensor(obs['node_x'], dtype=torch.float)  # 节点特征
        adj = obs['adj']  # 邻接矩阵
        rows, cols = np.nonzero(adj)  # 获取非零元素的行列索引
        edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)  # 边索引

        # 构建图对象
        graph = Data(x=x, edge_index=edge_index)
        batch_dag_data.append(graph)

        # 2. 处理邻居信息 (Padding)
        neighs = obs['neighbors']  # 邻居信息 [ID, dx, dy...]
        curr_len = len(neighs)

        padded_neigh = np.zeros((MAX_NEIGH, 3))  # 初始化邻居特征，维度为 (MAX_NEIGH, 3)
        mask = np.zeros(MAX_NEIGH + 1)  # 邻居掩码，+1 是因为还有 Self Token
        mask[0] = 1  # 自我节点总是有效的

        if curr_len > 0:
            valid_len = min(curr_len, MAX_NEIGH)  # 截断或填充邻居
            padded_neigh[:valid_len] = np.array(neighs)[:valid_len]  # 填充邻居特征
            mask[1:valid_len + 1] = 1  # 标记有效邻居

        batch_neigh_feats.append(padded_neigh)
        batch_neigh_mask.append(mask)

        # 3. 处理 Self 向量 (此处假设在网络内部进行处理，暂时不处理)
        pass

    # 打包数据，返回 PyG Batch 对象与邻居信息
    dag_batch = Batch.from_data_list(batch_dag_data).to(device)

    neigh_tensor = torch.tensor(np.array(batch_neigh_feats), dtype=torch.float).to(device)
    mask_tensor = torch.tensor(np.array(batch_neigh_mask), dtype=torch.bool).to(device)

    return dag_batch, neigh_tensor, mask_tensor
