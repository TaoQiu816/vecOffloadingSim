"""
DAG拓扑特征计算模块

功能：
- 计算前向层级（Forward Level）：从入口节点到当前节点的最大跳数
- 计算后向层级（Backward Level）：从当前节点到出口节点的最大跳数
- 计算最短路径距离矩阵：任意两点之间的最短路径跳数
- 计算Rank（GA-DRL启发）：用于注意力偏置的节点重要性先验
"""

import numpy as np
from typing import Tuple, Optional


def compute_forward_levels(adj_matrix: np.ndarray) -> np.ndarray:
    """
    计算前向层级（Forward Level）：从任意入口节点到节点i的最大跳数
    
    算法：使用BFS从所有入度为0的节点开始，计算到每个节点的最长路径
    
    Args:
        adj_matrix: 邻接矩阵 (NxN), adj[i][j]=1 表示 i -> j
        
    Returns:
        np.ndarray: [N], L_fwd[i] 表示从入口节点到节点i的最大跳数
    """
    N = adj_matrix.shape[0]
    L_fwd = np.zeros(N, dtype=int)
    
    # 计算入度
    in_degree = np.sum(adj_matrix, axis=0)
    
    # 找到所有入口节点（入度为0）
    entry_nodes = np.where(in_degree == 0)[0]
    if len(entry_nodes) == 0:
        # 如果没有入口节点（环？），返回全0
        return L_fwd
    
    # BFS队列：存储(node_id, current_level)
    from collections import deque
    queue = deque([(node, 0) for node in entry_nodes])
    visited = set(entry_nodes)
    
    # 初始化入口节点的层级为0
    for node in entry_nodes:
        L_fwd[node] = 0
    
    while queue:
        current, level = queue.popleft()
        
        # 遍历当前节点的所有后继节点
        successors = np.where(adj_matrix[current] > 0)[0]
        for succ in successors:
            new_level = level + 1
            # 更新层级（取最大值，因为是最大跳数）
            if new_level > L_fwd[succ]:
                L_fwd[succ] = new_level
            
            # 如果这个后继节点的所有前驱都已访问，将其加入队列
            # 但实际上，对于DAG，我们可以直接继续
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, new_level))
            else:
                # 如果已经访问过，但层级可能更新，需要重新处理其后继
                # 为了简化，这里直接加入队列（DAG保证不会无限循环）
                queue.append((succ, L_fwd[succ]))
    
    return L_fwd


def compute_backward_levels(adj_matrix: np.ndarray) -> np.ndarray:
    """
    计算后向层级（Backward Level）：从节点i到任意出口节点的最大跳数
    
    算法：在反向图上使用BFS，从所有出度为0的节点开始
    
    Args:
        adj_matrix: 邻接矩阵 (NxN), adj[i][j]=1 表示 i -> j
        
    Returns:
        np.ndarray: [N], L_bwd[i] 表示从节点i到出口节点的最大跳数
    """
    N = adj_matrix.shape[0]
    L_bwd = np.zeros(N, dtype=int)
    
    # 构建反向图（转置邻接矩阵）
    reverse_adj = adj_matrix.T
    
    # 计算出度（在原图中）
    out_degree = np.sum(adj_matrix, axis=1)
    
    # 找到所有出口节点（出度为0，在反向图中是入口节点）
    exit_nodes = np.where(out_degree == 0)[0]
    if len(exit_nodes) == 0:
        # 如果没有出口节点，返回全0
        return L_bwd
    
    # BFS队列：存储(node_id, current_level)
    from collections import deque
    queue = deque([(node, 0) for node in exit_nodes])
    visited = set(exit_nodes)
    
    # 初始化出口节点的层级为0
    for node in exit_nodes:
        L_bwd[node] = 0
    
    while queue:
        current, level = queue.popleft()
        
        # 遍历当前节点的所有前驱节点（在反向图中是后继）
        predecessors = np.where(reverse_adj[current] > 0)[0]
        for pred in predecessors:
            new_level = level + 1
            # 更新层级（取最大值）
            if new_level > L_bwd[pred]:
                L_bwd[pred] = new_level
            
            if pred not in visited:
                visited.add(pred)
                queue.append((pred, new_level))
            else:
                queue.append((pred, L_bwd[pred]))
    
    return L_bwd


def compute_shortest_path_matrix(adj_matrix: np.ndarray, max_nodes: int) -> np.ndarray:
    """
    计算最短路径距离矩阵：任意两点之间的最短路径跳数
    
    算法：对DAG，使用N次BFS（每个节点做一次起点）
    对于不连通的节点对，距离设为max_nodes
    
    Args:
        adj_matrix: 邻接矩阵 (NxN)
        max_nodes: 最大节点数（用于设置不连通值）
        
    Returns:
        np.ndarray: [N, N], Delta[i][j] 表示从节点i到节点j的最短路径跳数
                   如果不连通，值为max_nodes
    """
    N = adj_matrix.shape[0]
    Delta = np.full((N, N), max_nodes, dtype=int)
    
    # 对角线：节点到自己的距离为0
    np.fill_diagonal(Delta, 0)
    
    # 对每个节点作为起点，使用BFS计算到其他所有节点的最短距离
    from collections import deque
    
    for start in range(N):
        # BFS队列：存储(node_id, distance)
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current, dist = queue.popleft()
            
            # 遍历当前节点的所有后继节点
            successors = np.where(adj_matrix[current] > 0)[0]
            for succ in successors:
                if succ not in visited:
                    visited.add(succ)
                    new_dist = dist + 1
                    Delta[start, succ] = new_dist
                    queue.append((succ, new_dist))
    
    return Delta


def normalize_distance_matrix(dist_matrix: np.ndarray, max_nodes: int) -> np.ndarray:
    """
    归一化距离矩阵：确保不连通值统一为max_nodes
    
    在输出给神经网络之前，必须执行此操作
    
    Args:
        dist_matrix: 距离矩阵 (NxN)
        max_nodes: 最大节点数（不连通值）
        
    Returns:
        np.ndarray: 归一化后的距离矩阵，不连通值统一为max_nodes
    """
    normalized = dist_matrix.copy()
    
    # 将大于max_nodes的值设为max_nodes（包括infinity）
    normalized[normalized > max_nodes] = max_nodes
    normalized[normalized < 0] = max_nodes  # 负数也视为不连通
    
    # 处理NaN和infinity
    normalized = np.nan_to_num(normalized, nan=max_nodes, posinf=max_nodes, neginf=max_nodes)
    
    return normalized.astype(int)


def compute_rank(adj_matrix: np.ndarray,
                 comp_array: np.ndarray,
                 data_matrix: np.ndarray,
                 mean_cpu_freq: float = 2.0e9,
                 mean_rate: float = 20e6,
                 exec_cost_mode: str = 'mean_cpu',
                 comm_cost_mode: str = 'mean_rate') -> np.ndarray:
    """
    计算DAG节点的Rank值（GA-DRL启发的邻居重要性先验）

    【Rank定义与语义】
    递推公式: rank[i] = max_{j in preds[i]}( rank[j] + exec_cost(j) + comm_cost(j,i) )
    入口节点: rank = 0
    语义: rank越小 => 越靠近DAG入口 => 优先级越高 => 注意力偏置越大

    【与调度优先级的区别】
    - 调度优先级(L_bwd)：用于决定调度顺序，基于后向层级
    - Rank：用于注意力偏置，基于前向累积代价，强调关键路径上的瓶颈节点

    Args:
        adj_matrix: 邻接矩阵 (NxN), adj[i][j]=1 表示 i -> j
        comp_array: 节点计算量数组 [N], 单位cycles
        data_matrix: 边数据量矩阵 (NxN), 单位bits
        mean_cpu_freq: 平均CPU频率 (Hz), 用于static模式
        mean_rate: 平均通信速率 (bps), 用于static模式
        exec_cost_mode: exec_cost估计方式
            - 'mean_cpu': comp / mean_cpu_freq（推荐）
            - 'cycles': 直接使用comp cycles（需要后续归一化）
        comm_cost_mode: comm_cost估计方式
            - 'mean_rate': edge_data / mean_rate（推荐）
            - 'zero': 忽略通信开销

    Returns:
        np.ndarray: [N], 每个节点的rank值（未归一化）
                   rank越小表示越重要（越靠近入口）
    """
    N = adj_matrix.shape[0]
    rank = np.zeros(N, dtype=np.float64)

    # 计算exec_cost（节点计算代价）
    if exec_cost_mode == 'mean_cpu':
        exec_cost = comp_array / max(mean_cpu_freq, 1e-6)  # 秒
    elif exec_cost_mode == 'cycles':
        exec_cost = comp_array / 1e9  # 归一化到Gcycles
    else:
        exec_cost = comp_array / max(mean_cpu_freq, 1e-6)

    # 计算入度（找前驱）
    in_degree = np.sum(adj_matrix, axis=0)

    # 拓扑排序 + 递推计算rank
    # 使用Kahn算法的变体
    from collections import deque

    # 初始化：入口节点（入度为0）的rank为0
    remaining_in_degree = in_degree.copy()
    queue = deque()

    for i in range(N):
        if remaining_in_degree[i] == 0:
            queue.append(i)
            rank[i] = 0.0

    # 拓扑遍历
    while queue:
        current = queue.popleft()

        # 找到current的所有后继节点
        successors = np.where(adj_matrix[current] > 0)[0]

        for succ in successors:
            # 计算从current到succ的边通信代价
            if comm_cost_mode == 'mean_rate':
                edge_data = data_matrix[current, succ] if data_matrix is not None else 0
                comm_cost = edge_data / max(mean_rate, 1e-6)  # 秒
            else:
                comm_cost = 0.0

            # 递推公式: rank[succ] = max(..., rank[current] + exec_cost[current] + comm_cost)
            new_rank = rank[current] + exec_cost[current] + comm_cost
            rank[succ] = max(rank[succ], new_rank)

            # 减少后继的剩余入度
            remaining_in_degree[succ] -= 1

            # 如果后继的所有前驱都处理完，加入队列
            if remaining_in_degree[succ] == 0:
                queue.append(succ)

    return rank


def normalize_rank(rank: np.ndarray,
                   mode: str = 'minmax',
                   valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    对Rank值进行归一化

    Args:
        rank: 原始rank数组 [N]
        mode: 归一化方式
            - 'minmax': (rank - min) / (max - min + eps) -> [0,1]
            - 'zscore': (rank - mean) / (std + eps)
            - 'none': 不归一化
        valid_mask: 有效节点掩码 [N], True表示有效节点
                   仅对有效节点计算统计量

    Returns:
        np.ndarray: 归一化后的rank值 [N]
    """
    if mode == 'none':
        return rank.copy()

    if valid_mask is not None:
        valid_rank = rank[valid_mask]
    else:
        valid_rank = rank

    if len(valid_rank) == 0:
        return rank.copy()

    if mode == 'minmax':
        r_min = np.min(valid_rank)
        r_max = np.max(valid_rank)
        eps = 1e-8
        if r_max - r_min < eps:
            # 所有值相同，返回0
            normalized = np.zeros_like(rank)
        else:
            normalized = (rank - r_min) / (r_max - r_min + eps)

    elif mode == 'zscore':
        r_mean = np.mean(valid_rank)
        r_std = np.std(valid_rank)
        eps = 1e-8
        normalized = (rank - r_mean) / (r_std + eps)

    else:
        normalized = rank.copy()

    return normalized

