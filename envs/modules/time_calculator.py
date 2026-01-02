"""
时间计算模块 (Time Calculator Module)

职责：
- 计算DAG任务的EST（Earliest Start Time）和CT（Completion Time）
- 计算完全本地执行时间（t_local）
- 提供拓扑排序功能

核心公式：
1. EST_i = max(处理器FAT, 上传完成时间, max(前驱CT + 传输时间))
2. CT_i = EST_i + 执行时间
3. CFT = max(终止节点的CT)

注意：
- 这是预测性计算，不修改实际的FAT状态
- 依赖数据传输使用当前车辆的发射功率估计速率
- 子任务输入数据上传使用车辆当前发射功率估计
"""

import numpy as np
from configs.config import SystemConfig as Cfg


def get_topological_order(adj_matrix):
    """
    获取DAG的拓扑排序顺序（Kahn算法）
    
    拓扑排序保证：对于任意边 (i, j)，节点i在排序中一定出现在节点j之前
    这对于按依赖关系顺序计算EST/CT是必要的。
    
    Args:
        adj_matrix: 邻接矩阵 (NxN), adj[i][j]=1 表示任务 i 指向任务 j (i -> j)
    
    Returns:
        list: 拓扑排序后的节点ID列表
    
    Example:
        >>> adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        >>> order = get_topological_order(adj)
        >>> # 结果: [0, 1, 2]（节点0是入口，节点2是出口）
    """
    n = len(adj_matrix)
    # 计算每个节点的入度（有多少前驱节点）
    in_degree = np.sum(adj_matrix, axis=0)
    
    # 初始化：将所有入度为0的节点（入口节点）加入队列
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    
    # Kahn算法：逐步处理入度为0的节点
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        
        # 处理该节点的所有后继节点：减少它们的入度
        for j in range(n):
            if adj_matrix[node, j] == 1:
                in_degree[j] -= 1
                # 如果入度变为0，说明该节点的所有前驱都已处理，可以加入队列
                if in_degree[j] == 0:
                    queue.append(j)
    
    return topo_order


def _get_vehicle_by_id(vehicles_list, veh_id):
    for veh in vehicles_list:
        if veh.id == veh_id:
            return veh
    return None


def calculate_est_ct(vehicle, dag, task_locations, channel_model, rsus, vehicles_list, current_time, v2i_user_count=None):
    """
    计算DAG任务的EST（Earliest Start Time）和CT（Completion Time）
    
    核心逻辑：
    1. 按拓扑排序顺序处理每个节点
    2. 对每个节点计算：
       - EST = max(处理器FAT, 上传完成时间, 前驱节点CT+传输时间的最大值)
       - CT = EST + 执行时间
    3. CFT = 终止节点的最大CT
    
    关键设计：
    - 这是预测性计算，只读取FAT值，不修改实际FAT状态
    - 依赖数据传输使用当前车辆的发射功率估计
    - 子任务输入数据上传的上传完成时间考虑上传信道FAT
    
    Args:
        vehicle: 车辆对象（任务所属的车辆）
        dag: DAGTask对象（包含任务依赖关系、计算量、数据量等信息）
        task_locations: 每个子任务的执行位置列表
                       ['Local' | ('RSU', rsu_id) | int(vehicle_id)]
        channel_model: 信道模型对象（用于计算传输速率）
        rsus: RSU列表
        vehicles_list: 车辆列表
        current_time: 当前仿真时间
        v2i_user_count: 估算V2I用户数（保持与环境口径一致）
    
    Returns:
        tuple: (EST数组, CT数组, CFT)
            - EST: np.array, 每个子任务的执行开始时间
            - CT: np.array, 每个子任务的完成时间
            - CFT: float, 任务的完成时间（所有终止节点的最大CT）
    
    Note:
        - EST和CT数组中，-1.0表示未计算（已完成节点使用current_time）
        - 已完成的节点（status==3）直接使用current_time作为CT
    """
    n = dag.num_subtasks
    EST = np.full(n, -1.0, dtype=np.float32)
    CT = np.full(n, -1.0, dtype=np.float32)
    
    # 步骤1：获取拓扑排序（确保按依赖关系顺序计算）
    topo_order = get_topological_order(dag.adj)
    
    # 步骤2：构建前驱节点映射（加速查找）
    predecessors = {}
    for i in range(n):
        preds = np.where(dag.adj[:, i] == 1)[0]
        predecessors[i] = list(preds)
    
    # 步骤3：按拓扑顺序计算每个节点的EST和CT
    for node_id in topo_order:
        loc = task_locations[node_id]
        status = int(dag.status[node_id])
        
        # 特殊情况：节点已完成
        # 已完成节点的CT使用当前时间（实际完成时间）
        if status == 3:
            EST[node_id] = current_time
            CT[node_id] = current_time
            continue
        
        # ============================================================
        # 步骤3.1：获取处理器FAT（Earliest Available Time）
        # ============================================================
        # 注意：这里只读取FAT值，不修改，因为这是预测计算
        # 已经在执行的节点：处理器已被占用，用当前时间作为最早启动时间，避免重复叠加等待
        if status == 2:
            processor_fat = current_time
        else:
            processor_fat = 0.0
            if loc == 'Local':
                # 本地执行：使用车辆本地处理器的FAT
                processor_fat = vehicle.fat_processor
            elif isinstance(loc, tuple) and loc[0] == 'RSU':
                # RSU执行：使用RSU中最小FAT的处理器（负载均衡后的选择）
                rsu_id = loc[1]
                if 0 <= rsu_id < len(rsus):
                    rsu = rsus[rsu_id]
                    processor_fat = rsu.get_min_processor_fat()
            elif isinstance(loc, int):
                # 其他车辆执行：使用目标车辆处理器的FAT
                target_veh = _get_vehicle_by_id(vehicles_list, loc)
                if target_veh is not None:
                    processor_fat = target_veh.fat_processor
        
        # ============================================================
        # 步骤3.2：计算上传完成时间（如果任务在远程执行）
        # ============================================================
        # 公式：上传完成时间 = max(上传信道FAT, current_time) + 上传数据量 / 传输速率
        # 注意：上传数据量是子任务的输入数据（total_data[node_id]），不是依赖数据
        upload_completion_time = current_time if status == 2 else 0.0
        if loc != 'Local':
            # 上传数据量：使用剩余输入数据，确保动态剩余时间正确
            upload_data = float(dag.rem_data[node_id])
            
            # 根据目标类型选择传输速率（V2I或V2V）
            if isinstance(loc, tuple) and loc[0] == 'RSU':
                # V2I上传
                rsu_id = loc[1]
                if 0 <= rsu_id < len(rsus):
                    rsu_pos = rsus[rsu_id].position
                    upload_rate = channel_model.compute_one_rate(
                        vehicle, rsu_pos, 'V2I', current_time,
                        v2i_user_count=v2i_user_count
                    )
                else:
                    upload_rate = 1e6  # 默认速率（错误情况）
            elif isinstance(loc, int):
                # V2V上传
                target_veh = _get_vehicle_by_id(vehicles_list, loc)
                if target_veh is not None:
                    upload_rate = channel_model.compute_one_rate(vehicle, target_veh.pos, 'V2V', current_time)
                else:
                    upload_rate = 1e6  # 默认速率（错误情况）
            else:
                upload_rate = 1e6  # 默认速率（错误情况）
            
            # 计算上传时间
            upload_rate = max(upload_rate, 1e-6)  # 避免除零
            upload_time = upload_data / upload_rate
            # 上传完成时间考虑上传信道的FAT（串行传输）
            # 正在执行的任务已经占用上行，不再叠加额外FAT
            base_start = current_time if status == 2 else max(vehicle.fat_uplink, current_time)
            upload_completion_time = base_start + upload_time
        else:
            # 本地执行，无需上传
            upload_completion_time = current_time
        
        # ============================================================
        # 步骤3.3：计算前驱节点的最大完成时间+传输时间
        # ============================================================
        # 考虑所有前驱节点的完成时间，加上数据传输时间
        # 如果前驱和当前节点在同一位置执行，传输时间为0
        max_pred_completion = 0.0
        for pred_id in predecessors.get(node_id, []):
            # 获取前驱节点的完成时间
            pred_status = int(dag.status[pred_id])
            if pred_status == 3:
                # 前驱已完成：使用当前时间
                pred_ct = current_time
            else:
                # 前驱未完成：使用已计算的CT
                # 由于按拓扑顺序计算，前驱节点应该已经计算过
                if CT[pred_id] >= 0:
                    pred_ct = CT[pred_id]
                else:
                    # 理论上不应该发生，使用当前时间作为估计
                    pred_ct = current_time
            
            # 计算依赖数据传输时间
            transfer_data = dag.data_matrix[pred_id, node_id]
            transfer_time = 0.0
            
            if transfer_data > 1e-9:  # 有数据传输需求
                pred_loc = task_locations[pred_id]
                curr_loc = loc
                
                # 判断是否在同一位置执行
                same_location = False
                if isinstance(pred_loc, tuple) and isinstance(curr_loc, tuple):
                    same_location = (pred_loc == curr_loc)  # 同一RSU
                elif pred_loc == 'Local' and curr_loc == 'Local':
                    same_location = True  # 都在本地
                elif isinstance(pred_loc, int) and isinstance(curr_loc, int):
                    same_location = (pred_loc == curr_loc)  # 同一车辆
                
                if not same_location:
                    # 需要传输，使用当前发射功率估计速率
                    
                    # 确定接收方位置
                    if curr_loc == 'Local':
                        rx_pos = vehicle.pos
                        rx_vehicle = vehicle
                    elif isinstance(curr_loc, tuple) and curr_loc[0] == 'RSU':
                        rsu_id = curr_loc[1]
                        if 0 <= rsu_id < len(rsus):
                            rx_pos = rsus[rsu_id].position
                            rx_vehicle = None
                        else:
                            rx_pos = Cfg.RSU_POS
                            rx_vehicle = None
                    elif isinstance(curr_loc, int):
                        rx_vehicle = _get_vehicle_by_id(vehicles_list, curr_loc)
                        rx_pos = rx_vehicle.pos if rx_vehicle is not None else vehicle.pos
                    else:
                        rx_pos = vehicle.pos
                        rx_vehicle = vehicle
                    
                    # 确定发送方位置和链路类型
                    if pred_loc == 'Local':
                        # 前驱在本地，当前车辆发送
                        tx_vehicle = vehicle
                        if isinstance(curr_loc, tuple) and curr_loc[0] == 'RSU':
                            link_type = 'V2I'
                        else:
                            link_type = 'V2V'
                    elif isinstance(pred_loc, tuple) and pred_loc[0] == 'RSU':
                        # 前驱在RSU，简化处理：假设当前车辆从RSU获取数据
                        # 实际应该是V2I链路
                        tx_vehicle = vehicle
                        link_type = 'V2I'
                    elif isinstance(pred_loc, int):
                        # 前驱在其他车辆
                        tx_vehicle = _get_vehicle_by_id(vehicles_list, pred_loc)
                        if tx_vehicle is None:
                            tx_vehicle = vehicle
                        if isinstance(curr_loc, tuple) and curr_loc[0] == 'RSU':
                            link_type = 'V2I'
                        else:
                            link_type = 'V2V'
                    else:
                        tx_vehicle = vehicle
                        link_type = 'V2V'
                    
                    if link_type == 'V2I':
                        transfer_rate = channel_model.compute_one_rate(
                            tx_vehicle, rx_pos, link_type, current_time,
                            v2i_user_count=v2i_user_count
                        )
                    else:
                        transfer_rate = channel_model.compute_one_rate(tx_vehicle, rx_pos, link_type, current_time)
                    
                    transfer_rate = max(transfer_rate, 1e-6)  # 避免除零
                    transfer_time = transfer_data / transfer_rate
            
            # 前驱完成+传输时间
            completion_with_transfer = pred_ct + transfer_time
            max_pred_completion = max(max_pred_completion, completion_with_transfer)
        
        # ============================================================
        # 步骤3.4：计算EST（Earliest Start Time）
        # ============================================================
        # EST = max(处理器FAT, 上传完成时间, 前驱完成+传输时间的最大值)
        EST[node_id] = max(processor_fat, upload_completion_time, max_pred_completion)
        
        # ============================================================
        # 步骤3.5：计算执行时间
        # ============================================================
        # 根据执行位置获取CPU频率
        if loc == 'Local':
            cpu_freq = vehicle.cpu_freq
        elif isinstance(loc, tuple) and loc[0] == 'RSU':
            rsu_id = loc[1]
            if 0 <= rsu_id < len(rsus):
                cpu_freq = rsus[rsu_id].cpu_freq
            else:
                cpu_freq = Cfg.F_RSU  # 默认RSU频率
        elif isinstance(loc, int):
            target_veh = _get_vehicle_by_id(vehicles_list, loc)
            if target_veh is not None:
                cpu_freq = target_veh.cpu_freq
            else:
                cpu_freq = vehicle.cpu_freq  # 错误情况，使用车辆频率
        else:
            cpu_freq = vehicle.cpu_freq  # 默认

        # 执行时间 = 剩余计算量 / CPU频率
        execution_time = float(dag.rem_comp[node_id]) / max(cpu_freq, 1e-6)
        
        # ============================================================
        # 步骤3.6：计算CT（Completion Time）
        # ============================================================
        CT[node_id] = EST[node_id] + execution_time
        
        # 注意：这里不更新实际的FAT，因为这是预测计算
        # 实际的FAT更新在任务完成时进行（在环境层的step()函数中）
    
    # ============================================================
    # 步骤4：计算CFT（任务完成时间）
    # ============================================================
    # CFT = 所有终止节点的最大CT
    # 终止节点：出度为0的节点（没有后继节点）
    termination_nodes = [i for i in range(n) if dag.out_degree[i] == 0]
    if termination_nodes:
        valid_cts = [CT[tn] for tn in termination_nodes if CT[tn] >= 0]
        if valid_cts:
            CFT = max(valid_cts)
        else:
            # 所有终止节点都未计算（异常情况），使用当前时间
            CFT = current_time
    else:
        # 没有终止节点（异常情况），使用所有节点的最大CT
        valid_cts = [ct for ct in CT if ct >= 0]
        if valid_cts:
            CFT = max(valid_cts)
        else:
            CFT = current_time
    
    return EST, CT, CFT


def calculate_t_local(vehicle, dag, channel_model):
    """
    计算完全本地执行时间（t_local）
    
    核心逻辑：
    - 假设所有子任务都在本地车辆上执行
    - 考虑DAG依赖关系
    - 单处理器串行执行（任务必须等待前一个任务完成）
    - 依赖数据传输使用本地传输速率估计
    
    用途：
    - 用于全局奖励计算：R_global = λ₁ * tanh(t_local / t - 1) + ...
    - 衡量卸载策略相比完全本地执行的性能提升
    
    Args:
        vehicle: 车辆对象（执行任务的车辆）
        dag: DAGTask对象
        channel_model: 信道模型对象（用于计算本地传输时间）
    
    Returns:
        float: 完全本地执行时间（t_local），即所有子任务在本地串行执行的总时间
    
    Note:
        - 这是虚拟计算，不修改车辆状态
        - 本地传输速率使用当前发射功率估计
    """
    n = dag.num_subtasks
    local_fat = 0.0  # 本地处理器FAT（初始为0）
    CT_local = np.zeros(n, dtype=np.float32)
    
    # 步骤1：获取拓扑排序
    topo_order = get_topological_order(dag.adj)
    
    # 步骤2：构建前驱节点映射
    predecessors = {}
    for i in range(n):
        preds = np.where(dag.adj[:, i] == 1)[0]
        predecessors[i] = list(preds)
    
    # 步骤3：按拓扑顺序计算每个节点的本地执行完成时间
    for node_id in topo_order:
        # 计算EST（考虑前驱完成时间和传输时间）
        max_pred_ct = 0.0
        for pred_id in predecessors.get(node_id, []):
            pred_ct = CT_local[pred_id]
            
            # 依赖数据传输时间（本地传输，但仍有开销）
            transfer_data = dag.data_matrix[pred_id, node_id]
            if transfer_data > 1e-9:
                # 本地传输使用V2V模型（车辆内部传输）
                transfer_rate = channel_model.compute_one_rate(vehicle, vehicle.pos, 'V2V', 0.0)
                
                transfer_rate = max(transfer_rate, 1e-6)  # 避免除零
                transfer_time = transfer_data / transfer_rate
            else:
                transfer_time = 0.0
            
            max_pred_ct = max(max_pred_ct, pred_ct + transfer_time)
        
        # EST = max(本地FAT, 前驱完成+传输时间)
        EST_local = max(local_fat, max_pred_ct)
        
        # 执行时间 = 计算量 / 本地CPU频率
        execution_time = dag.total_comp[node_id] / max(vehicle.cpu_freq, 1e-6)
        
        # CT = EST + 执行时间
        CT_local[node_id] = EST_local + execution_time
        
        # 更新本地FAT（单处理器串行：任务必须等待前一个任务完成）
        local_fat = CT_local[node_id]
    
    # 返回终止节点的CT（完全本地执行时间）
    termination_nodes = [i for i in range(n) if dag.out_degree[i] == 0]
    if termination_nodes:
        return max(CT_local[tn] for tn in termination_nodes)
    else:
        # 没有终止节点（异常情况），返回最大CT
        return max(CT_local) if len(CT_local) > 0 else 0.0
