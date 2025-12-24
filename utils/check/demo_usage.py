# from utils.dag_generator import DAGGenerator
# import numpy as np
#
# # 1. 配置你想要的范围
# config = {
#     # 计算量：500M ~ 1000M 周期
#     'comp_range': (500 * 1e6, 1000 * 1e6),
#
#     # 任务间传输：10MB ~ 20MB
#     'trans_range': (10 * 1e6, 20 * 1e6),
#
#     # 入口任务的原始数据：50MB ~ 100MB
#     'input_range': (50 * 1e6, 100 * 1e6),
#
#     # 拓扑形状参数
#     'fat': 0.6,
#     'density': 0.4
# }
#
# # 2. 初始化
# generator = DAGGenerator(**config)
#
# # 3. 生成
# num_nodes = 5
# adj, profiles, data_mat = generator.generate(num_nodes)
#
# # 4. 打印验证
# print(f"=== 生成 {num_nodes} 节点 DAG ===")
# print(f"设定计算范围: {config['comp_range']}")
# print(f"设定传输范围: {config['trans_range']}")
# print("-" * 30)
#
# for i, p in enumerate(profiles):
#     print(f"Task {i}:")
#     print(f"  - 计算量: {p['comp'] / 1e6:.2f} Mcycles "
#           f"(符合范围? {config['comp_range'][0] <= p['comp'] <= config['comp_range'][1]})")
#
#     if p['input_data'] > 0:
#         print(f"  - [入口] 输入数据: {p['input_data'] / 1e6:.2f} MB "
#               f"(符合范围? {config['input_range'][0] <= p['input_data'] <= config['input_range'][1]})")
#
# print("-" * 30)
# edges = np.argwhere(adj > 0)
# for u, v in edges:
#     w = data_mat[u, v]
#     print(f"Edge {u}->{v}: 传输 {w / 1e6:.2f} MB "
#           f"(符合范围? {config['trans_range'][0] <= w <= config['trans_range'][1]})")