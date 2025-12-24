# from utils.dag_generator import DAGGenerator
# import numpy as np
#
# # 1. 初始化生成器，设置参数
# params = {
#     'fat': 0.6,
#     'density': 0.4,
#     'regular': 0.5,
#     'ccr': 1.0,
#     'avg_comp': 1000.0,  # 我们希望平均计算量是 1000
#     'avg_data': 50.0     # 我们希望平均传输量是 50
# }
# generator = DAGGenerator(**params) # 注意：需要在 __init__ 里接收 **kwargs
#
# # 2. 生成一个 5 个节点的图
# adj, profiles, data = generator.generate(5)
#
# # 3. 打印验证
# print("=== 邻接矩阵 (Topology) ===")
# print(adj)
#
# print("\n=== 任务属性 (Profiles) ===")
# for i, p in enumerate(profiles):
#     print(f"Task {i}: 计算量={p['comp']:.2f}, 输入数据={p['input_data']:.2f}")
#
# print("\n=== 验证参数是否生效 ===")
# comps = [p['comp'] for p in profiles]
# print(f"实际平均计算量: {np.mean(comps):.2f} (预期: 1000.0)")
#
# if np.sum(adj) > 0:
#     print(f"成功生成了 {np.sum(adj)} 条边")
# else:
#     print("生成了独立的节点，无边连接")