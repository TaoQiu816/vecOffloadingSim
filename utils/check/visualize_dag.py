# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# from utils.dag_generator import DAGGenerator
#
#
# def visualize_dag(num_nodes=6):
#     # 1. 初始化生成器
#     # 设定平均计算量 200 Mcycles, 平均数据传输 20 MB
#     params = {
#         'fat': 0.6,
#         'density': 0.5,
#         'regular': 0.5,
#         'ccr': 1.0,
#         'avg_comp': 200 * 1e6,  # 200 Mcycles
#         'avg_data': 20 * 1e6  # 20 MB
#     }
#     generator = DAGGenerator(**params)
#
#     # 2. 生成数据
#     adj, profiles, data_mat = generator.generate(num_nodes)
#
#     # 3. 构建 NetworkX 图对象
#     G = nx.DiGraph()
#
#     # --- 添加节点 ---
#     entry_nodes = []
#     node_labels = {}
#
#     for i in range(num_nodes):
#         comp_val = profiles[i]['comp']
#         input_data = profiles[i]['input_data']
#
#         # 单位转换：除以 10^6 转换为 M (Mega)
#         comp_str = f"{comp_val / 1e6:.1f}M"
#
#         G.add_node(i, comp=comp_val, input_data=input_data)
#
#         # 标签显示：ID + 计算量
#         label = f"Task {i}\nComp: {comp_str}"
#
#         # 如果是入口任务，额外显示输入数据量
#         if input_data > 0:
#             entry_nodes.append(i)
#             label += f"\nIn: {input_data:.1f}MB"
#
#         node_labels[i] = label
#
#     # --- 添加边 ---
#     edge_labels = {}
#     rows, cols = np.where(adj == 1)
#     for u, v in zip(rows, cols):
#         w = data_mat[u, v]
#         # 单位转换：除以 10^6 转换为 MB
#         w_str = f"{w / 1e6:.1f} MB"
#
#         G.add_edge(u, v, weight=w)
#         edge_labels[(u, v)] = w_str
#
#     # 4. 开始绘图
#     plt.figure(figsize=(12, 8))
#
#     # 布局算法：使用 shell_layout 或 circular_layout 这种分层布局比较适合 DAG
#     # 或者用 spring_layout 模拟力导向
#     try:
#         # 尝试使用 graphviz 布局（如果有安装的话效果最好），没有则回退
#         pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
#     except:
#         # 使用 spring 布局，k值越大节点越分散
#         pos = nx.spring_layout(G, seed=42, k=1.5)
#
#     # 绘制普通节点 (蓝色)
#     non_entry_nodes = [n for n in G.nodes() if n not in entry_nodes]
#     nx.draw_networkx_nodes(G, pos, nodelist=non_entry_nodes,
#                            node_color='lightblue', node_size=2000, edgecolors='black')
#
#     # 绘制入口节点 (绿色，突出显示)
#     nx.draw_networkx_nodes(G, pos, nodelist=entry_nodes,
#                            node_color='lightgreen', node_size=2000, edgecolors='black')
#
#     # 绘制边
#     nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
#
#     # 绘制节点标签
#     nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
#
#     # 绘制边标签 (传输数据量)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='red')
#
#     plt.title(f"DAG Visualization (N={num_nodes})\n"
#               f"Unit: Comp in Mcycles (10^6), Data in MB (10^6)", fontsize=14)
#     plt.axis('off')
#
#     # 显示图形
#     plt.tight_layout()
#     print("生成可视化图表...")
#     plt.show()
#
#
# if __name__ == '__main__':
#     visualize_dag(num_nodes=6)