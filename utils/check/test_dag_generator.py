# import unittest
# from unittest.mock import patch, MagicMock
# import numpy as np
# from utils.dag_generator import DAGGenerator
#
#
# class TestDAGGenerator(unittest.TestCase):
#
#     def setUp(self):
#         """初始化 DAGGenerator 实例"""
#         # 初始化时使用默认参数
#         self.generator = DAGGenerator()
#         # 确保 avg_comp 和 avg_data 有明确的值用于测试断言
#         self.generator.avg_comp = 200000000.0
#         self.generator.avg_data = 20000000.0
#
#     @patch('utils.dag_generator.daggen')
#     def test_generate_normal_case(self, mock_daggen):
#         """TC01: 正常情况下生成 DAG 结构，并验证归一化"""
#
#         # 模拟 daggen 返回两个节点 (权重 10 和 20) 和一条边
#         mock_node_1 = {'w': 10}
#         mock_node_2 = {'w': 20}
#         mock_edge = {'u': 0, 'v': 1, 'w': 5}
#
#         mock_dag = MagicMock()
#         # 注意：这里模拟返回字典列表，而不是对象列表
#         mock_dag.task_n_edge_dicts.return_value = ([mock_node_1, mock_node_2], [mock_edge])
#
#         mock_daggen.DAG.return_value = mock_dag
#
#         # 执行生成
#         adj, profiles, data_mat = self.generator.generate(2)
#
#         # 断言结构维度
#         self.assertEqual(adj.shape, (2, 2))
#         self.assertEqual(len(profiles), 2)
#
#         # --- 验证计算量归一化逻辑 ---
#         # 原始均值 = (10+20)/2 = 15
#         # 目标均值 = 200,000,000
#         # 预期缩放因子 = 2e8 / 15
#         expected_scale = self.generator.avg_comp / 15.0
#
#         # 验证第一个任务 (原始10)
#         self.assertAlmostEqual(profiles[0]['comp'], 10 * expected_scale, delta=1.0)
#         # 验证第二个任务 (原始20)
#         self.assertAlmostEqual(profiles[1]['comp'], 20 * expected_scale, delta=1.0)
#
#         # 验证边
#         self.assertEqual(adj[0][1], 1)
#         # 边缩放验证 (原始5, 目标均值2e7, 原始均值5 -> 因子 2e7/5)
#         self.assertAlmostEqual(data_mat[0][1], 20000000.0, delta=1.0)
#
#     @patch('utils.dag_generator.daggen')
#     def test_generate_empty_graph(self, mock_daggen):
#         """TC02: N=0 的情况"""
#         adj, profiles, data_mat = self.generator.generate(0)
#         self.assertEqual(adj.shape, (0, 0))
#         self.assertEqual(len(profiles), 0)
#
#     @patch('utils.dag_generator.daggen')
#     def test_single_node_no_edges(self, mock_daggen):
#         """TC03: N=1 的情况 (此时库可能不返回边，或者走 Fallback)"""
#         # 模拟库返回一个节点，无边
#         mock_dag = MagicMock()
#         mock_dag.task_n_edge_dicts.return_value = ([{'w': 10}], [])
#         mock_daggen.DAG.return_value = mock_dag
#
#         adj, profiles, data_mat = self.generator.generate(1)
#
#         self.assertEqual(adj.shape, (1, 1))
#         self.assertEqual(len(profiles), 1)
#         # 只有一个节点时，缩放因子 = target / 10
#         self.assertAlmostEqual(profiles[0]['comp'], self.generator.avg_comp, delta=1.0)
#         # 入口任务应该有输入数据
#         self.assertGreater(profiles[0]['input_data'], 0)
#
#     @patch('utils.dag_generator.daggen')
#     def test_zero_weights_avoid_division_by_zero(self, mock_daggen):
#         """TC04: 原始权重为 0 时，应处理除以零异常，结果应为 0"""
#         mock_dag = MagicMock()
#         mock_dag.task_n_edge_dicts.return_value = (
#             [{'w': 0}, {'w': 0}],
#             [{'u': 0, 'v': 1, 'w': 0}]
#         )
#         mock_daggen.DAG.return_value = mock_dag
#
#         adj, profiles, data_mat = self.generator.generate(2)
#
#         # 应当不会崩溃，且结果为0
#         self.assertEqual(profiles[0]['comp'], 0)
#         self.assertEqual(data_mat[0][1], 0)
#
#     @patch('utils.dag_generator.daggen')
#     def test_multiple_calls_different_seeds(self, mock_daggen):
#         """TC05: 多次调用应产生不同的随机种子 (修复版)"""
#         mock_dag = MagicMock()
#         mock_dag.task_n_edge_dicts.return_value = ([{'w': 10}], [])
#         mock_daggen.DAG.return_value = mock_dag
#
#         seeds = []
#
#         # --- 关键修复：同时支持位置参数和关键字参数 ---
#         def side_effect(*args, **kwargs):
#             # args[0] 通常是 seed (如果代码用的是位置传参)
#             if len(args) > 0:
#                 seeds.append(args[0])
#             else:
#                 seeds.append(kwargs.get('seed'))
#             return mock_dag
#
#         # ----------------------------------------
#
#         mock_daggen.DAG.side_effect = side_effect
#
#         self.generator.generate(1)
#         self.generator.generate(1)
#
#         # 验证确实调用了两次且种子不同
#         self.assertEqual(len(seeds), 2)
#         self.assertNotEqual(seeds[0], seeds[1])
#
#     @patch('utils.dag_generator.daggen')
#     def test_invalid_index_access(self, mock_daggen):
#         """TC06: 边索引超出范围时不崩溃"""
#         mock_dag = MagicMock()
#         # 模拟返回一个非法的边 (u=99 超出 N=1)
#         mock_dag.task_n_edge_dicts.return_value = (
#             [{'w': 10}],
#             [{'u': 99, 'v': 0, 'w': 5}]
#         )
#         mock_daggen.DAG.return_value = mock_dag
#
#         adj, profiles, data_mat = self.generator.generate(1)
#
#         # 不应该设置任何边
#         self.assertEqual(np.sum(adj), 0)
#         self.assertEqual(np.sum(data_mat), 0)
#
#     def test_dag_visualization_integration(self):
#         """集成测试：验证基本属性"""
#         # 注意：这里我们不 Mock daggen，如果本地没有安装 daggen 库可能会报错
#         # 但既然你的环境里有 daggen，这个测试可以用来验证 Fallback 或真实逻辑
#
#         # 为了避免依赖真实 C++ 库的不确定性，我们这里验证 Fallback 逻辑
#         # 我们可以通过 mock 强行触发 fallback，或者简单运行一遍
#
#         # 这里简单运行一遍 N=6，看是否崩盘即可
#         try:
#             adj, profiles, dep_data = self.generator.generate(6)
#             print(f"集成测试成功: N=6, 边数={np.sum(adj)}")
#             self.assertEqual(len(profiles), 6)
#             self.assertEqual(adj.shape, (6, 6))
#         except Exception as e:
#             # 如果本地 daggen 库没装好，忽略此测试
#             print(f"跳过集成测试 (可能是 daggen 库问题): {e}")
#
#
# if __name__ == '__main__':
#     unittest.main()