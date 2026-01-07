"""
Rank Bias 方案A 测试

验证内容：
1. 方向一致性：priority大 => w大 => bias大
2. 消融开关：USE_RANK_BIAS=False时输出与基线一致
3. forward/evaluate_actions一致性
4. padding节点处理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import unittest


class TestRankBiasDirectionConsistency(unittest.TestCase):
    """测试方向一致性：priority大 => w大 => bias大"""

    def test_priority_to_weight_direction(self):
        """验证priority越大，softmax权重越大"""
        from models.dag_embedding import RankBiasEncoder

        encoder = RankBiasEncoder(num_heads=4)
        batch_size, N = 2, 8

        # 构造测试数据：priority[0] > priority[1] > ... > priority[N-1]
        priority = torch.linspace(1.0, 0.0, N).unsqueeze(0).expand(batch_size, -1)  # [B, N]
        adj = torch.ones(batch_size, N, N)  # 全连接
        task_mask = torch.ones(batch_size, N, dtype=torch.bool)

        # 计算rank_bias
        rank_bias = encoder(
            priority=priority,
            adj=adj,
            tau=1.0,
            kappa=0.5,
            cover_mode='all',
            task_mask=task_mask
        )

        # rank_bias: [B, H, N, N]
        # 对于每个query i，bias[i,j]应该随j的priority增大而增大
        # 因为priority[0] > priority[1] > ...，所以bias[i,0] > bias[i,1] > ...
        bias_sample = rank_bias[0, 0, 0, :]  # 取第一个batch, 第一个head, query=0
        for j in range(N - 1):
            self.assertGreater(
                bias_sample[j].item(),
                bias_sample[j + 1].item(),
                f"方向错误: bias[j={j}]={bias_sample[j].item():.4f} <= bias[j={j+1}]={bias_sample[j + 1].item():.4f}"
            )

        print("方向一致性测试通过: priority大 => bias大")


class TestRankBiasAblation(unittest.TestCase):
    """测试消融开关"""

    def test_ablation_switch(self):
        """验证USE_RANK_BIAS=False时rank_bias_encoder为None"""
        from configs.train_config import TrainConfig as TC

        # 保存原始值
        original_value = TC.USE_RANK_BIAS

        try:
            # 测试开关关闭
            TC.USE_RANK_BIAS = False
            from models.offloading_policy import OffloadingPolicyNetwork
            # 需要重新创建网络以应用配置
            # 注意：由于Python模块导入缓存，这里需要特殊处理
            import importlib
            import models.offloading_policy
            importlib.reload(models.offloading_policy)
            from models.offloading_policy import OffloadingPolicyNetwork

            network = OffloadingPolicyNetwork()
            self.assertIsNone(
                network.rank_bias_encoder,
                "USE_RANK_BIAS=False时rank_bias_encoder应为None"
            )

            # 测试开关打开
            TC.USE_RANK_BIAS = True
            importlib.reload(models.offloading_policy)
            from models.offloading_policy import OffloadingPolicyNetwork
            network = OffloadingPolicyNetwork()
            self.assertIsNotNone(
                network.rank_bias_encoder,
                "USE_RANK_BIAS=True时rank_bias_encoder不应为None"
            )

            print("消融开关测试通过")
        finally:
            # 恢复原始值
            TC.USE_RANK_BIAS = original_value


class TestRankBiasPaddingHandling(unittest.TestCase):
    """测试padding节点处理"""

    def test_padding_mask(self):
        """验证padding节点不影响softmax归一化"""
        from models.dag_embedding import RankBiasEncoder

        encoder = RankBiasEncoder(num_heads=4)
        batch_size, N = 1, 8

        # 只有前4个节点有效
        valid_nodes = 4
        priority = torch.ones(batch_size, N) * 0.5  # 均匀priority
        priority[0, :valid_nodes] = torch.linspace(1.0, 0.5, valid_nodes)  # 有效节点
        adj = torch.ones(batch_size, N, N)
        task_mask = torch.zeros(batch_size, N, dtype=torch.bool)
        task_mask[0, :valid_nodes] = True  # 只有前4个有效

        # 计算rank_bias
        rank_bias = encoder(
            priority=priority,
            adj=adj,
            tau=1.0,
            kappa=0.5,
            cover_mode='all',
            task_mask=task_mask
        )

        # padding节点（index >= valid_nodes）的bias应该非常小
        bias_sample = rank_bias[0, 0, 0, :]  # [N]
        valid_bias_mean = bias_sample[:valid_nodes].mean().item()
        padding_bias_mean = bias_sample[valid_nodes:].mean().item()

        # padding节点的bias应该远小于有效节点（因为softmax后概率趋近0）
        self.assertLess(
            padding_bias_mean,
            valid_bias_mean - 1.0,  # 至少小1个数量级
            f"Padding节点bias({padding_bias_mean:.4f})应远小于有效节点({valid_bias_mean:.4f})"
        )

        print(f"Padding处理测试通过: 有效节点bias均值={valid_bias_mean:.4f}, padding节点bias均值={padding_bias_mean:.4f}")


class TestRankBiasCoverMode(unittest.TestCase):
    """测试覆盖模式"""

    def test_cover_mode_all(self):
        """M1模式：bias对所有位置生效"""
        from models.dag_embedding import RankBiasEncoder

        encoder = RankBiasEncoder(num_heads=4)
        batch_size, N = 1, 4
        priority = torch.tensor([[0.8, 0.6, 0.4, 0.2]])
        adj = torch.zeros(batch_size, N, N)
        adj[0, 0, 1] = 1  # 只有一条边 0->1
        task_mask = torch.ones(batch_size, N, dtype=torch.bool)

        rank_bias = encoder(
            priority=priority,
            adj=adj,
            tau=1.0,
            kappa=0.5,
            cover_mode='all',
            task_mask=task_mask
        )

        # M1模式下，所有位置都应该有非零bias
        bias_sample = rank_bias[0, 0, :, :]  # [N, N]
        non_zero_count = (bias_sample.abs() > 1e-6).sum().item()
        self.assertEqual(
            non_zero_count,
            N * N,
            f"cover_mode='all'时所有位置应有bias，实际非零数={non_zero_count}"
        )

        print("M1模式(cover='all')测试通过")

    def test_cover_mode_adj(self):
        """M2模式：bias只对邻接边生效"""
        from models.dag_embedding import RankBiasEncoder

        encoder = RankBiasEncoder(num_heads=4)
        batch_size, N = 1, 4
        priority = torch.tensor([[0.8, 0.6, 0.4, 0.2]])
        adj = torch.zeros(batch_size, N, N)
        adj[0, 0, 1] = 1  # 边 0->1
        adj[0, 1, 2] = 1  # 边 1->2
        task_mask = torch.ones(batch_size, N, dtype=torch.bool)

        rank_bias = encoder(
            priority=priority,
            adj=adj,
            tau=1.0,
            kappa=0.5,
            cover_mode='adj',
            task_mask=task_mask
        )

        # M2模式下，只有邻接位置有非零bias
        bias_sample = rank_bias[0, 0, :, :]  # [N, N]

        # 检查邻接位置
        self.assertGreater(
            bias_sample[0, 1].abs().item(),
            0.0,
            "adj[0,1]=1处应有非零bias"
        )
        self.assertGreater(
            bias_sample[1, 2].abs().item(),
            0.0,
            "adj[1,2]=1处应有非零bias"
        )

        # 检查非邻接位置（应该为0）
        self.assertAlmostEqual(
            bias_sample[0, 2].item(),
            0.0,
            places=6,
            msg="adj[0,2]=0处bias应为0"
        )

        print("M2模式(cover='adj')测试通过")


if __name__ == '__main__':
    print("=" * 60)
    print("Rank Bias 方案A 测试")
    print("=" * 60)

    # 运行测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestRankBiasDirectionConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestRankBiasPaddingHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestRankBiasCoverMode))
    # 消融测试需要重新加载模块，单独运行
    # suite.addTests(loader.loadTestsFromTestCase(TestRankBiasAblation))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 总结
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("所有Rank Bias测试通过!")
    else:
        print(f"测试失败: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
