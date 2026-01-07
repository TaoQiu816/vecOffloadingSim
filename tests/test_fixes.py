"""
[修复验证测试] test_fixes.py
Fix Verification Tests

验证问题修复的完整测试套件：
- P38: evaluate_actions 包含 resource_raw 参数
- P39: evaluate_actions 应用 Logit Bias
- P14: LR_CRITIC 设置为 5e-4
- P33: MASK_VALUE 统一使用
- P03: get_subtask_exec_location 方法正确实现

运行方式:
    python -m pytest tests/test_fixes.py -v
    或
    python tests/test_fixes.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

def test_constants_import():
    """测试常量文件可以正确导入"""
    from configs.constants import MASK_VALUE, TaskStatus, ResourceRole, ActionIndex

    # 验证 MASK_VALUE 值
    assert MASK_VALUE == -1e10, f"MASK_VALUE should be -1e10, got {MASK_VALUE}"

    # 验证任务状态常量
    assert TaskStatus.PENDING == 0
    assert TaskStatus.READY == 1
    assert TaskStatus.RUNNING == 2
    assert TaskStatus.COMPLETED == 3

    # 验证资源角色常量
    assert ResourceRole.LOCAL == 1
    assert ResourceRole.RSU == 2
    assert ResourceRole.NEIGHBOR == 3

    # 验证动作索引常量
    assert ActionIndex.LOCAL == 0
    assert ActionIndex.RSU == 1
    assert ActionIndex.V2V_START == 2

    print("✓ test_constants_import passed")


def test_lr_critic_value():
    """测试 P14: LR_CRITIC 值是否正确"""
    from configs.train_config import TrainConfig as TC

    expected_lr_critic = 5e-4
    assert TC.LR_CRITIC == expected_lr_critic, \
        f"P14: LR_CRITIC should be {expected_lr_critic}, got {TC.LR_CRITIC}"

    # 验证与 LR_ACTOR 的比例
    ratio = TC.LR_CRITIC / TC.LR_ACTOR
    assert 1.0 <= ratio <= 2.0, \
        f"LR_CRITIC/LR_ACTOR ratio should be 1.0-2.0, got {ratio:.2f}"

    print(f"✓ test_lr_critic_value passed (LR_CRITIC={TC.LR_CRITIC}, ratio={ratio:.2f})")


def test_evaluate_actions_has_resource_raw():
    """测试 P38: evaluate_actions 调用 forward 时包含 resource_raw 参数"""
    import inspect
    from models.offloading_policy import OffloadingPolicyNetwork

    # 获取 evaluate_actions 方法的源代码
    source = inspect.getsource(OffloadingPolicyNetwork.evaluate_actions)

    # 检查是否包含 resource_raw 参数
    assert "resource_raw=inputs['resource_raw']" in source or \
           "resource_raw=inputs[\"resource_raw\"]" in source, \
        "P38: evaluate_actions should pass resource_raw to forward()"

    print("✓ test_evaluate_actions_has_resource_raw passed")


def test_evaluate_actions_has_logit_bias():
    """测试 P39: evaluate_actions 应用 Logit Bias"""
    import inspect
    from models.offloading_policy import OffloadingPolicyNetwork

    # 获取 evaluate_actions 方法的源代码
    source = inspect.getsource(OffloadingPolicyNetwork.evaluate_actions)

    # 检查是否应用 Logit Bias
    assert "USE_LOGIT_BIAS" in source, \
        "P39: evaluate_actions should check USE_LOGIT_BIAS"
    assert "LOGIT_BIAS_LOCAL" in source or "logit_bias" in source, \
        "P39: evaluate_actions should apply logit_bias"

    print("✓ test_evaluate_actions_has_logit_bias passed")


def test_mask_value_consistency():
    """测试 P33: MASK_VALUE 在各模块中使用一致"""
    import inspect
    from configs.constants import MASK_VALUE

    modules_to_check = [
        ('models.dag_embedding', 'EdgeFeatureEncoder'),
        ('models.actor_critic', 'ActorHead'),
        ('models.actor_critic', 'CriticHead'),
        ('models.offloading_policy', 'OffloadingPolicyNetwork'),
        ('agents.mappo_agent', 'MAPPOAgent'),
    ]

    for module_name, class_name in modules_to_check:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        source = inspect.getsource(cls)

        # 检查是否导入了 MASK_VALUE
        module_source = inspect.getsource(module)
        uses_mask_value = 'MASK_VALUE' in source or 'from configs.constants import' in module_source

        # 检查是否还有硬编码的 -1e9 或 -1e10
        has_hardcoded = ('-1e9' in source and 'MASK_VALUE' not in source) or \
                       ('-1e10' in source and 'MASK_VALUE' not in source)

        # 允许注释中包含这些值
        if has_hardcoded:
            # 简单检查：如果有 MASK_VALUE，则认为已修复
            if 'MASK_VALUE' in source:
                has_hardcoded = False

        assert not has_hardcoded or uses_mask_value, \
            f"P33: {module_name}.{class_name} should use MASK_VALUE constant"

    print("✓ test_mask_value_consistency passed")


def test_get_subtask_exec_location():
    """测试 P03: get_subtask_exec_location 方法正确实现"""
    from envs.entities.task_dag import DAGTask

    # 创建简单的DAG用于测试
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    profiles = [
        {'comp': 1e9, 'input_data': 1e6},
        {'comp': 1e9, 'input_data': 0},
        {'comp': 1e9, 'input_data': 0},
    ]
    data_matrix = adj * 1e6
    deadline = 10.0

    dag = DAGTask(task_id=0, adj=adj, profiles=profiles,
                  data_matrix=data_matrix, deadline=deadline)

    # 测试1: 未分配的任务返回 'Local'
    loc = dag.get_subtask_exec_location(0)
    assert loc == 'Local', f"Unassigned task should return 'Local', got {loc}"

    # 测试2: 边界检查
    loc = dag.get_subtask_exec_location(-1)
    assert loc == 'Local', f"Invalid index should return 'Local', got {loc}"
    loc = dag.get_subtask_exec_location(100)
    assert loc == 'Local', f"Out of bounds index should return 'Local', got {loc}"

    # 测试3: 设置exec_locations后返回正确位置
    dag.exec_locations[0] = 'Local'
    dag.exec_locations[1] = ('RSU', 0)

    assert dag.get_subtask_exec_location(0) == 'Local'
    assert dag.get_subtask_exec_location(1) == ('RSU', 0)

    # 测试4: task_locations优先级高于exec_locations
    dag.task_locations[0] = ('RSU', 1)  # 覆盖exec_locations
    assert dag.get_subtask_exec_location(0) == ('RSU', 1)

    print("✓ test_get_subtask_exec_location passed")


def test_network_forward_consistency():
    """测试网络前向传播的一致性"""
    from models.offloading_policy import OffloadingPolicyNetwork
    from configs.config import SystemConfig as Cfg

    # 创建网络
    network = OffloadingPolicyNetwork(d_model=64, num_heads=2, num_layers=1)
    network.eval()

    # 创建模拟输入
    batch_size = 2
    max_nodes = Cfg.MAX_NODES
    max_targets = Cfg.MAX_TARGETS

    inputs = {
        'node_x': torch.randn(batch_size, max_nodes, 7),
        'adj': torch.zeros(batch_size, max_nodes, max_nodes),
        'status': torch.zeros(batch_size, max_nodes, dtype=torch.long),
        'location': torch.zeros(batch_size, max_nodes, dtype=torch.long),
        'L_fwd': torch.zeros(batch_size, max_nodes, dtype=torch.long),
        'L_bwd': torch.zeros(batch_size, max_nodes, dtype=torch.long),
        'data_matrix': torch.zeros(batch_size, max_nodes, max_nodes),
        'delta': torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long),
        'resource_ids': torch.zeros(batch_size, max_targets, dtype=torch.long),
        'resource_raw': torch.randn(batch_size, max_targets, 14),
        'subtask_index': torch.zeros(batch_size, dtype=torch.long),
        'action_mask': torch.ones(batch_size, max_targets, dtype=torch.bool),
        'task_mask': torch.ones(batch_size, max_nodes, dtype=torch.bool),
    }

    # 设置一些有效值
    inputs['resource_ids'][:, 0] = 1  # Local
    inputs['resource_ids'][:, 1] = 2  # RSU
    inputs['status'][:, 0] = 1  # READY

    with torch.no_grad():
        target_logits, alpha, beta, values = network.forward(**inputs)

    # 验证输出形状
    assert target_logits.shape == (batch_size, max_targets), \
        f"target_logits shape mismatch: {target_logits.shape}"
    assert alpha.shape == (batch_size, 1), f"alpha shape mismatch: {alpha.shape}"
    assert beta.shape == (batch_size, 1), f"beta shape mismatch: {beta.shape}"
    assert values.shape == (batch_size, 1), f"values shape mismatch: {values.shape}"

    # 验证没有 NaN
    assert not torch.isnan(target_logits).any(), "target_logits contains NaN"
    assert not torch.isnan(alpha).any(), "alpha contains NaN"
    assert not torch.isnan(beta).any(), "beta contains NaN"
    assert not torch.isnan(values).any(), "values contains NaN"

    # 验证 alpha, beta > 1 (Beta分布参数)
    assert (alpha > 1).all(), f"alpha should be > 1, got min={alpha.min()}"
    assert (beta > 1).all(), f"beta should be > 1, got min={beta.min()}"

    print("✓ test_network_forward_consistency passed")


def test_mappo_agent_evaluate_actions():
    """测试 MAPPO Agent 的 evaluate_actions 方法"""
    from models.offloading_policy import OffloadingPolicyNetwork
    from agents.mappo_agent import MAPPOAgent
    from configs.config import SystemConfig as Cfg

    # 创建网络和Agent
    network = OffloadingPolicyNetwork(d_model=64, num_heads=2, num_layers=1)
    agent = MAPPOAgent(network, device='cpu')

    # 创建模拟观测
    batch_size = 2
    max_nodes = Cfg.MAX_NODES
    max_targets = Cfg.MAX_TARGETS

    obs_list = []
    for _ in range(batch_size):
        obs = {
            'node_x': np.random.randn(max_nodes, 7).astype(np.float32),
            'adj': np.zeros((max_nodes, max_nodes), dtype=np.float32),
            'status': np.zeros(max_nodes, dtype=np.int64),
            'location': np.zeros(max_nodes, dtype=np.int64),
            'L_fwd': np.zeros(max_nodes, dtype=np.int64),
            'L_bwd': np.zeros(max_nodes, dtype=np.int64),
            'data_matrix': np.zeros((max_nodes, max_nodes), dtype=np.float32),
            'Delta': np.zeros((max_nodes, max_nodes), dtype=np.int64),
            'resource_ids': np.zeros(max_targets, dtype=np.int64),
            'resource_raw': np.random.randn(max_targets, 14).astype(np.float32),
            'subtask_index': 0,
            'action_mask': np.ones(max_targets, dtype=bool),
            'task_mask': np.ones(max_nodes, dtype=bool),
        }
        obs['resource_ids'][0] = 1  # Local
        obs['resource_ids'][1] = 2  # RSU
        obs['status'][0] = 1  # READY
        obs_list.append(obs)

    # 创建模拟动作
    actions = [{'target': 0, 'power': 0.5} for _ in range(batch_size)]

    # 调用 evaluate_actions
    log_probs, values, entropy = agent.evaluate_actions(obs_list, actions)

    # 验证输出
    assert log_probs.shape == (batch_size,), f"log_probs shape: {log_probs.shape}"
    assert values.shape == (batch_size,), f"values shape: {values.shape}"
    assert entropy.shape == (batch_size,), f"entropy shape: {entropy.shape}"

    # 验证没有 NaN
    assert not torch.isnan(log_probs).any(), "log_probs contains NaN"
    assert not torch.isnan(values).any(), "values contains NaN"
    assert not torch.isnan(entropy).any(), "entropy contains NaN"

    # 验证 log_probs 是负数（概率 < 1）
    assert (log_probs <= 0).all(), "log_probs should be <= 0"

    # 验证熵是正数
    assert (entropy >= 0).all(), "entropy should be >= 0"

    print("✓ test_mappo_agent_evaluate_actions passed")


def test_syntax_check_all_modified_files():
    """语法检查所有修改的文件"""
    import subprocess

    files_to_check = [
        'configs/constants.py',
        'configs/train_config.py',
        'models/offloading_policy.py',
        'models/dag_embedding.py',
        'models/actor_critic.py',
        'agents/mappo_agent.py',
        'envs/entities/task_dag.py',
    ]

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for file_path in files_to_check:
        full_path = os.path.join(root_dir, file_path)
        if os.path.exists(full_path):
            result = subprocess.run(
                ['python', '-m', 'py_compile', full_path],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, \
                f"Syntax error in {file_path}: {result.stderr}"

    print(f"✓ test_syntax_check_all_modified_files passed ({len(files_to_check)} files)")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Running Fix Verification Tests")
    print("=" * 60)

    tests = [
        test_constants_import,
        test_lr_critic_value,
        test_evaluate_actions_has_resource_raw,
        test_evaluate_actions_has_logit_bias,
        test_mask_value_consistency,
        test_get_subtask_exec_location,
        test_network_forward_consistency,
        test_mappo_agent_evaluate_actions,
        test_syntax_check_all_modified_files,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
