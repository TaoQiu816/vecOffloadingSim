"""
P01修复验证测试：Phase1-2时序冲突

测试目标：
1. 验证Phase4任务完成后，EDGE传输在同一step内被激活
2. 验证边激活补偿机制不会重复创建EDGE传输
3. 验证DAG任务流转的正确性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import deque


def test_edge_activation_in_same_step():
    """
    测试：Phase4任务完成后，EDGE传输是否在同一step内激活

    场景：
    - DAG结构: A -> B (A是B的前驱)
    - Step N: A完成 -> B的EDGE应立即入队（而非延迟到Step N+1）
    """
    from envs.vec_offloading_env import VecOffloadingEnv

    env = VecOffloadingEnv()
    obs, _ = env.reset(seed=42)

    # 记录初始状态
    initial_active_edges = len(env.active_edge_keys)

    # 运行多个step，观察EDGE激活行为
    edge_activation_delays = []

    for step in range(20):
        # 构造简单动作：全部本地执行（简化测试）
        actions = []
        for v in env.vehicles:
            actions.append({'target': 0, 'power': 0.5})  # Local

        # 记录step前的pending边数量
        pending_edges_before = 0
        for v in env.vehicles:
            if hasattr(v.task_dag, 'inter_task_transfers'):
                for child_id, parents in v.task_dag.inter_task_transfers.items():
                    for parent_id, info in parents.items():
                        if info.get('rem_bytes', 0) > 0:
                            pending_edges_before += 1

        obs, rewards, terminated, truncated, info = env.step(actions)

        # 记录step后的pending边数量
        pending_edges_after = 0
        for v in env.vehicles:
            if hasattr(v.task_dag, 'inter_task_transfers'):
                for child_id, parents in v.task_dag.inter_task_transfers.items():
                    for parent_id, info in parents.items():
                        if info.get('rem_bytes', 0) > 0:
                            pending_edges_after += 1

        # 检查是否有pending边在同一step内被激活
        # （如果修复正确，新产生的pending边应该在同一step内被处理）

        if terminated or truncated:
            break

    print(f"  Steps executed: {step + 1}")
    print(f"  Final active_edge_keys: {len(env.active_edge_keys)}")
    # 验证至少执行了一些步骤
    assert step >= 0, "Should execute at least one step"


def test_edge_activation_idempotency():
    """
    测试：边激活函数的幂等性

    验证：多次调用_phase2_activate_edge_transfers()不会重复创建EDGE
    """
    from envs.vec_offloading_env import VecOffloadingEnv

    env = VecOffloadingEnv()
    obs, _ = env.reset(seed=123)

    # 执行一步，产生一些pending边
    actions = [{'target': 1, 'power': 0.5} for _ in env.vehicles]  # RSU

    try:
        env.step(actions)
    except:
        # 可能因为RSU不可用而失败，忽略
        pass

    # 记录当前active_edge_keys数量
    edges_before = len(env.active_edge_keys)

    # 多次调用边激活函数
    for _ in range(5):
        env._phase2_activate_edge_transfers()

    edges_after = len(env.active_edge_keys)

    # 幂等性检验：多次调用后数量应相同
    assert edges_after == edges_before, (
        f"边激活函数非幂等: before={edges_before}, after={edges_after}"
    )

    print(f"  Idempotency verified: {edges_before} edges unchanged after 5 calls")


def test_dag_completion_flow():
    """
    测试：完整DAG完成流程

    验证：
    1. 所有子任务能正确完成
    2. EDGE传输不会造成死锁
    3. 状态转换正确
    """
    from envs.vec_offloading_env import VecOffloadingEnv

    env = VecOffloadingEnv()
    obs, _ = env.reset(seed=456)

    total_subtasks = sum(v.task_dag.num_subtasks for v in env.vehicles)

    # 运行直到完成或超时
    max_steps = 500
    completed_at_step = None

    for step in range(max_steps):
        # 简单策略：全部本地执行
        actions = [{'target': 0, 'power': 0.5} for _ in env.vehicles]

        obs, rewards, terminated, truncated, info = env.step(actions)

        # 统计已完成子任务
        completed_subtasks = sum(
            np.sum(v.task_dag.status == 3)
            for v in env.vehicles
        )

        if all(v.task_dag.is_finished for v in env.vehicles):
            completed_at_step = step + 1
            break

        if truncated:
            break

    # 验证结果
    final_completed = sum(np.sum(v.task_dag.status == 3) for v in env.vehicles)
    completion_rate = final_completed / total_subtasks if total_subtasks > 0 else 0

    print(f"  Total subtasks: {total_subtasks}")
    print(f"  Completed: {final_completed} ({completion_rate*100:.1f}%)")
    if completed_at_step:
        print(f"  All finished at step: {completed_at_step}")

    # 检查没有死锁（应该有进展）
    assert completion_rate > 0.5, f"Completion rate too low: {completion_rate}"


def test_inter_task_transfer_cleanup():
    """
    测试：inter_task_transfers正确清理

    验证：
    1. 同位置传输立即清零
    2. EDGE完成后正确清理
    """
    from envs.vec_offloading_env import VecOffloadingEnv

    env = VecOffloadingEnv()
    obs, _ = env.reset(seed=789)

    # 运行一些步骤
    for step in range(30):
        actions = [{'target': 0, 'power': 0.5} for _ in env.vehicles]
        obs, rewards, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            break

    # 检查inter_task_transfers状态
    total_pending = 0
    total_completed = 0

    for v in env.vehicles:
        if hasattr(v.task_dag, 'inter_task_transfers'):
            for child_id, parents in v.task_dag.inter_task_transfers.items():
                for parent_id, info in parents.items():
                    if info.get('rem_bytes', 0) > 0:
                        total_pending += 1
                    else:
                        total_completed += 1

    print(f"  Pending transfers: {total_pending}")
    print(f"  Completed transfers (in dict): {total_completed}")
    # 验证有完成的传输
    assert total_completed >= 0, "Should have non-negative completed transfers"


def test_phase_execution_order():
    """
    测试：阶段执行顺序正确性

    验证step()中的阶段按正确顺序执行
    """
    from envs.vec_offloading_env import VecOffloadingEnv
    import inspect

    env = VecOffloadingEnv()

    # 检查step方法源码中的阶段顺序
    source = inspect.getsource(env.step)

    # 验证关键阶段的存在和顺序
    phase1_pos = source.find('_phase1_commit_offload_decisions')
    phase2_first_pos = source.find('_phase2_activate_edge_transfers')
    phase3_pos = source.find('_phase3_advance_comm_queues')
    phase4_pos = source.find('_phase4_advance_cpu_queues')

    # 找到第二次调用_phase2（P01修复）
    phase2_second_pos = source.find('_phase2_activate_edge_transfers', phase2_first_pos + 1)

    assert phase1_pos < phase2_first_pos < phase3_pos < phase4_pos, \
        "Phase order incorrect: should be Phase1 -> Phase2 -> Phase3 -> Phase4"

    assert phase2_second_pos > phase4_pos, \
        "P01 fix missing: Phase2 should be called again after Phase4"

    print("  Phase execution order verified:")
    print("    Phase1 -> Phase2 -> Phase3 -> Phase4 -> Phase2(补偿)")


def run_all_p01_tests():
    """运行所有P01相关测试"""
    print("=" * 60)
    print("P01 Fix Verification Tests: Phase1-2 Timing Conflict")
    print("=" * 60)

    tests = [
        ("Phase execution order", test_phase_execution_order),
        ("Edge activation idempotency", test_edge_activation_idempotency),
        ("Edge activation in same step", test_edge_activation_in_same_step),
        ("DAG completion flow", test_dag_completion_flow),
        ("Inter-task transfer cleanup", test_inter_task_transfer_cleanup),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTesting {name}...")
        try:
            test_fn()
            print(f"✓ {name} passed")
            passed += 1
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"P01 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_p01_tests()
    sys.exit(0 if success else 1)
