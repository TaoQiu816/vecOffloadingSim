import numpy as np

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv
from baselines.local_only_policy import LocalOnlyPolicy


def _summarize_decisions(env):
    """辅助函数：读取环境内部决策计数器。"""
    counters = getattr(env, "_reward_stats", None).counters if hasattr(env, "_reward_stats") else {}
    return {
        "local": counters.get("decision_local", 0),
        "rsu": counters.get("decision_rsu", 0),
        "v2v": counters.get("decision_v2v", 0),
        "total": counters.get("decision_total", 0),
    }


def run_verification():
    print("=== 开始环境逻辑与难度校准测试 ===")

    # 核心物理参数检查
    print(f"[Config] MIN_COMP={Cfg.MIN_COMP:.2e}, MAX_COMP={Cfg.MAX_COMP:.2e}, MEAN_COMP_LOAD={Cfg.MEAN_COMP_LOAD:.2e}")
    print(f"[Config] MAX_VEHICLE_CPU_FREQ={Cfg.MAX_VEHICLE_CPU_FREQ:.2e}, DT={Cfg.DT}, MAX_STEPS={Cfg.MAX_STEPS}")
    theoretical_time = Cfg.MEAN_COMP_LOAD / Cfg.MAX_VEHICLE_CPU_FREQ
    steps_needed = theoretical_time / Cfg.DT
    print(f"[Physics] 理论本地处理时间: {theoretical_time:.4f}s ({steps_needed:.1f} steps)")
    if steps_needed < 1.0:
        print("!!! 警告: 任务太简单，1个step内即可完成，无需卸载 !!!")
    else:
        print(">>> 正常: 任务需要多个step，卸载具有潜在价值。")

    # 1) Local-Only 动作映射验证
    print("\n--- 测试 Local-Only 动作索引 ---")
    env = VecOffloadingEnv()
    policy = LocalOnlyPolicy()
    obs_list, _ = env.reset(seed=0)
    actions = policy.select_action(obs_list)
    obs_list, rewards, terminated, truncated, info = env.step(actions)
    decision_counters = _summarize_decisions(env)
    curr_targets = [v.curr_target for v in env.vehicles]
    print(f"[Validation] 发送动作 target=0 后，实际 curr_target 列表: {curr_targets}")
    print(f"[Counters] local={decision_counters['local']}, rsu={decision_counters['rsu']}, v2v={decision_counters['v2v']}, total={decision_counters['total']}")

    # 2) 仅本地策略在高负载下的成功率
    print("\n--- 运行 Local-Only 压力测试 (高负载) ---")
    env2 = VecOffloadingEnv()
    policy.reset()
    obs_list, _ = env2.reset(seed=7)
    for step in range(int(Cfg.MAX_STEPS)):
        actions = policy.select_action(obs_list)
        obs_list, rewards, terminated, truncated, info = env2.step(actions)
        if terminated or truncated:
            break
    success_count = sum(1 for v in env2.vehicles if v.task_dag.is_finished)
    success_rate = success_count / max(len(env2.vehicles), 1)
    print(f"[Result] 仅本地策略完成率: {success_rate*100:.1f}% (车辆完成/总数 = {success_count}/{len(env2.vehicles)})")
    print("=== 测试结束 ===")


if __name__ == "__main__":
    run_verification()
