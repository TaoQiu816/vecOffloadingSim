"""
[基准策略评估脚本] eval_baselines.py
Baseline Policy Evaluation Script

作用 (Purpose):
    评估多种基准策略并与训练好的MAPPO智能体进行对比，验证强化学习方法的有效性。
    Evaluates multiple baseline policies and compares them with trained MAPPO agent 
    to validate the effectiveness of reinforcement learning approach.

评估策略 (Evaluated Policies):
    1. Random Policy - 随机选择卸载目标（Local/RSU/V2V）
    2. Local-Only Policy - 所有任务在本地执行（无卸载）
    3. Greedy Policy - 贪婪选择计算能力最强的节点
    4. MAPPO (Trained) - 训练好的MAPPO智能体

评估指标 (Evaluation Metrics):
    - 平均回合奖励 (Average Episode Reward)
    - 任务成功率 (Task Success Rate) - 车辆级、子任务级
    - 平均完成时间 (Average Completion Time)
    - 卸载决策分布 (Offloading Decision Distribution) - Local/RSU/V2V比例
    - 平均队列长度 (Average Queue Length)
    - 平均功率消耗 (Average Power Consumption)

使用方法 (Usage):
    python eval_baselines.py --model-path runs/run_XXX/models/best_model.pth --num-episodes 100
    python eval_baselines.py --num-episodes 50 --seed 42
"""

import numpy as np
import torch
import os
import sys
import json
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from configs.exp_dynamic_config import apply_exp_dynamic_config
from envs.vec_offloading_env import VecOffloadingEnv
from baselines import RandomPolicy, LocalOnlyPolicy, GreedyPolicy, EFTPPolicy
from models.offloading_policy import OffloadingPolicyNetwork


def _bind_eval_horizon():
    eval_steps = int(getattr(Cfg, "EVAL_MAX_STEPS", getattr(Cfg, "MAX_STEPS", 0)))
    if eval_steps <= 0:
        raise ValueError(f"EVAL_MAX_STEPS must be positive, got {eval_steps}")
    Cfg.MAX_STEPS = eval_steps
    TC.MAX_STEPS = eval_steps
    return eval_steps


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _classify_action_type(action, obs):
    """Classify action as local/rsu/v2v using observation metadata first."""
    target = int(action.get("target", 0))
    candidate_types = obs.get("candidate_types")
    if candidate_types is not None and 0 <= target < len(candidate_types):
        ctype = int(candidate_types[target])
        if ctype == 1:
            return "local"
        if ctype == 2:
            return "rsu"
        if ctype == 3:
            return "v2v"

    # Fallback path for older observations without candidate_types.
    explicit_type = action.get("target_type")
    if isinstance(explicit_type, str):
        t = explicit_type.strip().lower()
        if t in ("local", "rsu", "v2v"):
            return t
    if target == 0:
        return "local"
    if getattr(Cfg, "ENABLE_RSU_SELECTION", False):
        if 1 <= target <= int(getattr(Cfg, "NUM_RSU", 1)):
            return "rsu"
        return "v2v"
    return "rsu" if target == 1 else "v2v"


def evaluate_policy(env, policy, policy_name, num_episodes=50, use_network=False):
    """
    评估单个策略
    
    Args:
        env: 环境实例
        policy: 策略实例
        policy_name: 策略名称
        num_episodes: 评估回合数
        use_network: 是否使用神经网络策略（MAPPO）
    
    Returns:
        results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估策略: {policy_name}")
    print(f"{'='*60}")
    
    # 统计容器
    episode_rewards = []
    vehicle_success_rates = []
    task_success_rates = []
    subtask_success_rates = []
    avg_completion_times = []
    decision_stats = {'local': 0, 'rsu': 0, 'v2v': 0}
    avg_queue_lengths = []
    avg_powers = []
    avg_energy_consumptions = []
    avg_makespans = []
    deadline_meet_ratios = []
    
    eval_steps = _bind_eval_horizon()
    for ep in tqdm(range(num_episodes), desc=f"{policy_name}"):
        obs_list, _ = env.reset(seed=ep)
        if hasattr(policy, "reset") and callable(getattr(policy, "reset")):
            policy.reset()
        
        ep_reward = 0
        ep_decisions = {'local': 0, 'rsu': 0, 'v2v': 0}
        ep_queue_sum = 0
        ep_power_sum = 0
        total_decisions = 0
        last_info = None
        
        for step in range(eval_steps):
            # 获取动作
            current_obs = obs_list
            if use_network:
                with torch.no_grad():
                    subtask_actions, target_actions, power_actions, _, _, _, _ = policy.get_action_and_value(
                        obs_list, deterministic=True, device='cpu'
                    )
                    subtask_actions = subtask_actions.numpy()
                    target_actions = target_actions.numpy()
                    power_actions = power_actions.numpy()
                
                actions = []
                for i in range(len(obs_list)):
                    act = {
                        'subtask': int(subtask_actions[i]),
                        'target': int(target_actions[i]),
                        'power': float(power_actions[i]),
                    }
                    if "obs_stamp" in obs_list[i]:
                        act["obs_stamp"] = int(obs_list[i]["obs_stamp"])
                    actions.append(act)
            else:
                actions = policy.select_action(obs_list)
            
            # 环境步进
            obs_list, rewards, terminated, truncated, info = env.step(actions)
            last_info = info
            done = terminated or truncated
            
            # 统计
            ep_reward += sum(rewards) / len(rewards)
            
            for i, action in enumerate(actions):
                action_type = _classify_action_type(action, current_obs[i] if i < len(current_obs) else {})
                power = action['power']
                
                # 决策分布统计
                if action_type == "local":
                    ep_decisions['local'] += 1
                elif action_type == "rsu":
                    ep_decisions['rsu'] += 1
                else:
                    ep_decisions['v2v'] += 1
                
                # 队列和功率统计
                ep_queue_sum += env.vehicles[i].task_queue_len
                ep_power_sum += power
                total_decisions += 1
            
            if done:
                break
        
        # 回合结束统计
        episode_rewards.append(ep_reward)
        
        # 成功率统计
        success_count = sum([1 for v in env.vehicles if v.task_dag.is_finished])
        vehicle_success_rates.append(success_count / max(len(env.vehicles), 1))
        task_success_rates.append(success_count / max(len(env.vehicles), 1))
        
        total_subtasks = 0
        completed_subtasks = 0
        completion_times = []
        for v in env.vehicles:
            total_subtasks += v.task_dag.num_subtasks
            completed_subtasks += np.sum(v.task_dag.status == 3)
            
            # 统计完成时间（仅统计已完成的任务）
            if v.task_dag.is_finished:
                termination_nodes = [i for i in range(v.task_dag.num_subtasks) 
                                    if v.task_dag.out_degree[i] == 0]
                if termination_nodes:
                    ct_max = max(v.task_dag.CT[tn] for tn in termination_nodes)
                    completion_times.append(ct_max)
        
        subtask_success_rates.append(
            completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
        )
        
        if completion_times:
            avg_completion_times.append(np.mean(completion_times))
        epm = last_info.get("episode_metrics", {}) if isinstance(last_info, dict) else {}
        # 论文指标：ACT / Makespan / Energy / Deadline Meet Ratio
        energy_mean = epm.get("energy_norm_mean")
        if energy_mean is not None:
            try:
                avg_energy_consumptions.append(float(energy_mean))
            except Exception:
                pass
        makespan = epm.get("episode_time_seconds")
        if makespan is not None:
            try:
                avg_makespans.append(float(makespan))
            except Exception:
                pass
        dmr = epm.get("deadline_miss_rate")
        if dmr is not None:
            try:
                deadline_meet_ratios.append(float(np.clip(1.0 - float(dmr), 0.0, 1.0)))
            except Exception:
                pass
        
        # 决策分布累加
        decision_stats['local'] += ep_decisions['local']
        decision_stats['rsu'] += ep_decisions['rsu']
        decision_stats['v2v'] += ep_decisions['v2v']
        
        # 队列和功率统计
        avg_queue_lengths.append(ep_queue_sum / total_decisions if total_decisions > 0 else 0)
        avg_powers.append(ep_power_sum / total_decisions if total_decisions > 0 else 0)
    
    # 计算平均值
    total_decisions_all = sum(decision_stats.values())
    decision_distribution = {
        'local': decision_stats['local'] / total_decisions_all if total_decisions_all > 0 else 0.0,
        'rsu': decision_stats['rsu'] / total_decisions_all if total_decisions_all > 0 else 0.0,
        'v2v': decision_stats['v2v'] / total_decisions_all if total_decisions_all > 0 else 0.0
    }
    
    results = {
        'policy_name': policy_name,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_vehicle_success_rate': np.mean(vehicle_success_rates),
        'std_vehicle_success_rate': np.std(vehicle_success_rates),
        'avg_task_success_rate': np.mean(task_success_rates),
        'std_task_success_rate': np.std(task_success_rates),
        'avg_subtask_success_rate': np.mean(subtask_success_rates),
        'std_subtask_success_rate': np.std(subtask_success_rates),
        'avg_completion_time': np.mean(avg_completion_times) if avg_completion_times else 0,
        'std_completion_time': np.std(avg_completion_times) if avg_completion_times else 0,
        'avg_makespan': np.mean(avg_makespans) if avg_makespans else 0,
        'std_makespan': np.std(avg_makespans) if avg_makespans else 0,
        'avg_energy_consumption': np.mean(avg_energy_consumptions) if avg_energy_consumptions else 0,
        'std_energy_consumption': np.std(avg_energy_consumptions) if avg_energy_consumptions else 0,
        'deadline_meet_ratio': np.mean(deadline_meet_ratios) if deadline_meet_ratios else 0,
        'decision_distribution': decision_distribution,
        'avg_queue_length': np.mean(avg_queue_lengths),
        'avg_power': np.mean(avg_powers)
    }
    
    # 打印结果
    print(f"\n结果摘要:")
    print(f"  平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  车辆成功率: {results['avg_vehicle_success_rate']*100:.1f}% ± {results['std_vehicle_success_rate']*100:.1f}%")
    print(f"  子任务成功率: {results['avg_subtask_success_rate']*100:.1f}% ± {results['std_subtask_success_rate']*100:.1f}%")
    print(f"  平均完成时间: {results['avg_completion_time']:.2f}s ± {results['std_completion_time']:.2f}s")
    print(f"  平均Makespan: {results['avg_makespan']:.2f}s ± {results['std_makespan']:.2f}s")
    print(f"  平均能耗(归一化): {results['avg_energy_consumption']:.4f} ± {results['std_energy_consumption']:.4f}")
    print(f"  截止满足率: {results['deadline_meet_ratio']*100:.1f}%")
    print(f"  决策分布: Local={decision_distribution['local']*100:.1f}%, "
          f"RSU={decision_distribution['rsu']*100:.1f}%, V2V={decision_distribution['v2v']*100:.1f}%")
    print(f"  平均队列长度: {results['avg_queue_length']:.2f}")
    print(f"  平均功率: {results['avg_power']:.2f}")
    
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baselines and trained MAPPO policy.")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--exp-dynamic", action="store_true", default=False, help="Apply configs.exp_dynamic_config profile before evaluation.")
    return parser.parse_args()


def main():
    """主评估流程"""
    args = _parse_args()
    print("="*60)
    print("基准策略对比评估")
    print("="*60)
    if args.exp_dynamic:
        info = apply_exp_dynamic_config(Cfg, TC)
        print(f"[eval_baselines] dynamic profile applied: veh={info['num_vehicles']} rsu={info['num_rsu']} dag={info['dag_source']}")
    
    # 创建环境
    env = VecOffloadingEnv()
    
    # 评估配置
    num_episodes = int(args.num_episodes)
    
    all_results = []
    
    # 1. 评估随机策略
    random_policy = RandomPolicy(seed=int(args.seed))
    results_random = evaluate_policy(env, random_policy, "Random Policy", num_episodes)
    all_results.append(results_random)
    
    # 2. 评估全本地执行策略
    local_policy = LocalOnlyPolicy()
    results_local = evaluate_policy(env, local_policy, "Local-Only Policy", num_episodes)
    all_results.append(results_local)
    
    # 3. 评估贪婪策略
    greedy_policy = GreedyPolicy(env)
    results_greedy = evaluate_policy(env, greedy_policy, "Greedy Policy", num_episodes)
    all_results.append(results_greedy)

    # 4. 评估 EFT / HEFT-style 启发式
    eft_policy = EFTPPolicy(env)
    results_eft = evaluate_policy(env, eft_policy, "EFT Policy", num_episodes)
    all_results.append(results_eft)
    
    # 5. 评估训练好的MAPPO（如果存在）
    mappo_model_path = args.model_path
    if os.path.exists(mappo_model_path):
        print(f"\n检测到训练好的MAPPO模型: {mappo_model_path}")
        
        # 加载模型
        network = OffloadingPolicyNetwork(
            d_model=TC.EMBED_DIM,
            num_heads=TC.NUM_HEADS,
            num_layers=TC.NUM_LAYERS
        )
        
        checkpoint = torch.load(mappo_model_path, map_location='cpu')
        network.load_state_dict(checkpoint['network_state_dict'], strict=False)
        network.eval()
        
        results_mappo = evaluate_policy(
            env, network, "MAPPO (Trained)", num_episodes, use_network=True
        )
        all_results.append(results_mappo)
    else:
        print(f"\n未找到训练好的MAPPO模型，跳过MAPPO评估")
    
    # 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/baseline_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    # 论文核心指标汇总CSV
    core_csv = os.path.join(output_dir, "baseline_core_metrics.csv")
    with open(core_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy_name",
                "avg_completion_time",
                "avg_makespan",
                "avg_energy_consumption",
                "avg_task_success_rate",
                "deadline_meet_ratio",
                "avg_reward",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "policy_name": r.get("policy_name"),
                "avg_completion_time": r.get("avg_completion_time"),
                "avg_makespan": r.get("avg_makespan"),
                "avg_energy_consumption": r.get("avg_energy_consumption"),
                "avg_task_success_rate": r.get("avg_task_success_rate"),
                "deadline_meet_ratio": r.get("deadline_meet_ratio"),
                "avg_reward": r.get("avg_reward"),
            })
    
    print(f"\n{'='*60}")
    print(f"评估完成！结果已保存到: {output_dir}/baseline_comparison.json")
    print(f"论文核心指标CSV已保存到: {core_csv}")
    print(f"{'='*60}")
    
    # 生成对比表格
    print("\n对比表格:")
    print(f"{'策略':<20} {'平均奖励':<12} {'车辆成功率':<12} {'子任务成功率':<12} {'平均完成时间':<12}")
    print("-" * 68)
    for result in all_results:
        print(f"{result['policy_name']:<20} "
              f"{result['avg_reward']:<12.2f} "
              f"{result['avg_vehicle_success_rate']*100:<12.1f} "
              f"{result['avg_subtask_success_rate']*100:<12.1f} "
              f"{result['avg_completion_time']:<12.2f}")
    env.close()


if __name__ == "__main__":
    main()
