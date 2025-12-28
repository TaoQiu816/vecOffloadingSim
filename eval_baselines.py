"""
基准策略评估脚本

评估以下策略并对比：
1. Random Policy: 随机卸载
2. Local-Only Policy: 全本地执行
3. Greedy Policy: 贪婪卸载（选择计算能力最强的节点）
4. MAPPO (Trained): 训练好的MAPPO智能体

评估指标：
- 平均回合奖励
- 任务成功率（车辆级、子任务级）
- 平均完成时间
- 卸载决策分布（Local/RSU/V2V比例）
- 平均队列长度
- 平均功率消耗
"""

import numpy as np
import torch
import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import SystemConfig as Cfg
from configs.train_config import TrainConfig as TC
from envs.vec_offloading_env import VecOffloadingEnv
from baselines import RandomPolicy, LocalOnlyPolicy, GreedyPolicy
from models.offloading_policy import OffloadingPolicyNetwork


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
    subtask_success_rates = []
    avg_completion_times = []
    decision_stats = {'local': 0, 'rsu': 0, 'v2v': 0}
    avg_queue_lengths = []
    avg_powers = []
    
    for ep in tqdm(range(num_episodes), desc=f"{policy_name}"):
        obs_list, _ = env.reset()
        policy.reset()
        
        ep_reward = 0
        ep_decisions = {'local': 0, 'rsu': 0, 'v2v': 0}
        ep_queue_sum = 0
        ep_power_sum = 0
        total_decisions = 0
        
        for step in range(TC.MAX_STEPS):
            # 获取动作
            if use_network:
                with torch.no_grad():
                    target_actions, power_actions, _, _ = policy.get_action_and_value(
                        obs_list, deterministic=True, device='cpu'
                    )
                    target_actions = target_actions.numpy()
                    power_actions = power_actions.numpy()
                
                actions = [
                    {'target': int(target_actions[i]), 'power': float(power_actions[i])}
                    for i in range(len(obs_list))
                ]
            else:
                actions = policy.select_action(obs_list)
            
            # 环境步进
            obs_list, rewards, done, info = env.step(actions)
            
            # 统计
            ep_reward += sum(rewards) / len(rewards)
            
            for i, action in enumerate(actions):
                target = action['target']
                power = action['power']
                
                # 决策分布统计
                if target == 0:
                    ep_decisions['local'] += 1
                elif target == 1:
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
        vehicle_success_rates.append(success_count / Cfg.NUM_VEHICLES * 100)
        
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
            completed_subtasks / total_subtasks * 100 if total_subtasks > 0 else 0
        )
        
        if completion_times:
            avg_completion_times.append(np.mean(completion_times))
        
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
        'local': decision_stats['local'] / total_decisions_all * 100 if total_decisions_all > 0 else 0,
        'rsu': decision_stats['rsu'] / total_decisions_all * 100 if total_decisions_all > 0 else 0,
        'v2v': decision_stats['v2v'] / total_decisions_all * 100 if total_decisions_all > 0 else 0
    }
    
    results = {
        'policy_name': policy_name,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_vehicle_success_rate': np.mean(vehicle_success_rates),
        'std_vehicle_success_rate': np.std(vehicle_success_rates),
        'avg_subtask_success_rate': np.mean(subtask_success_rates),
        'std_subtask_success_rate': np.std(subtask_success_rates),
        'avg_completion_time': np.mean(avg_completion_times) if avg_completion_times else 0,
        'std_completion_time': np.std(avg_completion_times) if avg_completion_times else 0,
        'decision_distribution': decision_distribution,
        'avg_queue_length': np.mean(avg_queue_lengths),
        'avg_power': np.mean(avg_powers)
    }
    
    # 打印结果
    print(f"\n结果摘要:")
    print(f"  平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  车辆成功率: {results['avg_vehicle_success_rate']:.1f}% ± {results['std_vehicle_success_rate']:.1f}%")
    print(f"  子任务成功率: {results['avg_subtask_success_rate']:.1f}% ± {results['std_subtask_success_rate']:.1f}%")
    print(f"  平均完成时间: {results['avg_completion_time']:.2f}s ± {results['std_completion_time']:.2f}s")
    print(f"  决策分布: Local={decision_distribution['local']:.1f}%, "
          f"RSU={decision_distribution['rsu']:.1f}%, V2V={decision_distribution['v2v']:.1f}%")
    print(f"  平均队列长度: {results['avg_queue_length']:.2f}")
    print(f"  平均功率: {results['avg_power']:.2f}")
    
    return results


def main():
    """主评估流程"""
    print("="*60)
    print("基准策略对比评估")
    print("="*60)
    
    # 创建环境
    env = VecOffloadingEnv()
    
    # 评估配置
    num_episodes = 50  # 每个策略评估50个回合
    
    all_results = []
    
    # 1. 评估随机策略
    random_policy = RandomPolicy(seed=42)
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
    
    # 4. 评估训练好的MAPPO（如果存在）
    mappo_model_path = "checkpoints/best_model.pth"
    if os.path.exists(mappo_model_path):
        print(f"\n检测到训练好的MAPPO模型: {mappo_model_path}")
        
        # 加载模型
        network = OffloadingPolicyNetwork(
            d_model=TC.EMBED_DIM,
            num_heads=TC.NUM_HEADS,
            num_layers=TC.NUM_LAYERS
        )
        
        checkpoint = torch.load(mappo_model_path, map_location='cpu')
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()
        
        results_mappo = evaluate_policy(
            env, network, "MAPPO (Trained)", num_episodes, use_network=True
        )
        all_results.append(results_mappo)
    else:
        print(f"\n未找到训练好的MAPPO模型，跳过MAPPO评估")
    
    # 保存结果
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/baseline_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("评估完成！结果已保存到: eval_results/baseline_comparison.json")
    print(f"{'='*60}")
    
    # 生成对比表格
    print("\n对比表格:")
    print(f"{'策略':<20} {'平均奖励':<12} {'车辆成功率':<12} {'子任务成功率':<12} {'平均完成时间':<12}")
    print("-" * 68)
    for result in all_results:
        print(f"{result['policy_name']:<20} "
              f"{result['avg_reward']:<12.2f} "
              f"{result['avg_vehicle_success_rate']:<12.1f} "
              f"{result['avg_subtask_success_rate']:<12.1f} "
              f"{result['avg_completion_time']:<12.2f}")


if __name__ == "__main__":
    main()

