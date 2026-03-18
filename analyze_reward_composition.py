#!/usr/bin/env python3
"""
分析当前奖励函数的组成和各部分占比
"""
import json
import numpy as np
from pathlib import Path

def analyze_reward_composition(jsonl_path):
    """分析奖励组成"""
    
    # 读取最后100个episode的数据
    episodes = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    
    # 取最后100个episode
    recent_episodes = episodes[-100:] if len(episodes) > 100 else episodes
    
    print(f"分析最近 {len(recent_episodes)} 个episode的奖励组成\n")
    print("=" * 80)
    
    # 收集所有奖励分项的绝对值
    r_prog_abs_all = []
    r_term_abs_all = []
    r_illegal_abs_all = []
    cost_power_all = []
    cost_trust_all = []
    
    for ep in recent_episodes:
        metrics = ep.get('metrics', {})
        
        # 收集绝对值均值
        r_prog_abs = metrics.get('r_prog_abs', {}).get('abs_mean', 0.0)
        r_term_abs = metrics.get('r_term_abs', {}).get('abs_mean', 0.0)
        r_illegal_abs = metrics.get('r_illegal_abs', {}).get('abs_mean', 0.0)
        cost_power = metrics.get('cost_power', {}).get('mean', 0.0)
        cost_trust = metrics.get('cost_trust', {}).get('mean', 0.0)
        
        r_prog_abs_all.append(r_prog_abs)
        r_term_abs_all.append(r_term_abs)
        r_illegal_abs_all.append(r_illegal_abs)
        cost_power_all.append(cost_power)
        cost_trust_all.append(cost_trust)
    
    # 计算均值
    mean_r_prog_abs = np.mean(r_prog_abs_all)
    mean_r_term_abs = np.mean(r_term_abs_all)
    mean_r_illegal_abs = np.mean(r_illegal_abs_all)
    mean_cost_power = np.mean(cost_power_all)
    mean_cost_trust = np.mean(cost_trust_all)
    
    # 计算总和（用于占比计算）
    total_abs = mean_r_prog_abs + mean_r_term_abs + mean_r_illegal_abs
    
    print("\n【当前奖励函数方案】UNIFIED")
    print("-" * 80)
    print("\n奖励公式:")
    print("  r_total = r_step + r_term")
    print("  r_step = r_prog + r_illegal")
    print("  其中:")
    print("    - r_prog: 进度奖励，基于post-decision phi的改善")
    print("    - r_term: 终止奖励")
    print("    - r_illegal: 非法动作惩罚")
    print("\n辅助成本指标（不直接参与奖励计算，仅用于监控）:")
    print("    - cost_power: 能耗成本")
    print("    - cost_trust: 信任成本")
    
    print("\n" + "=" * 80)
    print("\n【奖励权重配置】")
    print("-" * 80)
    print(f"  W_ILLEGAL = 30.0          # 非法动作惩罚权重")
    print(f"  E_REF_UNIFIED = 0.02      # 能耗归一化参考值")
    print(f"  I_REF_D0 = V2V_RANGE/2    # 干扰归一化参考距离")
    print(f"  REWARD_PROGRESS_TNORM = 1.0  # 进度奖励时间归一化")
    print(f"  REWARD_PROGRESS_RMAX = 2.0   # 进度奖励上限")
    
    print("\n" + "=" * 80)
    print("\n【各部分绝对值均值】(最近{0}个episode)".format(len(recent_episodes)))
    print("-" * 80)
    print(f"  r_prog (进度奖励):      {mean_r_prog_abs:.6f}")
    print(f"  r_term (终止奖励):      {mean_r_term_abs:.6f}")
    print(f"  r_illegal (非法惩罚):   {mean_r_illegal_abs:.6f}")
    print(f"  ----------------------------------------")
    print(f"  总和:                   {total_abs:.6f}")
    
    print("\n辅助成本指标:")
    print(f"  cost_power (能耗):      {mean_cost_power:.6f}")
    print(f"  cost_trust (信任):      {mean_cost_trust:.6f}")
    
    print("\n" + "=" * 80)
    print("\n【各部分占比】(基于绝对值)")
    print("-" * 80)
    if total_abs > 0:
        ratio_prog = mean_r_prog_abs / total_abs * 100
        ratio_term = mean_r_term_abs / total_abs * 100
        ratio_illegal = mean_r_illegal_abs / total_abs * 100
        
        print(f"  r_prog:    {ratio_prog:6.2f}%  {'█' * int(ratio_prog/2)}")
        print(f"  r_term:    {ratio_term:6.2f}%  {'█' * int(ratio_term/2)}")
        print(f"  r_illegal: {ratio_illegal:6.2f}%  {'█' * int(ratio_illegal/2)}")
    else:
        print("  无法计算占比（总和为0）")
    
    print("\n" + "=" * 80)
    print("\n【奖励范围分析】")
    print("-" * 80)
    
    # 分析r_prog范围
    r_prog_means = []
    r_prog_mins = []
    r_prog_maxs = []
    
    for ep in recent_episodes:
        metrics = ep.get('metrics', {})
        r_prog_means.append(metrics.get('r_prog', {}).get('mean', 0.0))
        r_prog_mins.append(metrics.get('r_prog', {}).get('min', 0.0))
        r_prog_maxs.append(metrics.get('r_prog', {}).get('max', 0.0))
    
    print(f"\nr_prog (进度奖励):")
    print(f"  理论范围: [-1.0, 1.0]")
    print(f"  实际均值: [{np.min(r_prog_means):.4f}, {np.max(r_prog_means):.4f}]")
    print(f"  实际极值: [{np.min(r_prog_mins):.4f}, {np.max(r_prog_maxs):.4f}]")
    
    # 分析r_term范围
    r_term_means = []
    r_term_mins = []
    r_term_maxs = []
    
    for ep in recent_episodes:
        metrics = ep.get('metrics', {})
        r_term_means.append(metrics.get('r_term', {}).get('mean', 0.0))
        r_term_mins.append(metrics.get('r_term', {}).get('min', 0.0))
        r_term_maxs.append(metrics.get('r_term', {}).get('max', 0.0))
    
    print(f"\nr_term (终止奖励):")
    print(f"  理论范围: [-2.0, 2.0]")
    print(f"  实际均值: [{np.min(r_term_means):.4f}, {np.max(r_term_means):.4f}]")
    print(f"  实际极值: [{np.min(r_term_mins):.4f}, {np.max(r_term_maxs):.4f}]")
    print(f"  注: 终止奖励仅在episode结束时触发，大部分step为0")
    
    # 分析r_illegal
    r_illegal_means = []
    for ep in recent_episodes:
        metrics = ep.get('metrics', {})
        r_illegal_means.append(metrics.get('r_illegal', {}).get('mean', 0.0))
    
    print(f"\nr_illegal (非法动作惩罚):")
    print(f"  理论值: -30.0 (触发时) 或 0.0 (未触发)")
    print(f"  实际均值: [{np.min(r_illegal_means):.4f}, {np.max(r_illegal_means):.4f}]")
    
    print("\n" + "=" * 80)
    print("\n【关键观察】")
    print("-" * 80)
    
    # 计算非零率
    r_prog_nonzero_rates = []
    r_term_nonzero_rates = []
    r_illegal_nonzero_rates = []
    
    for ep in recent_episodes:
        metrics = ep.get('metrics', {})
        r_prog_stat = metrics.get('r_prog', {})
        r_term_stat = metrics.get('r_term', {})
        r_illegal_stat = metrics.get('r_illegal', {})
        
        if r_prog_stat.get('count', 0) > 0:
            r_prog_nonzero_rates.append(r_prog_stat.get('nonzero_count', 0) / r_prog_stat.get('count', 1))
        if r_term_stat.get('count', 0) > 0:
            r_term_nonzero_rates.append(r_term_stat.get('nonzero_count', 0) / r_term_stat.get('count', 1))
        if r_illegal_stat.get('count', 0) > 0:
            r_illegal_nonzero_rates.append(r_illegal_stat.get('nonzero_count', 0) / r_illegal_stat.get('count', 1))
    
    print(f"\n1. 激活频率:")
    print(f"   - r_prog 非零率:    {np.mean(r_prog_nonzero_rates)*100:.1f}%  (几乎每步都有)")
    print(f"   - r_term 非零率:    {np.mean(r_term_nonzero_rates)*100:.1f}%  (仅终止时)")
    print(f"   - r_illegal 非零率: {np.mean(r_illegal_nonzero_rates)*100:.1f}%  (训练良好时应为0)")
    
    print(f"\n2. 主导因素:")
    if total_abs > 0:
        if ratio_prog > 50:
            print(f"   - r_prog 占主导 ({ratio_prog:.1f}%)，说明进度信号是主要学习信号")
        elif ratio_term > 50:
            print(f"   - r_term 占主导 ({ratio_term:.1f}%)，说明终止奖励影响较大")
        else:
            print(f"   - 各部分较为均衡")
    
    print(f"\n3. 辅助成本:")
    print(f"   - cost_power 和 cost_trust 不直接参与奖励计算")
    print(f"   - 仅作为监控指标，用于分析策略的能耗和信任特性")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 分析最新的训练日志
    log_path = Path("runs/run_20260221_015014/logs/env_reward.jsonl")
    
    if log_path.exists():
        print(f"\n正在分析: {log_path}\n")
        analyze_reward_composition(log_path)
    else:
        print(f"错误: 找不到日志文件 {log_path}")
