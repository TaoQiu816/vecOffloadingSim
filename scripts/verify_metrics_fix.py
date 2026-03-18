#!/usr/bin/env python3
"""
验证脚本：测试时延和能耗指标修复效果

运行方式：
    python scripts/verify_metrics_fix.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path


def verify_metrics_from_logs():
    """从最近的训练日志验证指标"""
    print("=" * 80)
    print("验证时延能耗指标修复效果")
    print("=" * 80)
    
    # 查找最近的运行目录
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ runs目录不存在")
        return False
    
    # 获取最新的运行目录
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        print("❌ 没有找到运行目录")
        return False
    
    latest_run = run_dirs[-1]
    print(f"\n检查运行目录: {latest_run}")
    
    # 检查env_reward.jsonl
    reward_log = latest_run / "logs" / "env_reward.jsonl"
    if not reward_log.exists():
        print(f"❌ 日志文件不存在: {reward_log}")
        return False
    
    print(f"\n读取日志文件: {reward_log}")
    
    # 读取最后几行
    with open(reward_log, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ 日志文件为空")
        return False
    
    print(f"日志文件共 {len(lines)} 行")
    
    # 分析最后几个episode
    issues = []
    success_count = 0
    
    for i, line in enumerate(lines[-10:], start=max(0, len(lines)-10)):
        try:
            data = json.loads(line)
            episode = data.get('episode', i)
            
            task_duration_mean = data.get('task_duration_mean')
            energy_norm_mean = data.get('energy_norm_mean')
            completed_tasks = data.get('completed_tasks_count', 0)
            task_sr = data.get('task_success_rate', 0.0)
            
            print(f"\nEpisode {episode}:")
            print(f"  task_duration_mean: {task_duration_mean}")
            print(f"  energy_norm_mean: {energy_norm_mean}")
            print(f"  completed_tasks_count: {completed_tasks}")
            print(f"  task_success_rate: {task_sr:.1%}")
            
            # 检查问题
            if completed_tasks > 0:
                if task_duration_mean is None or task_duration_mean < 0.001:
                    issues.append(f"Episode {episode}: 有完成任务但时延为 {task_duration_mean}")
                else:
                    success_count += 1
                    print(f"  ✅ 时延指标正常")
                
                if energy_norm_mean is None or energy_norm_mean < 0.001:
                    issues.append(f"Episode {episode}: 能耗指标异常 {energy_norm_mean}")
                else:
                    print(f"  ✅ 能耗指标正常")
            
        except json.JSONDecodeError:
            print(f"⚠️  第 {i} 行JSON解析失败")
            continue
        except Exception as e:
            print(f"⚠️  第 {i} 行处理失败: {e}")
            continue
    
    # 输出结论
    print("\n" + "=" * 80)
    print("验证结论")
    print("=" * 80)
    
    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✅ 验证通过！成功记录了 {success_count} 个episode的指标")
        print("\n修复效果:")
        print("  - 任务完成时间（Lat）正确记录")
        print("  - 能耗归一化值（En）正确记录")
        print("  - 指标统计逻辑正常工作")
        return True


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 80)
    print("使用指南")
    print("=" * 80)
    print("\n1. 运行训练测试:")
    print("   python train.py --max-episodes 5 --device cpu")
    print("\n2. 验证修复效果:")
    print("   python scripts/verify_metrics_fix.py")
    print("\n3. 查看训练日志:")
    print("   tail -f runs/run_*/logs/env_reward.jsonl")
    print("\n4. 预期输出:")
    print("   Ep    Wall    R/step     T_SR     V_SR     S_SR            L/R/V   Lat(s)       En")
    print("      1    14.2s   -0.0117     5.0%     5.0%    47.5%      33%/14%/53%    0.523    0.145")
    print("                                                                          ^^^^     ^^^^")
    print("                                                                       (非0.000)  (非-)")


if __name__ == "__main__":
    success = verify_metrics_from_logs()
    print_usage_guide()
    
    sys.exit(0 if success else 1)
