#!/usr/bin/env python3
"""训练指标数据完整性验证脚本

用法:
    python scripts/verify_training_metrics.py <training_stats.csv>

功能:
    - 检查成功率>0但时延=0的异常情况
    - 检查成功率>0但能耗=0的异常情况
    - 检查决策步数>0但物理步数=0的异常情况
"""

import pandas as pd
import sys
import os


def verify_metrics(csv_path):
    """验证训练指标数据完整性
    
    Args:
        csv_path: training_stats.csv 文件路径
        
    Returns:
        bool: True表示检查通过，False表示发现问题
    """
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return False
    
    issues = []
    
    # 检查1：成功率>0但时延=0
    if 'task_sr' in df.columns and 'task_duration_mean' in df.columns:
        mask = (df['task_sr'] > 0.01) & (df['task_duration_mean'] < 0.001)
        if mask.any():
            count = mask.sum()
            episodes = df[mask]['episode'].tolist()
            issues.append(
                f"发现 {count} 个episode：task_sr>1% 但 task_duration_mean≈0\n"
                f"  异常episode: {episodes[:10]}{'...' if len(episodes) > 10 else ''}"
            )
    
    # 检查2：成功率>0但能耗=0
    if 'task_sr' in df.columns and 'energy_mean' in df.columns:
        mask = (df['task_sr'] > 0.01) & (df['energy_mean'] < 0.001)
        if mask.any():
            count = mask.sum()
            episodes = df[mask]['episode'].tolist()
            issues.append(
                f"发现 {count} 个episode：task_sr>1% 但 energy_mean≈0\n"
                f"  异常episode: {episodes[:10]}{'...' if len(episodes) > 10 else ''}"
            )
    
    # 检查3：决策步数>0但物理步数=0
    if 'decision_steps' in df.columns and 'physical_steps' in df.columns:
        mask = (df['decision_steps'] > 0) & (df['physical_steps'] == 0)
        if mask.any():
            count = mask.sum()
            episodes = df[mask]['episode'].tolist()
            issues.append(
                f"发现 {count} 个episode：有决策但无物理推进\n"
                f"  异常episode: {episodes[:10]}{'...' if len(episodes) > 10 else ''}"
            )
    
    # 检查4：completed_tasks与task_sr的一致性
    if 'completed_tasks' in df.columns and 'task_sr' in df.columns:
        mask = (df['task_sr'] > 0.01) & (df['completed_tasks'] == 0)
        if mask.any():
            count = mask.sum()
            episodes = df[mask]['episode'].tolist()
            issues.append(
                f"发现 {count} 个episode：task_sr>1% 但 completed_tasks=0\n"
                f"  异常episode: {episodes[:10]}{'...' if len(episodes) > 10 else ''}"
            )
    
    # 输出结果
    if issues:
        print("❌ 数据完整性检查失败：\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}\n")
        return False
    else:
        print("✅ 数据完整性检查通过")
        print(f"   总episode数: {len(df)}")
        if 'task_sr' in df.columns:
            print(f"   平均成功率: {df['task_sr'].mean()*100:.2f}%")
        if 'task_duration_mean' in df.columns:
            valid_durations = df[df['task_duration_mean'] > 0]['task_duration_mean']
            if len(valid_durations) > 0:
                print(f"   平均任务时长: {valid_durations.mean():.3f}s")
        return True


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/verify_training_metrics.py <training_stats.csv>")
        print("\n示例:")
        print("  python scripts/verify_training_metrics.py runs/run_*/logs/training_stats.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    success = verify_metrics(csv_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
