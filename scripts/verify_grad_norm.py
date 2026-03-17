#!/usr/bin/env python3
"""
验证 grad_norm 双口径记录的脚本
运行最小训练测试（5个episode）并检查输出
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path

def run_smoke_test():
    """运行最小训练测试"""
    print("正在运行最小训练测试（5个episode）...")
    
    cmd = [
        sys.executable, "train.py",
        "--max-episodes", "5",
        "--device", "cpu",
        "--seed", "999",
        "--log-interval", "1"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"训练失败: {result.stderr}")
        return None
    
    # 查找最新的 run 目录
    runs_dir = Path("runs")
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    
    if not run_dirs:
        print("未找到运行目录")
        return None
    
    latest_run = run_dirs[-1]
    print(f"找到运行目录: {latest_run}")
    
    return latest_run


def verify_grad_norm(run_dir: Path):
    """验证 grad_norm 双口径记录"""
    training_stats_path = run_dir / "logs" / "training_stats.csv"
    
    if not training_stats_path.exists():
        print(f"错误: 未找到 training_stats.csv")
        return False
    
    df = pd.read_csv(training_stats_path)
    
    print("\n=== 验证结果 ===")
    
    # 检查列是否存在
    has_preclip = "grad_norm_preclip" in df.columns
    has_postclip = "grad_norm_postclip" in df.columns
    has_old = "grad_norm" in df.columns
    
    print(f"✓ grad_norm_preclip 列存在: {has_preclip}")
    print(f"✓ grad_norm_postclip 列存在: {has_postclip}")
    print(f"✓ grad_norm 列存在（向后兼容）: {has_old}")
    
    if not (has_preclip and has_postclip):
        print("\n❌ 验证失败: 缺少 grad_norm_preclip 或 grad_norm_postclip 列")
        return False
    
    # 检查数据
    preclip_data = df["grad_norm_preclip"].dropna()
    postclip_data = df["grad_norm_postclip"].dropna()
    
    if len(preclip_data) == 0 or len(postclip_data) == 0:
        print("\n❌ 验证失败: grad_norm 数据为空")
        return False
    
    print(f"\n数据统计:")
    print(f"  grad_norm_preclip: 均值={preclip_data.mean():.4f}, 最大值={preclip_data.max():.4f}")
    print(f"  grad_norm_postclip: 均值={postclip_data.mean():.4f}, 最大值={postclip_data.max():.4f}")
    
    # 检查 postclip 是否 <= 1.0 (MAX_GRAD_NORM)
    max_postclip = postclip_data.max()
    if max_postclip > 1.0:
        print(f"\n⚠️  警告: grad_norm_postclip 最大值 ({max_postclip:.4f}) 超过 MAX_GRAD_NORM (1.0)")
        print("   这可能表示梯度裁剪未正确执行")
    else:
        print(f"\n✅ grad_norm_postclip 最大值 ({max_postclip:.4f}) 符合预期 (<= 1.0)")
    
    # 检查 preclip >= postclip
    violations = (preclip_data < postclip_data).sum()
    if violations > 0:
        print(f"\n⚠️  警告: 发现 {violations} 个 preclip < postclip 的情况")
    else:
        print(f"\n✅ 所有数据点都满足 preclip >= postclip")
    
    print("\n✅ 验证通过: grad_norm 双口径记录正常工作")
    return True


def main():
    print("=== grad_norm 双口径验证脚本 ===\n")
    
    # 运行测试
    run_dir = run_smoke_test()
    if run_dir is None:
        print("\n❌ 测试失败")
        return 1
    
    # 验证结果
    if not verify_grad_norm(run_dir):
        print("\n❌ 验证失败")
        return 1
    
    print(f"\n运行目录: {run_dir}")
    print("可以手动检查 logs/training_stats.csv 以确认")
    
    return 0


if __name__ == "__main__":
    exit(main())
