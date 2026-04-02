#!/usr/bin/env python3
"""
主导出脚本 - 协调所有实验组的评估和图表生成
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "runs/paper_final_results_20260327"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_script(script_path: Path, description: str):
    """运行脚本并显示进度"""
    print(f"\n{'=' * 70}")
    print(f"执行: {description}")
    print(f"脚本: {script_path}")
    print(f"{'=' * 70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    print("=" * 70)
    print("论文实验结果主导出脚本")
    print("=" * 70)
    print(f"\n输出目录: {OUTPUT_ROOT}")
    
    # 检查脚本目录
    scripts_dir = ROOT / "scripts/paper"
    if not scripts_dir.exists():
        print(f"错误: 脚本目录不存在: {scripts_dir}")
        return 1
    
    results = {}
    
    # 第二组：综合性能对比
    print(f"\n{'#' * 70}")
    print("# 第二组：综合性能对比实验")
    print("#" * 70)
    eval_script = scripts_dir / "eval_group2_comprehensive.py"
    if eval_script.exists():
        results["group2_eval"] = run_script(eval_script, "第二组 - 评估")
    else:
        print(f"⚠ 脚本不存在: {eval_script}")
    
    # 第三组：消融实验
    print(f"\n{'#' * 70}")
    print("# 第三组：消融实验")
    print("#" * 70)
    eval_script = scripts_dir / "eval_group3_ablation.py"
    if eval_script.exists():
        results["group3_eval"] = run_script(eval_script, "第三组 - 评估")
    else:
        print(f"⚠ 脚本不存在: {eval_script}")
    
    # 第四组：任务复杂度与截止期敏感性
    print(f"\n{'#' * 70}")
    print("# 第四组：任务复杂度与截止期敏感性")
    print("#" * 70)
    eval_script = scripts_dir / "eval_group4_complexity.py"
    if eval_script.exists():
        results["group4_eval"] = run_script(eval_script, "第四组 - 评估")
    else:
        print(f"⚠ 脚本不存在: {eval_script}")
    
    # 第五组：系统负载与资源竞争
    print(f"\n{'#' * 70}")
    print("# 第五组：系统负载与资源竞争")
    print("#" * 70)
    eval_script = scripts_dir / "eval_group5_system_load.py"
    if eval_script.exists():
        results["group5_eval"] = run_script(eval_script, "第五组 - 评估")
    else:
        print(f"⚠ 脚本不存在: {eval_script}")
    
    # 第六组：机制分析
    print(f"\n{'#' * 70}")
    print("# 第六组：机制分析")
    print("#" * 70)
    eval_script = scripts_dir / "eval_group6_mechanism.py"
    if eval_script.exists():
        results["group6_eval"] = run_script(eval_script, "第六组 - 数据收集")
    else:
        print(f"⚠ 脚本不存在: {eval_script}")
    
    # 生成所有图表
    print(f"\n{'#' * 70}")
    print("# 生成所有图表")
    print("#" * 70)
    plot_script = scripts_dir / "plot_all_figures.py"
    if plot_script.exists():
        results["plot_all"] = run_script(plot_script, "绘制所有图表")
    else:
        print(f"⚠ 脚本不存在: {plot_script}")
    
    # 总结
    print(f"\n{'=' * 70}")
    print("执行总结")
    print(f"{'=' * 70}")
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {success_count}/{total_count} 成功")
    print(f"\n所有结果已保存到: {OUTPUT_ROOT}")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
