#!/usr/bin/env python3
"""主执行脚本：运行所有评估和绘图"""
import sys
import subprocess
from pathlib import Path


def run_script(script_path: str, description: str):
    """运行Python脚本"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"脚本: {script_path}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True
    )
    
    # 打印输出
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"\n错误: {description} 执行失败")
        return False
    
    print(f"\n✓ {description} 完成")
    return True


def main():
    scripts_dir = Path(__file__).parent
    
    # 定义执行顺序
    tasks = [
        ("scripts/eval_baselines.py", "评估Baseline方法（LO/NRO/EFT-H）"),
        ("scripts/eval_models.py", "评估RL模型（F-MAPPO/TERA-MAPPO/IPPO-H/消融）"),
        ("runs/paper_final_results/group2_comprehensive_comparison/plot.py", "绘制Group 2综合对比图"),
        ("runs/paper_final_results/group3_ablation/plot.py", "绘制Group 3消融实验图"),
    ]
    
    print("开始执行论文最终结果生成流程...")
    print(f"总共 {len(tasks)} 个任务\n")
    
    for i, (script, desc) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {desc}")
        
        script_path = Path(__file__).parent.parent / script
        if not script_path.exists():
            print(f"错误: 脚本不存在 {script_path}")
            continue
        
        success = run_script(str(script_path), desc)
        if not success:
            print(f"\n任务失败，停止执行")
            return 1
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print("结果保存在: runs/paper_final_results/")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
