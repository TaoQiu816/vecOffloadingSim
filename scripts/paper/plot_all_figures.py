#!/usr/bin/env python3
"""
统一绘图脚本 - 生成所有论文图表
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess


def run_script(script_path: Path) -> bool:
    """运行单个脚本"""
    try:
        print(f"\n{'='*60}")
        print(f"运行: {script_path.name}")
        print('='*60)
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        
        if result.returncode == 0:
            print(f"✓ {script_path.name} 完成")
            return True
        else:
            print(f"✗ {script_path.name} 失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"✗ {script_path.name} 出错: {e}")
        return False


def main():
    scripts_dir = Path(__file__).parent
    
    plot_scripts = [
        "plot_group2_bars.py",
        "plot_group3_bars.py",
        "plot_group4_lines.py",
        "plot_group5_lines.py",
        "plot_group6_analysis.py"
    ]
    
    print("开始生成所有论文图表...")
    print(f"脚本目录: {scripts_dir}")
    
    results = {}
    for script_name in plot_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"警告: {script_name} 不存在，跳过")
            results[script_name] = False
            continue
        
        results[script_name] = run_script(script_path)
    
    # 汇总结果
    print("\n" + "="*60)
    print("绘图任务汇总")
    print("="*60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for script_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {script_name}")
    
    print(f"\n完成: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n所有图表生成成功！")
        return 0
    else:
        print(f"\n警告: {total_count - success_count} 个脚本失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
