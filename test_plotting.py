"""
测试新增绘图功能

用法:
    python test_plotting.py --run-dir runs/run_20260105_021203
"""

import argparse
import os
import sys
from utils.data_recorder import DataRecorder


def main():
    parser = argparse.ArgumentParser(description='测试训练结果可视化')
    parser.add_argument('--run-dir', type=str, required=True, 
                       help='训练运行目录（例如: runs/run_20260105_021203）')
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.run_dir):
        print(f"错误: 目录不存在: {args.run_dir}")
        sys.exit(1)
    
    episode_log = os.path.join(args.run_dir, 'episode_log.csv')
    if not os.path.exists(episode_log):
        print(f"错误: 未找到episode_log.csv")
        sys.exit(1)
    
    print(f"📊 开始生成图表...")
    print(f"输入: {episode_log}")
    print(f"输出: {os.path.join(args.run_dir, 'plots')}")
    print()
    
    # 创建DataRecorder并生成图表
    recorder = DataRecorder(base_dir=args.run_dir, quiet=False)
    recorder.auto_plot()
    
    print()
    print("="*70)
    print("✅ 图表生成完成！")
    print("="*70)
    
    # 统计生成的图表数量
    plot_dir = os.path.join(args.run_dir, 'plots')
    if os.path.exists(plot_dir):
        plots = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        print(f"\n生成图表数量: {len(plots)}")
        print("\n图表列表:")
        for i, plot in enumerate(sorted(plots), 1):
            print(f"  {i:2d}. {plot}")
    
    print(f"\n详细说明请查看: PLOTTING_GUIDE.md")


if __name__ == '__main__':
    main()

