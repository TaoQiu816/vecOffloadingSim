#!/usr/bin/env python3
"""
图表分析脚本
分析 plots/ 目录中的所有图表文件，生成详细的分析报告
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import sys

# 图表分类定义
PLOT_CATEGORIES = {
    "收敛性分析": {
        "patterns": ["convergence", "reward_curve", "training_stability"],
        "description": "评估训练过程的收敛性和稳定性",
        "importance": "critical"
    },
    "性能指标": {
        "patterns": ["success_rate", "latency", "performance"],
        "description": "衡量算法的核心性能表现",
        "importance": "critical"
    },
    "训练健康度": {
        "patterns": ["training_diagnostics", "loss_curve", "training_overview"],
        "description": "监控训练过程的健康状态",
        "importance": "high"
    },
    "奖励分解": {
        "patterns": ["reward_decomposition", "reward_analysis"],
        "description": "分析奖励函数各组成部分的贡献",
        "importance": "high"
    },
    "多智能体指标": {
        "patterns": ["multi_agent", "ma_"],
        "description": "评估多智能体协作和公平性",
        "importance": "high"
    },
    "系统指标": {
        "patterns": ["system_metrics", "constraint_health", "physical_metrics"],
        "description": "监控系统资源和约束满足情况",
        "importance": "medium"
    },
    "策略演化": {
        "patterns": ["policy_evolution", "decision_distribution", "offloading_ratio"],
        "description": "追踪策略的演化过程",
        "importance": "medium"
    },
    "资源利用": {
        "patterns": ["resource_utilization", "cpu_efficiency", "queue"],
        "description": "分析资源使用效率",
        "importance": "medium"
    },
    "综合仪表盘": {
        "patterns": ["summary_dashboard", "overview"],
        "description": "提供整体训练情况的综合视图",
        "importance": "high"
    }
}


class PlotAnalyzer:
    """图表分析器"""
    
    def __init__(self, plots_dir: str):
        self.plots_dir = Path(plots_dir)
        self.manifest_path = self.plots_dir / "plot_manifest.json"
        self.manifest = None
        self.plots_by_category = {}
        
    def load_manifest(self) -> bool:
        """加载 plot_manifest.json"""
        if not self.manifest_path.exists():
            print(f"⚠ 未找到 plot_manifest.json")
            return False
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        print(f"✓ 加载 manifest: {self.manifest['plot_count']} 个图表")
        return True
    
    def categorize_plots(self):
        """将图表按类别分类"""
        if not self.manifest:
            return
        
        # 初始化分类
        for category in PLOT_CATEGORIES:
            self.plots_by_category[category] = []
        self.plots_by_category["其他"] = []
        
        # 分类图表
        for fig in self.manifest['figures']:
            filename = fig['file']
            categorized = False
            
            for category, info in PLOT_CATEGORIES.items():
                for pattern in info['patterns']:
                    if pattern in filename.lower():
                        self.plots_by_category[category].append(fig)
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                self.plots_by_category["其他"].append(fig)
    
    def get_image_info(self, filepath: str) -> Dict:
        """获取图像文件信息"""
        try:
            with Image.open(filepath) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode
                }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_plot(self, fig: Dict) -> Dict:
        """分析单个图表"""
        filepath = fig['path']
        analysis = {
            "file": fig['file'],
            "exists": os.path.exists(filepath),
            "size_kb": fig['size_bytes'] / 1024,
            "tags": fig.get('tags', [])
        }
        
        if analysis['exists']:
            img_info = self.get_image_info(filepath)
            analysis.update(img_info)
        
        return analysis
    
    def generate_report(self) -> str:
        """生成分析报告"""
        lines = []
        
        # 标题
        lines.append("# 图表分析报告")
        lines.append("")
        lines.append(f"**生成时间**: {self.manifest['generated_at']}")
        lines.append(f"**图表总数**: {self.manifest['plot_count']}")
        lines.append(f"**图表目录**: `{self.plots_dir}`")
        lines.append("")
        
        # 标签统计
        lines.append("## 标签统计")
        lines.append("")
        for tag, count in self.manifest['tag_counts'].items():
            lines.append(f"- **{tag}**: {count} 个图表")
        lines.append("")
        
        # 按类别分析
        lines.append("## 图表分类分析")
        lines.append("")
        
        for category, info in PLOT_CATEGORIES.items():
            plots = self.plots_by_category.get(category, [])
            if not plots:
                continue
            
            lines.append(f"### {category}")
            lines.append("")
            lines.append(f"**作用**: {info['description']}")
            lines.append(f"**重要性**: {info['importance'].upper()}")
            lines.append(f"**图表数量**: {len(plots)}")
            lines.append("")
            
            # 列出该类别的图表
            for fig in plots:
                analysis = self.analyze_plot(fig)
                size_mb = analysis['size_kb'] / 1024
                
                lines.append(f"#### [{fig['file']}](plots/{fig['file']})")
                lines.append("")
                lines.append(f"- **文件大小**: {size_mb:.2f} MB")
                
                if analysis['exists']:
                    if 'width' in analysis:
                        lines.append(f"- **分辨率**: {analysis['width']} × {analysis['height']}")
                        lines.append(f"- **格式**: {analysis['format']}")
                else:
                    lines.append(f"- ⚠ **文件不存在**")
                
                if analysis['tags']:
                    lines.append(f"- **标签**: {', '.join(analysis['tags'])}")
                
                lines.append("")
        
        # 其他图表
        other_plots = self.plots_by_category.get("其他", [])
        if other_plots:
            lines.append("### 其他图表")
            lines.append("")
            for fig in other_plots:
                analysis = self.analyze_plot(fig)
                size_mb = analysis['size_kb'] / 1024
                lines.append(f"- **{fig['file']}** ({size_mb:.2f} MB)")
            lines.append("")
        
        # 关键图表推荐
        lines.append("## 关键图表推荐")
        lines.append("")
        lines.append("基于重要性和文件大小，以下是最值得关注的图表：")
        lines.append("")
        
        # 收集关键图表
        key_plots = []
        for category, info in PLOT_CATEGORIES.items():
            if info['importance'] in ['critical', 'high']:
                plots = self.plots_by_category.get(category, [])
                for fig in plots:
                    key_plots.append((category, fig, info['importance']))
        
        # 按重要性和大小排序
        key_plots.sort(key=lambda x: (
            0 if x[2] == 'critical' else 1,
            -x[1]['size_bytes']
        ))
        
        for i, (category, fig, importance) in enumerate(key_plots[:10], 1):
            size_mb = fig['size_bytes'] / 1024 / 1024
            lines.append(f"{i}. **[{fig['file']}](plots/{fig['file']})** ({category})")
            lines.append(f"   - 重要性: {importance.upper()}")
            lines.append(f"   - 大小: {size_mb:.2f} MB")
            if fig.get('tags'):
                lines.append(f"   - 标签: {', '.join(fig['tags'])}")
            lines.append("")
        
        # 图表解读建议
        lines.append("## 图表解读建议")
        lines.append("")
        
        lines.append("### 1. 收敛性评估")
        lines.append("")
        lines.append("查看以下图表判断训练是否收敛：")
        lines.append("")
        conv_plots = self.plots_by_category.get("收敛性分析", [])
        for fig in conv_plots:
            lines.append(f"- [`{fig['file']}`](plots/{fig['file']})")
        lines.append("")
        lines.append("**关注点**:")
        lines.append("- 奖励曲线是否趋于稳定")
        lines.append("- 是否存在剧烈波动或发散")
        lines.append("- 与基线方法的对比情况")
        lines.append("")
        
        lines.append("### 2. 性能指标分析")
        lines.append("")
        lines.append("评估算法性能的核心指标：")
        lines.append("")
        perf_plots = self.plots_by_category.get("性能指标", [])
        for fig in perf_plots:
            lines.append(f"- [`{fig['file']}`](plots/{fig['file']})")
        lines.append("")
        lines.append("**关注点**:")
        lines.append("- 成功率是否达到预期")
        lines.append("- 延迟指标是否满足约束")
        lines.append("- 与基线方法的性能差距")
        lines.append("")
        
        lines.append("### 3. 训练健康度检查")
        lines.append("")
        lines.append("确保训练过程健康：")
        lines.append("")
        health_plots = self.plots_by_category.get("训练健康度", [])
        for fig in health_plots:
            lines.append(f"- [`{fig['file']}`](plots/{fig['file']})")
        lines.append("")
        lines.append("**关注点**:")
        lines.append("- Loss 是否正常下降")
        lines.append("- 是否存在梯度爆炸/消失")
        lines.append("- 训练稳定性指标")
        lines.append("")
        
        lines.append("### 4. 奖励分解分析")
        lines.append("")
        lines.append("理解奖励函数的组成：")
        lines.append("")
        reward_plots = self.plots_by_category.get("奖励分解", [])
        for fig in reward_plots:
            lines.append(f"- [`{fig['file']}`](plots/{fig['file']})")
        lines.append("")
        lines.append("**关注点**:")
        lines.append("- 各奖励分量的贡献比例")
        lines.append("- 是否存在某个分量主导")
        lines.append("- 奖励设计是否合理")
        lines.append("")
        
        lines.append("### 5. 多智能体协作")
        lines.append("")
        lines.append("评估多智能体系统的协作效果：")
        lines.append("")
        ma_plots = self.plots_by_category.get("多智能体指标", [])
        for fig in ma_plots:
            lines.append(f"- [`{fig['file']}`](plots/{fig['file']})")
        lines.append("")
        lines.append("**关注点**:")
        lines.append("- 智能体间的协作程度")
        lines.append("- 奖励分配的公平性")
        lines.append("- 是否存在搭便车现象")
        lines.append("")
        
        # 数据质量评估
        lines.append("## 数据质量评估")
        lines.append("")
        
        total_size = sum(fig['size_bytes'] for fig in self.manifest['figures'])
        lines.append(f"- **图表总大小**: {total_size / 1024 / 1024:.2f} MB")
        lines.append(f"- **平均图表大小**: {total_size / len(self.manifest['figures']) / 1024:.2f} KB")
        lines.append("")
        
        # 找出最大的图表
        largest = max(self.manifest['figures'], key=lambda x: x['size_bytes'])
        lines.append(f"- **最大图表**: `{largest['file']}` ({largest['size_bytes'] / 1024 / 1024:.2f} MB)")
        
        # 找出最小的图表
        smallest = min(self.manifest['figures'], key=lambda x: x['size_bytes'])
        lines.append(f"- **最小图表**: `{smallest['file']}` ({smallest['size_bytes'] / 1024:.2f} KB)")
        lines.append("")
        
        # 生成任务统计
        lines.append("## 生成任务统计")
        lines.append("")
        for job in self.manifest['jobs']:
            status = "✓" if job['ok'] else "✗"
            skipped = " (跳过)" if job.get('skipped', False) else ""
            lines.append(f"- {status} **{job['job']}**{skipped}: {job['seconds']:.2f}s")
        lines.append("")
        
        # 结论
        lines.append("## 结论")
        lines.append("")
        lines.append(f"本次训练共生成 {self.manifest['plot_count']} 个图表，覆盖了训练过程的各个方面。")
        lines.append("")
        
        # 统计关键类别的图表数量
        critical_count = sum(len(self.plots_by_category.get(cat, [])) 
                           for cat, info in PLOT_CATEGORIES.items() 
                           if info['importance'] == 'critical')
        lines.append(f"- **关键图表**: {critical_count} 个")
        
        high_count = sum(len(self.plots_by_category.get(cat, [])) 
                        for cat, info in PLOT_CATEGORIES.items() 
                        if info['importance'] == 'high')
        lines.append(f"- **高重要性图表**: {high_count} 个")
        lines.append("")
        
        lines.append("建议优先查看「关键图表推荐」部分列出的图表，以快速了解训练效果。")
        lines.append("详细的性能分析请参考 [`TRAINING_DATA_ANALYSIS.md`](TRAINING_DATA_ANALYSIS.md)。")
        lines.append("")
        
        return "\n".join(lines)
    
    def run(self, output_path: str):
        """运行分析并生成报告"""
        print("=" * 60)
        print("图表分析脚本")
        print("=" * 60)
        print()
        
        # 加载 manifest
        if not self.load_manifest():
            print("✗ 无法加载 plot_manifest.json，退出")
            return False
        
        # 分类图表
        print("分类图表...")
        self.categorize_plots()
        
        # 统计各类别
        print()
        print("类别统计:")
        for category, plots in self.plots_by_category.items():
            if plots:
                print(f"  - {category}: {len(plots)} 个")
        print()
        
        # 生成报告
        print("生成分析报告...")
        report = self.generate_report()
        
        # 保存报告
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 报告已保存: {output_path}")
        print(f"  文件大小: {len(report) / 1024:.2f} KB")
        print()
        
        return True


def main():
    """主函数"""
    # 确定路径
    script_dir = Path(__file__).parent
    plots_dir = script_dir / "plots"
    output_path = script_dir / "PLOT_ANALYSIS.md"
    
    # 检查 plots 目录
    if not plots_dir.exists():
        print(f"✗ 图表目录不存在: {plots_dir}")
        return 1
    
    # 运行分析
    analyzer = PlotAnalyzer(plots_dir)
    success = analyzer.run(output_path)
    
    if success:
        print("=" * 60)
        print("分析完成！")
        print("=" * 60)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
