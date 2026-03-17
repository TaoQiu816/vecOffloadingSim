#!/usr/bin/env python3
"""
主线训练分析脚本
只关注当前主线有效指标，生成健康度面���报告
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# 定义当前主线有效指标
MAINLINE_METRICS = {
    "成功率指标": [
        "task_success_rate",
        "subtask_success_rate", 
        "deadline_miss_rate",
        "time_limit_rate"
    ],
    "决策分布指标": [
        "decision_frac_local",
        "decision_frac_rsu",
        "decision_frac_v2v"
    ],
    "训练指标": [
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "grad_norm_preclip",
        "grad_norm_postclip"
    ],
    "Oracle对比指标": [
        "oracle_match_rate",
        "action_regret_mean",
        "action_regret_p95"
    ],
    "物理指标": [
        "mean_cft_est",
        "energy_norm_mean",
        "power_ratio_mean",
        "I_caused_mean",
        "risk_penalty_mean"
    ]
}

# 遗留指标（不再使用但可能存在于日志中）
LEGACY_METRICS = [
    "policy_loss",
    "value_loss",
    "grad_norm",  # 已被 grad_norm_preclip/postclip 替代
    "mode_aux_loss",
    "mode_aux_acc"
]


def load_data(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载训练日志数据"""
    metrics_path = run_dir / "logs" / "metrics.csv"
    training_stats_path = run_dir / "logs" / "training_stats.csv"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到 metrics.csv: {metrics_path}")
    if not training_stats_path.exists():
        raise FileNotFoundError(f"未找到 training_stats.csv: {training_stats_path}")
    
    metrics_df = pd.read_csv(metrics_path)
    training_stats_df = pd.read_csv(training_stats_path)
    
    return metrics_df, training_stats_df


def compute_statistics(df: pd.DataFrame, metric: str) -> Dict:
    """计算指标的统计摘要"""
    if metric not in df.columns:
        return {
            "available": False,
            "reason": "指标不存在于数据中"
        }
    
    data = df[metric].dropna()
    if len(data) == 0:
        return {
            "available": False,
            "reason": "指标数据全为空"
        }
    
    # 计算趋势（线性回归斜率）
    x = np.arange(len(data))
    if len(data) > 1:
        slope = np.polyfit(x, data, 1)[0]
        trend = "上升" if slope > 0.001 else ("下降" if slope < -0.001 else "稳定")
    else:
        slope = 0
        trend = "数据不足"
    
    return {
        "available": True,
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "median": float(data.median()),
        "trend": trend,
        "slope": float(slope),
        "count": len(data)
    }


def check_convergence(df: pd.DataFrame, metric: str, window: int = 50) -> Dict:
    """检查指标收敛性"""
    if metric not in df.columns:
        return {"converged": False, "reason": "指标不存在"}
    
    data = df[metric].dropna()
    if len(data) < window * 2:
        return {"converged": False, "reason": "数据不足"}
    
    # 比较最后 window 个数据点与前 window 个数据点的方差
    tail = data.iloc[-window:]
    head = data.iloc[:window]
    
    tail_std = tail.std()
    head_std = head.std()
    
    # 如果尾部标准差显著小于头部，认为已收敛
    converged = tail_std < head_std * 0.5
    
    return {
        "converged": converged,
        "tail_std": float(tail_std),
        "head_std": float(head_std),
        "tail_mean": float(tail.mean()),
        "head_mean": float(head.mean())
    }


def detect_anomalies(df: pd.DataFrame, metric: str, threshold: float = 3.0) -> Dict:
    """检测异常值（基于 Z-score）"""
    if metric not in df.columns:
        return {"has_anomalies": False, "reason": "指标不存在"}
    
    data = df[metric].dropna()
    if len(data) < 10:
        return {"has_anomalies": False, "reason": "数据不足"}
    
    mean = data.mean()
    std = data.std()
    
    if std < 1e-8:
        return {"has_anomalies": False, "reason": "标准差过小"}
    
    z_scores = np.abs((data - mean) / std)
    anomalies = z_scores > threshold
    
    return {
        "has_anomalies": anomalies.any(),
        "anomaly_count": int(anomalies.sum()),
        "anomaly_rate": float(anomalies.mean()),
        "max_z_score": float(z_scores.max())
    }


def generate_report(run_dir: Path, metrics_df: pd.DataFrame, training_stats_df: pd.DataFrame) -> str:
    """生成 Markdown 报告"""
    report = []
    report.append("# 主线训练健康度面板\n")
    report.append(f"**运行目录**: `{run_dir.name}`\n")
    report.append(f"**总 Episode 数**: {len(metrics_df)}\n")
    report.append(f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("\n---\n")
    
    # 对每个指标类别生成报告
    for category, metrics in MAINLINE_METRICS.items():
        report.append(f"\n## {category}\n")
        
        for metric in metrics:
            report.append(f"\n### {metric}\n")
            
            # 优先从 metrics.csv 读取，如果不存在则从 training_stats.csv 读取
            if metric in metrics_df.columns:
                df = metrics_df
                source = "metrics.csv"
            elif metric in training_stats_df.columns:
                df = training_stats_df
                source = "training_stats.csv"
            else:
                report.append(f"⚠️ **指标不可用**（未在日志中找到）\n")
                continue
            
            # 统计摘要
            stats = compute_statistics(df, metric)
            if not stats["available"]:
                report.append(f"⚠️ **指标不可用**: {stats['reason']}\n")
                continue
            
            report.append(f"- **数据来源**: {source}\n")
            report.append(f"- **均值**: {stats['mean']:.4f}\n")
            report.append(f"- **标准差**: {stats['std']:.4f}\n")
            report.append(f"- **中位数**: {stats['median']:.4f}\n")
            report.append(f"- **范围**: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            report.append(f"- **趋势**: {stats['trend']} (斜率: {stats['slope']:.6f})\n")
            
            # 收敛性分析
            conv = check_convergence(df, metric)
            if conv["converged"]:
                report.append(f"- **收敛状态**: ✅ 已收敛\n")
                report.append(f"  - 尾部均值: {conv['tail_mean']:.4f}, 标准差: {conv['tail_std']:.4f}\n")
            else:
                report.append(f"- **收敛状态**: ⏳ 未收敛 ({conv.get('reason', '标准差未显著下降')})\n")
            
            # 异常检测
            anom = detect_anomalies(df, metric)
            if anom["has_anomalies"]:
                report.append(f"- **异常检测**: ⚠️ 发现 {anom['anomaly_count']} 个异常值 ({anom['anomaly_rate']*100:.1f}%)\n")
            else:
                report.append(f"- **异常检测**: ✅ 无显著异常\n")
    
    # 遗留指标说明
    report.append("\n---\n")
    report.append("\n## 遗留指标说明\n")
    report.append("\n以下指标已不再作为主线指标使用，但可能仍存在于日志中：\n")
    for metric in LEGACY_METRICS:
        if metric in metrics_df.columns or metric in training_stats_df.columns:
            report.append(f"- `{metric}`: 存在于日志中（遗留）\n")
        else:
            report.append(f"- `{metric}`: 不存在于日志中\n")
    
    # 特殊说明：grad_norm 的演进
    report.append("\n### grad_norm 指标演进\n")
    report.append("- `grad_norm`: 旧版本指标，已被 `grad_norm_preclip` 和 `grad_norm_postclip` 替代\n")
    report.append("- `grad_norm_preclip`: 梯度裁剪前的梯度范数（新增）\n")
    report.append("- `grad_norm_postclip`: 梯度裁剪后的梯度范数（新增）\n")
    report.append("- 预期行为：`grad_norm_postclip` 应 <= MAX_GRAD_NORM (1.0)\n")
    
    # 检查 grad_norm_postclip 是否符合预期
    if "grad_norm_postclip" in training_stats_df.columns:
        postclip_data = training_stats_df["grad_norm_postclip"].dropna()
        if len(postclip_data) > 0:
            max_postclip = postclip_data.max()
            if max_postclip > 1.0:
                report.append(f"\n⚠️ **警告**: grad_norm_postclip 最大值 ({max_postclip:.4f}) 超过 MAX_GRAD_NORM (1.0)\n")
            else:
                report.append(f"\n✅ grad_norm_postclip 最大值 ({max_postclip:.4f}) 符合预期 (<= 1.0)\n")
    
    report.append("\n---\n")
    report.append("\n## 总结\n")
    report.append("\n本报告仅包含当前主线有效指标的分析。\n")
    report.append("遗留指标已被明确标注，不参与健康度评估。\n")
    
    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description="主线训练分析脚本")
    parser.add_argument("run_dir", type=str, help="运行目录路径")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"错误: 运行目录不存在: {run_dir}")
        return 1
    
    print(f"正在分析运行目录: {run_dir}")
    
    # 加载数据
    try:
        metrics_df, training_stats_df = load_data(run_dir)
        print(f"已加载 metrics.csv: {len(metrics_df)} 行")
        print(f"已加载 training_stats.csv: {len(training_stats_df)} 行")
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        return 1
    
    # 生成报告
    try:
        report = generate_report(run_dir, metrics_df, training_stats_df)
    except Exception as e:
        print(f"错误: 生成报告失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 保存报告
    output_path = run_dir / "MAINLINE_ANALYSIS.md"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ 报告已保存: {output_path}")
    except Exception as e:
        print(f"错误: 保存报告失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
