#!/usr/bin/env python3
"""
第三组：消融实验评估脚本
评估 w/o TDE, w/o CARE, TERA-MAPPO
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "runs/paper_final_results_20260327/group3_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练数据路径
ABLATION_ROOT = ROOT / "runs/rc1_ablation_1500ep_20260322_180707"
RUNS = {
    "TERA-MAPPO": ABLATION_ROOT / "full",
    "w/o TDE": ABLATION_ROOT / "wo_dag",
    "w/o CARE": ABLATION_ROOT / "wo_resource",
}


def extract_tail_metrics(training_stats_path: Path, n: int = 100) -> dict:
    """从训练数据中提取最后N个episode的平均指标"""
    if not training_stats_path.exists():
        print(f"警告: 文件不存在 {training_stats_path}")
        return {}
    
    df = pd.read_csv(training_stats_path)
    tail_df = df.tail(n)
    
    metrics = {
        "task_sr": tail_df["task_sr"].mean(),
        "task_sr_std": tail_df["task_sr"].std(),
        "mean_cft": tail_df["mean_cft_completed"].mean(),
        "mean_cft_std": tail_df["mean_cft_completed"].std(),
        "p95_cft": tail_df.get("task_duration_p95", pd.Series([0])).mean(),
        "tx_waiting": tail_df.get("t_tx_mean", pd.Series([0])).mean(),
        "comp_waiting": tail_df.get("avg_rsu_queue", pd.Series([0])).mean(),
    }
    
    return metrics


def main():
    print("=" * 60)
    print("第三组：消融实验评估")
    print("=" * 60)
    
    results = {}
    
    for method, run_path in RUNS.items():
        print(f"\n处理 {method}...")
        stats_path = run_path / "logs/training_stats.csv"
        
        if not stats_path.exists():
            print(f"  警告: 训练数据不存在")
            continue
        
        metrics = extract_tail_metrics(stats_path, n=100)
        results[method] = metrics
        print(f"  SR={metrics.get('task_sr', 0):.3f}")
    
    # 保存结果
    json_path = OUTPUT_DIR / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {json_path}")
    
    # 生成表格
    table_data = []
    for method, metrics in results.items():
        table_data.append({
            "Variant": method,
            "SR": f"{metrics.get('task_sr', 0):.3f}±{metrics.get('task_sr_std', 0):.3f}",
            "Mean CFT": f"{metrics.get('mean_cft', 0):.3f}±{metrics.get('mean_cft_std', 0):.3f}",
            "P95 CFT": f"{metrics.get('p95_cft', 0):.3f}",
        })
    
    df = pd.DataFrame(table_data)
    csv_path = OUTPUT_DIR / "tables" / "ablation_table.csv"
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"表格已保存: {csv_path}")


if __name__ == "__main__":
    main()
