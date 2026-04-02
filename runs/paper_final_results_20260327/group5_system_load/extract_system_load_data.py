#!/usr/bin/env python3
"""
Group 5: 系统负载实验数据提取
提取不同车辆规模和RSU算力下的SR和Mean CFT数据
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import json
import numpy as np


def extract_rl_metrics(metrics_csv_path):
    """从RL训练的metrics.csv提取最后50个episode的平均指标"""
    df = pd.read_csv(metrics_csv_path)
    # 取最后50个episode
    df_tail = df.tail(50)

    return {
        'mean_cft': df_tail['mean_cft_completed'].mean(),
        'success_rate': df_tail['task_success_rate'].mean(),
    }


def extract_baseline_metrics(baseline_csv_path):
    """从baseline的baseline_eval_core_summary.csv提取指标"""
    df = pd.read_csv(baseline_csv_path)

    return {
        'mean_cft': df['mean_cft_completed_mean'].iloc[0],
        'success_rate': df['task_success_rate_mean'].iloc[0],
    }


def main():
    base_dir = Path("runs")
    output_dir = Path("runs/paper_final_results_20260327/group5_system_load")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. 车辆规模实验 ==========
    vehicle_batch_dirs = [
        base_dir / "rc1_batch2_vehicle_20260324_181254",
        base_dir / "rc1_batch2_vehicle_fmappo_20260328_224844"
    ]

    vehicle_configs = ['vehicle_10', 'vehicle_20', 'vehicle_30']
    vehicle_numbers = [10, 20, 30]

    # 算法映射
    algo_map = {
        'mappo_full': 'MAPPO',
        'fmappo_flat': 'F-MAPPO',
        'ippo': 'IPPO',
        'greedy': 'NRO',
        'eft': 'EFT',
        'cp_eft': 'CP-EFT',
        'local_only': 'LO'
    }

    vehicle_data = []

    for veh_config, veh_num in zip(vehicle_configs, vehicle_numbers):
        for batch_dir in vehicle_batch_dirs:
            config_dir = batch_dir / veh_config
            if not config_dir.exists():
                continue

            for algo_dir_name, algo_label in algo_map.items():
                algo_dir = config_dir / algo_dir_name
                if not algo_dir.exists():
                    continue

                # RL算法
                if algo_dir_name in ['mappo_full', 'fmappo_flat', 'ippo']:
                    metrics_path = algo_dir / "logs" / "metrics.csv"
                    if metrics_path.exists():
                        metrics = extract_rl_metrics(metrics_path)
                        vehicle_data.append({
                            'num_vehicles': veh_num,
                            'algorithm': algo_label,
                            'mean_cft': metrics['mean_cft'],
                            'success_rate': metrics['success_rate']
                        })
                        print(f"✓ 提取 {veh_config}/{algo_dir_name}: SR={metrics['success_rate']:.3f}, CFT={metrics['mean_cft']:.3f}")

                # Baseline算法
                else:
                    baseline_path = algo_dir / "logs" / "baseline_eval_core_summary.csv"
                    if baseline_path.exists():
                        metrics = extract_baseline_metrics(baseline_path)
                        vehicle_data.append({
                            'num_vehicles': veh_num,
                            'algorithm': algo_label,
                            'mean_cft': metrics['mean_cft'],
                            'success_rate': metrics['success_rate']
                        })
                        print(f"✓ 提取 {veh_config}/{algo_dir_name}: SR={metrics['success_rate']:.3f}, CFT={metrics['mean_cft']:.3f}")

    # 保存车辆规模数据
    df_vehicle = pd.DataFrame(vehicle_data)
    df_vehicle = df_vehicle.sort_values(['num_vehicles', 'algorithm'])
    vehicle_csv_path = output_dir / "vehicle_scale_data.csv"
    df_vehicle.to_csv(vehicle_csv_path, index=False)
    print(f"\n✓ 车辆规模数据已保存: {vehicle_csv_path}")
    print(f"  共 {len(df_vehicle)} 条记录")

    # ========== 2. RSU算力实验 ==========
    rsu_batch_dirs = [
        base_dir / "rc1_batch3_frsu_20260325_163701",
        base_dir / "rc1_batch3_frsu_fmappo_20260328_224844"
    ]

    rsu_configs = ['frsu_4', 'frsu_6', 'frsu_8']
    rsu_factors = [4, 6, 8]

    rsu_data = []

    for rsu_config, rsu_factor in zip(rsu_configs, rsu_factors):
        for batch_dir in rsu_batch_dirs:
            config_dir = batch_dir / rsu_config
            if not config_dir.exists():
                continue

            for algo_dir_name, algo_label in algo_map.items():
                algo_dir = config_dir / algo_dir_name
                if not algo_dir.exists():
                    continue

                # RL算法
                if algo_dir_name in ['mappo_full', 'fmappo_flat', 'ippo']:
                    metrics_path = algo_dir / "logs" / "metrics.csv"
                    if metrics_path.exists():
                        metrics = extract_rl_metrics(metrics_path)
                        rsu_data.append({
                            'rsu_cpu_factor': rsu_factor,
                            'algorithm': algo_label,
                            'mean_cft': metrics['mean_cft'],
                            'success_rate': metrics['success_rate']
                        })
                        print(f"✓ 提取 {rsu_config}/{algo_dir_name}: SR={metrics['success_rate']:.3f}, CFT={metrics['mean_cft']:.3f}")

                # Baseline算法
                else:
                    baseline_path = algo_dir / "logs" / "baseline_eval_core_summary.csv"
                    if baseline_path.exists():
                        metrics = extract_baseline_metrics(baseline_path)
                        rsu_data.append({
                            'rsu_cpu_factor': rsu_factor,
                            'algorithm': algo_label,
                            'mean_cft': metrics['mean_cft'],
                            'success_rate': metrics['success_rate']
                        })
                        print(f"✓ 提取 {rsu_config}/{algo_dir_name}: SR={metrics['success_rate']:.3f}, CFT={metrics['mean_cft']:.3f}")

    # 保存RSU算力数据
    df_rsu = pd.DataFrame(rsu_data)
    df_rsu = df_rsu.sort_values(['rsu_cpu_factor', 'algorithm'])
    rsu_csv_path = output_dir / "rsu_cpu_data.csv"
    df_rsu.to_csv(rsu_csv_path, index=False)
    print(f"\n✓ RSU算力数据已保存: {rsu_csv_path}")
    print(f"  共 {len(df_rsu)} 条记录")

    # 生成统计摘要
    summary = {
        'vehicle_scale': {
            'num_records': len(df_vehicle),
            'algorithms': df_vehicle['algorithm'].unique().tolist(),
            'vehicle_numbers': df_vehicle['num_vehicles'].unique().tolist()
        },
        'rsu_cpu': {
            'num_records': len(df_rsu),
            'algorithms': df_rsu['algorithm'].unique().tolist(),
            'rsu_factors': df_rsu['rsu_cpu_factor'].unique().tolist()
        }
    }

    summary_path = output_dir / "data_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 数据摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
