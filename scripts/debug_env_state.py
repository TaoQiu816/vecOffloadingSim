#!/usr/bin/env python
"""
快速调试脚本：检查环境初始状态

检查：
- RSU位置和数量
- 车辆初始位置
- 距离关系
- 队列状态
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import SystemConfig as Cfg
from envs.vec_offloading_env import VecOffloadingEnv

def main():
    print("="*80)
    print("环境初始状态检查")
    print("="*80)
    
    # 创建环境
    np.random.seed(42)
    env = VecOffloadingEnv()
    obs, info = env.reset()
    
    # 检查RSU
    print(f"\n【RSU配置】")
    print(f"RSU数量: {len(env.rsus)}")
    print(f"RSU_RANGE: {Cfg.RSU_RANGE}m")
    print(f"NUM_RSU: {Cfg.NUM_RSU}")
    print(f"MAP_SIZE: {Cfg.MAP_SIZE}m")
    
    for i, rsu in enumerate(env.rsus):
        print(f"\nRSU {i}:")
        print(f"  位置: {rsu.position}")
        print(f"  CPU频率: {rsu.cpu_freq/1e9:.1f} GHz")
        print(f"  核心数: {rsu.num_processors}")
        print(f"  覆盖范围: {rsu.coverage_range}m")
        
        # 检查队列状态
        is_full = rsu.is_queue_full(new_task_cycles=1e9)
        print(f"  队列是否满（1Gcycles测试）: {is_full}")
    
    # 检查车辆
    print(f"\n【车辆配置】")
    print(f"车辆数量: {len(env.vehicles)}")
    print(f"V2V_RANGE: {Cfg.V2V_RANGE}m")
    
    for i, v in enumerate(env.vehicles[:3]):  # 只打印前3辆
        print(f"\n车辆 {v.id}:")
        print(f"  位置: {v.pos}")
        print(f"  速度: {v.vel}")
        print(f"  CPU频率: {v.cpu_freq/1e9:.1f} GHz")
        
        # 检查到RSU的距离
        for j, rsu in enumerate(env.rsus):
            dist = np.linalg.norm(v.pos - rsu.position)
            in_coverage = dist <= rsu.coverage_range
            print(f"  到RSU{j}距离: {dist:.1f}m (覆盖={in_coverage})")
        
        # 检查到其他车辆的距离
        v2v_neighbors = 0
        for other in env.vehicles:
            if v.id == other.id:
                continue
            dist = np.linalg.norm(v.pos - other.pos)
            if dist <= Cfg.V2V_RANGE:
                v2v_neighbors += 1
        print(f"  V2V邻居数（距离<{Cfg.V2V_RANGE}m）: {v2v_neighbors}")
    
    # 调用_select_best_rsu检查
    print(f"\n【RSU选择测试】")
    for i, v in enumerate(env.vehicles[:3]):
        task_comp = 1.5e9  # 1.5 Gcycles
        task_data = 2e6    # 2 Mbits
        rsu_id, rsu_rate, rsu_wait, rsu_dist, rsu_contact = env._select_best_rsu(v, task_comp, task_data)
        
        print(f"\n车辆{v.id}:")
        print(f"  选中RSU: {rsu_id}")
        if rsu_id is not None:
            print(f"  速率: {rsu_rate/1e6:.2f} Mbps")
            print(f"  等待时间: {rsu_wait:.3f}s")
            print(f"  距离: {rsu_dist:.1f}m")
            print(f"  接触时间: {rsu_contact:.3f}s")
        else:
            print(f"  原因: 无可用RSU")
            # 详细检查原因
            for j, rsu in enumerate(env.rsus):
                dist = rsu.get_distance(v.pos)
                in_coverage = rsu.is_in_coverage(v.pos)
                is_full = rsu.is_queue_full(new_task_cycles=task_comp)
                print(f"    RSU{j}: dist={dist:.1f}m, in_coverage={in_coverage}, queue_full={is_full}")

if __name__ == "__main__":
    main()

