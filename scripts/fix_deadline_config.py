#!/usr/bin/env python
"""
快速修复脚本：根据审计报告放宽Deadline配置

审计发现：当前Deadline过紧（1.0-1.2倍本地时间）导致成功率0%
建议修复：放宽至1.5-2.0倍本地时间

此脚本会：
1. 修改configs/config.py中的DEADLINE_TIGHTENING参数
2. 验证修改后的配置
3. 运行5个episode测试成功率
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def update_config():
    """更新config.py中的deadline参数"""
    config_path = 'configs/config.py'
    
    print("="*80)
    print("修复Deadline配置")
    print("="*80)
    
    # 读取配置文件
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 备份原配置
    original_min = re.search(r'DEADLINE_TIGHTENING_MIN\s*=\s*([0-9.]+)', content)
    original_max = re.search(r'DEADLINE_TIGHTENING_MAX\s*=\s*([0-9.]+)', content)
    
    if original_min and original_max:
        print(f"\n当前配置:")
        print(f"  DEADLINE_TIGHTENING_MIN = {original_min.group(1)}")
        print(f"  DEADLINE_TIGHTENING_MAX = {original_max.group(1)}")
    
    # 修改参数
    content = re.sub(
        r'(DEADLINE_TIGHTENING_MIN\s*=\s*)([0-9.]+)',
        r'\g<1>1.5',
        content
    )
    content = re.sub(
        r'(DEADLINE_TIGHTENING_MAX\s*=\s*)([0-9.]+)',
        r'\g<1>2.0',
        content
    )
    
    # 写回文件
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"\n✅ 修改后配置:")
    print(f"  DEADLINE_TIGHTENING_MIN = 1.5")
    print(f"  DEADLINE_TIGHTENING_MAX = 2.0")
    print(f"\n原因：审计显示成功率0%，deadline过紧导致任务无法完成")
    print(f"预期：放宽后成功率应提升至>10%")

def verify_and_test():
    """验证配置并快速测试"""
    print(f"\n" + "="*80)
    print("快速验证测试（5 episodes）")
    print("="*80)
    
    from configs.config import SystemConfig as Cfg
    print(f"\n加载配置成功:")
    print(f"  DEADLINE_TIGHTENING_MIN = {Cfg.DEADLINE_TIGHTENING_MIN}")
    print(f"  DEADLINE_TIGHTENING_MAX = {Cfg.DEADLINE_TIGHTENING_MAX}")
    
    # 运行简单测试
    from envs.vec_offloading_env import VecOffloadingEnv
    import numpy as np
    
    env = VecOffloadingEnv()
    np.random.seed(42)
    
    success_count = 0
    total_eps = 5
    
    for ep in range(total_eps):
        obs, info = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Greedy策略
            actions = []
            for v in env.vehicles:
                if v.task_dag.get_top_priority_task() is None:
                    actions.append({"target": 0, "power": 1.0})
                    continue
                
                rsu_id = env._last_rsu_choice.get(v.id)
                target = 1 if rsu_id is not None else 0
                actions.append({"target": target, "power": 1.0})
            
            obs, rewards, done, truncated, info = env.step(actions)
        
        # 统计成功率
        task_success_rate = info.get('task_success_rate', 0.0)
        if task_success_rate > 0:
            success_count += 1
        
        print(f"  Episode {ep+1}: 成功率={task_success_rate:.1%}")
    
    print(f"\n测试结果:")
    print(f"  有成功任务的episode: {success_count}/{total_eps}")
    
    if success_count > 0:
        print(f"  ✅ 配置修复有效！成功率已提升")
    else:
        print(f"  ⚠️  成功率仍为0，可能需要进一步放宽deadline")

if __name__ == "__main__":
    update_config()
    verify_and_test()
    
    print(f"\n" + "="*80)
    print("建议下一步:")
    print("  1. 运行完整训练: python train.py --max-episodes 100")
    print("  2. 运行完整审计: python scripts/debug_rollout_audit.py --episodes 20")
    print("  3. 检查策略是否出现V2V崩溃")
    print("="*80)

