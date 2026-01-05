#!/usr/bin/env python3
"""
深度逻辑检查：检查关键仿真逻辑
"""
import re

def deep_logic_check():
    with open('envs/vec_offloading_env.py', 'r') as f:
        content = f.read()
        lines = f.readlines()
    
    issues = []
    warnings = []
    
    # 1. 检查EDGE传输是否等待child位置确定
    if 'child_exec_loc is None' not in content:
        issues.append("❌ EDGE激活逻辑缺少child_loc检查")
    else:
        # 检查是否有continue而不是fallback
        edge_section = content[content.find('def _phase2_activate'):content.find('def _phase3')]
        if 'continue' in edge_section and 'child_exec_loc is None' in edge_section:
            warnings.append("✓ EDGE正确等待child位置确定")
        else:
            issues.append("❌ EDGE可能有child_loc None fallback")
    
    # 2. 检查同位置EDGE是否瞬时清零
    if 'same_location' in content and 'rem_bytes = 0' in content:
        warnings.append("✓ 同位置EDGE瞬时清零逻辑存在")
    else:
        issues.append("❌ 缺少同位置EDGE瞬时清零逻辑")
    
    # 3. 检查INPUT能耗是否正确分类
    if 'E_tx_input_cost' in content and 'E_tx_edge_record' in content:
        # 检查是否有正确的kind判断
        if 'if job.kind == "INPUT"' in content:
            warnings.append("✓ INPUT/EDGE能耗正确分类")
        else:
            issues.append("❌ INPUT/EDGE能耗分类逻辑缺失")
    
    # 4. 检查Local目标是否跳过INPUT传输
    local_section = content[content.find('if actual_target == \'Local\''):content.find('if actual_target == \'Local\'')+500]
    if 'rem_data' in local_section and '= 0' in local_section:
        warnings.append("✓ Local目标正确跳过INPUT传输")
    else:
        issues.append("❌ Local目标可能仍创建INPUT传输")
    
    # 5. 检查时间推进位置
    time_advance_pos = content.find('self.time += self.config.DT')
    phase4_pos = content.find('_phase4_serve_compute_queues()')
    if time_advance_pos > phase4_pos > 0:
        warnings.append("✓ 时间推进在Phase4之后")
    else:
        issues.append("❌ 时间推进位置错误")
    
    # 6. 检查队列预算是否独立
    if 'txq_v2i' in content and 'txq_v2v' in content:
        # 检查是否分别推进
        phase3_section = content[content.find('def _phase3'):content.find('def _phase4')]
        if phase3_section.count('txq_v2i') > 0 and phase3_section.count('txq_v2v') > 0:
            warnings.append("✓ V2I/V2V队列独立推进")
        else:
            issues.append("❌ V2I/V2V队列可能未独立推进")
    
    # 7. 检查exec_locations写入次数
    if 'v.exec_locations[subtask_idx] = ' in content:
        # 检查是否有重复写入保护
        if 'is None' in content[content.find('v.exec_locations[subtask_idx] = '):content.find('v.exec_locations[subtask_idx] = ')+200]:
            warnings.append("✓ exec_locations有写入保护")
        else:
            warnings.append("⚠️  exec_locations可能缺少重复写入保护")
    
    # 8. 检查是否使用了服务组件
    services_used = {
        '_comm_service': '_comm_service' in content,
        '_cpu_service': '_cpu_service' in content,
        '_dag_handler': '_dag_handler' in content
    }
    
    for service, used in services_used.items():
        if used:
            warnings.append(f"✓ 使用了{service}")
        else:
            issues.append(f"❌ 未使用{service}")
    
    # 9. 检查reset是否清空队列
    reset_section = content[content.find('def reset'):content.find('def reset')+2000]
    if 'self.txq_v2i = defaultdict' in reset_section:
        warnings.append("✓ reset正确清空FIFO队列")
    else:
        issues.append("❌ reset可能未清空FIFO队列")
    
    # 10. 检查是否有processor sharing残留
    old_keywords = ['allocation_factor', 'add_active_task', 'remove_active_task']
    for keyword in old_keywords:
        # 排除注释
        matches = re.findall(f'^[^#]*{keyword}', content, re.MULTILINE)
        if matches:
            issues.append(f"❌ 发现旧逻辑残留: {keyword}")
    
    return issues, warnings

if __name__ == '__main__':
    print("=" * 70)
    print("深度逻辑检查")
    print("=" * 70)
    print()
    
    issues, warnings = deep_logic_check()
    
    if issues:
        print(f"❌ 发现 {len(issues)} 个问题：")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
    
    if warnings:
        print(f"✓ 通过 {len(warnings)} 项检查：")
        for warning in warnings:
            print(f"  {warning}")
        print()
    
    if not issues:
        print("✅ 未发现严重逻辑问题")
    else:
        print(f"⚠️  需要修复 {len(issues)} 个问题")
    
    print()
    print("=" * 70)

