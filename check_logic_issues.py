#!/usr/bin/env python3
"""
检查仿真环境逻辑问题
"""

def check_logic_issues():
    issues = []
    
    with open('envs/vec_offloading_env.py', 'r') as f:
        lines = f.readlines()
    
    # 检查1: 时间推进是否在正确位置
    time_advance_found = False
    phase4_found = False
    for i, line in enumerate(lines):
        if '_phase4_serve_compute_queues' in line:
            phase4_found = True
            phase4_line = i
        if 'self.time += self.config.DT' in line:
            time_advance_found = True
            time_advance_line = i
    
    if time_advance_found and phase4_found:
        if time_advance_line < phase4_line:
            issues.append(f"⚠️  时间推进在Phase4之前（行{time_advance_line+1}），应该在Phase4之后")
    
    # 检查2: reset中是否清空了所有队列
    reset_clears_queues = False
    for i, line in enumerate(lines):
        if 'def reset(' in line:
            # 检查后100行内是否清空队列
            for j in range(i, min(i+100, len(lines))):
                if 'self.txq_v2i' in lines[j] and 'defaultdict' in lines[j]:
                    reset_clears_queues = True
                    break
            break
    
    if not reset_clears_queues:
        issues.append("⚠️  reset()中可能未正确清空FIFO队列")
    
    # 检查3: 是否有未使用的旧方法调用
    old_methods = ['ActiveTaskManager', 'add_active_task', 'step_progress']
    for method in old_methods:
        for i, line in enumerate(lines):
            if method in line and not line.strip().startswith('#'):
                issues.append(f"⚠️  发现旧方法调用: {method} (行{i+1})")
                break
    
    # 检查4: exec_locations是否正确初始化
    exec_locations_init = False
    for i, line in enumerate(lines):
        if 'v.exec_locations' in line and '= [None]' in line:
            exec_locations_init = True
            break
    
    if not exec_locations_init:
        issues.append("⚠️  exec_locations可能未正确初始化为None列表")
    
    # 检查5: 能耗账本是否正确分类
    energy_separation = {
        'E_tx_input_cost': False,
        'E_tx_edge_record': False,
        'E_cpu_local_cost': False
    }
    for line in lines:
        for key in energy_separation:
            if key in line:
                energy_separation[key] = True
    
    for key, found in energy_separation.items():
        if not found:
            issues.append(f"⚠️  能耗账本{key}未找到")
    
    # 检查6: Phase调用顺序
    phase_order = []
    in_step_method = False
    for i, line in enumerate(lines):
        if 'def step(' in line:
            in_step_method = True
        elif in_step_method and 'def ' in line and not line.strip().startswith('#'):
            break
        elif in_step_method:
            if '_phase1_commit' in line:
                phase_order.append(('phase1', i))
            elif '_phase2_activate' in line:
                phase_order.append(('phase2', i))
            elif '_phase3_serve_communication' in line:
                phase_order.append(('phase3', i))
            elif '_phase4_serve_compute' in line:
                phase_order.append(('phase4', i))
    
    expected_order = ['phase1', 'phase2', 'phase3', 'phase4']
    actual_order = [p[0] for p in phase_order]
    if actual_order != expected_order:
        issues.append(f"⚠️  Phase调用顺序错误: {actual_order} (期望: {expected_order})")
    
    # 检查7: 是否有pass # 已清理但实际应该有逻辑的地方
    suspicious_pass = []
    for i, line in enumerate(lines):
        if 'pass  # 已清理' in line:
            # 检查上下文，看是否是关键逻辑
            if i > 0:
                prev_line = lines[i-1].strip()
                if any(keyword in prev_line for keyword in ['if hard_triggered:', 'if reward', 'if illegal']):
                    suspicious_pass.append((i+1, prev_line))
    
    if suspicious_pass:
        for line_num, context in suspicious_pass[:3]:  # 只显示前3个
            issues.append(f"⚠️  可疑的pass语句（行{line_num}）: {context}")
    
    return issues

if __name__ == '__main__':
    print("=" * 60)
    print("仿真环境逻辑检查")
    print("=" * 60)
    print()
    
    issues = check_logic_issues()
    
    if not issues:
        print("✓ 未发现明显逻辑问题")
    else:
        print(f"发现 {len(issues)} 个潜在问题：")
        print()
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    print()
    print("=" * 60)

