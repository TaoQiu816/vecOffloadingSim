#!/usr/bin/env python3
"""
检查仿真环境与算法衔接的一致性
"""
import re

def check_interface_consistency():
    issues = []
    warnings = []
    
    with open('envs/vec_offloading_env.py', 'r') as f:
        env_content = f.read()
        env_lines = env_content.split('\n')
    
    # 1. 检查动作空间定义与解析的一致性
    print("=" * 70)
    print("1. 动作空间一致性检查")
    print("=" * 70)
    
    # 查找action_space定义
    action_space_def = None
    for i, line in enumerate(env_lines):
        if 'self.action_space' in line and 'gym.spaces' in line:
            action_space_def = line
            print(f"✓ 找到action_space定义（行{i+1}）:")
            print(f"  {line.strip()}")
            break
    
    # 查找动作解析逻辑
    action_parse_found = False
    for i, line in enumerate(env_lines):
        if 'def _plan_actions_snapshot' in line:
            action_parse_found = True
            # 检查后50行
            parse_section = '\n'.join(env_lines[i:i+50])
            if 'isinstance(act, dict)' in parse_section or 'act[0]' in parse_section:
                print(f"✓ 找到动作解析逻辑（行{i+1}）")
                # 检查是否同时支持dict和array格式
                if 'isinstance(act, dict)' in parse_section and 'act[0]' in parse_section:
                    print("  ✓ 支持dict和array两种格式")
                elif 'isinstance(act, dict)' in parse_section:
                    warnings.append("动作解析只支持dict格式，可能与MultiDiscrete不兼容")
                elif 'act[0]' in parse_section:
                    print("  ✓ 支持array格式（MultiDiscrete）")
            break
    
    if not action_parse_found:
        issues.append("❌ 未找到动作解析逻辑")
    
    print()
    
    # 2. 检查位置编码的一致性
    print("=" * 70)
    print("2. 位置编码一致性检查")
    print("=" * 70)
    
    # 查找位置编码的使用
    location_encodings = {
        "'Local'": 0,
        "('RSU',": 0,
        "isinstance(.*int)": 0
    }
    
    for line in env_lines:
        if "'Local'" in line and '=' in line:
            location_encodings["'Local'"] += 1
        if "('RSU'," in line or '("RSU",' in line:
            location_encodings["('RSU',"] += 1
        if 'isinstance' in line and 'int' in line and ('loc' in line or 'target' in line):
            location_encodings["isinstance(.*int)"] += 1
    
    print("位置编码使用统计:")
    print(f"  'Local': {location_encodings[\"'Local'\"]} 处")
    print(f"  ('RSU', id): {location_encodings[\"('RSU',\"]} 处")
    print(f"  int (vehicle_id): {location_encodings[\"isinstance(.*int)\"]} 处")
    
    if all(v > 0 for v in location_encodings.values()):
        print("  ✓ 三种位置编码都有使用")
    else:
        issues.append("❌ 某些位置编码未使用")
    
    # 检查exec_locations和task_locations的一致性
    exec_loc_usage = env_content.count('exec_locations')
    task_loc_usage = env_content.count('task_locations')
    print(f"\n位置字段使用:")
    print(f"  exec_locations: {exec_loc_usage} 处")
    print(f"  task_locations: {task_loc_usage} 处")
    
    if exec_loc_usage > 0 and task_loc_usage > 0:
        print("  ✓ 两种位置字段都有使用")
    else:
        issues.append("❌ 位置字段使用不完整")
    
    print()
    
    # 3. 检查观测空间与_get_obs的一致性
    print("=" * 70)
    print("3. 观测空间一致性检查")
    print("=" * 70)
    
    # 查找observation_space定义
    obs_space_keys = []
    in_obs_space = False
    for i, line in enumerate(env_lines):
        if 'self.observation_space = gym.spaces.Dict' in line:
            in_obs_space = True
            print(f"✓ 找到observation_space定义（行{i+1}）")
        elif in_obs_space:
            if "'" in line and ':' in line:
                key = line.split("'")[1] if "'" in line else None
                if key:
                    obs_space_keys.append(key)
            if '})' in line:
                break
    
    print(f"  定义的观测键: {obs_space_keys}")
    
    # 查找_get_obs返回的键
    obs_return_keys = []
    for i, line in enumerate(env_lines):
        if 'def _get_obs' in line:
            # 检查后500行
            obs_section = '\n'.join(env_lines[i:i+500])
            # 查找字典键
            for match in re.finditer(r"['\"](\w+)['\"]:\s*", obs_section):
                key = match.group(1)
                if key not in obs_return_keys:
                    obs_return_keys.append(key)
            break
    
    print(f"  _get_obs返回的键: {obs_return_keys[:10]}...")  # 只显示前10个
    
    # 检查关键键是否匹配
    critical_keys = ['node_x', 'self_info', 'adj', 'task_mask']
    missing_keys = [k for k in critical_keys if k not in obs_return_keys]
    if missing_keys:
        issues.append(f"❌ _get_obs缺少关键键: {missing_keys}")
    else:
        print("  ✓ 关键观测键都存在")
    
    print()
    
    # 4. 检查奖励计算的一致性
    print("=" * 70)
    print("4. 奖励计算一致性检查")
    print("=" * 70)
    
    # 查找奖励计算相关
    reward_functions = []
    for i, line in enumerate(env_lines):
        if 'def compute_absolute_reward' in line or 'def calculate_agent_reward' in line:
            reward_functions.append((i+1, line.strip()))
    
    print(f"找到 {len(reward_functions)} 个奖励函数:")
    for line_num, func in reward_functions:
        print(f"  行{line_num}: {func}")
    
    # 检查能耗是否正确使用
    energy_usage = {
        'E_tx_input_cost': env_content.count('E_tx_input_cost'),
        'E_tx_edge_record': env_content.count('E_tx_edge_record'),
        'E_cpu_local_cost': env_content.count('E_cpu_local_cost')
    }
    
    print(f"\n能耗账本使用:")
    for key, count in energy_usage.items():
        print(f"  {key}: {count} 处")
    
    if all(v > 0 for v in energy_usage.values()):
        print("  ✓ 所有能耗账本都有使用")
    else:
        issues.append("❌ 某些能耗账本未使用")
    
    print()
    
    # 5. 检查资源ID映射的一致性
    print("=" * 70)
    print("5. 资源ID映射一致性检查")
    print("=" * 70)
    
    # 查找resource_id_list的定义
    resource_id_found = False
    for i, line in enumerate(env_lines):
        if 'resource_id_list' in line and '= np.zeros' in line:
            resource_id_found = True
            print(f"✓ 找到resource_id_list定义（行{i+1}）")
            # 检查后20行的映射规则
            mapping_section = '\n'.join(env_lines[i:i+20])
            if 'resource_id_list[0] = 1' in mapping_section:
                print("  ✓ Local映射为1")
            if 'resource_id_list[1] = 2' in mapping_section:
                print("  ✓ RSU映射为2")
            if '3 + candidate_id' in mapping_section or '3 + ' in mapping_section:
                print("  ✓ V2V映射为3+vehicle_id")
            break
    
    if not resource_id_found:
        warnings.append("⚠️  未找到resource_id_list映射")
    
    print()
    
    # 6. 检查target解析与位置编码的一致性
    print("=" * 70)
    print("6. Target解析与位置编码一致性")
    print("=" * 70)
    
    # 查找target解析逻辑
    target_parse_found = False
    for i, line in enumerate(env_lines):
        if 'actual_target' in line and '=' in line:
            target_parse_found = True
            # 检查后30行
            parse_section = '\n'.join(env_lines[i:i+30])
            
            checks = {
                "Local": "'Local'" in parse_section,
                "RSU": "('RSU'," in parse_section or '("RSU",' in parse_section,
                "V2V": "int" in parse_section and "candidate" in parse_section
            }
            
            print("Target解析逻辑:")
            for target_type, found in checks.items():
                status = "✓" if found else "❌"
                print(f"  {status} {target_type}")
            
            if not all(checks.values()):
                issues.append("❌ Target解析逻辑不完整")
            break
    
    if not target_parse_found:
        issues.append("❌ 未找到target解析逻辑")
    
    print()
    
    # 7. 检查Phase方法的参数传递
    print("=" * 70)
    print("7. Phase方法参数传递检查")
    print("=" * 70)
    
    phase_methods = {
        '_phase1_commit_decisions': [],
        '_phase2_activate_edge_transfers': [],
        '_phase3_serve_communication_queues': [],
        '_phase4_serve_compute_queues': []
    }
    
    for method in phase_methods.keys():
        for i, line in enumerate(env_lines):
            if f'def {method}' in line:
                # 提取参数
                if '(' in line and ')' in line:
                    params = line[line.find('(')+1:line.find(')')].strip()
                    phase_methods[method] = params.split(',') if params else ['self']
                    print(f"✓ {method}({params})")
                break
    
    # 检查调用是否匹配
    for method, params in phase_methods.items():
        call_pattern = f'self.{method}('
        if call_pattern in env_content:
            print(f"  ✓ {method} 有调用")
        else:
            warnings.append(f"⚠️  {method} 可能未被调用")
    
    print()
    
    return issues, warnings

if __name__ == '__main__':
    print("\n")
    print("=" * 70)
    print("仿真环境与算法衔接一致性检查")
    print("=" * 70)
    print("\n")
    
    issues, warnings = check_interface_consistency()
    
    print("=" * 70)
    print("检查结果汇总")
    print("=" * 70)
    print()
    
    if issues:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
    
    if warnings:
        print(f"⚠️  {len(warnings)} 个警告:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
        print()
    
    if not issues and not warnings:
        print("✅ 所有接口一致性检查通过")
    elif not issues:
        print("✅ 未发现严重问题，仅有警告")
    else:
        print(f"⚠️  需要修复 {len(issues)} 个问题")
    
    print()

