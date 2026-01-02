import numpy as np
import torch
import gym
import sys
import os
import importlib

# --- 1. 路径环境修复 ---
# 获取当前脚本所在目录 (即 vecOffloadingSim 根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将该目录加入 Python 搜索路径，确保能找到 configs 和 envs 包
if current_dir not in sys.path:
    sys.path.append(current_dir)

print(f"当前工作目录: {os.getcwd()}")
print(f"脚本扫描路径: {current_dir}")

# --- 2. 动态配置加载器 ---
def load_config_object():
    """
    智能加载配置：
    1. 尝试从 configs.config 导入 SystemConfig 类
    2. 如果失败，尝试导入 Cfg 类
    3. 最后尝试直接返回 configs.config 模块本身
    """
    print("\n[Config Loader] 正在加载配置...")
    
    try:
        # 尝试方案 A: 导入 SystemConfig 类 (项目实际使用的类名)
        from configs.config import SystemConfig
        print("  -> 成功导入 'SystemConfig' 类作为配置对象")
        return SystemConfig
    except ImportError:
        pass
    
    try:
        # 尝试方案 B: 导入 Cfg 类 (常见别名)
        from configs.config import Cfg
        print("  -> 成功导入 'Cfg' 类作为配置对象")
        return Cfg
    except ImportError:
        pass
    
    try:
        # 尝试方案 C: 直接导入模块 (参数直接写在文件中)
        import configs.config as cfg_module
        print("  -> 成功导入 'configs.config' 模块作为配置对象")
        return cfg_module
    except ImportError as e:
        print(f"  -> [Fatal] 无法加载配置模块: {e}")
        return None

def run_system_check():
    print("\n=== 开始全系统集成测试 (System Integrity Check) ===")
    
    # [Step 1] 加载配置
    config = load_config_object()
    if config is None:
        print("❌ 无法加载配置，测试终止。请检查 configs/config.py 是否存在。")
        return

    # 检查配置是否包含我们刚才修改的关键参数 (RESOURCE_RAW_DIM)
    # getattr(obj, name, default) 可以安全地读取属性
    actual_dim = getattr(config, 'RESOURCE_RAW_DIM', 11) 
    expected_dim = 14
    print(f"    Check Config: RESOURCE_RAW_DIM = {actual_dim} (Expect {expected_dim})")

    # [Step 2] 初始化环境
    print("\n[Step 2] 初始化环境...")
    try:
        # 导入环境类
        from envs.vec_offloading_env import VecOffloadingEnv
        
        # 实例化环境
        # VecOffloadingEnv 不需要传入 config 参数（使用全局 Cfg）
        env = VecOffloadingEnv()
        
        # Reset 环境
        print("    -> Env Initialized. Resetting...")
        reset_result = env.reset()
        
        # 处理 Gym 版本差异 (Obs vs (Obs, Info))
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # [Step 3] 检查资源特征维度与物理语义
    print("\n[Step 3] 检查资源特征维度与物理语义...")
    if not obs:
        print("❌ 观测为空 (Obs is None/Empty)!")
        return

    # 环境返回的是列表格式的观测 (多智能体)
    # obs 是一个列表，每个元素是一个车辆的观测字典
    if isinstance(obs, list):
        if len(obs) == 0:
            print("❌ 观测列表为空!")
            return
        first_veh_obs = obs[0]  # 取第一个车辆的观测
        print(f"    观测格式: 列表 (长度={len(obs)})")
    elif isinstance(obs, dict):
        # 如果是字典格式 (agent_id -> obs)
        first_agent_id = list(obs.keys())[0]
        first_veh_obs = obs[first_agent_id]
        print(f"    观测格式: 字典 (智能体数={len(obs)})")
    else:
        print(f"❌ 未知的观测格式: {type(obs)}")
        return
    
    if 'resource_raw' not in first_veh_obs:
        print(f"❌ 观测中缺少 'resource_raw' 键。现有键: {list(first_veh_obs.keys())}")
        return

    res_raw = first_veh_obs['resource_raw'] # [MAX_TARGETS, RAW_DIM]
    print(f"    Feature Shape: {res_raw.shape}")
    
    # [断言 1] 特征维度
    if res_raw.shape[1] != expected_dim:
        print(f"❌ [FAIL] 特征维度为 {res_raw.shape[1]}，期望为 {expected_dim}！")
        print("    >> 请确保 configs/config.py 中的 RESOURCE_RAW_DIM 已修改为 14")
        print("    >> 请确保 models/resource_features.py 中的 ResourceFeatureBuilder 已更新")
    else:
        print(f"✅ [PASS] 特征维度正确 ({expected_dim})。")
        
    # [断言 2] Local Rate 归零 (Index 3)
    # 注意：这里假设 ResourceBuilder 的顺序没有变，Rate 是第 4 个元素
    local_feat = res_raw[0]
    rate_val = local_feat[3]
    print(f"    Local Node Raw (Partial): Rate={rate_val:.4f}")
    
    if rate_val == 0.0:
         print("✅ [PASS] Local Rate 正确归零 (无传输)。")
    else:
         print(f"❌ [FAIL] Local Rate 为 {rate_val}，期望为 0.0。")
         
    # [断言 3] 时间特征 (最后3维)
    time_feats = local_feat[-3:]
    print(f"    Time Features (Exec, Comm, Wait): {time_feats}")
    if np.any(time_feats > 0):
        print("✅ [PASS] 时间特征已注入且数值合理 (非全0)。")
    else:
        print("⚠️ [WARN] 时间特征全为 0。如果是刚开始且任务极小，可能是正常的；否则需检查计算逻辑。")

    # [Step 4] 检查 Mask 语义
    print("\n[Step 4] 检查 Action Mask...")
    if 'action_mask' not in first_veh_obs:
        print("❌ 观测中缺少 'action_mask'。")
        return

    mask = first_veh_obs['action_mask']
    print(f"    Mask Shape: {mask.shape}")
    
    local_available = mask[0]
    total_available = np.sum(mask)
    
    print(f"    Local Action Available: {local_available}")
    print(f"    Total Actions Available: {total_available}")
    
    if local_available:
        print("✅ [PASS] Local Action (Index 0) 可选 (保底机制生效)。")
    else:
        print("❌ [FAIL] Local Action 不可选！死胡同保底逻辑可能失效。")
        
    if total_available == 0:
        print("❌ [FAIL] Mask 全为 False！Agent 无路可走。")
    
    # [Step 5] 试运行 Step
    print("\n[Step 5] 试运行 Step (全 Local 策略)...")
    
    # 根据观测格式构造动作
    if isinstance(obs, list):
        # 列表格式：每个车辆一个动作字典
        actions = [{'target': 0, 'power': 1.0} for _ in obs]
        print(f"    动作格式: 列表 (长度={len(actions)})")
    elif isinstance(obs, dict):
        # 字典格式：agent_id -> action
        actions = {agent_id: {'target': 0, 'power': 1.0} for agent_id in obs.keys()}
        print(f"    动作格式: 字典 (智能体数={len(actions)})")
    
    try:
        step_result = env.step(actions)
        # 兼容 Gym 返回值解包
        if len(step_result) == 4:
            next_obs, rewards, dones, infos = step_result
        elif len(step_result) == 5:
            next_obs, rewards, terms, truncs, infos = step_result
            
        print("✅ [PASS] Step 运行成功，无崩溃。")
        
        # 处理奖励格式
        if isinstance(rewards, list):
            if len(rewards) > 0:
                sample_reward = rewards[0]
                print(f"    Rewards Sample: {sample_reward:.4f}")
            else:
                print("⚠️ [WARN] Rewards 列表为空。")
        elif isinstance(rewards, dict):
            if rewards:
                sample_reward = list(rewards.values())[0]
                print(f"    Rewards Sample: {sample_reward:.4f}")
            else:
                print("⚠️ [WARN] Rewards 字典为空。")
        else:
            print(f"⚠️ [WARN] 未知的 Rewards 格式: {type(rewards)}")
            
    except Exception as e:
        print(f"❌ [FAIL] Step 运行崩溃: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 测试结束 ===")

if __name__ == "__main__":
    run_system_check()