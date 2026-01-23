"""
三组对比实验配置文件
Experimental Configuration for Ablation Study

使用方法：
python train.py --config A  # 基准延长训练
python train.py --config B  # 高压力场景
python train.py --config C  # 大规模车辆场景

或通过环境变量：
EXP_CONFIG=A python train.py
"""

# =============================================================================
# 方案A：基准延长训练 (Baseline Extended)
# =============================================================================
CONFIG_A = {
    "name": "baseline_extended",
    "description": "基准配置延长至1500ep，验证收敛极限",
    
    # 训练参数
    "MAX_EPISODES": 1500,
    "LR_DECAY_STEPS": 200,
    "LR_DECAY_RATE": 0.95,
    "SAVE_INTERVAL": 200,
    
    # 其他参数保持默认
}

# =============================================================================
# 方案B：高压力场景 (Tight Deadline + Dense Task)
# =============================================================================
CONFIG_B = {
    "name": "high_pressure",
    "description": "紧迫deadline+大规模任务，测试算法极限",
    
    # 训练参数
    "MAX_EPISODES": 1500,
    "LR_DECAY_STEPS": 200,
    "LR_DECAY_RATE": 0.95,
    "SAVE_INTERVAL": 200,
    
    # Deadline压力增加
    "DEADLINE_TIGHTENING_MIN": 0.65,
    "DEADLINE_TIGHTENING_MAX": 0.85,
    
    # 任务规模增加
    "MIN_NODES": 22,
    "MAX_NODES": 30,
    "MAX_COMP": 4.0e9,
    "NORM_MAX_COMP": 5.0e9,  # 配合MAX_COMP调整
    
    # 队列限制适度放宽
    "RSU_QUEUE_CYCLES_LIMIT": 180.0e9,
}

# =============================================================================
# 方案C：大规模车辆场景 (Scalability Test)
# =============================================================================
CONFIG_C = {
    "name": "large_scale",
    "description": "50车辆+4RSU，测试算法可扩展性",
    
    # 训练参数
    "MAX_EPISODES": 1500,
    "LR_DECAY_STEPS": 200,
    "LR_DECAY_RATE": 0.95,
    "SAVE_INTERVAL": 200,
    "ENTROPY_COEF": 0.003,  # 更多探索
    
    # 场景规模增加
    "NUM_VEHICLES": 50,
    "V2V_TOP_K": 7,
    "MAP_SIZE": 1500.0,
    "NUM_RSU": 4,
    "VEHICLE_SPAWN_X_MAX": 0.85,
    
    # 队列限制适度放宽
    "RSU_QUEUE_CYCLES_LIMIT": 200.0e9,
    "VEHICLE_QUEUE_CYCLES_LIMIT": 15.0e9,
}

# =============================================================================
# 配置选择器
# =============================================================================
CONFIGS = {
    "A": CONFIG_A,
    "B": CONFIG_B,
    "C": CONFIG_C,
}

def get_config(config_name):
    """获取指定配置"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]

def print_config_diff(config_name):
    """打印配置与默认值的差异"""
    config = get_config(config_name)
    print(f"\n{'='*60}")
    print(f"实验配置: {config['name']}")
    print(f"说明: {config['description']}")
    print(f"{'='*60}")
    for key, value in config.items():
        if key not in ['name', 'description']:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")
