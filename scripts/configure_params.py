#!/usr/bin/env python3
"""
交互式参数配置工具

用法：
    python configure_params.py                    # 交互式配置
    python configure_params.py --show             # 显示当前配置
    python configure_params.py --preset quick     # 使用预设
"""

import argparse
from configs.config import SystemConfig
from configs.train_config import TrainConfig

# ============================================================================
# 预设配置方案
# ============================================================================

PRESETS = {
    "quick": {
        "name": "快速验证（300 episodes，小网络）",
        "SystemConfig": {
            "NUM_VEHICLES": 8,
            "MIN_NODES": 4,
            "MAX_NODES": 8,
        },
        "TrainConfig": {
            "MAX_EPISODES": 300,
            "EMBED_DIM": 64,
            "NUM_LAYERS": 2,
            "MINI_BATCH_SIZE": 128,
        }
    },
    "standard": {
        "name": "标准训练（当前默认配置）",
        "SystemConfig": {},
        "TrainConfig": {}
    },
    "long": {
        "name": "长期训练（5000 episodes）",
        "SystemConfig": {},
        "TrainConfig": {
            "MAX_EPISODES": 5000,
            "LR_DECAY_STEPS": 200,
            "BIAS_DECAY_EVERY_EP": 200,
        }
    },
    "high_capacity": {
        "name": "高容量网络（增强表达能力）",
        "SystemConfig": {},
        "TrainConfig": {
            "EMBED_DIM": 256,
            "NUM_LAYERS": 4,
            "D_FF": 1024,
            "NUM_HEADS": 8,
        }
    },
    "easy_task": {
        "name": "简化任务（降低难度）",
        "SystemConfig": {
            "MIN_NODES": 4,
            "MAX_NODES": 6,
            "MAX_COMP": 1.0e8,
            "DEADLINE_TIGHTENING_MIN": 2.0,
            "DEADLINE_TIGHTENING_MAX": 3.5,
        },
        "TrainConfig": {}
    },
}

# ============================================================================
# 工具函数
# ============================================================================

def show_current_config():
    """显示当前关键配置"""
    print("\n" + "="*70)
    print("当前系统配置（关键参数）")
    print("="*70)

    print("\n【场景参数】")
    print(f"  车辆数:              {SystemConfig.NUM_VEHICLES}")
    print(f"  DAG 节点范围:        [{SystemConfig.MIN_NODES}, {SystemConfig.MAX_NODES}]")
    print(f"  计算量范围:          [{SystemConfig.MIN_COMP/1e6:.0f}M, {SystemConfig.MAX_COMP/1e6:.0f}M] cycles")
    print(f"  V2I 带宽:            {SystemConfig.BW_V2I/1e6:.0f} MHz")
    print(f"  RSU CPU:             {SystemConfig.F_RSU/1e9:.0f} GHz")
    print(f"  Deadline 松紧:       [{SystemConfig.DEADLINE_TIGHTENING_MIN}, {SystemConfig.DEADLINE_TIGHTENING_MAX}]")

    print("\n【训练参数】")
    print(f"  最大 Episodes:       {TrainConfig.MAX_EPISODES}")
    print(f"  网络维度:            {TrainConfig.EMBED_DIM}")
    print(f"  Transformer 层数:    {TrainConfig.NUM_LAYERS}")
    print(f"  学习率（Actor):      {TrainConfig.LR_ACTOR}")
    print(f"  学习率（Critic):     {TrainConfig.LR_CRITIC}")
    print(f"  Logit Bias (RSU):    {TrainConfig.LOGIT_BIAS_RSU}")
    print(f"  Logit Bias (Local):  {TrainConfig.LOGIT_BIAS_LOCAL}")

    print("\n【奖励函数】")
    print(f"  方案:                {SystemConfig.REWARD_SCHEME}")
    print(f"  PBRS Alpha:          {SystemConfig.REWARD_ALPHA}")
    print(f"  PBRS Beta:           {SystemConfig.REWARD_BETA}")
    print(f"  成功奖励:            {SystemConfig.SUCCESS_BONUS}")
    print(f"  失败惩罚:            {SystemConfig.PENALTY_FAILURE}")
    print(f"  Stage 1 修复:        ✅ 已启用")

    print("\n" + "="*70)


def show_presets():
    """显示所有预设方案"""
    print("\n" + "="*70)
    print("可用预设方案")
    print("="*70)

    for key, preset in PRESETS.items():
        print(f"\n[{key}] {preset['name']}")
        if preset['SystemConfig']:
            print("  场景修改:")
            for k, v in preset['SystemConfig'].items():
                print(f"    - {k} = {v}")
        if preset['TrainConfig']:
            print("  训练修改:")
            for k, v in preset['TrainConfig'].items():
                print(f"    - {k} = {v}")

    print("\n" + "="*70)


def apply_preset(preset_name):
    """应用预设配置（生成配置代码）"""
    if preset_name not in PRESETS:
        print(f"❌ 错误：未找到预设 '{preset_name}'")
        print(f"可用预设: {', '.join(PRESETS.keys())}")
        return

    preset = PRESETS[preset_name]
    print(f"\n应用预设: {preset['name']}")

    # 生成配置代码
    print("\n将以下代码添加到训练脚本或配置文件中：\n")
    print("="*70)
    print("# " + preset['name'])
    print("="*70)

    if preset['SystemConfig']:
        print("\n# 场景参数修改")
        for k, v in preset['SystemConfig'].items():
            if isinstance(v, str):
                print(f"env.config.{k} = '{v}'")
            else:
                print(f"env.config.{k} = {v}")

    if preset['TrainConfig']:
        print("\n# 训练参数修改")
        for k, v in preset['TrainConfig'].items():
            if isinstance(v, str):
                print(f"TC.{k} = '{v}'")
            else:
                print(f"TC.{k} = {v}")

    print("\n" + "="*70)
    print("\n或者直接修改配置文件:")
    print("  - configs/config.py      (SystemConfig)")
    print("  - configs/train_config.py (TrainConfig)")


def interactive_config():
    """交互式配置向导"""
    print("\n" + "="*70)
    print("交互式参数配置向导")
    print("="*70)

    print("\n选择配置模式:")
    print("  1. 使用预设方案")
    print("  2. 自定义配置（逐项设置）")
    print("  3. 显示当前配置")
    print("  4. 退出")

    choice = input("\n请输入选项 (1-4): ").strip()

    if choice == "1":
        show_presets()
        preset_name = input("\n请输入预设名称: ").strip()
        apply_preset(preset_name)

    elif choice == "2":
        print("\n自定义配置功能开发中...")
        print("当前请直接编辑配置文件:")
        print("  - configs/config.py")
        print("  - configs/train_config.py")
        print("\n参考文档: docs/parameters_reference.md")

    elif choice == "3":
        show_current_config()

    elif choice == "4":
        print("\n退出配置向导")
        return

    else:
        print("\n❌ 无效选项，请重新选择")
        interactive_config()


# ============================================================================
# 参数验证
# ============================================================================

def validate_config():
    """验证配置合理性"""
    issues = []

    # 检查关键一致性
    if SystemConfig.REWARD_GAMMA != TrainConfig.GAMMA:
        issues.append(
            f"⚠️ REWARD_GAMMA ({SystemConfig.REWARD_GAMMA}) != GAMMA ({TrainConfig.GAMMA})"
        )

    if SystemConfig.MAX_STEPS != TrainConfig.MAX_STEPS:
        issues.append(
            f"⚠️ SystemConfig.MAX_STEPS ({SystemConfig.MAX_STEPS}) != TrainConfig.MAX_STEPS ({TrainConfig.MAX_STEPS})"
        )

    if TrainConfig.EMBED_DIM % TrainConfig.NUM_HEADS != 0:
        issues.append(
            f"❌ EMBED_DIM ({TrainConfig.EMBED_DIM}) 必须能被 NUM_HEADS ({TrainConfig.NUM_HEADS}) 整除"
        )

    # 检查资源可行性
    max_task_time = SystemConfig.MAX_COMP / SystemConfig.MIN_VEHICLE_CPU_FREQ
    episode_time = SystemConfig.MAX_STEPS * SystemConfig.DT
    if max_task_time > episode_time:
        issues.append(
            f"⚠️ 最大任务执行时间 ({max_task_time:.1f}s) > Episode 时长 ({episode_time:.1f}s)"
        )

    # 报告结果
    if issues:
        print("\n" + "="*70)
        print("配置验证结果：发现问题")
        print("="*70)
        for issue in issues:
            print(issue)
        print("\n建议修复后再开始训练")
        return False
    else:
        print("\n" + "="*70)
        print("✅ 配置验证通过！")
        print("="*70)
        return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="交互式参数配置工具")
    parser.add_argument("--show", action="store_true", help="显示当前配置")
    parser.add_argument("--preset", type=str, help="应用预设方案")
    parser.add_argument("--validate", action="store_true", help="验证配置合理性")
    parser.add_argument("--list-presets", action="store_true", help="列出所有预设")

    args = parser.parse_args()

    if args.show:
        show_current_config()
    elif args.preset:
        apply_preset(args.preset)
    elif args.validate:
        validate_config()
    elif args.list_presets:
        show_presets()
    else:
        # 默认进入交互模式
        interactive_config()


if __name__ == "__main__":
    main()
