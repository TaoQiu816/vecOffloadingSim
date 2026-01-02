#!/bin/bash
# Warmup训练一键启动脚本
# 先运行回归测试确保环境正常，再启动200ep训练

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Warmup训练启动脚本"
echo "=========================================="
echo ""

# 步骤1: 运行回归测试（快速验证环境）
echo "[1/2] 运行回归测试..."
python scripts/regression_local_no_tx_deadlock.py --episodes 3 --seed 42
if [ $? -ne 0 ]; then
    echo "❌ 回归测试失败，请先修复环境问题"
    exit 1
fi
echo "✅ 回归测试通过"
echo ""

# 步骤2: 启动训练
echo "[2/2] 启动Warmup训练 (200 episodes)..."
python train.py \
    --run-id warmup_200ep \
    --max-episodes 200 \
    --seed 42 \
    --device cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Warmup训练完成！"
    echo "=========================================="
    echo "结果保存在: runs/warmup_200ep/"
    echo ""
    echo "关键指标检查："
    echo "  - deadlock_vehicle_count 应始终为 0"
    echo "  - audit_deadline_misses 应接近 0"
    echo "  - vehicle_success_rate 应明显 > 0.6 且上升"
    echo "  - RSU占比应 >= 30%"
    echo ""
else
    echo ""
    echo "❌ 训练失败，请检查日志"
    exit 1
fi

