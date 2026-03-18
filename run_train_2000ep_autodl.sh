#!/bin/bash
# 2000 episode训练脚本（AutoDL环境）
# 训练完成后自动同步代码到git并关机

set -e  # 遇到错误立即退出

echo "========================================="
echo "开始2000 episode训练"
echo "时间: $(date)"
echo "========================================="

# 1. 运行训练
echo ""
echo "[1/4] 开始训练..."
python train.py \
    --max-episodes 2000 \
    --seed 42 \
    --device cuda \
    --save-interval 100 \
    --eval-interval 100 \
    --log-interval 10

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "✓ 训练完成"
else
    echo "✗ 训练失败"
    exit 1
fi

# 2. 同步代码到git
echo ""
echo "[2/4] 同步代码到git..."
git add -A
git commit -m "训练完成: 2000ep reward修复后 $(date +%Y%m%d_%H%M%S)" || echo "无新改动需要提交"
git push origin main || git push origin master || echo "推送失败，请检查git配置"

if [ $? -eq 0 ]; then
    echo "✓ 代码已同步到git"
else
    echo "⚠ git同步可能失败，但继续执行"
fi

# 3. 等待2分钟
echo ""
echo "[3/4] 等待2分钟后关机..."
for i in {120..1}; do
    echo -ne "\r剩余 $i 秒...  "
    sleep 1
done
echo ""

# 4. 关机（AutoDL）
echo ""
echo "[4/4] 关机..."
echo "训练完成，正在关机..."
sudo shutdown -h now

echo "========================================="
echo "脚本执行完毕"
echo "========================================="
