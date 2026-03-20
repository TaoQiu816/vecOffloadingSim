#!/bin/bash
set -e

# 1. 创建实验目录
mkdir -p runs/run_1000ep_A_20260320 runs/run_1000ep_B_20260320

# 2. 检查初始配置
echo "=== 初始配置检查 ==="
grep -E "LR_CRITIC|ENTROPY_COEF" configs/train_config.py

# 3. 启动实验A（screen后台窗口）
echo -e "\n=== 启动实验A ==="
screen -dmS exp_A python train.py --run-dir runs/run_1000ep_A_20260320 --exact-run-dir --max-episodes 1000 --save-interval 100
# 等待3秒确保进程启动，然后查找实验A的python训练进程PID
sleep 3
PID_A=$(pgrep -f "train.py --run-dir runs/run_1000ep_A_20260320")
echo "实验A PID: $PID_A (screen窗口：exp_A)"

# 4. 修改实验B配置
echo -e "\n=== 修改实验B配置 ==="
sed -i 's/LR_CRITIC = 2e-4/LR_CRITIC = 5e-4/' configs/train_config.py
sed -i 's/ENTROPY_COEF_START = 0\.02/ENTROPY_COEF_START = 0.012/' configs/train_config.py
sed -i 's/ENTROPY_COEF_END = 0\.02/ENTROPY_COEF_END = 0.012/' configs/train_config.py

# 5. 检查修改后配置
echo "=== 修改后配置检查 ==="
grep -E "LR_CRITIC|ENTROPY_COEF" configs/train_config.py

# 6. 启动实验B（screen后台窗口）
echo -e "\n=== 启动实验B ==="
screen -dmS exp_B python train.py --run-dir runs/run_1000ep_B_20260320 --exact-run-dir --max-episodes 1000 --save-interval 100
# 等待3秒确保进程启动，查找实验B的python训练进程PID
sleep 3
PID_B=$(pgrep -f "train.py --run-dir runs/run_1000ep_B_20260320")
echo "实验B PID: $PID_B (screen窗口：exp_B)"

# 7. 等待所有训练完成
echo -e "\n=== 等待实验A/B训练完成 ==="
while true; do
  # 检查PID是否存活（0=存活，非0=结束）
  kill -0 $PID_A > /dev/null 2>&1
  STATUS_A=$?
  kill -0 $PID_B > /dev/null 2>&1
  STATUS_B=$?

  # 两个进程都结束则退出循环
  if [ $STATUS_A -ne 0 ] && [ $STATUS_B -ne 0 ]; then
    echo "=== 所有训练已完成 ==="
    break
  fi

  # 每30秒检查一次，输出状态
  sleep 30
  STATUS_A_STR=$(if [ $STATUS_A -eq 0 ]; then echo "运行中"; else echo "已完成"; fi)
  STATUS_B_STR=$(if [ $STATUS_B -eq 0 ]; then echo "运行中"; else echo "已完成"; fi)
  echo "等待中... 实验A: $STATUS_A_STR | 实验B: $STATUS_B_STR"
done

# 8. 提交Git结果
echo -e "\n=== 提交训练结果 ==="
git add runs/run_1000ep_A_20260320/ runs/run_1000ep_B_20260320/ configs/train_config.py envs/vec_offloading_env.py train.py .gitignore && \
git commit -m "run_1000ep_AB_20260320: A(lr=2e-4,ent=0.02) B(lr=5e-4,ent=0.012)" && \
git push

# 9. 等待2分钟后关机
echo -e "\n=== 所有操作完成，2分钟后关闭服务器（取消关机执行：shutdown -c） ==="
sleep 120
shutdown -h now
