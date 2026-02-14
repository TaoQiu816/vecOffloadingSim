# DIAGNOSIS_PATCH_NOTES（论文级闭环修复）

## 一、修复范围与约束
- 仅修改：判定逻辑/统计日志/baseline实现/绘图统计脚本/训练偏置与退火参数。
- 未修改：通信与计算动力学公式（V2V RB-SINR、V2I share、5-phase推进、队列推进）。

## 二、修复项 -> 修改 -> 验证

### 1) EFT 单位口径修复（bits/s）
- 问题：baseline EFT 传输时延存在 `task_data*8/rate` 风险，和环境 `bits/rate` 口径不一致。
- 修改：`baselines/eft_policy.py`
  - 新增 `EFTPolicy._tx_time_seconds(task_data_bits, rate_bps)`。
  - RSU/V2V 两处改为 `t_tx = task_data_bits / rate_bps`。
- 代码：`baselines/eft_policy.py:116`, `baselines/eft_policy.py:147`, `baselines/eft_policy.py:168`。
- 单测：`tests/test_eft_policy.py:46`（通过）。

### 2) 严谨图统计修复（禁止 forward-fill 参与统计）
- 问题：旧版 rigorous 对 baseline 做前向填充后直接统计，均值/方差/p值偏差。
- 修改：`scripts/final/plot_rigorous_compare_cn.py`
  - 绘图曲线可延展；统计和显著性仅用 baseline raw；
  - matched-tail：`K=min(len(RL窗口), len(baseline_raw))`；
  - 增加 Welch t-test + bootstrap 95% CI；
  - 增加 on-task 条件统计输出 `rigorous_on_task_summary.csv`。
- 代码：`scripts/final/plot_rigorous_compare_cn.py:213`, `scripts/final/plot_rigorous_compare_cn.py:374`, `scripts/final/plot_rigorous_compare_cn.py:449`。
- 单测：`tests/test_rigorous_compare_stats.py:10`（通过）。
- 验证结果：`final_rigorous_cn/rigorous_window_summary.csv` 中 baseline `n` 仅为 `100/120`，且 `stat_basis=matched_tail_raw_baseline`。

### 3) 干扰奖励极值稳定化（不改动力学）
- 问题：`I_ref` 过小时，`(I_caused/I_ref)^p` 可爆到 `-1e9` 级。
- 修改：
  - `configs/config.py` 增加 `I_REF_MIN_UNIFIED=1e-8`、`INTERF_RATIO_CLIP_UNIFIED=20.0`。
  - `envs/rl/reward_functions.py` 对 `I_ref` 加 floor，对 ratio clip。
- 代码：`configs/config.py:658`, `configs/config.py:660`, `envs/rl/reward_functions.py:91-103`。
- 单测：`tests/test_unified_interf_stability.py:19`（通过）。
- 审计：
  - 修复前旧 run（1000ep）：`r_interf min=-6.329e9`。
  - 修复后短训 run（10ep×80）：`r_interf min=-9.03`（PASS 阈值 -1e4）。
  - 修复后随机策略2ep：`r_interf min=-44.72`（PASS 阈值 -1e4）。

### 4) train/baseline CSV 同口径关键列
- 修改：`train.py`, `envs/vec_offloading_env.py`, `scripts/run_baselines.py`
  - 新增并贯通：`on_task_rate`, `has_task_available_rate`。
  - 保证 `illegal/no_task/unified_illegal_trigger` 与延迟/功率/干扰/风险关键列都在 train + baseline。
- 代码：
  - `train.py:75-138`, `train.py:1494-1496`, `train.py:2519-2520`, `train.py:2743-2744`
  - `envs/vec_offloading_env.py:6867-6873`
  - `scripts/run_baselines.py:297-351`
- 契约验证：`python scripts/assert_csv_contract.py --run-dir runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1` 通过。

### 5) 运行可追溯元数据
- `baseline_run_meta.json` 写入 `git_commit/config_hash/seed/max_steps/num_episodes`。
- 代码：`scripts/run_baselines.py:217-227`。

### 6) 探索偏置与熵退火（最小调整）
- 修改：`configs/train_config.py` + `train.py`
  - `LOGIT_BIAS_RSU=0.0`, `LOGIT_BIAS_V2V_INIT=0.8`，慢退火；
  - 熵退火 `ENTROPY_COEF_START=0.004 -> END=0.001`。
- 代码：`configs/train_config.py:196-199`, `configs/train_config.py:247`, `configs/train_config.py:283-285`, `train.py:279-293`, `train.py:2136-2139`。

## 三、重跑与图表产物

### 1) Baseline 重跑（120ep）
- 目录：`runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1/logs/baseline_stats.csv`
- 状态：7 个 baseline 全部 120ep 完成并合并（`baseline_parts/*.csv` 各 121 行含表头）。

### 2) 图与统计（可直接用于论文）
- 严谨对比（修复统计口径）：
  - `runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1/final_rigorous_cn/`
- 常规对比：
  - `runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1/final_compare/`
- 标准窗口对比：
  - `runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1/final_standard_compare/`

## 四、前后对比（基于代码+数据）

### 1) MAPPO 主结果（1000ep，末100）
- `task_success_rate=0.8205`
- `deadline_miss_rate=0.1795`
- `mean_cft_est=11.3359`
- `time_limit_rate=0`
- `illegal_action_rate=0`, `abs_ratio_r_illegal=0`

### 2) 仍存在的核心现状
- RSU 塌缩仍明显：`decision_frac_rsu≈0.9986`，`decision_frac_v2v≈0.0011`（末100）。
- 干扰后期失活：`I_total_p95=0`，`abs_ratio_r_interf=0`（末100）。
- 决策稀疏：`no_task_rate≈0.756`。
- trust 改善有限：`trust_failure_rate≈0.227`。

### 3) EFT/CP-EFT/LB-Greedy 修复后变化（与旧 baseline 对比）
- EFT（改善方向正确）：
  - `task_sr 0.2504 -> 0.2850`
  - `deadline_miss_rate 0.7496 -> 0.7150`
  - `time_limit_rate 0.9167 -> 0.8583`
  - `mean_cft_est 23.6830 -> 23.3701`
- CP-EFT / LB-Greedy：成功率与超时率变化不大（部分略波动），说明“单位问题并非唯一瓶颈”，其性能还受当前场景/策略假设影响。
- 注意：`reward_mean` 在多数 baseline 上显著上移，主要受干扰项数值稳定化影响，**不宜单独用于修复效果主结论**。

## 五、可写论文 vs 暂不写
- 可写：
  - 修复后 rigorous 统计（raw baseline + matched-tail）得到的显著性结论；
  - illegal/no_task 口径分离与 CSV 同口径；
  - 干扰奖励极值修复前后数值稳定对比。
- 暂不写：
  - 修复前 `final_rigorous_cn` 的统计显著性（forward-fill 污染）。
  - 把 CP-EFT/LB-Greedy 的小幅波动直接解释为“算法劣化/优化”，需结合场景和接口再做单独分析。

## 六、回归与复现命令
```bash
# 1) 单测
pytest -q tests/test_eft_policy.py tests/test_rigorous_compare_stats.py tests/test_unified_interf_stability.py

# 2) baseline 重跑（并行）
bash scripts/run_baselines_cpu_parallel.sh \
  runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1 42 120 200

# 3) CSV 契约
python scripts/assert_csv_contract.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1

# 4) 严谨图
python scripts/final/plot_rigorous_compare_cn.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1 \
  --out-name final_rigorous_cn --window 100

# 5) 常规与标准图
python scripts/final/plot_final_compare.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1 \
  --out-name final_compare --window 100
python scripts/final/plot_final_compare.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1 \
  --out-name final_standard_compare --window 50
```
