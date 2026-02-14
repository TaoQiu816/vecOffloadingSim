# DIAGNOSIS PATCH NOTES（基于当前仓库代码与已落盘数据）

## 0. 范围与原则
- 本次修复遵循最小侵入：不改通信/计算动力学公式（V2V RB-SINR、队列推进、5-phase 流程不改）。
- 允许修改：baseline 单位口径、统计脚本、奖励数值稳定、训练偏置/退火、日志字段、审计脚本。
- 结论区分为：
  - `可写论文`：统计口径正确、数据来源可复现。
  - `暂不写论文`：修复前口径有偏差或修复后重跑尚未完成。

## 1. 修复点与验证

### 1.1 EFT/CP-EFT/LB-Greedy 通信时延单位口径修复
- 问题：baseline 里通信时延用了 `task_data*8/rate`，而环境口径是 `bits/rate`。
- 修改：`baselines/eft_policy.py` 统一改为 `t_tx = task_data_bits / rate_bps`，并新增 `_tx_time_seconds()` 注释固化单位契约。
- 代码位置：`baselines/eft_policy.py:116`、`baselines/eft_policy.py:147`、`baselines/eft_policy.py:168`。
- 验证：`tests/test_eft_policy.py:46` 新增单测，断言 baseline 与 env 的 bits/s 口径一致。

### 1.2 严谨对比统计口径修复（raw baseline + matched-tail）
- 问题：旧版 rigorous 对比把 baseline 120ep 前向填充到 1000ep 后参与统计，导致均值/方差/p 值扭曲。
- 修改：
  - forward-fill 仅用于趋势画图；
  - 统计/显著性只用 baseline raw 样本；
  - 采用 matched-tail：`K=min(len(RL窗口), len(baseline_raw))`；
  - 增加 Welch t-test + bootstrap 95% CI；
  - 增加 on-task 条件统计表。
- 代码位置：`scripts/final/plot_rigorous_compare_cn.py:213`、`scripts/final/plot_rigorous_compare_cn.py:374`、`scripts/final/plot_rigorous_compare_cn.py:449`。
- 验证：`tests/test_rigorous_compare_stats.py:10`。
- 前后对比证据：
  - 修复前：`final_rigorous_cn/rigorous_window_summary.csv` 中 baseline `n=1000`（无效）。
  - 修复后：`final_rigorous_cn_patch_v2/rigorous_window_summary.csv` 中 baseline `n=120`，`matched_k=120`，`stat_basis=matched_tail_raw_baseline`（有效）。

### 1.3 UNIFIED 干扰项数值稳定修复（不改动力学）
- 问题：`I_ref` 过小时 `(I_caused/I_ref)^p` 会爆表，出现 `-1e9` 级 `r_interf`，破坏早期学习。
- 修改：
  - 新增 `I_REF_MIN_UNIFIED`（`configs/config.py:658`）；
  - 新增 `INTERF_RATIO_CLIP_UNIFIED`（`configs/config.py:660`）；
  - 在 `compute_unified_step_reward` 中对 `I_ref` 加 floor、对 ratio clip。
- 代码位置：`envs/rl/reward_functions.py:91-103`。
- 验证：
  - 单测：`tests/test_unified_interf_stability.py:19`；
  - 审计脚本：`scripts/audit/check_interf_extremes.py`。
  - 结果：
    - 修复前旧 run（1000ep）：`r_interf min = -6.329e9`（见 `final_diagnosis_cn_patch_v2/interf_extreme_audit.txt`）。
    - 修复后短训 run：`r_interf min = -9.03`（`runs/patch_stageB_biascheck_10ep80/audit_results/interf_extreme_audit.txt`）。
    - 修复后随机2ep：`r_interf min = -44.72`（`runs/patch_stageB_biascheck_10ep80/audit_results/interf_extreme_random2ep.txt`）。

### 1.4 关键字段写入主 CSV（train + baseline）
- 修改：训练和 baseline 输出补齐并对齐：
  - `illegal_action_rate`, `no_task_rate`, `on_task_rate`, `has_task_available_rate`, `unified_illegal_trigger_rate`
  - `decision_frac_local/rsu/v2v`, `mean_cft_est`, `episode_time_seconds`, `deadline_miss_rate`, `time_limit_rate`
  - `power_ratio_mean/p95`, `avg_power`, `I_total_p50/p95`, `I_caused_mean/p95`
  - `trust_failure_rate`, `rho_selected_p10`, `uncertainty_selected_p90`
- 代码位置：
  - `train.py:75-138`, `train.py:1494-1496`, `train.py:2519-2520`, `train.py:2743-2744`
  - `envs/vec_offloading_env.py:6867-6873`
  - `scripts/run_baselines.py:297-351`
- 契约验证：`python scripts/assert_csv_contract.py --run-dir runs/formal_rerun_n20_all_seed42_latest` 通过。

### 1.5 运行元数据与可复现性
- 修改：baseline 侧补写 `baseline_run_meta.json`（含 `git_commit`/`config_hash`/seed/steps/episodes）。
- 代码位置：`scripts/run_baselines.py:217-227`。

### 1.6 探索偏置与熵退火（最小调整）
- 修改：
  - `LOGIT_BIAS_RSU=0.0`，`LOGIT_BIAS_V2V_INIT=0.8`，慢退火；
  - 熵系数改为起高后降：`ENTROPY_COEF_START=0.004 -> ENTROPY_COEF_END=0.001`；
  - 训练时按全局步数线性退火生效。
- 代码位置：`configs/train_config.py:196-199,247,283-285`，`train.py:279-293,2136-2139,2175`。
- 快速验证（10ep×80step, StageB-like）：
  - `decision_frac_v2v` 均值约 `0.834`，`I_total_p95` 非零（约 `6.09e-05`），说明干扰信号被点亮。
  - 同时 `abs_ratio_r_interf` 接近 1，提示 StageB 下干扰项可能反向主导，需后续权重再平衡（见第3节）。

## 2. 当前结果（基于 1000ep 主 run + 修复后严谨统计脚本）
数据源：`runs/formal_rerun_n20_all_seed42_latest`

### 2.1 MAPPO 末100ep主要指标
- `task_success_rate = 0.8205`
- `deadline_miss_rate = 0.1795`
- `mean_cft_est = 11.3359`
- `time_limit_rate = 0.0`
- `illegal_action_rate = 0.0`
- `abs_ratio_r_illegal = 0.0`

### 2.2 明确存在的问题（客观）
- 仍有显著 RSU 塌缩：`decision_frac_rsu ≈ 0.9986`，`decision_frac_v2v ≈ 0.0011`（末100ep）。
- 干扰目标后期失活：`I_total_p95 = 0`，`abs_ratio_r_interf = 0`（末100ep）。
- 决策稀疏：`no_task_rate ≈ 0.756`（末100ep）。
- 信誉改进有限：`trust_failure_rate ≈ 0.227`（末100ep）。

### 2.3 严谨统计后的“可写”结论
- 修复后统计口径下（raw+matched-tail），MAPPO 在 `task_success_rate / deadline_miss_rate / mean_cft_est` 上相对多数 baseline 仍显著更优（见 `final_rigorous_cn_patch_v2/rigorous_pairwise_significance.csv`）。
- 但对 `trust_failure_rate` 的优势不稳定/不显著。

## 3. 仍需继续的工作（按优先级）
1. 完成 baseline patch-run 全量重跑并固化最终对比（当前 `runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1` 正在跑）。
2. StageB 下调平衡奖励量纲，避免 `r_interf` 在高V2V阶段过主导（建议先调 `W_INTERF` 或 `INTERF_RATIO_CLIP_UNIFIED`，不改动力学）。
3. 用 on-task 口径补一版对比表：`P(action|on_task)`、`I_total/SINR` 激活率、`power_std`，用于解释“模块确实被使用”。
4. 按 Stage 匹配做消融：
   - StageA：Transformer / risk / PBRS
   - StageB：interf / power

## 4. 哪些可写论文，哪些不可以
- `可写`：
  - 修复后 rigorous 脚本输出（`final_rigorous_cn_patch_v2/*`）的统计结论；
  - illegal/no_task 分离与 CSV 对齐后的指标解释；
  - 干扰极值修复前后的数值稳定证据。
- `暂不写`：
  - 修复前 `final_rigorous_cn` 中基于 forward-fill baseline 的统计显著性结论（口径无效）；
  - EFT 单位修复的最终强结论（需等 patch-run baseline 全量完成后定稿）。

## 5. 复现命令（本次修复相关）
```bash
# 单元测试
pytest -q tests/test_eft_policy.py tests/test_rigorous_compare_stats.py tests/test_unified_interf_stability.py

# 严谨对比图（修复后统计逻辑）
python scripts/final/plot_rigorous_compare_cn.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest \
  --out-name final_rigorous_cn_patch_v2 --window 100

# 常规对比图
python scripts/final/plot_final_compare.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest \
  --out-name final_compare_patch_v2 --window 100
python scripts/final/plot_final_compare.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest \
  --out-name final_standard_compare_patch_v2 --window 50

# CSV 契约检查
python scripts/assert_csv_contract.py --run-dir runs/formal_rerun_n20_all_seed42_latest

# 干扰极值审计（修复前 run，可复现“爆表”）
python scripts/audit/check_interf_extremes.py \
  --run-dir runs/formal_rerun_n20_all_seed42_latest \
  --random-episodes 0 --threshold=-1e4

# 干扰极值审计（修复后短训 run，应 PASS）
python scripts/audit/check_interf_extremes.py \
  --run-dir runs/patch_stageB_biascheck_10ep80 \
  --random-episodes 0 --threshold=-1e4
```
