# Ablation Visibility Check

- metrics_path: `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/runs/formal_rerun_n20_all_seed42_latest_patch_paperfix_v1/logs/metrics.csv`
- tail episodes: 100

## Summary
- episodes: 100
- on_task_rate_mean: 0.24371404430823207
- p_local_on_task: 0.0002538411985061
- p_rsu_on_task: 0.9986327685858077
- p_v2v_on_task: 0.0011133902156861921
- decision_collapse_max_frac: 0.9986327685858077
- entropy_mean: 0.3128289262147709
- power_ratio_std: 0.01572808102844795
- interf_metric_nonzero_rate: 0.0
- r_interf_abs_nonzero_rate: 0.0
- r_risk_abs_nonzero_rate: 1.0
- flag_policy_collapse: 1.0
- flag_interf_not_activated: 1.0
- flag_power_head_nearly_constant: 0.0
- flag_no_task_dilution: 1.0

## Interpret flags
- `flag_policy_collapse=1`: target distribution collapsed; ablation differences may be hidden.
- `flag_interf_not_activated=1`: interference objective likely unobservable under current stage.
- `flag_power_head_nearly_constant=1`: fixed-power-like behavior; power ablation may appear ineffective.
- `flag_no_task_dilution=1`: all-step metrics likely diluted by high no_task; use on_task-conditioned stats.