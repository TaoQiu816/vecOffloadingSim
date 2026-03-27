# LR Critic Final Figure Pack

## Export Settings
- Smooth window: 36
- Band window: 28
- Tail window: 100
- Source experiments: run_1000ep_A_20260320, run_1000ep_A_lrcritic_3e4_20260321, run_1000ep_A_lrcritic_5e4_20260321
- Actual source training horizon: 1000 episodes for all three learning-rate runs

## Source Run Inventory
- lr_c=2e-4: run=run_1000ep_A_20260320, episode_max=1000, unique_episode_count=1000
- lr_c=3e-4: run=run_1000ep_A_lrcritic_3e4_20260321, episode_max=1000, unique_episode_count=1000
- lr_c=5e-4: run=run_1000ep_A_lrcritic_5e4_20260321, episode_max=1000, unique_episode_count=1000

## Tail Summary
- lr_c=2e-4: reward_mean=0.0108, reward_total=0.2141, task_sr=0.9285, deadline_miss=0.0715, avg_rsu_queue=1.2608
- lr_c=3e-4: reward_mean=0.0106, reward_total=0.2123, task_sr=0.8960, deadline_miss=0.1040, avg_rsu_queue=1.5143
- lr_c=5e-4: reward_mean=0.0091, reward_total=0.1876, task_sr=0.9130, deadline_miss=0.0870, avg_rsu_queue=7.1309

## Visual Takeaways
- Highest tail task success rate: lr_c=2e-4
- Lowest tail deadline miss rate: lr_c=2e-4
- Highest tail average reward: lr_c=2e-4
- Highest tail total reward: lr_c=2e-4

## Exported Files
- `tables/lr_critic_main_training_table.csv`
- `tables/lr_critic_diagnostics_table.csv`
- `tables/lr_critic_decision_mix_table.csv`
- `tables/lr_critic_tail_summary_table.csv`
- `figures/fig_reward_mean_final.png`
- `figures/fig_reward_total_final.png`
- `figures/fig_task_sr_final.png`
- `figures/fig_deadline_miss_rate_final.png`
- `figures/fig_mean_cft_completed_final.png`
- `figures/fig_avg_rsu_queue_final.png`
- `figures/fig_approx_kl_final.png`
- `figures/fig_entropy_final.png`
- `figures/fig_clip_frac_final.png`
- `figures/fig_ratio_local_final.png`
- `figures/fig_ratio_rsu_final.png`
- `figures/fig_ratio_v2v_final.png`
- `figures/fig_tail_reward_mean_final.png`
- `figures/fig_tail_reward_total_final.png`
- `figures/fig_tail_task_sr_final.png`
- `figures/fig_tail_deadline_miss_rate_final.png`
- `figures/fig_tail_avg_rsu_queue_final.png`
