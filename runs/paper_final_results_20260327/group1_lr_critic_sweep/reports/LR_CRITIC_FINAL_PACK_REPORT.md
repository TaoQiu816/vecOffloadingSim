# LR Critic Final Figure Pack

## Export Settings
- Smooth window: 36
- Band window: 28
- Tail window: 100
- Source experiments: lr_critic_1500ep_20260327_163712/lr_c2e4, lr_c3e4, lr_c5e4
- Actual source training horizon: 1500 episodes for all three learning-rate runs

## Source Run Inventory
- lr_c=2e-4: run=lr_c2e4, episode_max=1500, unique_episode_count=1500
- lr_c=3e-4: run=lr_c3e4, episode_max=1500, unique_episode_count=1500
- lr_c=5e-4: run=lr_c5e4, episode_max=1500, unique_episode_count=1500

## Tail Summary
- lr_c=2e-4: reward_mean=0.0123, reward_total=0.2306, task_sr=0.9530, deadline_miss=0.0470, avg_rsu_queue=0.9400
- lr_c=3e-4: reward_mean=0.0107, reward_total=0.2145, task_sr=0.9070, deadline_miss=0.0930, avg_rsu_queue=1.6584
- lr_c=5e-4: reward_mean=0.0058, reward_total=0.1466, task_sr=0.7040, deadline_miss=0.2885, avg_rsu_queue=4.7495

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
