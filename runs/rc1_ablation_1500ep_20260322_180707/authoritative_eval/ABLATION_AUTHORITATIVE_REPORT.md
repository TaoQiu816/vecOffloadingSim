# RC1 Ablation AUTHORITATIVE Eval Report

> Generated: 2026-03-23 02:29:43
> Protocol: seeds 9000-9009, episodes=10, MAX_STEPS=300, deterministic, frozen scene

## Summary Table

| Policy | task_success_rate_B | deadline_miss_rate | mean_cft | subtask_sr | local | rsu | v2v | score S |
|------|------:|------:|------:|------:|------:|------:|------:|------|
| w/o DAG-Feature::last_model | 0.9700 | 0.0000 | 1.5148 | 0.9962 | 0.4724 | 0.5199 | 0.0077 | `(0.9700, -0.0000, -1.5148)` |
| w/o DAG-Feature::best_model | 0.9700 | 0.0000 | 1.5264 | 0.9974 | 0.5348 | 0.4209 | 0.0443 | `(0.9700, -0.0000, -1.5264)` |
| Full-MAPPO::last_model | 0.9500 | 0.0000 | 1.5657 | 0.9965 | 0.3557 | 0.2593 | 0.3850 | `(0.9500, -0.0000, -1.5657)` |
| Full-MAPPO::best_model | 0.9350 | 0.0000 | 1.6963 | 0.9940 | 0.3326 | 0.2234 | 0.4440 | `(0.9350, -0.0000, -1.6963)` |
| Local-Only | 0.8750 | 0.0000 | 1.4804 | 0.9837 | 1.0000 | 0.0000 | 0.0000 | `(0.8750, -0.0000, -1.4804)` |
| Greedy-Local | 0.8750 | 0.0000 | 1.4804 | 0.9837 | 1.0000 | 0.0000 | 0.0000 | `(0.8750, -0.0000, -1.4804)` |
| w/o DAG & Resource::last_model | 0.8750 | 0.0000 | 1.4982 | 0.9833 | 0.9919 | 0.0044 | 0.0037 | `(0.8750, -0.0000, -1.4982)` |
| w/o DAG & Resource::best_model | 0.8750 | 0.0000 | 1.5030 | 0.9839 | 0.9924 | 0.0032 | 0.0044 | `(0.8750, -0.0000, -1.5030)` |
| w/o Resource-Feature::best_model | 0.8750 | 0.0000 | 1.5043 | 0.9842 | 0.9896 | 0.0033 | 0.0071 | `(0.8750, -0.0000, -1.5043)` |
| w/o Resource-Feature::last_model | 0.8700 | 0.0000 | 1.5007 | 0.9847 | 0.9922 | 0.0037 | 0.0041 | `(0.8700, -0.0000, -1.5007)` |
| Legal-Random | 0.2400 | 0.0000 | 1.4037 | 0.6206 | 0.7623 | 0.0243 | 0.2134 | `(0.2400, -0.0000, -1.4037)` |

## Best Checkpoint Per Ablation Run

| Run | Winner | Score S |
|------|------|------|
| full | Full-MAPPO::last_model | `(0.9500, -0.0000, -1.5657)` |
| wo_dag | w/o DAG-Feature::last_model | `(0.9700, -0.0000, -1.5148)` |
| wo_resource | w/o Resource-Feature::best_model | `(0.8750, -0.0000, -1.5043)` |
| wo_dag_resource | w/o DAG & Resource::last_model | `(0.8750, -0.0000, -1.4982)` |

## Overall Ranking

1. `w/o DAG-Feature::last_model` — S=(0.9700, -0.0000, -1.5148)
2. `w/o DAG-Feature::best_model` — S=(0.9700, -0.0000, -1.5264)
3. `Full-MAPPO::last_model` — S=(0.9500, -0.0000, -1.5657)
4. `Full-MAPPO::best_model` — S=(0.9350, -0.0000, -1.6963)
5. `Local-Only` — S=(0.8750, -0.0000, -1.4804)
6. `Greedy-Local` — S=(0.8750, -0.0000, -1.4804)
7. `w/o DAG & Resource::last_model` — S=(0.8750, -0.0000, -1.4982)
8. `w/o DAG & Resource::best_model` — S=(0.8750, -0.0000, -1.5030)
9. `w/o Resource-Feature::best_model` — S=(0.8750, -0.0000, -1.5043)
10. `w/o Resource-Feature::last_model` — S=(0.8700, -0.0000, -1.5007)
11. `Legal-Random` — S=(0.2400, -0.0000, -1.4037)
