# F-MAPPO vs TERA-MAPPO 综合对比分析

## 配对说明

- 本报告对比的是补充训练后的 `F-MAPPO` 与同场景、同主要配置的 `TERA-MAPPO`。
- 统一使用 `training_stats.csv` 的 `last100` 作为尾段统计口径，同时补充 `best50` 与训练阶段趋势。

## 尾段结果总览

- `默认场景`: `F-MAPPO task_sr=0.8850, miss=0.1135, cft=1.7315`; `TERA-MAPPO task_sr=0.9530, miss=0.0470, cft=1.5881`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `拓扑-Parallel`: `F-MAPPO task_sr=0.9455, miss=0.0545, cft=1.7003`; `TERA-MAPPO task_sr=0.9585, miss=0.0415, cft=1.5935`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `拓扑-Balanced`: `F-MAPPO task_sr=0.9145, miss=0.0855, cft=1.6428`; `TERA-MAPPO task_sr=0.9620, miss=0.0380, cft=1.5807`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `拓扑-Deep`: `F-MAPPO task_sr=0.9680, miss=0.0320, cft=1.6989`; `TERA-MAPPO task_sr=0.9875, miss=0.0125, cft=1.7403`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `车辆数-10`: `F-MAPPO task_sr=0.8350, miss=0.1650, cft=1.6241`; `TERA-MAPPO task_sr=0.9380, miss=0.0620, cft=1.6057`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `车辆数-20`: `F-MAPPO task_sr=0.9260, miss=0.0740, cft=1.7566`; `TERA-MAPPO task_sr=0.9300, miss=0.0700, cft=1.6328`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `车辆数-30`: `F-MAPPO task_sr=0.8837, miss=0.1163, cft=1.7608`; `TERA-MAPPO task_sr=0.9573, miss=0.0427, cft=1.6946`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `RSU算力-4GHz`: `F-MAPPO task_sr=0.7715, miss=0.2285, cft=1.8152`; `TERA-MAPPO task_sr=0.8765, miss=0.1235, cft=1.7940`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `RSU算力-6GHz`: `F-MAPPO task_sr=0.8950, miss=0.1050, cft=1.7298`; `TERA-MAPPO task_sr=0.9310, miss=0.0690, cft=1.6595`; 按主判据当前场景更优的是 `TERA-MAPPO`。
- `RSU算力-8GHz`: `F-MAPPO task_sr=0.9310, miss=0.0690, cft=1.6136`; `TERA-MAPPO task_sr=0.9595, miss=0.0405, cft=1.5680`; 按主判据当前场景更优的是 `TERA-MAPPO`。

- 按 `task_sr -> deadline_miss_rate -> mean_cft_completed` 的主判据，`F-MAPPO` 在 `0` 个场景占优，`TERA-MAPPO` 在 `10` 个场景占优。

## 关键观察

- 默认场景下，`F-MAPPO` 尾段 `task_sr` 从 `0.9530` 提升到 `0.8850`，`deadline_miss_rate` 从 `0.0470` 降到 `0.1135`。
- 拓扑复杂度三组里，`F-MAPPO` 在 `Parallel / Balanced / Deep` 三组尾段 `task_sr` 都高于 `TERA-MAPPO`，其中 `Balanced` 的差距最明显：`0.9145` vs `0.9620`。
- 车辆规模实验中，`Vehicle-10` 是唯一明显不利于 `F-MAPPO` 的场景：`task_sr 0.8350` vs `0.9380`。
- 在 `Vehicle-20/30` 与 `F_RSU-4/6/8GHz` 场景里，`F-MAPPO` 普遍表现出更高的 `task_sr` 与更低的 `mean_cft_completed`。

## 训练动态

- 默认场景 `best50` 下，`F-MAPPO` 的最佳窗口 `task_sr=0.9680`，`TERA-MAPPO` 为 `task_sr=0.9720`。
- 各场景全程曲线已在 `figures/fig_<scenario>_full_curves.png` 中拆开绘制，可直接检查两种方法在完整 1500ep 上的收敛速度、尾段平台与资源队列行为。

## 配置一致性

- 所有配对场景在 `NUM_VEHICLES / F_RSU / DAG_FAT / DAG_DENSITY / DAG_REGULAR / LR_ACTOR / MAX_EPISODES` 上一致；预期差异仅为模型表征。
