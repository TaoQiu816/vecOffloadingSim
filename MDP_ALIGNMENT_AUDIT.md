# MDP Alignment Audit

## Purpose

This document records the strict alignment check for the current IoV/VEC simulator.
The standard formulation used here is:

- the policy chooses the executable subtask
- then chooses the execution target
- then chooses transmission power for remote execution

The focus is model-definition correctness, not parameter tuning.

## Confirmed Structural Problems

### 1. Observation-action mismatch existed in the previous mainline

Evidence:

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/envs/vec_offloading_env.py`
  - `_get_obs()` previously used `selected_subtask_idx = v.task_dag.get_top_priority_task()`
  - task-dependent quantities were then built from this environment-selected subtask
- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/models/offloading_policy.py`
  - the policy still sampled an independent subtask action

Why this is invalid:

- observation and resource candidates were prepared for one subtask
- the policy could execute another subtask
- therefore the action semantics and observation semantics were inconsistent

### 2. Target-side resource encoding was task-bound before the policy chose the task

Evidence:

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/envs/vec_offloading_env.py`
  - Local/RSU/V2V `resource_raw`
  - V2V candidate ranking
  - finish estimates
  were previously tied to a preselected task

Why this is invalid:

- if the policy selects the subtask, target-related candidate features must not be preconditioned on an environment-selected subtask

### 3. Observation-phase best-mode/cost statistics were not the correct ground truth

Evidence:

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/envs/vec_offloading_env.py`
  - `best_mode_*` and `cost_*` were accumulated during `_get_obs()`
- but the actual subtask is only determined after action planning

Why this is invalid:

- for standard modeling, oracle comparison must be computed from the actually executed subtask, not from an observation-phase placeholder task

## Implemented Fixes

### A. Restore standard task-selection semantics

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/models/offloading_policy.py`
  - policy now always samples subtask actions
  - temporary compatibility logic that reused environment-provided `subtask_index` as the action was removed

### B. Make observation-side resource encoding task-agnostic

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/envs/vec_offloading_env.py`
  - `_get_obs()` no longer builds candidate ranking and `resource_raw` from an environment-preselected task
  - resource features now encode current resource state:
    - queue/load
    - current expected rate
    - distance/contact
    - current state-based finish proxy

Task-specific legality and realized cost are still handled in `step()` after the policy chooses the subtask.

### C. Use executed-task oracle records as the only source of mode/cost diagnostics

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/envs/vec_offloading_env.py`
  - episode-level `best_mode_*`
  - `v2v_beats_rsu_rate`
  - mean cost statistics
  are now derived from `_episode_oracle_records`

This is the correct source because these records are computed after the action plan fixes the executed subtask.

### D. Keep value estimation aligned with sampled subtask

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/models/offloading_policy.py`
  - `get_action_and_value()` now refreshes `values_env` in the second forward pass under the sampled subtask
  - `evaluate_actions()` likewise refreshes `values` under the provided subtask action

## Important Remaining Issue

### Reward objective is still not aligned with snapshot completion-cost oracle

Evidence:

- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/configs/config.py`
  - default `UNIFIED_MAIN_REWARD_MODE = "margin_term_illegal"`
- `/Users/qiutao/研/毕设/毕设/vecOffloadingSim/envs/vec_offloading_env.py`
  - default main reward excludes `r_time` from the optimized objective

Implication:

- snapshot oracle compares latency-like completion cost
- PPO still mainly optimizes terminal success and margin

This is a real remaining issue, but it is separate from the MDP alignment fixes above.

## Deleted Redundant Logic

- removed the temporary branch that let the policy bypass subtask sampling
- removed obsolete observation-phase `best_mode_*` / `cost_*` accumulators as diagnostic ground truth

## Next Required Check

After these structural fixes, the next strict check should be:

1. run a mid-length training job on the aligned code
2. compare executed-task oracle mode distribution vs chosen mode distribution
3. compare oracle match and regret
4. only then decide whether the remaining bottleneck is reward-objective mismatch or learning instability
