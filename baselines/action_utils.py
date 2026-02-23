"""
Baseline action helpers for the stage-3 autoregressive DAG scheduler.

These helpers let classic baselines (which historically only chose target/power)
emit a valid `subtask` field without rewriting each heuristic's core logic.
"""

from typing import Dict, Optional
import numpy as np


def choose_subtask_from_obs(obs: Dict, preferred: Optional[int] = None) -> int:
    """
    Choose a valid subtask index from observation masks.

    Priority:
    1) explicit preferred index if valid in subtask_mask
    2) env-provided subtask_index if valid in subtask_mask
    3) first valid index in subtask_mask
    4) first valid index in node_valid_mask (fallback for no-task states)
    5) 0
    """
    subtask_mask = np.asarray(obs.get("subtask_mask", obs.get("task_mask", [])), dtype=np.float32).reshape(-1)
    node_valid_mask = np.asarray(obs.get("node_valid_mask", obs.get("task_mask", [])), dtype=np.float32).reshape(-1)

    valid_subtasks = np.where(subtask_mask > 0)[0]
    if preferred is not None:
        try:
            p = int(preferred)
            if 0 <= p < len(subtask_mask) and subtask_mask[p] > 0:
                return p
        except Exception:
            pass

    env_idx = obs.get("subtask_index", None)
    try:
        if env_idx is not None:
            e = int(env_idx)
            if 0 <= e < len(subtask_mask) and subtask_mask[e] > 0:
                return e
    except Exception:
        pass

    if valid_subtasks.size > 0:
        return int(valid_subtasks[0])

    valid_nodes = np.where(node_valid_mask > 0)[0]
    if valid_nodes.size > 0:
        return int(valid_nodes[0])
    return 0


def attach_subtask(obs: Dict, action: Dict, preferred: Optional[int] = None) -> Dict:
    """Return a shallow-copied action dict with a valid `subtask` field attached."""
    out = dict(action)
    out["subtask"] = int(choose_subtask_from_obs(obs, preferred=preferred))
    return out

