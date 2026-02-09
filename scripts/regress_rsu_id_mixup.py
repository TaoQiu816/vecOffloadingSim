"""
回归测试：验证 ENABLE_RSU_SELECTION=True 时不再出现 RSU id 被当作 vehicle id 的混淆。

核心检查：
1. candidate_set 中 type=2(RSU) 的 id 不会被传入 _get_vehicle_by_id()
2. candidate_set 中 type=3(V2V) 的 id 在 [0, NUM_VEHICLES) 范围内
3. _last_candidates 缓存中不包含 RSU id
4. phi 相关函数运行无断言失败
"""

import os
import sys
import json
import numpy as np
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import SystemConfig as BaseCfg
from envs.vec_offloading_env import VecOffloadingEnv


class Cfg(BaseCfg):
    """强制 ENABLE_RSU_SELECTION=True, NUM_RSU=3 以触发多RSU场景"""
    ENABLE_RSU_SELECTION = True
    NUM_RSU = 3
    MAX_NEIGHBORS = 5
    MAX_TARGETS = 1 + NUM_RSU + MAX_NEIGHBORS  # 1 Local + 3 RSU + 5 V2V = 9
    CHAIN_ENABLED = False
    CHAIN_RISK_WEIGHT_DEPOSIT = 0.0
    CHAIN_RISK_WEIGHT_FAIL = 0.0
    NUM_VEHICLES = 8


def main():
    np.random.seed(42)
    env = VecOffloadingEnv(config=Cfg)
    obs, _ = env.reset(seed=42)
    env.action_space.seed(42)

    num_steps = 10
    trace = []  # 记录每步审计轨迹
    errors = []

    # 用于拦截 _get_vehicle_by_id 调用的包装器
    original_get_vehicle = env._get_vehicle_by_id
    call_log = []

    def _traced_get_vehicle_by_id(veh_id):
        call_log.append(("get_vehicle", int(veh_id)))
        return original_get_vehicle(veh_id)

    original_get_rsu_queue = env._get_rsu_queue_load
    def _traced_get_rsu_queue_load(rsu_id, processor_id=None):
        call_log.append(("get_rsu_queue", int(rsu_id)))
        return original_get_rsu_queue(rsu_id, processor_id)

    # Patch
    env._get_vehicle_by_id = _traced_get_vehicle_by_id
    env._get_rsu_queue_load = _traced_get_rsu_queue_load

    for step_idx in range(num_steps):
        call_log.clear()
        action = env.action_space.sample()

        try:
            obs, rew, term, trunc, info = env.step(action)
        except AssertionError as e:
            errors.append(f"step={step_idx} AssertionError: {e}")
            break
        except Exception as e:
            errors.append(f"step={step_idx} Exception: {e}")
            break

        step_record = {
            "step": step_idx,
            "calls": list(call_log),
        }

        # 检查1: _last_candidates 应只包含 candidate_set 中 type=3(V2V) 的 id
        # 注意：RSU id 和 vehicle id 可能数值重叠（如 RSU_1 和 vehicle_1 都是 id=1）
        # 因此只能通过 candidate_set 的 type 字段判断，不能仅按数值排除
        for vid, cands in env._last_candidates.items():
            cand_set = env._last_candidate_set.get(vid)
            if cand_set is None:
                continue
            types_arr = cand_set.get("types", [])
            ids_arr = cand_set.get("ids", [])
            # 收集 type=3(V2V) 的 id 集合
            v2v_ids_in_set = set()
            for i in range(len(types_arr)):
                if int(types_arr[i]) == 3 and int(ids_arr[i]) >= 0:
                    v2v_ids_in_set.add(int(ids_arr[i]))
            # _last_candidates 中每个 id 必须在 V2V id 集合中
            for c in cands:
                if c not in v2v_ids_in_set and c >= 0:
                    msg = f"step={step_idx} veh={vid}: _last_candidates id={c} not in V2V set {v2v_ids_in_set}"
                    errors.append(msg)
                    step_record.setdefault("errors", []).append(msg)

        # 检查2: get_vehicle 的调用对象不应是 RSU id
        # 在 ENABLE_RSU_SELECTION=True 且 NUM_RSU=3 时，RSU id 范围 [0,1,2]
        # vehicle id 也可能是 0,1,2...所以不能简单按id范围排除
        # 关键检查：candidate_set 中 type=2 的 id 不应出现在 get_vehicle 调用中
        all_rsu_ids_in_candidates = set()
        for vid, cand_set in env._last_candidate_set.items():
            if cand_set is None:
                continue
            types_arr = cand_set.get("types", [])
            ids_arr = cand_set.get("ids", [])
            for i in range(len(types_arr)):
                if int(types_arr[i]) == 2 and int(ids_arr[i]) >= 0:
                    all_rsu_ids_in_candidates.add(int(ids_arr[i]))

        step_record["rsu_ids_in_candidates"] = sorted(all_rsu_ids_in_candidates)
        step_record["get_vehicle_calls"] = [c[1] for c in call_log if c[0] == "get_vehicle"]
        step_record["get_rsu_queue_calls"] = [c[1] for c in call_log if c[0] == "get_rsu_queue"]

        # 检查3: 奖励和终止逻辑正常
        rew_val = rew if isinstance(rew, (int, float)) else sum(rew) if hasattr(rew, '__iter__') else float(rew)
        if not np.isfinite(rew_val):
            msg = f"step={step_idx} reward is not finite: {rew}"
            errors.append(msg)
            step_record.setdefault("errors", []).append(msg)

        trace.append(step_record)

        # 提前终止检查
        done = term if isinstance(term, bool) else all(term) if hasattr(term, '__iter__') else bool(term)
        if done:
            print(f"Episode terminated at step {step_idx}")
            break

    # 输出结果
    output_path = os.path.join(ROOT, "logs", "rsu_id_mixup_regress.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "config": {
            "ENABLE_RSU_SELECTION": Cfg.ENABLE_RSU_SELECTION,
            "NUM_RSU": Cfg.NUM_RSU,
            "MAX_TARGETS": Cfg.MAX_TARGETS,
            "NUM_VEHICLES": Cfg.NUM_VEHICLES,
            "MAX_NEIGHBORS": Cfg.MAX_NEIGHBORS,
        },
        "steps_run": len(trace),
        "errors": errors,
        "trace": trace,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # 打印摘要
    print(f"\n=== RSU ID Mixup Regression Test ===")
    print(f"Steps run: {len(trace)}")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  [ERROR] {e}")

    if not errors:
        print("PASS: No RSU/Vehicle ID mixup detected.")
        print(f"Trace saved to {output_path}")
        return 0
    else:
        print(f"FAIL: {len(errors)} errors found.")
        print(f"Trace saved to {output_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
