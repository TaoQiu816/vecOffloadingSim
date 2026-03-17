"""
Candidate Set Manager

负责每步构建动作候选集合（Local/RSU/V2V），仅选择并排序V2V邻车候选。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class CandidateSetManager:
    def __init__(self, config):
        self.config = config

    def _v2v_gate_reason(self, info: Dict) -> Optional[str]:
        if not getattr(self.config, "V2V_CAND_GATE_ENABLED", False):
            return None
        rem_deadline = max(float(info.get("remaining_deadline", 0.0)), 0.0)
        if rem_deadline <= 0.0:
            return "deadline_nonpositive"
        finish = float(info.get("predicted_finish_time", info.get("total_time", float("inf"))))
        if not np.isfinite(finish):
            return "finish_nonfinite"
        return None

    def _passes_v2v_gate(self, info: Dict) -> bool:
        return self._v2v_gate_reason(info) is None

    def _rsu_gate_reason(self, info: Dict) -> Optional[str]:
        if not getattr(self.config, "RSU_CAND_GATE_ENABLED", False):
            return None
        rem_deadline = max(float(info.get("remaining_deadline", 0.0)), 0.0)
        if rem_deadline <= 0.0:
            return "deadline_nonpositive"
        finish = float(info.get("predicted_finish_time", info.get("total_time", float("inf"))))
        if not np.isfinite(finish):
            return "finish_nonfinite"
        return None

    def _passes_rsu_gate(self, info: Dict) -> bool:
        return self._rsu_gate_reason(info) is None

    def _sort_candidates(self, candidates: List[Dict]) -> List[Dict]:
        sort_by = getattr(self.config, "CANDIDATE_SORT_BY", "t_finish")
        if sort_by == "distance":
            key_fn = lambda x: (x.get("dist", float("inf")), x.get("id", -1))
        elif sort_by == "rate":
            key_fn = lambda x: (-x.get("rate", 0.0), x.get("dist", float("inf")), x.get("id", -1))
        else:
            key_fn = lambda x: (x.get("total_time", float("inf")), x.get("dist", float("inf")), x.get("id", -1))
        return sorted(candidates, key=key_fn)

    def _apply_dynamic_filter(self, sorted_info: List[Dict], max_k: Optional[int] = None) -> List[Dict]:
        if max_k is None:
            max_k = int(getattr(self.config, "V2V_TOP_K", self.config.MAX_NEIGHBORS))
        if not getattr(self.config, "V2V_DYNAMIC_K", False):
            return sorted_info[:max_k] if max_k > 0 else []
        if not sorted_info:
            return []
        min_k = int(getattr(self.config, "V2V_TOP_K_MIN", 1))
        max_k = max(0, min(max_k, self.config.MAX_NEIGHBORS))
        best_time = sorted_info[0].get("total_time", None)
        if best_time is None:
            return sorted_info[:max_k] if max_k > 0 else []
        rel_tol = float(getattr(self.config, "V2V_CANDIDATE_REL_TOL", 0.25))
        abs_tol = float(getattr(self.config, "V2V_CANDIDATE_ABS_TOL", 0.2))
        time_limit = best_time * (1.0 + rel_tol) + abs_tol
        filtered = [info for info in sorted_info if info.get("total_time", float("inf")) <= time_limit]
        if len(filtered) < min_k:
            filtered = sorted_info[:min_k]
        if max_k > 0 and len(filtered) > max_k:
            filtered = filtered[:max_k]
        return filtered

    def build_candidate_set(
        self,
        vehicle,
        v2v_candidates: List[Dict],
        serving_rsu_id: Optional[int],
        rsus_in_range: Optional[List[int]] = None,  # 新增：覆盖范围内的RSU列表
        rsu_candidates: Optional[List[Dict]] = None,
    ) -> Dict:
        max_targets = int(getattr(self.config, "MAX_TARGETS", 2))
        enable_rsu_selection = True
        num_rsu = int(getattr(self.config, "NUM_RSU", 3))
        
        # 计算RSU和V2V的索引边界
        rsu_start_idx = 1
        rsu_end_idx = 1 + num_rsu
        v2v_start_idx = rsu_end_idx
        max_neighbors = max(0, max_targets - v2v_start_idx)
        
        ids = np.full(max_targets, -1, dtype=np.int64)
        types = np.zeros(max_targets, dtype=np.int8)
        mask = np.zeros(max_targets, dtype=bool)

        # Index 0: Local (always available)
        ids[0] = int(vehicle.id)
        types[0] = 1
        mask[0] = True

        rsus_available = set(rsus_in_range) if rsus_in_range else set()
        rsu_info_map = {
            int(info.get("id")): info
            for info in (rsu_candidates or [])
            if info is not None and info.get("id") is not None
        }
        rsu_gate_total = 0
        rsu_gate_blocked = 0
        rsu_gate_reason_counts: Dict[str, int] = {}
        for rsu_id in range(num_rsu):
            idx = rsu_start_idx + rsu_id
            if idx < max_targets:
                ids[idx] = rsu_id
                types[idx] = 2
                mask_val = bool(rsu_id in rsus_available)
                if mask_val:
                    rsu_gate_total += 1
                    rsu_info = rsu_info_map.get(rsu_id)
                    gate_reason = self._rsu_gate_reason(rsu_info) if rsu_info is not None else None
                    if gate_reason is not None:
                        mask_val = False
                        rsu_gate_blocked += 1
                        rsu_gate_reason_counts[gate_reason] = rsu_gate_reason_counts.get(gate_reason, 0) + 1
                mask[idx] = mask_val

        # V2V candidates
        admissible_v2v = [info for info in v2v_candidates if self._passes_v2v_gate(info)]
        sorted_info = self._sort_candidates(admissible_v2v)
        reachable_cnt = len(sorted_info)
        mode = str(getattr(self.config, "CANDIDATE_MODE", "TOPK")).upper()
        if mode == "ALL":
            # ALL means every currently in-range and action-mask-feasible helper.
            # Fixed-length V2V slots are only padding carriers for the observation tensor.
            if reachable_cnt > max_neighbors:
                raise AssertionError(
                    f"ALL mode requires padding capacity for all feasible helpers: "
                    f"reachable_cnt={reachable_cnt}, max_neighbors={max_neighbors}"
                )
            selected_info = sorted_info
        elif mode == "RANDOMK":
            max_k = int(getattr(self.config, "RANDOMK_K", self.config.MAX_NEIGHBORS))
            max_k = max(0, min(max_k, self.config.MAX_NEIGHBORS))
            if max_k <= 0 or not sorted_info:
                selected_info = []
            else:
                k = min(max_k, len(sorted_info))
                idx = np.random.choice(len(sorted_info), size=k, replace=False)
                selected_info = [sorted_info[i] for i in idx]
        else:
            max_k = int(getattr(self.config, "TOPK_K", self.config.MAX_NEIGHBORS))
            selected_info = self._apply_dynamic_filter(sorted_info, max_k=max_k)
        used_ids = set()
        v2v_slots: List[Optional[Dict]] = [None] * max_neighbors
        slot_idx = 0
        for info in selected_info:
            if slot_idx >= max_neighbors:
                break
            cand_id = info.get("id", None)
            if cand_id is None:
                continue
            if cand_id == vehicle.id or cand_id in used_ids:
                continue
            used_ids.add(cand_id)
            target_idx = v2v_start_idx + slot_idx
            if target_idx < max_targets:
                ids[target_idx] = int(cand_id)
                types[target_idx] = 3
                mask[target_idx] = True
                v2v_slots[slot_idx] = info
            slot_idx += 1

        selected_cnt = sum(1 for info in v2v_slots if info is not None)
        dropped_cnt = max(0, reachable_cnt - selected_cnt)
        padded_cnt_v2v = max_neighbors - selected_cnt  # padding空位数
        if getattr(self.config, "DEBUG_CANDIDATE_SET", False):
            info_list = []
            for idx in range(max_targets):
                info_list.append(f"{idx}:{types[idx]}/{ids[idx]}/{int(mask[idx])}")
            print(f"[Debug] candidate_set veh={vehicle.id} serving_rsu={serving_rsu_id} rsus_in_range={rsus_in_range} -> {', '.join(info_list)}")

        return {
            "ids": ids,
            "types": types,
            "mask": mask,
            "v2v_slots": v2v_slots,
            "rsu_start_idx": rsu_start_idx,
            "rsu_end_idx": rsu_end_idx,
            "v2v_start_idx": v2v_start_idx,
            "max_neighbors": max_neighbors,
            "reachable_cnt": reachable_cnt,
            "dropped_cnt": dropped_cnt,
            "feasible_cnt_v2v": selected_cnt,
            "padded_cnt_v2v": padded_cnt_v2v,
            "masked_cnt_total": int(np.sum(mask)),
            "rsu_gate_total": int(rsu_gate_total),
            "rsu_gate_blocked": int(rsu_gate_blocked),
            "rsu_gate_reason_counts": dict(rsu_gate_reason_counts),
        }


__all__ = ["CandidateSetManager"]
