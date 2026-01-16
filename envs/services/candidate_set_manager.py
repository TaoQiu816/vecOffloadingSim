"""
Candidate Set Manager

负责每步构建动作候选集合（Local/RSU/V2V），仅选择并排序V2V邻车候选。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class CandidateSetManager:
    def __init__(self, config):
        self.config = config

    def _sort_candidates(self, candidates: List[Dict]) -> List[Dict]:
        sort_by = getattr(self.config, "CANDIDATE_SORT_BY", "t_finish")
        if sort_by == "distance":
            key_fn = lambda x: (x.get("dist", float("inf")), x.get("id", -1))
        elif sort_by == "rate":
            key_fn = lambda x: (-x.get("rate", 0.0), x.get("dist", float("inf")), x.get("id", -1))
        else:
            key_fn = lambda x: (x.get("total_time", float("inf")), x.get("dist", float("inf")), x.get("id", -1))
        return sorted(candidates, key=key_fn)

    def _apply_dynamic_filter(self, sorted_info: List[Dict]) -> List[Dict]:
        if not getattr(self.config, "V2V_DYNAMIC_K", False):
            return sorted_info
        if not sorted_info:
            return []
        min_k = int(getattr(self.config, "V2V_TOP_K_MIN", 1))
        max_k = int(getattr(self.config, "V2V_TOP_K", self.config.MAX_NEIGHBORS))
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
    ) -> Dict:
        max_targets = int(getattr(self.config, "MAX_TARGETS", 2))
        max_neighbors = max(0, max_targets - 2)
        ids = np.full(max_targets, -1, dtype=np.int64)
        types = np.zeros(max_targets, dtype=np.int8)
        mask = np.zeros(max_targets, dtype=bool)

        # Index 0: Local
        ids[0] = int(vehicle.id)
        types[0] = 1
        mask[0] = True

        # Index 1: Serving RSU
        types[1] = 2
        if serving_rsu_id is not None:
            ids[1] = int(serving_rsu_id)
            mask[1] = True

        sorted_info = self._sort_candidates(v2v_candidates)
        selected_info = self._apply_dynamic_filter(sorted_info)

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
            ids[2 + slot_idx] = int(cand_id)
            types[2 + slot_idx] = 3
            mask[2 + slot_idx] = True
            v2v_slots[slot_idx] = info
            slot_idx += 1

        if getattr(self.config, "DEBUG_CANDIDATE_SET", False):
            info_list = []
            for idx in range(max_targets):
                info_list.append(f"{idx}:{types[idx]}/{ids[idx]}/{int(mask[idx])}")
            print(f"[Debug] candidate_set veh={vehicle.id} serving_rsu={serving_rsu_id} -> {', '.join(info_list)}")

        return {
            "ids": ids,
            "types": types,
            "mask": mask,
            "v2v_slots": v2v_slots,
        }


__all__ = ["CandidateSetManager"]
