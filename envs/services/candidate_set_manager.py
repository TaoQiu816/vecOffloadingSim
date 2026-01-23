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
        rsus_in_range: Optional[List[int]] = None,  # 新增：覆盖范围内的RSU列表
    ) -> Dict:
        max_targets = int(getattr(self.config, "MAX_TARGETS", 2))
        enable_rsu_selection = getattr(self.config, "ENABLE_RSU_SELECTION", False)
        num_rsu = int(getattr(self.config, "NUM_RSU", 3))
        
        # 计算RSU和V2V的索引边界
        rsu_start_idx = 1
        rsu_end_idx = (1 + num_rsu) if enable_rsu_selection else 2
        v2v_start_idx = rsu_end_idx
        max_neighbors = max(0, max_targets - v2v_start_idx)
        
        ids = np.full(max_targets, -1, dtype=np.int64)
        types = np.zeros(max_targets, dtype=np.int8)
        mask = np.zeros(max_targets, dtype=bool)

        # Index 0: Local (always available)
        ids[0] = int(vehicle.id)
        types[0] = 1
        mask[0] = True

        if enable_rsu_selection:
            # 新模式：每个RSU作为独立选项，仅mask覆盖范围内的RSU
            rsus_available = set(rsus_in_range) if rsus_in_range else set()
            for rsu_id in range(num_rsu):
                idx = rsu_start_idx + rsu_id  # index 1,2,3 -> RSU_0,1,2
                if idx < max_targets:
                    ids[idx] = rsu_id
                    types[idx] = 2
                    mask[idx] = (rsu_id in rsus_available)
        else:
            # 旧模式：单一RSU选项（由env选择serving RSU）
            types[1] = 2
            if serving_rsu_id is not None:
                ids[1] = int(serving_rsu_id)
                mask[1] = True

        # V2V candidates
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
            target_idx = v2v_start_idx + slot_idx
            if target_idx < max_targets:
                ids[target_idx] = int(cand_id)
                types[target_idx] = 3
                mask[target_idx] = True
                v2v_slots[slot_idx] = info
            slot_idx += 1

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
        }


__all__ = ["CandidateSetManager"]
