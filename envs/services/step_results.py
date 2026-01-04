from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from envs.vec_offloading_env import TransferJob, ComputeJob  # type: ignore
else:
    TransferJob = Any  # for runtime to avoid circular import
    ComputeJob = Any


@dataclass
class CommStepResult:
    completed_jobs: List[TransferJob] = field(default_factory=list)
    energy_delta_cost: Dict[int, float] = field(default_factory=dict)  # INPUT计成本
    energy_delta_record_edge: Dict[int, float] = field(default_factory=dict)  # EDGE仅记录
    time_used_v2i: Dict[Tuple[str, int], float] = field(default_factory=dict)
    time_used_v2v: Dict[Tuple[str, int], float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CpuStepResult:
    completed_jobs: List[ComputeJob] = field(default_factory=list)
    energy_delta_cost_local: Dict[int, float] = field(default_factory=dict)
    cycles_done_local: Dict[int, float] = field(default_factory=dict)
    cycles_done_rsu_record: Dict[int, float] = field(default_factory=dict)
    time_used_by_proc: Dict[Any, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
