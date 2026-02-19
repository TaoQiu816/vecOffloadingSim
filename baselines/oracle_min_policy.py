"""
Oracle-Min baseline.

Purpose:
- Uses the same action interface as other baselines (target + power).
- Selects the target with minimum EFT-style estimated completion time.
- Uses max transmit power to approximate an optimistic upper bound.
"""

from baselines.eft_policy import EFTPolicy


class OracleMinPolicy(EFTPolicy):
    """Optimistic min-completion-time baseline (upper-bound style)."""

    def __init__(self, env):
        super().__init__(env, target_snr_db=10.0)

    def _min_power_a(self, distance: float, link_type: str = "V2V") -> float:
        _ = (distance, link_type)
        return 1.0

