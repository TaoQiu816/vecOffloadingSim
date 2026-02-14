#!/usr/bin/env python3
"""Train entry with StageC (trust/risk) parameter overlay."""

from configs.exp.config_stageC_trust import apply_stage_c_profile
from train import main


if __name__ == "__main__":
    apply_stage_c_profile()
    main()

