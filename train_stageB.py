#!/usr/bin/env python3
"""Train entry with StageB (interference) parameter overlay."""

from configs.exp.config_stageB_interf import apply_stage_b_profile
from train import main


if __name__ == "__main__":
    apply_stage_b_profile()
    main()

