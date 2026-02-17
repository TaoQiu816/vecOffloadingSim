#!/usr/bin/env python3
"""Train entry with StageA (main) parameter overlay."""

from configs.exp.config_stageA_main import apply_stage_a_profile
from train import main


if __name__ == "__main__":
    apply_stage_a_profile()
    main()
