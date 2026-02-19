#!/usr/bin/env python3
"""Train entry with final paper profile overlay."""

from configs.exp.config_final_paper import apply_final_paper_profile
from train import main


if __name__ == "__main__":
    apply_final_paper_profile()
    main()
