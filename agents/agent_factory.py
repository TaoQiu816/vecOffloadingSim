from __future__ import annotations

from configs.train_config import TrainConfig as TC
from agents.mappo_agent import MAPPOAgent


def get_agent_class(algo_mode: str | None = None):
    mode = str(algo_mode or getattr(TC, "ALGO_MODE", "mappo")).strip().lower()
    if mode == "ippo":
        from agents.ippo_agent import IPPOAgent

        return IPPOAgent
    return MAPPOAgent


def build_agent(network, device: str = "cpu", algo_mode: str | None = None):
    return get_agent_class(algo_mode)(network, device=device)
