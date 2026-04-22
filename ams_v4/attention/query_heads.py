"""BundleQueryHeads — three per-bundle projections of the decoder's hidden
state into each bundle's fiber space.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn

from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class BundleQueryHeads(nn.Module):
    """Three linear heads: hidden_state → {time, topic, ctx} queries."""

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.d_LLM)
        self.q_time = nn.Linear(cfg.d_LLM, cfg.d_F_time)
        self.q_topic = nn.Linear(cfg.d_LLM, cfg.d_F_topic)
        self.q_ctx = nn.Linear(cfg.d_LLM, cfg.d_F_ctx)

    def forward(self, hidden_state: Tensor) -> Dict[str, Tensor]:
        assert hidden_state.dim() == 2 and hidden_state.shape[-1] == self.cfg.d_LLM
        h = self.ln(hidden_state)
        return {
            "time":  self.q_time(h),
            "topic": self.q_topic(h),
            "ctx":   self.q_ctx(h),
        }
