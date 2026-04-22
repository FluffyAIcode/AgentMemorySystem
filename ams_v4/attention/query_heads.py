"""BundleQueryHeads — three per-bundle projection heads mapping the
decoder's hidden state into each bundle's query space.

One head per bundle. Each head outputs a query vector of the *same dim as
that bundle's fiber space*, so the bundle's keys/values (which are fibers
or fiber-derived) can be attended over directly.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn

from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class BundleQueryHeads(nn.Module):
    """Three linear heads: hidden_state → (q_time, q_topic, q_ctx)."""

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        # v4.4 implementation:
        #   self.q_time  = nn.Linear(cfg.d_LLM, cfg.d_F_time)
        #   self.q_topic = nn.Linear(cfg.d_LLM, cfg.d_F_topic)
        #   self.q_ctx   = nn.Linear(cfg.d_LLM, cfg.d_F_ctx)
        #   Plus LayerNorm on input and per-head output.
        raise NotImplementedError("v4-skel: BundleQueryHeads.__init__ — lands in v4.4")

    def forward(self, hidden_state: Tensor) -> Dict[str, Tensor]:
        """hidden_state: (B, d_LLM) → {"time": (B, d_F_time),
                                       "topic": (B, d_F_topic),
                                       "ctx":   (B, d_F_ctx)}.
        """
        raise NotImplementedError("v4-skel: BundleQueryHeads.forward — lands in v4.4")
