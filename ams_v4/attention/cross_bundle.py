"""CrossBundleAttention — the attention mechanism that forms the context window.

Pulls three per-bundle attention results and combines them into a prefix
that is delivered into the backbone's forward pass. This is the §1.5
component of the abstract architecture.

Contract (§6 invariant 6): output shape = (effective_prefix_slots, d_LLM),
where effective_prefix_slots = Cfg4.L_mem, split as
  Cfg4.prefix_slots_time + Cfg4.prefix_slots_topic + Cfg4.prefix_slots_ctx
  == Cfg4.L_mem.

Attention strategy: one attention *per bundle*, not a single mixed-bundle
attention. This keeps the per-bundle signal clean (topic attention does not
get distracted by temporal fibers, etc.) and lets the bundles specialize.
The combination is concatenative across slots, not additive in a single slot.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ams_v4.attention.query_heads import BundleQueryHeads
from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import MemEntry
from ams_v4.core.types import Tensor


class CrossBundleAttention(nn.Module):
    """Three per-bundle multi-head attentions + a concatenative output projection."""

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        # v4.4 implementation:
        #   self.query_heads = BundleQueryHeads(cfg)
        #   self.attn_time  = nn.MultiheadAttention(cfg.d_F_time,  cfg.n_heads_time,  batch_first=True)
        #   self.attn_topic = nn.MultiheadAttention(cfg.d_F_topic, cfg.n_heads_topic, batch_first=True)
        #   self.attn_ctx   = nn.MultiheadAttention(cfg.d_F_ctx,   cfg.n_heads_ctx,   batch_first=True)
        #
        #   # Per-slot lift heads: each slot is its own learned linear lift from
        #   # the bundle's fiber dim to d_LLM. (prefix_slots_time × d_F_time → d_LLM per slot.)
        #   self.lift_time  = nn.ModuleList([nn.Linear(cfg.d_F_time,  cfg.d_LLM)
        #                                    for _ in range(cfg.prefix_slots_time)])
        #   self.lift_topic = nn.ModuleList([nn.Linear(cfg.d_F_topic, cfg.d_LLM)
        #                                    for _ in range(cfg.prefix_slots_topic)])
        #   self.lift_ctx   = nn.ModuleList([nn.Linear(cfg.d_F_ctx,   cfg.d_LLM)
        #                                    for _ in range(cfg.prefix_slots_ctx)])
        #
        #   # LayerNorm on the final prefix for stability when injected into the backbone.
        #   self.prefix_ln = nn.LayerNorm(cfg.d_LLM)
        raise NotImplementedError("v4-skel: CrossBundleAttention.__init__ — lands in v4.4")

    def forward(self, hidden_state: Tensor, entries: List[MemEntry],
                mem_mask: Optional[Tensor] = None) -> Tensor:
        """Produce a prefix tensor.

        hidden_state:  (B, d_LLM)      — current query hidden state
        entries:       list of MemEntry — memories to attend over (length M)
        mem_mask:      (B, M) bool      — optional key-padding mask

        Returns: prefix of shape (B, L_mem, d_LLM).

        Pipeline:
          1. q = BundleQueryHeads(hidden_state) → three per-bundle queries.
          2. For each bundle:
             a. Stack the bundle's fibers across entries → K = V = (B, M, d_F_bundle)
             b. out_bundle = attn(q_bundle, K, V)     (B, d_F_bundle)
          3. For each bundle, run out_bundle through its prefix_slots_bundle
             lift heads → (B, prefix_slots_bundle, d_LLM).
          4. Concatenate across bundles along the slot dim (prefix_slots_time
             + prefix_slots_topic + prefix_slots_ctx == L_mem).
          5. prefix_ln(result).
        """
        raise NotImplementedError("v4-skel: CrossBundleAttention.forward — lands in v4.4")
