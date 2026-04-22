"""CrossBundleAttention — three per-bundle attentions + slot-concat to (L_mem, d_LLM)."""
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn

from ams_v4.attention.query_heads import BundleQueryHeads
from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import MemEntry
from ams_v4.core.types import Tensor


class CrossBundleAttention(nn.Module):
    """Three per-bundle multi-head attentions + per-slot lifts to d_LLM."""

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self.query_heads = BundleQueryHeads(cfg)
        self.attn_time = nn.MultiheadAttention(
            cfg.d_F_time, cfg.n_heads_time, batch_first=True,
        )
        self.attn_topic = nn.MultiheadAttention(
            cfg.d_F_topic, cfg.n_heads_topic, batch_first=True,
        )
        self.attn_ctx = nn.MultiheadAttention(
            cfg.d_F_ctx, cfg.n_heads_ctx, batch_first=True,
        )
        self.lift_time = nn.ModuleList([
            nn.Linear(cfg.d_F_time, cfg.d_LLM) for _ in range(cfg.prefix_slots_time)
        ])
        self.lift_topic = nn.ModuleList([
            nn.Linear(cfg.d_F_topic, cfg.d_LLM) for _ in range(cfg.prefix_slots_topic)
        ])
        self.lift_ctx = nn.ModuleList([
            nn.Linear(cfg.d_F_ctx, cfg.d_LLM) for _ in range(cfg.prefix_slots_ctx)
        ])
        self.prefix_ln = nn.LayerNorm(cfg.d_LLM)

    def forward(self, hidden_state: Tensor, entries: List[MemEntry],
                mem_mask: Optional[Tensor] = None) -> Tensor:
        """hidden_state: (B, d_LLM); entries: list of MemEntry (length M);
        mem_mask:   (B, M) bool (True = ignore this mem) or None.

        Returns: prefix of shape (B, L_mem, d_LLM).
        """
        assert hidden_state.dim() == 2
        assert hidden_state.shape[-1] == self.cfg.d_LLM
        assert len(entries) >= 1, "CrossBundleAttention requires ≥ 1 memory entry"
        B = hidden_state.shape[0]
        M = len(entries)
        dev = hidden_state.device
        dtype = hidden_state.dtype

        q = self.query_heads(hidden_state)  # three (B, d_F_*) queries

        def _stack_fibers(attr: str, d_F: int) -> Tensor:
            stacked = torch.stack([getattr(e, attr) for e in entries], dim=0)  # (M, d_F_*)
            stacked = stacked.to(device=dev, dtype=dtype)
            return stacked.unsqueeze(0).expand(B, M, d_F)

        K_time = V_time = _stack_fibers("time_fiber", self.cfg.d_F_time)
        K_topic = V_topic = _stack_fibers("topic_fiber", self.cfg.d_F_topic)
        K_ctx = V_ctx = _stack_fibers("ctx_fiber", self.cfg.d_F_ctx)

        # If mem_mask provided, it is (B, M) with True = pad. Otherwise None.
        out_time, _ = self.attn_time(
            q["time"].unsqueeze(1), K_time, V_time, key_padding_mask=mem_mask,
        )
        out_topic, _ = self.attn_topic(
            q["topic"].unsqueeze(1), K_topic, V_topic, key_padding_mask=mem_mask,
        )
        out_ctx, _ = self.attn_ctx(
            q["ctx"].unsqueeze(1), K_ctx, V_ctx, key_padding_mask=mem_mask,
        )

        out_time = out_time.squeeze(1)    # (B, d_F_time)
        out_topic = out_topic.squeeze(1)  # (B, d_F_topic)
        out_ctx = out_ctx.squeeze(1)      # (B, d_F_ctx)

        # Lift to (B, prefix_slots_*, d_LLM) via per-slot Linears
        slots_time = torch.stack(
            [lh(out_time) for lh in self.lift_time], dim=1,
        )
        slots_topic = torch.stack(
            [lh(out_topic) for lh in self.lift_topic], dim=1,
        )
        slots_ctx = torch.stack(
            [lh(out_ctx) for lh in self.lift_ctx], dim=1,
        )

        prefix = torch.cat([slots_time, slots_topic, slots_ctx], dim=1)
        # Post-attention layer norm for decoder stability
        prefix = self.prefix_ln(prefix)
        assert prefix.shape == (B, self.cfg.L_mem, self.cfg.d_LLM), \
            f"prefix shape invariant: got {tuple(prefix.shape)}"
        return prefix
