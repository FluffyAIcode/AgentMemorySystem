"""ContextBundle — carries background/situational memory encoding.

Base space B_ctx = R^{d_ctx}. A point in B_ctx is a learned compression of
the session state at the moment of write — who was talking, what task, which
prior turns mattered. Distinct from topic: topic = what the memory is about;
context = the framing in which it was created.

Canonical axis: the "session-mean" direction — a learned attractor toward
the typical session embedding. Used by the Kakeya alignment so context-axis
compression factorizes session-invariant features cleanly from session-specific
ones.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ams_v4.bundles.base import (
    Bundle, RiemannianMetric, FiberConnection, FiberTransporter, GeodesicSolver,
)
from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class ContextEncoder(nn.Module):
    """Encodes (hidden_state, session_summary, prev_turns) →
                 (ctx_base, ctx_fiber, ctx_dirn).

    session_summary: (B, d_LLM) — a running pooled hidden state of the session so far.
    prev_turns: (B, T_prev, d_LLM) — recent turn hidden states, optional attention source.
    """
    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        # arch sketch (v4.2):
        #   attn = AttentionPool(query=hidden, kv=prev_turns)       -> (B, d_LLM)
        #   mixed = Linear(hidden + session_summary + attn)         -> (B, d_ctx)
        #   base  = LN(mixed)                                       -> (B, d_ctx)
        #   fiber = MLP(concat(hidden, base, session_summary_proj)) -> (B, d_F_ctx)
        #   dirn  = normalize(base)                                 -> (B, d_ctx)
        raise NotImplementedError("v4-skel: ContextEncoder.__init__ — lands in v4.2")

    def forward(self, hidden_state: Tensor, session_summary: Tensor,
                prev_turns: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """hidden_state: (B, d_LLM); session_summary: (B, d_LLM);
        prev_turns: (B, T_prev, d_LLM) or None.

        Returns (base, fiber, dirn), shapes (B, d_ctx), (B, d_F_ctx), (B, d_ctx).
        """
        raise NotImplementedError("v4-skel: ContextEncoder.forward — lands in v4.2")


class ContextBundle(Bundle):
    """Fiber bundle with R^{d_ctx} as base, F_ctx as typical fiber."""

    def __init__(self, cfg: Cfg4):
        super().__init__(name="ctx", cfg=cfg)
        self.d_base = cfg.d_ctx
        self.d_fiber = cfg.d_F_ctx
        # v4.1 parts — same shape as TemporalBundle.
        raise NotImplementedError("v4-skel: ContextBundle.__init__ — lands in v4.1/v4.2")

    def canonical_axis(self) -> Tensor:
        raise NotImplementedError("v4-skel: ContextBundle.canonical_axis — lands in v4.2")

    def encode(self, hidden_state: Tensor, *, session_summary: Tensor,
               prev_turns: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("v4-skel: ContextBundle.encode — lands in v4.2")

    def transport(self, fiber_src: Tensor, base_src: Tensor, base_dst: Tensor) -> Tensor:
        raise NotImplementedError("v4-skel: ContextBundle.transport — lands in v4.2")
