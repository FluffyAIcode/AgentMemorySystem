"""ContextBundle — carries background/situational memory encoding.

Base space B_ctx = R^{d_ctx}. Distinct from topic: topic = what the memory
is about; context = the framing in which it was created (session state,
task framing, recent conversation).
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ams_v4.bundles.base import Bundle
from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class ContextEncoder(nn.Module):
    """Encodes (hidden_state, session_summary, prev_turns) → (base, fiber, dirn).

    session_summary: (B, d_LLM) — pooled running hidden state of the session.
    prev_turns: optional (B, T_prev, d_LLM) — hidden states of recent turns.
    """

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self.d_ctx = cfg.d_ctx
        self.d_F_ctx = cfg.d_F_ctx
        self.d_LLM = cfg.d_LLM

        # Attention pool over prev_turns: single-head, dim d_attn
        self.d_attn = max(cfg.d_ctx * 4, 32)
        self.q_proj = nn.Linear(cfg.d_LLM, self.d_attn)
        self.k_proj = nn.Linear(cfg.d_LLM, self.d_attn)
        self.v_proj = nn.Linear(cfg.d_LLM, self.d_attn)
        self.attn_scale = 1.0 / math.sqrt(self.d_attn)

        hidden = max(4 * cfg.d_ctx, 64)
        self.mix_mlp = nn.Sequential(
            nn.Linear(cfg.d_LLM + cfg.d_LLM + self.d_attn, hidden), nn.SiLU(),
            nn.Linear(hidden, cfg.d_ctx),
        )
        self.base_ln = nn.LayerNorm(cfg.d_ctx)

        fiber_hidden = max(4 * cfg.d_F_ctx, 96)
        self.fiber_mlp = nn.Sequential(
            nn.Linear(cfg.d_LLM + cfg.d_ctx + cfg.d_LLM, fiber_hidden), nn.SiLU(),
            nn.Linear(fiber_hidden, cfg.d_F_ctx),
        )

    def _attention_pool(self, hidden_state: Tensor, prev_turns: Optional[Tensor]) -> Tensor:
        """Returns (B, d_attn). Zeros if prev_turns is None or empty."""
        B = hidden_state.shape[0]
        if prev_turns is None or prev_turns.shape[1] == 0:
            return torch.zeros(B, self.d_attn, device=hidden_state.device,
                               dtype=hidden_state.dtype)
        q = self.q_proj(hidden_state).unsqueeze(1)           # (B, 1, d_attn)
        k = self.k_proj(prev_turns)                          # (B, T_prev, d_attn)
        v = self.v_proj(prev_turns)                          # (B, T_prev, d_attn)
        scores = (q @ k.transpose(-1, -2)) * self.attn_scale  # (B, 1, T_prev)
        w = F.softmax(scores, dim=-1)
        out = (w @ v).squeeze(1)                              # (B, d_attn)
        return out

    def forward(self, hidden_state: Tensor, session_summary: Tensor,
                prev_turns: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """hidden_state: (B, d_LLM); session_summary: (B, d_LLM);
        prev_turns: (B, T_prev, d_LLM) or None.

        Returns (base, fiber, dirn) of shapes (B, d_ctx), (B, d_F_ctx), (B, d_ctx).
        """
        assert hidden_state.dim() == 2 and hidden_state.shape[-1] == self.d_LLM
        assert session_summary.dim() == 2 and session_summary.shape[-1] == self.d_LLM
        if prev_turns is not None:
            assert prev_turns.dim() == 3 and prev_turns.shape[-1] == self.d_LLM
            assert prev_turns.shape[0] == hidden_state.shape[0]

        attn = self._attention_pool(hidden_state, prev_turns)
        mixed = self.mix_mlp(torch.cat([hidden_state, session_summary, attn], dim=-1))
        base = self.base_ln(mixed)
        fiber = self.fiber_mlp(torch.cat([hidden_state, base, session_summary], dim=-1))
        dirn = F.normalize(base, dim=-1, eps=1e-8)
        return base, fiber, dirn


class ContextBundle(Bundle):
    """Fiber bundle with R^{d_ctx} as base."""

    def __init__(self, cfg: Cfg4):
        super().__init__(name="ctx", cfg=cfg, d_base=cfg.d_ctx, d_fiber=cfg.d_F_ctx)
        self.encoder = ContextEncoder(cfg)

    def encode(self, hidden_state: Tensor, *, session_summary: Tensor,
               prev_turns: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        return self.encoder(hidden_state, session_summary, prev_turns)
