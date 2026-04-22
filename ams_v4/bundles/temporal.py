"""TemporalBundle — carries time-axis memory encoding.

Base space B_time = R^{d_time}. A point in B_time is a learned embedding of
(absolute wall-clock time, recency = now - last_access, write-count).

Canonical axis: a learned unit direction in B_time (inherited from Bundle).
It is trained (in v4.6) to align with the "pure recency" direction — the
direction along which projection monotonically tracks `ts`. v4.2 just lets
it be random; training lands in v4.6.
"""
from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ams_v4.bundles.base import Bundle
from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


def _fourier_features(scalars: Tensor, n_features: int, max_period: float = 1e4) -> Tensor:
    """Sinusoidal Fourier features with exponentially-spaced frequencies.

    scalars: (B, n_scalars) — each scalar gets its own set of features.
    Returns: (B, n_scalars * n_features) with n_features = 2 * k, ...cos/sin pairs.

    Same trick as NeRF / Transformer positional encoding; prevents the MLP
    from having to learn time-scale invariance from scratch.
    """
    assert n_features % 2 == 0, "n_features must be even (cos/sin pairs)"
    B, n_scalars = scalars.shape
    k = n_features // 2
    freqs = torch.exp(
        torch.linspace(0, math.log(max_period), k, device=scalars.device)
    )  # (k,)
    # (B, n_scalars, 1) * (k,) -> (B, n_scalars, k)
    args = scalars.unsqueeze(-1) * freqs.view(1, 1, -1)
    out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, n_scalars, 2k)
    return out.reshape(B, n_scalars * n_features)


class TimeEncoder(nn.Module):
    """Encodes (hidden_state, time_scalars, surprise) → (base, fiber, dirn).

    time_scalars: (B, 3) = (absolute_ts, recency, cnt).
    surprise:     (B,) or (B, 1).
    """

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self.d_time = cfg.d_time
        self.d_F_time = cfg.d_F_time
        self.d_LLM = cfg.d_LLM

        # Fourier features: 3 scalars x n_feat dims each. Pick n_feat so the
        # fourier block is comparable in size to the hidden projection.
        self.n_fourier_per_scalar = max(4, 2 * ((cfg.d_time + 1) // 2))
        fourier_dim = 3 * self.n_fourier_per_scalar

        hidden = max(4 * cfg.d_time, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, cfg.d_time),
        )
        self.hidden_proj = nn.Linear(cfg.d_LLM, cfg.d_time)
        self.base_ln = nn.LayerNorm(cfg.d_time)

        fiber_hidden = max(4 * cfg.d_F_time, 64)
        self.fiber_mlp = nn.Sequential(
            nn.Linear(cfg.d_LLM + cfg.d_time + 1, fiber_hidden), nn.SiLU(),
            nn.Linear(fiber_hidden, cfg.d_F_time),
        )

    def forward(self, hidden_state: Tensor, time_scalars: Tensor,
                surprise: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """hidden_state: (B, d_LLM); time_scalars: (B, 3); surprise: (B,) or (B, 1).

        Returns (base, fiber, dirn) of shapes (B, d_time), (B, d_F_time), (B, d_time).
        """
        assert hidden_state.dim() == 2 and hidden_state.shape[-1] == self.d_LLM
        assert time_scalars.dim() == 2 and time_scalars.shape[-1] == 3
        if surprise.dim() == 1:
            surprise = surprise.unsqueeze(-1)
        assert surprise.dim() == 2 and surprise.shape[-1] == 1

        ff = _fourier_features(time_scalars, self.n_fourier_per_scalar)
        time_embed = self.time_mlp(ff)                              # (B, d_time)
        base = self.base_ln(time_embed + self.hidden_proj(hidden_state))  # (B, d_time)
        fiber = self.fiber_mlp(torch.cat([hidden_state, base, surprise], dim=-1))  # (B, d_F_time)
        dirn = F.normalize(base, dim=-1, eps=1e-8)                  # (B, d_time)
        return base, fiber, dirn


class TemporalBundle(Bundle):
    """Fiber bundle with B_time as base, F_time as typical fiber."""

    def __init__(self, cfg: Cfg4):
        super().__init__(name="time", cfg=cfg, d_base=cfg.d_time, d_fiber=cfg.d_F_time)
        self.encoder = TimeEncoder(cfg)

    def encode(self, hidden_state: Tensor, *, time_scalars: Tensor,
               surprise: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.encoder(hidden_state, time_scalars, surprise)
