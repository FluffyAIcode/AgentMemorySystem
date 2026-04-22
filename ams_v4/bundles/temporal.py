"""TemporalBundle — carries time-axis memory encoding.

Base space B_time = R^{d_time}. A point in B_time is a learned embedding of
(absolute wall-clock time, recency = now - last_access, write-count).

Canonical axis: the pure-recency direction (the direction in B_time along
which "more recent" monotonically increases). Learned during v4.1 training
to maximize correlation between projection onto this axis and `ts`.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from ams_v4.bundles.base import (
    Bundle, RiemannianMetric, FiberConnection, FiberTransporter, GeodesicSolver,
)
from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class TimeEncoder(nn.Module):
    """Encodes (hidden_state, time_scalars) → (time_base, time_fiber, time_dirn).

    time_scalars: (B, 3) = (absolute_ts, recency = ts - last_access, cnt).
    """
    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        # arch sketch (v4.2):
        #   time_embed = MLP(sinusoidal_encode(time_scalars))    -> (B, d_time)
        #   base       = LN(time_embed + hidden_proj(hidden))    -> (B, d_time)
        #   fiber      = MLP(concat(hidden, base, surprise))     -> (B, d_F_time)
        #   dirn       = normalize(base)                         -> (B, d_time)
        raise NotImplementedError("v4-skel: TimeEncoder.__init__ — lands in v4.2")

    def forward(self, hidden_state: Tensor, time_scalars: Tensor,
                surprise: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """hidden_state: (B, d_LLM); time_scalars: (B, 3); surprise: (B, 1).

        Returns (base, fiber, dirn), shapes (B, d_time), (B, d_F_time), (B, d_time).
        """
        raise NotImplementedError("v4-skel: TimeEncoder.forward — lands in v4.2")


class TemporalBundle(Bundle):
    """Fiber bundle with B_time as base, F_time as typical fiber."""

    def __init__(self, cfg: Cfg4):
        super().__init__(name="time", cfg=cfg)
        self.d_base = cfg.d_time
        self.d_fiber = cfg.d_F_time
        # instantiated in v4.1:
        #   self.metric = RiemannianMetric(d_base=cfg.d_time)
        #   self.conn   = FiberConnection(cfg.d_time, cfg.d_F_time, self.metric)
        #   self.trans  = FiberTransporter(self.conn, cfg)
        #   self.solver = GeodesicSolver(self.metric, cfg)
        # The canonical-axis parameter (learned):
        #   self._axis = nn.Parameter(torch.randn(cfg.d_time))
        raise NotImplementedError("v4-skel: TemporalBundle.__init__ — lands in v4.1/v4.2")

    def canonical_axis(self) -> Tensor:
        raise NotImplementedError("v4-skel: TemporalBundle.canonical_axis — lands in v4.2")

    def encode(self, hidden_state: Tensor, *, time_scalars: Tensor,
               surprise: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("v4-skel: TemporalBundle.encode — lands in v4.2")

    def transport(self, fiber_src: Tensor, base_src: Tensor, base_dst: Tensor) -> Tensor:
        raise NotImplementedError("v4-skel: TemporalBundle.transport — lands in v4.2")
