"""TopicBundle — carries topic-axis memory encoding.

Base space B_topic = S^{d_topic - 1}. Closed-form geodesics (great-circle)
instead of GeodesicSolver.
"""
from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ams_v4.bundles.base import (
    Bundle, RiemannianMetric, FiberConnection, FiberTransporter, GeodesicSolver,
)
from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


def _idf_weighted_centroid(token_ids: Sequence[int], wte_normed: Tensor,
                           idf: Optional[Tensor] = None,
                           idf_floor: float = 0.1) -> Tensor:
    """Compute the IDF-weighted mean of `wte_normed[token_ids]`.

    wte_normed: (V, d_LLM) L2-normalized rows.
    idf: optional (V,) corpus-derived IDF weights. If None, uniform weighting.

    Returns: (d_LLM,). If token_ids is empty or all-OOV, returns zeros.
    """
    V, d = wte_normed.shape
    valid = [t for t in token_ids if 0 <= int(t) < V]
    if not valid:
        return torch.zeros(d, device=wte_normed.device, dtype=wte_normed.dtype)
    ids = torch.tensor(valid, device=wte_normed.device, dtype=torch.long)
    vecs = wte_normed[ids]  # (L, d)
    if idf is not None:
        w = idf[ids].clamp(min=idf_floor)
    else:
        w = torch.ones(ids.shape[0], device=wte_normed.device, dtype=vecs.dtype)
    w = w.unsqueeze(-1)
    return (vecs * w).sum(dim=0) / w.sum().clamp(min=1e-8)


class TopicEncoder(nn.Module):
    """Encodes (hidden_state, content_token_ids, wte_normed) → (base, fiber, dirn).

    content_token_ids is either a flat list (single-batch case) or a list of
    lists (ragged batch). base lives on S^{d_topic-1} by construction.
    """

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self.d_topic = cfg.d_topic
        self.d_F_topic = cfg.d_F_topic
        self.d_LLM = cfg.d_LLM

        hidden = max(4 * cfg.d_topic, 64)
        self.down_project = nn.Sequential(
            nn.Linear(cfg.d_LLM, hidden), nn.SiLU(),
            nn.Linear(hidden, cfg.d_topic),
        )
        self.hidden_to_topic = nn.Linear(cfg.d_LLM, cfg.d_topic)

        fiber_hidden = max(4 * cfg.d_F_topic, 128)
        self.fiber_mlp = nn.Sequential(
            nn.Linear(cfg.d_LLM + cfg.d_topic, fiber_hidden), nn.SiLU(),
            nn.Linear(fiber_hidden, cfg.d_F_topic),
        )

    def forward(self, hidden_state: Tensor,
                content_token_ids: Sequence,
                wte_normed: Tensor,
                idf: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """hidden_state: (B, d_LLM); content_token_ids: list-of-int or list-of-lists;
        wte_normed: (V, d_LLM) L2-normalized; idf: optional (V,).

        Returns (base, fiber, dirn), each (B, d_topic) or (B, d_F_topic).
        """
        assert hidden_state.dim() == 2 and hidden_state.shape[-1] == self.d_LLM
        B = hidden_state.shape[0]

        # Support ragged list or list-of-lists; normalize to list-of-lists.
        if len(content_token_ids) > 0 and not isinstance(content_token_ids[0], (list, tuple)):
            # Single-example flat list
            assert B == 1, (
                "content_token_ids is a flat list but hidden_state batch > 1; "
                "pass list-of-lists for batched input"
            )
            token_lists: List[List[int]] = [list(content_token_ids)]
        else:
            token_lists = [list(x) for x in content_token_ids]
            assert len(token_lists) == B

        # Per-batch IDF-weighted centroid in d_LLM, then project to d_topic
        centroids = torch.stack([
            _idf_weighted_centroid(tl, wte_normed, idf) for tl in token_lists
        ], dim=0)  # (B, d_LLM)

        mixed = self.down_project(centroids) + self.hidden_to_topic(hidden_state)  # (B, d_topic)
        base = F.normalize(mixed, dim=-1, eps=1e-8)  # on the sphere
        fiber = self.fiber_mlp(torch.cat([hidden_state, base], dim=-1))  # (B, d_F_topic)
        dirn = base  # already unit-norm
        return base, fiber, dirn


class TopicBundle(Bundle):
    """Fiber bundle with S^{d_topic-1} as base.

    Transport along great-circle paths (closed form), skipping GeodesicSolver.
    """

    def __init__(self, cfg: Cfg4):
        super().__init__(name="topic", cfg=cfg, d_base=cfg.d_topic, d_fiber=cfg.d_F_topic)
        # Override solver: not needed for sphere (we use _great_circle_path)
        self._solver = None
        self.encoder = TopicEncoder(cfg)

    def canonical_axis(self) -> Tensor:
        # Also on the sphere for topic
        return F.normalize(self._axis_raw, dim=0, eps=1e-8)

    def encode(self, hidden_state: Tensor, *,
               content_token_ids, wte_normed: Tensor,
               idf: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        return self.encoder(hidden_state, content_token_ids, wte_normed, idf=idf)

    def _great_circle_path(self, base_src: Tensor, base_dst: Tensor,
                            n_pts: int) -> Tensor:
        """Interpolate on the great circle between unit vectors base_src, base_dst.

        base_src, base_dst: (B, d_topic) unit.
        Returns: (B, n_pts, d_topic).

        Uses slerp. Handles near-antipodal / identical edge cases.
        """
        B = base_src.shape[0]
        t = torch.linspace(0, 1, n_pts, device=base_src.device).view(1, n_pts, 1)
        dot = (base_src * base_dst).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(dot)  # (B, 1)
        sin_theta = torch.sin(theta).clamp(min=1e-7)  # (B, 1)
        # (B, n_pts, 1) expansion
        theta_e = theta.unsqueeze(1)         # (B, 1, 1)
        sin_theta_e = sin_theta.unsqueeze(1) # (B, 1, 1)
        a = torch.sin((1 - t) * theta_e) / sin_theta_e  # (B, n_pts, 1)
        b = torch.sin(t * theta_e) / sin_theta_e        # (B, n_pts, 1)
        path = a * base_src.unsqueeze(1) + b * base_dst.unsqueeze(1)  # (B, n_pts, d_topic)
        return path

    def transport(self, fiber_src: Tensor, base_src: Tensor, base_dst: Tensor) -> Tensor:
        """Great-circle transport: build slerp path, then RK4 on fiber."""
        n_pts = self.cfg.n_geo_pts + 2
        path = self._great_circle_path(base_src, base_dst, n_pts)
        return self.trans(fiber_src, path)
