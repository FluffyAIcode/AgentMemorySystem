"""TopicBundle — carries topic-axis memory encoding.

Base space B_topic = S^{d_topic - 1} (the unit sphere). A point is a topic
direction: a dense representation of *what* the memory is about.

Why the sphere: topic similarity is naturally cosine-based, and closed-form
geodesics on the sphere let TopicBundle skip GeodesicSolver (use great-circle
paths), which is faster and more stable than gradient descent in R^d.

Canonical axis: the dominant topic direction of the store's content
population. Updated when the store reclusters (see KakeyaRegistry.rebuild()).
"""
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn

from ams_v4.bundles.base import (
    Bundle, RiemannianMetric, FiberConnection, FiberTransporter,
)
from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class TopicEncoder(nn.Module):
    """Encodes (hidden_state, content_token_ids, wte_normed) →
                 (topic_base, topic_fiber, topic_dirn).

    topic_base is computed as the L2-normalized IDF-weighted mean of
    wte_normed[content_token_ids] projected onto R^{d_topic} via a learned
    projection. That gives a point on S^{d_topic - 1} directly — no separate
    normalization step in the loss.
    """
    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        # arch sketch (v4.2):
        #   content_centroid = idf_weighted_mean(wte_normed[ids])  -> (B, d_LLM)
        #   base = normalize(Linear_down(content_centroid + hidden_proj(hidden))) -> (B, d_topic)
        #   fiber = MLP(concat(hidden, base))                      -> (B, d_F_topic)
        #   dirn  = base                                            (already unit)
        raise NotImplementedError("v4-skel: TopicEncoder.__init__ — lands in v4.2")

    def forward(self, hidden_state: Tensor, content_token_ids: List[int],
                wte_normed: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """hidden_state: (B, d_LLM); content_token_ids: list[int];
        wte_normed: (V, d_LLM).

        Returns (base, fiber, dirn), shapes (B, d_topic), (B, d_F_topic), (B, d_topic).
        """
        raise NotImplementedError("v4-skel: TopicEncoder.forward — lands in v4.2")


class TopicBundle(Bundle):
    """Fiber bundle with S^{d_topic-1} as base, F_topic as typical fiber."""

    def __init__(self, cfg: Cfg4):
        super().__init__(name="topic", cfg=cfg)
        self.d_base = cfg.d_topic
        self.d_fiber = cfg.d_F_topic
        # v4.1: metric + connection + transporter on the sphere
        # No GeodesicSolver — topic transport uses great-circle paths.
        raise NotImplementedError("v4-skel: TopicBundle.__init__ — lands in v4.1/v4.2")

    def canonical_axis(self) -> Tensor:
        raise NotImplementedError("v4-skel: TopicBundle.canonical_axis — lands in v4.2")

    def encode(self, hidden_state: Tensor, *, content_token_ids: List[int],
               wte_normed: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("v4-skel: TopicBundle.encode — lands in v4.2")

    def transport(self, fiber_src: Tensor, base_src: Tensor, base_dst: Tensor) -> Tensor:
        """Great-circle transport. Closed form:
          θ = arccos(base_src · base_dst)
          path = great_circle(base_src, base_dst, n_geo_pts)
        Then run FiberTransporter over that path.
        """
        raise NotImplementedError("v4-skel: TopicBundle.transport — lands in v4.2")
