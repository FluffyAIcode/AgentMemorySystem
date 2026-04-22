"""Abstract Bundle + geometry primitives.

RiemannianMetric, FiberConnection, FiberTransporter, GeodesicSolver are the
four pieces that implement parallel transport on a fiber bundle. v3.46's
scheme_b_v344.py already has correct implementations of all four; v4.1 will
port them here with minimal edits (change d_M / d_F names to per-bundle
dims and make them generic over bundle dims).

Bundle is a new abstract class that ties (metric, connection, transporter,
solver) to one named bundle (time | topic | ctx) and a fixed (d_base, d_fiber)
pair. Concrete subclasses are TemporalBundle, TopicBundle, ContextBundle.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class RiemannianMetric(nn.Module):
    """Learned Riemannian metric g(x) on a base manifold of dim d_base.

    Signature mirrors v3.46 RiemannianMetric. Bundle-generic: instantiate
    one metric per bundle with its own d_base.
    """
    def __init__(self, d_base: int, hidden: int = 64):
        super().__init__()
        self.d_base = d_base
        # Implementation ported in v4.1 from scheme_b_v344.py::RiemannianMetric
        raise NotImplementedError("v4-skel: RiemannianMetric.__init__ — lands in v4.1")

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, d_base) → g: (B, d_base, d_base), SPD."""
        raise NotImplementedError("v4-skel: RiemannianMetric.forward — lands in v4.1")


class FiberConnection(nn.Module):
    """Antisymmetric connection A(x, v) ∈ so(d_fiber), parameterized by a
    metric-aware MLP.

    Ported in v4.1 from scheme_b_v344.py::FiberConnection. Per-bundle:
    instantiate one with that bundle's (d_base, d_fiber).
    """
    def __init__(self, d_base: int, d_fiber: int, metric: RiemannianMetric,
                 grad_coupling: bool = True):
        super().__init__()
        self.d_base = d_base
        self.d_fiber = d_fiber
        self.metric = metric
        self.grad_coupling = grad_coupling
        raise NotImplementedError("v4-skel: FiberConnection.__init__ — lands in v4.1")

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """x: (B, d_base), v: (B, d_base) → A: (B, d_fiber, d_fiber), antisym."""
        raise NotImplementedError("v4-skel: FiberConnection.forward — lands in v4.1")


class FiberTransporter(nn.Module):
    """Parallel transport of a fiber along a piecewise-linear path on B.

    RK4 with periodic norm correction. Ported in v4.1 from
    scheme_b_v344.py::FiberTransporter.
    """
    def __init__(self, conn: FiberConnection, cfg: Cfg4):
        super().__init__()
        self.conn = conn
        self.cfg = cfg
        raise NotImplementedError("v4-skel: FiberTransporter.__init__ — lands in v4.1")

    def forward(self, fiber: Tensor, path: Tensor) -> Tensor:
        """fiber: (B, d_fiber), path: (B, n_pts, d_base) → transported: (B, d_fiber)."""
        raise NotImplementedError("v4-skel: FiberTransporter.forward — lands in v4.1")


class GeodesicSolver(nn.Module):
    """Gradient-descent geodesic solver on B under a given metric.

    Ported in v4.1. Not used by every bundle — TopicBundle skips it (its
    base is the sphere, closed-form geodesics).
    """
    def __init__(self, metric: RiemannianMetric, cfg: Cfg4):
        super().__init__()
        self.metric = metric
        self.cfg = cfg
        raise NotImplementedError("v4-skel: GeodesicSolver.__init__ — lands in v4.1")

    def forward(self, p0: Tensor, p1: Tensor) -> Tensor:
        """p0, p1: (B, d_base) → path: (B, n_pts, d_base)."""
        raise NotImplementedError("v4-skel: GeodesicSolver.forward — lands in v4.1")


class Bundle(ABC, nn.Module):
    """Abstract bundle. A concrete bundle = (name, d_base, d_fiber, metric,
    connection, transporter, solver) + a canonical axis direction.

    The canonical axis is a fixed or learned unit vector in R^{d_base}. It
    represents the "time axis" in the temporal bundle, the "dominant topic
    direction" in the topic bundle, etc. KakeyaSet.t_dir must align with
    the pushforward of this axis into the kakeya PCA subspace (§1.3).
    """
    name: str
    d_base: int
    d_fiber: int

    def __init__(self, name: str, cfg: Cfg4):
        super().__init__()
        self.name = name
        self.cfg = cfg

    @abstractmethod
    def canonical_axis(self) -> Tensor:
        """Return the (d_base,) unit vector this bundle's Kakeya set aligns to."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, hidden_state: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Produce (base, fiber, dirn) for a new memory.

        hidden_state: (B, T, d_LLM) or (B, d_LLM) — bundle-specific.
        Returns:
          base:  (B, d_base)
          fiber: (B, d_fiber)
          dirn:  (B, d_base), unit-norm
        """
        raise NotImplementedError

    @abstractmethod
    def transport(self, fiber_src: Tensor, base_src: Tensor, base_dst: Tensor) -> Tensor:
        """Parallel-transport a fiber from base_src to base_dst along the
        bundle's preferred path. Returns: (B, d_fiber).
        """
        raise NotImplementedError
