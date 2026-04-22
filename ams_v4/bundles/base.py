"""Abstract Bundle + geometry primitives (v4.1).

Ports from scheme_b_v344.py (v3.46):
  RiemannianMetric   (@lines 554-590)    — same parameterization; d is now per-bundle
  GeodesicSolver     (@lines 595-624)    — same algorithm; stays as a plain class (not nn.Module)
  FiberConnection    (@lines 626-638)    — same parameterization; (d_base, d_fiber) per-bundle
  FiberTransporter   (@lines 640-653)    — same RK4; uses cfg.norm_correction_interval only

What changed vs v3.46:
  - d_M / d_F renamed to d_base / d_fiber, wired per-bundle rather than globally
  - Bundle abstract class (§1.3) is new
  - NamedTuple GeodesicResult ported verbatim
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


# ─── Riemannian metric ────────────────────────────────────────────────────

class RiemannianMetric(nn.Module):
    """Learned SPD metric g(x) on a base manifold of dim d_base.

    Parameterization: produces the lower-triangular Cholesky factor L via an
    MLP, and returns g = L L^T. The diagonal is softplus'd + ε > 0 so g is
    strictly positive-definite.
    """

    def __init__(self, d_base: int, hidden_mult: int = 4):
        super().__init__()
        self.d_base = d_base
        n_tri = d_base * (d_base + 1) // 2
        h = hidden_mult * d_base
        self.net = nn.Sequential(
            nn.Linear(d_base, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, n_tri),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)
        r, c = [], []
        for i in range(d_base):
            for j in range(i + 1):
                r.append(i); c.append(j)
        self.register_buffer("_r", torch.tensor(r))
        self.register_buffer("_c", torch.tensor(c))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, d_base) → g: (B, d_base, d_base) SPD."""
        B = x.shape[0]
        d = self.d_base
        v = self.net(x)
        L = x.new_zeros(B, d, d)
        L[:, self._r, self._c] = v
        di = torch.arange(d, device=x.device)
        L[:, di, di] = F.softplus(L[:, di, di]) + 1e-3
        return L @ L.transpose(1, 2)

    def midpoint_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Approximate geodesic distance by evaluating g at the midpoint."""
        diff = x - y
        mid = (x + y) / 2
        with torch.no_grad():
            g = self.forward(mid)
        return torch.einsum("bi,bij,bj->b", diff, g, diff).clamp(min=0).sqrt()


# ─── Geodesic solver ──────────────────────────────────────────────────────

class GeodesicResult(NamedTuple):
    path: Tensor       # (B, n_pts, d_base)
    energy: float
    converged: bool
    iterations: int


class GeodesicSolver:
    """Finds an approximate geodesic between two points by gradient descent
    on path energy. Not an nn.Module — holds no parameters of its own.
    """

    def __init__(self, metric: RiemannianMetric, cfg: Cfg4):
        self.metric = metric
        self.cfg = cfg

    def solve(self, xs: Tensor, xe: Tensor) -> GeodesicResult:
        """xs, xe: (B, d_base) → path: (B, n_geo_pts+2, d_base)."""
        B, d = xs.shape
        N = self.cfg.n_geo_pts
        dev = xs.device
        t = torch.linspace(0, 1, N + 2, device=dev)[1:-1]

        # Freeze metric params during path search; restore after
        ps = {n: p.requires_grad for n, p in self.metric.named_parameters()}
        for p in self.metric.parameters():
            p.requires_grad_(False)

        try:
            with torch.enable_grad():
                interior = (xs.detach().unsqueeze(1) * (1 - t[None, :, None])
                            + xe.detach().unsqueeze(1) * t[None, :, None]
                            ).detach().clone().requires_grad_(True)
                opt = torch.optim.Adam([interior], lr=self.cfg.geo_lr)
                prev = float("inf"); converged = False; iters = 0; cur = prev
                for it in range(self.cfg.geo_max_steps):
                    opt.zero_grad()
                    path = torch.cat([xs.detach().unsqueeze(1), interior,
                                      xe.detach().unsqueeze(1)], dim=1)
                    dx = path[:, 1:] - path[:, :-1]
                    mid = (path[:, 1:] + path[:, :-1]) / 2
                    g = self.metric(mid.reshape(-1, d)).reshape(B, N + 1, d, d)
                    energy = torch.einsum("bni,bnij,bnj->", dx, g, dx)
                    if not torch.isfinite(energy):
                        t_full = torch.linspace(0, 1, N + 2, device=dev).view(1, -1, 1)
                        lin = xs.unsqueeze(1) * (1 - t_full) + xe.unsqueeze(1) * t_full
                        return GeodesicResult(lin, float("inf"), False, it)
                    energy.backward()
                    opt.step()
                    iters = it + 1
                    cur = energy.item()
                    if abs(prev - cur) / (abs(prev) + 1e-10) < self.cfg.geo_tol:
                        converged = True
                        break
                    prev = cur
        finally:
            for n, p in self.metric.named_parameters():
                p.requires_grad_(ps[n])

        final = torch.cat([xs.unsqueeze(1), interior.detach(), xe.unsqueeze(1)], dim=1)
        return GeodesicResult(final, cur, converged, iters)

    def linear_path(self, xs: Tensor, xe: Tensor) -> Tensor:
        """Fallback: straight-line path in R^{d_base}. (B, n_geo_pts+2, d_base)."""
        N = self.cfg.n_geo_pts
        t_full = torch.linspace(0, 1, N + 2, device=xs.device).view(1, -1, 1)
        return xs.unsqueeze(1) * (1 - t_full) + xe.unsqueeze(1) * t_full


# ─── Fiber connection ─────────────────────────────────────────────────────

class FiberConnection(nn.Module):
    """Antisymmetric connection A(x, v) ∈ so(d_fiber).

    Parameterized by a metric-aware MLP on concat(x, v, tri(g(x))).
    """

    def __init__(self, d_base: int, d_fiber: int, metric: RiemannianMetric,
                 grad_coupling: bool = True):
        super().__init__()
        self.d_base = d_base
        self.d_fiber = d_fiber
        self.metric = metric
        self.grad_coupling = grad_coupling
        d_g = d_base * (d_base + 1) // 2
        self.net = nn.Sequential(
            nn.Linear(2 * d_base + d_g, 4 * d_fiber), nn.SiLU(),
            nn.Linear(4 * d_fiber, 4 * d_fiber), nn.SiLU(),
            nn.Linear(4 * d_fiber, d_fiber * d_fiber),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.normal_(self.net[-1].bias, std=0.01)

    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        """x, v: (B, d_base) → A: (B, d_fiber, d_fiber) antisymmetric."""
        g = self.metric(x)
        d = g.shape[-1]
        idx = torch.triu_indices(d, d, device=x.device)
        gf = g[:, idx[0], idx[1]]
        if not self.grad_coupling:
            gf = gf.detach()
        raw = self.net(torch.cat([x, v, gf], dim=-1)).reshape(-1, self.d_fiber, self.d_fiber)
        return (raw - raw.transpose(1, 2)) / 2


# ─── Fiber transporter ────────────────────────────────────────────────────

class FiberTransporter(nn.Module):
    """RK4 parallel transport of a fiber along a piecewise-linear path on B.

    Applies periodic norm correction every `cfg.norm_correction_interval`
    steps to prevent numerical drift.
    """

    def __init__(self, conn: FiberConnection, cfg: Cfg4):
        super().__init__()
        self.conn = conn
        self.cfg = cfg

    def forward(self, fiber: Tensor, path: Tensor) -> Tensor:
        """fiber: (B, d_fiber), path: (B, n_pts, d_base) → transported: (B, d_fiber)."""
        f = fiber
        n0 = fiber.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        nci = self.cfg.norm_correction_interval
        for k in range(path.shape[1] - 1):
            p0, p1 = path[:, k], path[:, k + 1]
            v = p1 - p0
            mid = (p0 + p1) / 2
            k1 = -(self.conn(p0, v) @ f.unsqueeze(-1)).squeeze(-1)
            k2 = -(self.conn(mid, v) @ (f + 0.5 * k1).unsqueeze(-1)).squeeze(-1)
            k3 = -(self.conn(mid, v) @ (f + 0.5 * k2).unsqueeze(-1)).squeeze(-1)
            k4 = -(self.conn(p1, v) @ (f + k3).unsqueeze(-1)).squeeze(-1)
            f = f + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if (k + 1) % nci == 0:
                f = f * (n0 / f.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        return f


# ─── Abstract Bundle ──────────────────────────────────────────────────────

class Bundle(ABC, nn.Module):
    """Abstract bundle: name, (d_base, d_fiber), plus its geometry objects
    and a canonical axis direction.

    Concrete subclasses (TemporalBundle, TopicBundle, ContextBundle) each
    instantiate (metric, conn, transporter) and implement encode + transport.

    The canonical_axis (a learned unit parameter in R^{d_base}) is the axis
    along which this bundle's Kakeya sets align their distinguished direction
    (see alignment.py).
    """

    def __init__(self, name: str, cfg: Cfg4, d_base: int, d_fiber: int):
        super().__init__()
        assert name in ("time", "topic", "ctx"), \
            f"Bundle name must be time/topic/ctx, got {name}"
        self.name = name
        self.cfg = cfg
        self.d_base = d_base
        self.d_fiber = d_fiber

        self.metric = RiemannianMetric(d_base)
        self.conn = FiberConnection(d_base, d_fiber, self.metric)
        self.trans = FiberTransporter(self.conn, cfg)
        # Solver is optional; topic bundle skips it (great-circle closed form)
        self._solver: Optional[GeodesicSolver] = GeodesicSolver(self.metric, cfg)

        # Canonical axis (the "t-direction" for this bundle's kakeya sets).
        # Initialized random then unit-normalized every access.
        self._axis_raw = nn.Parameter(torch.randn(d_base) * 0.1)

    def canonical_axis(self) -> Tensor:
        """Return the (d_base,) unit-norm canonical axis for this bundle."""
        return F.normalize(self._axis_raw, dim=0, eps=1e-8)

    @abstractmethod
    def encode(self, hidden_state: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (base, fiber, dirn). See concrete bundles for kwarg contract."""
        raise NotImplementedError

    def transport(self, fiber_src: Tensor, base_src: Tensor, base_dst: Tensor) -> Tensor:
        """Default: geodesic path from base_src to base_dst, then RK4 transport."""
        if self._solver is None:
            path = self.trans.cfg  # type: ignore  # unreachable; set to None only if subclass overrides
        res = self._solver.solve(base_src, base_dst)
        return self.trans(fiber_src, res.path)
