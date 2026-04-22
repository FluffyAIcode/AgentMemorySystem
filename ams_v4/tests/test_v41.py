"""v4.1 tests — geometry primitives, MemStore, DirectionTreeV4.

Run:
  python3 ams_v4/tests/test_v41.py
"""
from __future__ import annotations
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from ams_v4 import Cfg4, MemEntry, MemStore
from ams_v4.bundles.base import (
    Bundle, RiemannianMetric, FiberConnection, FiberTransporter, GeodesicSolver,
)


def _mk_entry(mid=-1, d_time=8, d_F_time=32, d_topic=16, d_F_topic=64,
              d_ctx=12, d_F_ctx=48):
    """Build a minimal MemEntry with random unit-norm dirns."""
    tb = torch.randn(d_time); td = torch.nn.functional.normalize(tb, dim=0)
    pb = torch.nn.functional.normalize(torch.randn(d_topic), dim=0); pd = pb.clone()
    cb = torch.randn(d_ctx); cd = torch.nn.functional.normalize(cb, dim=0)
    return MemEntry(
        mid=mid,
        time_base=tb, time_fiber=torch.randn(d_F_time), time_dirn=td,
        topic_base=pb, topic_fiber=torch.randn(d_F_topic), topic_dirn=pd,
        ctx_base=cb, ctx_fiber=torch.randn(d_F_ctx), ctx_dirn=cd,
        surprise=0.0, ts=0.0, last=0.0, cnt=0,
        source_text="t",
    )


# ─── Metric ───────────────────────────────────────────────────────────────

def test_metric_spd():
    torch.manual_seed(0)
    g_mod = RiemannianMetric(d_base=8)
    x = torch.randn(4, 8)
    g = g_mod(x)
    assert g.shape == (4, 8, 8)
    # Symmetry
    sym = (g - g.transpose(-1, -2)).abs().max().item()
    assert sym < 1e-5, f"metric not symmetric, max asym = {sym}"
    # PD via eigenvalues
    evals = torch.linalg.eigvalsh(g)
    assert evals.min().item() > 0, f"metric not PD, min eig = {evals.min().item()}"


# ─── Connection ───────────────────────────────────────────────────────────

def test_connection_antisymmetric():
    torch.manual_seed(0)
    m = RiemannianMetric(8)
    c = FiberConnection(d_base=8, d_fiber=32, metric=m)
    x = torch.randn(3, 8); v = torch.randn(3, 8)
    A = c(x, v)
    assert A.shape == (3, 32, 32)
    asym = (A + A.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-5, f"connection not antisymmetric, max = {asym}"


# ─── Transporter ──────────────────────────────────────────────────────────

def test_transporter_preserves_norm():
    torch.manual_seed(0)
    cfg = Cfg4()
    m = RiemannianMetric(8)
    c = FiberConnection(8, 32, m)
    t = FiberTransporter(c, cfg)
    fiber = torch.randn(2, 32)
    # Closed-loop path: start and end at same point, via detour
    p0 = torch.zeros(2, 8); p1 = torch.ones(2, 8) * 0.1; p2 = torch.zeros(2, 8)
    path = torch.stack([p0, p1, p2], dim=1)  # (2, 3, 8)
    out = t(fiber, path)
    n_in = fiber.norm(dim=-1); n_out = out.norm(dim=-1)
    rel = ((n_out - n_in) / n_in).abs().max().item()
    # Note: v3.46 does periodic norm renormalization; over a short path
    # tolerance of 10% is generous and stable across seeds.
    assert rel < 0.1, f"transporter norm drift too large: rel = {rel}"


# ─── Geodesic ─────────────────────────────────────────────────────────────

def test_geodesic_endpoints():
    torch.manual_seed(0)
    cfg = Cfg4()
    m = RiemannianMetric(8)
    s = GeodesicSolver(m, cfg)
    xs = torch.randn(2, 8); xe = torch.randn(2, 8)
    res = s.solve(xs, xe)
    assert res.path.shape == (2, cfg.n_geo_pts + 2, 8)
    assert (res.path[:, 0] - xs).abs().max() < 1e-4
    assert (res.path[:, -1] - xe).abs().max() < 1e-4


def test_geodesic_linear_fallback():
    cfg = Cfg4()
    m = RiemannianMetric(8)
    s = GeodesicSolver(m, cfg)
    xs = torch.zeros(1, 8); xe = torch.ones(1, 8)
    lin = s.linear_path(xs, xe)
    assert lin.shape == (1, cfg.n_geo_pts + 2, 8)
    # First point is xs, last is xe, monotone interp
    assert (lin[:, 0] - xs).abs().max() < 1e-6
    assert (lin[:, -1] - xe).abs().max() < 1e-6


# ─── DirectionTreeV4 + MemStore ───────────────────────────────────────────

def test_memstore_add_routes_to_all_three_trees():
    cfg = Cfg4()
    store = MemStore(cfg)
    for _ in range(5):
        store.add(_mk_entry(
            d_time=cfg.d_time, d_F_time=cfg.d_F_time,
            d_topic=cfg.d_topic, d_F_topic=cfg.d_F_topic,
            d_ctx=cfg.d_ctx, d_F_ctx=cfg.d_F_ctx,
        ))
    assert len(store) == 5
    assert store.tree_time.size() == 5
    assert store.tree_topic.size() == 5
    assert store.tree_ctx.size() == 5


def test_direction_tree_insert_retrieve():
    torch.manual_seed(1)
    cfg = Cfg4()
    store = MemStore(cfg)
    n = 20
    for _ in range(n):
        store.add(_mk_entry(
            d_time=cfg.d_time, d_F_time=cfg.d_F_time,
            d_topic=cfg.d_topic, d_F_topic=cfg.d_F_topic,
            d_ctx=cfg.d_ctx, d_F_ctx=cfg.d_F_ctx,
        ))
    # Pick an arbitrary memory as the query
    target_mid = 7
    target_dirn = store.get(target_mid).topic_dirn
    hits = store.tree_topic.retrieve(target_dirn, beam=cfg.retrieval_beam)
    top_mids = [mid for mid, _ in hits[:3]]
    assert target_mid in top_mids, \
        f"target mid {target_mid} not in top-3 of retrieval: {top_mids}"


def test_memstore_remove_updates_trees():
    cfg = Cfg4()
    store = MemStore(cfg)
    mids = []
    for _ in range(6):
        mids.append(store.add(_mk_entry(
            d_time=cfg.d_time, d_F_time=cfg.d_F_time,
            d_topic=cfg.d_topic, d_F_topic=cfg.d_F_topic,
            d_ctx=cfg.d_ctx, d_F_ctx=cfg.d_F_ctx,
        )))
    store.remove(mids[2])
    assert len(store) == 5
    assert store.tree_time.size() == 5
    assert store.tree_topic.size() == 5
    assert store.tree_ctx.size() == 5


def test_memstore_verify_consistency_empty():
    cfg = Cfg4()
    store = MemStore(cfg)
    errs = store.verify_consistency()
    assert errs == [], f"empty store should have no errors, got: {errs}"


def test_memstore_verify_consistency_populated():
    cfg = Cfg4()
    store = MemStore(cfg)
    for _ in range(4):
        store.add(_mk_entry(
            d_time=cfg.d_time, d_F_time=cfg.d_F_time,
            d_topic=cfg.d_topic, d_F_topic=cfg.d_F_topic,
            d_ctx=cfg.d_ctx, d_F_ctx=cfg.d_F_ctx,
        ))
    errs = store.verify_consistency()
    assert errs == [], f"populated store with all valid entries should have no errors, got: {errs}"


def test_memstore_invariant_no_raw_large_fields():
    cfg = Cfg4()
    store = MemStore(cfg)
    e = _mk_entry(
        d_time=cfg.d_time, d_F_time=cfg.d_F_time,
        d_topic=cfg.d_topic, d_F_topic=cfg.d_F_topic,
        d_ctx=cfg.d_ctx, d_F_ctx=cfg.d_F_ctx,
    )
    # Attach a raw 1536-dim tensor to the entry (simulating a drift back to v3.46 style)
    e.__dict__["semantic_emb_raw"] = torch.randn(cfg.d_LLM)
    store.add(e)
    try:
        store.assert_all_large_fields_compressed()
    except AssertionError as ex:
        assert "semantic_emb_raw" in str(ex) or "1536" in str(ex) or str(cfg.d_LLM) in str(ex)
        return
    raise AssertionError("expected AssertionError for raw large field")


# ─── Runner ───────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_metric_spd,
        test_connection_antisymmetric,
        test_transporter_preserves_norm,
        test_geodesic_endpoints,
        test_geodesic_linear_fallback,
        test_memstore_add_routes_to_all_three_trees,
        test_direction_tree_insert_retrieve,
        test_memstore_remove_updates_trees,
        test_memstore_verify_consistency_empty,
        test_memstore_verify_consistency_populated,
        test_memstore_invariant_no_raw_large_fields,
    ]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} / {len(tests)} failed: {failed}")
        sys.exit(1)
    print(f"\nall {len(tests)} v4.1 tests passed")


if __name__ == "__main__":
    _run_all()
