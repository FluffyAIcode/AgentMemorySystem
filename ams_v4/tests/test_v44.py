"""v4.4 tests — BundleQueryHeads + CrossBundleAttention."""
from __future__ import annotations
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from ams_v4 import Cfg4, MemEntry
from ams_v4.attention.cross_bundle import CrossBundleAttention
from ams_v4.attention.query_heads import BundleQueryHeads


def _mk_entry(cfg, mid=0):
    return MemEntry(
        mid=mid,
        time_base=torch.randn(cfg.d_time),
        time_fiber=torch.randn(cfg.d_F_time),
        time_dirn=F.normalize(torch.randn(cfg.d_time), dim=0),
        topic_base=F.normalize(torch.randn(cfg.d_topic), dim=0),
        topic_fiber=torch.randn(cfg.d_F_topic),
        topic_dirn=F.normalize(torch.randn(cfg.d_topic), dim=0),
        ctx_base=torch.randn(cfg.d_ctx),
        ctx_fiber=torch.randn(cfg.d_F_ctx),
        ctx_dirn=F.normalize(torch.randn(cfg.d_ctx), dim=0),
        surprise=0.0, ts=0.0, last=0.0, cnt=0,
    )


# ─── BundleQueryHeads ────────────────────────────────────────────────────

def test_query_heads_shapes():
    torch.manual_seed(0)
    cfg = Cfg4()
    qh = BundleQueryHeads(cfg)
    h = torch.randn(3, cfg.d_LLM)
    q = qh(h)
    assert q["time"].shape == (3, cfg.d_F_time)
    assert q["topic"].shape == (3, cfg.d_F_topic)
    assert q["ctx"].shape == (3, cfg.d_F_ctx)


def test_query_heads_distinct():
    """Queries should be actually different across bundles (not the same tensor)."""
    torch.manual_seed(1)
    cfg = Cfg4()
    qh = BundleQueryHeads(cfg)
    h = torch.randn(2, cfg.d_LLM)
    q = qh(h)
    # Pairs have different dims so we only check that at least the first
    # four elements are not numerically identical (sanity check, not
    # strong assertion).
    assert not torch.allclose(q["time"][:, :4], q["topic"][:, :4], atol=0), \
        "time and topic queries unexpectedly identical"


# ─── CrossBundleAttention ────────────────────────────────────────────────

def test_cross_bundle_forward_shape():
    torch.manual_seed(2)
    cfg = Cfg4()
    cba = CrossBundleAttention(cfg)
    entries = [_mk_entry(cfg, mid=i) for i in range(5)]
    h = torch.randn(2, cfg.d_LLM)
    prefix = cba(h, entries)
    assert prefix.shape == (2, cfg.L_mem, cfg.d_LLM), \
        f"prefix shape {tuple(prefix.shape)}"


def test_cross_bundle_requires_at_least_one_entry():
    cfg = Cfg4()
    cba = CrossBundleAttention(cfg)
    h = torch.randn(1, cfg.d_LLM)
    try:
        cba(h, [])
    except AssertionError:
        return
    raise AssertionError("expected AssertionError when entries is empty")


def test_cross_bundle_gradient_flow():
    """prefix.sum().backward() produces non-zero gradient on q_time.weight."""
    torch.manual_seed(3)
    cfg = Cfg4()
    cba = CrossBundleAttention(cfg)
    entries = [_mk_entry(cfg, mid=i) for i in range(4)]
    h = torch.randn(1, cfg.d_LLM, requires_grad=False)
    prefix = cba(h, entries)
    prefix.sum().backward()
    g = cba.query_heads.q_time.weight.grad
    assert g is not None and g.abs().sum().item() > 0, \
        "no gradient flowed through query_heads.q_time.weight"


def test_cross_bundle_finite_with_random_fibers():
    """Check numerical stability on random init."""
    torch.manual_seed(4)
    cfg = Cfg4()
    cba = CrossBundleAttention(cfg)
    entries = [_mk_entry(cfg, mid=i) for i in range(8)]
    h = torch.randn(3, cfg.d_LLM)
    prefix = cba(h, entries)
    assert torch.isfinite(prefix).all(), "prefix has non-finite values"


def test_cross_bundle_batch_determinism():
    """Same inputs twice in eval() should produce same outputs."""
    torch.manual_seed(5)
    cfg = Cfg4()
    cba = CrossBundleAttention(cfg).eval()
    entries = [_mk_entry(cfg, mid=i) for i in range(4)]
    h = torch.randn(2, cfg.d_LLM)
    with torch.no_grad():
        p1 = cba(h, entries)
        p2 = cba(h, entries)
    assert torch.allclose(p1, p2, atol=1e-6)


def test_cross_bundle_slot_allocation_matches_cfg():
    """prefix[:, :prefix_slots_time] come from time lifts; prove the slot
    allocation is consistent with Cfg4 values by checking that with
    random queries, the time-slot rows change when we perturb only the
    time fibers.
    """
    torch.manual_seed(6)
    cfg = Cfg4()
    cba = CrossBundleAttention(cfg).eval()
    entries = [_mk_entry(cfg, mid=i) for i in range(4)]
    h = torch.randn(1, cfg.d_LLM)

    with torch.no_grad():
        p_before = cba(h, entries)

    # Perturb every entry's time_fiber only
    for e in entries:
        e.time_fiber = e.time_fiber + 10.0 * torch.randn_like(e.time_fiber)

    with torch.no_grad():
        p_after = cba(h, entries)

    # time slots should change
    time_slots = slice(0, cfg.prefix_slots_time)
    topic_slots = slice(cfg.prefix_slots_time, cfg.prefix_slots_time + cfg.prefix_slots_topic)
    ctx_slots = slice(cfg.prefix_slots_time + cfg.prefix_slots_topic, cfg.L_mem)

    d_time_slots = (p_after[:, time_slots] - p_before[:, time_slots]).abs().mean()
    d_topic_slots = (p_after[:, topic_slots] - p_before[:, topic_slots]).abs().mean()
    d_ctx_slots = (p_after[:, ctx_slots] - p_before[:, ctx_slots]).abs().mean()

    # After LayerNorm, cross-slot coupling is non-zero but time slots should
    # change the most on a time-fiber perturbation.
    assert d_time_slots > d_topic_slots, (
        f"time slots change ({d_time_slots}) should exceed topic slots "
        f"({d_topic_slots}) on a time-fiber perturbation"
    )
    assert d_time_slots > d_ctx_slots, (
        f"time slots change ({d_time_slots}) should exceed ctx slots "
        f"({d_ctx_slots}) on a time-fiber perturbation"
    )


# ─── Runner ──────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_query_heads_shapes,
        test_query_heads_distinct,
        test_cross_bundle_forward_shape,
        test_cross_bundle_requires_at_least_one_entry,
        test_cross_bundle_gradient_flow,
        test_cross_bundle_finite_with_random_fibers,
        test_cross_bundle_batch_determinism,
        test_cross_bundle_slot_allocation_matches_cfg,
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
    print(f"\nall {len(tests)} v4.4 tests passed")


if __name__ == "__main__":
    _run_all()
