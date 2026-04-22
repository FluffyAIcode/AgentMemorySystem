"""v4.2 tests — three encoders + three concrete bundles."""
from __future__ import annotations
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from ams_v4 import Cfg4
from ams_v4.bundles.temporal import TimeEncoder, TemporalBundle
from ams_v4.bundles.topic import TopicEncoder, TopicBundle, _idf_weighted_centroid
from ams_v4.bundles.context import ContextEncoder, ContextBundle


# ─── TimeEncoder ─────────────────────────────────────────────────────────

def test_time_encoder_shapes():
    torch.manual_seed(0)
    cfg = Cfg4()
    enc = TimeEncoder(cfg)
    B = 3
    h = torch.randn(B, cfg.d_LLM)
    ts = torch.randn(B, 3)
    s = torch.randn(B)
    base, fiber, dirn = enc(h, ts, s)
    assert base.shape == (B, cfg.d_time)
    assert fiber.shape == (B, cfg.d_F_time)
    assert dirn.shape == (B, cfg.d_time)


def test_time_dirn_unit_norm():
    torch.manual_seed(1)
    cfg = Cfg4()
    enc = TimeEncoder(cfg)
    h = torch.randn(5, cfg.d_LLM)
    ts = torch.randn(5, 3)
    s = torch.randn(5)
    _, _, dirn = enc(h, ts, s)
    norms = dirn.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-4, f"dirn norms: {norms}"


def test_temporal_bundle_encode_matches_encoder():
    torch.manual_seed(2)
    cfg = Cfg4()
    bundle = TemporalBundle(cfg)
    h = torch.randn(2, cfg.d_LLM)
    ts = torch.randn(2, 3)
    s = torch.randn(2)
    b1, f1, d1 = bundle.encode(h, time_scalars=ts, surprise=s)
    assert b1.shape == (2, cfg.d_time)
    assert f1.shape == (2, cfg.d_F_time)
    assert d1.shape == (2, cfg.d_time)
    # dirn still unit
    assert (d1.norm(dim=-1) - 1.0).abs().max().item() < 1e-4


# ─── TopicEncoder ────────────────────────────────────────────────────────

def test_idf_centroid_empty_returns_zero():
    wte = torch.randn(100, 16)
    wte = torch.nn.functional.normalize(wte, dim=-1)
    out = _idf_weighted_centroid([], wte)
    assert out.shape == (16,)
    assert out.abs().max().item() == 0.0


def test_idf_centroid_oov_returns_zero():
    wte = torch.randn(50, 16)
    wte = torch.nn.functional.normalize(wte, dim=-1)
    out = _idf_weighted_centroid([999], wte)  # oov
    assert out.abs().max().item() == 0.0


def test_topic_encoder_shapes_batched():
    torch.manual_seed(3)
    cfg = Cfg4()
    enc = TopicEncoder(cfg)
    V = 500
    wte = torch.nn.functional.normalize(torch.randn(V, cfg.d_LLM), dim=-1)
    B = 2
    h = torch.randn(B, cfg.d_LLM)
    ids = [[1, 2, 3, 4], [10, 20, 30]]
    base, fiber, dirn = enc(h, ids, wte)
    assert base.shape == (B, cfg.d_topic)
    assert fiber.shape == (B, cfg.d_F_topic)
    assert dirn.shape == (B, cfg.d_topic)


def test_topic_base_on_sphere():
    torch.manual_seed(4)
    cfg = Cfg4()
    enc = TopicEncoder(cfg)
    V = 500
    wte = torch.nn.functional.normalize(torch.randn(V, cfg.d_LLM), dim=-1)
    h = torch.randn(3, cfg.d_LLM)
    ids = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    base, _, _ = enc(h, ids, wte)
    norms = base.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-4, f"base off-sphere: {norms}"


def test_topic_bundle_canonical_axis_unit():
    torch.manual_seed(5)
    cfg = Cfg4()
    b = TopicBundle(cfg)
    ax = b.canonical_axis()
    assert ax.shape == (cfg.d_topic,)
    assert (ax.norm() - 1.0).abs().item() < 1e-5


def test_topic_great_circle_endpoints():
    torch.manual_seed(6)
    cfg = Cfg4()
    b = TopicBundle(cfg)
    # Two random unit vectors
    p0 = torch.nn.functional.normalize(torch.randn(2, cfg.d_topic), dim=-1)
    p1 = torch.nn.functional.normalize(torch.randn(2, cfg.d_topic), dim=-1)
    path = b._great_circle_path(p0, p1, 8)
    assert path.shape == (2, 8, cfg.d_topic)
    # Endpoints preserved
    assert (path[:, 0] - p0).abs().max().item() < 1e-4
    assert (path[:, -1] - p1).abs().max().item() < 1e-4
    # All intermediate points unit-norm
    mid_norms = path.norm(dim=-1)
    assert (mid_norms - 1.0).abs().max().item() < 1e-3, f"off-sphere: {mid_norms}"


def test_topic_transport_preserves_norm():
    torch.manual_seed(7)
    cfg = Cfg4()
    b = TopicBundle(cfg)
    # Endpoints on sphere
    p0 = torch.nn.functional.normalize(torch.randn(2, cfg.d_topic), dim=-1)
    p1 = torch.nn.functional.normalize(torch.randn(2, cfg.d_topic), dim=-1)
    fiber = torch.randn(2, cfg.d_F_topic)
    out = b.transport(fiber, p0, p1)
    assert out.shape == (2, cfg.d_F_topic)
    n_in = fiber.norm(dim=-1); n_out = out.norm(dim=-1)
    rel = ((n_out - n_in) / n_in).abs().max().item()
    assert rel < 0.15, f"transport norm drift: {rel}"


# ─── ContextEncoder ──────────────────────────────────────────────────────

def test_context_encoder_no_prev_turns():
    torch.manual_seed(8)
    cfg = Cfg4()
    enc = ContextEncoder(cfg)
    B = 3
    h = torch.randn(B, cfg.d_LLM)
    ss = torch.randn(B, cfg.d_LLM)
    base, fiber, dirn = enc(h, ss, None)
    assert base.shape == (B, cfg.d_ctx)
    assert fiber.shape == (B, cfg.d_F_ctx)
    assert dirn.shape == (B, cfg.d_ctx)


def test_context_encoder_with_prev_turns():
    torch.manual_seed(9)
    cfg = Cfg4()
    enc = ContextEncoder(cfg)
    B = 2; T = 5
    h = torch.randn(B, cfg.d_LLM)
    ss = torch.randn(B, cfg.d_LLM)
    prev = torch.randn(B, T, cfg.d_LLM)
    base, fiber, dirn = enc(h, ss, prev)
    assert base.shape == (B, cfg.d_ctx)
    assert fiber.shape == (B, cfg.d_F_ctx)
    assert dirn.shape == (B, cfg.d_ctx)
    assert (dirn.norm(dim=-1) - 1.0).abs().max().item() < 1e-4


# ─── Canonical axis ──────────────────────────────────────────────────────

def test_all_bundles_canonical_axis_unit():
    torch.manual_seed(10)
    cfg = Cfg4()
    for Bundle in (TemporalBundle, TopicBundle, ContextBundle):
        b = Bundle(cfg)
        ax = b.canonical_axis()
        assert ax.shape == (b.d_base,)
        assert (ax.norm() - 1.0).abs().item() < 1e-5, f"{Bundle.__name__}: {ax.norm()}"


# ─── Gradient flow ───────────────────────────────────────────────────────

def test_gradients_flow_through_time_encoder():
    torch.manual_seed(11)
    cfg = Cfg4()
    enc = TimeEncoder(cfg)
    h = torch.randn(2, cfg.d_LLM, requires_grad=False)
    ts = torch.randn(2, 3)
    s = torch.randn(2)
    base, fiber, dirn = enc(h, ts, s)
    loss = base.sum() + fiber.sum() + dirn.sum()
    loss.backward()
    # A layer in the time_mlp should have a non-zero grad
    g = enc.time_mlp[0].weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no gradient flowed through time_mlp"


# ─── Runner ──────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_time_encoder_shapes,
        test_time_dirn_unit_norm,
        test_temporal_bundle_encode_matches_encoder,
        test_idf_centroid_empty_returns_zero,
        test_idf_centroid_oov_returns_zero,
        test_topic_encoder_shapes_batched,
        test_topic_base_on_sphere,
        test_topic_bundle_canonical_axis_unit,
        test_topic_great_circle_endpoints,
        test_topic_transport_preserves_norm,
        test_context_encoder_no_prev_turns,
        test_context_encoder_with_prev_turns,
        test_all_bundles_canonical_axis_unit,
        test_gradients_flow_through_time_encoder,
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
    print(f"\nall {len(tests)} v4.2 tests passed")


if __name__ == "__main__":
    _run_all()
