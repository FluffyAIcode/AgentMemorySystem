"""v4.3 tests — KakeyaSet + KakeyaRegistry + alignment math."""
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

from ams_v4 import Cfg4
from ams_v4.kakeya.alignment import (
    alignment_error, project_into_pca, pushforward, solve_aligned_t_dir,
)
from ams_v4.kakeya.set import KakeyaSet, _compute_pca, _spherical_kmeans
from ams_v4.kakeya.registry import KakeyaRegistry


# ─── Alignment math ──────────────────────────────────────────────────────

def test_pushforward_matches_matmul():
    axis = torch.randn(8)
    M = torch.randn(8, 32)
    out = pushforward(axis, M)
    assert out.shape == (32,)
    assert torch.allclose(out, axis @ M, atol=1e-6)


def test_project_into_pca_shape():
    d = torch.randn(64)
    basis = torch.randn(10, 64)
    out = project_into_pca(d, basis)
    assert out.shape == (10,)
    assert torch.allclose(out, basis @ d, atol=1e-6)


def test_alignment_error_zero_when_equal():
    v = torch.randn(10)
    v_n = F.normalize(v, dim=0)
    err = alignment_error(v_n, v)  # target will be normalized inside
    assert err < 1e-5


def test_alignment_error_nonzero_when_different():
    a = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=0)
    b = F.normalize(torch.tensor([0.0, 1.0, 0.0]), dim=0)
    err = alignment_error(a, b)
    assert err > 1.0


def test_solve_aligned_t_dir_normalizes():
    target = torch.tensor([3.0, 4.0, 0.0])
    t_dir, err = solve_aligned_t_dir(target)
    assert (t_dir.norm() - 1.0).abs() < 1e-5
    assert err < 1e-5


def test_solve_aligned_t_dir_degenerate():
    target = torch.zeros(4)
    t_dir, err = solve_aligned_t_dir(target)
    assert (t_dir.norm() - 1.0).abs() < 1e-5
    assert err == 1.0


# ─── _compute_pca / _spherical_kmeans ────────────────────────────────────

def test_pca_retains_variance_ratio():
    torch.manual_seed(0)
    N, d = 50, 32
    # Low-rank data: only 4 principal directions have variance
    u = torch.randn(N, 4); v = torch.randn(4, d)
    data = u @ v + 0.01 * torch.randn(N, d)
    basis, mean, d_eff = _compute_pca(data, variance_ratio=0.99)
    # d_eff should be small (close to 4)
    assert 2 <= d_eff <= 10, f"d_eff = {d_eff}"
    assert basis.shape == (d_eff, d)


def test_spherical_kmeans_produces_k_centers():
    torch.manual_seed(0)
    N, d = 50, 8
    dirs = F.normalize(torch.randn(N, d), dim=-1)
    centers, assgn = _spherical_kmeans(dirs, K=5, max_iter=30)
    assert centers.shape == (5, d)
    assert assgn.shape == (N,)
    # Centers are unit-ish
    assert (centers.norm(dim=-1) - 1.0).abs().max() < 0.2


# ─── KakeyaSet ───────────────────────────────────────────────────────────

def _random_field_corpus(N: int, d_field: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    # Low-rank + noise → PCA meaningful
    u = torch.randn(N, 8)
    v = torch.randn(8, d_field)
    return u @ v + 0.1 * torch.randn(N, d_field)


def test_kakeya_set_build_activates():
    cfg = Cfg4()
    vecs = _random_field_corpus(N=30, d_field=64)
    axis_in_field = F.normalize(torch.randn(64), dim=0)
    kset = KakeyaSet(0, "time", ["semantic_emb"], cfg)
    assert not kset.is_active
    kset.build(vecs, axis_in_field)
    assert kset.is_active
    assert kset.skeleton.basis.shape[-1] == 64


def test_kakeya_set_alignment_near_zero():
    cfg = Cfg4()
    vecs = _random_field_corpus(N=30, d_field=64)
    axis_in_field = F.normalize(torch.randn(64), dim=0)
    kset = KakeyaSet(0, "time", ["semantic_emb"], cfg)
    kset.build(vecs, axis_in_field)
    err = kset.verify_alignment(axis_in_field)
    assert err < cfg.kakeya_alignment_tol, (
        f"alignment err {err:.4e} exceeds tol {cfg.kakeya_alignment_tol}"
    )


def test_kakeya_set_encode_decode_roundtrip():
    torch.manual_seed(2)
    cfg = Cfg4()
    N, d_field = 100, 128
    vecs = _random_field_corpus(N, d_field, seed=2)
    axis_in_field = F.normalize(torch.randn(d_field), dim=0)
    kset = KakeyaSet(0, "time", ["semantic_emb"], cfg)
    kset.build(vecs, axis_in_field)

    # Round-trip the training set
    rels = []
    for i in range(N):
        v = vecs[i]
        cv = kset.encode(v)
        v_hat = kset.decode(cv, device=v.device)
        rel = (v - v_hat).norm() / v.norm().clamp(min=1e-8)
        rels.append(rel.item())
    med = sorted(rels)[len(rels) // 2]
    mx = max(rels)
    # §6 invariant 5 uses median ≤ 0.15 as the bar; we allow some outliers
    assert med <= cfg.kakeya_reconstruction_tol, \
        f"median reconstruction error {med} > tol {cfg.kakeya_reconstruction_tol}"
    # Allow some outliers but not wild ones
    assert mx < 3 * cfg.kakeya_reconstruction_tol + 0.2, \
        f"max reconstruction error {mx} too large"


def test_kakeya_set_rejects_wrong_d_field():
    cfg = Cfg4()
    vecs = _random_field_corpus(N=20, d_field=64)
    axis = F.normalize(torch.randn(64), dim=0)
    kset = KakeyaSet(0, "time", ["semantic_emb"], cfg)
    kset.build(vecs, axis)
    try:
        kset.encode(torch.randn(32))
    except AssertionError:
        return
    raise AssertionError("encode should have rejected wrong d_field")


# ─── KakeyaRegistry ──────────────────────────────────────────────────────

def _mk_registry_with_corpus(cfg: Cfg4, N: int = 30):
    # Three field corpora with different d_field; registry uses compression_min_dim
    # but we're testing the registry, so we pick small dims for speed
    torch.manual_seed(3)
    field_corpus = {
        "semantic_emb":       _random_field_corpus(N, 64, seed=10),
        "content_wte_mean":   _random_field_corpus(N, 64, seed=11),
        "context_descriptor": _random_field_corpus(N, 32, seed=12),
    }
    bundle_axes = {
        "time":  F.normalize(torch.randn(cfg.d_time), dim=0),
        "topic": F.normalize(torch.randn(cfg.d_topic), dim=0),
        "ctx":   F.normalize(torch.randn(cfg.d_ctx), dim=0),
    }
    reg = KakeyaRegistry(cfg)
    reg.build(field_corpus, bundle_axes)
    return reg, bundle_axes, field_corpus


def test_registry_default_routing_has_4_sets():
    cfg = Cfg4()
    reg, _, _ = _mk_registry_with_corpus(cfg)
    assert len(reg.sets) == 4, f"expected 4 sets, got {len(reg.sets)}"
    n_active = sum(1 for s in reg.sets if s.is_active)
    assert n_active >= 2, f"at least 2 sets should be active, got {n_active}"


def test_registry_custom_routing():
    cfg = Cfg4()
    reg = KakeyaRegistry(cfg)
    reg.define_sets([
        ("time",  ["semantic_emb"]),
        ("topic", ["semantic_emb"]),
    ])
    assert len(reg._routing) == 2


def test_registry_rejects_short_routing():
    cfg = Cfg4()
    reg = KakeyaRegistry(cfg)
    try:
        reg.define_sets([("time", ["semantic_emb"])])
    except AssertionError as e:
        assert "multiple-kakeya-sets" in str(e) or "≥ 2" in str(e)
        return
    raise AssertionError("expected AssertionError for routing of length 1")


def test_registry_encode_handle_covers_all_fields():
    cfg = Cfg4()
    reg, _, field_corpus = _mk_registry_with_corpus(cfg)
    # One memory's worth of fields (take the first row of each)
    one_mem = {f: field_corpus[f][0] for f in field_corpus}
    handle = reg.encode_memory_fields(one_mem)
    # Every field in routing should be present
    expected_fields = set()
    for _, fs in reg._routing:
        expected_fields.update(fs)
    assert set(handle.entries.keys()) == expected_fields, (
        f"handle fields {set(handle.entries.keys())} != expected {expected_fields}"
    )


def test_registry_decode_field_roundtrip():
    cfg = Cfg4()
    reg, _, field_corpus = _mk_registry_with_corpus(cfg)
    one_mem = {f: field_corpus[f][5] for f in field_corpus}
    handle = reg.encode_memory_fields(one_mem)
    for f, orig in one_mem.items():
        dec = reg.decode_field(handle, f)
        assert dec is not None, f"decode_field returned None for {f}"
        assert dec.shape == orig.shape, \
            f"{f}: decoded shape {tuple(dec.shape)} != original {tuple(orig.shape)}"
        rel = (dec - orig).norm() / orig.norm().clamp(min=1e-8)
        # Looser bar than per-set test because the registry's base_to_field is
        # random-init and one mem may be atypical
        assert rel.item() < 0.5, f"{f}: reconstruction rel err {rel.item()}"


def test_registry_verify_invariants_passes_on_healthy_build():
    cfg = Cfg4()
    reg, bundle_axes, _ = _mk_registry_with_corpus(cfg, N=20)
    errs = reg.verify_invariants(20, bundle_axes=bundle_axes)
    assert errs == [], f"invariants failed: {errs}"


def test_registry_verify_invariants_flags_insufficient_sets():
    cfg = Cfg4()
    reg = KakeyaRegistry(cfg)
    # No build → 0 active sets; with n_entries >= min_entries, invariant 3 fires
    errs = reg.verify_invariants(cfg.kakeya_min_entries + 1)
    assert any("invariant 3" in e for e in errs), errs


# ─── Runner ──────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_pushforward_matches_matmul,
        test_project_into_pca_shape,
        test_alignment_error_zero_when_equal,
        test_alignment_error_nonzero_when_different,
        test_solve_aligned_t_dir_normalizes,
        test_solve_aligned_t_dir_degenerate,
        test_pca_retains_variance_ratio,
        test_spherical_kmeans_produces_k_centers,
        test_kakeya_set_build_activates,
        test_kakeya_set_alignment_near_zero,
        test_kakeya_set_encode_decode_roundtrip,
        test_kakeya_set_rejects_wrong_d_field,
        test_registry_default_routing_has_4_sets,
        test_registry_custom_routing,
        test_registry_rejects_short_routing,
        test_registry_encode_handle_covers_all_fields,
        test_registry_decode_field_roundtrip,
        test_registry_verify_invariants_passes_on_healthy_build,
        test_registry_verify_invariants_flags_insufficient_sets,
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
    print(f"\nall {len(tests)} v4.3 tests passed")


if __name__ == "__main__":
    _run_all()
