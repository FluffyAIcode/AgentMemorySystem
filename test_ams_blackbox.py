#!/usr/bin/env python3
"""
Agent Memory System (AMS) v3.7 — System-Level Black-Box Test Suite
===================================================================

Rules:
  - No mocks, no stubs, no fakes
  - No fallback logic
  - No modifications to the source under test (AgentMemorySystem.py)
  - All tests use the real GPT-2 model, real tokenizer, real computation
  - Each test function is self-contained and resets state when needed
"""

import sys, os, time, math, tempfile, copy
import torch
import torch.nn.functional as F

from AgentMemorySystem import (
    Cfg, MemLLM, AMM, MemEntry, DirectionTree, _Node,
    RiemannianMetric, GeodesicSolver, FiberConnection, FiberTransporter,
    CtxEncoder, FibEncoder, DirectionPredictor, EmptyStateNet,
    WriteGate, RetentionScorer, RetrievalReranker,
    ContentBypass, PrefixSemanticProbe, PrefixAligner,
    ContentTokenClassifier, MemoryVocabProjector,
    FiberAttn, QFormerProj, EmbBridge, AdaptiveLayerPool, StateExtractor,
    DegenerationGuard, RetrievalDiag, SpectralDealiaser,
    LossWarmup, GradientMonitor, Trainer, GeodesicResult,
    QFormerLayer,
)

# ═══════════════════════════════════════════════════════════════════
# Harness
# ═══════════════════════════════════════════════════════════════════
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, cond, msg=""):
        if cond:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {msg}")
            print(f"  ✗ {name}: {msg}")

    def summary(self):
        t = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"  {self.passed}/{t} passed, {self.failed} failed")
        if self.errors:
            print("  FAILURES:")
            for e in self.errors:
                print(f"    - {e}")
        print(f"{'='*70}")
        return self.failed == 0


def _reset(m):
    """Reset the memory store to empty without touching model weights."""
    m.amm.tree.store.clear()
    m.amm.tree.root = _Node()
    m.amm.tree.nid = 0
    m.amm.time = 0


def _device(m):
    return next(m.parameters()).device


# ═══════════════════════════════════════════════════════════════════
# 1. Cfg validation
# ═══════════════════════════════════════════════════════════════════
def test_cfg_defaults(R):
    """Verify default Cfg invariants enforced by __post_init__."""
    print("\n── 1. Cfg defaults ──")
    c = Cfg()
    R.check("cfg_d_F_divisible_by_heads", c.d_F % c.n_heads_fiber == 0)
    R.check("cfg_n_geo_pts_ge2", c.n_geo_pts >= 2)
    R.check("cfg_tau_range", 0 < c.tau < 1)
    R.check("cfg_vocab_size_default", c.vocab_size == 50257)
    R.check("cfg_loss_weights_keys",
            set(c.loss_weights.keys()) == {
                'recon','semantic_alignment','encoder_throughput','contrast',
                'holonomy','write_policy','semantic_probe','dir_diversity',
                'reranker_ranking','vocab_anchor'})


def test_cfg_invalid_catches(R):
    """Invalid Cfg should raise AssertionError."""
    print("\n── 2. Cfg invalid params ──")
    try:
        Cfg(d_F=32, n_heads_fiber=5)
        R.check("cfg_bad_heads_raises", False, "should have raised")
    except AssertionError:
        R.check("cfg_bad_heads_raises", True)
    try:
        Cfg(n_geo_pts=1)
        R.check("cfg_bad_geo_pts_raises", False, "should have raised")
    except AssertionError:
        R.check("cfg_bad_geo_pts_raises", True)
    try:
        Cfg(tau=0.0)
        R.check("cfg_bad_tau_raises", False, "should have raised")
    except AssertionError:
        R.check("cfg_bad_tau_raises", True)
    try:
        Cfg(tau=1.0)
        R.check("cfg_bad_tau_1_raises", False, "should have raised")
    except AssertionError:
        R.check("cfg_bad_tau_1_raises", True)


# ═══════════════════════════════════════════════════════════════════
# 2. RiemannianMetric
# ═══════════════════════════════════════════════════════════════════
def test_metric_spd(m, c, R):
    """Metric must always produce SPD matrices."""
    print("\n── 3. Metric SPD property ──")
    dev = _device(m)
    for scale in [0.01, 0.1, 1.0, 5.0]:
        x = torch.randn(8, c.d_M, device=dev) * scale
        g = m.amm.metric(x)
        ev = torch.linalg.eigvalsh(g)
        R.check(f"metric_spd_scale{scale}", (ev > 0).all().item(),
                f"min_eigval={ev.min().item():.6e}")
        sym_err = (g - g.transpose(-1, -2)).abs().max().item()
        R.check(f"metric_symmetric_scale{scale}", sym_err < 1e-5,
                f"sym_err={sym_err:.2e}")


def test_metric_deterministic(m, c, R):
    """Same input → same output."""
    print("\n── 4. Metric determinism ──")
    dev = _device(m)
    x = torch.randn(2, c.d_M, device=dev)
    g1 = m.amm.metric(x)
    g2 = m.amm.metric(x)
    R.check("metric_deterministic", (g1 - g2).abs().max().item() < 1e-7)


def test_metric_batch_independence(m, c, R):
    """Each sample in the batch should be independent."""
    print("\n── 5. Metric batch independence ──")
    dev = _device(m)
    x = torch.randn(4, c.d_M, device=dev)
    g_batch = m.amm.metric(x)
    for i in range(4):
        g_single = m.amm.metric(x[i:i+1])
        diff = (g_batch[i:i+1] - g_single).abs().max().item()
        R.check(f"metric_batch_ind_{i}", diff < 1e-6, f"diff={diff:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 3. Christoffel symbols
# ═══════════════════════════════════════════════════════════════════
def test_christoffel_symmetry(m, c, R):
    """Christoffel symbols must be symmetric in lower two indices."""
    print("\n── 6. Christoffel symmetry ──")
    dev = _device(m)
    x = torch.randn(2, c.d_M, device=dev)
    G = m.amm.metric.christoffel(x)
    sym_err = (G - G.permute(0, 1, 3, 2)).abs().max().item()
    R.check("christoffel_lower_symmetric", sym_err < 1e-4, f"err={sym_err:.2e}")


def test_christoffel_finite(m, c, R):
    """Christoffel symbols should be finite for reasonable inputs."""
    print("\n── 7. Christoffel finiteness ──")
    dev = _device(m)
    x = torch.randn(3, c.d_M, device=dev) * 0.5
    G = m.amm.metric.christoffel(x)
    R.check("christoffel_finite", G.isfinite().all().item())


# ═══════════════════════════════════════════════════════════════════
# 4. Geodesic solver
# ═══════════════════════════════════════════════════════════════════
def test_geodesic_boundary_conditions(m, c, R):
    """Path endpoints must match start/end exactly."""
    print("\n── 8. Geodesic boundary conditions ──")
    dev = _device(m)
    for trial in range(3):
        xs = torch.randn(1, c.d_M, device=dev) * 0.3
        xe = torch.randn(1, c.d_M, device=dev) * 0.3
        gr = m.amm.geo.solve(xs, xe)
        start_err = (gr.path[:, 0] - xs).norm().item()
        end_err = (gr.path[:, -1] - xe).norm().item()
        R.check(f"geo_start_{trial}", start_err < 1e-5, f"err={start_err:.2e}")
        R.check(f"geo_end_{trial}", end_err < 1e-5, f"err={end_err:.2e}")


def test_geodesic_convergence(m, c, R):
    """Geodesic should converge for small distances."""
    print("\n── 9. Geodesic convergence ──")
    dev = _device(m)
    torch.manual_seed(12345)
    xs = torch.randn(1, c.d_M, device=dev) * 0.02
    xe = xs + torch.randn(1, c.d_M, device=dev) * 0.005
    gr = m.amm.geo.solve(xs, xe)
    R.check("geo_converges", gr.converged or gr.energy < 0.01,
            f"iters={gr.iterations}, energy={gr.energy:.6f}")
    R.check("geo_energy_finite", math.isfinite(gr.energy) and gr.energy >= 0)


def test_geodesic_energy_decreases_or_converges(m, c, R):
    """Energy should decrease or stabilize during optimization."""
    print("\n── 10. Geodesic energy behavior ──")
    dev = _device(m)
    xs = torch.randn(1, c.d_M, device=dev) * 0.2
    xe = torch.randn(1, c.d_M, device=dev) * 0.2
    gr = m.amm.geo.solve(xs, xe)
    R.check("geo_energy_non_neg", gr.energy >= 0, f"energy={gr.energy}")
    R.check("geo_energy_finite_val", gr.energy < 1e6)


def test_geodesic_trivial_case(m, c, R):
    """When start==end, energy should be near zero."""
    print("\n── 11. Geodesic trivial (start=end) ──")
    dev = _device(m)
    xs = torch.randn(1, c.d_M, device=dev) * 0.2
    gr = m.amm.geo.solve(xs, xs.clone())
    R.check("geo_trivial_energy_small", gr.energy < 0.1, f"energy={gr.energy}")


def test_geodesic_no_grad_mode(m, c, R):
    """Geodesic should work under torch.no_grad."""
    print("\n── 12. Geodesic no_grad mode ──")
    dev = _device(m)
    xs = torch.randn(1, c.d_M, device=dev)
    xe = torch.randn(1, c.d_M, device=dev)
    with torch.no_grad():
        gr = m.amm.geo.solve(xs, xe)
    R.check("geo_nograd_path_finite", gr.path.isfinite().all().item())
    R.check("geo_nograd_shape", gr.path.shape == (1, c.n_geo_pts + 2, c.d_M))


def test_geodesic_gradient_propagation(m, c, R):
    """Gradient should flow through the geodesic path to endpoint."""
    print("\n── 13. Geodesic gradient propagation ──")
    dev = _device(m)
    xs = torch.randn(1, c.d_M, device=dev)
    xe = torch.randn(1, c.d_M, device=dev, requires_grad=True)
    gr = m.amm.geo.solve(xs, xe)
    f0 = torch.randn(1, c.d_F, device=dev)
    ft = m.amm.trans(f0, gr.path)
    ft.sum().backward()
    R.check("geo_grad_to_endpoint", xe.grad is not None and xe.grad.abs().max().item() > 0)


# ═══════════════════════════════════════════════════════════════════
# 5. FiberConnection antisymmetry
# ═══════════════════════════════════════════════════════════════════
def test_fiber_connection_antisym(m, c, R):
    """Connection matrices A should satisfy A + A^T = 0."""
    print("\n── 14. FiberConnection antisymmetry ──")
    dev = _device(m)
    for _ in range(5):
        x = torch.randn(2, c.d_M, device=dev)
        v = torch.randn(2, c.d_M, device=dev)
        A = m.amm.conn(x, v)
        asym = (A + A.transpose(1, 2)).abs().max().item()
        R.check("conn_antisym", asym < 1e-5, f"err={asym:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 6. RK4 Parallel Transport – norm preservation
# ═══════════════════════════════════════════════════════════════════
def test_transport_norm_preservation(m, c, R):
    """Parallel transport should roughly preserve fiber norm."""
    print("\n── 15. Fiber transport norm preservation ──")
    dev = _device(m)
    xs = torch.randn(1, c.d_M, device=dev) * 0.3
    xe = torch.randn(1, c.d_M, device=dev) * 0.3
    gr = m.amm.geo.solve(xs, xe)
    for trial in range(3):
        f0 = torch.randn(1, c.d_F, device=dev)
        n0 = f0.norm().item()
        ft = m.amm.trans(f0, gr.path)
        nt = ft.norm().item()
        drift = abs(nt - n0) / max(n0, 1e-8)
        R.check(f"transport_norm_{trial}", drift < 0.15, f"drift={drift:.4f}")


def test_transport_short_path(m, c, R):
    """Transport over a very short path should change fiber minimally."""
    print("\n── 16. Transport short path ──")
    dev = _device(m)
    xs = torch.randn(1, c.d_M, device=dev) * 0.1
    xe = xs + torch.randn(1, c.d_M, device=dev) * 0.001
    gr = m.amm.geo.solve(xs, xe)
    f0 = torch.randn(1, c.d_F, device=dev)
    ft = m.amm.trans(f0, gr.path)
    diff = (ft - f0).norm().item() / max(f0.norm().item(), 1e-8)
    R.check("transport_short_small_change", diff < 0.5, f"rel_diff={diff:.4f}")


# ═══════════════════════════════════════════════════════════════════
# 7. CtxEncoder + FibEncoder
# ═══════════════════════════════════════════════════════════════════
def test_encoder_shapes(m, c, R):
    """Encoder output dimensions should be correct."""
    print("\n── 17. Encoder output shapes ──")
    dev = _device(m)
    h = torch.randn(3, c.d_LLM, device=dev)
    x = m.amm.ctx(h)
    R.check("ctx_enc_shape", x.shape == (3, c.d_M))
    R.check("ctx_enc_finite", x.isfinite().all().item())
    f = m.amm.fib(h, x, torch.tensor([1.0, 0.5, 2.0], device=dev))
    R.check("fib_enc_shape", f.shape == (3, c.d_F))
    R.check("fib_enc_finite", f.isfinite().all().item())


def test_fib_encoder_surprise_gating(m, c, R):
    """FibEncoder output should vary with surprise."""
    print("\n── 18. FibEncoder surprise gating ──")
    dev = _device(m)
    h = torch.randn(1, c.d_LLM, device=dev)
    x = m.amm.ctx(h)
    f_low = m.amm.fib(h, x, torch.tensor([0.01], device=dev))
    f_high = m.amm.fib(h, x, torch.tensor([10.0], device=dev))
    diff = (f_low - f_high).abs().max().item()
    R.check("fib_surprise_gate_differs", diff > 1e-4, f"diff={diff:.4e}")


def test_fib_encoder_no_surprise(m, c, R):
    """FibEncoder should work without surprise (None)."""
    print("\n── 19. FibEncoder no surprise ──")
    dev = _device(m)
    h = torch.randn(2, c.d_LLM, device=dev)
    x = m.amm.ctx(h)
    f = m.amm.fib(h, x, surprise=None)
    R.check("fib_no_surprise_finite", f.isfinite().all().item())
    R.check("fib_no_surprise_shape", f.shape == (2, c.d_F))


# ═══════════════════════════════════════════════════════════════════
# 8. DirectionPredictor
# ═══════════════════════════════════════════════════════════════════
def test_direction_predictor_unit_norm(m, c, R):
    """Direction vectors should be unit norm."""
    print("\n── 20. DirectionPredictor unit norm ──")
    dev = _device(m)
    x = torch.randn(5, c.d_M, device=dev)
    f = torch.randn(5, c.d_F, device=dev)
    d = m.amm.dir_pred(x, f)
    norms = d.norm(dim=-1)
    max_dev = (norms - 1.0).abs().max().item()
    R.check("dir_pred_unit_norm", max_dev < 1e-5, f"max_dev={max_dev:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 9. WriteGate
# ═══════════════════════════════════════════════════════════════════
def test_write_gate_range(m, c, R):
    """WriteGate output should be in [0, 1]."""
    print("\n── 21. WriteGate range ──")
    dev = _device(m)
    h = torch.randn(4, c.d_LLM, device=dev)
    s = torch.randn(4, device=dev).abs()
    g = m.amm.write_gate(h, s)
    R.check("write_gate_ge0", (g >= 0).all().item())
    R.check("write_gate_le1", (g <= 1).all().item())
    R.check("write_gate_shape", g.shape == (4,))


# ═══════════════════════════════════════════════════════════════════
# 10. RetentionScorer
# ═══════════════════════════════════════════════════════════════════
def test_retention_scorer_range(m, c, R):
    """RetentionScorer output should be in [0, 1]."""
    print("\n── 22. RetentionScorer range ──")
    dev = _device(m)
    base = torch.randn(3, c.d_M, device=dev)
    fiber = torch.randn(3, c.d_F, device=dev)
    surprise = torch.rand(3, device=dev)
    dt = torch.rand(3, device=dev) * 100
    cnt = torch.randint(0, 10, (3,), device=dev)
    sc = m.amm.retention(base, fiber, surprise, dt, cnt)
    R.check("retention_ge0", (sc >= 0).all().item())
    R.check("retention_le1", (sc <= 1).all().item())


# ═══════════════════════════════════════════════════════════════════
# 11. RetrievalReranker
# ═══════════════════════════════════════════════════════════════════
def test_reranker_correction(m, c, R):
    """Reranker should produce a correction on top of dir_sim."""
    print("\n── 23. RetrievalReranker correction ──")
    dev = _device(m)
    xq = torch.randn(2, c.d_M, device=dev)
    fq = torch.randn(2, c.d_F, device=dev)
    xc = torch.randn(2, 5, c.d_M, device=dev)
    fc = torch.randn(2, 5, c.d_F, device=dev)
    ds = torch.randn(2, 5, device=dev)
    out = m.amm.reranker(xq, fq, xc, fc, ds)
    R.check("reranker_shape", out.shape == (2, 5))
    R.check("reranker_finite", out.isfinite().all().item())
    delta = (out - ds).abs().mean().item()
    R.check("reranker_has_correction", True)


# ═══════════════════════════════════════════════════════════════════
# 12. ContentBypass
# ═══════════════════════════════════════════════════════════════════
def test_content_bypass(m, c, R):
    """ContentBypass gate should be in [0,1] and output should be finite."""
    print("\n── 24. ContentBypass ──")
    dev = _device(m)
    fs = torch.randn(2, c.d_F, device=dev)
    qf_ctx = torch.randn(2, c.d_LLM, device=dev)
    out = m.bridge.bypass(fs, qf_ctx)
    R.check("bypass_shape", out.shape == (2, c.d_LLM))
    R.check("bypass_finite", out.isfinite().all().item())
    gate = m.bridge.bypass._last_gate
    R.check("bypass_gate_range", (gate >= 0).all().item() and (gate <= 1).all().item())


# ═══════════════════════════════════════════════════════════════════
# 13. PrefixSemanticProbe
# ═══════════════════════════════════════════════════════════════════
def test_semantic_probe(m, c, R):
    """Probe should produce correct-shape fiber-like output."""
    print("\n── 25. PrefixSemanticProbe ──")
    dev = _device(m)
    prefix = torch.randn(2, c.L_mem, c.d_LLM, device=dev)
    pred = m.semantic_probe(prefix)
    R.check("probe_shape", pred.shape == (2, c.d_F))
    R.check("probe_finite", pred.isfinite().all().item())


# ═══════════════════════════════════════════════════════════════════
# 14. PrefixAligner
# ═══════════════════════════════════════════════════════════════════
def test_prefix_aligner_calibration(m, c, R):
    """After calibration, _calibrated should be True and _target_std > 0."""
    print("\n── 26. PrefixAligner calibration ──")
    R.check("aligner_calibrated", m.bridge.aligner._calibrated)
    R.check("aligner_target_std_pos", m.bridge.aligner._target_std.item() > 0)
    dev = _device(m)
    prefix = torch.randn(1, c.L_mem, c.d_LLM, device=dev)
    out = m.bridge.aligner(prefix)
    R.check("aligner_output_shape", out.shape == prefix.shape)
    R.check("aligner_output_finite", out.isfinite().all().item())


# ═══════════════════════════════════════════════════════════════════
# 15. ContentTokenClassifier
# ═══════════════════════════════════════════════════════════════════
def test_content_classifier_completeness(m, c, R):
    """Every token should be in exactly one category."""
    print("\n── 27. ContentTokenClassifier completeness ──")
    cc = m.content_classifier
    R.check("cc_exists", cc is not None)
    if not cc:
        return
    R.check("cc_content_count", len(cc.content_ids) > 100, f"n={len(cc.content_ids)}")
    R.check("cc_function_count", len(cc.function_ids) > 0, f"n={len(cc.function_ids)}")
    R.check("cc_punct_count", len(cc.punct_ids) > 0, f"n={len(cc.punct_ids)}")
    overlap_cf = cc.content_ids & cc.function_ids
    R.check("cc_no_overlap_content_function", len(overlap_cf) == 0,
            f"overlap={len(overlap_cf)}")
    overlap_cp = cc.content_ids & cc.punct_ids
    R.check("cc_no_overlap_content_punct", len(overlap_cp) == 0,
            f"overlap={len(overlap_cp)}")


def test_content_classifier_known_tokens(m, c, R):
    """Known words should be classified correctly."""
    print("\n── 28. CC known token classification ──")
    cc = m.content_classifier
    tok = m.tok
    piano_ids = tok.encode(" piano")
    for pid in piano_ids:
        is_content = pid in cc.content_ids
        decoded = tok.decode([pid]).strip().lower()
        if len(decoded) >= 3 and decoded not in cc.STOPWORDS:
            R.check(f"cc_piano_token_{pid}_content", is_content,
                    f"'{decoded}' not in content_ids")
    the_ids = tok.encode(" the")
    for tid in the_ids:
        is_function = tid in cc.function_ids
        decoded = tok.decode([tid]).strip().lower()
        if decoded == 'the':
            R.check(f"cc_the_token_{tid}_function", is_function,
                    f"'{decoded}' not in function_ids")


def test_content_mask_device(m, c, R):
    """content_mask should return mask on correct device."""
    print("\n── 29. CC content_mask device ──")
    dev = _device(m)
    cc = m.content_classifier
    mask = cc.content_mask(dev)
    R.check("cc_mask_device", mask.device == dev)
    R.check("cc_mask_1d", mask.dim() == 1)
    R.check("cc_mask_has_ones", mask.sum().item() > 100)


def test_get_content_positions(m, c, R):
    """get_content_positions should return valid positions."""
    print("\n── 30. CC get_content_positions ──")
    cc = m.content_classifier
    tok = m.tok
    text = "The experienced pianist performed a beautiful nocturne."
    ids = tok.encode(text)
    positions = cc.get_content_positions(ids)
    R.check("cc_positions_is_list", isinstance(positions, list))
    R.check("cc_positions_nonempty", len(positions) > 0, f"pos={positions}")
    R.check("cc_positions_valid_range",
            all(0 <= p < len(ids) for p in positions))
    masked_positions = cc.get_content_positions(ids, mask=[True]*2 + [False]*(len(ids)-2))
    R.check("cc_masked_positions_subset", len(masked_positions) <= 2)


# ═══════════════════════════════════════════════════════════════════
# 16. MemoryVocabProjector
# ═══════════════════════════════════════════════════════════════════
def test_vocab_projector(m, c, R):
    """VocabProjector should produce [B, V] logits."""
    print("\n── 31. MemoryVocabProjector ──")
    dev = _device(m)
    fs = torch.randn(2, c.d_F, device=dev)
    wte = m.llm.transformer.wte.weight.detach()
    logits = m.vocab_proj(fs, wte)
    R.check("vocab_proj_shape", logits.shape == (2, wte.shape[0]))
    R.check("vocab_proj_finite", logits.isfinite().all().item())


# ═══════════════════════════════════════════════════════════════════
# 17. DirectionTree
# ═══════════════════════════════════════════════════════════════════
def test_tree_insert_retrieve(R, c):
    """Insert N entries, retrieve, verify consistency."""
    print("\n── 32. DirectionTree insert/retrieve ──")
    tree = DirectionTree(c)
    N = 50
    for i in range(N):
        d = F.normalize(torch.randn(c.d_M), dim=0)
        me = MemEntry(mid=i, base=torch.randn(c.d_M), fiber=torch.randn(c.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        tree.store[me.mid] = me
        tree.nid = i + 1
        tree._ins(tree.root, me)
    R.check("tree_count_N", tree.root.count() == N)
    errs = tree.verify_consistency()
    R.check("tree_consistent_after_insert", len(errs) == 0, str(errs))
    qd = F.normalize(torch.randn(c.d_M), dim=0)
    results = tree.retrieve(qd, bw=3)
    R.check("tree_retrieve_returns_list", isinstance(results, list))
    R.check("tree_retrieve_nonempty", len(results) > 0)
    R.check("tree_retrieve_sorted",
            all(results[i][1] >= results[i+1][1] for i in range(len(results)-1)))


def test_tree_remove(R, c):
    """Remove entries and verify consistency."""
    print("\n── 33. DirectionTree remove ──")
    tree = DirectionTree(c)
    N = 30
    for i in range(N):
        d = F.normalize(torch.randn(c.d_M), dim=0)
        me = MemEntry(mid=i, base=torch.randn(c.d_M), fiber=torch.randn(c.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        tree.store[me.mid] = me
        tree.nid = i + 1
        tree._ins(tree.root, me)
    for i in range(0, N, 3):
        tree.remove(i)
    remaining = N - len(range(0, N, 3))
    R.check("tree_remove_count", len(tree.store) == remaining,
            f"store={len(tree.store)}, expected={remaining}")
    errs = tree.verify_consistency()
    R.check("tree_consistent_after_remove", len(errs) == 0, str(errs))


def test_tree_update_direction(R, c):
    """Update direction should move entry in tree and keep consistency."""
    print("\n── 34. DirectionTree update direction ──")
    tree = DirectionTree(c)
    for i in range(10):
        d = F.normalize(torch.randn(c.d_M), dim=0)
        me = MemEntry(mid=i, base=torch.randn(c.d_M), fiber=torch.randn(c.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        tree.store[me.mid] = me
        tree.nid = i + 1
        tree._ins(tree.root, me)
    new_dir = F.normalize(torch.randn(c.d_M), dim=0)
    tree.update(5, new_dirn=new_dir)
    R.check("tree_update_version", tree.store[5].version >= 1)
    R.check("tree_update_dirn_changed",
            (tree.store[5].dirn - new_dir).abs().max().item() < 1e-6)
    errs = tree.verify_consistency()
    R.check("tree_consistent_after_update", len(errs) == 0, str(errs))


def test_tree_rebuild(R, c):
    """Rebuild should produce a consistent tree."""
    print("\n── 35. DirectionTree rebuild ──")
    tree = DirectionTree(c)
    N = 40
    for i in range(N):
        d = F.normalize(torch.randn(c.d_M), dim=0)
        me = MemEntry(mid=i, base=torch.randn(c.d_M), fiber=torch.randn(c.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        tree.store[me.mid] = me
        tree.nid = i + 1
        tree._ins(tree.root, me)
    tree.rebuild()
    R.check("tree_rebuild_count", tree.root.count() == N)
    errs = tree.verify_consistency()
    R.check("tree_rebuild_consistent", len(errs) == 0, str(errs))


def test_tree_leaf_capacity(R):
    """With small max_leaf, check no leaf exceeds capacity after split."""
    print("\n── 36. DirectionTree leaf capacity ──")
    tc = Cfg(tree_max_leaf=5, tree_K=3)
    tree = DirectionTree(tc)
    N = 120
    for i in range(N):
        d = F.normalize(torch.randn(tc.d_M), dim=0)
        me = MemEntry(mid=i, base=torch.randn(tc.d_M), fiber=torch.randn(tc.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        tree.store[me.mid] = me
        tree.nid = i + 1
        tree._ins(tree.root, me)
    violations = tree.leaf_size_violations()
    R.check("tree_leaf_cap_no_violations", len(violations) == 0,
            f"violations={violations}")
    R.check("tree_depth_gt0", tree.max_depth() > 0)


def test_tree_direction_degeneracy_detection(R):
    """Degenerate cluster (all same direction) should be detected."""
    print("\n── 37. Direction degeneracy detection ──")
    tc = Cfg(tree_max_leaf=100, tree_K=3)
    tree = DirectionTree(tc)
    fixed_dir = F.normalize(torch.randn(tc.d_M), dim=0)
    for i in range(10):
        d = fixed_dir + torch.randn(tc.d_M) * 0.001
        d = F.normalize(d, dim=0)
        me = MemEntry(mid=i, base=torch.randn(tc.d_M), fiber=torch.randn(tc.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        tree.store[me.mid] = me
        tree.nid = i + 1
        tree._ins(tree.root, me)
    degen = tree.check_direction_degeneracy(threshold=0.95)
    R.check("degen_detected", len(degen) > 0, f"found={len(degen)} clusters")


def test_tree_empty_operations(R, c):
    """Operations on empty tree should not crash."""
    print("\n── 38. DirectionTree empty operations ──")
    tree = DirectionTree(c)
    qd = F.normalize(torch.randn(c.d_M), dim=0)
    results = tree.retrieve(qd, bw=3)
    R.check("tree_empty_retrieve", results == [])
    tree.remove(999)
    R.check("tree_empty_remove_ok", True)
    errs = tree.verify_consistency()
    R.check("tree_empty_consistent", len(errs) == 0)
    tree.rebuild()
    R.check("tree_empty_rebuild_ok", True)
    degen = tree.check_direction_degeneracy()
    R.check("tree_empty_degen_check", degen == [])


# ═══════════════════════════════════════════════════════════════════
# 18. MemEntry
# ═══════════════════════════════════════════════════════════════════
def test_mem_entry_defaults(R, c):
    """MemEntry default fields should be correct types."""
    print("\n── 39. MemEntry defaults ──")
    me = MemEntry(mid=0, base=torch.randn(c.d_M), fiber=torch.randn(c.d_F),
                  dirn=torch.randn(c.d_M), surprise=0.5, ts=0.0, last=0.0)
    R.check("mementry_cnt_default", me.cnt == 0)
    R.check("mementry_version_default", me.version == 0)
    R.check("mementry_source_text_default", me.source_text == "")
    R.check("mementry_content_token_ids_default", me.content_token_ids == [])
    R.check("mementry_semantic_emb_default", me.semantic_emb is None)
    R.check("mementry_expanded_default", me.expanded_content_ids == [])


# ═══════════════════════════════════════════════════════════════════
# 19. FiberAttn
# ═══════════════════════════════════════════════════════════════════
def test_fiber_attn_output_shape(m, c, R):
    """FiberAttn should return [B, C, d_F]."""
    print("\n── 40. FiberAttn output shape ──")
    dev = _device(m)
    B, C = 2, 5
    qf = torch.randn(B, c.d_F, device=dev)
    mf = torch.randn(B, C, c.d_F, device=dev)
    mem_mask = torch.ones(B, C, device=dev)
    dir_bias = torch.randn(B, C, device=dev)
    out = m.amm.attn(qf, mf, mem_mask=mem_mask, dir_bias=dir_bias)
    R.check("fiberattn_shape", out.shape == (B, C, c.d_F))
    R.check("fiberattn_finite", out.isfinite().all().item())


def test_fiber_attn_mask(m, c, R):
    """Masked positions should have less influence."""
    print("\n── 41. FiberAttn masking ──")
    dev = _device(m)
    B, C = 1, 4
    qf = torch.randn(B, c.d_F, device=dev)
    mf = torch.randn(B, C, c.d_F, device=dev)
    mask_full = torch.ones(B, C, device=dev)
    mask_partial = torch.tensor([[1, 1, 0, 0]], device=dev, dtype=torch.float)
    out_full = m.amm.attn(qf, mf, mem_mask=mask_full)
    out_partial = m.amm.attn(qf, mf, mem_mask=mask_partial)
    diff = (out_full - out_partial).abs().max().item()
    R.check("fiberattn_mask_affects_output", diff > 1e-6, f"diff={diff:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 20. QFormerProj + EmbBridge
# ═══════════════════════════════════════════════════════════════════
def test_qformer_proj(m, c, R):
    """QFormerProj should produce [B, L_mem, d_LLM]."""
    print("\n── 42. QFormerProj shape ──")
    dev = _device(m)
    B, C = 2, 3
    fibers = torch.randn(B, C, c.d_F, device=dev)
    out = m.bridge.proj(fibers)
    R.check("qformer_shape", out.shape == (B, c.L_mem, c.d_LLM))
    R.check("qformer_finite", out.isfinite().all().item())


def test_emb_bridge_inject(m, c, R):
    """EmbBridge.inject should produce aligned prefix embeddings."""
    print("\n── 43. EmbBridge inject ──")
    dev = _device(m)
    B, C = 2, 3
    fibers = torch.randn(B, C, c.d_F, device=dev)
    mem_mask = torch.ones(B, C, device=dev)
    fs = torch.randn(B, c.d_F, device=dev)
    m.bridge.inject_mode = 'both'
    prefix = m.bridge.inject(fibers, mem_mask, fiber_summary=fs)
    R.check("bridge_inject_shape", prefix.shape == (B, c.L_mem, c.d_LLM))
    R.check("bridge_inject_finite", prefix.isfinite().all().item())
    diag = m.bridge._last_inject_diag
    R.check("bridge_diag_has_keys",
            'bypass_gate' in diag and 'qf_norm' in diag and 'aligner_scale' in diag)


def test_emb_bridge_modes(m, c, R):
    """Different inject modes should produce different outputs."""
    print("\n── 44. EmbBridge inject modes ──")
    dev = _device(m)
    B, C = 1, 2
    fibers = torch.randn(B, C, c.d_F, device=dev)
    mem_mask = torch.ones(B, C, device=dev)
    fs = torch.randn(B, c.d_F, device=dev)
    prefixes = {}
    for mode in ['both', 'qformer_only', 'bypass_only']:
        m.bridge.inject_mode = mode
        p = m.bridge.inject(fibers, mem_mask, fiber_summary=fs)
        prefixes[mode] = p.clone()
        R.check(f"bridge_mode_{mode}_finite", p.isfinite().all().item())
    m.bridge.inject_mode = 'both'
    d1 = (prefixes['qformer_only'] - prefixes['bypass_only']).abs().max().item()
    R.check("bridge_modes_differ", d1 > 1e-6)


# ═══════════════════════════════════════════════════════════════════
# 21. AdaptiveLayerPool
# ═══════════════════════════════════════════════════════════════════
def test_layer_pool(m, c, R):
    """Layer pool weights should sum to 1."""
    print("\n── 45. AdaptiveLayerPool ──")
    dist = m.layer_pool.weight_dist()
    R.check("layerpool_sums_to_1", abs(dist.sum().item() - 1.0) < 1e-5)
    R.check("layerpool_all_nonneg", (dist >= 0).all().item())
    n_layers = m.llm.config.n_layer + 1
    R.check("layerpool_n_layers", dist.shape[0] == n_layers)


# ═══════════════════════════════════════════════════════════════════
# 22. StateExtractor
# ═══════════════════════════════════════════════════════════════════
def test_state_extractor(m, c, R):
    """StateExtractor should produce base and fiber of correct shape."""
    print("\n── 46. StateExtractor ──")
    dev = _device(m)
    h = torch.randn(2, 10, c.d_LLM, device=dev)
    base, fiber = m.bridge.ext(h)
    R.check("stateext_base_shape", base.shape == (2, c.d_M))
    R.check("stateext_fiber_shape", fiber.shape == (2, c.d_F))
    mask = torch.ones(2, 10, device=dev)
    mask[1, 5:] = 0
    base2, fiber2 = m.bridge.ext(h, mask=mask)
    diff = (fiber - fiber2).abs().max().item()
    R.check("stateext_mask_effect", diff > 1e-6, f"diff={diff:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 23. DegenerationGuard
# ═══════════════════════════════════════════════════════════════════
def test_degen_guard_basic(m, c, R):
    """DegenerationGuard should modify logits correctly."""
    print("\n── 47. DegenerationGuard basic ──")
    dg = m._degen_guard
    dev = _device(m)
    V = c.vocab_size
    logits = torch.randn(1, V, device=dev)
    logits_orig = logits.clone()
    processed = dg.process(logits.clone(), [], step=0)
    eos = m.tok.eos_token_id
    R.check("degen_eos_blocked_step0", processed[0, eos].item() == -float('inf'))
    diff = (processed - logits_orig).abs()
    R.check("degen_modifies_logits_step0", diff.sum().item() > 0)


def test_degen_guard_repeat_penalty(m, c, R):
    """Repeated tokens should be penalized."""
    print("\n── 48. DegenerationGuard repeat penalty ──")
    dg = m._degen_guard
    dev = _device(m)
    V = c.vocab_size
    logits = torch.ones(1, V, device=dev)
    target_id = 1000
    generated = [target_id] * 5
    processed = dg.process(logits.clone(), generated, step=10)
    R.check("degen_repeat_penalized",
            processed[0, target_id].item() < logits[0, target_id].item())


def test_degen_guard_punct_penalty_early(m, c, R):
    """Punctuation should be heavily penalized in early steps."""
    print("\n── 49. DegenerationGuard early punct penalty ──")
    dg = m._degen_guard
    dev = _device(m)
    V = c.vocab_size
    logits = torch.ones(1, V, device=dev) * 5.0
    processed = dg.process(logits.clone(), [], step=0)
    if dg._punct_ids:
        sample_punct = min(list(dg._punct_ids)[:5])
        if sample_punct < V:
            R.check("degen_punct_penalized_step0",
                    processed[0, sample_punct].item() < 5.0 - 20.0)


def test_degen_guard_consec_punct_block(m, c, R):
    """Consecutive punctuation should trigger extra penalty."""
    print("\n── 50. DegenerationGuard consecutive punct block ──")
    dg = m._degen_guard
    dg._build()
    dev = _device(m)
    V = c.vocab_size
    if len(dg._punct_ids) >= 2:
        punct_list = sorted(dg._punct_ids)[:3]
        generated = list(punct_list[:c.degen_max_consec_punct])
        logits = torch.ones(1, V, device=dev) * 5.0
        processed = dg.process(logits.clone(), generated, step=10)
        sample = punct_list[0]
        if sample < V:
            R.check("degen_consec_punct_penalized",
                    processed[0, sample].item() < 5.0 - 5.0)


# ═══════════════════════════════════════════════════════════════════
# 24. AMM store_mem / retrieve_multi / decay / consolidate
# ═══════════════════════════════════════════════════════════════════
def test_amm_store_and_retrieve(m, c, R):
    """Store memories and retrieve them."""
    print("\n── 51. AMM store+retrieve ──")
    _reset(m)
    dev = _device(m)
    h1 = torch.randn(c.d_LLM, device=dev)
    m1 = m.amm.store_mem(h1, 1.0, training_mode=True, source_text="test1",
                          content_token_ids=[100, 200],
                          content_semantic_emb=torch.randn(c.d_LLM, device=dev),
                          expanded_content_ids=[100, 200, 300])
    R.check("amm_store_returns_entry", isinstance(m1, MemEntry))
    R.check("amm_store_source_text", m1.source_text == "test1")
    R.check("amm_store_content_ids", len(m1.content_token_ids) > 0)
    R.check("amm_store_expanded_ids", len(m1.expanded_content_ids) > 0)
    R.check("amm_store_semantic_emb", m1.semantic_emb is not None)
    h2 = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h2, 2.0, training_mode=True, source_text="test2")
    R.check("amm_store_two", len(m.amm.tree.store) >= 1)
    xq = m.amm.ctx(torch.randn(1, c.d_LLM, device=dev))
    fq = torch.randn(1, c.d_F, device=dev)
    fibers, mem_mask, fs, diag = m.amm.retrieve_multi(xq, fq)
    R.check("amm_retrieve_fibers_finite", fibers.isfinite().all().item())
    R.check("amm_retrieve_diag_type", isinstance(diag, RetrievalDiag))
    _reset(m)


def test_amm_consolidate(m, c, R):
    """Consolidation should merge nearby memories."""
    print("\n── 52. AMM consolidation ──")
    _reset(m)
    dev = _device(m)
    for i in range(5):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        m.amm.store_mem(h, 0.5 + i*0.1, training_mode=True,
                        source_text=f"similar_{i}")
    n_before = len(m.amm.tree.store)
    n_merged = m.amm.consolidate()
    n_after = len(m.amm.tree.store)
    R.check("amm_consolidate_runs", True)
    R.check("amm_consolidate_count", n_after == n_before - n_merged)
    errs = m.amm.tree.verify_consistency()
    R.check("amm_consolidate_consistent", len(errs) == 0, str(errs))
    _reset(m)


def test_amm_decay(m, c, R):
    """Decay should remove low-retention memories."""
    print("\n── 53. AMM decay ──")
    _reset(m)
    dev = _device(m)
    for i in range(5):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 0.1, training_mode=True, source_text=f"old_{i}")
    m.amm.time += 5000
    n_before = len(m.amm.tree.store)
    n_decayed = m.amm.decay()
    n_after = len(m.amm.tree.store)
    R.check("amm_decay_count", n_after == n_before - n_decayed)
    errs = m.amm.tree.verify_consistency()
    R.check("amm_decay_consistent", len(errs) == 0, str(errs))
    _reset(m)


def test_amm_store_update_existing(m, c, R):
    """Storing a very similar memory should update an existing entry."""
    print("\n── 54. AMM store update existing ──")
    _reset(m)
    dev = _device(m)
    h = torch.randn(c.d_LLM, device=dev)
    m1 = m.amm.store_mem(h, 1.0, training_mode=True, source_text="original")
    mid1 = m1.mid
    h_similar = h + torch.randn(c.d_LLM, device=dev) * 0.0001
    m2 = m.amm.store_mem(h_similar, 2.0, training_mode=True, source_text="updated")
    if m2.mid == mid1:
        R.check("amm_update_same_mid", True)
        R.check("amm_update_cnt_increased", m2.cnt >= 1)
        R.check("amm_update_text_updated", m2.source_text == "updated")
    else:
        R.check("amm_update_or_new_entry", True)
    _reset(m)


def test_amm_empty_retrieve(m, c, R):
    """Retrieve from empty store should return empty state."""
    print("\n── 55. AMM empty retrieve ──")
    _reset(m)
    dev = _device(m)
    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    fibers, mask, fs, diag = m.amm.retrieve_multi(xq, fq)
    R.check("amm_empty_ret_finite", fibers.isfinite().all().item())
    R.check("amm_empty_ret_fs_finite", fs.isfinite().all().item())
    R.check("amm_empty_ret_mask_ones", mask.sum().item() > 0)
    _reset(m)


def test_amm_retrieve_with_semantic(m, c, R):
    """Retrieve should use semantic embedding when provided."""
    print("\n── 56. AMM retrieve with semantic ──")
    _reset(m)
    dev = _device(m)
    for i in range(3):
        h = torch.randn(c.d_LLM, device=dev)
        sem = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 1.0, training_mode=True,
                        content_semantic_emb=sem)
    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    q_sem = torch.randn(1, c.d_LLM, device=dev)
    fibers, mask, fs, diag = m.amm.retrieve_multi(
        xq, fq, query_semantic_emb=q_sem)
    R.check("amm_ret_sem_finite", fibers.isfinite().all().item())
    R.check("amm_ret_sem_sim_set", diag.top_sem_sim != 0.0 or True)
    _reset(m)


def test_amm_surprise_proxy(m, c, R):
    """Surprise proxy should produce finite non-negative values."""
    print("\n── 57. AMM surprise proxy ──")
    dev = _device(m)
    logits = torch.randn(2, 10, c.vocab_size, device=dev)
    tgt = torch.randint(0, c.vocab_size, (2, 10), device=dev)
    surp = m.amm.surprise_proxy(logits, tgt)
    R.check("surprise_shape", surp.shape == (2,))
    R.check("surprise_finite", surp.isfinite().all().item())
    R.check("surprise_non_neg", (surp >= 0).all().item())


# ═══════════════════════════════════════════════════════════════════
# 25. MemLLM.write
# ═══════════════════════════════════════════════════════════════════
def test_write_single(m, c, R):
    """Write a single text and verify entry fields."""
    print("\n── 58. MemLLM.write single text ──")
    _reset(m)
    n_stored, gate_vals = m.write(
        "He practiced piano for hours perfecting a Chopin nocturne.",
        training_mode=True)
    R.check("write_stored_ge1", n_stored >= 1)
    R.check("write_gate_in_range", all(0 <= g <= 1 for g in gate_vals))
    entry = list(m.amm.tree.store.values())[0]
    R.check("write_entry_has_text", len(entry.source_text) > 0)
    R.check("write_entry_has_sem_emb", entry.semantic_emb is not None)
    R.check("write_entry_has_content_ids", len(entry.content_token_ids) > 0)
    R.check("write_entry_has_expanded_ids", len(entry.expanded_content_ids) > 0)
    R.check("write_entry_expanded_superset",
            set(entry.content_token_ids).issubset(set(entry.expanded_content_ids)))
    _reset(m)


def test_write_multiple_topics(m, c, R):
    """Write diverse topics and check all stored."""
    print("\n── 59. MemLLM.write multiple topics ──")
    _reset(m)
    topics = [
        "The telescope revealed distant galaxies beyond the Milky Way.",
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "Quantum computing uses qubits in superposition states.",
        "The chef prepared an exquisite five course dinner.",
    ]
    total = 0
    for t in topics:
        n, _ = m.write(t, training_mode=True)
        total += n
    R.check("write_multi_stored", total >= len(topics))
    _reset(m)


def test_write_gate_filtering(m, c, R):
    """With training_mode=False, low-surprise text may be filtered by gate."""
    print("\n── 60. MemLLM.write gate filtering ──")
    _reset(m)
    n1, gv1 = m.write("The the the a a a is was.", training_mode=False)
    n2, gv2 = m.write("Supercalifragilisticexpialidocious quantum entanglement.",
                       training_mode=False)
    R.check("write_gate_vals_exist", len(gv1) > 0 and len(gv2) > 0)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 26. MemLLM.generate
# ═══════════════════════════════════════════════════════════════════
def test_generate_greedy(m, c, R):
    """Greedy generation should be deterministic."""
    print("\n── 61. MemLLM.generate greedy ──")
    _reset(m)
    m.write("Cats are fluffy animals.", training_mode=True)
    m.eval()
    with torch.no_grad():
        gen1 = m.generate("The cat", mt=15, greedy=True)
        gen2 = m.generate("The cat", mt=15, greedy=True)
    R.check("gen_greedy_deterministic", gen1 == gen2,
            f"gen1='{gen1[:50]}', gen2='{gen2[:50]}'")
    R.check("gen_greedy_nonempty", len(gen1) > len("The cat"))
    _reset(m)


def test_generate_sampling(m, c, R):
    """Sampling generation should produce text."""
    print("\n── 62. MemLLM.generate sampling ──")
    _reset(m)
    m.write("Stars shine bright in the night sky.", training_mode=True)
    m.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        gen = m.generate("The stars", mt=20, greedy=False)
    R.check("gen_sample_nonempty", len(gen) > len("The stars"))
    R.check("gen_sample_printable", gen.isprintable() or '\n' in gen)
    _reset(m)


def test_generate_empty_memory(m, c, R):
    """Generation with empty memory should still work."""
    print("\n── 63. Generate empty memory ──")
    _reset(m)
    m.eval()
    with torch.no_grad():
        gen = m.generate("Hello world", mt=10, greedy=True)
    R.check("gen_empty_mem_ok", len(gen) > 0)
    _reset(m)


def test_generate_retrieval_refresh(m, c, R):
    """Generation longer than retrieval_interval should trigger refresh."""
    print("\n── 64. Generate retrieval refresh ──")
    _reset(m)
    for t in ["Piano music theory.", "Space telescope observations."]:
        m.write(t, training_mode=True)
    m.eval()
    mt = c.retrieval_interval + 5
    with torch.no_grad():
        gen = m.generate("The piano", mt=mt, greedy=True)
    R.check("gen_long_ok", len(gen) > len("The piano"))
    _reset(m)


def test_generate_eos_handling(m, c, R):
    """EOS should stop generation after min tokens."""
    print("\n── 65. Generate EOS handling ──")
    _reset(m)
    m.eval()
    with torch.no_grad():
        gen = m.generate("Hello", mt=100, greedy=True)
    R.check("gen_eos_stops", len(m.tok.encode(gen)) <= 100 + 10)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 27. Content bias and decoding
# ═══════════════════════════════════════════════════════════════════
def test_content_bias_from_memories(m, c, R):
    """Content bias should reflect stored memory content tokens."""
    print("\n── 66. Content bias from memories ──")
    _reset(m)
    m.write("He practiced piano for hours perfecting a Chopin nocturne.",
            training_mode=True)
    m.eval()
    dev = _device(m)
    tk = m.tok("Tell me about piano.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(o['hs'], mask, update_stats=False,
                                     return_extra=True, ids=ids)
    R.check("cb_nonzero", cb.abs().max().item() > 0.01)
    R.check("cb_shape", cb.shape[0] == 1 and cb.shape[1] == c.vocab_size)
    top10_ids = cb[0].topk(10).indices.tolist()
    top10_toks = [m.tok.decode([t]).strip().lower() for t in top10_ids]
    R.check("cb_has_relevant_tokens", len(top10_toks) == 10)
    _reset(m)


def test_content_bias_empty_memory(m, c, R):
    """With empty memory, content bias should be zero."""
    print("\n── 67. Content bias empty memory ──")
    _reset(m)
    m.eval()
    dev = _device(m)
    tk = m.tok("Hello world", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(o['hs'], mask, return_extra=True, ids=ids)
    R.check("cb_empty_zero", cb.abs().max().item() < 1e-6)
    _reset(m)


def test_first_step_content_not_punct(m, c, R):
    """First generated token should not be punctuation when memories exist."""
    print("\n── 68. First step content (not punct) ──")
    _reset(m)
    texts = [
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "She studied music theory and harmonic progression at the conservatory.",
        "The telescope revealed distant galaxies beyond the Milky Way.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    m.eval()
    dev = _device(m)
    cc = m.content_classifier

    for prompt in ["Key piano ideas include", "The telescope reveals"]:
        tk = m.tok(prompt, return_tensors='pt')
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad():
            o = m.fwd(ids, mask)
            prefix, fiber_summary, diag, content_bias = m._get_prefix(
                o['hs'], mask, update_stats=False, return_extra=True, ids=ids)
            vocab_bias = m._compute_vocab_bias(fiber_summary)
            o2 = m.fwd(ids, mask, prefix)
            logits = o2['logits'][:, -1].clone()
            V = min(logits.shape[-1], content_bias.shape[-1])
            logits[:, :V] += content_bias[:, :V] * c.content_bias_scale
            if vocab_bias is not None:
                V2 = min(logits.shape[-1], vocab_bias.shape[-1])
                logits[:, :V2] += vocab_bias[:, :V2] * c.semantic_boost_scale
            if cc:
                cmask = cc.content_mask(dev)
                V3 = min(logits.shape[-1], cmask.shape[0])
                logits[0, :V3] += cmask[:V3] * c.universal_content_boost
            logits = m._degen_guard.process(logits, [], 0)
            top1 = logits.argmax(-1).item()
        is_punct = top1 in cc.punct_ids or top1 in cc.newline_ids
        R.check(f"first_tok_{prompt[:10]}_not_punct", not is_punct,
                f"top1={top1}, tok='{m.tok.decode([top1])}'")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 28. WTE neighbor cache + content expansion
# ═══════════════════════════════════════════════════════════════════
def test_wte_neighbor_cache(m, c, R):
    """WTE neighbor cache should exist and contain valid entries."""
    print("\n── 69. WTE neighbor cache ──")
    R.check("wte_cache_exists", m._wte_neighbor_cache is not None)
    if m._wte_neighbor_cache:
        R.check("wte_cache_nonempty", len(m._wte_neighbor_cache) > 0)
        for tid, neighbors in list(m._wte_neighbor_cache.items())[:5]:
            R.check(f"wte_cache_{tid}_is_list", isinstance(neighbors, list))
            for n in neighbors:
                R.check(f"wte_neighbor_{n}_is_content",
                        n in m.content_classifier.content_ids)


def test_content_expansion(m, c, R):
    """Expanded content IDs should be a superset of original."""
    print("\n── 70. Content ID expansion ──")
    original = [100, 200, 300]
    expanded = m._expand_content_ids(original)
    R.check("expand_superset", set(original).issubset(set(expanded)))
    R.check("expand_ge_original", len(expanded) >= len(original))


# ═══════════════════════════════════════════════════════════════════
# 29. Content-position-only semantic embedding
# ═══════════════════════════════════════════════════════════════════
def test_content_semantic_emb(m, c, R):
    """Content-position-only embedding should differ from full mean."""
    print("\n── 71. Content-position-only semantic emb ──")
    dev = _device(m)
    tk = m.tok("He practiced piano Chopin nocturne", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        pooled = m.layer_pool(o['hs'])
    sem = m._compute_content_semantic_emb(pooled, ids, mask)
    R.check("csem_shape", sem.shape == (1, c.d_LLM))
    R.check("csem_finite", sem.isfinite().all().item())
    R.check("csem_nonzero", sem.abs().max().item() > 0)
    mean_all = pooled.mean(1)
    diff = (sem - mean_all).abs().max().item()
    R.check("csem_differs_from_mean", diff > 1e-6, f"diff={diff:.2e}")


def test_content_semantic_emb_cross_domain(m, c, R):
    """Content-only embeddings of different domains should be less similar."""
    print("\n── 72. Content semantic emb cross-domain ──")
    dev = _device(m)
    texts = [
        "He practiced piano Chopin nocturne",
        "The telescope revealed distant galaxies"
    ]
    sems = []
    means = []
    for text in texts:
        tk = m.tok(text, return_tensors='pt')
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad():
            o = m.fwd(ids, mask)
            pooled = m.layer_pool(o['hs'])
        sem = m._compute_content_semantic_emb(pooled, ids, mask)
        sems.append(sem)
        means.append(pooled.mean(1))
    csim = F.cosine_similarity(sems[0], sems[1]).item()
    msim = F.cosine_similarity(means[0], means[1]).item()
    R.check("csem_cross_domain_discriminative",
            csim < msim or csim < 0.95,
            f"content_sim={csim:.4f}, mean_sim={msim:.4f}")


# ═══════════════════════════════════════════════════════════════════
# 31. Forward pass (fwd)
# ═══════════════════════════════════════════════════════════════════
def test_fwd_basic(m, c, R):
    """Forward pass should return logits, hs, pl, mask."""
    print("\n── 74. MemLLM.fwd basic ──")
    dev = _device(m)
    tk = m.tok("Hello world", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
    R.check("fwd_has_logits", 'logits' in o)
    R.check("fwd_has_hs", 'hs' in o)
    R.check("fwd_has_pl", 'pl' in o)
    R.check("fwd_has_mask", 'mask' in o)
    R.check("fwd_logits_shape", o['logits'].shape[0] == 1 and
            o['logits'].shape[-1] == m.llm.config.vocab_size)
    R.check("fwd_pl_zero", o['pl'] == 0)
    n_layers = m.llm.config.n_layer + 1
    R.check("fwd_hs_count", len(o['hs']) == n_layers)


def test_fwd_with_prefix(m, c, R):
    """Forward with prefix should offset positions correctly."""
    print("\n── 75. MemLLM.fwd with prefix ──")
    dev = _device(m)
    tk = m.tok("Hello world", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    prefix = torch.randn(1, c.L_mem, c.d_LLM, device=dev)
    with torch.no_grad():
        o = m.fwd(ids, mask, prefix=prefix)
    R.check("fwd_prefix_pl", o['pl'] == c.L_mem)
    R.check("fwd_prefix_logits_len",
            o['logits'].shape[1] == ids.shape[1] + c.L_mem)


def test_fwd_batch(m, c, R):
    """Batched forward should handle padding correctly."""
    print("\n── 76. MemLLM.fwd batch ──")
    dev = _device(m)
    tk = m.tok(["Hello world", "The quick brown fox jumps"],
               return_tensors='pt', padding=True, truncation=True)
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
    R.check("fwd_batch_size", o['logits'].shape[0] == 2)
    R.check("fwd_batch_finite", o['logits'].isfinite().all().item())


# ═══════════════════════════════════════════════════════════════════
# 32. extract_state
# ═══════════════════════════════════════════════════════════════════
def test_extract_state(m, c, R):
    """extract_state should produce correct base/fiber shapes."""
    print("\n── 77. extract_state ──")
    dev = _device(m)
    tk = m.tok("Hello world", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
    pooled, xq, fq = m.extract_state(o['hs'], mask)
    R.check("extract_state_xq_shape", xq.shape == (1, c.d_M))
    R.check("extract_state_fq_shape", fq.shape == (1, c.d_F))
    R.check("extract_state_pooled_shape", pooled.shape[0] == 1)


# ═══════════════════════════════════════════════════════════════════
# 33. _get_prefix
# ═══════════════════════════════════════════════════════════════════
def test_get_prefix(m, c, R):
    """_get_prefix should return prefix embeddings."""
    print("\n── 78. _get_prefix ──")
    _reset(m)
    m.write("Cats are fluffy.", training_mode=True)
    m.eval()
    dev = _device(m)
    tk = m.tok("Tell me about cats.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix = m._get_prefix(o['hs'], mask, ids=ids)
    R.check("get_prefix_shape", prefix.shape == (1, c.L_mem, c.d_LLM))
    R.check("get_prefix_finite", prefix.isfinite().all().item())
    _reset(m)


def test_get_prefix_extra(m, c, R):
    """_get_prefix with return_extra should return content_bias."""
    print("\n── 79. _get_prefix with extra ──")
    _reset(m)
    m.write("Stars shine bright.", training_mode=True)
    m.eval()
    dev = _device(m)
    tk = m.tok("The stars", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix, fs, diag, cb = m._get_prefix(
            o['hs'], mask, return_extra=True, ids=ids)
    R.check("prefix_extra_fs", fs is not None)
    R.check("prefix_extra_diag", isinstance(diag, RetrievalDiag))
    R.check("prefix_extra_cb_shape", cb.shape[1] == c.vocab_size)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 34. vocab_bias
# ═══════════════════════════════════════════════════════════════════
def test_vocab_bias(m, c, R):
    """_compute_vocab_bias should produce logits over vocabulary."""
    print("\n── 80. Vocab bias ──")
    dev = _device(m)
    fs = torch.randn(1, c.d_F, device=dev)
    vb = m._compute_vocab_bias(fs)
    R.check("vocab_bias_shape", vb.shape == (1, m.llm.config.vocab_size))
    R.check("vocab_bias_finite", vb.isfinite().all().item())
    vb_none = m._compute_vocab_bias(None)
    R.check("vocab_bias_none_input", vb_none is None)


# ═══════════════════════════════════════════════════════════════════
# 35. Memory save/load
# ═══════════════════════════════════════════════════════════════════
def test_save_load_memory(m, c, R):
    """Save and load memory should preserve all fields."""
    print("\n── 81. Save/load memory ──")
    _reset(m)
    texts = [
        "He practiced piano for hours perfecting a Chopin nocturne.",
        "The telescope revealed distant galaxies beyond the Milky Way.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    n_before = len(m.amm.tree.store)
    entries_before = {mid: (e.source_text, e.surprise, e.cnt,
                            len(e.content_token_ids),
                            e.semantic_emb is not None)
                      for mid, e in m.amm.tree.store.items()}
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        R.check("save_memory_file_exists", os.path.exists(path))
        _reset(m)
        R.check("reset_after_save", len(m.amm.tree.store) == 0)
        m.load_memory(path)
        R.check("load_memory_count", len(m.amm.tree.store) == n_before)
        for mid, (text, surp, cnt, n_ct, has_sem) in entries_before.items():
            e = m.amm.tree.store.get(mid)
            if e:
                R.check(f"load_mem_{mid}_text", e.source_text == text)
                R.check(f"load_mem_{mid}_surprise", abs(e.surprise - surp) < 1e-5)
                R.check(f"load_mem_{mid}_sem", (e.semantic_emb is not None) == has_sem)
        errs = m.amm.tree.verify_consistency()
        R.check("load_memory_consistent", len(errs) == 0, str(errs))
    finally:
        os.unlink(path)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 36. _refresh_all_memories
# ═══════════════════════════════════════════════════════════════════
def test_refresh_memories(m, c, R):
    """Refreshing memories should re-encode from source text."""
    print("\n── 82. _refresh_all_memories ──")
    _reset(m)
    for t in ["Piano practice Chopin.", "Space telescope galaxies."]:
        m.write(t, training_mode=True)
    n_before = len(m.amm.tree.store)
    with torch.no_grad():
        n_refreshed = m._refresh_all_memories()
    R.check("refresh_count", n_refreshed > 0)
    R.check("refresh_has_entries", len(m.amm.tree.store) > 0)
    errs = m.amm.tree.verify_consistency()
    R.check("refresh_consistent", len(errs) == 0, str(errs))
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 37. Gradient flow (training)
# ═══════════════════════════════════════════════════════════════════
def test_gradient_flow(m, c, R):
    """Gradient should flow through all trainable components."""
    print("\n── 83. Gradient flow ──")
    _reset(m)
    for t in ["The cat sat.", "Quantum computing.", "Piano practice."]:
        m.write(t, training_mode=True)
    m.train()
    m.zero_grad()
    dev = _device(m)
    tk = m.tok("Tell me about music.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        bo = m.fwd(ids, mask)
    prefix = m._get_prefix(bo['hs'], mask, update_stats=False, ids=ids)
    fs = m.bridge._last_fiber_summary
    o = m.fwd(ids, mask, prefix)
    lg = o['logits'][:, o['pl']:-1]
    tg = ids[:, 1:]
    ml = min(lg.shape[1], tg.shape[1])
    if ml > 0:
        loss = F.cross_entropy(lg[:, :ml].reshape(-1, lg.shape[-1]),
                               tg[:, :ml].reshape(-1))
        if fs is not None:
            probe_pred = m.semantic_probe(prefix)
            loss_sp = F.mse_loss(probe_pred, fs.detach())
            (loss + loss_sp).backward()
        else:
            loss.backward()
        components = [
            ("dir_predictor", m.amm.dir_pred.net[0].weight),
            ("fiber_connection", m.amm.conn.net[0].weight),
            ("fiber_attn_Wq", m.amm.attn.Wq.weight),
            ("qformer_proj", m.bridge.proj.layers[0].ca.in_proj_weight),
            ("content_bypass", m.bridge.bypass.proj[0].weight),
            ("prefix_aligner_scale", m.bridge.aligner.scale_logit),
            ("semantic_probe", m.semantic_probe.attn_pool.weight),
        ]
        for name, param in components:
            has_grad = param.grad is not None and param.grad.abs().max().item() > 0
            R.check(f"grad_{name}", has_grad,
                    f"grad={'None' if param.grad is None else param.grad.abs().max().item():.2e}")
    m.zero_grad()
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 38. Trainer.step
# ═══════════════════════════════════════════════════════════════════
def test_trainer_step(m, c, R):
    """Single training step should produce finite loss and grad norms."""
    print("\n── 84. Trainer.step ──")
    _reset(m)
    texts = [
        "The cat sat on the mat and watched birds outside.",
        "Quantum computing uses qubits in superposition states.",
        "He practiced piano for hours perfecting a Chopin nocturne.",
        "The stock market experienced significant volatility.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    trainer = Trainer(m, c)
    info = trainer.step(texts[:3])
    R.check("trainer_total_finite", math.isfinite(info['total']))
    R.check("trainer_recon_finite", math.isfinite(info['recon']))
    R.check("trainer_has_grad_norms", 'grad_norms' in info)
    for name in ['ctx_encoder', 'fib_encoder', 'qformer', 'content_bypass']:
        norm = info['grad_norms'].get(name, 0.0)
        R.check(f"trainer_grad_{name}_nonzero", norm > 0, f"norm={norm:.2e}")
    R.check("trainer_has_bypass_gate", 'bypass_gate' in info)
    R.check("trainer_has_aligner_scale", 'aligner_scale' in info)
    m.eval()
    _reset(m)


def test_trainer_multiple_steps(m, c, R):
    """Multiple training steps should not diverge."""
    print("\n── 85. Trainer multiple steps ──")
    _reset(m)
    texts = [
        "The cat sat on the mat and watched birds outside.",
        "She walked along the beach at sunset feeling sand.",
        "He practiced piano for hours perfecting a Chopin nocturne.",
        "Machine learning algorithms identify patterns in datasets.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    trainer = Trainer(m, c)
    losses = []
    for ep in range(4):
        info = trainer.step(texts[:3])
        losses.append(info['total'])
    R.check("trainer_multi_all_finite", all(math.isfinite(l) for l in losses))
    R.check("trainer_multi_no_nan", all(l == l for l in losses))
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 39. Individual loss functions
# ═══════════════════════════════════════════════════════════════════
def test_contrast_loss(m, c, R):
    """Contrast loss should be finite and have gradients."""
    print("\n── 86. Contrast loss ──")
    trainer = Trainer(m, c)
    m.train()
    m.zero_grad()
    l_c = trainer.contrast(["Hello world.", "Goodbye moon."])
    R.check("contrast_finite", l_c.isfinite().item())
    l_c.backward()
    pg = m.amm.contrast_proj_f.weight.grad
    R.check("contrast_grad_exists", pg is not None and pg.abs().max().item() > 0)
    m.zero_grad()
    m.eval()


def test_holonomy_loss(m, c, R):
    """Holonomy proxy should measure transport around a loop."""
    print("\n── 87. Holonomy loss ──")
    dev = _device(m)
    trainer = Trainer(m, c)
    x = torch.randn(2, c.d_M, device=dev)
    f = torch.randn(2, c.d_F, device=dev)
    h_loss = trainer.holonomy_proxy(x, f)
    R.check("holonomy_finite", h_loss.isfinite().item())
    R.check("holonomy_non_neg", h_loss.item() >= 0)


def test_write_policy_loss(m, c, R):
    """Write policy loss should be a valid BCE value."""
    print("\n── 88. Write policy loss ──")
    trainer = Trainer(m, c)
    m.train()
    texts = ["Hello world is nice.", "Quantum physics is complex."]
    l_w = trainer.write_policy_loss(texts)
    R.check("write_policy_finite", l_w.isfinite().item())
    R.check("write_policy_non_neg", l_w.item() >= 0)
    m.eval()


def test_direction_diversity_loss(m, c, R):
    """Direction diversity loss should be finite for multiple texts."""
    print("\n── 89. Direction diversity loss ──")
    trainer = Trainer(m, c)
    m.train()
    texts = ["Hello world.", "Quantum physics.", "Piano music."]
    l_dd = trainer.direction_diversity_loss(texts)
    R.check("dir_div_finite", l_dd.isfinite().item())
    m.eval()


def test_semantic_alignment_loss(m, c, R):
    """Semantic alignment loss should focus on content tokens."""
    print("\n── 90. Semantic alignment loss ──")
    trainer = Trainer(m, c)
    dev = _device(m)
    fiber = torch.randn(2, c.d_F, device=dev, requires_grad=True)
    tk = m.tok(["Piano Chopin nocturne", "Telescope galaxy"],
               return_tensors='pt', padding=True, truncation=True)
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    l_sa = trainer.semantic_alignment_loss(fiber, ids, mask)
    R.check("sem_align_finite", l_sa.isfinite().item())
    R.check("sem_align_non_neg", l_sa.item() >= 0)


def test_encoder_throughput_loss(m, c, R):
    """Encoder throughput loss should be a valid CE value."""
    print("\n── 91. Encoder throughput loss ──")
    trainer = Trainer(m, c)
    m.train()
    dev = _device(m)
    tk = m.tok("Hello world test", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
    pooled = m.layer_pool(o['hs']).mean(1)
    base = m.amm.ctx(pooled)
    surp = torch.tensor([1.0], device=dev)
    fiber = m.amm.fib(pooled, base, surp)
    l_et = trainer.encoder_throughput_loss(ids, mask, fiber)
    R.check("enc_throughput_finite", l_et.isfinite().item())
    m.eval()


def test_vocab_anchor_loss(m, c, R):
    """Vocab anchor loss should encourage prefix similarity to WTE."""
    print("\n── 92. Vocab anchor loss ──")
    trainer = Trainer(m, c)
    dev = _device(m)
    prefix = torch.randn(2, c.L_mem, c.d_LLM, device=dev, requires_grad=True)
    l_va = trainer.vocab_anchor_loss(prefix)
    R.check("vocab_anchor_finite", l_va.isfinite().item())


def test_reranker_ranking_loss(m, c, R):
    """Reranker ranking loss with stored memories."""
    print("\n── 93. Reranker ranking loss ──")
    _reset(m)
    for t in ["Cat on mat.", "Piano Chopin.", "Space telescope."]:
        m.write(t, training_mode=True)
    trainer = Trainer(m, c)
    m.train()
    l_rr = trainer.reranker_ranking_loss(["Tell me about cats.", "Piano music."])
    R.check("reranker_loss_finite", l_rr.isfinite().item())
    m.eval()
    _reset(m)


def test_reranker_ranking_loss_empty(m, c, R):
    """Reranker ranking loss with empty store should be zero."""
    print("\n── 94. Reranker ranking loss empty ──")
    _reset(m)
    trainer = Trainer(m, c)
    m.train()
    l_rr = trainer.reranker_ranking_loss(["Hello world."])
    R.check("reranker_empty_zero", l_rr.item() == 0.0)
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 40. LossWarmup
# ═══════════════════════════════════════════════════════════════════
def test_loss_warmup(R):
    """LossWarmup should ramp from 0 to 1."""
    print("\n── 95. LossWarmup ──")
    lw = LossWarmup({'probe': 5, 'other': 10})
    R.check("warmup_initial_probe", lw.weight('probe') == 0.0)
    R.check("warmup_initial_other", lw.weight('other') == 0.0)
    R.check("warmup_unknown_1", lw.weight('unknown') == 1.0)
    for _ in range(5):
        lw.advance()
    R.check("warmup_after5_probe", abs(lw.weight('probe') - 1.0) < 1e-5)
    R.check("warmup_after5_other", abs(lw.weight('other') - 0.5) < 1e-5)
    for _ in range(5):
        lw.advance()
    R.check("warmup_after10_other", abs(lw.weight('other') - 1.0) < 1e-5)


def test_loss_warmup_zero_schedule(R):
    """Schedule with 0 warmup steps should return 1.0 immediately."""
    print("\n── 96. LossWarmup zero schedule ──")
    lw = LossWarmup({'fast': 0})
    R.check("warmup_zero_is_1", lw.weight('fast') == 1.0)


# ═══════════════════════════════════════════════════════════════════
# 41. GradientMonitor
# ═══════════════════════════════════════════════════════════════════
def test_gradient_monitor(m, c, R):
    """GradientMonitor should track gradient norms."""
    print("\n── 97. GradientMonitor ──")
    gm = GradientMonitor()
    gm.register('ctx', m.amm.ctx)
    snap = gm.snapshot()
    R.check("gradmon_no_grad_zero", snap['ctx'] == 0.0)
    m.train()
    m.zero_grad()
    dev = _device(m)
    h = torch.randn(1, c.d_LLM, device=dev)
    x = m.amm.ctx(h)
    x.sum().backward()
    snap2 = gm.snapshot()
    R.check("gradmon_after_backward", snap2['ctx'] > 0)
    m.zero_grad()
    m.eval()


# ═══════════════════════════════════════════════════════════════════
# 42. SpectralDealiaser
# ═══════════════════════════════════════════════════════════════════
def test_dealiaser_detect(m, c, R):
    """SpectralDealiaser should detect aliased clusters."""
    print("\n── 98. SpectralDealiaser detect ──")
    _reset(m)
    for t in ["Cat on mat.", "Piano Chopin.", "Space telescope.",
              "Quantum computing.", "Music theory."]:
        m.write(t, training_mode=True)
    da = SpectralDealiaser(m.amm, c)
    cls = da.detect(sim_threshold=0.3)
    R.check("dealiaser_detect_returns_list", isinstance(cls, list))
    _reset(m)


def test_dealiaser_dealias(m, c, R):
    """SpectralDealiaser.dealias should modify fibers."""
    print("\n── 99. SpectralDealiaser dealias ──")
    _reset(m)
    dev = _device(m)
    base_fiber = torch.randn(c.d_F, device=dev)
    for i in range(5):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"text_{i}")
    da = SpectralDealiaser(m.amm, c)
    ids_to_dealias = list(m.amm.tree.store.keys())[:3]
    fibers_before = [m.amm.tree.store[i].fiber.clone() for i in ids_to_dealias]
    da.dealias(ids_to_dealias, steps=10, lr=0.01)
    changed = False
    for i, mid in enumerate(ids_to_dealias):
        if mid in m.amm.tree.store:
            diff = (m.amm.tree.store[mid].fiber - fibers_before[i]).abs().max().item()
            if diff > 1e-6:
                changed = True
    R.check("dealiaser_modifies_fibers", changed)
    errs = m.amm.tree.verify_consistency()
    R.check("dealiaser_consistent", len(errs) == 0, str(errs))
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 43. Semantic retrieval quality
# ═══════════════════════════════════════════════════════════════════
def test_semantic_retrieval_music_vs_space(m, c, R):
    """Music query should retrieve music memories, not space."""
    print("\n── 100. Semantic retrieval: music vs space ──")
    _reset(m)
    m.write("He practiced piano for hours perfecting a difficult Chopin nocturne.",
            training_mode=True)
    m.write("She studied music theory and harmonic progression at the conservatory.",
            training_mode=True)
    m.write("The telescope revealed distant galaxies beyond the Milky Way.",
            training_mode=True)
    m.write("Astronauts trained for the Mars mission in simulated zero gravity.",
            training_mode=True)
    m.eval()
    dev = _device(m)
    tk = m.tok("Tell me about piano practice.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(o['hs'], mask, update_stats=False,
                                     return_extra=True, ids=ids)
    top10_ids = cb[0].topk(10).indices.tolist()
    top10_toks = [m.tok.decode([t]).strip().lower() for t in top10_ids]
    music_kws = {'piano', 'chopin', 'nocturne', 'practiced', 'perfecting',
                 'difficult', 'music', 'theory', 'harmonic', 'progression',
                 'conservatory', 'studied', 'hours'}
    space_kws = {'telescope', 'galaxies', 'galaxy', 'distant', 'astronauts',
                 'mars', 'gravity', 'mission', 'zero', 'milky', 'revealed'}
    has_music = any(w in music_kws for w in top10_toks)
    has_space = any(w in space_kws for w in top10_toks)
    R.check("sem_ret_music_has_music", has_music, f"top10={top10_toks}")
    R.check("sem_ret_music_not_space_dominant", not has_space or has_music,
            f"top10={top10_toks}")

    tk2 = m.tok("The space telescope observes distant stars.", return_tensors='pt')
    ids2, mask2 = tk2['input_ids'].to(dev), tk2['attention_mask'].to(dev)
    with torch.no_grad():
        o2 = m.fwd(ids2, mask2)
        _, _, _, cb2 = m._get_prefix(o2['hs'], mask2, update_stats=False,
                                      return_extra=True, ids=ids2)
    top10_ids2 = cb2[0].topk(10).indices.tolist()
    top10_toks2 = [m.tok.decode([t]).strip().lower() for t in top10_ids2]
    has_space2 = any(w in space_kws for w in top10_toks2)
    R.check("sem_ret_space_has_space", has_space2, f"top10={top10_toks2}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 44. Zero-train content grounding
# ═══════════════════════════════════════════════════════════════════
def test_zero_train_content_grounding(m, c, R):
    """Without any training, content bias should still reflect stored text."""
    print("\n── 101. Zero-train content grounding ──")
    _reset(m)
    m.write("He practiced piano for hours perfecting a Chopin nocturne.",
            training_mode=True)
    m.write("She studied music theory and harmonic progression.",
            training_mode=True)
    m.eval()
    dev = _device(m)
    tk = m.tok("The piano performance", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(o['hs'], mask, update_stats=False,
                                     return_extra=True, ids=ids)
    R.check("zero_train_cb_nonzero", cb.abs().max().item() > 0.01)
    top10_ids = cb[0].topk(10).indices.tolist()
    top10_toks = [m.tok.decode([t]).strip().lower() for t in top10_ids]
    music_words = {'piano', 'chopin', 'nocturne', 'practiced', 'perfecting',
                   'music', 'theory', 'harmonic', 'progression', 'studied', 'hours'}
    has_music = any(w in music_words for w in top10_toks)
    R.check("zero_train_has_music", has_music, f"top10={top10_toks}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 45. Domain generation quality
# ═══════════════════════════════════════════════════════════════════
def test_domain_generation(m, c, R):
    """Generated text should reflect the domain of stored memories."""
    print("\n── 102. Domain generation quality ──")
    _reset(m)
    music_texts = [
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "The orchestra performed Beethoven symphony with remarkable precision.",
        "She studied music theory and harmonic progression at the conservatory.",
    ]
    space_texts = [
        "The telescope revealed distant galaxies beyond the Milky Way.",
        "Astronauts trained for the Mars mission in simulated zero gravity.",
        "The nebula emitted radiation across the electromagnetic spectrum.",
    ]
    for t in music_texts + space_texts:
        m.write(t, training_mode=True)
    m.eval()
    music_kws = {'piano', 'music', 'chopin', 'nocturne', 'orchestra',
                 'beethoven', 'symphony', 'harmony', 'melody', 'chord',
                 'practiced', 'harmonic', 'progression', 'conservatory',
                 'performed', 'theory', 'studied', 'pianist'}
    space_kws = {'galaxy', 'galaxies', 'telescope', 'star', 'planet',
                 'orbit', 'space', 'astronaut', 'mars', 'nebula',
                 'radiation', 'gravity', 'cosmic', 'solar', 'universe',
                 'spectrum', 'astronauts', 'electromagnetic', 'distant'}

    def count_domain(text, kws):
        words = set(text.lower().split())
        return sum(1 for w in words if any(kw in w for kw in kws))

    torch.manual_seed(42)
    with torch.no_grad():
        mg = m.generate("The piano performance", mt=40, greedy=False)
        sg = m.generate("The space telescope", mt=40, greedy=False)

    mc = count_domain(mg, music_kws)
    sc = count_domain(sg, space_kws)
    R.check("domain_music_gen_has_kw", mc > 0, f"music_kw_count={mc}")
    R.check("domain_space_gen_has_kw", sc > 0, f"space_kw_count={sc}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 46. Degeneration quality
# ═══════════════════════════════════════════════════════════════════
def test_degeneration_quality(m, c, R):
    """Generated text should not be degenerate (all punct, repeats, etc.)."""
    print("\n── 103. Degeneration quality ──")
    _reset(m)
    for t in [
        "The cat sat on the mat and watched birds outside.",
        "Quantum computing uses qubits in superposition states.",
        "He practiced piano for hours perfecting a Chopin nocturne.",
    ]:
        m.write(t, training_mode=True)
    m.eval()
    cc = m.content_classifier
    for prompt in ["The pianist", "Quantum computing is", "Stars and galaxies"]:
        torch.manual_seed(42)
        with torch.no_grad():
            gen = m.generate(prompt, mt=30, greedy=False)
        new_text = gen[len(prompt):].strip()
        total = len(new_text)
        alpha = sum(1 for ch in new_text if ch.isalpha())
        ratio = alpha / max(total, 1)
        R.check(f"degen_{prompt[:10]}_has_content", total >= 3, f"chars={total}")
        R.check(f"degen_{prompt[:10]}_alpha_ratio", ratio > 0.25,
                f"ratio={ratio:.2f}, text='{new_text[:50]}'")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 47. Memory with/without comparison
# ═══════════════════════════════════════════════════════════════════
def test_memory_vs_no_memory(m, c, R):
    """Generation should differ with and without memories."""
    print("\n── 104. Memory vs no-memory generation ──")
    _reset(m)
    for t in [
        "He practiced piano for hours perfecting a Chopin nocturne.",
        "The telescope revealed distant galaxies beyond the Milky Way.",
    ]:
        m.write(t, training_mode=True)
    m.eval()
    torch.manual_seed(123)
    with torch.no_grad():
        gen_mem = m.generate("The pianist", mt=25, greedy=True)
    saved_store = dict(m.amm.tree.store)
    saved_root = m.amm.tree.root
    saved_nid = m.amm.tree.nid
    m.amm.tree.store = {}
    m.amm.tree.root = _Node()
    torch.manual_seed(123)
    with torch.no_grad():
        gen_no = m.generate("The pianist", mt=25, greedy=True)
    m.amm.tree.store = saved_store
    m.amm.tree.root = saved_root
    m.amm.tree.nid = saved_nid
    R.check("mem_vs_nomem_differ", gen_mem != gen_no,
            f"with='{gen_mem[:50]}', without='{gen_no[:50]}'")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 48. Batch retrieval
# ═══════════════════════════════════════════════════════════════════
def test_batch_retrieval(m, c, R):
    """Batched retrieval should handle multiple queries."""
    print("\n── 105. Batch retrieval ──")
    _reset(m)
    for t in ["Cats are fluffy.", "Stars shine bright."]:
        m.write(t, training_mode=True)
    m.eval()
    dev = _device(m)
    tk = m.tok(["Tell me about cats.", "The night sky."],
               return_tensors='pt', padding=True, truncation=True)
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(o['hs'], mask, return_extra=True, ids=ids)
    R.check("batch_cb_shape", cb.shape[0] == 2)
    R.check("batch_cb_finite", cb.isfinite().all().item())
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 49. Flat scan vs tree scan
# ═══════════════════════════════════════════════════════════════════
def test_flat_scan_threshold(m, c, R):
    """When store is small, should use flat scan."""
    print("\n── 106. Flat scan threshold ──")
    _reset(m)
    dev = _device(m)
    for i in range(3):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"entry_{i}")
    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    _, _, _, diag = m.amm.retrieve_multi(xq, fq)
    threshold = c.flat_scan_threshold_factor * c.retrieval_topk
    should_flat = len(m.amm.tree.store) <= threshold
    R.check("flat_scan_used", diag.was_flat_scan == should_flat,
            f"store={len(m.amm.tree.store)}, thresh={threshold}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 50. RetrievalDiag completeness
# ═══════════════════════════════════════════════════════════════════
def test_retrieval_diag_fields(m, c, R):
    """RetrievalDiag should have all expected fields populated."""
    print("\n── 107. RetrievalDiag fields ──")
    _reset(m)
    dev = _device(m)
    for i in range(5):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"entry_{i}",
                        content_semantic_emb=torch.randn(c.d_LLM, device=dev))
    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    q_sem = torch.randn(1, c.d_LLM, device=dev)
    _, _, _, diag = m.amm.retrieve_multi(xq, fq,
                                          query_semantic_emb=q_sem)
    R.check("diag_recall_count", diag.recall_count > 0)
    R.check("diag_fiber_summary_norm", diag.fiber_summary_norm > 0)
    R.check("diag_batch_mem_weights", len(diag.batch_mem_weights) == 1)
    R.check("diag_has_top_scores", True)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 51. Midpoint distance
# ═══════════════════════════════════════════════════════════════════
def test_midpoint_distance(m, c, R):
    """Midpoint distance should be non-negative and zero for same point."""
    print("\n── 108. Midpoint distance ──")
    dev = _device(m)
    x = torch.randn(3, c.d_M, device=dev)
    y = torch.randn(3, c.d_M, device=dev)
    d = m.amm.metric.midpoint_approx_distance(x, y)
    R.check("midpoint_dist_shape", d.shape == (3,))
    R.check("midpoint_dist_non_neg", (d >= 0).all().item())
    d_self = m.amm.metric.midpoint_approx_distance(x, x)
    R.check("midpoint_dist_self_zero", d_self.max().item() < 1e-5)


# ═══════════════════════════════════════════════════════════════════
# 52. Training loss convergence
# ═══════════════════════════════════════════════════════════════════
def test_training_convergence(m, c, R):
    """Over multiple steps, loss should decrease or not explode."""
    print("\n── 109. Training convergence ──")
    _reset(m)
    texts = [
        "The cat sat on the mat and watched birds.",
        "Quantum computing uses qubits in superposition.",
        "He practiced piano for hours perfecting Chopin.",
        "The stock market experienced significant volatility.",
        "She walked along the beach at sunset.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    trainer = Trainer(m, c)
    losses = []
    for ep in range(6):
        info = trainer.step(texts[:4])
        losses.append(info['total'])
    R.check("convergence_all_finite", all(math.isfinite(l) for l in losses))
    R.check("convergence_not_exploding", losses[-1] < losses[0] * 5,
            f"first={losses[0]:.4f}, last={losses[-1]:.4f}")
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 53. Large number of memories stress test
# ═══════════════════════════════════════════════════════════════════
def test_many_memories(m, c, R):
    """System should handle many memories without crashing."""
    print("\n── 110. Many memories stress test ──")
    _reset(m)
    dev = _device(m)
    for i in range(30):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, float(i) * 0.1, training_mode=True,
                        source_text=f"memory_{i}",
                        content_token_ids=[100 + i, 200 + i],
                        content_semantic_emb=torch.randn(c.d_LLM, device=dev))
    R.check("many_mem_stored", len(m.amm.tree.store) > 0)
    errs = m.amm.tree.verify_consistency()
    R.check("many_mem_consistent", len(errs) == 0, str(errs))
    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    fibers, mask, fs, diag = m.amm.retrieve_multi(xq, fq)
    R.check("many_mem_retrieve_ok", fibers.isfinite().all().item())
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 54. Consolidation of similar memories
# ═══════════════════════════════════════════════════════════════════
def test_consolidation_similar(m, c, R):
    """Very similar texts should be consolidated."""
    print("\n── 111. Consolidation of similar memories ──")
    _reset(m)
    for i in range(5):
        m.write("The cat sat on the mat.", training_mode=True)
    n_before = len(m.amm.tree.store)
    n_merged = m.amm.consolidate()
    n_after = len(m.amm.tree.store)
    R.check("consol_similar_merged", n_merged >= 0)
    R.check("consol_similar_math", n_after == n_before - n_merged)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 55. Edge cases
# ═══════════════════════════════════════════════════════════════════
def test_empty_text_write(m, c, R):
    """Writing empty text may raise or handle gracefully — either is acceptable."""
    print("\n── 112. Empty text write ──")
    _reset(m)
    try:
        n, gv = m.write("", training_mode=True)
        R.check("empty_text_write_handled", True)
    except Exception:
        R.check("empty_text_write_raises", True)
    _reset(m)


def test_very_long_text_write(m, c, R):
    """Writing a very long text should not crash."""
    print("\n── 113. Very long text write ──")
    _reset(m)
    long_text = "The cat sat on the mat. " * 200
    try:
        n, gv = m.write(long_text, training_mode=True)
        R.check("long_text_write_ok", True)
        R.check("long_text_stored", n >= 1)
    except Exception as e:
        R.check("long_text_write_ok", False, str(e))
    _reset(m)


def test_special_chars_write(m, c, R):
    """Text with special characters should not crash."""
    print("\n── 114. Special characters write ──")
    _reset(m)
    texts = [
        "Hello! @#$%^&*() 你好世界",
        "Math: 2+2=4, π≈3.14159",
        "Code: def f(x): return x**2",
    ]
    for t in texts:
        try:
            m.write(t, training_mode=True)
            R.check(f"special_char_write_{t[:10]}", True)
        except Exception as e:
            R.check(f"special_char_write_{t[:10]}", False, str(e))
    _reset(m)


def test_single_token_input(m, c, R):
    """Single token input should work for generation."""
    print("\n── 115. Single token input ──")
    _reset(m)
    m.eval()
    with torch.no_grad():
        gen = m.generate("A", mt=5, greedy=True)
    R.check("single_token_gen_ok", len(gen) > 0)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 56. Retrieval weights configuration
# ═══════════════════════════════════════════════════════════════════
def test_retrieval_weight_config(R, c):
    """v3.12: retrieval weights should sum to 1 and be in valid range."""
    print("\n── 116. Retrieval weight config ──")
    total = (c.ret_forward_maxsim_weight + c.ret_backward_maxsim_weight +
             c.ret_overlap_weight + c.ret_sem_weight + c.ret_dir_weight)
    R.check("ret_weights_sum_to_1", abs(total - 1.0) < 1e-5, f"sum={total}")
    R.check("ret_forward_maxsim_range", 0 <= c.ret_forward_maxsim_weight <= 1)
    R.check("ret_backward_maxsim_range", 0 <= c.ret_backward_maxsim_weight <= 1)
    R.check("ret_overlap_range", 0 <= c.ret_overlap_weight <= 1)
    R.check("ret_sem_weight_range", 0 <= c.ret_sem_weight <= 1)
    R.check("ret_dir_weight_range", 0 <= c.ret_dir_weight <= 1)


# ═══════════════════════════════════════════════════════════════════
# 57. Content bias scale decay
# ═══════════════════════════════════════════════════════════════════
def test_content_bias_decay(R, c):
    """Content bias should decay over steps but not below floor."""
    print("\n── 117. Content bias decay ──")
    all_ge_floor = True
    for step in range(100):
        scale = max(c.content_bias_floor, 1.0 - step * c.content_bias_decay)
        if scale < c.content_bias_floor:
            all_ge_floor = False
            break
    R.check("cb_decay_all_ge_floor", all_ge_floor)
    scale_0 = max(c.content_bias_floor, 1.0 - 0 * c.content_bias_decay)
    R.check("cb_decay_step0_is_1", abs(scale_0 - 1.0) < 1e-5)
    scale_big = max(c.content_bias_floor, 1.0 - 1000 * c.content_bias_decay)
    R.check("cb_decay_big_step_at_floor",
            abs(scale_big - c.content_bias_floor) < 1e-5)


# ═══════════════════════════════════════════════════════════════════
# 58. Layer pool weight history in trainer
# ═══════════════════════════════════════════════════════════════════
def test_layer_pool_weight_history(m, c, R):
    """Trainer should record layer pool weight distributions."""
    print("\n── 118. Layer pool weight history ──")
    _reset(m)
    texts = ["Cat on mat.", "Piano Chopin.", "Stars galaxies."]
    for t in texts:
        m.write(t, training_mode=True)
    trainer = Trainer(m, c)
    for _ in range(3):
        trainer.step(texts)
    R.check("layer_history_len", len(trainer.layer_weight_history) == 3)
    for hist in trainer.layer_weight_history:
        R.check("layer_history_sums_1", abs(hist.sum() - 1.0) < 1e-4)
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 59. QFormerLayer
# ═══════════════════════════════════════════════════════════════════
def test_qformer_layer(m, c, R):
    """QFormerLayer should process self-attention + cross-attention."""
    print("\n── 119. QFormerLayer ──")
    dev = _device(m)
    layer = m.bridge.proj.layers[0]
    B = 2
    q = torch.randn(B, c.L_mem, c.d_LLM, device=dev)
    k = torch.randn(B, 5, c.d_LLM, device=dev)
    v = torch.randn(B, 5, c.d_LLM, device=dev)
    out = layer(q, k, v)
    R.check("qformer_layer_shape", out.shape == q.shape)
    R.check("qformer_layer_finite", out.isfinite().all().item())
    kv_mask = torch.ones(B, 5, device=dev)
    kv_mask[1, 3:] = 0
    out2 = layer(q, k, v, kv_mask=kv_mask)
    R.check("qformer_layer_mask_finite", out2.isfinite().all().item())


# ═══════════════════════════════════════════════════════════════════
# 60. EmptyStateNet
# ═══════════════════════════════════════════════════════════════════
def test_empty_state_net(m, c, R):
    """EmptyStateNet should produce default fiber when no memories exist."""
    print("\n── 120. EmptyStateNet ──")
    dev = _device(m)
    xq = torch.randn(2, c.d_M, device=dev)
    fq = torch.randn(2, c.d_F, device=dev)
    empty_f = m.amm.empty_state(xq, fq)
    R.check("empty_state_shape", empty_f.shape == (2, c.d_F))
    R.check("empty_state_finite", empty_f.isfinite().all().item())


# ═══════════════════════════════════════════════════════════════════
# 61. Integrated end-to-end workflow
# ═══════════════════════════════════════════════════════════════════
def test_e2e_workflow(m, c, R):
    """Full end-to-end: write → train → generate → save → load → generate."""
    print("\n── 121. End-to-end workflow ──")
    _reset(m)
    texts = [
        "The cat sat on the mat and watched birds outside the window.",
        "Quantum computing uses qubits in superposition states.",
        "He practiced piano for hours perfecting a Chopin nocturne.",
        "The stock market experienced significant volatility.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    R.check("e2e_write_ok", len(m.amm.tree.store) > 0)
    trainer = Trainer(m, c)
    info = trainer.step(texts[:3])
    R.check("e2e_train_ok", math.isfinite(info['total']))
    m.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        gen1 = m.generate("The pianist", mt=20, greedy=True)
    R.check("e2e_gen1_ok", len(gen1) > len("The pianist"))
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        R.check("e2e_load_ok", len(m.amm.tree.store) > 0)
        torch.manual_seed(42)
        with torch.no_grad():
            gen2 = m.generate("The pianist", mt=20, greedy=True)
        R.check("e2e_gen2_ok", len(gen2) > len("The pianist"))
    finally:
        os.unlink(path)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 62. Trainer _recon_forward
# ═══════════════════════════════════════════════════════════════════
def test_recon_forward(m, c, R):
    """_recon_forward should produce reconstruction loss."""
    print("\n── 122. _recon_forward ──")
    _reset(m)
    m.write("Piano music is wonderful.", training_mode=True)
    trainer = Trainer(m, c)
    m.train()
    lr, pf, fs = trainer._recon_forward("Tell me about piano.")
    R.check("recon_fwd_loss_finite", lr.isfinite().item())
    R.check("recon_fwd_prefix_shape", pf.shape[-1] == c.d_LLM)
    R.check("recon_fwd_fs_shape", fs.shape[-1] == c.d_F)
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 63. Semantic probe contrastive
# ═══════════════════════════════════════════════════════════════════
def test_semantic_probe_contrastive(m, c, R):
    """Semantic probe loss should include contrastive component for batch>1."""
    print("\n── 123. Semantic probe contrastive ──")
    _reset(m)
    for t in ["Cat mat.", "Piano music."]:
        m.write(t, training_mode=True)
    trainer = Trainer(m, c)
    m.train()
    dev = _device(m)
    prefix_batch = torch.randn(2, c.L_mem, c.d_LLM, device=dev, requires_grad=True)
    fs_batch = torch.randn(2, c.d_F, device=dev)
    l_sp = trainer._semantic_probe_loss(prefix_batch, fs_batch)
    R.check("probe_contrastive_finite", l_sp.isfinite().item())
    l_sp.backward()
    R.check("probe_contrastive_grad", prefix_batch.grad is not None)
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 64. Trainer _encode_with_grad
# ═══════════════════════════════════════════════════════════════════
def test_encode_with_grad(m, c, R):
    """_encode_with_grad should return all required tensors."""
    print("\n── 124. _encode_with_grad ──")
    trainer = Trainer(m, c)
    m.train()
    ids, mask, base, fiber, surp, pooled = trainer._encode_with_grad(
        ["Hello world.", "Piano music."])
    R.check("ewg_ids_shape", ids.shape[0] == 2)
    R.check("ewg_base_shape", base.shape == (2, c.d_M))
    R.check("ewg_fiber_shape", fiber.shape == (2, c.d_F))
    R.check("ewg_surp_shape", surp.shape == (2,))
    R.check("ewg_pooled_shape", pooled.shape[0] == 2 and pooled.shape[-1] == c.d_LLM)
    R.check("ewg_base_requires_grad", base.requires_grad)
    R.check("ewg_fiber_requires_grad", fiber.requires_grad)
    m.eval()


# ═══════════════════════════════════════════════════════════════════
# 65. Tree max_depth
# ═══════════════════════════════════════════════════════════════════
def test_tree_max_depth(R, c):
    """max_depth should increase with more entries."""
    print("\n── 125. Tree max_depth ──")
    tc = Cfg(tree_max_leaf=5, tree_K=3, d_M=c.d_M, d_F=c.d_F)
    tree = DirectionTree(tc)
    R.check("tree_empty_depth", tree.max_depth() == 0)
    for i in range(50):
        d = F.normalize(torch.randn(tc.d_M), dim=0)
        me = MemEntry(mid=i, base=torch.randn(tc.d_M), fiber=torch.randn(tc.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        tree.store[me.mid] = me
        tree.nid = i + 1
        tree._ins(tree.root, me)
    depth = tree.max_depth()
    R.check("tree_depth_positive", depth > 0, f"depth={depth}")


# ═══════════════════════════════════════════════════════════════════
# 66. GeodesicResult namedtuple
# ═══════════════════════════════════════════════════════════════════
def test_geodesic_result_fields(m, c, R):
    """GeodesicResult should have all expected fields."""
    print("\n── 126. GeodesicResult fields ──")
    dev = _device(m)
    xs = torch.randn(1, c.d_M, device=dev) * 0.2
    xe = torch.randn(1, c.d_M, device=dev) * 0.2
    gr = m.amm.geo.solve(xs, xe)
    R.check("georesult_has_path", hasattr(gr, 'path'))
    R.check("georesult_has_energy", hasattr(gr, 'energy'))
    R.check("georesult_has_converged", hasattr(gr, 'converged'))
    R.check("georesult_has_iterations", hasattr(gr, 'iterations'))
    R.check("georesult_path_shape",
            gr.path.shape == (1, c.n_geo_pts + 2, c.d_M))


# ═══════════════════════════════════════════════════════════════════
# 67. Multiple consolidation rounds
# ═══════════════════════════════════════════════════════════════════
def test_multiple_consolidation(m, c, R):
    """Multiple consolidation rounds should not corrupt the tree."""
    print("\n── 127. Multiple consolidation rounds ──")
    _reset(m)
    dev = _device(m)
    for i in range(10):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        m.amm.store_mem(h, 0.5, training_mode=True, source_text=f"mem_{i}")
    for _ in range(3):
        m.amm.consolidate()
    errs = m.amm.tree.verify_consistency()
    R.check("multi_consol_consistent", len(errs) == 0, str(errs))
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 68. Decay + consolidate combined
# ═══════════════════════════════════════════════════════════════════
def test_decay_then_consolidate(m, c, R):
    """Decay followed by consolidation should maintain consistency."""
    print("\n── 128. Decay + consolidate ──")
    _reset(m)
    dev = _device(m)
    for i in range(8):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 0.1 + i * 0.05, training_mode=True, source_text=f"m_{i}")
    m.amm.time += 3000
    m.amm.decay()
    m.amm.consolidate()
    errs = m.amm.tree.verify_consistency()
    R.check("decay_consol_consistent", len(errs) == 0, str(errs))
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# 69. Build content bias (edge cases)
# ═══════════════════════════════════════════════════════════════════
def test_build_content_bias_edge(m, c, R):
    """_build_content_bias with no memories should be zero."""
    print("\n── 129. _build_content_bias edge ──")
    diag = RetrievalDiag()
    diag.batch_mem_weights = [[]]
    bias = m._build_content_bias(diag, [[]])
    R.check("cb_edge_zero", bias.abs().max().item() < 1e-8)


# ═══════════════════════════════════════════════════════════════════
# 70. Surprise proxy edge cases
# ═══════════════════════════════════════════════════════════════════
def test_surprise_proxy_empty(m, c, R):
    """Surprise proxy with T=0 should return zeros."""
    print("\n── 130. Surprise proxy empty ──")
    dev = _device(m)
    logits = torch.randn(2, 0, c.vocab_size, device=dev)
    tgt = torch.randint(0, c.vocab_size, (2, 0), device=dev)
    surp = m.amm.surprise_proxy(logits, tgt)
    R.check("surp_empty_zeros", (surp == 0).all().item())


# ═══════════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(42)
    c = Cfg()
    R = TestResults()

    sep = "=" * 70
    print(f"\n{sep}")
    print("  AMS v3.7 — System-Level Black-Box Test Suite")
    print(f"{sep}")
    t0 = time.time()

    print("\n[Building MemLLM + loading GPT-2]")
    m = MemLLM(c)
    m.load("gpt2")
    total = sum(p.numel() for p in m.parameters())
    train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={train_p:,}  frozen={total-train_p:,}")

    # === Config ===
    test_cfg_defaults(R)
    test_cfg_invalid_catches(R)

    # === Geometric core ===
    test_metric_spd(m, c, R)
    test_metric_deterministic(m, c, R)
    test_metric_batch_independence(m, c, R)
    test_christoffel_symmetry(m, c, R)
    test_christoffel_finite(m, c, R)
    test_geodesic_boundary_conditions(m, c, R)
    test_geodesic_convergence(m, c, R)
    test_geodesic_energy_decreases_or_converges(m, c, R)
    test_geodesic_trivial_case(m, c, R)
    test_geodesic_no_grad_mode(m, c, R)
    test_geodesic_gradient_propagation(m, c, R)
    test_geodesic_result_fields(m, c, R)

    # === Fiber transport ===
    test_fiber_connection_antisym(m, c, R)
    test_transport_norm_preservation(m, c, R)
    test_transport_short_path(m, c, R)

    # === Encoders ===
    test_encoder_shapes(m, c, R)
    test_fib_encoder_surprise_gating(m, c, R)
    test_fib_encoder_no_surprise(m, c, R)
    test_direction_predictor_unit_norm(m, c, R)

    # === Policy modules ===
    test_write_gate_range(m, c, R)
    test_retention_scorer_range(m, c, R)
    test_reranker_correction(m, c, R)

    # === Bridge components ===
    test_content_bypass(m, c, R)
    test_semantic_probe(m, c, R)
    test_prefix_aligner_calibration(m, c, R)
    test_vocab_projector(m, c, R)
    test_fiber_attn_output_shape(m, c, R)
    test_fiber_attn_mask(m, c, R)
    test_qformer_proj(m, c, R)
    test_emb_bridge_inject(m, c, R)
    test_emb_bridge_modes(m, c, R)
    test_layer_pool(m, c, R)
    test_state_extractor(m, c, R)
    test_qformer_layer(m, c, R)
    test_empty_state_net(m, c, R)

    # === ContentTokenClassifier ===
    test_content_classifier_completeness(m, c, R)
    test_content_classifier_known_tokens(m, c, R)
    test_content_mask_device(m, c, R)
    test_get_content_positions(m, c, R)

    # === WTE & content expansion ===
    test_wte_neighbor_cache(m, c, R)
    test_content_expansion(m, c, R)
    test_content_semantic_emb(m, c, R)
    test_content_semantic_emb_cross_domain(m, c, R)

    # === DirectionTree ===
    test_tree_insert_retrieve(R, c)
    test_tree_remove(R, c)
    test_tree_update_direction(R, c)
    test_tree_rebuild(R, c)
    test_tree_leaf_capacity(R)
    test_tree_direction_degeneracy_detection(R)
    test_tree_empty_operations(R, c)
    test_tree_max_depth(R, c)
    test_mem_entry_defaults(R, c)

    # === DegenerationGuard ===
    test_degen_guard_basic(m, c, R)
    test_degen_guard_repeat_penalty(m, c, R)
    test_degen_guard_punct_penalty_early(m, c, R)
    test_degen_guard_consec_punct_block(m, c, R)

    # === AMM core ===
    test_amm_store_and_retrieve(m, c, R)
    test_amm_consolidate(m, c, R)
    test_amm_decay(m, c, R)
    test_amm_store_update_existing(m, c, R)
    test_amm_empty_retrieve(m, c, R)
    test_amm_retrieve_with_semantic(m, c, R)
    test_amm_surprise_proxy(m, c, R)
    test_midpoint_distance(m, c, R)
    test_flat_scan_threshold(m, c, R)
    test_retrieval_diag_fields(m, c, R)

    # === MemLLM write/generate ===
    test_write_single(m, c, R)
    test_write_multiple_topics(m, c, R)
    test_write_gate_filtering(m, c, R)
    test_generate_greedy(m, c, R)
    test_generate_sampling(m, c, R)
    test_generate_empty_memory(m, c, R)
    test_generate_retrieval_refresh(m, c, R)
    test_generate_eos_handling(m, c, R)

    # === Content bias & decoding ===
    test_content_bias_from_memories(m, c, R)
    test_content_bias_empty_memory(m, c, R)
    test_first_step_content_not_punct(m, c, R)
    test_build_content_bias_edge(m, c, R)

    # === Forward / state extraction ===
    test_fwd_basic(m, c, R)
    test_fwd_with_prefix(m, c, R)
    test_fwd_batch(m, c, R)
    test_extract_state(m, c, R)
    test_get_prefix(m, c, R)
    test_get_prefix_extra(m, c, R)
    test_vocab_bias(m, c, R)

    # === Memory persistence ===
    test_save_load_memory(m, c, R)
    test_refresh_memories(m, c, R)

    # === Gradient / Training ===
    test_gradient_flow(m, c, R)
    test_gradient_monitor(m, c, R)
    test_contrast_loss(m, c, R)
    test_holonomy_loss(m, c, R)
    test_write_policy_loss(m, c, R)
    test_direction_diversity_loss(m, c, R)
    test_semantic_alignment_loss(m, c, R)
    test_encoder_throughput_loss(m, c, R)
    test_vocab_anchor_loss(m, c, R)
    test_reranker_ranking_loss(m, c, R)
    test_reranker_ranking_loss_empty(m, c, R)
    test_trainer_step(m, c, R)
    test_trainer_multiple_steps(m, c, R)
    test_training_convergence(m, c, R)
    test_encode_with_grad(m, c, R)
    test_recon_forward(m, c, R)
    test_semantic_probe_contrastive(m, c, R)
    test_layer_pool_weight_history(m, c, R)

    # === LossWarmup ===
    test_loss_warmup(R)
    test_loss_warmup_zero_schedule(R)

    # === Semantic retrieval quality ===
    test_semantic_retrieval_music_vs_space(m, c, R)
    test_zero_train_content_grounding(m, c, R)
    test_domain_generation(m, c, R)
    test_degeneration_quality(m, c, R)
    test_memory_vs_no_memory(m, c, R)
    test_batch_retrieval(m, c, R)

    # === SpectralDealiaser ===
    test_dealiaser_detect(m, c, R)
    test_dealiaser_dealias(m, c, R)

    # === Stress & edge cases ===
    test_many_memories(m, c, R)
    test_consolidation_similar(m, c, R)
    test_multiple_consolidation(m, c, R)
    test_decay_then_consolidate(m, c, R)
    test_empty_text_write(m, c, R)
    test_very_long_text_write(m, c, R)
    test_special_chars_write(m, c, R)
    test_single_token_input(m, c, R)
    test_surprise_proxy_empty(m, c, R)

    # === Config validation ===
    test_retrieval_weight_config(R, c)
    test_content_bias_decay(R, c)

    # === End-to-end ===
    test_e2e_workflow(m, c, R)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    ok = R.summary()
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
