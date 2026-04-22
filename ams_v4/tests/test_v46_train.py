"""v4.6 tests — loss shapes, grad flow, and a 3-step CPU smoke train."""
from __future__ import annotations
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from ams_v4 import Cfg4, MemLLM4
from ams_v4.training.batch_encode import encode_batch_for_training
from ams_v4.training.losses import (
    loss_bundle_axis_alignment,
    loss_cross_bundle_independence,
    loss_prefix_semantic_anchor,
    loss_recon,
    loss_write_policy,
)
from ams_v4.training.trainer import Trainer4


TEXTS = [
    "The cat sat on the mat.",
    "Pianos have 88 keys.",
    "Paris is the capital of France.",
    "Python is a programming language.",
    "The Pacific is the largest ocean.",
    "Chess has 16 pieces per side.",
]


def _tiny_cfg() -> Cfg4:
    return Cfg4(
        llm_name="distilgpt2",
        llm_dtype="fp32",
        d_LLM=768,
        vocab_size=50257,
        d_time=8, d_F_time=16, n_heads_time=2,
        d_topic=16, d_F_topic=32, n_heads_topic=4,
        d_ctx=8, d_F_ctx=16, n_heads_ctx=2,
        L_mem=6, prefix_slots_time=2, prefix_slots_topic=2, prefix_slots_ctx=2,
        n_kakeya_sets=4, kakeya_min_entries=4, kakeya_K=4, kakeya_d_res=5,
        n_geo_pts=4, geo_max_steps=20,
    )


def _fresh_model():
    cfg = _tiny_cfg()
    m = MemLLM4(cfg)
    m.load()
    # Populate the store so kakeya registry activates
    for t in TEXTS:
        m.write(t)
    return m


# ─── Batch encode ────────────────────────────────────────────────────────

def test_encode_batch_for_training_shapes():
    torch.manual_seed(0)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:3])
    assert be.hidden.shape == (3, m.cfg.d_LLM)
    assert be.time_base.shape == (3, m.cfg.d_time)
    assert be.topic_base.shape == (3, m.cfg.d_topic)
    assert be.ctx_base.shape == (3, m.cfg.d_ctx)
    assert be.time_fiber.shape == (3, m.cfg.d_F_time)
    assert be.topic_fiber.shape == (3, m.cfg.d_F_topic)
    assert be.ctx_fiber.shape == (3, m.cfg.d_F_ctx)
    # Gradients retained
    assert be.time_fiber.requires_grad
    assert be.topic_fiber.requires_grad
    assert be.ctx_fiber.requires_grad


# ─── Individual losses ──────────────────────────────────────────────────

def test_loss_prefix_semantic_anchor_scalar_and_finite():
    torch.manual_seed(1)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:3])
    lv = loss_prefix_semantic_anchor(m, be)
    assert lv.dim() == 0
    assert torch.isfinite(lv)
    assert lv.item() > 0


def test_loss_bundle_axis_alignment_nonneg():
    torch.manual_seed(2)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:4])
    lv = loss_bundle_axis_alignment(m, be)
    assert lv.dim() == 0
    assert torch.isfinite(lv)
    assert lv.item() >= 0


def test_loss_cross_bundle_independence_nonneg():
    torch.manual_seed(3)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:4])
    lv = loss_cross_bundle_independence(m, be)
    assert lv.dim() == 0
    assert torch.isfinite(lv)
    assert lv.item() >= 0


def test_loss_recon_finite():
    torch.manual_seed(4)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:3])
    lv = loss_recon(m, be)
    assert lv.dim() == 0
    assert torch.isfinite(lv)


def test_loss_write_policy_finite():
    torch.manual_seed(5)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:3])
    lv = loss_write_policy(m, be)
    assert lv.dim() == 0
    assert torch.isfinite(lv)
    assert lv.item() >= 0


# ─── Gradient flow ──────────────────────────────────────────────────────

def test_loss_prefix_anchor_gradient_flow_cross_attn():
    """prefix_semantic_anchor gradient must reach cross_attn lift_time[0].weight."""
    torch.manual_seed(6)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:3])
    lv = loss_prefix_semantic_anchor(m, be)
    # zero pre-existing grads
    for p in m.parameters():
        if p.grad is not None:
            p.grad.zero_()
    lv.backward()
    g = m.cross_attn.lift_time[0].weight.grad
    assert g is not None, "no grad on cross_attn.lift_time[0].weight"
    assert g.abs().sum().item() > 0, "grad is zero — computation graph didn't reach lift_time"


def test_loss_bundle_axis_alignment_gradient_flow():
    """bundle_axis_alignment loss drives bundle_time._axis_raw."""
    torch.manual_seed(7)
    m = _fresh_model()
    be = encode_batch_for_training(m, TEXTS[:4])
    lv = loss_bundle_axis_alignment(m, be)
    for p in m.parameters():
        if p.grad is not None:
            p.grad.zero_()
    lv.backward()
    g = m.bundle_time._axis_raw.grad
    assert g is not None, "no grad on bundle_time._axis_raw"
    assert g.abs().sum().item() > 0, "grad is zero on time axis"


# ─── Trainer ────────────────────────────────────────────────────────────

def test_trainer_three_step_cpu_smoke():
    """Three trainer steps must run without raising, and total loss must change."""
    torch.manual_seed(8)
    m = _fresh_model()
    trainer = Trainer4(m, m.cfg, lr=1e-3)

    losses = []
    for i in range(3):
        batch = TEXTS[(i * 2) % 6: (i * 2 + 3) % 6 + 1]
        if len(batch) < 2:  # wrap-around safety
            batch = TEXTS[:3]
        st = trainer.step(batch)
        losses.append(st.total)
        assert st.dt_s > 0
        for k in ("prefix_semantic_anchor", "bundle_axis_alignment",
                  "cross_bundle_independence", "recon", "write_policy"):
            assert k in st.components

    # Expect *some* variation across steps (not strictly monotone on 3 steps,
    # but two identical totals would be suspicious).
    assert len(set(f"{l:.6f}" for l in losses)) >= 2, f"losses didn't vary: {losses}"


def test_trainer_save_and_reload_roundtrip():
    """Train for 1 step, save, new model, load_trained_weights, param matches."""
    import tempfile
    torch.manual_seed(9)
    m1 = _fresh_model()
    trainer = Trainer4(m1, m1.cfg, lr=1e-3)
    trainer.step(TEXTS[:3])

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "v4_test.pt")
        trainer.save(ckpt_path, steps=1, elapsed_s=1.0,
                     pre_probe=trainer.probe_weights(),
                     post_probe=trainer.probe_weights())
        assert os.path.exists(ckpt_path)

        # Capture one weight from m1 for comparison
        ref = m1.cross_attn.lift_time[0].weight.detach().cpu().clone()

        # Fresh model, load ckpt
        m2 = MemLLM4(_tiny_cfg())
        m2.load()
        stats = m2.load_trained_weights(ckpt_path)
        assert stats["loaded"] > 0, f"nothing loaded: {stats}"
        assert stats["shape_errs"] == 0, f"shape errors: {stats}"
        now = m2.cross_attn.lift_time[0].weight.detach().cpu()
        # After loading, the weight should equal m1's
        diff = (now - ref).abs().max().item()
        assert diff < 1e-5, f"reloaded weight differs by {diff}"


# ─── Runner ──────────────────────────────────────────────────────────────

def _run_all():
    tests = [
        test_encode_batch_for_training_shapes,
        test_loss_prefix_semantic_anchor_scalar_and_finite,
        test_loss_bundle_axis_alignment_nonneg,
        test_loss_cross_bundle_independence_nonneg,
        test_loss_recon_finite,
        test_loss_write_policy_finite,
        test_loss_prefix_anchor_gradient_flow_cross_attn,
        test_loss_bundle_axis_alignment_gradient_flow,
        test_trainer_three_step_cpu_smoke,
        test_trainer_save_and_reload_roundtrip,
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
    print(f"\nall {len(tests)} v4.6 training tests passed")


if __name__ == "__main__":
    _run_all()
