"""v4.5 smoke test — end-to-end MemLLM4 on a tiny backbone (sshleifer/tiny-gpt2).

This is intentionally CPU-runnable: tiny-gpt2 has hidden_size=2 and ~7K params
(per-layer), so forward passes are sub-millisecond on CPU. The test only
asserts that the stack composes, runs to completion, and satisfies §6
invariants on live data.

It does NOT assert hit-rate or generation quality — those are v4.6 goals.
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

from ams_v4 import Cfg4
from ams_v4.bridge.memllm import MemLLM4


def _tiny_cfg() -> Cfg4:
    # distilgpt2: hidden_size=768, vocab_size=50257, 82M params.
    # Small enough to run on CPU in ~15 s per test; large enough that d_LLM
    # fields are meaningfully compressible by kakeya.
    return Cfg4(
        llm_name="distilgpt2",
        llm_dtype="fp32",
        d_LLM=768,
        vocab_size=50257,
        # Small bundle dims keep the v4 module param count low on CPU
        d_time=8, d_F_time=16, n_heads_time=2,
        d_topic=16, d_F_topic=32, n_heads_topic=4,
        d_ctx=8, d_F_ctx=16, n_heads_ctx=2,
        L_mem=6, prefix_slots_time=2, prefix_slots_topic=2, prefix_slots_ctx=2,
        n_kakeya_sets=4, kakeya_min_entries=4, kakeya_K=4,
        kakeya_d_res=5,
        # Geometry: smaller path for speed
        n_geo_pts=4, geo_max_steps=20,
    )


def test_v45_cpu_smoke():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    m = MemLLM4(cfg)
    m.load()  # downloads tiny-gpt2 on first run

    texts = [
        "The cat sat on the mat.",
        "Pianos have 88 keys.",
        "Paris is the capital of France.",
        "Python is a programming language.",
        "The Pacific is the largest ocean.",
        "Chess has 16 pieces per side.",
    ]
    for t in texts:
        mid = m.write(t)
        assert mid is not None and mid >= 0, f"write returned {mid}"

    assert len(m.store) == 6

    # §6 invariant 1 — every entry has three triples (auto-asserted by MemStore.add)
    errs = m.store.verify_consistency()
    assert errs == [], f"store invariants failed: {errs}"
    # §6 invariant 2 — no raw d_LLM-sized tensor raw on any entry
    m.store.assert_all_large_fields_compressed()

    # §6 invariant 3 — kakeya registry has ≥ 2 active sets after build
    active = sum(1 for s in m.kakeya.sets if s.is_active)
    assert active >= 2, (
        f"abstract invariant: need ≥ 2 active KakeyaSets, got {active}. "
        f"n_entries = {len(m.store)}, kakeya_min_entries = {cfg.kakeya_min_entries}"
    )

    # §6 invariant 4 — kakeya alignment
    bundle_axes = {
        "time":  m.bundle_time.canonical_axis().detach().cpu().float(),
        "topic": m.bundle_topic.canonical_axis().detach().cpu().float(),
        "ctx":   m.bundle_ctx.canonical_axis().detach().cpu().float(),
    }
    reg_errs = m.kakeya.verify_invariants(len(m.store), bundle_axes=bundle_axes)
    assert reg_errs == [], f"kakeya invariants failed: {reg_errs}"

    # §6 invariant 6 — prefix shape
    ids, mask = m._tokenize("What does a cat do?")
    ctx = m.prepare_decode_context(ids, mask)
    assert ctx.prefix.shape == (1, cfg.L_mem, cfg.d_LLM), \
        f"prefix shape {tuple(ctx.prefix.shape)} != (1, {cfg.L_mem}, {cfg.d_LLM})"
    assert torch.isfinite(ctx.prefix).all(), "prefix has non-finite values"

    # Generate — just check it runs and returns a string
    out = m.generate("What does a cat do?", mt=8, greedy=True)
    assert isinstance(out, str), f"generate returned {type(out).__name__}"
    # With an untrained prefix + a random 7k-param model, the output is
    # gibberish. That's OK for v4.5; we just check shapes compose.
    print(f"    generated (meaningless by design in v4.5): {out!r}")


def _run_all():
    failed = []
    try:
        test_v45_cpu_smoke()
        print("PASS  test_v45_cpu_smoke")
    except Exception:
        print("FAIL  test_v45_cpu_smoke")
        traceback.print_exc()
        failed.append("test_v45_cpu_smoke")

    if failed:
        print(f"\n{len(failed)} failed")
        sys.exit(1)
    print("\nv4.5 smoke test passed")


if __name__ == "__main__":
    _run_all()
