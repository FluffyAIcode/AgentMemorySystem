# `ams_v4/` — AMS v4 realigned architecture

This package is the **design skeleton** for AMS v4. It is **compilable and importable** but every core method raises `NotImplementedError` with a `v4-skel:` marker. Implementation lands in PRs v4.1 through v4.5 (tracked in `ARCHITECTURE_v4.md` §4).

## Why this exists

A prior architectural audit (documented in `ARCHITECTURE_v4.md` §0) found that the abstract AMS spec —

> Multiple Kakeya sets compress the full context data. These Kakeya sets are linked on different fiber bundles. The fiber bundles carry memory encoding around time, topic, and background (context). An attention mechanism forms the current context window.

— had drifted in the v3.46 implementation: one Kakeya set instead of many, one bundle instead of three, no explicit time/topic/context axes, and no linkage between the Kakeya compression layer and the fiber bundle. This package realigns the code to the abstract spec.

## What this package guarantees *today*

1. `import ams_v4` succeeds.
2. `ams_v4.Cfg4()` constructs, validates six structural invariants at `__post_init__`, and rejects common misconfigurations.
3. Every class in the public surface exists with a full type signature.
4. `ams_v4/tests/test_shapes.py` passes (see below).
5. No code path accidentally "silently works" — every unimplemented method raises `NotImplementedError("v4-skel: <component> — lands in v4.X")` so downstream implementers cannot skip a step by accident.

## What this package does NOT do yet

Anything that requires a forward pass. No training, no inference, no checkpointing. Those come one module at a time:

| Follow-up | Module | Ported from v3.46 | New code |
|---|---|---|---|
| v4.1 | `core/` + `bundles/base.py` | `RiemannianMetric`, `FiberConnection`, `FiberTransporter`, `GeodesicSolver` | `Cfg4`, `MemEntry`, `MemStore`, `Bundle` abstract |
| v4.2 | `bundles/temporal.py`, `bundles/topic.py`, `bundles/context.py` | inspiration from `FibEncoder`, `CtxEncoder` | three per-bundle encoders |
| v4.3 | `kakeya/` | PCA + spherical-K-means from `kakeya_codec.py` helpers | `KakeyaSet`, `KakeyaRegistry`, alignment math |
| v4.4 | `attention/` | inspiration from `FiberAttn` | three-bundle attention + query heads |
| v4.5 | `projection/` + `bridge/` + parity harness | `EmbBridge.inject` prefix shape | `EmbBridge4`, `MemLLM4`, regression vs v3.46 |

Each follow-up must add unit tests in `ams_v4/tests/` and must not merge to `main` unless:
- the tests pass; and
- v4.5 specifically: the parity harness shows `MemLLM4` ≥ `MemLLM` v3.46 on the `session_viability.py` benchmark, with strict improvement on `A_ams_prefix` and `C_ams_hybrid` at N=20.

## Running the skeleton tests

```bash
python3 ams_v4/tests/test_shapes.py
# or
python3 -m pytest ams_v4/tests/test_shapes.py -v
```

All six tests should pass. Requires only Python 3.9+ and PyTorch.

## v3.46 coexistence

Nothing in `scheme_b_v344.py`, `kakeya_codec.py`, `train_v346.py`, or `session_viability.py` is modified by this branch. v3.46 is fully functional. PR #29's benchmarks continue to run unchanged. This is intentional: v4 proves itself against v3.46 as the baseline, not by replacing it.

## Further reading

- `ARCHITECTURE_v4.md` (workspace root) — full design document, abstract-to-concrete mapping, migration plan, invariants.
- `SPRINT_CLOSEOUT_v3.46.md` §10 / §10.9 — the decision trail that surfaced the need for this branch.
