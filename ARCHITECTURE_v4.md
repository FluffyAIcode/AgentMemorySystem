# AMS v4 architecture — realignment to the abstract spec

**Branch**: `AgentMemory/v347-architecture-realign-b7fa`
**Base**: `main` @ `c3b1422`
**Status**: design + compilable skeleton only. No runtime behavior changes to v3.12 / v3.46 paths; those continue to live in `scheme_b_v344.py` and `kakeya_codec.py` unchanged.

---

## 0. Why this branch exists

The abstract architecture AMS was supposed to implement is:

> **Multiple Kakeya sets compress the full context data. These Kakeya sets are linked on different fiber bundles. The fiber bundles carry memory encoding around time, topic, and background (context). An attention mechanism forms the current context window.**

An audit of `scheme_b_v344.py` + `kakeya_codec.py` on the PR #29 branch showed that four of the five structural claims in that sentence are drifted or missing:

| Abstract requirement | v3.46 reality | Status |
|---|---|---|
| **Multiple** Kakeya sets | Exactly one `KakeyaSkeleton` per store (single PCA + one t_dir + K segment centers on one compressed field) | drifted |
| Compress **full context data** | Only `semantic_emb` (1 field) is compressed; `base`/`fiber`/`dirn`/`context_descriptor`/`content_token_ids` are raw | drifted |
| Linked on **fiber bundles** (plural) | Kakeya sidecar and fiber bundle are two disjoint subsystems with zero cross-references; one bundle | drifted |
| Axes: **time / topic / background** | `FibEncoder.forward(h, base, surprise)` — none of the three axes is an input; `ts`/`last`/`cnt` are scalar bookkeeping; `context_descriptor` is a side-channel slot; `cluster_id` is an offline-KMeans integer tag | drifted |
| Attention forms context window | `FiberAttn` + `QFormerProj` + `EmbBridge.inject` (present, runs) | **kept** |

Consequence: PR #29's `A_ams_prefix` = 50 % / `C_ams_hybrid` = 70 % at N=20 is the **downstream symptom** of the three upstream drifts. The fiber bundle does not carry the information the attention step would need to retrieve, so tuning decode-side logit shaping (content_bias, overlap gates, rerank weights — what v3.3x–v3.4x has been doing) cannot close the gap.

This branch realigns the architecture. It does not delete v3.46 code; it defines `ams_v4/` as a parallel package that can coexist during migration. v3.46 stays callable for regression testing.

---

## 1. Abstract → concrete mapping

Five things the design has to get right, and how `ams_v4/` expresses each.

### 1.1 Multiple Kakeya sets

A `KakeyaSet` is a Kakeya-like skeleton (PCA basis + *one* distinguished direction + K segment centers on the perpendicular sphere + sparse residuals). The store holds **B** of them, not 1.

```
KakeyaRegistry
  ├── KakeyaSet[0]   owned_axis = "time"        skeleton_fields = {"semantic_emb", ...}
  ├── KakeyaSet[1]   owned_axis = "topic"       skeleton_fields = {"semantic_emb", "content_wte"}
  ├── KakeyaSet[2]   owned_axis = "background"  skeleton_fields = {"context_descriptor", ...}
  └── KakeyaSet[3..] ...
```

Each set is bound to **one** bundle axis (§1.3), inherits its distinguished direction from that axis's geometry, and compresses **whichever** memory fields are relevant to that axis. The registry owns routing: given a `MemEntry`, which set encodes which field.

Compression field set is no longer "just `semantic_emb`". The default v4 routing is:

| Memory field | Dim | Routed to set | Rationale |
|---|---:|---|---|
| `base` (point on M) | d_M = 8 | none (raw, it IS the bundle coordinate) | |
| `fiber` (vector in F) | d_F = 32 | none (raw) | |
| `dirn` (unit in M) | d_M = 8 | none (raw, indexed by `DirectionTree`) | |
| `semantic_emb` | d_LLM (1536) | time + topic sets (two skeletons per memory) | gives cross-axis redundancy that enables §1.4 attention to combine |
| `content_wte_mean` | d_LLM | topic set | |
| `context_descriptor` | d_LLM | background set | |
| `content_token_ids` | var | none (sparse, already small) | |

### 1.2 Compress **all** context data

Follows from §1.1: because the registry holds B sets each with its own field subset, any field with dim ≥ 256 is compressed by at least one set. No raw `d_LLM`-sized tensor is stored on `MemEntry` in v4; every large tensor is held as a per-set `CompressedVec` and reconstructed on demand. Small fields (integer ids, scalars) stay raw.

This is a hard storage invariant, enforced by `MemStore.assert_all_large_fields_compressed()` in debug builds.

### 1.3 Linked on **different fiber bundles**

There are **three fiber bundles**, not one:

| Bundle | Base space | Fiber space | What lives here |
|---|---|---|---|
| `TemporalBundle` | `B_time = R^{d_time}` — a learned embedding of (absolute time, recency, write-count) | `F_time ≅ R^{d_F_time}` | how each memory "looks" across time — decay, re-access, consolidation traces |
| `TopicBundle` | `B_topic = S^{d_topic - 1}` — a unit sphere where each point is a topic direction (content centroid) | `F_topic ≅ R^{d_F_topic}` | content-side encoding: what the memory is *about* |
| `ContextBundle` | `B_ctx = R^{d_ctx}` — the session/background embedding | `F_ctx ≅ R^{d_F_ctx}` | situational framing: who / where / why this memory was formed |

Each bundle has its own `RiemannianMetric`, `FiberConnection`, `FiberTransporter`, and `GeodesicSolver` — structurally the same building blocks as v3.46, but instantiated **three times** with different base spaces and independent parameters.

The "Kakeya sets are linked on different fiber bundles" clause is implemented by **bundle ownership**: each `KakeyaSet[i]` has exactly one `owner_bundle` field and its distinguished direction `t_dir_i` is constrained to equal the pushforward of that bundle's canonical axis into the compressed PCA subspace. See `ams_v4/kakeya/alignment.py` for the constraint.

### 1.4 Axes = time, topic, background

Each `MemEntry` gets three **coordinate tuples**, one per bundle, instead of v3.46's single `(base, fiber, dirn)`:

```python
@dataclass
class MemEntry:
    mid: int
    # Temporal bundle coordinates
    time_base: Tensor      # (d_time,)       -- point on B_time
    time_fiber: Tensor     # (d_F_time,)     -- fiber at time_base
    time_dirn: Tensor      # (d_time,)       -- unit, for DirectionTree indexing on time
    # Topic bundle coordinates
    topic_base: Tensor     # (d_topic,)      -- on S^{d_topic-1}
    topic_fiber: Tensor    # (d_F_topic,)
    topic_dirn: Tensor     # (d_topic,)
    # Context bundle coordinates
    ctx_base: Tensor       # (d_ctx,)
    ctx_fiber: Tensor      # (d_F_ctx,)
    ctx_dirn: Tensor       # (d_ctx,)
    # Scalars and raw text (unchanged from v3.46)
    surprise: float; ts: float; last: float; cnt: int = 0
    source_text: str = ""
    content_token_ids: List[int] = field(default_factory=list)
    # Large-field handles — COMPRESSED, not raw
    kakeya_handle: KakeyaHandle   # maps field-name -> (set_idx, CompressedVec)
```

The three `(base, fiber, dirn)` triples are produced by **three separate encoders**:

- `TimeEncoder(hidden_state, timestamps) → (time_base, time_fiber, time_dirn)`
- `TopicEncoder(hidden_state, content_tokens, wte_normed) → (topic_base, topic_fiber, topic_dirn)`
- `ContextEncoder(hidden_state, session_summary, prev_turns) → (ctx_base, ctx_fiber, ctx_dirn)`

This is the change that closes the v3.46 gap: the bundle inputs now **explicitly carry** the three axes the abstract spec calls out, instead of depending on whatever `FibEncoder(h, base, surprise)` happens to learn implicitly.

### 1.5 Attention forms the context window

The `CrossBundleAttention` module takes a query `q` (from the current hidden state) and returns a context window — a set of prefix embeddings — by attending over all three bundles simultaneously:

```
prefix = CrossBundleAttention(q)
       = W_o · concat(
             attn(q_time,  K_time,  V_time ),
             attn(q_topic, K_topic, V_topic),
             attn(q_ctx,   K_ctx,   V_ctx  )
         )
```

where `K_*` / `V_*` are derived from the corresponding bundle's fibers (reconstructed through the kakeya sets if compressed), and `q_*` is produced by three query heads (one per bundle) from the current hidden state.

This is what `A_ams_prefix` / `C_ams_hybrid` were benchmarking in v3.46, but now the attention sees **three separately-parameterized bundles** with explicit axes, not one black-box bundle.

---

## 2. Package layout — `ams_v4/`

```
ams_v4/
├── __init__.py                    re-exports public surface
├── core/
│   ├── config.py                  Cfg4  (one dataclass, strict invariants)
│   ├── mem_entry.py               MemEntry, KakeyaHandle
│   ├── mem_store.py               MemStore, DirectionTree-per-bundle
│   └── types.py                   Tensor type aliases, shape tags
├── bundles/
│   ├── base.py                    Bundle (abstract), RiemannianMetric, FiberConnection,
│   │                              FiberTransporter, GeodesicSolver
│   ├── temporal.py                TemporalBundle + TimeEncoder
│   ├── topic.py                   TopicBundle + TopicEncoder
│   └── context.py                 ContextBundle + ContextEncoder
├── kakeya/
│   ├── set.py                     KakeyaSet (single skeleton, one field group, one owner_bundle)
│   ├── registry.py                KakeyaRegistry (owns B sets, routes fields)
│   ├── alignment.py               bundle-axis ↔ kakeya-t_dir alignment constraint
│   └── codec.py                   KakeyaCodecV4 (unified encode/decode; supersedes kakeya_codec.py)
├── attention/
│   ├── cross_bundle.py            CrossBundleAttention
│   └── query_heads.py             three per-bundle query heads
├── projection/
│   └── bridge.py                  EmbBridge4 (prefix assembly + backbone injection)
├── bridge/
│   └── memllm.py                  MemLLM4 — top-level model, composes the above
└── tests/
    └── test_shapes.py             static shape/type checks; no end-to-end yet

ARCHITECTURE_v4.md                 this document
ams_v4/README.md                   short status + roadmap
```

---

## 3. Type contracts (what "compilable skeleton" means)

Every class has:
- full dataclass / `nn.Module` signature
- complete `__init__` field list with types and shapes
- `forward(...)` signature with declared input/output shapes in the docstring
- function body = `raise NotImplementedError("v4-skel: <component>.<method>")` where implementation is pending

This means:
- `python -c "import ams_v4"` succeeds.
- `from ams_v4 import Cfg4, MemLLM4; m = MemLLM4(Cfg4())` succeeds up to the first `forward` call (which raises a clear `NotImplementedError` with the component name — this is intentional, it is the scaffold).
- Static tools (mypy, IDEs, `pydoc`) see the full interface.
- Nothing in `ams_v4/` has unreachable behavior that could accidentally be depended on before implementation.

This PR adds exactly that scaffold. Runtime behavior lands in follow-up PRs, one per `ams_v4/` submodule, each tested in isolation.

---

## 4. Migration plan (v3.46 → v4)

Five follow-up PRs, each independently testable. Each reuses as much v3.46 machinery as possible — the geometry code (`FiberConnection`, `FiberTransporter`, `RiemannianMetric`, `GeodesicSolver`) is correct and **ports as-is into `ams_v4/bundles/base.py`**. Only the *composition* changes.

| PR | Scope | Reuses from v3.46 | New code |
|---|---|---|---|
| v4.1 | `core/` + `bundles/base.py` | `RiemannianMetric`, `GeodesicSolver`, `FiberConnection`, `FiberTransporter` (copy-with-minor-edits) | `Cfg4`, `MemEntry`, `MemStore`, `Bundle` abstract |
| v4.2 | `bundles/temporal.py`, `bundles/topic.py`, `bundles/context.py` + encoders | inspiration only from `FibEncoder`, `CtxEncoder` | three new encoders; time-embedding module |
| v4.3 | `kakeya/` full module | PCA + spherical K-means from `kakeya_codec.py::_compute_pca`, `_spherical_kmeans` | `KakeyaSet`, `KakeyaRegistry`, `alignment.py`, multi-set encode/decode |
| v4.4 | `attention/cross_bundle.py` + `query_heads.py` | inspiration from `FiberAttn` | three-bundle attention, per-bundle query heads |
| v4.5 | `projection/bridge.py` + `bridge/memllm.py` + parity harness against v3.46 | `EmbBridge.inject` prefix-assembly pattern | `EmbBridge4`, `MemLLM4`, regression harness |

**Gate between PRs**: each PR must add its own unit tests to `ams_v4/tests/` and pass them. No v4 PR merges to main until v4.5's parity harness shows `MemLLM4` matching or beating `MemLLM` v3.46 on the `session_viability.py` benchmark (not worse on any of the 5 modes, strictly better on `A_ams_prefix` and `C_ams_hybrid` at N=20).

**v3.46 stays callable** throughout. `scheme_b_v344.py` is not edited. `kakeya_codec.py` stays as-is (reference only — its PCA + spherical-K-means helpers are copied, not imported, into `ams_v4/kakeya/`).

---

## 5. What this design explicitly does not do

- **Not a RAG backend.** Mode B in PR #29 is retained as a retrieval-side diagnostic, not a product. `MemLLM4` has no "inject top-k source_text" code path.
- **Not a knowledge graph.** `DirectionTree` (now three of them, one per bundle) is a continuous-embedding routing structure; no entities, relations, or symbolic query surface.
- **Not a Cfg-knob turning exercise.** `Cfg4` ships with conservative defaults and strict `__post_init__` invariants; adding new Cfg flags requires touching the invariant list.
- **Not a rewrite-from-scratch.** Geometry (metric / connection / transport / geodesic) and compression primitives (PCA / spherical K-means) port directly.

---

## 6. Invariants the design commits to

These are assertable, will be checked in `ams_v4/core/mem_store.py::verify_consistency()`:

1. Every `MemEntry` has exactly one coordinate triple per bundle (three triples total).
2. No `MemEntry` field with shape `(d_LLM,)` or larger is stored raw — it lives in the `KakeyaRegistry`.
3. `KakeyaRegistry` has ≥ 2 sets whenever any bundle has ≥ `min_entries_to_build` memories.
4. Each `KakeyaSet.owner_bundle` is non-null and its `t_dir` satisfies the §1.3 alignment constraint within `alignment_tol = 1e-3`.
5. For every memory `m` and every field `f` listed in `m.kakeya_handle`, `decode(encode(v)) - v` has `||·||_2 / ||v||_2 ≤ reconstruction_tol = 0.15` (conservative initial bar, tightened in v4.3).
6. `CrossBundleAttention(q)` output shape equals `(effective_prefix_slots, d_LLM)`; no silent broadcasting.

---

## 7. Status of this PR

- Document (this file): complete.
- `ams_v4/` skeleton: compilable, all classes stubbed with `NotImplementedError`.
- `ams_v4/tests/test_shapes.py`: static import + construction test only (no forward pass).
- PR #29 / v3.46 code paths: untouched.
- `train_v346.py`, `session_viability.py`, `scheme_b_v344.py`, `kakeya_codec.py`: untouched.

Follow-up work tracked by v4.1–v4.5 in §4.
