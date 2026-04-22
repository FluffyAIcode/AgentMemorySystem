# v4 implementation spec — per-PR design for v4.1 through v4.5

Companion to `ARCHITECTURE_v4.md`. `ARCHITECTURE_v4.md` says **what** the system is; this document says **how** each follow-up PR builds it, which v3.46 code is ported vs rewritten, and what the tests assert.

Scope choice: **v4.5 ships the end-to-end inference path (write → encode → retrieve → attend → inject → generate) with a CPU smoke test on a small backbone (GPT-2 / `distilgpt2`).** A full Trainer4 + benchmark parity run on GPU is **v4.6**, scoped separately. Trainer is not in this batch because (a) v4 has five new loss terms, (b) training convergence on the real backbone takes its own engineering, and (c) conflating a design-drift fix with a training-convergence fix in one PR would make failures hard to diagnose. v4.5 proves the skeleton composes correctly; v4.6 proves it trains.

---

## 1. v4.1 — core + geometry primitives

**Scope files**: `ams_v4/core/mem_entry.py`, `ams_v4/core/mem_store.py`, `ams_v4/bundles/base.py`.

**What ports from v3.46 (with edits)**:

| v3.46 class | ams_v4 location | Edits |
|---|---|---|
| `RiemannianMetric` | `ams_v4/bundles/base.py` | Parameterize `d_base` per-bundle instead of `c.d_M` global; remove coupling to `Cfg` |
| `FiberConnection` | `ams_v4/bundles/base.py` | Parameterize `(d_base, d_fiber)` per-bundle |
| `FiberTransporter` | `ams_v4/bundles/base.py` | Accept per-bundle `cfg` (uses `norm_correction_interval` only) |
| `GeodesicSolver` | `ams_v4/bundles/base.py` | Accept per-bundle `cfg`; path shape unchanged |
| `DirectionTree` (internal `_Node`, `_ins`, `_split`, `_best`, `_beam_retrieve`) | `ams_v4/core/mem_store.py::DirectionTreeV4` | Drop v3.46's cluster-crowding rerank (that was a workaround for missing axes); drop the `AMM` cross-coupling; drop `wte_normed` / `content_classifier` path (topic bundle handles that natively in v4.3) |

**What is new**:

- `MemEntry.__post_init__`-style shape validation via `MemEntry.assert_no_raw_large_fields`.
- `MemStore.add(entry)` that inserts into **three** trees (`tree_time`, `tree_topic`, `tree_ctx`), uses each respective `*_dirn`.
- `MemStore.verify_consistency()` running §6 invariants 1–3 and 6 (4 and 5 require the kakeya registry which lands in v4.3; `verify_consistency` will accept an optional `registry` argument and skip those checks when it's None).

**Tests (ams_v4/tests/test_v41.py)**:

- `test_metric_spd`: `g(x)` is symmetric positive-definite for random `x`.
- `test_connection_antisymmetric`: `FiberConnection(x, v)` output is antisymmetric (`A + A^T ≈ 0`).
- `test_transporter_preserves_norm`: after `FiberTransporter(fiber, path)`, output norm is within 1 % of input norm (closed path).
- `test_geodesic_endpoints`: `GeodesicSolver(p0, p1)` returns path with `path[:, 0] = p0` and `path[:, -1] = p1` (both within 1e-4).
- `test_direction_tree_insert_retrieve`: insert 20 random MemEntries, retrieve with query == one of them, assert that memory is in top-3.
- `test_memstore_add_routes_to_all_three_trees`: after `add(entry)`, `mid` is present in `tree_time`, `tree_topic`, `tree_ctx`.
- `test_memstore_invariant_no_raw_large_fields`: try to attach a raw `(1536,)` tensor as an attribute on MemEntry, call `assert_no_raw_large_fields`, expect AssertionError.

**Exit criterion**: all tests pass on CPU.

---

## 2. v4.2 — three encoders + three concrete Bundle subclasses

**Scope files**: `ams_v4/bundles/temporal.py`, `ams_v4/bundles/topic.py`, `ams_v4/bundles/context.py`.

### 2.1 `TimeEncoder`

Input: `(hidden_state: (B, d_LLM), time_scalars: (B, 3), surprise: (B,))`.

```
sinusoidal_emb(time_scalars) → (B, 2 * d_time)             # 3 scalars × sin/cos bases
time_embed = MLP(sinusoidal_emb)                           # (B, d_time)
base = LayerNorm(time_embed + Linear(hidden))              # (B, d_time)
fiber = MLP(concat(hidden, base, surprise_broadcast))      # (B, d_F_time)
dirn  = F.normalize(base, dim=-1)                          # (B, d_time), unit
```

Sinusoidal encoding choice: each scalar → 2⌊d_time/3⌋ features with exponentially-spaced frequencies (standard Fourier feature trick). Prevents the MLP from having to learn time-scale invariance from scratch.

### 2.2 `TopicEncoder`

Input: `(hidden_state: (B, d_LLM), content_token_ids: List[int], wte_normed: (V, d_LLM))`.

```
# IDF-weighted content centroid
idf_w = idf[content_token_ids]                             # (L,)
c_mean = sum(idf_w[i] * wte_normed[id_i]) / sum(idf_w)     # (d_LLM,)
# Project to topic base space
mixed = Linear_down(c_mean + Linear(hidden))               # (d_topic,)
base  = F.normalize(mixed, dim=-1)                         # (d_topic,), ||·||=1
fiber = MLP(concat(hidden, base))                          # (d_F_topic,)
dirn  = base                                               # already unit
```

Notes:
- Batches: the IDF-weighted mean runs per-batch; `content_token_ids` becomes `List[List[int]]` with ragged length, handled via a loop in v4.2 (optimized later).
- IDF is computed over a corpus snapshot; if no corpus provided, fall back to uniform weighting.
- `base.shape[-1] = d_topic`; on the sphere by construction (no separate normalization loss).

### 2.3 `ContextEncoder`

Input: `(hidden_state: (B, d_LLM), session_summary: (B, d_LLM), prev_turns: Optional[(B, T_prev, d_LLM)])`.

```
if prev_turns is not None:
    # Attention pool over prev turns, with hidden as query
    q = Linear_q(hidden).unsqueeze(1)                      # (B, 1, d_attn)
    k = Linear_k(prev_turns)                               # (B, T_prev, d_attn)
    v = Linear_v(prev_turns)                               # (B, T_prev, d_attn)
    attn_out = softmax(q @ k.T / sqrt(d_attn)) @ v         # (B, 1, d_attn)
    attn_out = attn_out.squeeze(1)                         # (B, d_attn)
else:
    attn_out = zeros(B, d_attn)

mixed = Linear(concat(hidden, session_summary, attn_out))  # (d_ctx,)
base  = LayerNorm(mixed)                                   # (d_ctx,)
fiber = MLP(concat(hidden, base, session_summary))         # (d_F_ctx,)
dirn  = F.normalize(base, dim=-1)                          # (d_ctx,), unit
```

### 2.4 Three Bundle subclasses

Each subclass:
- owns its `RiemannianMetric`, `FiberConnection`, `FiberTransporter` (Topic skips `GeodesicSolver` — uses great-circle).
- owns its `canonical_axis` as an `nn.Parameter` of shape `(d_base,)`, initialized randomly + unit-normalized (re-normalized every forward or by a small penalty; v4.2 uses explicit re-normalize in `canonical_axis()` accessor).
- `encode` delegates to the corresponding encoder.
- `transport` delegates to its `FiberTransporter` with a path built from its geodesic (or great-circle for topic).

**Tests (ams_v4/tests/test_v42.py)**:

- `test_time_encoder_shapes`: output shapes match `(B, d_time)`, `(B, d_F_time)`, `(B, d_time)`.
- `test_time_dirn_unit_norm`: `dirn` is unit-norm within 1e-4.
- `test_topic_base_on_sphere`: `base` has `||·||=1` within 1e-4 for random hidden / ids.
- `test_context_encoder_no_prev_turns`: when `prev_turns=None`, no crash, shapes correct.
- `test_context_encoder_with_prev_turns`: `prev_turns` of shape `(2, 5, d_LLM)` consumed without shape error.
- `test_bundle_encode_returns_three_tensors`: each bundle's `.encode(...)` returns `(base, fiber, dirn)` of correct shape.
- `test_canonical_axis_unit_norm`: each bundle's `canonical_axis()` returns unit-norm tensor.
- `test_bundle_transport_preserves_norm`: `bundle.transport(fiber_src, base_src, base_dst)` preserves fiber norm within 1 %.

**Exit criterion**: all tests pass on CPU.

---

## 3. v4.3 — kakeya (multi-set + alignment)

**Scope files**: `ams_v4/kakeya/alignment.py`, `ams_v4/kakeya/set.py`, `ams_v4/kakeya/registry.py`.

### 3.1 `alignment.py` math

Four pure functions, no state:

```python
pushforward(axis_in_base, base_to_field) → direction_in_field   # @ matmul
project_into_pca(direction_in_field, basis) → coeff_in_pca      # basis @ direction
alignment_error(t_dir, target) → float                          # ||t_dir - normalize(target)||
solve_aligned_t_dir(coeffs, target_direction, tol) → (t_dir, err)
    # v4.3 impl: t_dir = target_direction / ||target_direction||
    # Future work can balance alignment vs coeff-variance.
```

`base_to_field` is a learned linear map that lives on the bundle, initialized randomly during first `KakeyaRegistry.build` and updated whenever the registry rebuilds. It is the bridge between `d_base = 8~16` and `d_field = 1536`.

### 3.2 `KakeyaSet`

```python
class KakeyaSet:
    def __init__(self, set_idx, owner_bundle_name, compressed_fields, cfg):
        ...  # skeleton None, inactive

    def build(self, vecs: (N, d_field), bundle_axis_pushforward: (d_field,)) -> None:
        # 1. PCA on vecs → (basis: (d_eff, d_field), mean: (d_field,), d_eff)
        basis, mean, d_eff = _compute_pca(vecs, cfg.kakeya_variance_ratio)
        # 2. Project pushforward into PCA → target coeff, normalize → t_dir
        proj = basis @ bundle_axis_pushforward
        t_dir = proj / (proj.norm() + 1e-8)
        # 3. coeffs = (vecs - mean) @ basis.T
        coeffs = (vecs - mean) @ basis.T
        # 4. perp = coeffs - (coeffs @ t_dir).unsqueeze(-1) * t_dir
        # 5. centers = spherical_kmeans(F.normalize(perp, -1), cfg.kakeya_K)
        # 6. store KakeyaSkeleton4

    def encode(self, v: (d_field,)) -> CompressedVec:
        # coeff = (v - mean) @ basis.T
        # alpha = coeff @ t_dir
        # perp = coeff - alpha * t_dir
        # seg_id = argmax(centers @ perp / ||perp||)
        # t = perp @ centers[seg_id]
        # residual = perp - t * centers[seg_id]
        # residual_idx = topk(|residual|, d_res).indices
        # residual_vals = residual[residual_idx]

    def decode(self, cv, device) -> (d_field,):
        # residual_full = scatter(residual_vals, residual_idx, zeros(d_eff))
        # perp_approx = cv.t * centers[cv.seg_id] + residual_full
        # coeff_approx = cv.alpha * t_dir + perp_approx
        # v_approx = coeff_approx @ basis + mean

    def verify_alignment(self, bundle_axis_pushforward) -> float:
        # alignment_error(self.skeleton.t_dir, project_into_pca(pushforward, basis))
```

`_compute_pca` and `spherical_kmeans` are **ported verbatim** from `kakeya_codec.py::KakeyaCodec._compute_pca` and `._spherical_kmeans` — they are correct and we don't gain anything by rewriting.

### 3.3 `KakeyaRegistry`

```python
class KakeyaRegistry:
    def __init__(self, cfg):
        self.sets: List[KakeyaSet] = []
        self._routing: List[Tuple[str, List[str]]] = [
            ("time",  ["semantic_emb"]),
            ("topic", ["semantic_emb", "content_wte_mean"]),
            ("ctx",   ["context_descriptor"]),
            ("topic", ["content_wte_mean"]),
        ]
        self._base_to_field_maps: Dict[str, Tensor] = {}  # per-bundle, per-field-concat

    def define_sets(self, routing): ...

    def build(self, field_corpus, bundle_axes):
        # field_corpus: {field_name: (N, d_field)}
        # bundle_axes:  {bundle_name: (d_base,)}
        for i, (owner, fields) in enumerate(self._routing):
            # Concat fields along last dim to form the set's input
            vecs = torch.cat([field_corpus[f] for f in fields], dim=-1)
            # Build or fetch the base_to_field map for this (owner, fields) combo
            key = f"{owner}::{'+'.join(fields)}"
            if key not in self._base_to_field_maps:
                d_base = {"time": cfg.d_time, "topic": cfg.d_topic, "ctx": cfg.d_ctx}[owner]
                d_field = vecs.shape[-1]
                self._base_to_field_maps[key] = torch.randn(d_base, d_field) / sqrt(d_base)
            # Pushforward
            axis = bundle_axes[owner]
            axis_in_field = axis @ self._base_to_field_maps[key]
            # Build set
            kset = KakeyaSet(set_idx=i, owner_bundle_name=owner,
                             compressed_fields=fields, cfg=cfg)
            kset.build(vecs, axis_in_field)
            self.sets.append(kset)

    def encode_memory_fields(self, fields: Dict[str, Tensor]) -> KakeyaHandle:
        handle = KakeyaHandle()
        for kset in self.sets:
            if not kset.is_active: continue
            # Build the set's concatenated input for this memory
            try:
                vec = torch.cat([fields[f] for f in kset.compressed_fields], dim=-1)
            except KeyError:
                continue  # memory is missing one of the fields; skip this set
            cv = kset.encode(vec)
            for f in kset.compressed_fields:
                handle.entries.setdefault(f, []).append(cv)
        return handle

    def decode_field(self, handle, field_name, preferred_set_idx=None, device=None):
        if field_name not in handle.entries: return None
        cvs = handle.entries[field_name]
        if preferred_set_idx is not None:
            cvs = [cv for cv in cvs if cv.set_idx == preferred_set_idx]
        if not cvs: return None
        cv = cvs[0]
        kset = self.sets[cv.set_idx]
        full = kset.decode(cv, device)
        # Slice out this field from the concatenated decoded vector
        offset, length = self._field_offset_in_set(kset, field_name)
        return full[offset:offset + length]

    def verify_invariants(self, n_entries):
        errs = []
        if n_entries >= cfg.kakeya_min_entries:
            n_active = sum(1 for s in self.sets if s.is_active)
            if n_active < 2:
                errs.append(f"invariant 3 violated: active sets = {n_active}")
        for kset in self.sets:
            if not kset.is_active: continue
            # Need bundle_axis_pushforward to verify, caller must recompute; here we
            # just verify skeleton exists (alignment fine-check is in tests)
        return errs
```

`_field_offset_in_set` is a tiny helper: when a set compresses `[f1, f2]` with dims `[d1, d2]`, decoded output is `(d1 + d2,)`; offsets are `{f1: (0, d1), f2: (d1, d2)}`.

### 3.4 Tests (ams_v4/tests/test_v43.py)

- `test_pushforward_matches_linear_map`: `pushforward(e_i, M)` returns `M[i]`.
- `test_project_into_pca_idempotent`: projecting a basis vector into its own PCA subspace returns a one-hot in PCA coords.
- `test_kakeya_set_build_activates`: after `build`, `is_active = True`, skeleton shapes correct.
- `test_kakeya_set_alignment`: `verify_alignment(pushforward)` ≤ `cfg.kakeya_alignment_tol`.
- `test_kakeya_set_encode_decode_roundtrip`: for random `v`, `||decode(encode(v)) - v||_2 / ||v||_2 ≤ cfg.kakeya_reconstruction_tol`.
- `test_registry_has_multiple_sets`: after `build`, `len(registry.sets) == 4`.
- `test_registry_encode_handle_covers_all_fields`: `handle.entries` includes every field listed in routing.
- `test_registry_decode_field_returns_right_shape`: decoded field tensor has shape `(d_field,)`.

**Exit criterion**: all tests pass; round-trip reconstruction error median ≤ 0.15 across 100 random vectors (conservative bar matching §6 invariant 5).

---

## 4. v4.4 — BundleQueryHeads + CrossBundleAttention

**Scope files**: `ams_v4/attention/query_heads.py`, `ams_v4/attention/cross_bundle.py`.

### 4.1 `BundleQueryHeads`

```python
class BundleQueryHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_LLM)
        self.q_time  = nn.Linear(cfg.d_LLM, cfg.d_F_time)
        self.q_topic = nn.Linear(cfg.d_LLM, cfg.d_F_topic)
        self.q_ctx   = nn.Linear(cfg.d_LLM, cfg.d_F_ctx)

    def forward(self, hidden: (B, d_LLM)) -> Dict[str, Tensor]:
        h = self.ln(hidden)
        return {
            "time":  self.q_time(h),       # (B, d_F_time)
            "topic": self.q_topic(h),      # (B, d_F_topic)
            "ctx":   self.q_ctx(h),        # (B, d_F_ctx)
        }
```

### 4.2 `CrossBundleAttention`

```python
class CrossBundleAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.query_heads = BundleQueryHeads(cfg)
        self.attn_time  = nn.MultiheadAttention(cfg.d_F_time,  cfg.n_heads_time,  batch_first=True)
        self.attn_topic = nn.MultiheadAttention(cfg.d_F_topic, cfg.n_heads_topic, batch_first=True)
        self.attn_ctx   = nn.MultiheadAttention(cfg.d_F_ctx,   cfg.n_heads_ctx,   batch_first=True)
        # Per-slot lift: one Linear per slot, bundle-local
        self.lift_time  = nn.ModuleList([nn.Linear(cfg.d_F_time,  cfg.d_LLM)
                                         for _ in range(cfg.prefix_slots_time)])
        self.lift_topic = nn.ModuleList([nn.Linear(cfg.d_F_topic, cfg.d_LLM)
                                         for _ in range(cfg.prefix_slots_topic)])
        self.lift_ctx   = nn.ModuleList([nn.Linear(cfg.d_F_ctx,   cfg.d_LLM)
                                         for _ in range(cfg.prefix_slots_ctx)])
        self.prefix_ln = nn.LayerNorm(cfg.d_LLM)

    def forward(self, hidden_state, entries, mem_mask=None):
        B = hidden_state.shape[0]
        M = len(entries)
        q = self.query_heads(hidden_state)         # 3 queries

        # Stack fibers from entries
        def stack_field(attr):
            return torch.stack([getattr(e, attr) for e in entries], dim=0)\
                     .unsqueeze(0).expand(B, -1, -1)    # (B, M, d_F_*)

        K_time  = V_time  = stack_field("time_fiber")   # (B, M, d_F_time)
        K_topic = V_topic = stack_field("topic_fiber")  # (B, M, d_F_topic)
        K_ctx   = V_ctx   = stack_field("ctx_fiber")    # (B, M, d_F_ctx)

        # Three attentions (per-bundle)
        out_time,  _ = self.attn_time (q["time"].unsqueeze(1),  K_time,  V_time,  key_padding_mask=mem_mask)
        out_topic, _ = self.attn_topic(q["topic"].unsqueeze(1), K_topic, V_topic, key_padding_mask=mem_mask)
        out_ctx,   _ = self.attn_ctx  (q["ctx"].unsqueeze(1),   K_ctx,   V_ctx,   key_padding_mask=mem_mask)

        out_time  = out_time.squeeze(1)   # (B, d_F_time)
        out_topic = out_topic.squeeze(1)  # (B, d_F_topic)
        out_ctx   = out_ctx.squeeze(1)    # (B, d_F_ctx)

        # Lift to d_LLM per slot
        slots_time  = torch.stack([lh(out_time)  for lh in self.lift_time],  dim=1)  # (B, prefix_slots_time, d_LLM)
        slots_topic = torch.stack([lh(out_topic) for lh in self.lift_topic], dim=1)
        slots_ctx   = torch.stack([lh(out_ctx)   for lh in self.lift_ctx],   dim=1)

        prefix = torch.cat([slots_time, slots_topic, slots_ctx], dim=1)  # (B, L_mem, d_LLM)
        return self.prefix_ln(prefix)
```

### 4.3 Tests (ams_v4/tests/test_v44.py)

- `test_query_heads_shapes`: three queries with correct fiber dims.
- `test_cross_bundle_forward_shape`: output is `(B, L_mem, d_LLM)` exactly.
- `test_cross_bundle_handles_empty_mem`: with `entries = []`, returns zero prefix (caller must check before calling; we assert that call raises a clear error here).
- `test_cross_bundle_mask_respected`: masking all but one entry concentrates attention on that entry (check via `attn.forward` with need_weights in a separate variant — for the SUT we only assert shape + finite).
- `test_cross_bundle_gradient_flows`: `prefix.sum().backward()` produces non-zero gradients on `query_heads.q_time.weight` etc.

---

## 5. v4.5 — EmbBridge4 + LLMBackbone4 + MemLLM4 + CPU smoke test

**Scope files**: `ams_v4/projection/bridge.py`, `ams_v4/bridge/backbone.py` (new), `ams_v4/bridge/memllm.py`, `ams_v4/tests/test_v45_smoke.py`.

### 5.1 `LLMBackbone4`

A thin wrapper over HF `AutoModelForCausalLM`:
- `__init__(cfg)`: stores cfg, defers model load to `.load(name=...)`.
- `.load(name=None)`: loads HF model by `cfg.llm_name` (default Qwen 2.5 1.5B) or an override. Keeps model in requested dtype. Freezes backbone parameters (we are NOT fine-tuning the backbone in v4).
- `.wte` property → `model.get_input_embeddings()`.
- `.forward_with_prefix(prefix: (B, L_mem, d_LLM), ids: (B, T), mask: (B, T))` → backbone output: hidden states and logits. Handles the prefix merge.
- `.generate_with_prefix(prefix, ids, mask, max_new_tokens, greedy)` → token sequence; uses HF `generate()` after prefix is prepended.

v3.46 `LLMBackbone` (at `scheme_b_v344.py:456`) is a good template; port it, strip the v3.46-specific hooks, add the prefix-merge helpers.

### 5.2 `EmbBridge4`

```python
class EmbBridge4(nn.Module):
    def __init__(self, cfg): super().__init__(); self.cfg = cfg
        self.prefix_post_ln = nn.LayerNorm(cfg.d_LLM)

    def build_inputs(self, prefix, ids, mask, wte):
        # prefix: (B, L_mem, d_LLM)
        # ids:    (B, T)
        # mask:   (B, T)
        tok_emb = wte(ids)                                        # (B, T, d_LLM)
        prefix_n = self.prefix_post_ln(prefix)
        input_embeds = torch.cat([prefix_n, tok_emb], dim=1)      # (B, L_mem + T, d_LLM)
        prefix_mask = torch.ones(mask.shape[0], self.cfg.L_mem,
                                 dtype=mask.dtype, device=mask.device)
        input_mask = torch.cat([prefix_mask, mask], dim=1)         # (B, L_mem + T)
        return input_embeds, input_mask
```

### 5.3 `MemLLM4`

```python
class MemLLM4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = LLMBackbone4(cfg)
        self.bundle_time  = TemporalBundle(cfg)
        self.bundle_topic = TopicBundle(cfg)
        self.bundle_ctx   = ContextBundle(cfg)
        self.kakeya       = KakeyaRegistry(cfg)
        self.cross_attn   = CrossBundleAttention(cfg)
        self.bridge       = EmbBridge4(cfg)
        self.store        = MemStore(cfg)
        self._session_summary = None  # updated in write(), used by ContextBundle
        self._writes_since_kakeya_build = 0

    def load(self, name=None):
        self.backbone.load(name)

    @property
    def tok(self): return self.backbone.tok

    def _hidden_of(self, text):
        ids, mask = self._tokenize(text)
        with torch.no_grad():
            hs = self.backbone.hidden_states(ids, mask)  # (B, T, d_LLM)
            pooled = hs.mean(dim=1)                       # (B, d_LLM); simple mean pooling in v4
        return pooled, ids, mask

    def write(self, text, training_mode=False):
        hidden, ids, mask = self._hidden_of(text)
        time_scalars = self._current_time_scalars(hidden.shape[0])
        surprise = torch.zeros(hidden.shape[0], device=hidden.device)  # v4 deferred
        time_b, time_f, time_d   = self.bundle_time.encode(hidden, time_scalars=time_scalars, surprise=surprise)
        content_ids = ids[0].tolist()
        topic_b, topic_f, topic_d = self.bundle_topic.encode(hidden, content_token_ids=content_ids,
                                                             wte_normed=self._wte_normed())
        ctx_b, ctx_f, ctx_d = self.bundle_ctx.encode(hidden, session_summary=self._get_session_summary(hidden),
                                                    prev_turns=None)
        # Build MemEntry
        entry = MemEntry(
            mid=-1,  # assigned by MemStore.add
            time_base=time_b[0], time_fiber=time_f[0], time_dirn=time_d[0],
            topic_base=topic_b[0], topic_fiber=topic_f[0], topic_dirn=topic_d[0],
            ctx_base=ctx_b[0], ctx_fiber=ctx_f[0], ctx_dirn=ctx_d[0],
            surprise=0.0, ts=self.store._next_mid, last=self.store._next_mid, cnt=0,
            source_text=text, content_token_ids=content_ids,
        )
        # Kakeya: encode large fields (semantic_emb, content_wte_mean, context_descriptor)
        large_fields = self._extract_large_fields(hidden, content_ids, ctx_b[0])
        if self.kakeya.sets and self.kakeya.sets[0].is_active:
            entry.kakeya_handle = self.kakeya.encode_memory_fields(large_fields)
        mid = self.store.add(entry)
        # Maybe build kakeya registry
        self._writes_since_kakeya_build += 1
        if (len(self.store) >= self.cfg.kakeya_min_entries
                and (not self.kakeya.sets or not self.kakeya.sets[0].is_active)):
            self._build_kakeya_from_store()
        return mid

    def prepare_decode_context(self, ids, mask, update_stats=False):
        with torch.no_grad():
            hs = self.backbone.hidden_states(ids, mask)       # (B, T, d_LLM)
            q_hidden = hs.mean(dim=1)                         # (B, d_LLM)
        entries = self.store.all_entries()
        if not entries:
            prefix = torch.zeros(q_hidden.shape[0], self.cfg.L_mem, self.cfg.d_LLM,
                                 device=q_hidden.device, dtype=q_hidden.dtype)
        else:
            prefix = self.cross_attn(q_hidden, entries)
        return DecodeContext4(prefix=prefix, n_memories=len(entries))

    def generate(self, prompt, mt=40, greedy=True):
        ids, mask = self._tokenize(prompt)
        ctx = self.prepare_decode_context(ids, mask)
        out_ids = self.backbone.generate_with_prefix(ctx.prefix, ids, mask,
                                                     max_new_tokens=mt, greedy=greedy)
        return self.tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
```

Helpers (`_tokenize`, `_current_time_scalars`, `_wte_normed`, `_get_session_summary`, `_extract_large_fields`, `_build_kakeya_from_store`) are small bookkeeping methods, documented inline.

### 5.4 CPU smoke test (ams_v4/tests/test_v45_smoke.py)

To stay CPU-runnable in the cloud agent env, the smoke test uses **`sshleifer/tiny-gpt2`** (a 7M-param GPT-2) instead of Qwen 2.5 1.5B. That's 1000× smaller and runs in seconds on CPU.

```python
def test_v45_cpu_smoke():
    cfg = Cfg4(
        llm_name="sshleifer/tiny-gpt2",
        d_LLM=2,       # tiny-gpt2 has hidden_size=2
        vocab_size=50257,
        # Keep bundle dims small too
        d_time=4, d_F_time=8, n_heads_time=2,
        d_topic=4, d_F_topic=8, n_heads_topic=2,
        d_ctx=4, d_F_ctx=8, n_heads_ctx=2,
        L_mem=6, prefix_slots_time=2, prefix_slots_topic=2, prefix_slots_ctx=2,
        kakeya_min_entries=4, n_kakeya_sets=4,
        strict_shape_checks=True,
    )
    m = MemLLM4(cfg)
    m.load()  # loads tiny-gpt2

    # Write 6 memories
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
        assert mid >= 0

    # Verify store invariants
    assert len(m.store) == 6
    errs = m.store.verify_consistency()
    assert not errs, errs
    m.store.assert_all_large_fields_compressed()

    # Verify kakeya activated
    active = sum(1 for s in m.kakeya.sets if s.is_active)
    assert active >= 2, f"abstract invariant 3: need >= 2 active sets, got {active}"

    # Generate — check it runs and returns a non-empty string of reasonable length
    out = m.generate("What does a cat do?", mt=10, greedy=True)
    assert isinstance(out, str)
    # The model is tiny and untrained on any fine-tuning; we only assert it ran.
```

**Exit criterion**: smoke test passes on CPU in the cloud agent env (< 60 s). This confirms:
1. The full v4 stack composes end-to-end.
2. §6 invariants 1, 2, 3 hold on live data.
3. Gradient / autograd doesn't choke on any tensor shape.
4. `MemLLM4.generate()` returns.

**Non-goals for v4.5 smoke test** (these are for v4.6):
- Training convergence.
- Hit-rate on any benchmark.
- Any claim that v4 beats v3.46. (It won't fresh-init; that's expected — training is what makes the architecture pay off.)

---

## 6. Beyond v4.5 — what v4.6 needs to do

Out of scope for this batch, stated here so nobody mistakes v4.5 for "done":

1. **Trainer4**: new loss terms per `Cfg4.loss_weights`:
   - `recon`: kakeya-decode ≈ original for each compressed field.
   - `bundle_axis_alignment`: each `KakeyaSet.t_dir` stays ≤ `alignment_tol` from its bundle's canonical axis pushforward, as `base_to_field` learns.
   - `cross_bundle_independence`: per-pair cosine between bundle outputs on mismatched content should be low (prevents the three bundles from collapsing to copies).
   - `prefix_semantic_anchor`: attention prefix, when decoded through backbone, has positive sim with target answer token embedding.
   - `write_policy`: same spirit as v3.46 write-gate but simplified.
2. **GPU training driver** (`train_v4.py`) matching v3.46's `train_v346.py`.
3. **Parity harness** (`session_viability_v4.py`) — same 5 modes, same N=10/20, same 10 queries, comparing `A_ams_prefix` / `C_ams_hybrid` between v3.46-trained and v4-trained.
4. Merge gate unchanged: `A_ams_prefix` / `C_ams_hybrid` at N=20 must be strictly higher on v4-trained than v3.46-trained.
