# v4.6 training spec

Companion to `ARCHITECTURE_v4.md` / `ARCHITECTURE_v4_IMPL.md`. Covers the trainer, loss terms, training driver, and what counts as "done" for the merge gate.

---

## 1. What trains and what doesn't

**Trainable** (`p.requires_grad = True`):
- `TemporalBundle.encoder` (`TimeEncoder` MLPs + `_axis_raw` canonical axis)
- `TopicBundle.encoder` (`TopicEncoder` MLPs + `_axis_raw`)
- `ContextBundle.encoder` (`ContextEncoder` MLPs + `_axis_raw`)
- Per-bundle `RiemannianMetric` + `FiberConnection` (inside each bundle)
- `CrossBundleAttention` (three `MultiheadAttention` + per-slot lifts + query heads + `prefix_ln`)
- `EmbBridge4` (none in v4.5 — nothing to train)
- `KakeyaRegistry._base_to_field_maps` — **NOT** `nn.Parameter` in v4.5; stay as random init and rebuild-by-corpus, per `KakeyaRegistry.build()`. The kakeya skeletons are rebuilt from store snapshots, not gradient-trained. (Rationale: PCA + spherical-K-means on the training-data corpus is already the "right" answer for compression; gradient-training kakeya would fight the analytic solution.)

**Frozen** (`p.requires_grad = False`):
- `LLMBackbone4.model` (entire HF model)

This is **identical to v3.46's training rule**: LM frozen, adapters trained. Total trainable params on the default `Cfg4()` are ~8M (most in the three `FiberConnection`s and the per-slot lifts).

## 2. Loss terms

Five terms, declared in `Cfg4.loss_weights`. Each has one clear job; one invariant to test.

### 2.1 `prefix_semantic_anchor` (weight 0.5) — MAIN SIGNAL

Computes the *teacher-forced next-token NLL* on a held-out piece of each training batch, using the v4 prefix produced by attending over the store.

```
# Setup
store has N=10 memories (seeded once per batch).
pick one memory m_tgt at random.
text_query, text_target = split(m_tgt.source_text, random_split_point)
# e.g. source_text="I collect vinyl records; latest is Kind of Blue by Miles Davis"
#      text_query="I collect vinyl records; latest is Kind of", text_target=" Blue by Miles Davis"

# Forward
ids_q, mask_q = tokenize(text_query)
ids_t, mask_t = tokenize(text_target)    # will be supervised
prefix = model.prepare_decode_context(ids_q, mask_q).prefix  # (1, L_mem, d_LLM)

# Concat (prefix, query_ids, target_ids), run backbone, shift, NLL on target positions
input_embeds, attn_mask = bridge.build_inputs(prefix, ids_full, mask_full, wte)
logits = backbone.model(inputs_embeds=input_embeds, attention_mask=attn_mask).logits
# Only supervise target positions
loss = cross_entropy(logits[:, :-1, :], ids_full[:, 1:], reduction over target slice)
```

This is the *only* loss that directly trains the prefix → LM pipeline. Everything else is an auxiliary regularizer. If this number goes down, the prefix is getting more informative about the target memory.

### 2.2 `bundle_axis_alignment` (weight 0.5)

Pulls each bundle's `canonical_axis` towards meaningful directions via *per-bundle contrastive targets*:

- **TimeBundle**: projection of a memory's `time_base` onto `canonical_axis` should monotonically track its `ts` (wall-clock write order). Implementation: across a batch of memories sorted by `ts`, require Spearman-like pairwise orders via a hinge loss `max(0, margin - (proj_new - proj_old))`.
- **TopicBundle**: for two memories with high *lexical* content overlap (Jaccard on content tokens), their `topic_base`s should be close in cosine; for two with low overlap, far. Implementation: triplet margin on `topic_base`.
- **ContextBundle**: for two writes within the same session (here = same epoch), their `ctx_base`s should be close; across-session, far. Implementation: triplet margin on `ctx_base`.

Sum the three sub-terms uniformly.

### 2.3 `cross_bundle_independence` (weight 0.2)

Prevents the three bundles from collapsing to copies of each other. For a batch of memories, compute the cross-bundle cosine matrix between `(time_fiber, topic_fiber, ctx_fiber)`. The term is the squared L2 distance between this matrix's off-diagonal entries and a target of 0.3 — low enough that bundles are distinct, nonzero so they can still correlate on meaningful signal.

### 2.4 `recon` (weight 1.0)

Round-trip reconstruction through `KakeyaRegistry`: for each memory in the batch's store, decode a sampled field and compute `||decode(encode(v)) − v||² / ||v||²` averaged across all fields touched. Asserts the compression pipeline doesn't silently drop information.

Note: kakeya skeletons are NOT gradient-trained (§1 above), but **the `base_to_field` pushforward maps ARE** — we make them `nn.Parameter` in v4.6. `recon` drives those maps to align kakeya's compressed subspace with the data distribution.

### 2.5 `write_policy` (weight 0.1)

Tiny regularizer on write statistics: penalize *trivial* memories (very short source_text, or where any `*_fiber` norm is unusually small). Keeps the bundles from collapsing to zero-vectors in degenerate cases. Weight is intentionally tiny — this is a sanity net, not a shaper.

## 3. Training data

Same **design** as v3.46's `train_v346.py` (§5.3 of `SPRINT_CLOSEOUT_v3.46.md`):
- 9 rotating sentences: 3 MUSIC + 3 SPACE + 6 GENERIC.
- Batch size 3 (rotating window over the 9).
- 60 steps default (matches v3.46 for a direct comparison).

This corpus is tiny on purpose: we are not trying to match GPT-2-on-WikiText here; we are trying to verify that the v4 stack can be moved off fresh-init by training. If the corpus works, scaling the corpus is a different experiment.

## 4. Trainer loop

```python
for step in range(n_steps):
    batch_texts = sample_batch(corpus, batch_size)
    # Seed store fresh each step — v4.5 write() is ~1 ms on GPU so this is cheap
    model.store = MemStore(cfg); model.kakeya = KakeyaRegistry(cfg)
    for t in batch_texts: model.write(t)

    # Compute losses
    losses = {}
    losses["prefix_semantic_anchor"] = loss_prefix_anchor(model, batch_texts)
    losses["bundle_axis_alignment"]  = loss_bundle_axis_alignment(model, batch_texts)
    losses["cross_bundle_independence"] = loss_cross_bundle_independence(model, batch_texts)
    losses["recon"]                  = loss_recon(model, batch_texts)
    losses["write_policy"]           = loss_write_policy(model, batch_texts)

    total = sum(cfg.loss_weights[k] * v for k, v in losses.items())

    opt.zero_grad()
    total.backward()
    # Clip grad norm on trainable params only
    nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
    opt.step()
```

Optimizer: AdamW, lr=3e-4, weight_decay=0.01. Same as v3.46's `Trainer`.

Checkpoint format (matches v3.46 style):
```python
{
    "state_dict": {name: p.detach().cpu() for name, p in m.named_parameters() if p.requires_grad},
    "cfg_snapshot": asdict(cfg),
    "provenance": "AgentMemory/v347-architecture-realign-b7fa",
    "steps": n_steps,
    "elapsed_s": float,
    "pre_probe": {...}, "post_probe": {...},
}
```

`pre_probe` / `post_probe` capture `abs().mean()` of key params (`cross_attn.lift_time[0].weight`, `bundle_topic.encoder.down_project[0].weight`, `bundle_time._axis_raw`) to check training moved something. Same pattern as v3.46 §5.6 "honest predictions".

## 5. Loader

`MemLLM4.load_trained_weights(path)`:
- Read checkpoint, assert `provenance` matches this branch's.
- Iterate `state_dict` items; assign into `self.named_parameters()` by name (strict=True on *trainable* subset — any mismatch raises, same bar as v3.46).
- Print `loaded=N skipped=M shape_errs=K` (same log format as v3.46 for muscle memory).

## 6. Merge gate

**Unchanged from PR #30 top:**

v4-trained numbers on `session_viability_v4.py`:
- `A_ams_prefix` at N=20: **strictly > 50%** (v3.46-trained).
- `C_ams_hybrid` at N=20: **strictly > 70%** (v3.46-trained).

If both gates clear on the same run, the branch is mergeable. If either fails, we investigate (scale the corpus? add another loss term? re-check alignment?) — but we do NOT ship decode-time logit-shaping patches.

---

## 7. What v4.6 explicitly does NOT do

- **Does not retune v3.46 hyperparameters.** `Cfg4.loss_weights` uses its own numbers.
- **Does not fine-tune the backbone.** `LLMBackbone4.model.parameters()` stays frozen.
- **Does not alter §6 invariants.** Trainer wraps the already-invariant-checked stack; if any invariant fails mid-training, `verify_consistency()` fires and training halts.
- **Does not ship a new decode-time patch.** If training doesn't close the gate, we debug training or architecture, not decode.
