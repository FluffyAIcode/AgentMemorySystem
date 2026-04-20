# v3.41 Black-Box Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Section 7 (Reporting Discipline).

## 1. Scope and configuration

- SUT: `scheme_b_v341.py` via `AgentMemorySystem.py` redirect.
- Runner: `v331_blackbox_eval.py`, unmodified.
- Spec: `V331_BLACKBOX_TEST_SPEC.md`, unmodified.
- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`, `llm_dtype=bf16`, CPU execution.
- Test set: 26 cases (§1–§4 + cipher-system probes 4.20–4.26).
- Elapsed: 1437.7 s.

## 2. Aggregate

- Checks passed: 17 / 26.
- Checks failed: 9 / 26.
- Constraints: present in report; not used as gating criterion per spec.

Comparison to v3.40 (16 / 26 pass):

| Transition | Count | Cases |
| --- | --- | --- |
| FAIL → PASS | 3 | 4.6 semantic_memory_grounding; 4.12 prefix_stepwise_drift_trajectory; 4.22 functional_token_suppression_probe |
| PASS → FAIL | 2 | 4.8 degeneration_quality; 4.21 decode_repetition_feedback_probe |
| PASS → PASS | 14 | (unchanged) |
| FAIL → FAIL | 7 | 4.7, 4.10, 4.15, 4.17, 4.23, 4.24, 4.25 |

Net change: +1 pass. Two new regressions are co-localised in the decode-time repetition-control subsystem and share a single mechanism (section 4.2).

## 3. Cases that transitioned FAIL → PASS

### 3.1 `semantic_memory_grounding` (4.6)

v3.40: FAIL. v3.41: PASS. Runner condition set met; no residual failure modes reported.
Plausible mechanism: `[G-2]` now attaches `content_bias` / `suppression_bias` and `guidance_active=True` to `prefix_cond` in the `return_extra=True` branch used by `prepare_decode_context`. The runner computes grounding by sampling under the retrieved-memory prefix; with bias now actually reaching `fwd`, the content-starter top-k set was pulled closer to the stored tokens enough to clear the case-specific threshold.

### 3.2 `prefix_stepwise_drift_trajectory` (4.12)

v3.40: FAIL (`first_bad_step=0`). v3.41: PASS.
Plausible mechanism: `[G-3]` removed the `√d_LLM` scaling of the rare-keyword WTE residual and moved the blend to a post-aligner additive at α=0.5 (native WTE element std ≈ 0.04 × α × sqrt(1536) ≈ 0.78 L2 contribution per slot). The attractor basin that in v3.40 forced a single keyword (e.g. "key") to dominate step 0 no longer exists at this magnitude.

### 3.3 `functional_token_suppression_probe` (4.22)

v3.40: FAIL (`avg_starter_delta_ge_1_5=False`, `margin_non_negative_ge_2_of_3=False`). v3.41: PASS (both True).
Plausible mechanism: `[G-2]` + `[G-6]` together. Guidance now propagates through `_get_prefix(return_extra=True)`, and the function-suppression term in `fwd` uses `fwd_function_suppression_scale=5.0` with `apply_dampen=False`, so it is not throttled by `fwd_path_bias_dampen=0.25`. The probe sees a net positive `top_star − top_func` margin on ≥ 2 of 3 prompts.

## 4. Cases that transitioned PASS → FAIL

### 4.1 `degeneration_quality` (4.8)

Metrics (v3.40 → v3.41):

| metric | v3.40 | v3.41 | runner threshold |
| --- | --- | --- | --- |
| `avg_unique_token_ratio` | 0.905 | 0.360 | ≥ 0.60 (case-default) |
| `avg_repeated_bigram_ratio` | 0.000 | 0.391 | ≤ 0.25 (case-default) |
| `avg_content_token_ratio` | 0.759 | 0.865 | ≥ 0.30 |
| `worst_max_token_run` | 2 | 2 | ≤ 3 |
| `avg_newline_ratio` | 0.009 | 0.000 | ≤ 0.10 |

Sample output under prompt `"The pianist"`:
`"The pianist pian pian Chop practiced midnight nocturnal night midnight practiced Chop nocturnal..."`.

### 4.2 `decode_repetition_feedback_probe` (4.21)

Metrics (v3.40 → v3.41):

| metric | v3.40 | v3.41 | threshold |
| --- | --- | --- | --- |
| `avg_max_repeat_per_content_token` | 2.00 | 3.67 | ≤ 3 |
| `avg_trigram_lock_count` | 0.00 | 1.33 | ≤ 1 |
| `min_first_bigram_repeat_index` | None | 11 | ≥ 4 |

Sample output under prompt `"The telescope"`:
`"The telescope telescope telescope stars neb spectral signatures captured distant stars captured signatures neb spectral distant power capture..."`.

### 4.3 Shared failure chain for 4.8 and 4.21

Both regressions exhibit the same pattern: content tokens and content-starter bigrams repeat with high frequency after step ~10.

Mechanism chain:
1. `[G-2]` attaches `content_bias` + `suppression_bias` to `prefix_cond` in the non-training path.
2. `fwd` now applies `content_bias × fwd_path_bias_dampen (0.25) × content_bias_scale (6.0) × logits_std × content_bias_std_multiplier (1.5)` on the last logit every step — this term was inactive in v3.40 because guidance was False.
3. `shape_step_logits` re-applies essentially the same additive term at the non-dampened scale (`cb_effective × logits_std × 1.5 × 6.0 × step_scale`). The decay mitigation (`content_bias_history_decay_rate=0.5`) is per-generated-content-token but only after a token actually appears in `generated_content_counts`; between the fwd-path add and the shape-step add, the same content token's effective additive boost is applied twice before first-token generation.
4. The double-add drives the same content token (from a small high-IDF set — "piano", "Chop", "noct", "midnight", "practiced", …) to dominate top-k on every early step, producing the observed bigram/trigram locks.

Falsifiable check: set `fwd_path_bias_dampen=0.0` (disables fwd-path content_bias add) and rerun 4.8/4.21; expect both to return to v3.40 metrics without breaking 4.22 (since `use_fwd_function_suppression` scale is now decoupled per `[G-6]`). If that check fails to restore 4.8/4.21 behaviour, the mechanism is elsewhere (e.g. `[G-3]` residual amplifying content-starter logits at post-aligner).

## 5. Cases that remained FAIL

### 5.1 `semantic_memory_counterfactual_pairs` (4.7)

Behaviour unchanged relative to v3.40. The probe compares the content-starter top-12 under music-prefix vs. space-prefix and requires ≥ 2 keyword-set differences per prompt. v3.41 output under prompt `"Describe the most important details a student should notice."` shows repetition of memory tokens (`student student expressive keyboard studied scales conservatory …`). The `fwd`-path double-add described in 4.3 also applies here; the memory does dominate, but the surface form is degenerate rather than discriminative between the two prefixes.

### 5.2 `retrieval_topk_semantic_shift` (4.10)

Unchanged. Prompt `"A strong explanation should mention"` top-12 without prefix is `[the, at, a, both, ...]` (stop-words at logits 19–21). With memory prefix, top-12 content-starter count is below the required shift threshold. This is the ceiling previously identified: `content_bias × fwd_path_bias_dampen=0.25` does not reverse an 8–10 logit gap on a generic English explanation prompt. `[G-6]` added function-suppression penalty (≈ `5.0 × logits_std × 1.5 × step_scale`) which moves some function tokens down but the runner measures the top-12 composition, where stop-words still dominate because the suppression applies to the logits tensor but the runner samples from position 0 (raw logits under prefix_cond, no shape_step_logits applied).

### 5.3 `stepwise_label_mass_alignment_audit` (4.15)

Unchanged. Per-step retrieved-majority label is correct for 12 / 12 steps but `logits_label_mass` for label tokens remains below threshold. This is the training-only component identified in v3.40 feedback: without trained `ContentSemanticTailHead` weights on label tokens, the decode-time bias cannot lift label-specific logit mass above the generic content mass.

### 5.4 `save_load_consistency` (4.17)

Unchanged. Sampled outputs diverge after a common prefix of 6 content tokens:

```
A: "... hours noct Chop pian class piano piano"
B: "... noct hours Chop pract act piano piano"
```

`[G-1]` removed the `mean_centering` and forced fp32 in `MemoryContextEncoder`, and added `.detach().contiguous().cpu()` in `save_memory`. The divergence persists, which indicates the non-determinism source is outside `MemoryContextEncoder`. Two remaining candidates identified in source:
- `PrefixAligner.calibrate` uses `torch.randperm` against the wte sample; this runs once at `load()` but is invoked in the same seeded order in both paths, so should not differ across the save→load cycle of the same process.
- `torch.multinomial` in `generate`: the probe seeds `torch.manual_seed(12345)` around both generates; but the `load_memory` path triggers `_refresh_rare_keyword_indices`, which mutates `mem.rare_keyword_ids` based on the re-computed IDF. If the IDF dictionary iteration order after reload differs from before save (Python `dict` order on int keys is insertion order), and `_compute_rare_keyword_ids` ranks by `-corpus_idf.get(t, floor)`, equal-IDF ties break differently. Save→load re-inserts mems in the same order (`store.items()` in save, then `data['store'].items()` in load), so the Python iteration order should be preserved, but internal `_ins` → `_split` tree rebalancing can reorder tree leaves and thus `store` iteration order is controlled by dict insertion order only.

Falsifiable check: call `m._refresh_rare_keyword_indices()` on instance A immediately before saving and compare `m.amm.tree.store[mid].rare_keyword_ids` pre-save vs. post-load for every mid. If any list differs, the IDF-tie-break ordering hypothesis holds. If all lists are identical, the divergence is downstream.

### 5.5 `keyword_specific_tail_slot_probe` (4.23)

Unchanged. `mean_intersection_size=0.0`, `hit_ratio_at_least_one=0.0` in both versions.

v3.41 evidence from the per-memory table:
- mid 0 `rare_keyword_ids=[43564, 32333]` → `[' practiced', ' midnight']`
- `tail_slot_top3_pieces=['-*', '信', ' current']`

The tail slot's top-3 vocab under `wte_normed @ tail_slot` does not contain either of the two rare keywords. `[G-3]` changed the residual scale (native WTE, α=0.5 post-aligner) so slot direction should include a ~0.78-L2 contribution from the rare-keyword WTE centroid. This contribution is present but the untrained `tail_head[1]` projection (random init, std=0.02) outputs a vector with L2 ≈ 0.03 × √1536 ≈ 1.2 after aligner, so residual dominates direction ratio roughly 0.65 / 1.2 ≈ 0.54. In practice the post-aligner add-sign test the runner performs is stricter: it requires the rare-keyword tokens specifically to appear in the top-3 of the entire vocab (151936 tokens), and a 0.54-direction-contribution does not push those specific tokens into top-3 when millions of WTE vectors have comparable cosine to a centroid.

Falsifiable check: compute `cos(tail_slot[1], wte[rare_keyword_id])` explicitly and compare to the top-3 cosines the slot aligns with. If the rare-keyword cosine is in the top 20 but not top 3, the magnitude hypothesis holds; raising α to 1.5–2.0 should flip to PASS at the cost of 4.12 (attractor basin) regressing.

### 5.6 `context_descriptor_cluster_probe` (4.24)

Metrics (v3.40 → v3.41):

| metric | v3.40 | v3.41 | threshold |
| --- | --- | --- | --- |
| `intra_music_mean_cos` | 0.924 | 0.304 | — |
| `intra_space_mean_cos` | 0.862 | 0.389 | — |
| `inter_domain_mean_cos` | 0.833 | 0.290 | — |
| implied gap (music vs. inter) | 0.091 | 0.014 | ≥ 0.15 |
| implied gap (space vs. inter) | 0.029 | 0.099 | ≥ 0.15 |

Structural change: v3.40 had near-collinear descriptors (all intra/inter ≈ 0.85) — large mean-collapse from the hidden-state-based encoder. `[G-4]` replaced the encoder input with WTE strict-starter centroids; the output now spreads over the unit sphere (0.30–0.39 intra, 0.29 inter). The per-domain gaps are 0.014 and 0.099 — both below 0.15. The expected structural gap from WTE alone is larger in principle (different-domain strict starters have near-orthogonal mean directions), but the untrained orthogonal MLP maps the centroid non-linearly through two `LN → SiLU` blocks that mix the signal with random noise.

Falsifiable check: bypass the MLP (return `F.normalize(wte_centroid_d_llm_subspace)` directly as the descriptor) and recompute gaps. If gap exceeds 0.15, the MLP is the noise source; if not, the strict-starter overlap across corpora is itself too high.

### 5.7 `prefix_length_scaling_probe` (4.25)

Metrics (v3.40 → v3.41):

| metric | v3.40 | v3.41 | threshold |
| --- | --- | --- | --- |
| `content_starters_top12_A` (L_mem=8) | 3 | 12 | — |
| `content_starters_top12_B` (L_mem=16) | 2 | 12 | — |
| `starter_count_B_ge_A_plus_1` | False | False | True required |
| `slot_norm_ratio_B_over_A` | 1.000 | 1.003 | in [0.85, 1.15] |

Net effect: v3.41 produces saturated content-starter top-12 at both L_mem values (12/12). The probe requires `B ≥ A + 1`, i.e. the larger L_mem must produce strictly more starters. With both saturated at 12, the inequality is false. Structurally the memory signal is now saturated at the L_mem=8 baseline so the L_mem=16 extra slots add no measurable starter count. The tied-extra heads (`[G-5]`) share weights with slot 1 and are differentiated only through per-slot rare-keyword residuals; with only 4 memories in the probe's corpus and `keyword_tail_top_k=8`, ranks 0–7 are exhausted but slots 2–7 at L_mem=16 all receive centroids of the same 4-memory rare-keyword set, so they contribute near-identical signal.

## 6. Constraints

The runner emits 4 constraint entries (non-gating per spec). They all report `PASS_or_not_implemented` except where the feature is fully exposed; no structural regression is reflected there.

## 7. Artifacts

- `reports/v341_blackbox/report.json` — machine-readable.
- `reports/v341_blackbox/report.md` — runner-authored Markdown.
- `reports/v341_blackbox/runner.log` — full stdout/stderr of the run.
- `reports/v341_blackbox/audit_feedback.md` — this file.

## 8. Summary of measured deltas

| Pass count | 16 → 17 | +1 |
| Elapsed | 1309.4 s → 1437.7 s | +128.3 s (+9.8 %) |
| FAIL → PASS | 3 cases | 4.6, 4.12, 4.22 |
| PASS → FAIL | 2 cases | 4.8, 4.21 |
| Persistent FAIL | 7 cases | 4.7, 4.10, 4.15, 4.17, 4.23, 4.24, 4.25 |
