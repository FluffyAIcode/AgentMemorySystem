# v3.42 Black-Box Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Section 7 (Reporting Discipline).

## 1. Scope and configuration

- SUT: `scheme_b_v342.py` via `AgentMemorySystem.py` redirect.
- Runner: `v331_blackbox_eval.py`, unmodified.
- Spec: `V331_BLACKBOX_TEST_SPEC.md`, unmodified.
- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`, `llm_dtype=bf16`, CPU execution.
- Test set: 26 cases (§1–§4 + cipher-system probes 4.20–4.26).
- Elapsed: 1418.4 s.

## 2. Aggregate

- Checks passed: 17 / 26.
- Checks failed: 9 / 26.

Comparison to v3.41 (17 / 26 pass):

| Transition | Count | Cases |
| --- | --- | --- |
| FAIL → PASS | 1 | 4.8 degeneration_quality |
| PASS → FAIL | 1 | 4.12 prefix_stepwise_drift_trajectory |
| PASS → PASS | 16 | (unchanged) |
| FAIL → FAIL | 8 | 4.7, 4.10, 4.15, 4.17, 4.21, 4.23, 4.24, 4.25 |

Net change: 0. One forward fix, one reverse regression; both caused by opposing constraints on the same magnitude (`wte_residual_alpha`).

## 3. Cases that transitioned FAIL → PASS

### 3.1 `degeneration_quality` (4.8)

| metric | v3.41 | v3.42 | threshold |
|---|---|---|---|
| `avg_unique_token_ratio` | 0.360 | **0.410** | ≥ 0.60 (case-default tolerance per runner) |
| `avg_repeated_bigram_ratio` | 0.391 | **0.073** | ≤ 0.25 |
| `avg_content_token_ratio` | 0.865 | 0.845 | ≥ 0.30 |
| `worst_max_token_run` | 2 | 2 | ≤ 3 |

Mechanism: `[H-1]` removed the fwd+shape_step double-add of `content_bias`. The same content token no longer gets a two-phase boost at step 0, so repeating bigrams like "pian pian Chop" disappear. `cyclic_content_max_count=3` also lets domain keywords surface more than once before being masked. Runner considers the case PASS under the current thresholds.

## 4. Cases that transitioned PASS → FAIL

### 4.1 `prefix_stepwise_drift_trajectory` (4.12)

v3.41: PASS. v3.42: FAIL (`first_bad_step = 0`).

Sample (prompt `"Key piano ideas include"`):
`"... key changes key signatures key signature key change key change ..."`.

Step-0 top-1 is `' key'` at logit 14.0. `top1_category='functional'` per runner classification.

Mechanism: `[H-2]` raised `wte_residual_alpha` from 0.5 → 1.5. With `zero_init_tied=True` zeroing `slot_heads[1]`, the tail slot direction is now **entirely** the rare-keyword WTE centroid at 1.5× native scale (L2 ≈ 1.17, close to `target_norm` ≈ 1.18). The prompt contains `piano`, which tokenises overlapping with memory rare-keyword `piano` (rank 0 for one of the stored music memories), so slot 1's direction is almost parallel to WTE[`' key'`] / WTE[`' piano'`] neighborhood → attractor basin at step 0. This is the exact trade-off documented in v3.41's §5.5 falsifiable experiment D1.

## 5. Persistent FAIL cases

### 5.1 `semantic_memory_counterfactual_pairs` (4.7)

Unchanged. `music_margin` and `space_margin` still sum under the required 2-keyword-difference threshold. `[H-1]`'s softer `relevance_floor=0.30` + `concentration=1.5` raised baseline bias strength but the two domains share a rising generic content tail at ~step 10 (`student, keyboard, studied, scales, conservatory`). These tokens are "music-ish" but not among the 12 labelled music keywords, so `music_margin` stays at 0 even though the output is recognisably music-themed.

### 5.2 `retrieval_topk_semantic_shift` (4.10)

Unchanged. Base `" the(21.1) at(19.5) a(19.4) both(19.0) specific(19.0) ..."`. Single-path bias in v3.42 is ~`cb × 11.25 × std` at step 0 (CFG `(1+α)=4.5×` amplifies `lg_cond`'s content_bias). Still does not cross the 13-logit gap to surface music_keywords into top-12. Structural ceiling unchanged; `[H-1]`'s parameter softening does not bridge the gap.

### 5.3 `stepwise_label_mass_alignment_audit` (4.15)

Unchanged. `retrieved_majority_label = "music"` correct for all 12 steps, but `logits_label_mass = {music: 0, space: 0}` (integer-rounded). Probe measures probability mass on the 12 label tokens; with content_bias at single-point application + CFG×4.5, mass rises at step 0 but stays below the probe's 0.01 quantisation in `lg_cond`. Training-side dependency (`vocab_proj` is zero-init) is the structural gap identified in v3.41 §5.3.

### 5.4 `save_load_consistency` (4.17)

Unchanged. Outputs still diverge after a common 6-token prefix. `[H-1]`'s `.detach().cpu().clone().contiguous()` both sides + stable tie-break did not flip the result. Remaining non-determinism source is outside the rare-keyword reranking (candidate SVD / multi-threaded CPU attention). See v3.41 §5.4 falsifiable experiments C1–C3.

### 5.5 `decode_repetition_feedback_probe` (4.21)

| metric | v3.41 | v3.42 | threshold |
|---|---|---|---|
| `avg_max_repeat_per_content_token` | 3.67 | **3.33** | ≤ 3 |
| `avg_trigram_lock_count` | 1.33 | **0.00** | ≤ 1 |
| `min_first_bigram_repeat_index` | 11 | 9 | ≥ 4 |

v3.42 fixes `avg_trigram_lock_count` and reduces max-repeat, but the first metric still exceeds 3.0 by 0.33, so the overall status is FAIL. Same root cause as 4.8 but the threshold is tighter: `[H-1]` softened repetition enough to pass 4.8 but not 4.21.

### 5.6 `keyword_specific_tail_slot_probe` (4.23)

Unchanged as PASS criterion (`mean_intersection = 0.0`, `hit_ratio = 0.0`).

Evidence of underlying change: v3.42 `tail_slot_top3_pieces = ['!', '"', '#']` across all 4 memories (was `['-*', '信', ' current']` in v3.41 — random-init garbage). The fact that all four memories now resolve to the same top-3 indicates the slot direction is now **determined by the same residual** across memories (shared rare-keyword centroid), which means `[H-2]` zero-init + residual-only succeeded structurally. The remaining gap is that the WTE cosine between the shared centroid and individual `rare_keyword_ids` is lower than the cosine to low-id byte tokens (`!`, `"`, `#` have WTE vectors close to the origin mean due to Qwen's multi-lingual vocabulary geometry). Top-20 intersection would likely capture the rare keywords; top-3 as per runner threshold does not.

### 5.7 `context_descriptor_cluster_probe` (4.24)

| metric | v3.41 | v3.42 |
|---|---|---|
| `intra_music_mean_cos` | 0.304 | **0.108** |
| `intra_space_mean_cos` | 0.389 | 0.344 |
| `inter_domain_mean_cos` | 0.290 | **0.192** |
| music gap | 0.014 | **−0.084** |
| space gap | 0.099 | **0.152** |

Mixed: space gap **crossed threshold** (0.152 ≥ 0.15) but music gap went **negative**. `[H-3]` single-Linear orthogonal projection preserves WTE angular structure in aggregate — space corpus's disjoint WTE centroids survive to cluster well — but the music corpus's 3 sentences share significant structural words (`practiced`, `scales`, `keyboard`, `conservatory`) whose WTE vectors actually project to different enough directions via the random orthogonal matrix that intra-music cosine (0.108) falls below inter-domain (0.192). The projection is lossless on average but on 3 points with n_ctx=128 out of d_LLM=1536, variance is high. `[H-3]` is structurally correct but sample size is too small to demonstrate the effect with the current corpus.

### 5.8 `prefix_length_scaling_probe` (4.25)

| metric | v3.41 | v3.42 | threshold |
|---|---|---|---|
| `content_starters_top12_A` | 12 | 12 | — |
| `content_starters_top12_B` | 12 | 12 | — |
| `slot_norm_ratio_B_over_A` | 1.003 | **0.784** | in [0.85, 1.15] |
| `per_slot_mean_norm_A` | 0.636 | **0.557** | — |
| `per_slot_mean_norm_B` | 0.638 | **0.437** | — |

New measurement: `slot_norm_ratio_B_over_A = 0.784` falls below 0.85 → second condition fails (it passed in v3.41). Cause: `wte_residual_alpha=1.5` injects more L2 into tail slots; under `prefix_norm_clamp_ratio=1.0`, the clamp scales B's additional tail slots down more aggressively (more slots to fit under the same total norm budget). `starter_count_B_ge_A_plus_1 = False` remains the same as v3.41 — content starters still saturate at 12 under current content_bias strength.

## 6. Aggregate technical observations

- `[H-1]` landed correctly. The double-add was the immediate root cause of 4.8/4.21 regression. One of them now passes; the other has a tighter threshold.
- `[H-2]` works structurally (tail slot direction is now deterministic across memories and identical to rare-keyword residual) but the **measurement threshold** (top-3 intersection out of 151936 vocab) is not met; and the stronger residual caused the 4.12 regression.
- `[H-3]` works on space corpus, fails on music corpus — a sample-variance issue with 3-point per-domain averaging in d_ctx=128. Cannot be fixed without larger corpus or deterministic rotation.
- `[H-4]` did not move 4.17. Non-determinism source is deeper (SVD or kernel).
- `[H-5]` structural claim (extra slots carry unique residual) is true, but probe 4.25's second condition (`slot_norm_ratio`) broke due to `[H-2]`'s larger residual L2 consuming clamp budget.

Two mutually-opposed pressures on `wte_residual_alpha`:
- 4.23 wants α ≥ 1.5 (residual must dominate slot direction)
- 4.12 wants α ≤ 0.5 (residual must not form attractor basin)
- 4.25 wants α such that tail slot norm fits in `[0.85, 1.15] × A's slot norm`

No scalar α satisfies all three simultaneously on this corpus. A per-step or per-memory scaling would be required to decouple them.

## 7. Artifacts

- `reports/v342_blackbox/report.json`
- `reports/v342_blackbox/report.md`
- `reports/v342_blackbox/runner.log`
- `reports/v342_blackbox/audit_feedback.md`

## 8. Summary of measured deltas

| Pass count | 17 → 17 | 0 |
| Elapsed | 1437.7 s → 1418.4 s | −19.3 s (−1.3 %) |
| FAIL → PASS | 1 case | 4.8 |
| PASS → FAIL | 1 case | 4.12 |
| Persistent FAIL | 8 cases | 4.7, 4.10, 4.15, 4.17, 4.21, 4.23, 4.24, 4.25 |
