# v3.39 Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Section 7 (Reporting Discipline).

## 1. Run parameters

| Field | Value |
| --- | --- |
| SUT | `scheme_b_v339.py` (via `AgentMemorySystem.py` → `from scheme_b_v339 import *`) |
| Runner | `v331_blackbox_eval.py` (unchanged from v3.38 run except for 4.26 probe body; 4.20 – 4.25 and 4.1 – 4.19 identical) |
| Seed policy | case-local fixed seeds per Section 4 of the spec |
| Device | CPU |
| Backbone | `Qwen/Qwen2.5-1.5B-Instruct`, `bf16` |
| Elapsed | 1268.88 s |
| Runner exit code | 1 |

## 2. Per-case result table

| # | case | passed | status | blocking | seed |
| --- | --- | --- | --- | --- | --- |
| 4.1 | leaf_capacity_stability | true | pass | false | 0..7 |
| 4.2 | degenerate_direction_boundary | true | pass | false | 17 |
| 4.3 | metric_trainability | true | pass | false | 23 |
| 4.4 | no_grad_generation | true | pass | false | 29 |
| 4.5 | counterfactual_memory_influence | true | pass | false | 31 |
| 4.6 | semantic_memory_grounding | false | fail | true | 33 |
| 4.7 | semantic_memory_counterfactual_pairs | false | fail | true | 35 |
| 4.8 | degeneration_quality | true | pass | false | 36 |
| 4.9 | prefix_logit_drift_audit | true | pass | false | 38 |
| 4.10 | retrieval_topk_semantic_shift | false | fail | true | 39 |
| 4.11 | repetition_segment_audit | true | pass | false | 40 |
| 4.12 | prefix_stepwise_drift_trajectory | true | pass | false | 44 |
| 4.13 | retrieval_generation_alignment_audit | true | pass | false | 45 |
| 4.14 | retrieval_prefix_decode_correlation_audit | false | fail | true | 46 |
| 4.15 | stepwise_label_mass_alignment_audit | false | fail | true | 48 |
| 4.16 | prompt_diversity_without_memory | true | pass | false | 37 |
| 4.17 | save_load_consistency | false | fail | true | 41 |
| 4.18 | training_cache_isolation | true | pass | false | 43 |
| 4.19 | cheating_heuristics | true | pass | false | 47 |
| 4.20 | rerank_stability_probe | false | fail | true | 49 |
| 4.21 | decode_repetition_feedback_probe | true | pass | false | 50 |
| 4.22 | functional_token_suppression_probe | false | fail | true | 51 |
| 4.23 | keyword_specific_tail_slot_probe | false | fail | false | 52 |
| 4.24 | context_descriptor_cluster_probe | false | fail | false | 53 |
| 4.25 | prefix_length_scaling_probe | false | fail | true | 54 |
| 4.26 | mixture_distribution_gate_probe | true | pass | false | 55 |

## 3. Count summary

| Metric | Count |
| --- | --- |
| total | 26 |
| pass | 15 |
| fail | 11 |
| not_implemented | 0 |
| error | 0 |
| blocking_fail | 9 |

## 4. Delta vs. v3.38

| case | prior_passed | current_passed | prior_status | current_status |
| --- | --- | --- | --- | --- |
| retrieval_prefix_decode_correlation_audit | true | false | pass | fail |
| save_load_consistency | true | false | pass | fail |
| context_descriptor_cluster_probe | false | false | not_implemented | fail |
| mixture_distribution_gate_probe | false | true | not_implemented | pass |

Cases not listed above did not change state between v3.38 and v3.39.

## 5. Per-failing-case evidence

### 4.6 semantic_memory_grounding
- Pass criterion (Section 4.6): `music_margin > 0` and `space_margin > 0` and at least one of `music_lift`, `space_lift` > 0.
- Measured: `music_margin = 0.15`, `space_margin = -0.0833`, `music_lift = 0.0833`, `space_lift = 0.0`.
- Gap: `space_margin` is negative; criterion requires `> 0`. Failed on the space arm.

### 4.7 semantic_memory_counterfactual_pairs
- Pass criterion (Section 4.7): for every prompt, music output favors music keywords and space output favors space keywords.
- Measured: 2 prompts evaluated; payload's `rows` contains per-prompt margins; one prompt's space margin did not exceed its music margin in the space-memory condition. Runner-reported `passed=false`.
- Full per-prompt margins: `reports/v339_blackbox/report.json → results.semantic_memory_counterfactual_pairs.rows`.

### 4.10 retrieval_topk_semantic_shift
- Pass criterion (Section 4.10): at least one prompt shows stronger domain alignment after prefix injection.
- Measured: neither of the two prompts' top-k exhibited increased domain keyword hit count or probability mass after prefix injection.
- Full top-k tables: `reports/v339_blackbox/report.json → results.retrieval_topk_semantic_shift.rows`.

### 4.14 retrieval_prefix_decode_correlation_audit
- Pass criterion (Section 4.17 of spec): `corr(retrieval_strength, bad_decode_score) <= 0.2`.
- Measured: `corr_retrieval_bad = 0.278259`.
- Gap: 0.0782 above threshold.

### 4.15 stepwise_label_mass_alignment_audit
- Pass criterion (Section 4.19 of spec): no row may accumulate retrieve-stage failure; no row may accumulate inject-stage failure.
- Measured: 2 rows, both reported accumulated inject-stage failures across the 12-step trace.
- Full stage counts per step: `reports/v339_blackbox/report.json → results.stepwise_label_mass_alignment_audit.rows`.

### 4.17 save_load_consistency
- Pass criterion (Section 4.13 of spec): both outputs are identical.
- Measured:
  - `output_a` = `"The pianist piano piano keys white feet happy singing music yellow purple green plant animal dog cat vehicle cool fast"`
  - `output_b` = `"The pianist piano piano keys white feet happy singing music yellow purple green plant grass red blue pink orange teal"`
- Longest common prefix ends at `"...plant"`. Divergence begins at the next token (`animal` vs `grass`) and persists.

### 4.20 rerank_stability_probe
- Pass criterion (Section 4.20 of spec): `jaccard(top5_a, top5_b) >= 0.6` for both pairs and `spearman(shared_ranks) >= 0.5` for at least one of the two pairs.
- Measured, pair `music_P1`: `top5_a=[1]`, `top5_b=[1]`, jaccard = 1.000, spearman = 0.000.
- Measured, pair `space_P2`: `top5_a=[5]`, `top5_b=[5]`, jaccard = 1.000, spearman = 0.000.
- Gap: jaccard condition met for both pairs. Spearman condition requires ≥ 0.5; achieved 0.0 on both pairs. The shared-rank set has size 1 on both pairs, on which Spearman is undefined; the runner emits 0.0.

### 4.22 functional_token_suppression_probe
- Pass criterion (Section 4.22 of spec): `avg(content_starter_count_with_prefix − content_starter_count_no_prefix) >= 1.5` and for ≥ 2 of 3 prompts `top_content_starter_logit >= top_functional_logit`.
- Measured: `avg_content_starter_delta = 0.3333`, `margin_non_negative_prompt_count = 0`.
- Gap: 1.167 below the delta threshold; 2 below the margin-count threshold.

### 4.23 keyword_specific_tail_slot_probe
- Pass criterion (Section 4.23 of spec): `mean_intersection_size >= 1.0` and `hit_ratio_at_least_one >= 0.5`.
- Measured over 4 memories: `mean_intersection_size = 0.0`, `hit_ratio_at_least_one = 0.0`.
- Gap: 1.0 below the mean threshold; 0.5 below the ratio threshold.

### 4.24 context_descriptor_cluster_probe
- Pass criterion (Section 4.24 of spec): `intra_domain_mean_cos - inter_domain_mean_cos >= 0.15` for both domains.
- Measured: `intra_music = 0.8974`, `intra_space = 0.8450`, `inter = 0.7823`.
- Differentials: `music - inter = 0.1151`, `space - inter = 0.0627`.
- Gap: 0.0349 below threshold on music arm; 0.0873 below on space arm.

### 4.25 prefix_length_scaling_probe
- Pass criterion (Section 4.25 of spec): `starters_B >= starters_A + 1` and slot-norm ratio B/A ∈ [0.85, 1.15].
- Measured: `L_mem_A = 8`, `L_mem_B = 16`, `starters_A = 3`, `starters_B = 2`, `per_slot_mean_norm_A = 0.6366`, `per_slot_mean_norm_B = 0.6361`, `slot_norm_ratio_B_over_A = 0.9993`.
- Gap: starter-count condition requires B ≥ 4; observed 2 (delta −2). Norm-ratio condition met.

## 6. Mechanism notes (non-normative, falsifiable)

This section records hypotheses linking failures to named code elements. Each hypothesis states a testable prediction. Hypotheses are not conclusions.

### H1 — 4.17 save_load_consistency divergence

- Code element: `MemLLM.write` calls `MemoryContextEncoder.encode(content_sem)` (scheme_b_v339.py, `write` body) and `MemLLM.load_memory` calls `_refresh_rare_keyword_indices` which depends on memory contents computed at write time.
- Observation: `output_a` and `output_b` share a prefix of 19 tokens and diverge at step 20 under greedy decoding with identical seeds.
- Prediction: disabling `use_memory_context_encoder` on both model_A and model_B (and therefore bypassing the `MemoryContextEncoder` forward pass) will cause 4.17 to return to pass. If it does not, the hypothesis is falsified.

### H2 — 4.14 retrieval_prefix_decode_correlation_audit correlation increase

- Code element: `MemLLM.shape_step_logits` decode-time functional suppression block (`[E-3]`) and `EmbBridge.inject` WTE residual addition to tail slot[1] (`[E-4]`).
- Observation: `corr(retrieval_strength, bad_decode_score)` increased from 0.19 (v3.38) to 0.28 (v3.39) while `retrieval_strength` values were on the same scale in both runs.
- Prediction: setting `use_decode_functional_suppression=False` and `use_wte_residual_tail=False` simultaneously will decrease the correlation back below 0.20. If the correlation remains above 0.20 with both flags off, the hypothesis is falsified.

### H3 — 4.20 rerank_stability_probe Spearman = 0

- Code element: `AMM._apply_min_keep_for_rerank` with `upstream_gate_min_keep_for_rerank = 3` and `strict_overlap_min_keep_for_rerank = 3`.
- Observation: `n_candidates_for_rerank` was 1 on both prompts of both near-paraphrase pairs, not ≥ 3.
- Prediction: an additional trace will show the strict-overlap gate and the subsequent hard_mask + score_keep + coherence_mask + bidi_gap + mean_center filters collapse the candidate set after `_apply_min_keep_for_rerank` has returned. If the candidate count stays ≥ 3 through all five filters with no config changes, the hypothesis is falsified.

### H4 — 4.23 keyword_specific_tail_slot_probe mean_intersection = 0

- Code element: `EmbBridge.inject` applies `PrefixAligner` (LayerNorm + learned scale) to the tail tensor after the WTE residual is added, producing `bridge._last_tail_slots` = post-aligner outputs.
- Observation: probe reads `bridge._last_tail_slots[0, -1]` and computes top-3 WTE cosine, obtaining 0 overlap with the memory's rare_keyword_ids.
- Prediction: reading the pre-aligner tensor — i.e., `tail_head(fiber, wte_residuals=rare_residual)` output directly, before `self.aligner(tail)` — will yield top-3 WTE cosine whose intersection with `rare_keyword_ids` is ≥ 1 for at least 50 % of the 4 memories. If the pre-aligner tensor also has 0 intersection, the hypothesis is falsified and the residual is not reaching slot[1] at all.

### H5 — 4.24 context_descriptor_cluster_probe differential < 0.15

- Code element: `MemoryContextEncoder` (random-initialized, untrained) projects `content_sem` (d_LLM = 1536) to `d_ctx = 128` and L2-normalizes.
- Observation: untrained `MemoryContextEncoder.encode` produces descriptors with intra_music cosine 0.8974 and inter cosine 0.7823, differential 0.1151 < threshold 0.15.
- Prediction: training `MemoryContextEncoder` on a supervised domain-classification loss over the corpus will increase the intra-inter differential above 0.15. If after 50 steps of such training the differential remains below 0.15, the hypothesis is falsified.

### H6 — 4.25 prefix_length_scaling_probe starter count regression on larger L_mem

- Code element: `Cfg.effective_tail_slots()` returns `max(content_tail_slots, L_mem // tail_L_mem_ratio)`, giving tail_slots=4 at L_mem=16 vs tail_slots=2 at L_mem=8. Additional tail slots are produced by untrained `tail_head.slot_heads[i]` for i ≥ 2.
- Observation: doubling L_mem decreased content-starter count in top-12 from 3 to 2 at matched per-slot norm.
- Prediction: setting `scale_tail_with_L_mem=False` (so L_mem=16 uses the same 2 tail slots as L_mem=8 with extra body slots only) will yield `starters_B >= starters_A`. If it still regresses, the hypothesis is falsified and the regression is caused by the body-slot expansion, not the extra tail slots.

## 7. Artifact links

- `reports/v339_blackbox/report.json`
- `reports/v339_blackbox/report.md`
- `reports/v339_blackbox/runner.log`
- Source under test: `scheme_b_v339.py`, `AgentMemorySystem.py` at commit `0c296bd`
- Audit artifacts commit: `a21fbdb`
