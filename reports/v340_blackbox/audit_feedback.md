# v3.40 Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Section 7 (Reporting Discipline).

## 1. Run parameters

| Field | Value |
| --- | --- |
| SUT | `scheme_b_v340.py` (via `AgentMemorySystem.py` → `from scheme_b_v340 import *`) |
| Runner | `v331_blackbox_eval.py` (unchanged from v3.39 run) |
| Seed policy | case-local fixed seeds per Section 4 of the spec |
| Device | CPU |
| Backbone | `Qwen/Qwen2.5-1.5B-Instruct`, `bf16` |
| Elapsed | 1309.40 s |
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
| 4.12 | prefix_stepwise_drift_trajectory | false | fail | true | 44 |
| 4.13 | retrieval_generation_alignment_audit | true | pass | false | 45 |
| 4.14 | retrieval_prefix_decode_correlation_audit | true | pass | false | 46 |
| 4.15 | stepwise_label_mass_alignment_audit | false | fail | true | 48 |
| 4.16 | prompt_diversity_without_memory | true | pass | false | 37 |
| 4.17 | save_load_consistency | false | fail | true | 41 |
| 4.18 | training_cache_isolation | true | pass | false | 43 |
| 4.19 | cheating_heuristics | true | pass | false | 47 |
| 4.20 | rerank_stability_probe | true | pass | false | 49 |
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
| pass | 16 |
| fail | 10 |
| not_implemented | 0 |
| error | 0 |
| blocking_fail | 8 |

## 4. Delta vs. v3.39

| case | prior_passed | current_passed | prior_status | current_status |
| --- | --- | --- | --- | --- |
| rerank_stability_probe | false | true | fail | pass |
| retrieval_prefix_decode_correlation_audit | false | true | fail | pass |
| prefix_stepwise_drift_trajectory | true | false | pass | fail |

Cases not listed above did not change state between v3.39 and v3.40.

## 5. Per-failing-case evidence

### 4.6 semantic_memory_grounding
- Pass criterion: `music_margin > 0`, `space_margin > 0`, and at least one of `music_lift` or `space_lift` > 0.
- Measured: `music_margin = 0.1579`, `space_margin = 0.0`, `music_lift = 0.0865`, `space_lift = 0.1111`.
- Gap: `space_margin` is `0.0`, criterion requires strict `> 0`. The space-memory arm's space-keyword score equals its music-keyword score (`0.1111` vs `0.1111`), producing zero margin.

### 4.7 semantic_memory_counterfactual_pairs
- Pass criterion: for every prompt, music output favors music keywords and space output favors space keywords.
- Measured: 2 prompts; at least one prompt's music-memory output did not produce a positive music margin. Full per-prompt margins: `reports/v340_blackbox/report.json → results.semantic_memory_counterfactual_pairs.rows`.

### 4.10 retrieval_topk_semantic_shift
- Pass criterion: at least one prompt shows stronger domain alignment after prefix injection (higher domain keyword hit count or probability mass in top-k of final-step logits).
- Measured: neither of the two prompts' top-k exhibited increased domain keyword hit count or probability mass after prefix injection. Full top-k tables: `reports/v340_blackbox/report.json → results.retrieval_topk_semantic_shift.rows`.

### 4.12 prefix_stepwise_drift_trajectory
- Pass criterion: `first_bad_step` is absent or `>= 3`.
- Measured, row 0 (prompt `Key piano ideas include`): `first_bad_step = 0`, `decoded_output = "Key piano ideas include key ideas related to key concepts, key themes, key themes, key themes,"`.
- Measured, row 1 (prompt `Explain the topic clearly`): `first_bad_step = 4`, `decoded_output = "Explain the topic clearly without adding extra words. 《红楼梦》是清代作家曹雪芹创作"`.
- Gap: row 0 fails by `3 − 0 = 3` steps. Row 1 satisfies the criterion. Suite-level FAIL because any single row failure blocks.

### 4.15 stepwise_label_mass_alignment_audit
- Pass criterion: no row may accumulate retrieve-stage failure; no row may accumulate inject-stage failure.
- Measured: 2 rows, both reported accumulated inject-stage failures across the 12-step trace. Full stage counts per step: `reports/v340_blackbox/report.json → results.stepwise_label_mass_alignment_audit.rows`.

### 4.17 save_load_consistency
- Pass criterion: `output_a == output_b`.
- Measured:
  - `output_a = "The pianist piano piano donald duck ducks \`@don <EMAIL>\`⁈disjon⁢tion"`
  - `output_b = "The pianist piano piano music finger fingers hands class Chopin Chopins nocturn\n\nAdd links within paragraphs"`
- Longest common prefix ends at `"piano piano"` (4 tokens). Divergence begins at the next token (`donald` vs `music`).

### 4.22 functional_token_suppression_probe
- Pass criterion: `avg(content_starter_count_with_prefix − content_starter_count_no_prefix) >= 1.5` AND for ≥ 2 of 3 prompts `top_content_starter_logit >= top_functional_logit`.
- Measured: `avg_content_starter_delta = 0.3333`, `margin_non_negative_prompt_count = 0`.
- Gap: 1.167 below delta threshold; 2 below margin-count threshold.

### 4.23 keyword_specific_tail_slot_probe
- Pass criterion: `mean_intersection_size >= 1.0` AND `hit_ratio_at_least_one >= 0.5`.
- Measured over 4 memories: `mean_intersection_size = 0.0`, `hit_ratio_at_least_one = 0.0`.
- Gap: 1.0 below mean threshold; 0.5 below ratio threshold.

### 4.24 context_descriptor_cluster_probe
- Pass criterion: `intra_domain_mean_cos − inter_domain_mean_cos >= 0.15` for both domains.
- Measured: `intra_music = 0.9242`, `intra_space = 0.8623`, `inter = 0.8333`.
- Differentials: `music − inter = 0.0909`, `space − inter = 0.0290`.
- Gap: 0.0591 below threshold on music arm; 0.1210 below on space arm.

### 4.25 prefix_length_scaling_probe
- Pass criterion: `starters_B >= starters_A + 1` AND `slot_norm_ratio_B_over_A ∈ [0.85, 1.15]`.
- Measured: `L_mem_A = 8`, `L_mem_B = 16`, `starters_A = 3`, `starters_B = 2`, `per_slot_mean_norm_A = 0.6361`, `per_slot_mean_norm_B = 0.6362`, `slot_norm_ratio_B_over_A = 1.0002`.
- Gap: starter-count condition requires `B >= 4`; observed `2` (delta `−2`). Norm-ratio condition met.

## 6. Mechanism notes (non-normative, falsifiable)

### H1 — 4.17 save_load_consistency divergence persists

- Code element: `MemLLM.write` calls `MemoryContextEncoder.encode(content_sem)` (scheme_b_v340.py). `MemLLM.load_memory` calls `_refresh_rare_keyword_indices` which re-invokes `_compute_rare_keyword_ids` using `_compute_corpus_idf`. The `[F-1]` change set `update_stats=False` in `prepare_decode_context` and `generate`, eliminating one mutation source. `output_a` and `output_b` still diverge at token index 4.
- Observation: common prefix `"The pianist piano piano"` (4 tokens), then `output_a` continues with `"donald duck..."` and `output_b` with `"music finger..."`.
- Prediction: replacing `MemoryContextEncoder.encode` output with `torch.zeros_like(content_sem[:, :c.d_ctx])` in both write and load paths will make `output_a == output_b`. If divergence persists when encode is constant-zero, the hypothesis is falsified and the source of non-determinism is elsewhere (e.g., dict iteration order in `tree.store`, non-deterministic `torch.randperm` in `PrefixAligner.calibrate`).

### H2 — 4.22 functional_token_suppression_probe unchanged from v3.39

- Code element: `MemLLM.fwd` adds `fwd_function_suppression_scale * logits_std * step_scale_fn * dampen * fn_mask` when `guidance_active` is True (`[F-3]`).
- Observation: probe reports `avg_content_starter_delta = 0.3333` (identical to v3.39 measurement) and `margin_non_negative_prompt_count = 0`. `[F-3]` is wired but probe metric did not move.
- Prediction: printing `guidance_active` immediately before the penalty block for each of the 3 probe prompts will show `guidance_active == False` for the majority. If guidance is True on all 3 prompts and the scale still has no effect, the hypothesis is falsified and `[F-3]` is inactive because its scale is being normalized away downstream.

### H3 — 4.23 keyword_specific_tail_slot_probe unchanged from v3.39

- Code element: `EmbBridge.inject` passes `rare_keyword_wte_residual` to `self.tail_head(fiber_summary, wte_residuals=...)`, then applies `self.aligner(tail)` which performs `LayerNorm` followed by scalar rescaling to `_target_std`. `[F-4]` changed the residual injection scale from `target_std·√d` to `√d_LLM`.
- Observation: `mean_intersection_size = 0.0` over 4 memories, identical to v3.39. The residual is added pre-LN but LN is a non-linear projection that can null out the component aligned with rare-keyword-centroid if the slot_head output already dominates.
- Prediction: reading `tail_head(fiber_summary, wte_residual)` output *before* the `self.aligner` call and computing top-20 WTE cosine will produce `mean_intersection >= 1.0`. If the intersection is 0 even pre-aligner, the residual is being zeroed at the `tail_head.slot_heads[i][1]` LayerNorm (the per-slot output LN), in which case the hypothesis is falsified.

### H4 — 4.24 context_descriptor_cluster_probe differential shrank

- Code element: `MemoryContextEncoder` changed in `[F-5]` from 2-Linear without intermediate LN to 3-Linear with orthogonal init and intermediate LN; `encode()` now applies per-sample mean-centering before L2-normalize.
- Observation: v3.39 differentials were `music − inter = 0.1151, space − inter = 0.0627`. v3.40 differentials are `0.0909, 0.0290`. Both arms shifted closer to `inter`.
- Prediction: disabling the `h = h - h.mean(-1, keepdim=True)` line in `encode()` and re-running 4.24 will yield differentials approximately matching v3.39's `0.1151/0.0627`. If removing mean-centering does not restore the differentials, the hypothesis is falsified and the cause is the orthogonal-init weight geometry rather than mean-centering.

### H5 — 4.25 prefix_length_scaling_probe starter count regression persists

- Code element: `Cfg.effective_tail_slots()` returns `2` at `L_mem=8` and `6` at `L_mem=16` (verified by direct call before audit). Slot indices `1..5` in the L_mem=16 model receive rare-keyword residuals via `_compute_rare_keyword_wte_residual` with distinct `kw_rank = slot_idx − 1`.
- Observation: top-12 content-starter count is `3` at L_mem=8 and `2` at L_mem=16 with `slot_norm_ratio = 1.0002`. Doubling L_mem removed one content-starter from the top-12 rather than adding one.
- Prediction: setting `use_wte_residual_tail=False` at L_mem=16 and re-running 4.25 will yield `starters_B >= starters_A`. If the regression persists, the hypothesis is falsified and the extra body slots (not the tail residuals) are drowning out the content-starter signal.

### H6 — 4.12 prefix_stepwise_drift_trajectory row 0 regressed from v3.39

- Code element: `[F-3]` `MemLLM.fwd` function-suppression path is active from step 0 onward when `guidance_active` is True. The penalty magnitude at step 0 is `fwd_function_suppression_scale * logits_std * 1.0 * dampen = 5.0 * logits_std * 0.25`.
- Observation: `first_bad_step = 0` on prompt `Key piano ideas include`. The decoded output begins with `"key ideas related to key concepts, key themes, key themes, key themes,"` — content-starter-dominated but trapped in a `"key"` repetition pattern. The degeneration detector fires at step 8 onward per `[D-4]` but step 0's first token is what the probe measures.
- Prediction: lowering `fwd_function_suppression_scale` from 5.0 to 2.5 (half) will shift the step-0 winner away from a function word but also reduce the magnitude of the `"key"` attractor, yielding `first_bad_step >= 3`. If lowering the scale does not move `first_bad_step` past 0, the hypothesis is falsified and `[F-3]` is not causally responsible.

## 7. Artifact links

- `reports/v340_blackbox/report.json`
- `reports/v340_blackbox/report.md`
- `reports/v340_blackbox/runner.log`
- Source under test: `scheme_b_v340.py`, `AgentMemorySystem.py` at commit `7429fcc`
- Prior version for delta: `reports/v339_blackbox/report.json` (branch `AgentMemory/v339-blackbox-audit-7e97`)
