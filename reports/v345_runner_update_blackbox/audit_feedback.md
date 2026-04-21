# v3.45-Runner-Update Black-Box Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Sections 7 and 7.7.

## 1. Run parameters

- SUT version: `scheme_b_v344.py` (unchanged from v3.44-Trained audit)
- Runner version: `v331_blackbox_eval.py` updated per SPEC Section 4.23 / 4.24 / 4.25 v3.45 correction + Section 4-meta.1 axis-coverage emission
- Weights: `ckpt/v344_trained.pt` (60-step Trainer checkpoint from v3.44-Trained run)
- Env: `AMS_TRAINED_WEIGHTS=ckpt/v344_trained.pt`, `AMS_DETERMINISTIC=1`
- Device: CPU (single-threaded under `torch.set_num_threads(1)`)
- Seed policy: per-case seeds as defined in SPEC Section 4
- Elapsed: 1476.3 s
- Exit code: 0

## 2. Axis coverage (SPEC 4-meta.1, v3.45+)

```json
{
  "axis_a_compression":   { "ratio": 8.97, "threshold": 10.0, "passed": false },
  "axis_b_injection_cost": { "per_step_floats": 164224, "depends_on_N": false, "passed": true },
  "axis_c_fidelity":      { "passed_over_total": "5/11", "threshold_K": 9, "passed": false },
  "axis_d_stability":     { "passed_over_total": "2/3",  "threshold_all_pass": true, "passed": false },
  "channel_passes_all_axes": false
}
```

Axis A is reported at `8.97` which is below the threshold `10.0`. This value is computed by the runner assuming `stored_floats_per_mem = d_M + d_F + d_M + d_ctx + d_LLM = 1712` (the `d_LLM=1536` comes from the `semantic_emb` field on `MemEntry`). A follow-up can refine the axis-A formula to exclude `semantic_emb` which is a cached hidden_mean, not part of the compressed code; under that definition stored_floats = 176 and ratio = 87. The current value is the literal sum of all optional fields and is reported as-is per Section 7.3.

Axis B passes: per-decode-step floats = `L_mem × d_LLM + V = 8 × 1536 + 151936 = 164224`, independent of `N`.

Axis C fails: 5 of 11 fidelity-dependent cases pass (threshold 9, = `ceil(0.75 × 11)`).

Axis D fails: 2 of 3 stability cases pass; `save_load_consistency` diverges after shared prefix length 1 (token `"piano"`).

## 3. Per-case result table

| case | passed | status | blocking | notes |
|---|---|---|---|---|
| 4.1 leaf_capacity_stability | true | pass | — | — |
| 4.2 degenerate_direction_boundary | true | pass | — | — |
| 4.3 metric_trainability | true | pass | — | — |
| 4.4 no_grad_generation | true | pass | — | — |
| 4.5 counterfactual_memory_influence | true | pass | — | — |
| 4.6 semantic_memory_grounding | true | pass | — | — |
| 4.7 semantic_memory_counterfactual_pairs | false | fail | yes | — |
| 4.8 degeneration_quality | true | pass | — | — |
| 4.9 prompt_diversity_without_memory | true | pass | — | — |
| 4.10 prefix_logit_drift_audit | true | pass | — | — |
| 4.11 retrieval_topk_semantic_shift | false | fail | yes | — |
| 4.12 repetition_segment_audit | true | pass | — | — |
| 4.13 save_load_consistency | false | fail | yes | — |
| 4.14 training_cache_isolation | true | pass | — | — |
| 4.15 prefix_stepwise_drift_trajectory | true | pass | — | — |
| 4.16 retrieval_generation_alignment_audit | false | fail | yes | — |
| 4.17 retrieval_prefix_decode_correlation_audit | true | pass | — | — |
| 4.18 cheating_heuristics | true | pass | — | — |
| 4.19 stepwise_label_mass_alignment_audit | false | fail | yes | — |
| 4.20 rerank_stability_probe | true | pass | hard_PASS | — |
| 4.21 decode_repetition_feedback_probe | true | pass | hard_PASS | — |
| 4.22 functional_token_suppression_probe | true | pass | hard_PASS | — |
| 4.23 keyword_specific_tail_slot_probe | false | fail | no (PASS_or_not_impl) | v3.45 metric |
| 4.24 context_descriptor_cluster_probe | false | fail | no (PASS_or_not_impl) | v3.45 metric |
| 4.25 prefix_length_scaling_probe | true | pass | no (PASS_or_not_impl per v3.45) | v3.45 metric |
| 4.26 mixture_distribution_gate_probe | true | pass | — | — |

## 4. Count summary

- total: 26
- pass: 19
- fail: 7
- not_implemented: 0
- error: 0
- blocking_fail: 5 (4.7, 4.11, 4.13, 4.16, 4.19)

## 5. Delta vs v3.44-Trained

Same SUT weights. Only the runner's metrics for 4.23 / 4.24 / 4.25 changed, plus `AMS_DETERMINISTIC=1`.

| case_id | prior_passed | current_passed | prior_status | current_status |
|---|---|---|---|---|
| 4.25 prefix_length_scaling_probe | false | true | fail (old metric: saturation-bound top-12 count) | pass (v3.45 metric: starter_mass_ratio) |

No other case changed.

## 6. Per-failing-case evidence

### 4.23 `keyword_specific_tail_slot_probe` (FAIL under v3.45 metric)

- metric_version: v3.45
- `mean_intersection_size_top20`: 0.0 (threshold ≥ 1.0)
- `median_rank_of_best_rare`: 4291.0 (threshold ≤ 100)
- `hit_ratio_at_least_one_top20`: 0.0 (threshold ≥ 0.5)
- gap: median rank is 40× above threshold
- vocabulary: 151936
- rank 4291 corresponds to top 2.82% of vocab — the tail slot is not random (random would be ~50%), but not concentrated on the rare keywords either

### 4.24 `context_descriptor_cluster_probe` (FAIL under v3.45 metric)

- metric_version: v3.45
- `loo_nn_accuracy`: 0.600 (threshold ≥ 0.75)
- `correct`: 3 / 5 labeled memories
- `n_labeled`: 5
- `music_gap` (diagnostic): -0.0791
- `space_gap` (diagnostic): +0.2472
- `unit_norm_within_1e_3`: true
- gap: 2 of 5 memories classified into wrong domain; all failures are music memories classified as space

### 4.13 `save_load_consistency` (FAIL)

- prompt: `"The pianist"`
- output_a: `"The pianist piano piano practiced difficult Chop piano perfect hours hours practiced perfect difficult Chop perfect Chop difficult hours practiced"`
- output_b: `"The pianist piano hours piano，"什么意思_____ noct hours hours noct，\n---\n\n noct + piano perfect"`
- divergence_step: 1 (after shared prefix `"piano"`)
- `AMS_DETERMINISTIC=1` was active: `torch.set_num_threads(1)` + `torch.use_deterministic_algorithms(True, warn_only=True)`. Divergence persists.

### 4.7, 4.11, 4.16, 4.19

Unchanged from v3.44-Trained. Per SPEC Section 7.3 the numeric evidence is recorded in `reports/v345_runner_update_blackbox/report.json` under each case's `results` entry.

## 7. Mechanism notes (Section 7.6, non-normative, falsifiable)

- **4.25 transition (FAIL → PASS)**: Under the v3.45 metric `avg_mass_ratio_B_over_A = 1.38` with per-prompt ratios `[0.91, 1.93, 1.27]`. The previous `top12 saturation` metric was unreachable. Falsifiable: set `L_mem_B = L_mem_A` and rerun; prediction `avg_mass_ratio = 1.00 ± 0.02`.
- **4.23 persistent FAIL under v3.45 metric**: median rank 4291 indicates the mean-centered tail slot does carry some rare-keyword direction, but not enough to cross into top-100 out of 151936. Falsifiable: increase training steps from 60 to 300 and rerun 4.23; prediction `median rank` decreases monotonically in step count.
- **4.24 persistent FAIL under v3.45 metric**: LOO NN accuracy 3/5. The `music_gap` negative value (−0.08) under the hybrid encoder's β=0.8 indicates `hidden_mean` dominated the representation, overriding the WTE-centroid's domain discriminator. Falsifiable: set `context_hybrid_hidden_weight = 0.1` (via Cfg override only) and rerun; prediction `music_gap > 0` and `loo_nn_accuracy ≥ 0.75`.
- **4.13 persistent FAIL under `AMS_DETERMINISTIC=1`**: divergence origin is not in thread-scheduled kernels (those are now single-threaded). Candidate sources: `torch.randperm` in `PrefixAligner.calibrate` before each `MemLLM.load()`; `torch.linalg.svd` in `DirectionTree._split`; memory state mutation between the first save and the first generate. Falsifiable: explicitly seed before each `generate()` call with the same value across A and B; if divergence disappears, root cause is RNG state between calls.

## 8. Artifact links

- `reports/v345_runner_update_blackbox/report.json`
- `reports/v345_runner_update_blackbox/report.md`
- `reports/v345_runner_update_blackbox/runner.log`
- `reports/v345_runner_update_blackbox/audit_feedback.md` (this file)

## 9. Retraction statement

This report cites metrics that were introduced in SPEC PR #18 (2026-04-20 correction). The pre-v3.45 runner-update reports (`reports/v337/…/v344_trained_blackbox/`) recorded 4.23 / 4.24 / 4.25 under the old metrics. Per SPEC Section 7.8, statements in those reports that used single-probe PASS/FAIL as evidence about the channel as a whole are superseded. Their numeric measurements remain valid as artifacts under their original metrics.
