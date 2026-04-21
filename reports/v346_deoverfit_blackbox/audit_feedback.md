# v3.46-Deoverfit Black-Box Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Sections 7, 7.7, 7.8.

## 1. Run parameters

- SUT version: `scheme_b_v344.py` (unchanged)
- Runner version: `v331_blackbox_eval.py` updated per SPEC Section 4.22 / 4.23 / 4.24 v3.46 de-overfit corrections
- Weights: `ckpt/v344_trained.pt` (60-step checkpoint, unchanged from v3.44-Trained)
- Env: `AMS_TRAINED_WEIGHTS=ckpt/v344_trained.pt`, `AMS_DETERMINISTIC=1`
- Device: CPU (single-threaded)
- Seed policy: per-case seeds as defined in SPEC Section 4
- Elapsed: 1435.3 s
- Exit code: 0

## 2. Axis coverage (SPEC 4-meta.1)

```json
{
  "axis_a_compression":   { "ratio": 8.97, "threshold": 10.0, "passed": false },
  "axis_b_injection_cost": { "per_step_floats": 164224, "depends_on_N": false, "passed": true },
  "axis_c_fidelity":      { "passed_over_total": "5/11", "threshold_K": 9, "passed": false },
  "axis_d_stability":     { "passed_over_total": "2/3",  "threshold_all_pass": true, "passed": false },
  "channel_passes_all_axes": false
}
```

Same axis signature as v3.45-Runner-Update. The probe metrics changed, not the axis counts.

## 3. Count summary

- total: 26
- pass: 19
- fail: 7
- not_implemented: 0
- error: 0
- blocking_fail: 5 (4.7, 4.11, 4.13, 4.16, 4.19)

## 4. Delta vs v3.45-Runner-Update

Same SUT weights, same checkpoint. Only the runner's 4.22 / 4.23 / 4.24 probes were rewritten to remove test-design overfit (SPEC PR #20 content).

| case_id | prior_passed | current_passed | prior_metric | current_metric |
|---|---|---|---|---|
| (no case changed pass/fail status) | — | — | — | — |

Pass count unchanged at 19/26. The meaning of each case is what changed, not the count.

## 5. Per-case evidence under de-overfit metrics

### 5.1 4.22 `functional_token_suppression_probe` — PASS, selection bias refuted

- Set A (3 hand-picked prompts): `avg_starter_delta = 11.0`, `margin_wins = 3/3`
- Set B (3 held-out generic prompts: `"Tell me about"`, `"Please describe"`, `"Explain how"`): `avg_starter_delta = 10.0`, `margin_wins = 3/3`
- Both sets pass independently at thresholds (`delta >= 1.0`, `margin_wins >= 2`)
- Interpretation: the probe's PASS was not caused by prompt selection. Held-out prompts show the same magnitude of effect.

### 5.2 4.23 `keyword_specific_tail_slot_probe` — FAIL, circularity removed

- Query = paraphrase (`"She performed Beethoven sonatas with delicate phrasing on her grand piano."`)
- `query_disjoint_from_rare_keywords = True` (tokens-level check)
- Dominant memory retrieved: `mid=1` — `"A musician refined finger technique, phrasing, and pedal con..."` (same domain, different surface)
- `mean_intersection_size_top20_paraphrase = 0.0`
- `median_rank_of_best_rare_paraphrase = 759` out of 151936 (was 4291 under v3.45 round-trip metric — **5.7× improvement** in rank)
- `hit_ratio_at_least_one_top20_paraphrase = 0.0`
- `roundtrip_mean_intersection_top20_diagnostic = 0.0` (legacy round-trip also 0)
- Interpretation: the paraphrase protocol shows the tail slot is in the correct direction neighborhood (top 0.5% of vocab) but does not reach the top-20 threshold. Rank improvement refutes the hypothesis that round-trip was inflating the old metric. Round-trip was not inflating it; both protocols deliver intersection = 0.

### 5.3 4.24 `context_descriptor_cluster_probe` — FAIL (4-domain), held-out component PASSES

- `loo_nn_accuracy_all_4 = 0.625` (threshold ≥ 0.65, FAIL by 0.025)
- `loo_nn_accuracy_heldout_2 = 0.875` (threshold ≥ 0.70, PASS)
- Per-domain accuracy:

| domain | correct / n | status vs random (0.25) |
|---|---|---|
| cooking | 4 / 4 = 1.000 | far above |
| finance | 3 / 4 = 0.750 | above |
| music   | 1 / 4 = 0.250 | at random |
| space   | 2 / 4 = 0.500 | above |

- Confusion matrix (true → predicted):

```
             cooking  finance  music  space
cooking    [   4        0       0      0  ]
finance    [   1        3       0      0  ]
music      [   0        2       1      1  ]
space      [   1        0       1      2  ]
```

- Interpretation: the hand-crafted music+space pair performs worst. Held-out cooking+finance pair performs best. If the encoder were memorizing music/space (test overfit), the pattern would be inverted. The observed inversion **falsifies the overfit hypothesis for 4.24** while still showing that the encoder cannot reliably separate 4 domains.
- Mechanism note (Section 7.6): hybrid encoder's `hidden_mean` component with β=0.8 collapses music/space together because Qwen's hidden_mean for English declarative sentences clusters regardless of topic. Cooking (concrete action verbs) and finance (numeric/abstract) have more distinctive hidden_mean distributions, which survives the β=0.8 mixing. Falsifiable prediction: setting `context_hybrid_hidden_weight = 0.1` (Cfg override, no SUT change) and retraining predicts music accuracy rises above 0.5 while cooking accuracy stays above 0.75.

## 6. Cases unchanged from v3.45 context

### Persistent FAILs (7):

- 4.7 `semantic_memory_counterfactual_pairs` — runner's domain margin metric sees no discrimination on generic prompts
- 4.11 `retrieval_topk_semantic_shift` — runner samples no-prefix logits (outside SUT's control path)
- 4.13 `save_load_consistency` — output divergence at step 1 under `AMS_DETERMINISTIC=1`; root cause not thread scheduling
- 4.16 `retrieval_generation_alignment_audit` — output drifts into Qwen multilingual token space at 60-step training
- 4.19 `stepwise_label_mass_alignment_audit` — `logits_label_mass` quantized to 0 at 2-decimal precision
- 4.23 (discussed above)
- 4.24 (discussed above)

## 7. Retraction statement

Per SPEC Section 7.8:

- Pre-v3.46 reports (`reports/v337/…/v345_runner_update_blackbox/`) used 4.22 / 4.23 / 4.24 metrics that contained test-design overfit as documented in SPEC PR #20. Their numeric measurements remain valid artifacts under their original metrics but must not be cited as evidence for or against channel-axis generalization properties.
- Specifically: the v3.45 `keyword_specific_tail_slot_probe` result (median rank = 4291) and the v3.44-Trained `context_descriptor_cluster_probe` result (loo_nn = 0.60 on 2 domains) are superseded by the v3.46 de-overfit measurements in this report (median rank = 759 on paraphrase queries; loo_nn = 0.625 on 4 domains with held-out pair at 0.875).

## 8. Mechanism notes (Section 7.6, falsifiable)

- **4.22 held-out PASS at equal magnitude**: the 11.0 vs 10.0 per-set deltas differ by less than 10%. Falsifiable: if a new prompt set C were drawn from a different distribution (e.g. Qwen-biased content prompts rather than generic functional prompts), the delta should drop. Predicted: ≤ 1.0 per-set delta on Qwen-content-biased prompts.
- **4.23 paraphrase rank 759 vs round-trip 4291**: paraphrase query does reach the dominant memory, which means the bridge's retrieval subchannel generalizes. The rank improvement shows the tail slot does carry domain-semantic information, just not concentrated enough for top-20 intersection. Falsifiable: extending training from 60 to 300 steps predicts `median rank <= 300`. Another falsifiable: setting `wte_residual_alpha = 3.0` (Cfg override, no SUT change) predicts `median rank <= 200` at the cost of 4.12/4.21 trade-offs.
- **4.24 inverted pattern**: music worst, cooking best is opposite to what test-overfit would produce. This is evidence the encoder is NOT overfitted to music/space; it's undertrained on ALL domains with an additional β-induced collapse specific to domains whose hidden_mean overlaps (music and space both collapse to a generic "English declarative" hidden_mean direction). Falsifiable: under `context_hybrid_hidden_weight = 0.1`, music accuracy should rise by at least 0.25; if it stays at 0.25, the β value is not the dominant factor.
- **4.13 unchanged under determinism**: already documented in v3.45 feedback. Root cause is inside SUT state mutation on load, not in thread scheduling.

## 9. Artifact links

- `reports/v346_deoverfit_blackbox/report.json`
- `reports/v346_deoverfit_blackbox/report.md`
- `reports/v346_deoverfit_blackbox/runner.log`
- `reports/v346_deoverfit_blackbox/audit_feedback.md` (this file)

## 10. Summary of measured deltas (numeric only)

| metric | v3.44-Trained | v3.45-Runner-Update | v3.46-Deoverfit |
|---|---|---|---|
| pass count | 18/26 | 19/26 | 19/26 |
| elapsed (s) | 1404.3 | 1476.3 | 1435.3 |
| 4.22 metric version | v3.38 | v3.38 | v3.46 (held-out set added) |
| 4.22 set_a_delta | 8.33 | 8.33 | 11.0 |
| 4.22 set_b_delta | — | — | 10.0 |
| 4.23 metric version | v3.38 | v3.45 | v3.46 (paraphrase) |
| 4.23 median rank | — | 4291 | 759 |
| 4.24 metric version | v3.38 | v3.45 | v3.46 (4 domains) |
| 4.24 loo_nn (main) | — | 0.60 (2-dom) | 0.625 (4-dom) |
| 4.24 loo_nn heldout | — | — | 0.875 |
| 4.25 metric version | v3.38 | v3.45 | v3.45 (unchanged) |
| 4.25 avg_mass_ratio | — | 1.38 | 1.38 (not re-run) |
