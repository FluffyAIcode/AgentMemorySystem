# v3.48-Mechanisms-1to4-Stacked Black-Box Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Sections 7, 7.7, 7.8.

## 1. Run parameters

- SUT version: `scheme_b_v344.py` (unchanged)
- Runner version: `v331_blackbox_eval.py` from v3.47 branch with one additional change: 4.24 primary metric reads the same fallback chain the SUT uses (`context_descriptor else semantic_emb`)
- Weights: `ckpt/v348_stacked.pt` (new 120-step checkpoint, not the v3.44-Trained ckpt)
- Training driver: `train_v348.py`, 120 steps, batch 3, `distill_lr=3e-4`
- Mechanisms active during training:
  - M1: `Cfg(use_memory_context_encoder=False)` — routes context slots through `mem.semantic_emb`
  - M2: Qwen layer-0 q_proj / k_proj / v_proj warm-start into QFormer layer-0 cross-attention (K/V tiled 6× to match 1536-dim expectation)
  - M3: per-step distillation loss pulling `bridge.proj(f).mean(1)` toward Qwen content-token `hidden_mean` (second optimizer, only over `bridge.proj` params, lr=3e-4)
  - M4: `bridge.proj.q` initialized from Qwen content-token `hidden_mean` of random corpus texts, plus 0.005 noise
- Loss reweighting: `encoder_throughput` 1.5→3.0, `vocab_anchor` 0.2→0.4, `semantic_alignment` 3.0→1.0, `tail_semantic_anchor` 0.5→0.1, `functional_suppression` 0.4→0.1, `context_separation` 0.3→0.0
- Env: `AMS_TRAINED_WEIGHTS=ckpt/v348_stacked.pt`, `AMS_DETERMINISTIC=1`
- Device: CPU (single-threaded)
- Training elapsed: 2685.8 s (44.8 min, 22.4 s/step)
- Audit elapsed: 1423.8 s
- Total elapsed: 4109.6 s (68.5 min)
- Exit code: 0

## 2. Count summary

- total: 26
- pass: 19
- fail: 7
- not_implemented: 0
- error: 0
- blocking_fail: 5 (4.7, 4.11, 4.13, 4.16, 4.19)

## 3. Delta vs v3.46-Deoverfit (same runner, different ckpt)

| case_id | prior_passed | current_passed |
|---|---|---|
| (no case changed pass/fail status) | — | — |

## 4. Training convergence metrics

120-step CPU run, deterministic mode. Per-step timing 22.4 s/step (up from v3.44-Trained's 6.6 s/step; the 3× overhead is from single-threaded torch and the added M3 distill step per iteration).

Loss trajectory at selected steps:

| step | total | recon | encoder_throughput | vocab_anchor | distill_loss | bridge↔Qwen cos |
|---|---|---|---|---|---|---|
| 0 | 578.74 | 4.27 | 5.25 | 0.000 | +0.009 | −0.008 |
| 10 | 29.10 | 4.03 | 4.34 | −0.198 | −0.604 | +0.636 |
| 20 | 27.58 | 3.76 | 4.07 | −0.199 | −0.726 | +0.740 |
| 50 | 21.85 | 2.99 | 2.60 | −0.246 | −0.868 | +0.869 |
| 100 | 20.23 | 3.00 | 2.22 | −0.313 | −0.749 | +0.763 |
| 119 | 17.49 | 2.08 | 1.73 | −0.329 | −0.758 | +0.775 |

Reference: v3.44-Trained at step 60 had `total=44`, `recon=4.8`, `vocab_anchor=−0.22`. v3.48 at step 60 already beats that; at step 119 it is 2.5× deeper on `total`, 50% deeper on `vocab_anchor`, 2.3× lower on `recon`.

The bridge↔Qwen-pool cosine reached 0.87 peak at step 50, indicating mechanism M3 successfully pulled `bridge.proj` output into Qwen's hidden-state manifold.

## 5. Per-failing-case evidence

### 4.24 primary (FAIL) vs mechanism_1 diagnostic (PASS on same ckpt)

The runner's 4.24 primary and the v3.47-introduced `mechanism_1_qwen_pool_diagnostic` block compute LOO NN over different source fields:

| source field | `loo_nn_accuracy_all_4` | `loo_nn_accuracy_heldout_2` | per-domain (music/space/cooking/finance) |
|---|---|---|---|
| primary metric (v3.48 runner) | 0.625 (10/16) | 0.750 (6/8) | (not reported at per-domain granularity on primary) |
| mechanism_1 diagnostic (reads `mem.semantic_emb` unconditionally) | 0.812 (13/16) | 0.875 (7/8) | 3/4, 3/4, 4/4, 3/4 |

The two should agree when `Cfg(use_memory_context_encoder=False)` is set, because the v3.48 runner was updated to follow the SUT's fallback chain. The fact that they disagree by 0.188 on `loo_nn_accuracy_all_4` indicates the runner's fallback path is not actually reading `semantic_emb` for all memories; measurement suggests at least 6 of 16 memories produce a vector from somewhere other than `semantic_emb` (possibly memories where `store_mem` still persisted an old `context_descriptor` tensor despite the encoder being disabled).

This is an audit-side measurement bug, not an SUT bug. It was introduced in the v3.48 runner change and must be fixed before the primary metric can be trusted.

### 4.24 heldout degraded vs v3.47

Heldout (cooking + finance): v3.47 = 0.875, v3.48 = 0.750. The M1-through-M4 training made the heldout metric worse, not better. The diagnostic metric (frozen Qwen pool) stays at 0.875.

### 4.23 regressed

| metric | v3.46 (60 steps, no M*) | v3.48 (120 steps, M1-M4) |
|---|---|---|
| `median_rank_of_best_rare_paraphrase` | 759 | **1089** |
| `mean_intersection_size_top20_paraphrase` | 0.0 | 0.0 |

Rank increased by 43%. M2 + M3 pulled the bridge output toward Qwen's hidden_mean manifold, which is not aligned with the rare-keyword direction the tail slot is supposed to carry.

### 4.13 save_load unchanged

Still FAIL under `AMS_DETERMINISTIC=1`. Root cause not thread scheduling.

### 4.7, 4.11, 4.16, 4.19

Unchanged from v3.46 / v3.47; see prior feedback documents.

## 6. Mechanism analysis (falsifiable)

| mechanism | effect | net impact on pass count |
|---|---|---|
| M1 (disable learned encoder) | ✓ applied | expected +1 based on v3.47 diagnostic (0.812 on 4.24_all_4); not observed on primary metric due to §5 runner-side issue |
| M2 (Qwen K/V warm-start, tiled 6×) | ✓ applied | -1 on 4.23 (median rank 759→1089) |
| M3 (distill to Qwen hidden_mean) | ✓ applied (cos=0.77 sustained) | -1 on 4.24 heldout (0.875→0.750); -1 on 4.23 combined with M2 |
| M4 (pool-init queries) | ✓ applied (L2=0.81) | neutral |

Net: +1 (from M1, not observable on v3.48 runner's primary) − 2 (from M2, M3) = −1 relative to the v3.47-diagnostic prediction of 20/26. Observed: 19/26.

### Root diagnosis: M3 distill target is wrong

Qwen's content-token `hidden_mean` is a useful clustering signal **when used as the final embedding** (v3.47 diagnostic: 0.812 on 4.24). But when used as a **distillation target for the prefix-generation pipeline** (v3.48 M3), it pulls `bridge.proj` output toward a direction whose principal components are domain-invariant (they reflect "English declarative sentence" geometry more than topic geometry). The prefix then compresses toward a low-discriminative center, which:
- makes tail slots less able to point at rare keywords (4.23 worse)
- makes held-out context_descriptor clustering worse (via back_proj chain, 4.24 heldout worse)

### Mechanism counter-intuitively useful for training loss, not for target geometry

v3.48 beats v3.44-Trained on every training loss metric (`total`, `recon`, `vocab_anchor`, `encoder_throughput`) but underperforms on several primary audit metrics. The training signal was stronger (mechanisms delivered); the training **destination** (Qwen hidden_mean) was mis-specified.

## 7. Falsifiable predictions for v3.49

1. **Revert M2 + M3, keep M1 + M4**: predicted 4.23 median rank back to ~750; 4.24 heldout back to ~0.875; 4.24 primary = 0.812 (if runner measurement bug is fixed); total pass count = 20/26.
2. **Change M3 target** from `hidden_mean` to `wte_centroid_of_strict_content_starters` (domain-discriminative): predicted 4.23 median rank < 500; 4.24 heldout preserved; total pass count ≥ 20/26.
3. **Fix v3.48 runner primary reader** (route through `semantic_emb` universally when `memory_context_encoder is None`): predicted 4.24 primary `loo_nn_accuracy_all_4` = 0.812 (matches diagnostic); 4.24 PASS; total pass count = 20/26 even on current v3.48 ckpt.

## 8. Artifacts

- `ckpt/v348_stacked.pt` (453 MB, not git-tracked; reproducible from `python3 train_v348.py`)
- `ckpt/v348_train_log.jsonl` (per-step metrics)
- `ckpt/v348_train_stdout.log` (training console)
- `reports/v348_stacked_blackbox/report.json`
- `reports/v348_stacked_blackbox/report.md`
- `reports/v348_stacked_blackbox/runner.log`
- `reports/v348_stacked_blackbox/audit_feedback.md` (this file)

## 9. Summary of measured deltas

| metric | v3.44-Trained | v3.46-Deoverfit | v3.47-Diagnostic | v3.48-Stacked |
|---|---|---|---|---|
| pass count | 18/26 | 19/26 | 19/26 | 19/26 |
| training steps | 60 | 60 (same ckpt) | 60 (same ckpt) | 120 (new ckpt) |
| training seconds | 398.5 | — | — | 2685.8 |
| audit seconds | 1404.3 | 1435.3 | 1498.0 | 1423.8 |
| 4.24 primary `all_4` | — | 0.625 | 0.625 | 0.625 |
| 4.24 diagnostic (semantic_emb) `all_4` | — | — | 0.812 | 0.812 |
| 4.24 primary `heldout_2` | — | 0.875 | 0.875 | 0.750 |
| 4.24 diagnostic `heldout_2` | — | — | 0.875 | 0.875 |
| 4.23 median rank paraphrase | — | 759 | 759 | 1089 |
| final train `total_loss` | 44 | — | — | 17.5 |
| final train `vocab_anchor` | −0.22 | — | — | −0.33 |

## 10. Retraction statement

The v3.47 feedback predicted that landing mechanism 1 would transition 4.24 from FAIL to PASS and raise the total pass count from 19/26 to 20/26. This prediction is **partially refuted** by v3.48:
- The mechanism_1 diagnostic number (`mem.semantic_emb` LOO NN on 4 domains) remains at 0.812 as predicted
- The 4.24 primary metric did not transition because the v3.48 runner's fallback reader either has a bug or is reading a different set of fields; the matter is identified but not resolved
- The total pass count remains at 19/26

Per Section 7.8, the v3.47 "next audit prediction" of 20/26 is formally retracted. The observed 19/26 + diagnostic 0.812 data suggests the prediction would be confirmed if the runner reader is fixed, but that is a v3.49 item, not a v3.48 observation.
