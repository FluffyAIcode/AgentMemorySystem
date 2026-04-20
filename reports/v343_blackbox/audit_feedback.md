# v3.43 Black-Box Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Section 7 (Reporting Discipline).

## 1. Scope and configuration

- SUT: `scheme_b_v343.py` via `AgentMemorySystem.py` redirect.
- Runner: `v331_blackbox_eval.py`, unmodified.
- Spec: `V331_BLACKBOX_TEST_SPEC.md`, unmodified.
- Backbone: `Qwen/Qwen2.5-1.5B-Instruct`, `llm_dtype=bf16`, CPU execution.
- Test set: 26 cases.
- Elapsed: 1452.3 s.

## 2. Aggregate

- Checks passed: 15 / 26.
- Checks failed: 11 / 26.

Comparison to v3.42 (17 / 26 pass):

| Transition | Count | Cases |
| --- | --- | --- |
| FAIL → PASS | 0 | — |
| PASS → FAIL | 2 | 4.8 degeneration_quality; 4.11 repetition_segment_audit |
| Persistent PASS | 15 | (unchanged) |
| Persistent FAIL | 9 | 4.7, 4.10, 4.12, 4.15, 4.17, 4.21, 4.23, 4.24, 4.25 |

Net change: **−2**. The `[I-1]` symmetric-CFG hypothesis was incorrect; algebraic analysis in §4.3 shows why.

## 3. Cases that transitioned PASS → FAIL

### 3.1 `degeneration_quality` (4.8)

| metric | v3.42 | v3.43 | threshold |
|---|---|---|---|
| `avg_unique_token_ratio` | 0.410 | **0.329** | ≥ 0.60 |
| `avg_repeated_bigram_ratio` | 0.073 | 0.110 | ≤ 0.25 |

Sample (prompt="The pianist"): `"practiced pian pian regularly cards pian Chop pian Chop Chop practiced Chop pian practiced practiced... Chop card practiced ..."`. 

Mechanism: see §4.3.

### 3.2 `repetition_segment_audit` (4.11)

`bad_segment_ratio = 0.143` (above threshold); `early_collapse_prompts = [The pianist, The telescope, The market analyst]` — all 3 test prompts. Sample: `"pian pian piano pianist pian piano piano pian piano perfect piano ..."`.

### 3.3 `decode_repetition_feedback_probe` (4.21, stays FAIL but worse)

`avg_max_repeat_per_content_token = 5.33` (was 3.33 in v3.42). Content tokens repeat up to 5 times within the 30-token window.

## 4. Mechanism analysis — I-1 algebraic error

### 4.1 Intended behaviour

`[I-1]` claim: attaching `cb` to both `prefix_cond` and `prefix_uncond` would make `fwd` add the same bias to `lg_cond` and `lg_uncond`, so that the CFG differential `(1+α)·lg_base − α·lg_uncond` cancels `cb`:

```
lg = (1+α)(lg_base_raw + cb) − α(lg_uncond_raw + cb)
   = (1+α)lg_base_raw − α·lg_uncond_raw + cb
```

The bias appears once in the final `lg`. This is actually the **correct** simplification. But at `dampen=1.0` (was 0.25 in v3.42, 0.0 in v3.40), the magnitude is:

`cb × logits_std(2.0) × std_multiplier(1.5) × content_bias_scale(6.0) × step_scale(1.0) × dampen(1.0) ≈ cb × 18 logit`

### 4.2 What happens in practice

Observation: `lg_cond` **before CFG** already contains bias, because fwd was applied to `prefix_cond`. Runner case 4.8 samples from `lg` (after CFG). In CFG:
- Each of `lg_base` and `lg_uncond` has bias at +18 logit scale
- After CFG `(1+α) lg_base − α lg_uncond`, bias coefficient is `(1+α) − α = 1` → bias appears at +18 logit scale.

Compared to v3.42:
- v3.42 (asymmetric, dampen=0.25): cb appears only in `lg_cond`, weight in CFG = `(1+α) = 4.5`, at dampen=0.25 magnitude: `0.25 × 18 × 4.5 = 20.3 logit` effective boost for lg_cond, **but only on cond side** — so lg_uncond_raw was bias-free and the CFG differential computed correctly (boost ≈ `4.5 × 4.5 = 20 logit`).
- v3.43 (symmetric, dampen=1.0): bias appears at `18 logit` in **both** lg_base and lg_uncond; CFG differential cancels to 1× so effective = 18 logit.

Net: v3.42 delivered ~20 logit via `(1+α) × damped bias`; v3.43 delivered ~18 logit via `full undamped bias × 1`. Magnitudes nearly equal — **not the issue**.

### 4.3 Actual root cause

`repetition_segment_audit` prompt `"The pianist"` sample shows the same content token ("pian", "piano", "Chop") repeating 4–5 times per segment. The memory corpus has only 3 music sentences; their content_bias vector concentrates ~80% of weight on 6 tokens: `piano, pianist, Chopin, practice, nocturne, midnight`.

With content_bias now **saturated at single-point ~18 logit boost** and no damping through decoding:
1. Step 0: `piano` hits +18, wins.
2. Step 1: `content_repeat_penalty=2.5 × 1.0 = 2.5` penalises `piano`; but bias is still +18 → `piano` wins again.
3. Step N: penalty = `2.5 × N` (linear). At N=7, penalty = 17.5 → bias (+18) still dominates by 0.5 logit → word still wins.

`[I-2]` linearised the repeat penalty (`exponent 1.5 → 1.0`). This was intended to prevent crashes after few repeats, but combined with `[I-1]`'s undamped bias it **removed the exponential cap** that previously broke repetition cycles.

Effective equilibrium:
- v3.42: `penalty = 3.5 × N^1.5` vs `bias = 4.5 logit (damped × CFG)`. Equilibrium at N=1.5 → token seen ~2 times then broken.
- v3.43: `penalty = 2.5 × N^1.0` vs `bias = 18 logit (undamped)`. Equilibrium at N=7.2 → token seen 7 times before broken.

Observed: `avg_max_repeat = 5.33` (4.21), matching predicted N ≈ 7 minus cyclic hard-mask `max_count=5`.

**Falsifiable experiment**: revert `fwd_path_bias_dampen` to 0.25 (keep `[I-1]` symmetric). Predicted effect: bias magnitude drops to 4.5 logit, equilibrium with `2.5 × N` penalty at N=1.8, restoring v3.42 behaviour for 4.8 and 4.11, without disturbing 4.10/4.15 because lg_cond still receives the full (1+α)× amplification via CFG.

## 5. Persistent FAIL

### 5.1 `context_descriptor_cluster_probe` (4.24)

| metric | v3.41 | v3.42 | v3.43 |
|---|---|---|---|
| `intra_music_mean_cos` | 0.304 | 0.108 | **0.900** |
| `intra_space_mean_cos` | 0.389 | 0.344 | **0.904** |
| `inter_domain_mean_cos` | 0.290 | 0.192 | **0.842** |
| music gap | 0.014 | −0.084 | 0.058 |
| space gap | 0.099 | 0.152 | 0.062 |

`[I-5]` hybrid encoder with `β=0.8`: intra values jumped to 0.90 as expected (hidden_mean gives strong intra cohesion), but `inter` also jumped to 0.84 because hidden_mean between music and space texts is still ≈ 0.85 (Qwen pre-trained representations pool to similar directions for English declarative sentences). Gap reduced to 0.06 on both sides. Hybrid trade-off: high intra comes with high inter.

### 5.2 `semantic_memory_counterfactual_pairs` (4.7)

Still `margin=0.0`. `[I-2]` made repetition cap looser but did not move domain-keyword overlap. Structural runner-side issue.

### 5.3 `retrieval_topk_semantic_shift` (4.10)

Still FAIL — the +18 logit bias at lg_cond (before CFG) **would** be enough to cross the 13-logit function-word gap, but runner samples `music_no_prefix` top-12 at `lg_cond[:, -1, :]` directly after `fwd(ids, mask, prefix=None)`, **without memory prefix**. This is the runner's baseline sampling point; I-1 does not affect it.

### 5.4 `prefix_stepwise_drift_trajectory` (4.12)

Still `first_bad_step = 0`. `[I-3]` excludes prompt content from residual — confirmed by sample where v3.43 output has more varied content (`"key changes throughout, dynamic changes, articulation, and phrasing"`) compared to v3.42 (`"key changes key signatures key signature"`). But step 0 still picks `' key'` (functional category per runner). Runner's `first_bad_step=0` is triggered by the category mismatch at step 0, not the repetition cycle. The first token after "include" being `key` was already triggered before [I-3] engaged because prompt `ids` at step 0 contain `key`, which [I-3] excludes from rare_keyword_ids — so the residual targets the second-rank keyword for step 0. But the CFG-amplified content_bias still boosts `key` via the general content_bias path (not residual path). Mechanism: content_bias is built from `mem.content_token_ids` which includes `key` (Qwen's `key` tokenises as strict starter). `[I-3]` only touches residual, not content_bias.

### 5.5 `stepwise_label_mass_alignment_audit` (4.15)

Still `logits_label_mass=0`. The bias DOES reach lg_cond now (+18 logit), but runner measures `probability mass`, not logit — and the bias applies only to memory's content tokens, not the runner's specific 12 label tokens unless they overlap. Partial overlap exists (`pianist, practiced, chopin, nocturnes, midnight`), but the probability sum is still below runner's quantisation (~0.01) because the 12 logit values each sit around `bias + baseline ≈ 18 − 3 = 15`, while competing functional tokens are around 18–21.

### 5.6 `save_load_consistency` (4.17)

Still divergent outputs. `[I-7]` `torch.set_num_threads(1)` + `use_deterministic_algorithms(True, warn_only=True)` only applied inside the test_save_load test function; runner's case 4.17 runs independently from SUT's internal test. Mechanism is SUT-side, requiring the deterministic scope to wrap the runner-level sampling. Not addressable within spec's black-box boundary.

### 5.7 `keyword_specific_tail_slot_probe` (4.23)

Still `top3 = [0,1,2]` equivalents. `[I-4]` subtracted `wte_global_mean` from residual centroid — but apparently the residual tensor's sign/scale interacts with the post-aligner additive such that after global mean subtraction the tail-slot direction, when inner-producted against non-mean-centered WTE, still hits token id 0/1/2 due to those tokens' WTE norms being unusually small (they lie near the WTE subspace where the mean-subtracted centroid points).

Falsifiable experiment: inner-product against `wte_centered = wte − wte_mean` for the top-K query. If rare keywords appear in top-3, the WTE geometry anomaly is confirmed.

### 5.8 `prefix_length_scaling_probe` (4.25)

`slot_norm_ratio = 0.785` — still < 0.85. `[I-6]` renormalised prefix_cond slots to target_norm, but the probe measures `per_slot_mean_norm` across body + tail + ctx slots. With `[I-1]`'s symmetric bias path, the uncond prefix goes through `build_neutral_prefix` which also applies renorm, but the **retrieved body-slot norms diverge** between cond (with content_bias) and uncond (without retrieval diversity). The runner averages across batch, and cond vs uncond geometry differs slightly.

Actually the test re-reads `per_slot_mean_norm_A/B` for two separate L_mem values on the **same** cond path. Re-inspection: `[I-6]` is active, `mean_norm_A=0.557` (L_mem=8, 5 body + 2 tail + 1 ctx, target ≈0.50). The renorm sets each slot to `target_std × √d_LLM = 0.03 × √1536 ≈ 1.18`. But actual reported value is 0.557, which is half. Cause: before renorm, the aligner already normalises with scale_logit; additional renorm to `target_norm` is nominal. But `[I-6]` is `use_slot_norm_renormalize=True` which multiplies `target_norm / cur_norms`. The mismatch shows aligner output norm after prefix_norm_clamp removal is different from before. This requires deeper inspection.

## 6. Aggregate technical observations

- `[I-1]` algebra analysis was correct in principle (bias appears once in final `lg`), but combined with `dampen=1.0` and `[I-2]`'s linearised repeat penalty the effective content_bias magnitude (~18 logit) overwhelms repetition suppression. Net: 4.8, 4.11 regress, 4.21 worse.
- `[I-2]` weakened repeat penalty in exchange for allowing domain keywords to appear more often; this removed the exponential cap needed to break content-bias-driven repetition cycles.
- `[I-3]` confirmed active (sample outputs diversified beyond `key key key`), but doesn't touch content_bias path.
- `[I-4]` — top-3 anomaly pattern continues; suggests mean-subtraction alone is insufficient.
- `[I-5]` shifted ctx descriptor magnitudes but did not widen gap.
- `[I-6]` — runner's `slot_norm_ratio` still < 0.85. Requires re-inspection of renormalisation scope.
- `[I-7]` not wired into runner's independent 4.17 case.

**Single-knob recovery path for v3.44**: set `fwd_path_bias_dampen = 0.25` while keeping `[I-1]` symmetric CFG. This restores v3.42's effective bias magnitude (~4.5 logit) with bias correctly visible in lg_cond. Predicted effects:
- 4.8, 4.11 → PASS (equilibrium moves to N≈1.8)
- 4.10, 4.15 → unchanged (both still sample `lg_cond` without prefix)
- 4.21 → PASS or borderline
- 4.12 → may improve (weaker bias → `key` not forced at step 0)
- 4.23, 4.24, 4.17, 4.25 → unchanged structural

## 7. Artifacts

- `reports/v343_blackbox/{report.json, report.md, runner.log, audit_feedback.md}`

## 8. Summary of measured deltas

| Pass count | 17 → 15 | −2 |
| Elapsed | 1418.4 s → 1452.3 s | +33.9 s (+2.4 %) |
| FAIL → PASS | 0 cases | — |
| PASS → FAIL | 2 cases | 4.8, 4.11 |
| Persistent FAIL | 9 cases | 4.7, 4.10, 4.12, 4.15, 4.17, 4.21, 4.23, 4.24, 4.25 |
