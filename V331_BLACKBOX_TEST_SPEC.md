# v3.31 Black-box Test Specification

This document records the complete external black-box test conditions and the concrete test cases currently used for `v3.31`.

The suite is the same external runner structure used for `scheme_b_v31_blackbox_eval.py`, with the tested target swapped to `scheme_b_v331`.

## 1. Test Policy

The suite is designed to evaluate the system through exported runtime behavior only.

Hard constraints:

- No `mock`
- No `fallback`
- No `overfit`
- No simplified replacement path
- No monkeypatching
- No reuse of the module-internal `test()`
- No source modification during the audit run

Allowed behavior:

- Real `torch`
- Real `transformers`
- Real HuggingFace causal LM
- Real memory write / retrieve / generate / train / save-load flow

## 2. Runner-Level Conditions

- External runner only
- Fixed seeds per case for reproducibility
- Black-box interaction through model construction and public runtime methods
- Detailed JSON + Markdown reporting
- Report fields include pass/fail, error details, and per-case metrics

## 3. Shared Corpora

### Music corpus

1. `The pianist practiced arpeggios and Chopin nocturnes until midnight.`
2. `A musician refined finger technique, phrasing, and pedal control on the piano.`
3. `Classical interpretation often depends on dynamics, tempo rubato, and touch.`
4. `A conservatory student studied etudes, scales, and expressive voicing on the keyboard.`

### Space corpus

1. `Astronomers observed distant galaxies, quasars, and stellar evolution in deep space.`
2. `Orbital mechanics explains how satellites and planets move under gravitational force.`
3. `A telescope captured nebulae, exoplanets, and spectral signatures from distant stars.`
4. `Cosmology studies dark matter, expansion, and the large scale structure of the universe.`

### General corpus

1. `The cat sat on the mat and watched the birds outside the window.`
2. `Quantum computing uses qubits existing in superposition states.`
3. `Machine learning algorithms identify patterns in large datasets.`
4. `The ancient temple was hidden deep within the tropical rainforest.`
5. `The stock market experienced significant volatility during the session.`
6. `He practiced piano for hours perfecting a difficult Chopin nocturne.`
7. `The restaurant served an exquisite five course meal with wine pairings.`
8. `The professor explained relativity using simple everyday analogies.`

## 4. Full Case List

### 4.1 `leaf_capacity_stability`

- Seed(s): `0..7`
- Input:
  - `Cfg(tree_max_leaf=5, tree_K=3)`
  - Insert 240 randomly directed `MemEntry` items into `DirectionTree`
- Observe:
  - `leaf_size_violations()`
  - `verify_consistency()`
  - depth and count per seed
- Pass:
  - no leaf overflow
  - no tree/store inconsistency
  - all seeds pass

### 4.2 `degenerate_direction_boundary`

- Seed: `17`
- Input:
  - `Cfg(tree_max_leaf=5, tree_K=3)`
  - 100 nearly collinear directions with only `1e-9`-scale perturbation
- Observe:
  - tree depth
  - count
  - `leaf_size_violations()`
  - `verify_consistency()`
- Pass:
  - consistency remains valid under extreme directional collapse

### 4.3 `metric_trainability`

- Seed: `23`
- Input:
  - build model
  - write `corpus_general()`
  - run one `Trainer.step(corpus_general()[:3])`
- Observe:
  - gradient norms of `model.amm.metric` parameters
  - parameter deltas after the step
  - training info payload
- Pass:
  - at least one metric parameter has non-zero gradient
  - at least one metric parameter changes after the step

### 4.4 `no_grad_generation`

- Seed: `29`
- Input:
  - build model
  - write `corpus_general()`
  - `with torch.no_grad(): generate("The pianist", mt=24, greedy=True)`
- Observe:
  - stored memory count
  - output string
- Pass:
  - memories were written
  - output is a non-empty string

### 4.5 `counterfactual_memory_influence`

- Seed: `31`
- Input:
  - music-only model
  - space-only model
  - prompt: `Tell me something about practice and performance.`
- Observe:
  - `music_output`
  - `space_output`
- Pass:
  - outputs differ

This checks that different memory states change answer content, not only surface token noise.

### 4.6 `semantic_memory_grounding`

- Seed: `33`
- Input:
  - blank model
  - music-memory model
  - space-memory model
  - prompt: `Explain what someone should focus on when improving technique and understanding the subject.`
- Observe:
  - keyword scores against derived music keywords
  - keyword scores against derived space keywords
  - blank baseline lift
- Pass:
  - `music_margin > 0`
  - `space_margin > 0`
  - at least one of `music_lift` or `space_lift` is positive

### 4.7 `semantic_memory_counterfactual_pairs`

- Seed: `35`
- Input prompts:
  - `Describe the most important details a student should notice.`
  - `Summarize the key ideas a learner should practice and remember.`
- Setup:
  - music-memory model
  - space-memory model
- Observe per prompt:
  - music output keyword margin
  - space output keyword margin
- Pass:
  - for every prompt, music output favors music keywords
  - for every prompt, space output favors space keywords

### 4.8 `degeneration_quality`

- Seed: `36`
- Input:
  - write `corpus_general + corpus_music + corpus_space`
  - prompts:
    - `The pianist`
    - `The telescope`
    - `The forest path`
    - `The market analyst`
    - `Explain the topic clearly`
- Observe aggregate text metrics:
  - `avg_unique_token_ratio`
  - `avg_repeated_bigram_ratio`
  - `avg_content_token_ratio`
  - `avg_newline_ratio`
  - `worst_max_token_run`
  - prompts judged short or hollow
- Pass thresholds:
  - `avg_unique_token_ratio >= 0.35`
  - `avg_repeated_bigram_ratio <= 0.20`
  - `avg_content_token_ratio >= 0.22`
  - `avg_newline_ratio <= 0.20`
  - `worst_max_token_run <= 4`
  - no short-or-hollow prompt

### 4.9 `prompt_diversity_without_memory`

- Seed: `37`
- Input prompts:
  - `The pianist`
  - `Quantum systems`
  - `The rainforest`
- Setup:
  - empty-memory model
- Observe:
  - outputs for all three prompts
  - unique output count
- Pass:
  - all outputs are distinct

### 4.10 `prefix_logit_drift_audit`

- Seed: `38`
- Input:
  - prompt: `Explain the topic in a precise and concrete way.`
  - blank model and memory-loaded model
  - compare `use_prefix=False` vs `use_prefix=True`
- Observe:
  - JS divergence of final-step logits
  - L2 shift of final-step logits
  - top-k overlap counts
  - entropy changes
- Pass:
  - memory condition shows stronger prefix-induced drift than blank condition by at least one of:
    - higher JS divergence
    - higher L2 shift
    - lower top-k overlap

### 4.11 `retrieval_topk_semantic_shift`

- Seed: `39`
- Input prompts:
  - `A strong explanation should mention`
  - `The most relevant idea is`
- Setup:
  - music-memory model
  - space-memory model
- Observe:
  - top-k logits before prefix
  - top-k logits after prefix
  - domain keyword hit count
  - domain keyword probability mass
- Pass:
  - at least one prompt shows stronger domain alignment after prefix injection

### 4.12 `repetition_segment_audit`

- Seed: `40`
- Input prompts:
  - `The pianist`
  - `The telescope`
  - `The market analyst`
  - `Explain the topic clearly`
- Setup:
  - write `corpus_general + corpus_music + corpus_space`
- Observe:
  - segment-level repetition statistics with `window=8`
  - bad-segment ratio
  - first bad segment index
  - early collapse prompts
- Pass:
  - `bad_segment_ratio <= 0.35`
  - at most one prompt collapses in segment `0` or `1`

### 4.13 `save_load_consistency`

- Seed: `41`
- Input:
  - model A writes `corpus_general()`
  - save memory to temp file
  - model B loads that memory
  - prompt: `The pianist`
- Observe:
  - `output_a`
  - `output_b`
- Pass:
  - both outputs are identical

### 4.14 `training_cache_isolation`

- Seed: `43`
- Input:
  - write `corpus_general()`
  - snapshot every memory entry's `(last, cnt)`
  - run `trainer.recon("Some query text that triggers retrieval.")`
- Observe:
  - any memory entries whose `last` or `cnt` changed
- Pass:
  - no cached training/reconstruction path mutates retrieval bookkeeping

### 4.15 `prefix_stepwise_drift_trajectory`

- Seed: `44`
- Input prompts:
  - `Key piano ideas include`
  - `Explain the topic clearly`
- Setup:
  - write `corpus_general + corpus_music`
- Observe:
  - 16-step decode trace under prefix
  - `first_bad_step`
  - stepwise token-category drift
- Pass:
  - `first_bad_step` is absent, or `>= 3`

This is meant to catch early-step collapse into function words or punctuation.

### 4.16 `retrieval_generation_alignment_audit`

- Seed: `45`
- Setup:
  - write labeled memory items from music and space corpora
- Input prompts:
  - `What improves piano technique and musical phrasing?` → expected `music`
  - `What explains satellites and orbital motion?` → expected `space`
  - `Summarize the subject with concrete domain details.` → expected `None`
- Observe:
  - retrieved memory ids
  - retrieved label majority
  - generated label from keyword scoring
  - diagnosis:
    - `aligned`
    - `retrieval_miss`
    - `bridge_unused`
    - `unknown`
- Pass:
  - no expected-domain case may fail due to wrong-domain retrieval
  - expected-domain cases should align retrieval and generation

### 4.17 `retrieval_prefix_decode_correlation_audit`

- Seed: `46`
- Setup:
  - labeled music + space memories
- Input prompts:
  - `What improves piano technique and musical phrasing?`
  - `What explains satellites and orbital motion?`
  - `Describe what a student should focus on first.`
  - `Summarize the subject with concrete domain details.`
  - `Key piano ideas include`
  - `Orbital motion depends on`
- Observe:
  - retrieval strength
  - prefix L2 shift
  - top-k non-semantic probability mass
  - bad-decode score
  - correlations:
    - retrieval strength vs prefix L2
    - retrieval strength vs bad decode
    - prefix L2 vs bad decode
- Pass:
  - no strong positive correlation showing that stronger retrieval/prefix perturbation makes decode worse:
    - `corr_retrieval_bad <= 0.2`
    - `corr_prefix_bad <= 0.2`

### 4.18 `cheating_heuristics`

- Seed: `47`
- Input prompts:
  - `The pianist`
  - `The telescope`
  - `The trader`
  - `The child`
- Observe:
  - whether all outputs are exactly the same
  - whether outputs are only the prompt itself
  - whether all outputs are too short to count as real generation
- Pass:
  - not exact-same across prompts
  - not prefix-only
  - not trivially short

This is the direct anti-shortcut / anti-test-fitting probe.

### 4.19 `stepwise_label_mass_alignment_audit`

- Seed: `48`
- Setup:
  - labeled music + space memories
- Input prompts:
  - `What improves piano technique and musical phrasing?` → expected `music`
  - `What explains satellites and orbital motion?` → expected `space`
- Observe:
  - 12-step alignment trace
  - stage diagnosis counts per step:
    - `retrieve`
    - `inject`
    - `decode`
    - others
- Pass:
  - no row may accumulate retrieve-stage failure
  - no row may accumulate inject-stage failure

### 4.20 `rerank_stability_probe`

> Cipher attribute: **invocation strategy (消歧 / 歧义消解)**.
> Maps to P0 proposal "C-6 confidence gating". Targets the 4.6 regression family.

- Seed: `49`
- Setup:
  - write `corpus_music() + corpus_space()` as labeled memories
- Input pairs (same domain, near paraphrase):
  - P1a: `What improves piano technique and musical phrasing?`
  - P1b: `How can one improve piano technique and musical expression?`
  - P2a: `What explains satellites and orbital motion?`
  - P2b: `What describes satellites and the motion of planets?`
- Observation protocol (purely via public behavior):
  - for each prompt, record the ordered list of memory ids reached through the public `prepare_decode_context` path, specifically the `dominant_per_batch` value and the first five entries of `batch_mem_weights[0]` sorted by weight
  - compute the Jaccard overlap between the two resulting top-5 mid sets for pair P1 and pair P2
  - compute the rank-correlation (Spearman) of the shared elements within each pair
- Pass:
  - `jaccard(P1a.top5_mids, P1b.top5_mids) >= 0.6`
  - `jaccard(P2a.top5_mids, P2b.top5_mids) >= 0.6`
  - `spearman(shared_ranks) >= 0.5` for at least one of the two pairs
- Anti-cheating: any attempt to short-circuit retrieval for these exact prompts (direct mid pinning / prompt-keyed router) invalidates the probe.

Rationale: the 4.6 regression in v3.37 was caused by C-6 rerank flipping the top-1 on borderline queries without sufficient confidence. A confidence-gated rerank should produce stable orderings across semantic-equivalent phrasings.

### 4.21 `decode_repetition_feedback_probe`

> Cipher attribute: **anti-collapse (抗塌缩)**.
> Maps to P0 proposal "generation-history feedback into bias". Targets 4.8.

- Seed: `50`
- Setup:
  - write `corpus_general() + corpus_music() + corpus_space()`
- Input prompts:
  - `The telescope`
  - `The pianist`
  - `The market analyst`
- Protocol:
  - run `generate(prompt, mt=30, greedy=True)` per prompt
  - tokenize the newly generated suffix (exclude the prompt tokens)
  - identify the multiset of content tokens among the first 20 generated tokens
  - compute, per prompt: `max_repeat_per_content_token`, `first_bigram_repeat_index`, and `trigram_lock_count` (number of distinct trigrams that appear twice or more)
- Pass (aggregate over the three prompts):
  - `avg(max_repeat_per_content_token) <= 3.0`
  - `min(first_bigram_repeat_index, default=∞) >= 4` on prompts where any bigram repeats
  - `avg(trigram_lock_count) <= 1.0`
- Anti-cheating: disabling the decode shaping path is not allowed; the probe must pass with the system's production decode-time pipeline.

Rationale: the `telescope telescope telescope` collapse pattern in 4.8 shows the cipher lacks feedback from already-emitted tokens. This probe measures exactly that.

### 4.22 `functional_token_suppression_probe`

> Cipher attribute: **expressive volume (声量)**.
> Maps to P1 proposal `L_functional_suppression`. Targets 4.7 / 4.10.

- Seed: `51`
- Setup:
  - music-memory model
- Input prompts (all chosen because Qwen's unconditional top-12 is dominated by functional tokens):
  - `A strong explanation should mention`
  - `The most relevant idea is`
  - `A learner should know about`
- Protocol:
  - for each prompt, compute top-12 of final-step logits under two conditions:
    - (A) no prefix (pure backbone, baseline functional-token concentration)
    - (B) with memory prefix from `prepare_decode_context` (cipher active)
  - count the number of content-starter tokens in each top-12
    (`content_starter_count_no_prefix`, `content_starter_count_with_prefix`)
  - also compute `logit_margin_best_content_starter_vs_best_functional` under condition (B)
- Pass (aggregate over the three prompts):
  - `avg(content_starter_count_with_prefix - content_starter_count_no_prefix) >= 1.5`
  - for at least 2 of 3 prompts: `logit_margin_best_content_starter_vs_best_functional >= 0`
- Anti-cheating: hard-masking functional tokens at decode time does not count as passing this probe; the cipher must raise content starters above functional tokens through prefix / bias, not through masking alone. The probe captures top-12 before any cyclic or newline hard-mask is applied.

Rationale: this is the core "声量不足" probe. If the bridge cannot push even one rare domain content starter into the top-12 relative to `" the"/" a"/" at"`, then 4.7 / 4.10 are unreachable by construction.

### 4.23 `keyword_specific_tail_slot_probe`

> Cipher attribute: **expressive vocabulary (词汇表宽度)**.
> Maps to P1 proposal "IDF-top-K keyword-specific tail slot". Targets 4.15 inject stage.

- Seed: `52`
- Setup:
  - music-memory model
- Protocol (pure API surface observation):
  - for each memory `m` in `amm.tree.store.values()`:
    - let `rare_keywords(m)` = the top-3 strict content starters in `m.content_token_ids` by descending corpus IDF (IDF is computed via the same code path as `_compute_corpus_idf`)
    - build a single-batch query that retrieves with `m` as the dominant memory (either by reusing `m.source_text` or by crafting a prompt containing its rare keywords)
    - obtain the runtime `fiber_summary` via `prepare_decode_context`
    - if `bridge._last_tail_slots` is not None, take the last tail slot, project it through `backbone.input_embedding_weight().T`, and read the top-3 vocabulary tokens
    - compute `intersection_size = |top3_tokens ∩ rare_keywords(m)|`
- Pass:
  - `mean(intersection_size) >= 1.0` across all memories that yielded a non-None tail slot
  - at least 50% of memories yield `intersection_size >= 1`
- Not-implemented path: if the system does not expose a keyword-specialized tail slot (the generic TailHead of v3.37 currently does not), the probe must record `status = "not_implemented"` rather than synthesize a shim. In that state the probe does not count toward suite PASS but must still be emitted for observability.

Rationale: `tail_semantic_anchor` in v3.37 trains to uniform content distribution, which is why 4.15 fails at inject stage. A specialized tail slot that projects onto the memory's rare strict starters is the minimal architectural delta needed.

### 4.24 `context_descriptor_cluster_probe`

> Cipher attribute: **invocation strategy (调用精细度)**.
> Maps to P2 proposal `MemEntry.context_descriptor`. Targets 4.6 / 4.9.

- Seed: `53`
- Setup:
  - write `corpus_music() + corpus_space()` (4 + 4 memories) into a fresh model
- Protocol:
  - for each stored memory, read `context_descriptor` from its `MemEntry`
  - partition memories by ground-truth domain label (music / space)
  - compute `intra_domain_cos_mean` (mean pairwise cosine within a domain) and `inter_domain_cos_mean` (mean pairwise cosine across domains)
- Pass:
  - `intra_domain_cos_mean - inter_domain_cos_mean >= 0.15` for both domains
  - every descriptor is unit-norm (tolerance `1e-3`) if the implementation advertises it as a direction
- Not-implemented path: if `MemEntry` does not carry `context_descriptor`, the probe records `status = "not_implemented"`. This is expected for v3.37; the probe is introduced here so that v3.38 and beyond have a concrete acceptance test for the upgrade.

Rationale: this probe defines the acceptance criterion for the "per-memory context descriptor" upgrade without committing to any specific clustering algorithm.

### 4.25 `prefix_length_scaling_probe`

> Cipher attribute: **expressive capacity (密语信道容量)**.
> Maps to P2 proposal "L_mem scaling". Targets 4.7 / 4.10.

- Seed: `54`
- Setup:
  - music-memory model, constructed twice under the same seed and corpus:
    - model A with `Cfg(L_mem = default)` (default is the production value, currently `8`)
    - model B with `Cfg(L_mem = 2 × default)`
  - identical write order; identical rerank / gate settings
- Input prompt:
  - `A strong explanation should mention`
- Observation:
  - for both models, record the count of content-starter tokens in the top-12 of the final-step logits after memory-prefix injection
    (`starters_A`, `starters_B`)
  - also record the L2 norm of the prefix tensor per slot as a sanity check
    (per-slot norms are expected to remain on the same scale due to `prefix_norm_clamp`)
- Pass:
  - `starters_B >= starters_A + 1`
  - the prefix L2 per slot remains within a `±15%` band between A and B
    (scaling length should not be confused with scaling magnitude)
- Anti-cheating: no separate training between A and B is permitted; both models must be loaded at eval-time from the same checkpoint if one exists, or initialized from the same seed without task-specific training. The probe is about **capacity**, not about re-optimization.

Rationale: longer prefix should monotonically expand cipher capacity for at least the "声量" axis. If doubling L_mem doesn't help, it signals that capacity is not the bottleneck (the bridge itself is) — a result as informative as a PASS.

### 4.26 `mixture_distribution_gate_probe`

> Cipher attribute: **expressive form (密语表达形式)**.
> Maps to P3 proposal "Mixture-of-Distributions gate". Targets 4.7 / 4.10 / 4.15 simultaneously if landed.

- Seed: `55`
- Setup:
  - music-memory model
- Protocol (API-level observation, no mocking):
  - call `prepare_decode_context` for a fixed input
  - inspect the returned `DecodeContext` (or equivalent object) for a per-token gate tensor `g` of shape `[B, V]` with values in `[0, 1]`
  - if present: verify that for 32 random prompt continuations the decoder output logit can be written as `(1 - g) * lg_raw + g * lg_memory` within numerical tolerance `1e-4`
  - also verify `g.mean()` behaves in a controlled way under `_mem_guidance_active = False` (should go to zero)
- Pass:
  - gate tensor exists and is bounded in `[0, 1]` element-wise
  - identity-decomposition check passes within tolerance
  - gate collapses to near-zero under inactive guidance (`mean < 0.05`)
- Not-implemented path: v3.37 does not expose a mixture gate; this probe records `status = "not_implemented"` and defines the acceptance criterion for v3.39+ if the P3 upgrade is taken.

Rationale: a mixture-of-distributions formulation is the most radical of the seven proposals because it changes the decode composition from additive (`lg += bias`) to convex (`lg = (1-g)·raw + g·mem`). Its acceptance probe needs to be specified explicitly because it is not backwards-compatible with v3.37's CFG path.

## 4-meta. Cipher-System Structural Probes Summary

Cases `4.20 – 4.26` form the `Cipher-System Structural Probes` subsuite, organized around the four cipher attributes:

| Case | Cipher attribute | Priority | Target pre-existing FAIL | Gating |
| --- | --- | --- | --- | --- |
| 4.20 rerank_stability_probe | invocation strategy | P0 | 4.6 | hard PASS |
| 4.21 decode_repetition_feedback_probe | anti-collapse | P0 | 4.8 | hard PASS |
| 4.22 functional_token_suppression_probe | expressive volume | P1 | 4.7, 4.10 | hard PASS |
| 4.23 keyword_specific_tail_slot_probe | expressive vocabulary | P1 | 4.15 inject | PASS or `not_implemented` |
| 4.24 context_descriptor_cluster_probe | invocation strategy | P2 | 4.6, 4.9 | PASS or `not_implemented` |
| 4.25 prefix_length_scaling_probe | expressive capacity | P2 | 4.7, 4.10 | hard PASS |
| 4.26 mixture_distribution_gate_probe | expressive form | P3 | 4.7, 4.10, 4.15 | PASS or `not_implemented` |

Interpretation rules:

- `hard PASS` probes must pass for the suite to be considered fully green. If the implementation does not support them, the corresponding FAIL is binding.
- `PASS or not_implemented` probes emit `status ∈ {"pass", "fail", "not_implemented"}`. Only `fail` blocks suite PASS. `not_implemented` is allowed for upgrades that have not yet landed, and must be truthful: a probe reported as `not_implemented` must come from an actual absence of the API surface, not from a silenced error path.
- None of these probes may be satisfied by prompt-keyed shortcuts, mocked return paths, or test-only code paths. The same "no mock / no fallback / no overfit" policy from Section 1 applies.

## 5. Anti-Cheating Interpretation

For `v3.31`, this suite is considered valid only if the following remain true during execution:

- no mocked return path
- no special-case keyword router for the listed prompts
- no hard-coded answer templates keyed to the audit corpus
- no inference-time shortcut pretending to be learned behavior
- no degraded alternate implementation that exists only to satisfy the suite

For the `Cipher-System Structural Probes` subsuite (`4.20 – 4.26`), the following additional constraints apply:

- a probe labelled `not_implemented` must be the result of a genuinely missing API surface, not a silent suppression. The runner must emit an explicit marker (e.g. `status = "not_implemented"` plus the name of the missing attribute or method) rather than a bare PASS.
- no probe may be satisfied by adding a helper path that activates only when one of the listed prompts is detected.
- the same `torch` / `transformers` / HuggingFace-backed model must be used; no dedicated small-stub model may replace the production backbone for the purposes of these probes.

## 6. Summary

This external suite is not a unit test collection for individual functions. It is a behavior-level black-box audit spanning:

- structural stability
- trainability
- no-grad generation
- counterfactual memory influence
- semantic grounding
- degeneration resistance
- prefix efficacy
- retrieval/decode alignment
- cache isolation
- anti-cheating checks
- cipher-system structural probes (4.20 – 4.26)

`v3.31` is judged against the full set above under the same no-mock / no-fallback / no-overfit / no-simplification policy.

The `Cipher-System Structural Probes` subsuite is forward-looking: it defines the acceptance criteria for the v3.38+ structural upgrades derived from the cipher-system analysis (expressive volume, expressive vocabulary, invocation strategy, anti-collapse, expressive capacity, expressive form). Probes that target upgrades not yet landed emit `not_implemented` rather than fail, which keeps the suite usable as a progress tracker across versions.

## 7. Reporting Discipline (mandatory)

All human-authored audit reports, PR descriptions, commit messages, change summaries, and inter-version comparisons produced against this suite MUST adhere to the following reporting discipline. This section is not stylistic; it is a normative part of the audit contract.

### 7.1 Banned language

The following categories of language are prohibited in audit reports:

- Celebratory framing: "wins", "胜利", "big improvement", "breakthrough", "major progress", "landmark", "historic best", "finally", "at last", "as expected", "as predicted".
- Self-congratulation or reassurance: "honest", "honest progress", "honest failure", "good news / bad news", "the good side / the bad side", "silver lining", "promising direction", "encouraging sign".
- Consolation or softening: "minor regression", "only slightly worse", "essentially the same", "negligible", "almost passes", "close to threshold", "one step away", "nearly green".
- Hype / marketing language: "state of the art", "best-in-class", "industry-leading", "game-changing", "elegant", "beautiful", "clean solution".
- Emotive adjectives attached to numbers: "strong", "weak", "healthy", "painful", "dramatic", "dramatic drop".

Any report containing the above phrases (in English or Chinese, including direct synonyms) MUST be rewritten before being merged or published.

### 7.2 Required report structure

Every audit report MUST contain the following sections in this order, and only these sections, plus artifact links:

1. **Run parameters**: SUT version, runner version, seed policy, device, elapsed seconds, exit code.
2. **Per-case result table**: one row per case, columns = `case_id`, `name`, `passed` (true/false), `status` (`pass`/`fail`/`not_implemented`/`error`), `blocking` (true/false per Section 4-meta), `seed`, `elapsed_seconds_case` (if measured).
3. **Count summary**: integer counts only, no narrative. Required counts: total, pass, fail, not_implemented, error, blocking_fail.
4. **Delta vs. prior version**: a table listing every case whose `(passed, status)` tuple changed between the previous audited version and the current one. Columns: `case_id`, `prior_passed`, `current_passed`, `prior_status`, `current_status`. Unchanged cases are omitted.
5. **Per-failing-case evidence**: for every case with `passed=false`, emit a raw evidence block containing (a) the measured metric(s) named in the pass criterion of Section 4, (b) the threshold, (c) the gap. No causal interpretation is permitted in this section.
6. **Mechanism notes (optional, non-normative)**: if the report author wishes to record a mechanism hypothesis linking a regression to a code change, it goes here. Every entry MUST be expressed as a falsifiable statement with (i) the named code element, (ii) the observed behavior, (iii) a testable prediction. No value judgments.
7. **Artifact links**: relative paths to `report.json`, `report.md`, `runner.log`, and any supporting files.

### 7.3 Writing rules

- State results as measurements. Example of compliant wording: "case 4.13 `save_load_consistency` failed; output_a and output_b diverge after the shared prefix of length 19 tokens." Example of non-compliant wording: "4.13 unfortunately regressed — an honest consequence of our improvements."
- Do not attribute intent to the system. "The bridge learned to ..." is banned; "`ContentSemanticTailHead.forward` produced slot[1] with cosine X to the rare keyword centroid" is required.
- Do not use comparative adjectives where a number would do. "Marginal" must be replaced by the numeric margin. "Significantly better" must be replaced by the delta.
- Do not hedge numerical FAILs with qualifiers. "FAIL at 0.278 vs threshold 0.20" is required; "narrowly FAIL" is banned.
- Do not characterize absence of PASS as progress. If the count decreased, it decreased. Report the count.
- Do not announce category winners. There are no winners in an audit. There are passing cases, failing cases, and measured numbers.

### 7.4 Counting conventions

- `blocking_fail` is a hard fail of any original case (4.1 – 4.19) or any `hard_PASS` probe (4.20 – 4.22, 4.25). `not_implemented` never counts as blocking. A non-blocking probe FAIL counts as a FAIL, not as a softer state.
- Version-to-version comparison tables MUST include all versions that have recorded artifacts in the repository; partial comparisons are not permitted.
- The total-pass line MUST be expressed as the raw integer over the total; no percentages, no "rate of improvement" calculations.

### 7.5 Error handling in reports

- When a case raises an exception, `status = "error"` is distinct from `status = "fail"`. The error traceback goes into the per-case evidence block verbatim. No paraphrase.
- When a probe reports `not_implemented`, the report MUST name the missing API literally (attribute name, method name, Cfg flag, or dataclass field), not describe it.

### 7.6 Enforcement

- A report violating Section 7.1 or 7.3 is itself invalid; the PR containing it is not mergeable until the report is rewritten.
- The audit runner and its output JSON are not subject to these rules (they are machine output). Only human-authored summaries, commit messages, PR descriptions, and analysis documents are.
- This section applies retroactively to all future audits starting at v3.40 and forward. Prior reports are not required to be rewritten, but may be rewritten voluntarily.
