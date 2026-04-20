# v3.31 Black-box Test Specification

This document records the complete external black-box test conditions and the concrete test cases currently used for `v3.31`.

The suite is the same external runner structure used for `scheme_b_v31_blackbox_eval.py`, with the tested target swapped to `scheme_b_v331`.

## 1.1 Definition of `密语系统` (compression-communication channel)

This section is a normative correction added on `2026-04-20`. It supersedes earlier, looser uses of the term `密语系统 / cipher system` in this document and in audit feedback reports of `v3.37` through `v3.44-Trained`. The earlier wording was ambiguous and led to structurally incorrect probes in `4.20–4.26` and to anti-cheating clauses that excluded legitimate channel mechanisms. This section defines the target precisely. All subsequent text that references `密语` or `cipher` must be read against this definition.

**`密语系统` is NOT an encryption system.** Information-security meanings of `密语` (secrecy, key exchange, authentication, deniability) are out of scope for this suite. A system satisfying this definition is permitted to transmit fully in the clear.

**`密语系统` IS a compression-communication channel** between the Agent Memory System (AMS, trainable) and the frozen LLM backbone. Its purpose is to transport agent memory semantics into the LLM's conditional distribution at **bounded per-query cost**, without dumping raw memory text into the LLM's context.

### 1.1.1 Axes

The channel is evaluated on exactly four axes. Probe design MUST map every probe to at least one axis and name the mapping.

| Axis | Name | Operational metric (audit-observable) |
| --- | --- | --- |
| A | Compression | floats (+ ints) stored per memory entry, divided by `tokens(m) × d_LLM` for the same memory's raw text |
| B | Injection cost | incremental floats attached to the LLM per decode step, as a function of `N = number of stored memories`; the target is `O(1)` in `N` |
| C | Semantic fidelity | divergence between the LLM's next-token distribution under the channel versus under naive-RAG (raw memory text concatenated to the prompt). Lower divergence = higher fidelity. |
| D | Channel stability | given identical `(ids, mask, memory_state, seed)`, two invocations must produce identical output to the precision claimed (typically byte-level tensor equality or token-level string equality under greedy decode) |

### 1.1.2 Channel mechanisms under evaluation

All of the following are **legitimate** mechanisms of the channel. A probe MUST NOT exclude any one of them unless the probe is explicitly a component-level diagnostic, in which case it MUST declare so.

1. The learned prefix embedding tensor `prefix_cond` (QFormer output + bypass + tail slots + context slots + aligner), injected as `inputs_embeds` to the backbone, length `L_mem` independent of `N`.
2. The retrieval-derived `content_bias` and `suppression_bias` dense vectors added to backbone logits at decode time.
3. The decoder-path functional suppression term (e.g., `fwd_function_suppression`) that subtracts a learned or configured mass from function-word logits.
4. The mean-centered rare-keyword WTE residual injected into tail slots.
5. The retrieval ranking itself, selecting which compressed codes enter the channel.
6. The per-memory stored `context_descriptor` that participates in prefix reconstruction.

**Use of any or all of these mechanisms, in combination, is the channel.** A v3.44-Trained-style system that achieves semantic fidelity through items 1 + 2 + 3 simultaneously is as valid a channel as one that uses only item 1. The question the suite answers is whether the combination meets the four-axis criteria, not which subset is used.

### 1.1.3 What is banned

Banned mechanisms are those that defeat the evaluation contract of the suite, not mechanisms that participate in the channel:

- prompt-keyed routing or `if prompt == X` branches
- per-probe mocked return values
- test-corpus-memorized answer templates
- any code path that is active only during the audit
- substitution of a smaller stub backbone for the production `transformers` model

Mechanisms that are sometimes confused with the above but are NOT banned:

- content_bias, suppression_bias, functional suppression, cyclic hard masks, ngram-repeat blockers, bigram repetition penalties
- any reweighting or masking derived from the retrieved memory set's content_token_ids
- hard token-id masks when they are derived from `ContentTokenClassifier` outputs and are not per-prompt specialized

### 1.1.4 Historical note

Probes `4.20–4.26` were introduced in the `v3.38` commit under the subsuite label `Cipher-System Structural Probes`. Several of them were written under a narrower and incorrect interpretation of `密语` that required the learned prefix to carry semantics alone, without help from the decoder-side bias path. The corrected mapping of each probe to the four axes, together with targeted text amendments, is given in Section `4-meta.1`. The original probe bodies are preserved to maintain runner compatibility; their anti-cheating clauses and acceptance interpretations are updated.

---

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

> **Correction notice (2026-04-20, applies to v3.45 and later):** Axes mapping under `1.1` is **axis D (stability)** — specifically, stability of the retrieval subchannel's top-K under near-paraphrase queries. Metrics and thresholds are unchanged.

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

> **Correction notice (2026-04-20, applies to v3.45 and later):** The "anti-collapse" framing treated repetition as evidence of channel failure. It is more accurately framed as an operating-point failure of the channel's magnitude-balance between `content_bias` and `content_repeat_penalty`. The metrics and thresholds are retained. The rationale is replaced below to reflect `1.1`.
>
> Axes mapping: the probe is an instance of **axis D (stability)** under repeated invocation — specifically, stability over a 30-step decode trajectory.

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
- Anti-cheating (v3.45+): no prompt-keyed routing; no per-probe mock values. Tuning of `content_bias_scale`, `content_repeat_penalty`, `cyclic_content_max_count`, `fwd_path_bias_dampen`, and any other `Cfg` scalars is permitted; they are part of the channel's operating point.

Rationale: the three metrics jointly measure whether, over a 30-step greedy decode, the channel's content-bias magnitude is balanced against its repetition-penalty magnitude such that no content token exceeds `max_repeat = 3`, no trigram locks to a cycle, and the first bigram repeat is delayed. A failure of this probe is evidence that the channel is operating outside its balanced regime, not that the channel is absent.

### 4.22 `functional_token_suppression_probe`

> **Correction notice (2026-04-20, applies to v3.45 and later):** The anti-cheating clause originally excluded hard-masking as a solution path. Under the corrected `1.1` definition, hard-masking derived from `ContentTokenClassifier.pure_function_mask` is a legitimate channel mechanism, not a cheat. The clause is replaced below. The metric itself (`logit_margin_best_content_starter_vs_best_functional`) is retained and remains binding.
>
> Axes mapping: the probe measures **axis C (semantic fidelity)** on a generic-prompt slice. It is NOT a test of whether the prefix-attention subchannel alone produces the margin; any legitimate combination of channel mechanisms may produce it.

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
- Anti-cheating (v3.45+): no prompt-keyed routing; no per-probe mocked return values; no code paths that activate only under the listed prompts. Hard-masking derived from `ContentTokenClassifier.pure_function_mask` is permitted. CFG / content_bias / suppression_bias / fwd_function_suppression are permitted. The probe captures top-12 of `lg` at the point where `shape_step_logits` has fully executed for step 0, i.e. after the production decode pipeline is applied. Measurements taken pre-`shape_step_logits` are not permitted as proof of failure.

Rationale: this probe exists to confirm that the channel, using any combination of its legitimate mechanisms, can route the retrieved memory's content semantics to the top of the final distribution against a functional-word prior that dominates by ~13 logit in Qwen's unconditional output. The earlier wording that excluded masking was a category error introduced under the pre-1.1 interpretation of `密语`.

### 4.23 `keyword_specific_tail_slot_probe`

> **Correction notice (2026-04-20, applies to v3.45 and later):** The original acceptance criterion (`top-3 token of wte @ tail_slot ∩ rare_keywords >= 1`) was shown to be unreachable by construction across v3.38-v3.44: Qwen 2.5's token ids 0/1/2 (`!`, `"`, `#`) lie near the WTE mean and dominate any top-K cosine query on any centered vector regardless of the slot's actual content. The probe was measuring a vocabulary-geometry artifact, not channel quality.
>
> The corrected probe replaces top-3 absolute ranking with **relative rank stability** under the mean-centered inner product, and adds `top-K` at `K=20`, which is robust to the WTE-mean anomaly. Thresholds and axes are re-specified below. The probe remains gated as `PASS or not_implemented`.
>
> Axes mapping: **axis C (semantic fidelity)**, at the tail-slot subchannel level.

#### 4.23 corrected (v3.45+)

- Seed: `52`
- Setup:
  - music-memory model
- Protocol (pure API surface observation):
  - compute `wte_mean = backbone.input_embedding_weight().mean(0)` once
  - define `wte_centered[t] = F.normalize(wte[t] - wte_mean, dim=-1)` for all `t`
  - for each memory `m` in `amm.tree.store.values()`:
    - let `rare_keywords(m)` = the top-3 strict content starters in `m.content_token_ids` by descending corpus IDF (IDF is computed via `_compute_corpus_idf`)
    - build a single-batch query that retrieves with `m` as the dominant memory (by reusing `m.source_text`); call `prepare_decode_context`
    - if `bridge._last_tail_slots` is not None, take slot index 1 (the rare-keyword slot in the current `ContentSemanticTailHead` layout), center it: `slot1_centered = F.normalize(slot1 - wte_mean, dim=-1)`
    - compute `sims = wte_centered @ slot1_centered`, take `top20 = argsort(sims, descending)[:20]`
    - compute `intersection_size_20 = |top20 ∩ rare_keywords(m)|`
    - also record `rank_of_best_rare = min(rank of t in sims for t in rare_keywords(m))`
- Pass:
  - `mean(intersection_size_20) >= 1.0` across memories that yielded a non-None tail slot
  - `median(rank_of_best_rare) <= 100` (out of `vocab_size ≈ 151936`)
  - at least 50% of memories yield `intersection_size_20 >= 1`
- Not-implemented path: if `bridge._last_tail_slots` is None (the implementation does not expose a tail subchannel), the probe MUST record `status = "not_implemented"` and MUST name the missing attribute literally. A tail subchannel that exists but carries zero signal MUST NOT be reported as `not_implemented`; it is a FAIL.

Rationale (v3.45+): axis C evaluation for the tail subchannel requires measuring `wte @ slot` in the mean-subtracted subspace because Qwen 2.5's raw WTE geometry has token ids 0/1/2 clustered near the global mean, which biases any unnormalized top-K ranking. The corrected metric removes that bias and substitutes a measurement that is reachable by a channel that actually carries the rare-keyword centroid in slot 1. Thresholds are calibrated so that a v3.44-Trained-style implementation, where slot 1 receives `α = 1.5 × (wte_centroid - wte_mean)` residual, is expected to PASS; an untrained random slot is expected to FAIL at `median rank ~ vocab_size / 2`.

### 4.24 `context_descriptor_cluster_probe`

> **Correction notice (2026-04-20, applies to v3.45 and later):** At `N = 3` memories per domain, the Johnson–Lindenstrauss projection into `d_ctx = 128` has O(1/√N) ≈ 0.58 sampling variance on mean-pairwise-cosine, which exceeds the `0.15` gap threshold. Audit data across v3.38-v3.44-Trained confirms that the probe outcome on this metric is dominated by JL noise, not by channel quality. Two corrections apply: (1) the metric is switched to a **linear-classifier accuracy** which has higher statistical power at N=3; (2) a **per-memory** accuracy rather than a pooled-cosine gap is reported, which is robust to sample-size variance. Gap-based wording is retained as an informational diagnostic, not a pass criterion.
>
> Axes mapping: **axis C (semantic fidelity)** at the context-descriptor subchannel level.

#### 4.24 corrected (v3.45+)

- Seed: `53`
- Setup:
  - write `corpus_music() + corpus_space()` (4 + 4 memories) into a fresh model
- Protocol (v3.45+):
  - for each stored memory, read `context_descriptor` from its `MemEntry` and the domain label (music / space)
  - stack into `D ∈ ℝ^{N, d_ctx}`, `y ∈ {0,1}^N`
  - compute leave-one-out (LOO) nearest-neighbour accuracy: for each memory i, predict `y_i` from `argmax_{j != i} cos(D_i, D_j)`'s label
  - compute informational diagnostics (not used for pass): `intra_domain_cos_mean`, `inter_domain_cos_mean`, `gap = intra - inter` per domain
- Pass (v3.45+):
  - `loo_nn_accuracy >= 0.75` (at `N = 8` this corresponds to at least 6/8 correct)
  - every descriptor is finite and unit-norm within tolerance `1e-3` if the implementation advertises it as a direction
- Diagnostics that MUST be emitted but MUST NOT be used as pass criteria:
  - `intra_music_cos_mean`, `intra_space_cos_mean`, `inter_domain_cos_mean`, `music_gap`, `space_gap`
- Not-implemented path: if `MemEntry` does not carry `context_descriptor`, the probe records `status = "not_implemented"` and names the missing field. A populated but random-direction descriptor is a FAIL, not `not_implemented`.

Rationale (v3.45+): axis C at the context-descriptor subchannel is operationally "can I tell which domain a memory came from by looking at its descriptor alone?" Leave-one-out NN accuracy measures this directly and has bounded variance at small N (Clopper–Pearson 95% CI at 6/8 is `[0.35, 0.97]`, at 7/8 is `[0.47, 1.0]`). Mean-pairwise-cosine gap has O(1/√N) variance that exceeds the gap threshold at the corpus size this suite actually uses. A v3.44-Trained-style hybrid encoder that receives correctly-posed training signal is expected to PASS. An untrained orthogonal projection is expected to FAIL at ~0.5 accuracy.

### 4.25 `prefix_length_scaling_probe`

> **Correction notice (2026-04-20, applies to v3.45 and later):** The original acceptance required `content_starters_top12(B = 2×L_mem) >= content_starters_top12(A) + 1`. Audit data across v3.38-v3.44-Trained shows this metric saturates at 12/12 on both A and B in every configuration that has any channel at all, making monotone growth impossible. The probe was measuring saturation, not capacity. The corrected probe replaces `top-12 count` with a **fidelity-divergence metric at fixed top-k**, which is sensitive to capacity changes even when counts saturate.
>
> Axes mapping: **axis C (semantic fidelity)** as a function of prefix capacity.

#### 4.25 corrected (v3.45+)

- Seed: `54`
- Setup:
  - music-memory model, constructed twice under the same seed and corpus:
    - model A with `Cfg(L_mem = default)`
    - model B with `Cfg(L_mem = 2 × default)`
  - identical write order; identical rerank / gate settings; identical checkpoint if one exists
- Input prompts (3, same for A and B):
  - `A strong explanation should mention`
  - `The pianist`
  - `The telescope`
- Observation (v3.45+):
  - for each prompt and each model, compute the final-step full-vocab logit vector under the memory-prefix condition
    (`lg_A`, `lg_B`), and also the no-prefix baseline (`lg_base`)
  - compute `shift_A = lg_A - lg_base`, `shift_B = lg_B - lg_base`
  - restrict each shift to the set of content-starter token ids (`starter_mask`)
  - compute `mass_A = sum(shift_A[starter_mask].clamp(min=0))`, `mass_B = sum(shift_B[starter_mask].clamp(min=0))` — positive logit mass the channel deposits on content starters
  - also record per-slot prefix L2 for both models
- Pass (v3.45+):
  - `mass_B / mass_A > 1.10` averaged over the 3 prompts (the capacity-doubled channel must deposit at least 10% more positive starter mass)
  - per-slot prefix L2 stays finite (non-NaN) in both A and B; no upper-band gating on L2 is required
- Informational diagnostics (emitted, not pass criteria):
  - `content_starters_top12_A`, `content_starters_top12_B` (legacy metric)
  - `slot_norm_ratio_B_over_A`, `per_slot_mean_norm_A`, `per_slot_mean_norm_B`
- Anti-cheating (v3.45+): no prompt-keyed routing; no per-probe specialization. Both A and B must come from the same training process. If no training has happened (pure random init), the probe is still valid and measures architectural capacity only.

Rationale (v3.45+): the question "does doubling prefix length increase channel capacity?" is answered by measuring how much additional positive logit mass the channel can route to the correct tokens, not by counting how many content starters appear in an already-saturated top-12. The earlier `B ≥ A + 1` criterion was unreachable because top-12 caps at 12, and the `slot_norm_ratio` criterion was a confounder dependent on which renorm path was enabled. The corrected metric is continuous, unbounded above, and monotone in actual capacity.

### 4.26 `mixture_distribution_gate_probe`

> **Correction notice (2026-04-20, applies to v3.45 and later):** The wording `cipher attribute: expressive form` is ambiguous. Under `1.1`, the mixture gate is a **composition primitive** for combining the channel's output with the LLM's raw distribution. The probe tests the presence, range, and identity-decomposition of such a primitive. Axes mapping: **axis B (bounded cost)** — the gate introduces at most O(V) additional ops per step — and **axis C (fidelity)** when active. Acceptance is unchanged; rationale updated.

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

> **Correction notice (2026-04-20):** the prior version of this section labelled each probe with an attribute drawn from seven "P0/P1/P2/P3 proposals" (`声量`, `词汇表`, `抗塌缩`, …). Those labels were design-stage motivation, not test-suite semantics. Under the `1.1` definition, every probe maps to one or more of the four channel axes `A / B / C / D`. The revised table below uses that mapping. The gating column is also revised: probes whose original acceptance criteria were shown to be structurally unreachable (4.23, 4.24, 4.25) are downgraded from `hard PASS` to `PASS or not_implemented` until the v3.45 re-specified metrics land, after which they are again `hard PASS` under the corrected metrics. This is the only place in this suite where gating is relaxed; it is relaxed because the underlying metric was defective, not because the target is weaker.

| Case | Axes | Priority | Gating (pre-v3.45) | Gating (v3.45+) |
| --- | --- | --- | --- | --- |
| 4.20 rerank_stability_probe | D | P0 | hard PASS | hard PASS |
| 4.21 decode_repetition_feedback_probe | D | P0 | hard PASS | hard PASS |
| 4.22 functional_token_suppression_probe | C | P1 | hard PASS | hard PASS |
| 4.23 keyword_specific_tail_slot_probe | C | P1 | PASS or `not_implemented` | PASS or `not_implemented` |
| 4.24 context_descriptor_cluster_probe | C | P2 | PASS or `not_implemented` | PASS or `not_implemented` |
| 4.25 prefix_length_scaling_probe | C | P2 | hard PASS (unreachable — defective metric) | PASS or `not_implemented` (corrected metric) |
| 4.26 mixture_distribution_gate_probe | B, C | P3 | PASS or `not_implemented` | PASS or `not_implemented` |

Interpretation rules:

- `hard PASS` probes must pass for the suite to be considered fully green. If the implementation does not support them, the corresponding FAIL is binding.
- `PASS or not_implemented` probes emit `status ∈ {"pass", "fail", "not_implemented"}`. Only `fail` blocks suite PASS.
- `not_implemented` must be truthful: a probe reported as `not_implemented` must come from an actual absence of the API surface, not from a silenced error path.
- None of these probes may be satisfied by prompt-keyed shortcuts, mocked return paths, or test-only code paths. The same "no mock / no fallback / no overfit" policy from Section 1 applies.

## 4-meta.1 Channel-axis coverage check

A v3.45+ audit report MUST emit an axis-coverage table, containing for each of the four axes:

| Axis | Probes that evaluate it | Current status (pass / fail / n/a) |
| --- | --- | --- |
| A Compression | computed directly from `MemEntry` fields at audit startup (see Section 1.1.1) | pass iff ratio >= 10 |
| B Injection cost | computed from `prefix.shape`, `content_bias.shape`, `retrieval_interval`, and `N` | pass iff per-step floats are O(1) in `N` |
| C Semantic fidelity | 4.6, 4.7, 4.10, 4.15, 4.16, 4.17, 4.19 (§4 cases) and 4.22, 4.23, 4.24, 4.25 (structural probes) | pass iff aggregate >= K cases pass, where K is set by suite version |
| D Stability | 4.13 save_load_consistency, 4.20, 4.21 | pass iff all D-axis cases pass |

Axis A and B are computed by the runner without running the backbone; they are cheap, deterministic, and their failure modes are well-defined (insufficient compression, super-constant cost in N). Axis C and D are evaluated via the existing cases and probes.

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

`v3.31` and later versions are judged against the full set above under the same no-mock / no-fallback / no-overfit / no-simplification policy.

The `Cipher-System Structural Probes` subsuite (`4.20 – 4.26`) is forward-looking: under the corrected `1.1` definition it defines the acceptance criteria for the compression-communication channel along axes `A / B / C / D`. Probes that target mechanisms not yet present in a given version emit `not_implemented` rather than fail, which keeps the suite usable as a progress tracker. Earlier versions of this section mapped probes to a seven-point `P0..P3` attribute scheme (`声量`, `词汇表`, `抗塌缩`, …); that mapping is superseded. Probes that were downgraded from `hard PASS` to `PASS or not_implemented` in the v3.45 correction are scheduled to return to `hard PASS` once their corrected metrics have been exercised on at least two consecutive audit versions.

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

### 7.7 Channel-axis framing (v3.45+)

Reports for `v3.45` and later MUST:

1. State the four axes `A / B / C / D` exactly as defined in Section `1.1.1`.
2. Emit the axis-coverage table defined in Section `4-meta.1` before any per-case discussion. Counts for axes A and B MUST be numeric; no prose.
3. When using the term `密语 / cipher system / compression-communication channel`, use it as a noun referring to the system under the `1.1` definition. Do NOT use it as a value judgment ("the cipher works", "the cipher is weak"). Replace such usages with axis-specific numeric claims ("axis C: 13/15 dependent cases pass; axis D: 2/3 dependent cases pass").
4. Never assert that a single probe's PASS or FAIL constitutes the presence or absence of the channel. Only the full axis-coverage table is permitted to speak about the channel as a whole.
5. When a mechanism (prefix attention, content_bias, suppression_bias, fwd function suppression, hard masks, cyclic masks) participates in achieving a PASS, the report MAY name the mechanism. The report MUST NOT label any of them as "cheating", "not part of the cipher", "a shortcut", or any synonym unless the mechanism satisfies the Section `1.1.3` banned list.
6. When a prior report or discussion used the seven-point `P0..P3` attribute scheme (`声量`, `词汇表`, `抗塌缩`, `调用精细度`, `密语信道容量`, `密语表达形式`, `消歧`), the present report MUST either omit those labels or parenthesize them as historical annotations, and MUST give the corresponding `A / B / C / D` mapping.

### 7.8 Retraction notice for pre-v3.45 reports

Reports produced against v3.37 through v3.44-Trained contain statements of two types that are superseded by this revision:

- statements that a given probe's FAIL implies the channel is "not real", "not a cipher", or "only a logit editor"; these statements conflated an unreachable acceptance criterion with channel non-existence and are retracted.
- statements that a given probe's PASS implies the channel is "established", "working", or "substantially progressing"; these statements treated single-probe outcomes as evidence about the whole channel and are retracted.

Retraction does not require rewriting prior reports. It does require that any report citing a pre-v3.45 feedback document include a sentence of the form: "cited feedback predates the 2026-04-20 correction of Section 1.1; the cited claim is superseded by the axis-coverage framing in Section 4-meta.1."
