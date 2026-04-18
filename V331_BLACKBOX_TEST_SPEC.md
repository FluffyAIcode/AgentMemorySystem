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

## 5. Anti-Cheating Interpretation

For `v3.31`, this suite is considered valid only if the following remain true during execution:

- no mocked return path
- no special-case keyword router for the listed prompts
- no hard-coded answer templates keyed to the audit corpus
- no inference-time shortcut pretending to be learned behavior
- no degraded alternate implementation that exists only to satisfy the suite

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

`v3.31` is judged against the full set above under the same no-mock / no-fallback / no-overfit / no-simplification policy.
