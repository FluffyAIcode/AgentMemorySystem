# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1508.0s`
- Passed: `20/26`
- Mode: fully external runner, no reuse of module-internal `test()`
- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts

## Axis Coverage (SPEC Section 4-meta.1, v3.45+)

```json
{
  "spec_section": "4-meta.1 v3.45+",
  "axis_a_compression": {
    "stored_floats_per_mem": 1712,
    "raw_floats_per_mem_typical_10_tokens": 15360,
    "ratio": 8.97196261682243,
    "threshold": 10.0,
    "passed": false
  },
  "axis_b_injection_cost": {
    "per_step_floats_formula": "L_mem * d_LLM + V",
    "per_step_floats_value": 164224,
    "depends_on_N": false,
    "passed": true
  },
  "axis_c_fidelity": {
    "dependent_cases": [
      "semantic_memory_grounding",
      "semantic_memory_counterfactual_pairs",
      "retrieval_topk_semantic_shift",
      "prefix_stepwise_drift_trajectory",
      "retrieval_generation_alignment_audit",
      "retrieval_prefix_decode_correlation_audit",
      "stepwise_label_mass_alignment_audit",
      "functional_token_suppression_probe",
      "keyword_specific_tail_slot_probe",
      "context_descriptor_cluster_probe",
      "prefix_length_scaling_probe"
    ],
    "passed_over_total": "7/11",
    "threshold_K": 9,
    "passed": false
  },
  "axis_d_stability": {
    "dependent_cases": [
      "save_load_consistency",
      "rerank_stability_probe",
      "decode_repetition_feedback_probe"
    ],
    "passed_over_total": "2/3",
    "threshold_all_pass": true,
    "passed": false
  },
  "channel_passes_all_axes": false
}
```

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 427.86248779296875, "recon": 3.2058675289154053, "contrast": 17888.765625, "holonomy": 5198.50439453125, "write_policy": 1.2801257371902466, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.648297071456909, "vocab_anchor": -0.0, "semantic_alignment": 9.940794944763184, "tail_semantic_anchor": 10.931857109069824, "functional_suppression": 0.0, "context_separation": 0.0, "slot_residual_alignment": 0.0, "inter_domain_margin": 0.0, "grad_norms": {"ctx_encoder": 2.2875472396664864e-17, "fib_encoder": 6.285003489921023e-15, "dir_predictor": 0.0, "fiber_connection": 1.315235037884401e-13, "fiber_attn": 1.2574421979159136e-16, "reranker": 2.6134262991325033e-19, "qformer": 8.558424465316837e-15, "content_bypass": 1.6076926531785598e-15, "semantic_probe": 0.0, "layer_pool": 5.479222219169355e-13, "prefix_aligner": 1.5589603208354674e-07, "vocab_proj": 2.7659061292216263e-06, "tail_head": 0.9999964974920388, "context_heads": 1.0952902730589725e-15, "memory_context_encoder": 1.5673766558890136e-15}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist hours piano perfect Chopsticks piano piano practiced difficult hours practiced perfect Chop Chop perfect hours Chop piano difficult practiced Course〕 Chop hours"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. Options later midnight Sure, practiced items mid noct midnight consistente performance.\n\" Baghd press practiced Items midnight noct Consistentent", "space_output": "Tell me something about practice and performance. Options group isAdmin extends Controller { fun space galaxies observed Tells spaceData(): Flow<Pair evolution BRAND QUESTION: deep dive\n#", "outputs_differ": true}
- `PASS` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Ohio Wesley Moo...\\ennocabularized explanation? Explain exactly enough technique tips.**\n\n  - Identify **muscle groups engaged**: Emphasis:\n\n- Hip flex", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. control Explain technique controlcontrol Understanding technique improvement finger Improve technique control control technique finger Hex control piano technique finger finger pedal control finger refined piano piano pedal pedal piano musician pedal", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. force explains mechanics mov
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. Conservation descriptions provide student conserv specialists (arch conserv conserv keyboardDescribe\nkeyboard describe:\n\n keyboard studied Describe \"study described\":\n\n** scales describes student keyboard", "space_output": "Describe the most important details a student should notice. large Describe helps break apart structure, roles scale describe xml scalescaledescribe.com\n\n studies\tdescribe provides XML scale descriptions.\n\n large expansion describes help dissect", "music_margin": 0.0, "space_margin": 0.0, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. touch summarize onTouchSummarize(java.nio depends Sum dependsExtract(key) {\ndepends interpretation touch rubricJava depends extract depends tempo summar ise touch", "space_output": "Summarize the key ideas a learner should practice and remember. large summarize Kotlin LargeSummarizer structure scale Sum summarizing Key large kotlin
- `FAIL` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist pian pian midnight Pell noct pian night pian practiced midnight midnight noct midnight pian noct noct Chop midnight practiced practiced noct practiced Chop practiced Chop Chop pian Chop", "token_count": 28, "unique_token_ratio": 0.25, "repeated_bigram_ratio": 0.037037037037037035, "max_token_run": 2, "punct_ratio": 0.0, "newline_ratio": 0.0, "alpha_ratio": 0.8527918781725888, "content_token_ratio": 1.0, "generated_preview": "pian pian midnight pell noct pian night pian practiced midnight midnight noct midnight pian noct noct chop midnight practiced practiced noct practiced chop practiced"}, {"prompt": "The telescope", "output": "The telescope telescope stars telescope css codes - Meaning, captured telescope telescope stars telescope , captured Telescope signatures meaning stars stars captured stars telescope captured spectral signatures captured signatures", "token_count": 25, "unique_token_ratio": 0.32, "repeated_bigram_ratio": 0.20833333333333334, "max_token_run": 2, "punct_ratio": 0.012987012987012988, "newline_ratio": 0.0, "alpha_ratio": 0.8658008658008658, "content_token_ratio": 0.96, "generated_preview": "telescope
- `PASS` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.33677369356155396, "l2_shift": 1042.322998046875, "topk_overlap_count": 3, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 5.552061080932617, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"t
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.0, "total_segments": 24, "bad_segments": 0, "early_collapse_prompts": []}, "rows": [{"prompt": "The pianist", "output": "The pianist pian piano hours pian pian 做 perfect hours piano pian practiced hours perfectedo difficult perfect practiced practiced perfect piano piano practiced midnight difficult hours difficult nocturn Chopsticks Chop Chop noct noct midnight midnight pian Chop midnight noct Chop practiced difficult difficult practiced noct", "generated_token_count": 42, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["pian", "piano", "hours", "pian", "pian", "perfect", "hours", "piano"], "unique_ratio": 0.5, "content_ratio": 0.75, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.375}, {"segment_idx": 1, "tokens": ["pian", "practiced", "hours", "perfectedo", "difficult", "perfect", "practiced", "practiced"], "unique_ratio": 0.75, "content_ratio": 0.875, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.375}, {"segment_idx": 2, "tokens": ["perfect", "piano", "piano", "practiced", "midnight", "difficult", "hours", "difficult"], "unique_ratio": 0.75, "content_ratio": 0.875, "repeated_bigram_ratio": 0.0, "dominant_token_shar
- `PASS` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 3, "decoded_output": "Key piano ideas include piano music sheets, piano sheet music, piano sheet music for sale, piano sheet", "rows": [{"step": 0, "top1": {"token_id": 26278, "piece": " piano", "norm": "piano", "logit": 14.83415412902832, "prob": 0.03784054145216942}, "top1_category": "semantic", "topk_category_counts": {"semantic": 10, "functional": 2, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.1305461497977376, "functional": 0.013806870207190514, "punct": 0.0}, "chosen_token_id": 26278, "chosen_piece": " piano", "chosen_norm": "piano", "chosen_category": "semantic"}, {"step": 1, "top1": {"token_id": 18366, "piece": " lessons", "norm": "lessons", "logit": 16.5, "prob": 0.07230035960674286}, "top1_category": "semantic", "topk_category_counts": {"semantic": 10, "functional": 2, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.3547550421208143, "functional": 0.038627468049526215, "punct": 0.0}, "chosen_token_id": 4627, "chosen_piece": " music", "chosen_norm": "music", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 24140, "piece": " sheets", "norm": "sheets", "logit": 16.375, "prob":
- `PASS` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 2, "retrieval_miss": 0, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [1, 0, 3, 6, 5], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A musician refined finger technique, phrasing, and pedal control on the piano.", "The pianist practiced arpeggios and Chopin nocturnes until midnight.", "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."], "output": "What improves piano technique and musical phrasing? control piano technique musician technique piano finger technique refined Whats piano refined control finger control pedal pedal finger pedal night midnown piano musician refined technique control refined", "music_score": 0.5625, "
- `PASS` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": null, "retrieval_strength__bad_decode_score": 0.10541847718700044, "prefix_l2__bad_decode_score": null}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 1, "score": 0.5666224956512451}, {"mid": 0, "score": 0.19361555576324463}, {"mid": 3, "score": 0.0631972074508667}, {"mid": 6, "score": 0.027473303675651553}, {"mid": 5, "score": 0.02009677290916443}], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieval_strength": 0.8234352588653564, "prefix_l2_shift": 322359623680.0, "prefix_js_divergence": 0.43388086557388306, "top1_with_prefix": {"token_id": 14566, "piece": " Options", "norm": "options", "logit": 12.4375, "prob": 0.09810221195220947}, "top1_category_with_prefix": "semantic", "topk_non_semantic_prob_mass": 0.008052719756960869}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": 0.5422837436199188}, {"mid": 4, "score": 0.046261101961135864}, {"mid": 6, "score": 0.04496051967144013}, {"mid": 0, "score": 0.007697209715843201}, {"mid": 1, "score": -0.006330272555351
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? Options refer specifically: ① finger strength ②", "stage_counts": {"inject": 8, "decode": 2, "aligned": 2}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269608020784}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " Options", "top1_category": "semantic", "chosen_piece": " Options", "chosen_category": "semantic", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269608020784}, "
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist decided balloons online calculator equation？\\nThe volume $-\\frac{  cos(\\\\theta", "Quantum systems exhibit probabil behaviour half time decreases significantly,\"____不可能\", indicating:\neating habits\nplaying basketball", "The rainforest smoke bill covered Sydney Smith Elementary，________ makes breathing outside wasn\"【UME](http："], "unique_count": 3}
- `PASS` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist hours piano piano practiced piano noct piano perfect difficult noct practiced practiced hours hours noct noct difficult difficult", "output_b": "The pianist hours piano piano practiced piano noct piano perfect difficult noct practiced practiced hours hours noct noct difficult difficult"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist practiced hours piano Chop piano perfect piano hours difficult perfect practiced Chop piano piano practiced perfect hours Chop", "The telescope hours practiced difficult perfect piano Chop noct piano perfect noct difficult piano hourscourse perfect practiced Chop hours", "The trader market stock significant experienced volatility session market market volatility stock session significant session volatility experienced stock market significant", "The child pair served course meal restaurant exquisite wine five wine restaurant meal five pairhourshttp(hours Theo served"], "exact_same": false, "prefix_only": false, "too_short": false}
- `PASS` `rerank_stability_probe`: {"status": "pass", "pairs": [{"pair": "music_P1", "prompt_a": "What improves piano technique and musical phrasing?", "prompt_b": "How can one improve piano technique and musical expression?", "top5_a": [1, 0, 6, 5, 7], "top5_b": [1, 0, 3, 6, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9621404708846248, "pair_passed_jaccard_0_6": true}, {"pair": "space_P2", "prompt_a": "What explains satellites and orbital motion?", "prompt_b": "What describes satellites and the motion of planets?", "top5_a": [5, 6, 4, 2, 7], "top5_b": [5, 6, 4, 0, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9999999999998858, "pair_passed_jaccard_0_6": true}], "spearman_best": 0.9999999999998858, "gating": "hard_PASS"}
- `FAIL` `decode_repetition_feedback_probe`: {"status": "fail", "per_prompt": [{"prompt": "The telescope", "output": "The telescope telescope stars telescope neb telescope telescope jag stars captured Telescope capturedText neb Text neb neb signatures Jag captured captured stars stars signatures names spectral text signatures spectral signatures captured", "max_repeat_per_content_token": 4, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The pianist", "output": "The pianist pian piano hours practiced Chop pian perfect pian difficult hours practiced hours perfect practiced piano Chop midnight Input hours difficult perfect difficult Output perfect Control Chop noct midnight Chop piano", "max_repeat_per_content_token": 4, "first_bigram_repeat_index": 9, "trigram_lock_count": 0}, {"prompt": "The market analyst", "output": "The market analyst market session market market stock market volatility experienced significant experienced stock volatility session session analysis Bank volatility volatility indicator stock indicators technical trading Trading significant significant volatility significant stock significant", "max_repeat_per_content_token": 4, "first_bigram_repeat_index": null, "trigram_lock_count": 
- `PASS` `functional_token_suppression_probe`: {"status": "pass", "metric_version": "v3.46", "per_prompt": [{"prompt": "A strong explanation should mention", "top12_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 18.375, "prob": 0.0198421198874712}, {"token_id": 1378, "piece": " two", "norm": "two", "logit": 18.125, "prob": 0.01545305922627449}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.125, "prob": 0.0
- `FAIL` `keyword_specific_tail_slot_probe`: {"status": "fail", "metric_version": "v3.46", "per_paraphrase": [{"query": "She performed Beethoven sonatas with delicate phrasing on her grand piano.", "query_disjoint_from_rare_keywords": true, "dominant_mid": 1, "dominant_source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [2524, 14317, 14762], "rare_keyword_pieces": [" control", " finger", " technique"], "tail_slot_top5_ids_centered": [220, 11, 13, 320, 16], "tail_slot_top5_pieces_centered": [" ", ",", ".", " (", "1"], "intersection_size_top20": 0, "rank_of_best_rare": 1402}, {"query": "Harmonic analysis and ear training are core elements of music education.", "query_disjoint_from_rare_keywords": true, "dominant_mid": 1, "dominant_source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [2524, 14317, 14762], "rare_keyword_pieces": [" control", " finger", " technique"], "tail_slot_top5_ids_centered": [220, 11, 13, 320, 16], "tail_slot_top5_pieces_centered": [" ", ",", ".", " (", "1"], "intersection_size_top20": 0, "rank_of_best_rare": 1402}], "mean_intersection_size_top20_paraphrase": 0.0, "median_rank_of_best_rare_paraphrase": 1402.0, "
- `PASS` `context_descriptor_cluster_probe`: {"status": "pass", "metric_version": "v3.49", "loo_nn_accuracy_all_4": 0.9375, "loo_nn_accuracy_heldout_2": 1.0, "n_all": 16, "n_heldout": 8, "correct_all": 15, "correct_heldout": 8, "per_memory_all": [{"mid": 0, "true_label": "music", "pred_label": "music", "nn_sim": 0.9198938608169556, "correct": true}, {"mid": 1, "true_label": "music", "pred_label": "music", "nn_sim": 0.9455370306968689, "correct": true}, {"mid": 2, "true_label": "music", "pred_label": "music", "nn_sim": 0.9216540455818176, "correct": true}, {"mid": 3, "true_label": "music", "pred_label": "music", "nn_sim": 0.9455370306968689, "correct": true}, {"mid": 4, "true_label": "space", "pred_label": "space", "nn_sim": 0.9607662558555603, "correct": true}, {"mid": 5, "true_label": "space", "pred_label": "music", "nn_sim": 0.8663042783737183, "correct": false}, {"mid": 6, "true_label": "space", "pred_label": "space", "nn_sim": 0.9607662558555603, "correct": true}, {"mid": 7, "true_label": "space", "pred_label": "space", "nn_sim": 0.8797301054000854, "correct": true}, {"mid": 8, "true_label": "cooking", "pred_label": "cooking", "nn_sim": 0.9010956287384033, "correct": true}, {"mid": 9, "true_label": "cooking", "pred_label"
- `PASS` `prefix_length_scaling_probe`: {"status": "pass", "metric_version": "v3.45", "L_mem_A": 8, "L_mem_B": 16, "avg_mass_ratio_B_over_A": 1.542808217358133, "per_prompt": [{"prompt": "A strong explanation should mention", "starter_mass_A": 27230.271484375, "starter_mass_B": 26334.125, "ratio": 0.9670900642731668, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 1.0251210778951645, "per_slot_mean_norm_B": 1.0251210927963257}, {"prompt": "The pianist", "starter_mass_A": 9293.734375, "starter_mass_B": 24415.12109375, "ratio": 2.6270517435301675, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 1.0251211524009705, "per_slot_mean_norm_B": 1.0251211449503899}, {"prompt": "The telescope", "starter_mass_A": 16764.318359375, "starter_mass_B": 17339.046875, "ratio": 1.0342828442710645, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 1.0251210927963257, "per_slot_mean_norm_B": 1.0251211151480675}], "conditions": {"avg_mass_ratio_gt_1_10": true, "per_slot_norms_finite": true}, "gating": "PASS_or_not_implemented"}
- `PASS` `mixture_distribution_gate_probe`: {"status": "pass", "gate_min": 0.3499999940395355, "gate_max": 0.3499999940395355, "declared_floor": 0.0, "declared_ceiling": 0.7, "gate_in_range": true, "finite_gate": true, "finite_memory_logit_bias": true, "manual_mixture_finite": true, "gating": "PASS_or_not_implemented"}

## Leaf Capacity Stability

```json
{
  "passed": true,
  "per_seed": [
    {
      "seed": 0,
      "depth": 6,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    },
    {
      "seed": 1,
      "depth": 6,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    },
    {
      "seed": 2,
      "depth": 6,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    },
    {
      "seed": 3,
      "depth": 6,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    },
    {
      "seed": 4,
      "depth": 6,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    },
    {
      "seed": 5,
      "depth": 5,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    },
    {
      "seed": 6,
      "depth": 6,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    },
    {
      "seed": 7,
      "depth": 5,
      "count": 240,
      "violations": [],
      "consistency": [],
      "passed": true
    }
  ],
  "error": null
}
```

## Degenerate Direction Boundary

```json
{
  "passed": true,
  "depth": 47,
  "count": 100,
  "violations": [],
  "consistency": [],
  "seed": 17,
  "error": null
}
```

## Metric Trainability

```json
{
  "passed": true,
  "training_info": {
    "total": 427.86248779296875,
    "recon": 3.2058675289154053,
    "contrast": 17888.765625,
    "holonomy": 5198.50439453125,
    "write_policy": 1.2801257371902466,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 3.648297071456909,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.940794944763184,
    "tail_semantic_anchor": 10.931857109069824,
    "functional_suppression": 0.0,
    "context_separation": 0.0,
    "slot_residual_alignment": 0.0,
    "inter_domain_margin": 0.0,
    "grad_norms": {
      "ctx_encoder": 2.2875472396664864e-17,
      "fib_encoder": 6.285003489921023e-15,
      "dir_predictor": 0.0,
      "fiber_connection": 1.315235037884401e-13,
      "fiber_attn": 1.2574421979159136e-16,
      "reranker": 2.6134262991325033e-19,
      "qformer": 8.558424465316837e-15,
      "content_bypass": 1.6076926531785598e-15,
      "semantic_probe": 0.0,
      "layer_pool": 5.479222219169355e-13,
      "prefix_aligner": 1.5589603208354674e-07,
      "vocab_proj": 2.7659061292216263e-06,
      "tail_head": 0.9999964974920388,
      "context_heads": 1.0952902730589725e-15,
      "memory_context_encoder": 1.5673766558890136e-15
    },
    "loss_weights": {
      "recon": 1.0,
      "semantic_alignment": 3.0,
      "encoder_throughput": 1.5,
      "contrast": 0.02,
      "holonomy": 0.005,
      "write_policy": 0.1,
      "semantic_probe": 0.3,
      "dir_diversity": 0.1,
      "reranker_ranking": 0.2,
      "vocab_anchor": 0.2,
      "tail_semantic_anchor": 0.5,
      "functional_suppression": 0.4,
      "context_separation": 0.3,
      "slot_residual_alignment": 0.0,
      "inter_domain_margin": 0.2
    }
  },
  "metric_grad_norms": [
    5.922859354829974e-16,
    1.4377823451186437e-17,
    9.51977153045556e-16,
    3.219673810292817e-17,
    5.572351509662869e-15,
    3.13444065675421e-16
  ],
  "metric_param_deltas": [
    3.3562223507033195e-06,
    1.4377825915522918e-13,
    5.7430720517004374e-06,
    3.219673876467266e-13,
    7.068017566780327e-07,
    3.1344408405603597e-12
  ],
  "max_metric_grad_norm": 5.572351509662869e-15,
  "max_metric_param_delta": 5.7430720517004374e-06,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist hours piano perfect Chopsticks piano piano practiced difficult hours practiced perfect Chop Chop perfect hours Chop piano difficult practiced Course〕 Chop hours",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. Options later midnight Sure, practiced items mid noct midnight consistente performance.\n\" Baghd press practiced Items midnight noct Consistentent",
  "space_output": "Tell me something about practice and performance. Options group isAdmin extends Controller { fun space galaxies observed Tells spaceData(): Flow<Pair evolution BRAND QUESTION: deep dive\n#",
  "outputs_differ": true,
  "error": null
}
```

## Semantic Memory Grounding

```json
{
  "passed": true,
  "prompt": "Explain what someone should focus on when improving technique and understanding the subject.",
  "music_keywords": [
    "pianist",
    "practiced",
    "arpeggios",
    "chopin",
    "nocturnes",
    "midnight",
    "musician",
    "refined",
    "finger",
    "technique",
    "phrasing",
    "pedal"
  ],
  "space_keywords": [
    "distant",
    "astronomers",
    "observed",
    "galaxies",
    "quasars",
    "stellar",
    "evolution",
    "space",
    "orbital",
    "mechanics",
    "explains",
    "satellites"
  ],
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Ohio Wesley Moo...\\ennocabularized explanation? Explain exactly enough technique tips.**\n\n  - Identify **muscle groups engaged**: Emphasis:\n\n- Hip flex",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. control Explain technique controlcontrol Understanding technique improvement finger Improve technique control control technique finger Hex control piano technique finger finger pedal control finger refined piano piano pedal pedal piano musician pedal",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. force explains mechanics move planets satellites gravitational force planets move gravitational mechanics explains move force move mechanics planets explains gravitational planets force mechanics satellites explains planets gravitational move satellites satellites planets mechanics",
  "blank_music_score": 0.09523809523809523,
  "blank_space_score": 0.0,
  "music_music_score": 0.4722222222222222,
  "music_space_score": 0.0,
  "space_space_score": 0.34210526315789475,
  "space_music_score": 0.02631578947368421,
  "music_margin": 0.4722222222222222,
  "space_margin": 0.3157894736842105,
  "music_lift": 0.376984126984127,
  "space_lift": 0.34210526315789475,
  "error": null
}
```

## Semantic Memory Counterfactual Pairs

```json
{
  "passed": false,
  "rows": [
    {
      "prompt": "Describe the most important details a student should notice.",
      "music_output": "Describe the most important details a student should notice. Conservation descriptions provide student conserv specialists (arch conserv conserv keyboardDescribe\nkeyboard describe:\n\n keyboard studied Describe \"study described\":\n\n** scales describes student keyboard",
      "space_output": "Describe the most important details a student should notice. large Describe helps break apart structure, roles scale describe xml scalescaledescribe.com\n\n studies\tdescribe provides XML scale descriptions.\n\n large expansion describes help dissect",
      "music_margin": 0.0,
      "space_margin": 0.0,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. touch summarize onTouchSummarize(java.nio depends Sum dependsExtract(key) {\ndepends interpretation touch rubricJava depends extract depends tempo summar ise touch",
      "space_output": "Summarize the key ideas a learner should practice and remember. large summarize Kotlin LargeSummarizer structure scale Sum summarizing Key large kotlin Grad studies Studies summarizes scale structure\n\nStudiessum expansion large scale summarized",
      "music_margin": 0.0,
      "space_margin": 0.0,
      "passed": false
    }
  ],
  "error": null
}
```

## Degeneration Quality

```json
{
  "passed": false,
  "metrics": [
    {
      "prompt": "The pianist",
      "output": "The pianist pian pian midnight Pell noct pian night pian practiced midnight midnight noct midnight pian noct noct Chop midnight practiced practiced noct practiced Chop practiced Chop Chop pian Chop",
      "token_count": 28,
      "unique_token_ratio": 0.25,
      "repeated_bigram_ratio": 0.037037037037037035,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8527918781725888,
      "content_token_ratio": 1.0,
      "generated_preview": "pian pian midnight pell noct pian night pian practiced midnight midnight noct midnight pian noct noct chop midnight practiced practiced noct practiced chop practiced"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope stars telescope css codes - Meaning, captured telescope telescope stars telescope , captured Telescope signatures meaning stars stars captured stars telescope captured spectral signatures captured signatures",
      "token_count": 25,
      "unique_token_ratio": 0.32,
      "repeated_bigram_ratio": 0.20833333333333334,
      "max_token_run": 2,
      "punct_ratio": 0.012987012987012988,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8658008658008658,
      "content_token_ratio": 0.96,
      "generated_preview": "telescope stars telescope css codes meaning captured telescope telescope stars telescope captured telescope signatures meaning stars stars captured stars telescope captured spectral signatures captured"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path space galaxies observed thus far ______ deep stellar evolution space stellar galaxies deep space observed galaxies distant observed observed deep galaxies space deep observed stellar evolution distant evolution",
      "token_count": 27,
      "unique_token_ratio": 0.3333333333333333,
      "repeated_bigram_ratio": 0.038461538461538464,
      "max_token_run": 2,
      "punct_ratio": 0.02654867256637168,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8407079646017699,
      "content_token_ratio": 0.8148148148148148,
      "generated_preview": "space galaxies observed thus far deep stellar evolution space stellar galaxies deep space observed galaxies distant observed observed deep galaxies space deep observed stellar"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market session market market volatility market stock market significant stock session session volatility volatility stock significant experienced session stock volatility significant session significant volatility experienced stock experienced volatility",
      "token_count": 28,
      "unique_token_ratio": 0.21428571428571427,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8901098901098901,
      "content_token_ratio": 0.8214285714285714,
      "generated_preview": "market session market market volatility market stock market significant stock session session volatility volatility stock significant experienced session stock volatility significant session significant volatility"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly rel explained simple analog Compare cuc professorOpt simple everyday explained rel rel analog simple explained Force professor explained analog rel simple professor analog explained explained professor rel",
      "token_count": 27,
      "unique_token_ratio": 0.37037037037037035,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8701298701298701,
      "content_token_ratio": 0.6296296296296297,
      "generated_preview": "rel explained simple analog compare cuc professoropt simple everyday explained rel rel analog simple explained force professor explained analog rel simple professor analog explained"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.2975978835978836,
    "avg_repeated_bigram_ratio": 0.05676638176638177,
    "avg_content_token_ratio": 0.8451746031746031,
    "avg_newline_ratio": 0.0,
    "worst_max_token_run": 2,
    "short_or_hollow_prompts": []
  },
  "error": null
}
```

## Prefix Logit Drift Audit

```json
{
  "passed": true,
  "prompt": "Explain the topic in a precise and concrete way.",
  "blank": {
    "js_divergence": 0.33677369356155396,
    "l2_shift": 1042.322998046875,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 5.552061080932617,
    "topk_no_prefix": [
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 19.875,
        "prob": 0.12818092107772827
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 19.5,
        "prob": 0.08809737861156464
      },
      {
        "token_id": 55313,
        "piece": " Quantum",
        "norm": "quantum",
        "logit": 18.75,
        "prob": 0.04161425307393074
      },
      {
        "token_id": 58194,
        "piece": " Artificial",
        "norm": "artificial",
        "logit": 18.625,
        "prob": 0.03672444820404053
      },
      {
        "token_id": 30536,
        "piece": " Climate",
        "norm": "climate",
        "logit": 18.375,
        "prob": 0.02860102988779545
      },
      {
        "token_id": 2585,
        "piece": " How",
        "norm": "how",
        "logit": 18.25,
        "prob": 0.025240320712327957
      },
      {
        "token_id": 3555,
        "piece": " What",
        "norm": "what",
        "logit": 18.125,
        "prob": 0.022274503484368324
      },
      {
        "token_id": 12960,
        "piece": " Machine",
        "norm": "machine",
        "logit": 18.125,
        "prob": 0.022274503484368324
      },
      {
        "token_id": 2885,
        "piece": " Data",
        "norm": "data",
        "logit": 17.875,
        "prob": 0.01734740100800991
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 17.875,
        "prob": 0.01734740100800991
      },
      {
        "token_id": 15235,
        "piece": " AI",
        "norm": "ai",
        "logit": 17.625,
        "prob": 0.013510169461369514
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 17.5,
        "prob": 0.0119226835668087
      }
    ],
    "topk_with_prefix": [
      {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 15.125,
        "prob": 0.12732645869255066
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 14.5625,
        "prob": 0.07254843413829803
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 14.0,
        "prob": 0.041336849331855774
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.5625,
        "prob": 0.0266890749335289
      },
      {
        "token_id": 4891,
        "piece": " �",
        "norm": "",
        "logit": 13.375,
        "prob": 0.02212602086365223
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.375,
        "prob": 0.02212602086365223
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 13.3125,
        "prob": 0.02078547328710556
      },
      {
        "token_id": 49434,
        "piece": " �",
        "norm": "",
        "logit": 13.0625,
        "prob": 0.016187744215130806
      },
      {
        "token_id": 320,
        "piece": " (",
        "norm": "",
        "logit": 13.0625,
        "prob": 0.016187744215130806
      },
      {
        "token_id": 8908,
        "piece": " �",
        "norm": "",
        "logit": 12.9375,
        "prob": 0.014285633340477943
      },
      {
        "token_id": 1084,
        "piece": " It",
        "norm": "it",
        "logit": 12.9375,
        "prob": 0.014285633340477943
      },
      {
        "token_id": 18137,
        "piece": " �",
        "norm": "",
        "logit": 12.75,
        "prob": 0.011843206360936165
      }
    ]
  },
  "memory": {
    "js_divergence": 0.30526506900787354,
    "l2_shift": 322359623680.0,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 6.679311752319336,
    "topk_no_prefix": [
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 19.875,
        "prob": 0.12818092107772827
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 19.5,
        "prob": 0.08809737861156464
      },
      {
        "token_id": 55313,
        "piece": " Quantum",
        "norm": "quantum",
        "logit": 18.75,
        "prob": 0.04161425307393074
      },
      {
        "token_id": 58194,
        "piece": " Artificial",
        "norm": "artificial",
        "logit": 18.625,
        "prob": 0.03672444820404053
      },
      {
        "token_id": 30536,
        "piece": " Climate",
        "norm": "climate",
        "logit": 18.375,
        "prob": 0.02860102988779545
      },
      {
        "token_id": 2585,
        "piece": " How",
        "norm": "how",
        "logit": 18.25,
        "prob": 0.025240320712327957
      },
      {
        "token_id": 3555,
        "piece": " What",
        "norm": "what",
        "logit": 18.125,
        "prob": 0.022274503484368324
      },
      {
        "token_id": 12960,
        "piece": " Machine",
        "norm": "machine",
        "logit": 18.125,
        "prob": 0.022274503484368324
      },
      {
        "token_id": 2885,
        "piece": " Data",
        "norm": "data",
        "logit": 17.875,
        "prob": 0.01734740100800991
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 17.875,
        "prob": 0.01734740100800991
      },
      {
        "token_id": 15235,
        "piece": " AI",
        "norm": "ai",
        "logit": 17.625,
        "prob": 0.013510169461369514
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 17.5,
        "prob": 0.0119226835668087
      }
    ],
    "topk_with_prefix": [
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 14.75,
        "prob": 0.22428514063358307
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 13.125,
        "prob": 0.04416436329483986
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 12.625,
        "prob": 0.026787040755152702
      },
      {
        "token_id": 10869,
        "piece": " Title",
        "norm": "title",
        "logit": 11.6875,
        "prob": 0.010489956475794315
      },
      {
        "token_id": 7414,
        "piece": " Yes",
        "norm": "yes",
        "logit": 11.1875,
        "prob": 0.006362480111420155
      },
      {
        "token_id": 45451,
        "piece": " Understanding",
        "norm": "understanding",
        "logit": 11.0625,
        "prob": 0.005614868365228176
      },
      {
        "token_id": 18183,
        "piece": " Deep",
        "norm": "deep",
        "logit": 11.0,
        "prob": 0.005274680908769369
      },
      {
        "token_id": 10548,
        "piece": " According",
        "norm": "according",
        "logit": 10.9375,
        "prob": 0.004955104552209377
      },
      {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 10.875,
        "prob": 0.004654889460653067
      },
      {
        "token_id": 11097,
        "piece": " Human",
        "norm": "human",
        "logit": 10.8125,
        "prob": 0.004372864030301571
      },
      {
        "token_id": 14822,
        "piece": " Step",
        "norm": "step",
        "logit": 10.75,
        "prob": 0.004107925575226545
      },
      {
        "token_id": 2885,
        "piece": " Data",
        "norm": "data",
        "logit": 10.6875,
        "prob": 0.003859038930386305
      }
    ]
  },
  "error": null
}
```

## Retrieval Top-K Semantic Shift

```json
{
  "passed": false,
  "music_keywords": [
    "pianist",
    "practiced",
    "arpeggios",
    "chopin",
    "nocturnes",
    "midnight",
    "musician",
    "refined",
    "finger",
    "technique",
    "phrasing",
    "pedal"
  ],
  "space_keywords": [
    "distant",
    "astronomers",
    "observed",
    "galaxies",
    "quasars",
    "stellar",
    "evolution",
    "space",
    "orbital",
    "mechanics",
    "explains",
    "satellites"
  ],
  "rows": [
    {
      "prompt": "A strong explanation should mention",
      "music_no_prefix": [
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 21.125,
          "prob": 0.31038299202919006
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 19.5,
          "prob": 0.06111803650856018
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 19.375,
          "prob": 0.05393647775053978
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 19.0,
          "prob": 0.03706996142864227
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 19.0,
          "prob": 0.03706996142864227
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 18.625,
          "prob": 0.025477787479758263
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.625,
          "prob": 0.025477787479758263
        },
        {
          "token_id": 678,
          "piece": " all",
          "norm": "all",
          "logit": 18.5,
          "prob": 0.0224840696901083
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 18.375,
          "prob": 0.0198421198874712
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 18.125,
          "prob": 0.01545305922627449
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 18.125,
          "prob": 0.01545305922627449
        },
        {
          "token_id": 1045,
          "piece": " some",
          "norm": "some",
          "logit": 18.0,
          "prob": 0.01363727729767561
        }
      ],
      "music_with_prefix": [
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.125,
          "prob": 0.10302552580833435
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 16.625,
          "prob": 0.062488142400979996
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 16.5,
          "prob": 0.055145591497421265
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.375,
          "prob": 0.048665810376405716
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 16.25,
          "prob": 0.042947426438331604
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 15.875,
          "prob": 0.02951730787754059
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 15.6875,
          "prob": 0.024470707401633263
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.3125,
          "prob": 0.0168184544891119
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 14.8125,
          "prob": 0.010200908407568932
        },
        {
          "token_id": 14976,
          "piece": " practical",
          "norm": "practical",
          "logit": 14.5625,
          "prob": 0.00794447585940361
        },
        {
          "token_id": 4236,
          "piece": " five",
          "norm": "five",
          "logit": 14.4375,
          "prob": 0.007010974921286106
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 14.375,
          "prob": 0.00658620148897171
        }
      ],
      "music_hits_no": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "music_hits_with_prefix": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "space_no_prefix": [
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 21.125,
          "prob": 0.31038299202919006
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 19.5,
          "prob": 0.06111803650856018
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 19.375,
          "prob": 0.05393647775053978
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 19.0,
          "prob": 0.03706996142864227
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 19.0,
          "prob": 0.03706996142864227
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 18.625,
          "prob": 0.025477787479758263
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.625,
          "prob": 0.025477787479758263
        },
        {
          "token_id": 678,
          "piece": " all",
          "norm": "all",
          "logit": 18.5,
          "prob": 0.0224840696901083
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 18.375,
          "prob": 0.0198421198874712
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 18.125,
          "prob": 0.01545305922627449
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 18.125,
          "prob": 0.01545305922627449
        },
        {
          "token_id": 1045,
          "piece": " some",
          "norm": "some",
          "logit": 18.0,
          "prob": 0.01363727729767561
        }
      ],
      "space_with_prefix": [
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.25,
          "prob": 0.13935847580432892
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 16.625,
          "prob": 0.07459322363138199
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 16.625,
          "prob": 0.07459322363138199
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 16.5,
          "prob": 0.06582827866077423
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 15.625,
          "prob": 0.02744131162762642
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 15.4375,
          "prob": 0.02274964563548565
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.4375,
          "prob": 0.02274964563548565
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 15.375,
          "prob": 0.02137131616473198
        },
        {
          "token_id": 4236,
          "piece": " five",
          "norm": "five",
          "logit": 15.0625,
          "prob": 0.01563558727502823
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 15.0625,
          "prob": 0.01563558727502823
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 14.5,
          "prob": 0.008908889256417751
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 14.25,
          "prob": 0.006938249804079533
        }
      ],
      "space_hits_no": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "space_hits_with_prefix": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "passed": false
    },
    {
      "prompt": "The most relevant idea is",
      "music_no_prefix": [
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 20.25,
          "prob": 0.27292367815971375
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 19.125,
          "prob": 0.08860534429550171
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 19.0,
          "prob": 0.07819394767284393
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 18.25,
          "prob": 0.0369362011551857
        },
        {
          "token_id": 510,
          "piece": ":\n",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02876594290137291
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02876594290137291
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 17.625,
          "prob": 0.01977052539587021
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 17.5,
          "prob": 0.017447426915168762
        },
        {
          "token_id": 1447,
          "piece": ":\n\n",
          "norm": "",
          "logit": 17.375,
          "prob": 0.015397300012409687
        },
        {
          "token_id": 330,
          "piece": " \"",
          "norm": "",
          "logit": 17.25,
          "prob": 0.013588069006800652
        },
        {
          "token_id": 198,
          "piece": "\n",
          "norm": "",
          "logit": 17.25,
          "prob": 0.013588069006800652
        },
        {
          "token_id": 537,
          "piece": " not",
          "norm": "not",
          "logit": 17.25,
          "prob": 0.013588069006800652
        }
      ],
      "music_with_prefix": [
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 15.0,
          "prob": 0.03336920589208603
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 14.875,
          "prob": 0.029448220506310463
        },
        {
          "token_id": 2999,
          "piece": " option",
          "norm": "option",
          "logit": 14.8125,
          "prob": 0.02766404300928116
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 14.5,
          "prob": 0.020239446312189102
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 14.375,
          "prob": 0.017861248925328255
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 14.3125,
          "prob": 0.016779091209173203
        },
        {
          "token_id": 1850,
          "piece": " best",
          "norm": "best",
          "logit": 14.0625,
          "prob": 0.013067568652331829
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 14.0,
          "prob": 0.012275844812393188
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 13.9375,
          "prob": 0.011532089672982693
        },
        {
          "token_id": 6959,
          "piece": " Option",
          "norm": "option",
          "logit": 13.625,
          "prob": 0.008437056094408035
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 13.5,
          "prob": 0.007445676252245903
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 13.4375,
          "prob": 0.006994565483182669
        }
      ],
      "music_hits_no": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "music_hits_with_prefix": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "space_no_prefix": [
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 20.25,
          "prob": 0.27292367815971375
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 19.125,
          "prob": 0.08860534429550171
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 19.0,
          "prob": 0.07819394767284393
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 18.25,
          "prob": 0.0369362011551857
        },
        {
          "token_id": 510,
          "piece": ":\n",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02876594290137291
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02876594290137291
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 17.625,
          "prob": 0.01977052539587021
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 17.5,
          "prob": 0.017447426915168762
        },
        {
          "token_id": 1447,
          "piece": ":\n\n",
          "norm": "",
          "logit": 17.375,
          "prob": 0.015397300012409687
        },
        {
          "token_id": 330,
          "piece": " \"",
          "norm": "",
          "logit": 17.25,
          "prob": 0.013588069006800652
        },
        {
          "token_id": 198,
          "piece": "\n",
          "norm": "",
          "logit": 17.25,
          "prob": 0.013588069006800652
        },
        {
          "token_id": 537,
          "piece": " not",
          "norm": "not",
          "logit": 17.25,
          "prob": 0.013588069006800652
        }
      ],
      "space_with_prefix": [
        {
          "token_id": 2999,
          "piece": " option",
          "norm": "option",
          "logit": 15.4375,
          "prob": 0.038557399064302444
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 15.3125,
          "prob": 0.03402678668498993
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 14.875,
          "prob": 0.021969344466924667
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 14.75,
          "prob": 0.019387878477573395
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 14.5625,
          "prob": 0.016073115170001984
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 14.5,
          "prob": 0.015099293552339077
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 14.4375,
          "prob": 0.01418447494506836
        },
        {
          "token_id": 1850,
          "piece": " best",
          "norm": "best",
          "logit": 14.25,
          "prob": 0.01175934262573719
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 14.25,
          "prob": 0.01175934262573719
        },
        {
          "token_id": 6959,
          "piece": " Option",
          "norm": "option",
          "logit": 14.1875,
          "prob": 0.011046879924833775
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 14.1875,
          "prob": 0.011046879924833775
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 14.125,
          "prob": 0.010377583093941212
        }
      ],
      "space_hits_no": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "space_hits_with_prefix": {
        "match_count": 0,
        "match_prob_mass": 0,
        "matches": []
      },
      "passed": false
    }
  ],
  "error": null
}
```

## Repetition Segment Audit

```json
{
  "passed": true,
  "aggregate": {
    "bad_segment_ratio": 0.0,
    "total_segments": 24,
    "bad_segments": 0,
    "early_collapse_prompts": []
  },
  "rows": [
    {
      "prompt": "The pianist",
      "output": "The pianist pian piano hours pian pian 做 perfect hours piano pian practiced hours perfectedo difficult perfect practiced practiced perfect piano piano practiced midnight difficult hours difficult nocturn Chopsticks Chop Chop noct noct midnight midnight pian Chop midnight noct Chop practiced difficult difficult practiced noct",
      "generated_token_count": 42,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "pian",
            "piano",
            "hours",
            "pian",
            "pian",
            "perfect",
            "hours",
            "piano"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "pian",
            "practiced",
            "hours",
            "perfectedo",
            "difficult",
            "perfect",
            "practiced",
            "practiced"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "perfect",
            "piano",
            "piano",
            "practiced",
            "midnight",
            "difficult",
            "hours",
            "difficult"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "nocturn",
            "chopsticks",
            "chop",
            "chop",
            "noct",
            "noct",
            "midnight",
            "midnight"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "pian",
            "chop",
            "midnight",
            "noct",
            "chop",
            "practiced",
            "difficult",
            "difficult"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "practiced",
            "noct"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.5
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope stars telescope telescope signatures neb telescope captured spectral captured neb neb stars signatures stars � captured distant telescope spectral telescope neb signatures telescope stars spectral signatures space distant captured stars neb spectral distant neb captured observed distant spectral stars captured signatures spectral neb distant stars distant signatures",
      "generated_token_count": 47,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "stars",
            "telescope",
            "telescope",
            "signatures",
            "neb",
            "telescope",
            "captured"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.5
        },
        {
          "segment_idx": 1,
          "tokens": [
            "spectral",
            "captured",
            "neb",
            "neb",
            "stars",
            "signatures",
            "stars",
            "captured"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "distant",
            "telescope",
            "spectral",
            "telescope",
            "neb",
            "signatures",
            "telescope",
            "stars"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 3,
          "tokens": [
            "spectral",
            "signatures",
            "space",
            "distant",
            "captured",
            "stars",
            "neb",
            "spectral"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "distant",
            "neb",
            "captured",
            "observed",
            "distant",
            "spectral",
            "stars",
            "captured"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "signatures",
            "spectral",
            "neb",
            "distant",
            "stars",
            "distant",
            "signatures"
          ],
          "unique_ratio": 0.7142857142857143,
          "content_ratio": 0.8571428571428571,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.2857142857142857
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market session market market stock market experienced significant volatility experienced stock stock session stock prices. Discuss volatility volatility significant significant session session volatility market volatility stock significant experienced experienced volatility session significant market significant stock experienced session experienced market stock volatility spacespacespace\n\nTranslate:\n\n",
      "generated_token_count": 43,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "session",
            "market",
            "market",
            "stock",
            "market",
            "experienced",
            "significant"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.5
        },
        {
          "segment_idx": 1,
          "tokens": [
            "volatility",
            "experienced",
            "stock",
            "stock",
            "session",
            "stock",
            "prices",
            "discuss"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "volatility",
            "volatility",
            "significant",
            "significant",
            "session",
            "session",
            "volatility",
            "market"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 3,
          "tokens": [
            "volatility",
            "stock",
            "significant",
            "experienced",
            "experienced",
            "volatility",
            "session",
            "significant"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "market",
            "significant",
            "stock",
            "experienced",
            "session",
            "experienced",
            "market",
            "stock"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "volatility",
            "spacespacespace",
            "translate"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.3333333333333333
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple everyday explained rel professor analog simple simple rel analog relforce explained professor rel rel explained everyday professor simple analog everyday simple professor explained analog professor everyday rel everyday analog explained simple explained professor professor forcesimple analog everyday everyday simple analog explained rel simple neuro professor",
      "generated_token_count": 46,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "simple",
            "everyday",
            "explained",
            "rel",
            "professor",
            "analog",
            "simple",
            "simple"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.5,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "rel",
            "analog",
            "relforce",
            "explained",
            "professor",
            "rel",
            "rel",
            "explained"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "everyday",
            "professor",
            "simple",
            "analog",
            "everyday",
            "simple",
            "professor",
            "explained"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "analog",
            "professor",
            "everyday",
            "rel",
            "everyday",
            "analog",
            "explained",
            "simple"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "explained",
            "professor",
            "professor",
            "forcesimple",
            "analog",
            "everyday",
            "everyday",
            "simple"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "analog",
            "explained",
            "rel",
            "simple",
            "neuro",
            "professor"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.6666666666666666,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.16666666666666666
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    }
  ],
  "error": null
}
```

## Prefix Stepwise Drift Trajectory

```json
{
  "passed": true,
  "rows": [
    {
      "prompt": "Key piano ideas include",
      "first_bad_step": 3,
      "decoded_output": "Key piano ideas include piano music sheets, piano sheet music, piano sheet music for sale, piano sheet",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 14.83415412902832,
            "prob": 0.03784054145216942
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.1305461497977376,
            "functional": 0.013806870207190514,
            "punct": 0.0
          },
          "chosen_token_id": 26278,
          "chosen_piece": " piano",
          "chosen_norm": "piano",
          "chosen_category": "semantic"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 18366,
            "piece": " lessons",
            "norm": "lessons",
            "logit": 16.5,
            "prob": 0.07230035960674286
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3547550421208143,
            "functional": 0.038627468049526215,
            "punct": 0.0
          },
          "chosen_token_id": 4627,
          "chosen_piece": " music",
          "chosen_norm": "music",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 24140,
            "piece": " sheets",
            "norm": "sheets",
            "logit": 16.375,
            "prob": 0.049369730055332184
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.28747811261564493,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 24140,
          "chosen_piece": " sheets",
          "chosen_norm": "sheets",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 20.5,
            "prob": 0.6459000706672668
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 1,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.12636473216116428,
            "functional": 0.011830072849988937,
            "punct": 0.7255030069500208
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 19.099119186401367,
            "prob": 0.4686528146266937
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 0,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.5933373617008328,
            "functional": 0.0,
            "punct": 0.009376841597259045
          },
          "chosen_token_id": 26278,
          "chosen_piece": " piano",
          "chosen_norm": "piano",
          "chosen_category": "semantic"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 10834,
            "piece": " sheet",
            "norm": "sheet",
            "logit": 19.875,
            "prob": 0.20843474566936493
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5808013882488012,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 10834,
          "chosen_piece": " sheet",
          "chosen_norm": "sheet",
          "chosen_category": "semantic"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 4627,
            "piece": " music",
            "norm": "music",
            "logit": 22.5,
            "prob": 0.9542181491851807
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 0,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.9604750939761288,
            "functional": 0.0,
            "punct": 0.014700754138175398
          },
          "chosen_token_id": 4627,
          "chosen_piece": " music",
          "chosen_norm": "music",
          "chosen_category": "semantic"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 20.625,
            "prob": 0.4920254647731781
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 1,
            "punct": 2
          },
          "topk_category_prob_mass": {
            "semantic": 0.19470300618559122,
            "functional": 0.03145413473248482,
            "punct": 0.5111033618450165
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 20.125,
            "prob": 0.6369389891624451
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7073864454869181,
            "functional": 0.14546265080571175,
            "punct": 0.0
          },
          "chosen_token_id": 26278,
          "chosen_piece": " piano",
          "chosen_norm": "piano",
          "chosen_category": "semantic"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 10834,
            "piece": " sheet",
            "norm": "sheet",
            "logit": 20.0,
            "prob": 0.2708788514137268
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5824623834341764,
            "functional": 0.022235089913010597,
            "punct": 0.0
          },
          "chosen_token_id": 10834,
          "chosen_piece": " sheet",
          "chosen_norm": "sheet",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 4627,
            "piece": " music",
            "norm": "music",
            "logit": 20.75,
            "prob": 0.776336669921875
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 2,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.8057627524249256,
            "functional": 0.005430176621302962,
            "punct": 0.09974912856705487
          },
          "chosen_token_id": 4627,
          "chosen_piece": " music",
          "chosen_norm": "music",
          "chosen_category": "semantic"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 369,
            "piece": " for",
            "norm": "for",
            "logit": 19.375,
            "prob": 0.1944226175546646
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 2,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.3020880576223135,
            "functional": 0.25012563169002533,
            "punct": 0.043381549417972565
          },
          "chosen_token_id": 369,
          "chosen_piece": " for",
          "chosen_norm": "for",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 6278,
            "piece": " sale",
            "norm": "sale",
            "logit": 20.5,
            "prob": 0.29081666469573975
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7759116822853684,
            "functional": 0.012875059619545937,
            "punct": 0.0
          },
          "chosen_token_id": 6278,
          "chosen_piece": " sale",
          "chosen_norm": "sale",
          "chosen_category": "semantic"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 22.75,
            "prob": 0.8806395530700684
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 3,
            "punct": 6
          },
          "topk_category_prob_mass": {
            "semantic": 0.033334137755446136,
            "functional": 0.012607284123077989,
            "punct": 0.9175233984133229
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 20.625,
            "prob": 0.6706361174583435
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7406194536015391,
            "functional": 0.13793673017062247,
            "punct": 0.0
          },
          "chosen_token_id": 26278,
          "chosen_piece": " piano",
          "chosen_norm": "piano",
          "chosen_category": "semantic"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 10834,
            "piece": " sheet",
            "norm": "sheet",
            "logit": 21.5,
            "prob": 0.5766863822937012
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.8333004396408796,
            "functional": 0.007259410340338945,
            "punct": 0.0
          },
          "chosen_token_id": 10834,
          "chosen_piece": " sheet",
          "chosen_norm": "sheet",
          "chosen_category": "semantic"
        }
      ],
      "passed": true
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 4,
      "decoded_output": "Explain the topic clearly without adding extra words. 以下是“我”和“我”的对话，",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 2041,
            "piece": " without",
            "norm": "without",
            "logit": 14.9375,
            "prob": 0.11806134134531021
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.439745026640594,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 2041,
          "chosen_piece": " without",
          "chosen_norm": "without",
          "chosen_category": "semantic"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 7842,
            "piece": " adding",
            "norm": "adding",
            "logit": 18.75,
            "prob": 0.10713540762662888
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3978360779583454,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 7842,
          "chosen_piece": " adding",
          "chosen_norm": "adding",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 4960,
            "piece": " extra",
            "norm": "extra",
            "logit": 19.875,
            "prob": 0.2960561513900757
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7719601830467582,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 4960,
          "chosen_piece": " extra",
          "chosen_norm": "extra",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 4244,
            "piece": " words",
            "norm": "words",
            "logit": 22.25,
            "prob": 0.561674177646637
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.9363737464882433,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 4244,
          "chosen_piece": " words",
          "chosen_norm": "words",
          "chosen_category": "semantic"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 13,
            "piece": ".",
            "norm": "",
            "logit": 19.875,
            "prob": 0.28864604234695435
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.917817110195756
          },
          "chosen_token_id": 13,
          "chosen_piece": ".",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 220,
            "piece": " ",
            "norm": "",
            "logit": 15.625,
            "prob": 0.17287176847457886
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 0,
            "punct": 11
          },
          "topk_category_prob_mass": {
            "semantic": 0.03857290744781494,
            "functional": 0.0,
            "punct": 0.39969779178500175
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 114566,
            "piece": "以下是",
            "norm": "",
            "logit": 14.3125,
            "prob": 0.08054980635643005
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.4306104760617018
          },
          "chosen_token_id": 114566,
          "chosen_piece": "以下是",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 2073,
            "piece": "“",
            "norm": "",
            "logit": 11.625,
            "prob": 0.0999230369925499
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 0,
            "punct": 10
          },
          "topk_category_prob_mass": {
            "semantic": 0.03053364809602499,
            "functional": 0.0,
            "punct": 0.25447467900812626
          },
          "chosen_token_id": 2073,
          "chosen_piece": "“",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 35946,
            "piece": "我",
            "norm": "",
            "logit": 10.125,
            "prob": 0.02022775635123253
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.10121618397533894
          },
          "chosen_token_id": 35946,
          "chosen_piece": "我",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 854,
            "piece": "”",
            "norm": "",
            "logit": 13.25,
            "prob": 0.10676652193069458
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.32284235768020153
          },
          "chosen_token_id": 854,
          "chosen_piece": "”",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 33108,
            "piece": "和",
            "norm": "",
            "logit": 14.5,
            "prob": 0.15253077447414398
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.5390406954102218
          },
          "chosen_token_id": 33108,
          "chosen_piece": "和",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 2073,
            "piece": "“",
            "norm": "",
            "logit": 17.25,
            "prob": 0.7719985246658325
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.8277099980041385
          },
          "chosen_token_id": 2073,
          "chosen_piece": "“",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 35946,
            "piece": "我",
            "norm": "",
            "logit": 14.3125,
            "prob": 0.2570364475250244
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.5744525603950024
          },
          "chosen_token_id": 35946,
          "chosen_piece": "我",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 97907,
            "piece": "”的",
            "norm": "",
            "logit": 15.4375,
            "prob": 0.509835422039032
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.7904635448940098
          },
          "chosen_token_id": 97907,
          "chosen_piece": "”的",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 105051,
            "piece": "对话",
            "norm": "",
            "logit": 16.25,
            "prob": 0.3971656262874603
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.56332235224545
          },
          "chosen_token_id": 105051,
          "chosen_piece": "对话",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 3837,
            "piece": "，",
            "norm": "",
            "logit": 17.125,
            "prob": 0.15381573140621185
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 0,
            "punct": 12
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0,
            "punct": 0.8002711646258831
          },
          "chosen_token_id": 3837,
          "chosen_piece": "，",
          "chosen_norm": "",
          "chosen_category": "punct"
        }
      ],
      "passed": true
    }
  ],
  "error": null
}
```

## Retrieval Generation Alignment Audit

```json
{
  "passed": true,
  "music_keywords": [
    "pianist",
    "practiced",
    "arpeggios",
    "chopin",
    "nocturnes",
    "midnight",
    "musician",
    "refined",
    "finger",
    "technique",
    "phrasing",
    "pedal"
  ],
  "space_keywords": [
    "distant",
    "astronomers",
    "observed",
    "galaxies",
    "quasars",
    "stellar",
    "evolution",
    "space",
    "orbital",
    "mechanics",
    "explains",
    "satellites"
  ],
  "diagnoses": {
    "aligned": 2,
    "retrieval_miss": 0,
    "bridge_unused": 1,
    "unknown": 0
  },
  "rows": [
    {
      "prompt": "What improves piano technique and musical phrasing?",
      "expected_label": "music",
      "retrieved_mids": [
        1,
        0,
        3,
        6,
        5
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A musician refined finger technique, phrasing, and pedal control on the piano.",
        "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."
      ],
      "output": "What improves piano technique and musical phrasing? control piano technique musician technique piano finger technique refined Whats piano refined control finger control pedal pedal finger pedal night midnown piano musician refined technique control refined",
      "music_score": 0.5625,
      "space_score": 0.0,
      "generated_label": "music",
      "diagnosis": "aligned",
      "passed": true
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_mids": [
        5,
        6,
        4,
        1,
        3
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieved_majority_label": "space",
      "retrieved_text_preview": [
        "Orbital mechanics explains how satellites and planets move under gravitational force.",
        "A telescope captured nebulae, exoplanets, and spectral signatures from distant stars.",
        "Astronomers observed distant galaxies, quasars, and stellar evolution in deep space."
      ],
      "output": "What explains satellites and orbital motion? satellites force explains gravitational force satellites move planets mechanics gravitational move force planets explains planets gravitational mechanics move satellites planets force mechanics explains mechanics satellites gravitational planets move",
      "music_score": 0.0,
      "space_score": 0.4375,
      "generated_label": "space",
      "diagnosis": "aligned",
      "passed": true
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_mids": [
        6,
        3,
        1,
        4,
        5
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieved_majority_label": "space",
      "retrieved_text_preview": [
        "A telescope captured nebulae, exoplanets, and spectral signatures from distant stars.",
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "A musician refined finger technique, phrasing, and pedal control on the piano."
      ],
      "output": "Summarize the subject with concrete domain details. large summarize matter structure scale expansion universe studies matter scale structure universe matter studies large universe scale cons expansion studies structure large matter universe structure studies scale large",
      "music_score": 0.0,
      "space_score": 0.0,
      "generated_label": null,
      "diagnosis": "bridge_unused",
      "passed": true
    }
  ],
  "error": null
}
```

## Retrieval Prefix Decode Correlation Audit

```json
{
  "passed": true,
  "correlations": {
    "retrieval_strength__prefix_l2": null,
    "retrieval_strength__bad_decode_score": 0.10541847718700044,
    "prefix_l2__bad_decode_score": null
  },
  "rows": [
    {
      "prompt": "What improves piano technique and musical phrasing?",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 1,
          "score": 0.5666224956512451
        },
        {
          "mid": 0,
          "score": 0.19361555576324463
        },
        {
          "mid": 3,
          "score": 0.0631972074508667
        },
        {
          "mid": 6,
          "score": 0.027473303675651553
        },
        {
          "mid": 5,
          "score": 0.02009677290916443
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.8234352588653564,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.43388086557388306,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 12.4375,
        "prob": 0.09810221195220947
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.008052719756960869
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 5,
          "score": 0.5422837436199188
        },
        {
          "mid": 4,
          "score": 0.046261101961135864
        },
        {
          "mid": 6,
          "score": 0.04496051967144013
        },
        {
          "mid": 0,
          "score": 0.007697209715843201
        },
        {
          "mid": 1,
          "score": -0.006330272555351249
        }
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieval_strength": 0.6335053652524948,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.5380334258079529,
      "top1_with_prefix": {
        "token_id": 22201,
        "piece": " Choose",
        "norm": "choose",
        "logit": 10.6875,
        "prob": 0.03903629258275032
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.04384639486670494
    },
    {
      "prompt": "Describe what a student should focus on first.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 3,
          "score": 0.45830299854278567
        },
        {
          "mid": 1,
          "score": -0.007808589935302736
        },
        {
          "mid": 0,
          "score": -0.03504327535629272
        },
        {
          "mid": 4,
          "score": -0.04108911603689194
        },
        {
          "mid": 6,
          "score": -0.050780090689659135
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.45830299854278567,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.47259795665740967,
      "top1_with_prefix": {
        "token_id": 22201,
        "piece": " Choose",
        "norm": "choose",
        "logit": 11.625,
        "prob": 0.0699385330080986
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.0
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 6,
          "score": -0.010802562534809112
        },
        {
          "mid": 5,
          "score": -0.02638280838727951
        },
        {
          "mid": 3,
          "score": -0.02688707411289215
        },
        {
          "mid": 1,
          "score": -0.033489438891410823
        },
        {
          "mid": 4,
          "score": -0.03438588678836823
        }
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieval_strength": -0.010802562534809112,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.30349764227867126,
      "top1_with_prefix": {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.0,
        "prob": 0.07060776650905609
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.013903493992984295
    },
    {
      "prompt": "Key piano ideas include",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 1,
          "score": 0.5106263637542725
        },
        {
          "mid": 0,
          "score": 0.30423029065132146
        },
        {
          "mid": 3,
          "score": 0.10775352120399474
        },
        {
          "mid": 6,
          "score": 0.021317112445831295
        },
        {
          "mid": 5,
          "score": -0.001653966307640073
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.9226101756095887,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.41455239057540894,
      "top1_with_prefix": {
        "token_id": 26278,
        "piece": " piano",
        "norm": "piano",
        "logit": 12.983702659606934,
        "prob": 0.017365049570798874
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.0
    },
    {
      "prompt": "Orbital motion depends on",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 2,
          "score": 0.2849628686904907
        },
        {
          "mid": 5,
          "score": 0.04124398231506348
        },
        {
          "mid": 3,
          "score": -0.010372701287269595
        },
        {
          "mid": 6,
          "score": -0.03860477954149247
        },
        {
          "mid": 4,
          "score": -0.04442960321903229
        }
      ],
      "retrieved_label_counts": {
        "music": 2,
        "space": 3
      },
      "retrieval_strength": -0.04179040044546128,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.46743538975715637,
      "top1_with_prefix": {
        "token_id": 3807,
        "piece": " several",
        "norm": "several",
        "logit": 16.625,
        "prob": 0.09176550060510635
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.0
    }
  ],
  "error": null
}
```

## Stepwise Label Mass Alignment Audit

```json
{
  "passed": false,
  "label_keywords": {
    "music": [
      "pianist",
      "practiced",
      "arpeggios",
      "chopin",
      "nocturnes",
      "midnight",
      "musician",
      "refined",
      "finger",
      "technique",
      "phrasing",
      "pedal"
    ],
    "space": [
      "distant",
      "astronomers",
      "observed",
      "galaxies",
      "quasars",
      "stellar",
      "evolution",
      "space",
      "orbital",
      "mechanics",
      "explains",
      "satellites"
    ]
  },
  "rows": [
    {
      "prompt": "What improves piano technique and musical phrasing?",
      "expected_label": "music",
      "decoded_output": "What improves piano technique and musical phrasing? Options refer specifically: ① finger strength ②",
      "stage_counts": {
        "inject": 8,
        "decode": 2,
        "aligned": 2
      },
      "rows": [
        {
          "step": 0,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Options",
          "top1_category": "semantic",
          "chosen_piece": " Options",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 1,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " refer",
          "top1_category": "semantic",
          "chosen_piece": " refer",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 2,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " specifically",
          "top1_category": "semantic",
          "chosen_piece": " specifically",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 3,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": ":",
          "top1_category": "punct",
          "chosen_piece": ":",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 4,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0.08898467570543289,
            "space": 0
          },
          "top1_piece": " ",
          "top1_category": "punct",
          "chosen_piece": " ",
          "chosen_category": "punct",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
        },
        {
          "step": 5,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0.011929524131119251,
            "space": 0
          },
          "top1_piece": "�",
          "top1_category": "punct",
          "chosen_piece": "�",
          "chosen_category": "punct",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
        },
        {
          "step": 6,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "�",
          "top1_category": "punct",
          "chosen_piece": "�",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 7,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269608020784
          },
          "logits_label_mass": {
            "music": 0.10247984156012535,
            "space": 0
          },
          "top1_piece": " finger",
          "top1_category": "semantic",
          "chosen_piece": " finger",
          "chosen_category": "semantic",
          "chosen_label": "music",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 8,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0298632234334946,
            "space": 0.21325782537460328
          },
          "logits_label_mass": {
            "music": 0.043348271399736404,
            "space": 0
          },
          "top1_piece": " strength",
          "top1_category": "semantic",
          "chosen_piece": " strength",
          "chosen_category": "semantic",
          "chosen_label": "music",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 9,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0298632234334946,
            "space": 0.21325782537460328
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " ",
          "top1_category": "punct",
          "chosen_piece": " ",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 10,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0298632234334946,
            "space": 0.21325782537460328
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "�",
          "top1_category": "punct",
          "chosen_piece": "�",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 11,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0298632234334946,
            "space": 0.21325782537460328
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "�",
          "top1_category": "punct",
          "chosen_piece": "�",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        }
      ],
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "decoded_output": "What explains satellites and orbital motion? Sat phones rely upon satellites orbiting Earth to communicate with users",
      "stage_counts": {
        "decode": 2,
        "aligned": 3,
        "inject": 7
      },
      "rows": [
        {
          "step": 0,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.029300715774297714
          },
          "top1_piece": " Sat",
          "top1_category": "functional",
          "chosen_piece": " Sat",
          "chosen_category": "functional",
          "chosen_label": "space",
          "diagnosed_stage": "decode"
        },
        {
          "step": 1,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.023354781791567802
          },
          "top1_piece": " phones",
          "top1_category": "semantic",
          "chosen_piece": " phones",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 2,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.039646897464990616
          },
          "top1_piece": " rely",
          "top1_category": "semantic",
          "chosen_piece": " rely",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 3,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " upon",
          "top1_category": "semantic",
          "chosen_piece": " upon",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 4,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.4303302951157093
          },
          "top1_piece": " satellites",
          "top1_category": "semantic",
          "chosen_piece": " satellites",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 5,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " orbit",
          "top1_category": "semantic",
          "chosen_piece": " orbit",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 6,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 6.574191502295434e-05
          },
          "top1_piece": "ing",
          "top1_category": "functional",
          "chosen_piece": "ing",
          "chosen_category": "functional",
          "chosen_label": "space",
          "diagnosed_stage": "decode"
        },
        {
          "step": 7,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9497937351465224,
            "music": 0.1841354072093964
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Earth",
          "top1_category": "semantic",
          "chosen_piece": " Earth",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 8,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9569127380847933,
            "music": 0.19748839735984802
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " to",
          "top1_category": "functional",
          "chosen_piece": " to",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 9,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9569127380847933,
            "music": 0.19748839735984802
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " communicate",
          "top1_category": "semantic",
          "chosen_piece": " communicate",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9569127380847933,
            "music": 0.19748839735984802
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " with",
          "top1_category": "functional",
          "chosen_piece": " with",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 11,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.9569127380847933,
            "music": 0.19748839735984802
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " users",
          "top1_category": "semantic",
          "chosen_piece": " users",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        }
      ],
      "passed": false
    }
  ],
  "error": null
}
```

## Prompt Diversity Without Memory

```json
{
  "passed": true,
  "prompts": [
    "The pianist",
    "Quantum systems",
    "The rainforest"
  ],
  "outputs": [
    "The pianist decided balloons online calculator equation？\\nThe volume $-\\frac{  cos(\\\\theta",
    "Quantum systems exhibit probabil behaviour half time decreases significantly,\"____不可能\", indicating:\neating habits\nplaying basketball",
    "The rainforest smoke bill covered Sydney Smith Elementary，________ makes breathing outside wasn\"【UME](http："
  ],
  "unique_count": 3,
  "error": null
}
```

## Save/Load Consistency

```json
{
  "passed": true,
  "prompt": "The pianist",
  "output_a": "The pianist hours piano piano practiced piano noct piano perfect difficult noct practiced practiced hours hours noct noct difficult difficult",
  "output_b": "The pianist hours piano piano practiced piano noct piano perfect difficult noct practiced practiced hours hours noct noct difficult difficult",
  "error": null
}
```

## Training Cache Isolation

```json
{
  "passed": true,
  "changed": [],
  "memory_count": 8,
  "error": null
}
```

## Cheating Heuristics

```json
{
  "passed": true,
  "outputs": [
    "The pianist practiced hours piano Chop piano perfect piano hours difficult perfect practiced Chop piano piano practiced perfect hours Chop",
    "The telescope hours practiced difficult perfect piano Chop noct piano perfect noct difficult piano hourscourse perfect practiced Chop hours",
    "The trader market stock significant experienced volatility session market market volatility stock session significant session volatility experienced stock market significant",
    "The child pair served course meal restaurant exquisite wine five wine restaurant meal five pairhourshttp(hours Theo served"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```