# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1250.1s`
- Passed: `18/26`
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
    "passed_over_total": "6/11",
    "threshold_K": 9,
    "passed": false
  },
  "axis_d_stability": {
    "dependent_cases": [
      "save_load_consistency",
      "rerank_stability_probe",
      "decode_repetition_feedback_probe"
    ],
    "passed_over_total": "1/3",
    "threshold_all_pass": true,
    "passed": false
  },
  "channel_passes_all_axes": false
}
```

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 41.98283386230469, "recon": 2.4085488319396973, "contrast": 43.46337127685547, "holonomy": 4.786942481994629, "write_policy": 1.0882740020751953, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.1604340076446533, "vocab_anchor": -0.0, "semantic_alignment": 9.469874382019043, "tail_semantic_anchor": 10.84397029876709, "functional_suppression": 0.0, "context_separation": 0.0, "slot_residual_alignment": 0.0, "inter_domain_margin": 0.0, "grad_norms": {"ctx_encoder": 0.0004906014182315579, "fib_encoder": 0.11028211643598784, "dir_predictor": 0.0, "fiber_connection": 0.041244823555418676, "fiber_attn": 0.00016646675964572253, "reranker": 1.862751849004779e-08, "qformer": 0.023817353122735294, "content_bypass": 0.03501559529968225, "semantic_probe": 0.0, "layer_pool": 0.003975302446633577, "prefix_aligner": 0.008598358558648532, "vocab_proj": 0.03414980954657789, "tail_head": 0.28745011541395643, "context_heads": 0.024231906034080493, "memory_context_encoder": 0.03503168573482837}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy":
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating hours *  hours\r\nelse"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. practiced midnight performances nocturnal practiced Midnight practiced noct midnight noct practiced practiced controlled noct noct midnight midnight practiced midnight controls noct midnight practices", "space_output": "Tell me something about practice and performance. distant distant space distant stars distant galaxies distant space observed observed space space stellar galaxies galaxies stellar evolution stellar stellar observed galaxies observed stellar", "outputs_differ": true}
- `PASS` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Mathematics education needs improvement mainly revolves around several key areas aimed</ Stuart Hallberg,\\\n improving problem-solving skills, improve the relevance/real-world applications,\n- Improve", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique control refers generally technique technique。 technique finger control control technique control piano musician， musician piano finger finger piano refined musician musician finger refined refined piano piano pedal Explain control", "space_output": "Explain what someone should focus on when improving technique and understanding the su
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. student student conservDesc.php studentdescCons conserv keyboard Describe student Keyboard music theory student studied describe important details:\n\n keyboard keyboard studies scales keyboard studied conserv", "space_output": "Describe the most important details a student should notice. large Describe matter large structure scale large universe scale matter studiesDescribe matter structure scale structure expansion universe studies large studies studies matter universe expansion studies universe structure", "music_margin": 0.0, "space_margin": 0.0, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. rub depends touch interpretation touch dynamics tempo often interpretation rub dynamics rub often touch often Control depends dynamics interpretation tempo touch rub interpretation dynamics touch depends often dynamics", "space_output": "Summarize the key ideas a learner should practice and remember. l
- `FAIL` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist pian piano pian pianette pian plays Chop Chop Chop hours piano piano hours pian piano perfect hours Chop hours perfect Chop midnight hours midnight perfect perfect midnight", "token_count": 27, "unique_token_ratio": 0.2962962962962963, "repeated_bigram_ratio": 0.11538461538461539, "max_token_run": 3, "punct_ratio": 0.0, "newline_ratio": 0.0, "alpha_ratio": 0.8478260869565217, "content_token_ratio": 0.8148148148148148, "generated_preview": "pian piano pian pianette pian plays chop chop chop hours piano piano hours pian piano perfect hours chop hours perfect chop midnight hours midnight"}, {"prompt": "The telescope", "output": "The telescope telescope stars telescopestarsStars amazing amazed telescope captured telescope stars stars captured stars distant telescope signatures captured captured distant captured nebula distant signatures signatures neb neb", "token_count": 25, "unique_token_ratio": 0.4, "repeated_bigram_ratio": 0.041666666666666664, "max_token_run": 2, "punct_ratio": 0.0, "newline_ratio": 0.0, "alpha_ratio": 0.8864628820960698, "content_token_ratio": 0.92, "generated_preview": "telescope stars telescopestarss
- `PASS` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.19956839084625244, "l2_shift": 586.2745361328125, "topk_overlap_count": 3, "entropy_no_prefix": 5.3277788162231445, "entropy_with_prefix": 6.950380802154541, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.75, "prob": 0.11376254260540009}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.0885983556509018}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04185090214014053}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.0369332879781723}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.5, "prob": 0.032593514770269394}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022401172667741776}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.125, "prob": 0.022401172667741776}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022401172667741776}, {"token_id": 52366, "piece": " Certainly", "norm": "certainly", "logit": 17.875, "prob": 0.01744605228304
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.3049025535583496}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.5, "prob": 0.06003887206315994}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.375, "prob": 0.05298411846160889}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03641541674733162}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03641541674733162}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.0250279251486063}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.0250279251486063}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.625, "prob": 0.0250279251486063}, {"token_id": 10295, 
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.0, "total_segments": 24, "bad_segments": 0, "early_collapse_prompts": []}, "rows": [{"prompt": "The pianist", "output": "The pianist pian piano pian pianette pian plays Chop Chop Chop hours piano piano hours pian piano perfect hours Chop hours perfect Chop midnight hours midnight perfect perfect midnight midnight pian perfect noct noct noct midnight noct pian noct Chop piano Chop perfect piano midnight Chop pian hours noct", "generated_token_count": 47, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["pian", "piano", "pian", "pianette", "pian", "plays", "chop", "chop"], "unique_ratio": 0.625, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.375}, {"segment_idx": 1, "tokens": ["chop", "hours", "piano", "piano", "hours", "pian", "piano", "perfect"], "unique_ratio": 0.625, "content_ratio": 0.75, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.375}, {"segment_idx": 2, "tokens": ["hours", "chop", "hours", "perfect", "chop", "midnight", "hours", "midnight"], "unique_ratio": 0.5, "content_ratio": 0.625, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.375}, {"segment_idx": 3, "tokens": ["perfect", "p
- `PASS` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 4, "decoded_output": "Key piano ideas include leg movements across keys, dynamic changes, and the use of the pedal. These", "rows": [{"step": 0, "top1": {"token_id": 3598, "piece": " major", "norm": "major", "logit": 16.25, "prob": 0.026983050629496574}, "top1_category": "semantic", "topk_category_counts": {"semantic": 11, "functional": 1, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.18486935831606388, "functional": 0.026983050629496574, "punct": 0.0}, "chosen_token_id": 2472, "chosen_piece": " leg", "chosen_norm": "leg", "chosen_category": "functional"}, {"step": 1, "top1": {"token_id": 19029, "piece": " movements", "norm": "movements", "logit": 14.375, "prob": 0.13023822009563446}, "top1_category": "semantic", "topk_category_counts": {"semantic": 11, "functional": 1, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.3965669944882393, "functional": 0.0113800885155797, "punct": 0.0}, "chosen_token_id": 19029, "chosen_piece": " movements", "chosen_norm": "movements", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 3941, "piece": " across", "norm": "across", "logit": 16.5, "prob": 0.0510
- `PASS` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 2, "retrieval_miss": 0, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [1, 0, 3, 6, 5], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A musician refined finger technique, phrasing, and pedal control on the piano.", "The pianist practiced arpeggios and Chopin nocturnes until midnight.", "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."], "output": "What improves piano technique and musical phrasing? piano technique control involves technique piano musician technique finger control piano piano musician control technique musician refined finger finger control finger technique piano finger refined refined pedal refined", "music_s
- `FAIL` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": null, "retrieval_strength__bad_decode_score": 0.21927202884584385, "prefix_l2__bad_decode_score": null}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 1, "score": 0.6172578841447831}, {"mid": 0, "score": 0.22511255741119385}, {"mid": 3, "score": 0.11276901960372926}, {"mid": 6, "score": 0.045475220680236815}, {"mid": 5, "score": 0.036619618535041816}], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieval_strength": 0.9551394611597062, "prefix_l2_shift": 322359623680.0, "prefix_js_divergence": 0.3171347379684448, "top1_with_prefix": {"token_id": 14566, "piece": " Options", "norm": "options", "logit": 16.375, "prob": 0.1110726147890091}, "top1_category_with_prefix": "semantic", "topk_non_semantic_prob_mass": 0.03182283788919449}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": 0.5634284257888794}, {"mid": 4, "score": 0.07376852035522463}, {"mid": 6, "score": 0.06803246438503266}, {"mid": 1, "score": 0.045463052392005925}, {"mid": 0, "score": 0.03999960422515869}]
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? Options tend towards improving piano technique, musical phrasing, and", "stage_counts": {"inject": 6, "aligned": 4, "decode": 2}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 0.9551394611597062, "space": 0.08209483921527863}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " Options", "top1_category": "semantic", "chosen_piece": " Options", "chosen_category": "semantic", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 0.9551394611597062, "space": 0
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist Xia points XYZ传感器 collects weather data based upon ____ protocol communication mode?\nBLE（Bluetooth）\n", "Quantum systems play central roles across cryptography due primarily?\\nThe Bose gas  |\n\n **Summary:\r\n\r\nWrite various", "The rainforest dataset typically refers specifically refering______. aviation charts. ____\nyes Explanation: \nFalse"], "unique_count": 3}
- `PASS` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating", "output_b": "The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating", "The telescope window watched cat sat mat outside birds window sat watched mat cat birds outside Market window cat watched", "The trader market stock volatility significant experienced 市 stock experienced significant market volatility experienced stock stock significant volatility", "The child learns Signs window window outside cat sat mat watched outside mat sat cat mat mat outside sat watched"], "exact_same": false, "prefix_only": false, "too_short": false}
- `FAIL` `rerank_stability_probe`: {"status": "fail", "pairs": [{"pair": "music_P1", "prompt_a": "What improves piano technique and musical phrasing?", "prompt_b": "How can one improve piano technique and musical expression?", "top5_a": [1, 0, 3, 4, 2], "top5_b": [1, 0, 3, 4, 2], "jaccard": 1.0, "spearman_shared": 0.9999999999998999, "pair_passed_jaccard_0_6": true}, {"pair": "space_P2", "prompt_a": "What explains satellites and orbital motion?", "prompt_b": "What describes satellites and the motion of planets?", "top5_a": [5, 0, 1, 3, 2], "top5_b": [5, 6, 4, 0, 1], "jaccard": 0.42857142857142855, "spearman_shared": 0.9607689228302918, "pair_passed_jaccard_0_6": false}], "spearman_best": 0.9999999999998999, "gating": "hard_PASS"}
- `FAIL` `decode_repetition_feedback_probe`: {"status": "fail", "per_prompt": [{"prompt": "The telescope", "output": "The telescope telescope stars telescopestarsStars amazing amazed telescope captured telescope stars stars captured stars distant telescope signatures captured captured distant captured nebula distant signatures signatures neb neb captured signatures", "max_repeat_per_content_token": 5, "first_bigram_repeat_index": 9, "trigram_lock_count": 0}, {"prompt": "The pianist", "output": "The pianist pian piano pian pianette pian plays Chop Chop Chop hours piano piano hours pian piano perfect hours Chop hours perfect Chop midnight hours midnight perfect perfect midnight midnight pian", "max_repeat_per_content_token": 5, "first_bigram_repeat_index": 8, "trigram_lock_count": 0}, {"prompt": "The market analyst", "output": "The market analyst market session sessessionssesess market market session session significant market volatility experienced stock market stock significant volatility experienced significant stock volatility significant session stock session volatility experienced volatility", "max_repeat_per_content_token": 5, "first_bigram_repeat_index": 8, "trigram_lock_count": 0}], "avg_max_repeat_per_content_token": 
- `PASS` `functional_token_suppression_probe`: {"status": "pass", "metric_version": "v3.46", "per_prompt": [{"prompt": "A strong explanation should mention", "top12_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.30489084124565125}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.5, "prob": 0.060036562383174896}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.375, "prob": 0.05298208072781563}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.036414019763469696}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.036414019763469696}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025026964023709297}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.625, "prob": 0.025026964023709297}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025026964023709297}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 18.5, "prob": 0.022086219862103462}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.25, "prob": 0.017200764268636703}, {"token_id": 1378, "piece": " two", "norm": "two", "logit": 18.125, "pro
- `PASS` `keyword_specific_tail_slot_probe`: {"status": "pass", "metric_version": "v3.50", "tail_slots_source": "bridge._last_cond_tail_slots", "per_paraphrase": [{"query": "She performed Beethoven sonatas with delicate phrasing on her grand piano.", "query_disjoint_from_rare_keywords": true, "dominant_mid": 1, "dominant_source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [2524, 14317, 14762], "rare_keyword_pieces": [" control", " finger", " technique"], "tail_slot_top5_ids_centered": [2524, 7779, 100359, 2865, 3273], "tail_slot_top5_pieces_centered": [" control", " Control", "控制", "control", "Control"], "intersection_size_top20": 1, "rank_of_best_rare": 1}, {"query": "Harmonic analysis and ear training are core elements of music education.", "query_disjoint_from_rare_keywords": true, "dominant_mid": 1, "dominant_source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [2524, 14317, 14762], "rare_keyword_pieces": [" control", " finger", " technique"], "tail_slot_top5_ids_centered": [2524, 7779, 100359, 2865, 3273], "tail_slot_top5_pieces_centered": [" control", " Control", "控制", "control", "Control"], "intersection_size_top20": 1, "ra
- `PASS` `context_descriptor_cluster_probe`: {"status": "pass", "metric_version": "v3.49", "loo_nn_accuracy_all_4": 0.6875, "loo_nn_accuracy_heldout_2": 0.875, "n_all": 16, "n_heldout": 8, "correct_all": 11, "correct_heldout": 7, "per_memory_all": [{"mid": 0, "true_label": "music", "pred_label": "space", "nn_sim": 0.10659328103065491, "correct": false}, {"mid": 1, "true_label": "music", "pred_label": "music", "nn_sim": 0.21885180473327637, "correct": true}, {"mid": 2, "true_label": "music", "pred_label": "space", "nn_sim": 0.7041908502578735, "correct": false}, {"mid": 3, "true_label": "music", "pred_label": "music", "nn_sim": 0.21885180473327637, "correct": true}, {"mid": 4, "true_label": "space", "pred_label": "space", "nn_sim": 0.6772083044052124, "correct": true}, {"mid": 5, "true_label": "space", "pred_label": "finance", "nn_sim": 0.5216456651687622, "correct": false}, {"mid": 6, "true_label": "space", "pred_label": "space", "nn_sim": 0.6772083044052124, "correct": true}, {"mid": 7, "true_label": "space", "pred_label": "music", "nn_sim": 0.7041908502578735, "correct": false}, {"mid": 8, "true_label": "cooking", "pred_label": "cooking", "nn_sim": 0.6417238712310791, "correct": true}, {"mid": 9, "true_label": "cooking", "p
- `FAIL` `prefix_length_scaling_probe`: {"status": "fail", "metric_version": "v3.45", "L_mem_A": 8, "L_mem_B": 16, "avg_mass_ratio_B_over_A": 0.8236899087743724, "per_prompt": [{"prompt": "A strong explanation should mention", "starter_mass_A": 36750.875, "starter_mass_B": 41343.8671875, "ratio": 1.124976403623043, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 1.0251211524009705, "per_slot_mean_norm_B": 1.0251210927963257}, {"prompt": "The pianist", "starter_mass_A": 22117.984375, "starter_mass_B": 14409.236328125, "ratio": 0.6514714941390314, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 1.0251210778951645, "per_slot_mean_norm_B": 1.0251211002469063}, {"prompt": "The telescope", "starter_mass_A": 14722.236328125, "starter_mass_B": 10226.38671875, "ratio": 0.6946218285610428, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 1.0251210778951645, "per_slot_mean_norm_B": 1.0251211076974869}], "conditions": {"avg_mass_ratio_gt_1_10": false, "per_slot_norms_finite": true}, "gating": "PASS_or_not_implemented"}
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
    "total": 41.98283386230469,
    "recon": 2.4085488319396973,
    "contrast": 43.46337127685547,
    "holonomy": 4.786942481994629,
    "write_policy": 1.0882740020751953,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 3.1604340076446533,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.469874382019043,
    "tail_semantic_anchor": 10.84397029876709,
    "functional_suppression": 0.0,
    "context_separation": 0.0,
    "slot_residual_alignment": 0.0,
    "inter_domain_margin": 0.0,
    "grad_norms": {
      "ctx_encoder": 0.0004906014182315579,
      "fib_encoder": 0.11028211643598784,
      "dir_predictor": 0.0,
      "fiber_connection": 0.041244823555418676,
      "fiber_attn": 0.00016646675964572253,
      "reranker": 1.862751849004779e-08,
      "qformer": 0.023817353122735294,
      "content_bypass": 0.03501559529968225,
      "semantic_probe": 0.0,
      "layer_pool": 0.003975302446633577,
      "prefix_aligner": 0.008598358558648532,
      "vocab_proj": 0.03414980954657789,
      "tail_head": 0.28745011541395643,
      "context_heads": 0.024231906034080493,
      "memory_context_encoder": 0.03503168573482837
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
    0.00019553887250367552,
    1.1630397239059675e-05,
    0.0002688287931960076,
    1.956606502062641e-05,
    0.0019679300021380186,
    0.00016431401309091598
  ],
  "metric_param_deltas": [
    0.0015214140294119716,
    0.0005180726875551045,
    0.002842925488948822,
    0.0005496913217939436,
    0.003378876717761159,
    0.0005994143430143595
  ],
  "max_metric_grad_norm": 0.0019679300021380186,
  "max_metric_param_delta": 0.003378876717761159,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating hours *  hours\r\nelse",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. practiced midnight performances nocturnal practiced Midnight practiced noct midnight noct practiced practiced controlled noct noct midnight midnight practiced midnight controls noct midnight practices",
  "space_output": "Tell me something about practice and performance. distant distant space distant stars distant galaxies distant space observed observed space space stellar galaxies galaxies stellar evolution stellar stellar observed galaxies observed stellar",
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Mathematics education needs improvement mainly revolves around several key areas aimed</ Stuart Hallberg,\\\n improving problem-solving skills, improve the relevance/real-world applications,\n- Improve",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique control refers generally technique technique。 technique finger control control technique control piano musician， musician piano finger finger piano refined musician musician finger refined refined piano piano pedal Explain control",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. mechanics force gravitational planets satellites explains move mechanics force gravitational planets satellites explains move move force move gravitational planets satellites explains mechanics move planets satellites explains force mechanics gravitational planets satellites force",
  "blank_music_score": 0.03571428571428571,
  "blank_space_score": 0.0,
  "music_music_score": 0.5,
  "music_space_score": 0.0,
  "space_space_score": 0.34210526315789475,
  "space_music_score": 0.02631578947368421,
  "music_margin": 0.5,
  "space_margin": 0.3157894736842105,
  "music_lift": 0.4642857142857143,
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
      "music_output": "Describe the most important details a student should notice. student student conservDesc.php studentdescCons conserv keyboard Describe student Keyboard music theory student studied describe important details:\n\n keyboard keyboard studies scales keyboard studied conserv",
      "space_output": "Describe the most important details a student should notice. large Describe matter large structure scale large universe scale matter studiesDescribe matter structure scale structure expansion universe studies large studies studies matter universe expansion studies universe structure",
      "music_margin": 0.0,
      "space_margin": 0.0,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. rub depends touch interpretation touch dynamics tempo often interpretation rub dynamics rub often touch often Control depends dynamics interpretation tempo touch rub interpretation dynamics touch depends often dynamics",
      "space_output": "Summarize the key ideas a learner should practice and remember. large large studies large Sum dark scale matter dark matter structureSum dark large scale scale expansion structure matter studies matter large learners key universe remember studies dark",
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
      "output": "The pianist pian piano pian pianette pian plays Chop Chop Chop hours piano piano hours pian piano perfect hours Chop hours perfect Chop midnight hours midnight perfect perfect midnight",
      "token_count": 27,
      "unique_token_ratio": 0.2962962962962963,
      "repeated_bigram_ratio": 0.11538461538461539,
      "max_token_run": 3,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8478260869565217,
      "content_token_ratio": 0.8148148148148148,
      "generated_preview": "pian piano pian pianette pian plays chop chop chop hours piano piano hours pian piano perfect hours chop hours perfect chop midnight hours midnight"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope stars telescopestarsStars amazing amazed telescope captured telescope stars stars captured stars distant telescope signatures captured captured distant captured nebula distant signatures signatures neb neb",
      "token_count": 25,
      "unique_token_ratio": 0.4,
      "repeated_bigram_ratio": 0.041666666666666664,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8864628820960698,
      "content_token_ratio": 0.92,
      "generated_preview": "telescope stars telescopestarsstars amazing amazed telescope captured telescope stars stars captured stars distant telescope signatures captured captured distant captured nebula distant signatures signatures neb"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path deep within ancient rain temple hidden tropical rain within hidden temple ancient deep hidden within tropical deep temple within ancient hidden rain deep tropical ancient temple rain hidden",
      "token_count": 28,
      "unique_token_ratio": 0.25,
      "repeated_bigram_ratio": 0.037037037037037035,
      "max_token_run": 1,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8536585365853658,
      "content_token_ratio": 0.8571428571428571,
      "generated_preview": "deep within ancient rain temple hidden tropical rain within hidden temple ancient deep hidden within tropical deep temple within ancient hidden rain deep tropical"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market session sessessionssesess market market session session significant market volatility experienced stock market stock significant volatility experienced significant stock volatility significant session stock session volatility",
      "token_count": 25,
      "unique_token_ratio": 0.28,
      "repeated_bigram_ratio": 0.08333333333333333,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8924302788844621,
      "content_token_ratio": 0.8,
      "generated_preview": "market session sessessionssesess market market session session significant market volatility experienced stock market stock significant volatility experienced significant stock volatility significant session stock session"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained simple simple analog simple rel everyday rel professor Professor explained explained analog explained rel Force professor everyday analog professor rel everyday professor analog everyday analog rel",
      "token_count": 28,
      "unique_token_ratio": 0.25,
      "repeated_bigram_ratio": 0.07407407407407407,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8708333333333333,
      "content_token_ratio": 0.6785714285714286,
      "generated_preview": "simple explained simple simple analog simple rel everyday rel professor professor explained explained analog explained rel force professor everyday analog professor rel everyday professor"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.2952592592592593,
    "avg_repeated_bigram_ratio": 0.0702991452991453,
    "avg_content_token_ratio": 0.81410582010582,
    "avg_newline_ratio": 0.0,
    "worst_max_token_run": 3,
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
    "js_divergence": 0.19956839084625244,
    "l2_shift": 586.2745361328125,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.3277788162231445,
    "entropy_with_prefix": 6.950380802154541,
    "topk_no_prefix": [
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 19.75,
        "prob": 0.11376254260540009
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 19.5,
        "prob": 0.0885983556509018
      },
      {
        "token_id": 55313,
        "piece": " Quantum",
        "norm": "quantum",
        "logit": 18.75,
        "prob": 0.04185090214014053
      },
      {
        "token_id": 58194,
        "piece": " Artificial",
        "norm": "artificial",
        "logit": 18.625,
        "prob": 0.0369332879781723
      },
      {
        "token_id": 30536,
        "piece": " Climate",
        "norm": "climate",
        "logit": 18.5,
        "prob": 0.032593514770269394
      },
      {
        "token_id": 12960,
        "piece": " Machine",
        "norm": "machine",
        "logit": 18.125,
        "prob": 0.022401172667741776
      },
      {
        "token_id": 2585,
        "piece": " How",
        "norm": "how",
        "logit": 18.125,
        "prob": 0.022401172667741776
      },
      {
        "token_id": 3555,
        "piece": " What",
        "norm": "what",
        "logit": 18.125,
        "prob": 0.022401172667741776
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 17.875,
        "prob": 0.01744605228304863
      },
      {
        "token_id": 2885,
        "piece": " Data",
        "norm": "data",
        "logit": 17.875,
        "prob": 0.01744605228304863
      },
      {
        "token_id": 15235,
        "piece": " AI",
        "norm": "ai",
        "logit": 17.625,
        "prob": 0.013586997985839844
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 17.5,
        "prob": 0.011990483850240707
      }
    ],
    "topk_with_prefix": [
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 15.75,
        "prob": 0.0856875479221344
      },
      {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 15.6875,
        "prob": 0.08049600571393967
      },
      {
        "token_id": 362,
        "piece": " A",
        "norm": "a",
        "logit": 14.5,
        "prob": 0.02454989403486252
      },
      {
        "token_id": 1096,
        "piece": " This",
        "norm": "this",
        "logit": 14.25,
        "prob": 0.01911947689950466
      },
      {
        "token_id": 1084,
        "piece": " It",
        "norm": "it",
        "logit": 14.0625,
        "prob": 0.015850603580474854
      },
      {
        "token_id": 4710,
        "piece": " \n\n",
        "norm": "",
        "logit": 13.9375,
        "prob": 0.013988107442855835
      },
      {
        "token_id": 758,
        "piece": " In",
        "norm": "in",
        "logit": 13.9375,
        "prob": 0.013988107442855835
      },
      {
        "token_id": 715,
        "piece": " \n",
        "norm": "",
        "logit": 13.8125,
        "prob": 0.012344461865723133
      },
      {
        "token_id": 330,
        "piece": " \"",
        "norm": "",
        "logit": 13.8125,
        "prob": 0.012344461865723133
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.75,
        "prob": 0.01159654837101698
      },
      {
        "token_id": 5692,
        "piece": " Here",
        "norm": "here",
        "logit": 13.5625,
        "prob": 0.009613876231014729
      },
      {
        "token_id": 2585,
        "piece": " How",
        "norm": "how",
        "logit": 13.5,
        "prob": 0.009031401015818119
      }
    ]
  },
  "memory": {
    "js_divergence": 0.2740609347820282,
    "l2_shift": 322359623680.0,
    "topk_overlap_count": 6,
    "entropy_no_prefix": 5.3277788162231445,
    "entropy_with_prefix": 7.505624294281006,
    "topk_no_prefix": [
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 19.75,
        "prob": 0.11376254260540009
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 19.5,
        "prob": 0.0885983556509018
      },
      {
        "token_id": 55313,
        "piece": " Quantum",
        "norm": "quantum",
        "logit": 18.75,
        "prob": 0.04185090214014053
      },
      {
        "token_id": 58194,
        "piece": " Artificial",
        "norm": "artificial",
        "logit": 18.625,
        "prob": 0.0369332879781723
      },
      {
        "token_id": 30536,
        "piece": " Climate",
        "norm": "climate",
        "logit": 18.5,
        "prob": 0.032593514770269394
      },
      {
        "token_id": 12960,
        "piece": " Machine",
        "norm": "machine",
        "logit": 18.125,
        "prob": 0.022401172667741776
      },
      {
        "token_id": 2585,
        "piece": " How",
        "norm": "how",
        "logit": 18.125,
        "prob": 0.022401172667741776
      },
      {
        "token_id": 3555,
        "piece": " What",
        "norm": "what",
        "logit": 18.125,
        "prob": 0.022401172667741776
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 17.875,
        "prob": 0.01744605228304863
      },
      {
        "token_id": 2885,
        "piece": " Data",
        "norm": "data",
        "logit": 17.875,
        "prob": 0.01744605228304863
      },
      {
        "token_id": 15235,
        "piece": " AI",
        "norm": "ai",
        "logit": 17.625,
        "prob": 0.013586997985839844
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 17.5,
        "prob": 0.011990483850240707
      }
    ],
    "topk_with_prefix": [
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 15.9375,
        "prob": 0.045294053852558136
      },
      {
        "token_id": 55313,
        "piece": " Quantum",
        "norm": "quantum",
        "logit": 15.4375,
        "prob": 0.027472233399748802
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 15.0,
        "prob": 0.017737405374646187
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 14.625,
        "prob": 0.01219072937965393
      },
      {
        "token_id": 30536,
        "piece": " Climate",
        "norm": "climate",
        "logit": 14.25,
        "prob": 0.008378557860851288
      },
      {
        "token_id": 58194,
        "piece": " Artificial",
        "norm": "artificial",
        "logit": 14.1875,
        "prob": 0.007870926521718502
      },
      {
        "token_id": 37444,
        "piece": " Nuclear",
        "norm": "nuclear",
        "logit": 14.0625,
        "prob": 0.0069460682570934296
      },
      {
        "token_id": 18183,
        "piece": " Deep",
        "norm": "deep",
        "logit": 14.0,
        "prob": 0.0065252273343503475
      },
      {
        "token_id": 2885,
        "piece": " Data",
        "norm": "data",
        "logit": 14.0,
        "prob": 0.0065252273343503475
      },
      {
        "token_id": 39502,
        "piece": " Hydro",
        "norm": "hydro",
        "logit": 14.0,
        "prob": 0.0065252273343503475
      },
      {
        "token_id": 12354,
        "piece": " Energy",
        "norm": "energy",
        "logit": 13.9375,
        "prob": 0.006129883695393801
      },
      {
        "token_id": 60477,
        "piece": " Neural",
        "norm": "neural",
        "logit": 13.875,
        "prob": 0.005758493207395077
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
          "prob": 0.3049025535583496
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 19.5,
          "prob": 0.06003887206315994
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 19.375,
          "prob": 0.05298411846160889
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 19.0,
          "prob": 0.03641541674733162
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 19.0,
          "prob": 0.03641541674733162
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 18.625,
          "prob": 0.0250279251486063
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.625,
          "prob": 0.0250279251486063
        },
        {
          "token_id": 678,
          "piece": " all",
          "norm": "all",
          "logit": 18.625,
          "prob": 0.0250279251486063
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 18.5,
          "prob": 0.022087067365646362
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 18.25,
          "prob": 0.01720142550766468
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 18.125,
          "prob": 0.015180204063653946
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 18.125,
          "prob": 0.015180204063653946
        }
      ],
      "music_with_prefix": [
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.875,
          "prob": 0.08923931419849396
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 17.375,
          "prob": 0.05412638187408447
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.125,
          "prob": 0.04215366765856743
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.0,
          "prob": 0.037200480699539185
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 16.875,
          "prob": 0.03282931074500084
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.5,
          "prob": 0.022563232108950615
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 16.375,
          "prob": 0.019911982119083405
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 16.25,
          "prob": 0.01757226325571537
        },
        {
          "token_id": 4650,
          "piece": " potential",
          "norm": "potential",
          "logit": 15.625,
          "prob": 0.009405754506587982
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 15.4375,
          "prob": 0.007797644007951021
        },
        {
          "token_id": 3425,
          "piece": " whether",
          "norm": "whether",
          "logit": 15.25,
          "prob": 0.006464474368840456
        },
        {
          "token_id": 1931,
          "piece": " real",
          "norm": "real",
          "logit": 15.1875,
          "prob": 0.00607281131669879
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
          "prob": 0.3049025535583496
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 19.5,
          "prob": 0.06003887206315994
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 19.375,
          "prob": 0.05298411846160889
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 19.0,
          "prob": 0.03641541674733162
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 19.0,
          "prob": 0.03641541674733162
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 18.625,
          "prob": 0.0250279251486063
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.625,
          "prob": 0.0250279251486063
        },
        {
          "token_id": 678,
          "piece": " all",
          "norm": "all",
          "logit": 18.625,
          "prob": 0.0250279251486063
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 18.5,
          "prob": 0.022087067365646362
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 18.25,
          "prob": 0.01720142550766468
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 18.125,
          "prob": 0.015180204063653946
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 18.125,
          "prob": 0.015180204063653946
        }
      ],
      "space_with_prefix": [
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 18.125,
          "prob": 0.11810589581727982
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 17.125,
          "prob": 0.04344873130321503
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.0,
          "prob": 0.03834336996078491
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 16.875,
          "prob": 0.03383790701627731
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 16.875,
          "prob": 0.03383790701627731
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.5,
          "prob": 0.02325643040239811
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 16.25,
          "prob": 0.018112126737833023
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 16.25,
          "prob": 0.018112126737833023
        },
        {
          "token_id": 3460,
          "piece": " large",
          "norm": "large",
          "logit": 15.5,
          "prob": 0.008555563166737556
        },
        {
          "token_id": 3425,
          "piece": " whether",
          "norm": "whether",
          "logit": 15.5,
          "prob": 0.008555563166737556
        },
        {
          "token_id": 4650,
          "piece": " potential",
          "norm": "potential",
          "logit": 15.4375,
          "prob": 0.008037206716835499
        },
        {
          "token_id": 5904,
          "piece": " evidence",
          "norm": "evidence",
          "logit": 15.1875,
          "prob": 0.006259383168071508
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
          "prob": 0.2765631675720215
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 19.125,
          "prob": 0.08978691697120667
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 19.0,
          "prob": 0.07923667877912521
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 18.25,
          "prob": 0.037428755313158035
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02914954163134098
        },
        {
          "token_id": 510,
          "piece": ":\n",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02914954163134098
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 17.5,
          "prob": 0.01768009178340435
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 17.5,
          "prob": 0.01768009178340435
        },
        {
          "token_id": 1447,
          "piece": ":\n\n",
          "norm": "",
          "logit": 17.375,
          "prob": 0.015602625906467438
        },
        {
          "token_id": 537,
          "piece": " not",
          "norm": "not",
          "logit": 17.25,
          "prob": 0.013769268989562988
        },
        {
          "token_id": 330,
          "piece": " \"",
          "norm": "",
          "logit": 17.25,
          "prob": 0.013769268989562988
        },
        {
          "token_id": 320,
          "piece": " (",
          "norm": "",
          "logit": 17.125,
          "prob": 0.012151338160037994
        }
      ],
      "music_with_prefix": [
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 16.0,
          "prob": 0.035965967923402786
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 15.625,
          "prob": 0.02471902407705784
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 15.5625,
          "prob": 0.023221375420689583
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.3125,
          "prob": 0.018084824085235596
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 15.25,
          "prob": 0.016989119350910187
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 15.0,
          "prob": 0.013231140561401844
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 14.9375,
          "prob": 0.01242950651794672
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 14.9375,
          "prob": 0.01242950651794672
        },
        {
          "token_id": 2999,
          "piece": " option",
          "norm": "option",
          "logit": 14.6875,
          "prob": 0.009680109098553658
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 14.6875,
          "prob": 0.009680109098553658
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 14.6875,
          "prob": 0.009680109098553658
        },
        {
          "token_id": 3520,
          "piece": " actually",
          "norm": "actually",
          "logit": 14.3125,
          "prob": 0.0066530355252325535
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
          "prob": 0.2765631675720215
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 19.125,
          "prob": 0.08978691697120667
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 19.0,
          "prob": 0.07923667877912521
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 18.25,
          "prob": 0.037428755313158035
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02914954163134098
        },
        {
          "token_id": 510,
          "piece": ":\n",
          "norm": "",
          "logit": 18.0,
          "prob": 0.02914954163134098
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 17.5,
          "prob": 0.01768009178340435
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 17.5,
          "prob": 0.01768009178340435
        },
        {
          "token_id": 1447,
          "piece": ":\n\n",
          "norm": "",
          "logit": 17.375,
          "prob": 0.015602625906467438
        },
        {
          "token_id": 537,
          "piece": " not",
          "norm": "not",
          "logit": 17.25,
          "prob": 0.013769268989562988
        },
        {
          "token_id": 330,
          "piece": " \"",
          "norm": "",
          "logit": 17.25,
          "prob": 0.013769268989562988
        },
        {
          "token_id": 320,
          "piece": " (",
          "norm": "",
          "logit": 17.125,
          "prob": 0.012151338160037994
        }
      ],
      "space_with_prefix": [
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 15.625,
          "prob": 0.024161575362086296
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 15.625,
          "prob": 0.024161575362086296
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 15.625,
          "prob": 0.024161575362086296
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 15.4375,
          "prob": 0.02003064937889576
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 15.375,
          "prob": 0.018817054107785225
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 15.375,
          "prob": 0.018817054107785225
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 15.0625,
          "prob": 0.013766851276159286
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.0,
          "prob": 0.012932759709656239
        },
        {
          "token_id": 10449,
          "piece": " presented",
          "norm": "presented",
          "logit": 14.875,
          "prob": 0.011413119733333588
        },
        {
          "token_id": 6839,
          "piece": " shown",
          "norm": "shown",
          "logit": 14.8125,
          "prob": 0.010721634142100811
        },
        {
          "token_id": 15251,
          "piece": " represented",
          "norm": "represented",
          "logit": 14.75,
          "prob": 0.010072043165564537
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 14.6875,
          "prob": 0.009461808949708939
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
      "output": "The pianist pian piano pian pianette pian plays Chop Chop Chop hours piano piano hours pian piano perfect hours Chop hours perfect Chop midnight hours midnight perfect perfect midnight midnight pian perfect noct noct noct midnight noct pian noct Chop piano Chop perfect piano midnight Chop pian hours noct",
      "generated_token_count": 47,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "pian",
            "piano",
            "pian",
            "pianette",
            "pian",
            "plays",
            "chop",
            "chop"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "chop",
            "hours",
            "piano",
            "piano",
            "hours",
            "pian",
            "piano",
            "perfect"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "hours",
            "chop",
            "hours",
            "perfect",
            "chop",
            "midnight",
            "hours",
            "midnight"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 3,
          "tokens": [
            "perfect",
            "perfect",
            "midnight",
            "midnight",
            "pian",
            "perfect",
            "noct",
            "noct"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 4,
          "tokens": [
            "noct",
            "midnight",
            "noct",
            "pian",
            "noct",
            "chop",
            "piano",
            "chop"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 5,
          "tokens": [
            "perfect",
            "piano",
            "midnight",
            "chop",
            "pian",
            "hours",
            "noct"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.8571428571428571,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.14285714285714285
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope stars telescopestarsStars amazing amazed telescope captured telescope stars stars captured stars distant telescope signatures captured captured distant captured nebula distant signatures signatures neb neb captured signatures spectral neb spectral spectral neb spectral stars neb signatures distant distant spectral signatures Space telescope neb stars spectral",
      "generated_token_count": 45,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "stars",
            "telescopestarsstars",
            "amazing",
            "amazed",
            "telescope",
            "captured",
            "telescope"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "stars",
            "stars",
            "captured",
            "stars",
            "distant",
            "telescope",
            "signatures",
            "captured"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "captured",
            "distant",
            "captured",
            "nebula",
            "distant",
            "signatures",
            "signatures",
            "neb"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "neb",
            "captured",
            "signatures",
            "spectral",
            "neb",
            "spectral",
            "spectral",
            "neb"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.14285714285714285,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 4,
          "tokens": [
            "spectral",
            "stars",
            "neb",
            "signatures",
            "distant",
            "distant",
            "spectral",
            "signatures"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "space",
            "telescope",
            "neb",
            "stars",
            "spectral"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.8,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.2
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market session sessessionssesess market market session session significant market volatility experienced stock market stock significant volatility experienced significant stock volatility significant session stock session volatility experienced volatility stock experienced session significant Market analysts often closely monitor market sessions involving significant events, particularly stock markets",
      "generated_token_count": 44,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "session",
            "sessessionssesess",
            "market",
            "market",
            "session",
            "session",
            "significant"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.14285714285714285,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "market",
            "volatility",
            "experienced",
            "stock",
            "market",
            "stock",
            "significant",
            "volatility"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "experienced",
            "significant",
            "stock",
            "volatility",
            "significant",
            "session",
            "stock",
            "session"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "volatility",
            "experienced",
            "volatility",
            "stock",
            "experienced",
            "session",
            "significant",
            "market"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "analysts",
            "often",
            "closely",
            "monitor",
            "market",
            "sessions",
            "involving",
            "significant"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 5,
          "tokens": [
            "events",
            "particularly",
            "stock",
            "markets"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained simple simple analog simple rel everyday rel professor Professor explained explained analog explained rel Force professor everyday analog professor rel everyday professor analog everyday analog rel simpleForce professor professor explained professor simple everyday explained everyday simple professor rel rel analog analog FORCE rel explainedFORCE",
      "generated_token_count": 46,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "simple",
            "explained",
            "simple",
            "simple",
            "analog",
            "simple",
            "rel",
            "everyday"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.375,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.5
        },
        {
          "segment_idx": 1,
          "tokens": [
            "rel",
            "professor",
            "professor",
            "explained",
            "explained",
            "analog",
            "explained",
            "rel"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "force",
            "professor",
            "everyday",
            "analog",
            "professor",
            "rel",
            "everyday",
            "professor"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 3,
          "tokens": [
            "analog",
            "everyday",
            "analog",
            "rel",
            "simpleforce",
            "professor",
            "professor",
            "explained"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "professor",
            "simple",
            "everyday",
            "explained",
            "everyday",
            "simple",
            "professor",
            "rel"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "rel",
            "analog",
            "analog",
            "force",
            "rel",
            "explainedforce"
          ],
          "unique_ratio": 0.6666666666666666,
          "content_ratio": 0.6666666666666666,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.3333333333333333
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
      "first_bad_step": 4,
      "decoded_output": "Key piano ideas include leg movements across keys, dynamic changes, and the use of the pedal. These",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 3598,
            "piece": " major",
            "norm": "major",
            "logit": 16.25,
            "prob": 0.026983050629496574
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.18486935831606388,
            "functional": 0.026983050629496574,
            "punct": 0.0
          },
          "chosen_token_id": 2472,
          "chosen_piece": " leg",
          "chosen_norm": "leg",
          "chosen_category": "functional"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 19029,
            "piece": " movements",
            "norm": "movements",
            "logit": 14.375,
            "prob": 0.13023822009563446
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3965669944882393,
            "functional": 0.0113800885155797,
            "punct": 0.0
          },
          "chosen_token_id": 19029,
          "chosen_piece": " movements",
          "chosen_norm": "movements",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 3941,
            "piece": " across",
            "norm": "across",
            "logit": 16.5,
            "prob": 0.05107051879167557
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.2263985425233841,
            "functional": 0.0767503883689642,
            "punct": 0.0
          },
          "chosen_token_id": 3941,
          "chosen_piece": " across",
          "chosen_norm": "across",
          "chosen_category": "functional"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 6894,
            "piece": " keys",
            "norm": "keys",
            "logit": 18.5,
            "prob": 0.09729984402656555
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.40497455187141895,
            "functional": 0.04631178267300129,
            "punct": 0.0
          },
          "chosen_token_id": 6894,
          "chosen_piece": " keys",
          "chosen_norm": "keys",
          "chosen_category": "semantic"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 21.125,
            "prob": 0.6922075748443604
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 0,
            "punct": 11
          },
          "topk_category_prob_mass": {
            "semantic": 0.004116016905754805,
            "functional": 0.0,
            "punct": 0.8863428700715303
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 8741,
            "piece": " dynamic",
            "norm": "dynamic",
            "logit": 17.625,
            "prob": 0.03767668455839157
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.1940950881689787,
            "functional": 0.04548138566315174,
            "punct": 0.0
          },
          "chosen_token_id": 8741,
          "chosen_piece": " dynamic",
          "chosen_norm": "dynamic",
          "chosen_category": "semantic"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 4344,
            "piece": " changes",
            "norm": "changes",
            "logit": 21.75,
            "prob": 0.42921698093414307
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 0,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.8407482951879501,
            "functional": 0.0,
            "punct": 0.008908114396035671
          },
          "chosen_token_id": 4344,
          "chosen_piece": " changes",
          "chosen_norm": "changes",
          "chosen_category": "semantic"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 25.375,
            "prob": 0.9306752681732178
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 2,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.0230137127218768,
            "functional": 0.005398477369453758,
            "punct": 0.961544852994848
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 323,
            "piece": " and",
            "norm": "and",
            "logit": 20.25,
            "prob": 0.4670189321041107
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.1295861303806305,
            "functional": 0.5177998133003712,
            "punct": 0.0
          },
          "chosen_token_id": 323,
          "chosen_piece": " and",
          "chosen_norm": "and",
          "chosen_category": "functional"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 279,
            "piece": " the",
            "norm": "the",
            "logit": 18.625,
            "prob": 0.1194610446691513
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.18513815943151712,
            "functional": 0.21467376872897148,
            "punct": 0.0
          },
          "chosen_token_id": 279,
          "chosen_piece": " the",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 990,
            "piece": " use",
            "norm": "use",
            "logit": 19.75,
            "prob": 0.22168958187103271
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.13133995607495308,
            "functional": 0.24505549110472202,
            "punct": 0.0
          },
          "chosen_token_id": 990,
          "chosen_piece": " use",
          "chosen_norm": "use",
          "chosen_category": "functional"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 315,
            "piece": " of",
            "norm": "of",
            "logit": 25.0,
            "prob": 0.9930819869041443
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 6,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.00010160254169022664,
            "functional": 0.9945033092226367,
            "punct": 0.00101397221442312
          },
          "chosen_token_id": 315,
          "chosen_piece": " of",
          "chosen_norm": "of",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 279,
            "piece": " the",
            "norm": "the",
            "logit": 19.125,
            "prob": 0.09505932033061981
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 8,
            "functional": 4,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.21602829732000828,
            "functional": 0.17841206304728985,
            "punct": 0.0
          },
          "chosen_token_id": 279,
          "chosen_piece": " the",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 48601,
            "piece": " pedal",
            "norm": "pedal",
            "logit": 18.375,
            "prob": 0.0746825560927391
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.42089940421283245,
            "functional": 0.09426561277359724,
            "punct": 0.0
          },
          "chosen_token_id": 48601,
          "chosen_piece": " pedal",
          "chosen_norm": "pedal",
          "chosen_category": "semantic"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 13,
            "piece": ".",
            "norm": "",
            "logit": 20.5,
            "prob": 0.38581112027168274
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 5,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.2004851959645748,
            "punct": 0.6528554670512676
          },
          "chosen_token_id": 13,
          "chosen_piece": ".",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 4220,
            "piece": " These",
            "norm": "these",
            "logit": 13.4375,
            "prob": 0.07124418765306473
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 5,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.08816616423428059,
            "functional": 0.15614240616559982,
            "punct": 0.11336426809430122
          },
          "chosen_token_id": 4220,
          "chosen_piece": " These",
          "chosen_norm": "these",
          "chosen_category": "semantic"
        }
      ],
      "passed": true
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 4,
      "decoded_output": "Explain the topic clearly based upon given context.  \"explain the topic\" is a phrase that means",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 3118,
            "piece": " based",
            "norm": "based",
            "logit": 14.1875,
            "prob": 0.17047074437141418
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.42534564854577184,
            "functional": 0.031533919274806976,
            "punct": 0.0
          },
          "chosen_token_id": 3118,
          "chosen_piece": " based",
          "chosen_norm": "based",
          "chosen_category": "semantic"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 5193,
            "piece": " upon",
            "norm": "upon",
            "logit": 17.5,
            "prob": 0.12673600018024445
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4612283743917942,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 5193,
          "chosen_piece": " upon",
          "chosen_norm": "upon",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 2661,
            "piece": " given",
            "norm": "given",
            "logit": 19.75,
            "prob": 0.23131124675273895
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5242105945944786,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 2661,
          "chosen_piece": " given",
          "chosen_norm": "given",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 2266,
            "piece": " context",
            "norm": "context",
            "logit": 21.625,
            "prob": 0.27016517519950867
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.780865266919136,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 2266,
          "chosen_piece": " context",
          "chosen_norm": "context",
          "chosen_category": "semantic"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 13,
            "piece": ".",
            "norm": "",
            "logit": 20.375,
            "prob": 0.28195127844810486
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
            "punct": 0.9160851284395903
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
            "logit": 16.625,
            "prob": 0.04581373557448387
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 7,
            "functional": 0,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.16006913781166077,
            "functional": 0.0,
            "punct": 0.09724485501646996
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 330,
            "piece": " \"",
            "norm": "",
            "logit": 14.5625,
            "prob": 0.07181069999933243
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 8,
            "functional": 0,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.18383791111409664,
            "functional": 0.0,
            "punct": 0.18584902863949537
          },
          "chosen_token_id": 330,
          "chosen_piece": " \"",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 94344,
            "piece": "explain",
            "norm": "explain",
            "logit": 12.6875,
            "prob": 0.01096130907535553
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 7,
            "functional": 5,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.04443507920950651,
            "functional": 0.02441536309197545,
            "punct": 0.0
          },
          "chosen_token_id": 94344,
          "chosen_piece": "explain",
          "chosen_norm": "explain",
          "chosen_category": "semantic"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 279,
            "piece": " the",
            "norm": "the",
            "logit": 19.5,
            "prob": 0.6197741031646729
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 3,
            "punct": 9
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.6259922899771482,
            "punct": 0.29147468809969723
          },
          "chosen_token_id": 279,
          "chosen_piece": " the",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 8544,
            "piece": " topic",
            "norm": "topic",
            "logit": 21.125,
            "prob": 0.5933138132095337
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7793927444145083,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 8544,
          "chosen_piece": " topic",
          "chosen_norm": "topic",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 1,
            "piece": "\"",
            "norm": "",
            "logit": 21.25,
            "prob": 0.2902170717716217
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 3,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.29939965903759,
            "functional": 0.18907508859410882,
            "punct": 0.39776377007365227
          },
          "chosen_token_id": 1,
          "chosen_piece": "\"",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 374,
            "piece": " is",
            "norm": "is",
            "logit": 15.625,
            "prob": 0.10762867331504822
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 1,
            "punct": 10
          },
          "topk_category_prob_mass": {
            "semantic": 0.07874282449483871,
            "functional": 0.10762867331504822,
            "punct": 0.23762445989996195
          },
          "chosen_token_id": 374,
          "chosen_piece": " is",
          "chosen_norm": "is",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 264,
            "piece": " a",
            "norm": "a",
            "logit": 21.25,
            "prob": 0.42951807379722595
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 8,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.06314869225025177,
            "functional": 0.7335648243315518,
            "punct": 0.0
          },
          "chosen_token_id": 264,
          "chosen_piece": " a",
          "chosen_norm": "a",
          "chosen_category": "functional"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 17133,
            "piece": " phrase",
            "norm": "phrase",
            "logit": 19.875,
            "prob": 0.16571058332920074
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4700012067332864,
            "functional": 0.012004034593701363,
            "punct": 0.0
          },
          "chosen_token_id": 17133,
          "chosen_piece": " phrase",
          "chosen_norm": "phrase",
          "chosen_category": "semantic"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 429,
            "piece": " that",
            "norm": "that",
            "logit": 23.0,
            "prob": 0.4704553186893463
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 7,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.2615935071371496,
            "functional": 0.6189748737961054,
            "punct": 0.007604201789945364
          },
          "chosen_token_id": 429,
          "chosen_piece": " that",
          "chosen_norm": "that",
          "chosen_category": "functional"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 3363,
            "piece": " means",
            "norm": "means",
            "logit": 21.125,
            "prob": 0.24727746844291687
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5248672068119049,
            "functional": 0.13997168745845556,
            "punct": 0.0
          },
          "chosen_token_id": 3363,
          "chosen_piece": " means",
          "chosen_norm": "means",
          "chosen_category": "semantic"
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
      "output": "What improves piano technique and musical phrasing? piano technique control involves technique piano musician technique finger control piano piano musician control technique musician refined finger finger control finger technique piano finger refined refined pedal refined",
      "music_score": 0.6060606060606061,
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
        4,
        6,
        1,
        0
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieved_majority_label": "space",
      "retrieved_text_preview": [
        "Orbital mechanics explains how satellites and planets move under gravitational force.",
        "Astronomers observed distant galaxies, quasars, and stellar evolution in deep space.",
        "A telescope captured nebulae, exoplanets, and spectral signatures from distant stars."
      ],
      "output": "What explains satellites and orbital motion? explains force satellites Force explains satellitesForce explains satellites force mechanics explains explains force satellites explain planets mechanics mechanics force mechanics gravitational force explains gravitational planets gravitational gravitational",
      "music_score": 0.0,
      "space_score": 0.5161290322580645,
      "generated_label": "space",
      "diagnosis": "aligned",
      "passed": true
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_mids": [
        3,
        1,
        6,
        0,
        5
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "A musician refined finger technique, phrasing, and pedal control on the piano.",
        "A telescope captured nebulae, exoplanets, and spectral signatures from distant stars."
      ],
      "output": "Summarize the subject with concrete domain details. touch interpretation often depends dynamics tempo rub dynamics rub tempo touch dynamics touch tempo interpretation dynamics interpretation controls interpretation rub rub touch often tempo tempo dynamics depends depends",
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
  "passed": false,
  "correlations": {
    "retrieval_strength__prefix_l2": null,
    "retrieval_strength__bad_decode_score": 0.21927202884584385,
    "prefix_l2__bad_decode_score": null
  },
  "rows": [
    {
      "prompt": "What improves piano technique and musical phrasing?",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 1,
          "score": 0.6172578841447831
        },
        {
          "mid": 0,
          "score": 0.22511255741119385
        },
        {
          "mid": 3,
          "score": 0.11276901960372926
        },
        {
          "mid": 6,
          "score": 0.045475220680236815
        },
        {
          "mid": 5,
          "score": 0.036619618535041816
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.9551394611597062,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.3171347379684448,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 16.375,
        "prob": 0.1110726147890091
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.03182283788919449
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 5,
          "score": 0.5634284257888794
        },
        {
          "mid": 4,
          "score": 0.07376852035522463
        },
        {
          "mid": 6,
          "score": 0.06803246438503266
        },
        {
          "mid": 1,
          "score": 0.045463052392005925
        },
        {
          "mid": 0,
          "score": 0.03999960422515869
        }
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieval_strength": 0.7052294105291367,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.46486830711364746,
      "top1_with_prefix": {
        "token_id": 13177,
        "piece": " Sat",
        "norm": "sat",
        "logit": 15.3125,
        "prob": 0.07889200001955032
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.1079147458076477
    },
    {
      "prompt": "Describe what a student should focus on first.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 3,
          "score": 0.5128585010766983
        },
        {
          "mid": 1,
          "score": 0.046858394145965584
        },
        {
          "mid": 0,
          "score": -0.0005610674619674696
        },
        {
          "mid": 4,
          "score": -0.011547431349754333
        },
        {
          "mid": 6,
          "score": -0.026388256251811976
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.5128585010766983,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.441089391708374,
      "top1_with_prefix": {
        "token_id": 22201,
        "piece": " Choose",
        "norm": "choose",
        "logit": 15.125,
        "prob": 0.12620772421360016
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.013302195817232132
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 3,
          "score": 0.021094447374343874
        },
        {
          "mid": 1,
          "score": 0.015311965346336366
        },
        {
          "mid": 6,
          "score": 0.004081499576568608
        },
        {
          "mid": 0,
          "score": -0.010262516140937806
        },
        {
          "mid": 5,
          "score": -0.012652482092380526
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.021094447374343874,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.32552844285964966,
      "top1_with_prefix": {
        "token_id": 58194,
        "piece": " Artificial",
        "norm": "artificial",
        "logit": 14.625,
        "prob": 0.009140501730144024
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.0
    },
    {
      "prompt": "Key piano ideas include",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 1,
          "score": 0.5411406040191651
        },
        {
          "mid": 0,
          "score": 0.3158708691596985
        },
        {
          "mid": 3,
          "score": 0.13700250387191773
        },
        {
          "mid": 6,
          "score": 0.016681492328643806
        },
        {
          "mid": 4,
          "score": -0.005892813205719001
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.9940139770507813,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.301528662443161,
      "top1_with_prefix": {
        "token_id": 3598,
        "piece": " major",
        "norm": "major",
        "logit": 16.0,
        "prob": 0.028910748660564423
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.022515714168548584
    },
    {
      "prompt": "Orbital motion depends on",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 2,
          "score": 0.3270561575889588
        },
        {
          "mid": 5,
          "score": 0.04361439943313599
        },
        {
          "mid": 3,
          "score": 0.024278688430786136
        },
        {
          "mid": 1,
          "score": -0.021913541853427882
        },
        {
          "mid": 6,
          "score": -0.033837710320949566
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.009776689112186425,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.4070507884025574,
      "top1_with_prefix": {
        "token_id": 3072,
        "piece": " mass",
        "norm": "mass",
        "logit": 18.625,
        "prob": 0.12673379480838776
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
      "decoded_output": "What improves piano technique and musical phrasing? Options tend towards improving piano technique, musical phrasing, and",
      "stage_counts": {
        "inject": 6,
        "aligned": 4,
        "decode": 2
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
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
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
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " tend",
          "top1_category": "semantic",
          "chosen_piece": " tend",
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
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
          },
          "logits_label_mass": {
            "music": 0.03443919029086828,
            "space": 0
          },
          "top1_piece": " towards",
          "top1_category": "semantic",
          "chosen_piece": " towards",
          "chosen_category": "semantic",
          "chosen_label": "music",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 3,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " improving",
          "top1_category": "semantic",
          "chosen_piece": " improving",
          "chosen_category": "semantic",
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
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
          },
          "logits_label_mass": {
            "music": 0.07181288627907634,
            "space": 0
          },
          "top1_piece": " piano",
          "top1_category": "semantic",
          "chosen_piece": " piano",
          "chosen_category": "semantic",
          "chosen_label": "music",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 5,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
          },
          "logits_label_mass": {
            "music": 0.9712017463753,
            "space": 0
          },
          "top1_piece": " technique",
          "top1_category": "semantic",
          "chosen_piece": " technique",
          "chosen_category": "semantic",
          "chosen_label": "music",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 6,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": ",",
          "top1_category": "punct",
          "chosen_piece": ",",
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
            "music": 0.9551394611597062,
            "space": 0.08209483921527863
          },
          "logits_label_mass": {
            "music": 0.03453451534733176,
            "space": 0
          },
          "top1_piece": " musical",
          "top1_category": "semantic",
          "chosen_piece": " musical",
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
            "music": 0.9551050901412963,
            "space": 0.09417556524276735
          },
          "logits_label_mass": {
            "music": 0.0019687179010361433,
            "space": 0
          },
          "top1_piece": " ph",
          "top1_category": "functional",
          "chosen_piece": " ph",
          "chosen_category": "functional",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
        },
        {
          "step": 9,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.9551050901412963,
            "space": 0.09417556524276735
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "rasing",
          "top1_category": "semantic",
          "chosen_piece": "rasing",
          "chosen_category": "semantic",
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
            "music": 0.9551050901412963,
            "space": 0.09417556524276735
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": ",",
          "top1_category": "punct",
          "chosen_piece": ",",
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
            "music": 0.9551050901412963,
            "space": 0.09417556524276735
          },
          "logits_label_mass": {
            "music": 0.02468138374388218,
            "space": 0
          },
          "top1_piece": " and",
          "top1_category": "functional",
          "chosen_piece": " and",
          "chosen_category": "functional",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
        }
      ],
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "decoded_output": "What explains satellites and orbital motion? Sat phones don' explain satellites, satellites are artificial objects that",
      "stage_counts": {
        "decode": 3,
        "aligned": 5,
        "inject": 4
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
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.010029993019998074
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
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.005930706858634949
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
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.032648902386426926
          },
          "top1_piece": " don",
          "top1_category": "functional",
          "chosen_piece": " don",
          "chosen_category": "functional",
          "chosen_label": "space",
          "diagnosed_stage": "decode"
        },
        {
          "step": 3,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "'",
          "top1_category": "punct",
          "chosen_piece": "'",
          "chosen_category": "punct",
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
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " explain",
          "top1_category": "semantic",
          "chosen_piece": " explain",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 5,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.3954301029443741
          },
          "top1_piece": " satellites",
          "top1_category": "semantic",
          "chosen_piece": " satellites",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 6,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.022754499688744545
          },
          "top1_piece": ",",
          "top1_category": "punct",
          "chosen_piece": ",",
          "chosen_category": "punct",
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
            "space": 0.7052294105291367,
            "music": 0.08546265661716462
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.2686969190835953
          },
          "top1_piece": " satellites",
          "top1_category": "semantic",
          "chosen_piece": " satellites",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 8,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.714495986700058,
            "music": 0.0803181141614914
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " are",
          "top1_category": "functional",
          "chosen_piece": " are",
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
            "space": 0.714495986700058,
            "music": 0.0803181141614914
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.03106600232422352
          },
          "top1_piece": " artificial",
          "top1_category": "semantic",
          "chosen_piece": " artificial",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.714495986700058,
            "music": 0.0803181141614914
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.33109524846076965
          },
          "top1_piece": " objects",
          "top1_category": "semantic",
          "chosen_piece": " objects",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 11,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.714495986700058,
            "music": 0.0803181141614914
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " that",
          "top1_category": "functional",
          "chosen_piece": " that",
          "chosen_category": "functional",
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
    "The pianist Xia points XYZ传感器 collects weather data based upon ____ protocol communication mode?\nBLE（Bluetooth）\n",
    "Quantum systems play central roles across cryptography due primarily?\\nThe Bose gas  |\n\n **Summary:\r\n\r\nWrite various",
    "The rainforest dataset typically refers specifically refering______. aviation charts. ____\nyes Explanation: \nFalse"
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
  "output_a": "The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating",
  "output_b": "The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating",
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
    "The pianist piano hours piano pianoHours = int(raw input desperation hours hours))\n piano perfectperfect print calculating",
    "The telescope window watched cat sat mat outside birds window sat watched mat cat birds outside Market window cat watched",
    "The trader market stock volatility significant experienced 市 stock experienced significant market volatility experienced stock stock significant volatility",
    "The child learns Signs window window outside cat sat mat watched outside mat sat cat mat mat outside sat watched"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```