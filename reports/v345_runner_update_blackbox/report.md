# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1476.3s`
- Passed: `19/26`
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
    "passed_over_total": "5/11",
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
- `PASS` `metric_trainability`: {"training_info": {"total": 39.28108215332031, "recon": 2.104579210281372, "contrast": 34.850242614746094, "holonomy": 7.79260778427124, "write_policy": 0.7723989486694336, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 1.7331069707870483, "vocab_anchor": -0.0, "semantic_alignment": 9.449036598205566, "tail_semantic_anchor": 10.83304214477539, "functional_suppression": 0.0, "context_separation": 0.0, "grad_norms": {"ctx_encoder": 0.0007482521274841787, "fib_encoder": 0.1965887709118549, "dir_predictor": 0.0, "fiber_connection": 0.07661381791164013, "fiber_attn": 0.00013147521659019666, "reranker": 5.52562567311736e-09, "qformer": 0.0058541068388556945, "content_bypass": 0.008790630492632524, "semantic_probe": 0.0, "layer_pool": 0.003010081360116601, "prefix_aligner": 0.0047493121169762675, "vocab_proj": 0.034365076759143263, "tail_head": 0.1648686377146804, "context_heads": 0.026186668693906123, "memory_context_encoder": 0.03793344280266559}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy": 0.1, "semantic_probe": 0.3, "dir_diversity": 0.1, "reranker_
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist piano piano practiced difficult Chop piano perfect hours hours practiced perfect difficult Chop perfect Chop difficult hours practiced piano. difficult practiced Chop hours"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. opened companies practiced pian performance，“ please briefly pian pian practiced。Antonio practiced performed company music open pian “什么事情自然灾害omething", "space_output": "Tell me something about practice and performance. distant distant galaxies（ space telescope stars planets distant space galaxies—— stellar evolution， � stellar evolution stellar space galaxies deep space observed", "outputs_differ": true}
- `PASS` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Watson dermat graph structure。\\omega´mesurer son impact sur les cons qui utilisent\n第一步介绍了大熊猫近年来在中国四川省、陕西省、云南省……\n\n 따라서", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique piano technique finger control， pedal control finger piano pedal piano finger control musician musician musician pedal technique\n\n学生的 focus � piano techniques control finger pedal。\n\n专注于技术和", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. mechanics explains force gravitational satellites move planets mechanics force planets move gravitati
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. student student studied student study 時aneous studied studied expressive 学\n\nAssistant-normal expressive expressive studied normal student・studied student studying expressive descriptive", "space_output": "Describe the most important details a student should notice. Política mechanics explains force studies— large scale force mechanics explains gravitational force explains mechanics – gravitational gravitational planets satellites move force laws explains planets move satellites planets", "music_margin": 0.0, "space_margin": 0.3, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. student studied keyboard scales practiced：（ student scales studied scales keyboard student keyboard � conserv expressive student\n\nstudent studied:\n\nAssistant conserv expressive expressive conserv", "space_output": "Summarize the key ideas a learner should practice and remember. structure dark matter studies universe e
- `PASS` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist 불구하고 opened pian piano，“出现在《开放式 HTML Technology typing ?的照片 \n rarely changed pian Tech news》。\r\n，我们可以很方便 mktime midnight piano tutorials", "token_count": 15, "unique_token_ratio": 0.8666666666666667, "repeated_bigram_ratio": 0.0, "max_token_run": 1, "punct_ratio": 0.047619047619047616, "newline_ratio": 0.013605442176870748, "alpha_ratio": 0.8027210884353742, "content_token_ratio": 1.0, "generated_preview": "opened pian piano html technology typing rarely changed pian tech news mktime midnight piano tutorials"}, {"prompt": "The telescope", "output": "The telescope telescope telescope spectral，“ telescope 是什么� spectral spectral distant stars captured nebula：\n\n neb stars distant captured captured distant neb\n\n telescope stars spectral power", "token_count": 21, "unique_token_ratio": 0.38095238095238093, "repeated_bigram_ratio": 0.05, "max_token_run": 2, "punct_ratio": 0.020942408376963352, "newline_ratio": 0.020942408376963352, "alpha_ratio": 0.837696335078534, "content_token_ratio": 0.9047619047619048, "generated_preview": "telescope telescope spectral telescope spectral spectral distant stars captured nebula neb sta
- `PASS` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.32981958985328674, "l2_shift": 1217.627685546875, "topk_overlap_count": 3, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 5.3402276039123535, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.1, "total_segments": 20, "bad_segments": 2, "early_collapse_prompts": []}, "rows": [{"prompt": "The pianist", "output": "The pianist 불구하고 opened pian piano，“出现在《开放式 HTML Technology typing ?的照片 \n rarely changed pian Tech news》。\r\n，我们可以很方便 mktime midnight piano tutorials python photos 技 open midnight midnight noct tech openings Changed greatly improved pian Technique typing spect hours opened reopened", "generated_token_count": 33, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["opened", "pian", "piano", "html", "technology", "typing", "rarely", "changed"], "unique_ratio": 1.0, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.125}, {"segment_idx": 1, "tokens": ["pian", "tech", "news", "mktime", "midnight", "piano", "tutorials", "python"], "unique_ratio": 1.0, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.125}, {"segment_idx": 2, "tokens": ["photos", "open", "midnight", "midnight", "noct", "tech", "openings", "changed"], "unique_ratio": 0.875, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.25}, {"segment_idx": 3, "tokens": ["greatly", "improved",
- `PASS` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 3, "decoded_output": "Key piano ideas include playing fast scales, playing legato, and playing in a legato style.", "rows": [{"step": 0, "top1": {"token_id": 5619, "piece": " playing", "norm": "playing", "logit": 16.625, "prob": 0.055965278297662735}, "top1_category": "semantic", "topk_category_counts": {"semantic": 11, "functional": 1, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.14633911196142435, "functional": 0.007115187123417854, "punct": 0.0}, "chosen_token_id": 5619, "chosen_piece": " playing", "chosen_norm": "playing", "chosen_category": "semantic"}, {"step": 1, "top1": {"token_id": 4937, "piece": " fast", "norm": "fast", "logit": 18.375, "prob": 0.12891888618469238}, "top1_category": "semantic", "topk_category_counts": {"semantic": 11, "functional": 1, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.4260465120896697, "functional": 0.01977035216987133, "punct": 0.0}, "chosen_token_id": 4937, "chosen_piece": " fast", "chosen_norm": "fast", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 46769, "piece": " passages", "norm": "passages", "logit": 18.5, "prob": 0.18950460851192474
- `FAIL` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 1, "retrieval_miss": 1, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [1, 0, 3, 2, 6], "retrieved_label_counts": {"music": 4, "space": 1}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A musician refined finger technique, phrasing, and pedal control on the piano.", "The pianist practiced arpeggios and Chopin nocturnes until midnight.", "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."], "output": "What improves piano technique and musical phrasing? piano technique piano musician technique，“ finger technique finger musician piano finger control musician pedal\n pedal control pedal musician control piano pedaling finger refined technique refined", "music_score": 0.6333333333333
- `PASS` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": null, "retrieval_strength__bad_decode_score": -0.433316342537437, "prefix_l2__bad_decode_score": null}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 1, "score": 0.6797175288200379}, {"mid": 0, "score": 0.2829789757728577}, {"mid": 3, "score": 0.17892389297485353}, {"mid": 2, "score": 0.11829279661178589}, {"mid": 6, "score": 0.07854197919368744}], "retrieved_label_counts": {"music": 4, "space": 1}, "retrieval_strength": 1.259913194179535, "prefix_l2_shift": 322359623680.0, "prefix_js_divergence": 0.6091209650039673, "top1_with_prefix": {"token_id": 14566, "piece": " Options", "norm": "options", "logit": 18.75, "prob": 0.6076661944389343}, "top1_category_with_prefix": "semantic", "topk_non_semantic_prob_mass": 0.0}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": 0.600679162144661}, {"mid": 1, "score": 0.11032906174659729}, {"mid": 2, "score": 0.1047287404537201}, {"mid": 4, "score": 0.1040426641702652}, {"mid": 3, "score": 0.10125940144062043}], "retrieved_label_counts"
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? Options omitted Answer: Practice. Question: What is the main", "stage_counts": {"inject": 12}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 4, "space": 1}, "retrieved_score_sum": {"music": 1.259913194179535, "space": 0.07854197919368744}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " Options", "top1_category": "semantic", "chosen_piece": " Options", "chosen_category": "semantic", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 4, "space": 1}, "retrieved_score_sum": {"music": 1.259913194179535, "space": 0.07854197919368744}, "logits_label_ma
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist performed performances worldwide mainly due _____．报告显示的时间、音乐会的形式_____.\n   \n\n\n leafage", "Quantum systems involve sub atomic particles instead, simplifies certain computational problems due correct?\nAnswer:\n\nExplanation", "The rainforest destruction leads air quality gets _____ gradually 牢ascar是一款世界上最著名的_____级别的 super的一种？\n"], "unique_count": 3}
- `FAIL` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist piano piano practiced difficult Chop piano perfect hours hours practiced perfect difficult Chop perfect Chop difficult hours practiced", "output_b": "The pianist piano hours piano，“什么意思_____ noct hours hours noct，\r\n---\n\n noct + piano perfect"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist piano piano practiced difficult Chop piano perfect hours hours practiced perfect difficult Chop perfect Chop difficult hours practiced", "The telescope perfect noct piano Chop hours difficult practiced”， difficult hours practiced perfect piano noct hours Chop perfect difficult", "The trader market volatility stock，“ experienced significant”，__ market experienced significant volatility？\nelder stock market stock volatility", "The child professor explained simple，“Look everyday five rel explained professor rel everyday rel simple explained everyday professor simple"], "exact_same": false, "prefix_only": false, "too_short": false}
- `PASS` `rerank_stability_probe`: {"status": "pass", "pairs": [{"pair": "music_P1", "prompt_a": "What improves piano technique and musical phrasing?", "prompt_b": "How can one improve piano technique and musical expression?", "top5_a": [1, 0, 6, 5, 7], "top5_b": [1, 0, 3, 6, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9621404708846248, "pair_passed_jaccard_0_6": true}, {"pair": "space_P2", "prompt_a": "What explains satellites and orbital motion?", "prompt_b": "What describes satellites and the motion of planets?", "top5_a": [5, 6, 4, 2, 7], "top5_b": [5, 6, 4, 0, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9999999999998858, "pair_passed_jaccard_0_6": true}], "spearman_best": 0.9999999999998858, "gating": "hard_PASS"}
- `PASS` `decode_repetition_feedback_probe`: {"status": "pass", "per_prompt": [{"prompt": "The telescope", "output": "The telescope telescope telescope spectral，“ telescope 是什么� spectral spectral distant stars captured nebula：\n\n neb stars distant captured captured distant neb\n\n telescope stars spectral power:\n\nspect", "max_repeat_per_content_token": 3, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The pianist", "output": "The pianist 불구하고 opened pian piano，“出现在《开放式 HTML Technology typing ?的照片 \n rarely changed pian Tech news》。\r\n，我们可以很方便 mktime midnight piano tutorials python photos", "max_repeat_per_content_token": 2, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The market analyst", "output": "The market analyst market market stock，“ market：__是什么 stock stock power rail__\n\n### Instruction:\n ahora market volatility stock price\n\nmarket: volatility volatility high/low �", "max_repeat_per_content_token": 4, "first_bigram_repeat_index": null, "trigram_lock_count": 0}], "avg_max_repeat_per_content_token": 3.0, "min_first_bigram_repeat_index": null, "avg_trigram_lock_count": 0.0, "conditions": {"avg_max_repeat_le_3": true, "min_first_bigram_ge_4": true, "avg_trigram_
- `PASS` `functional_token_suppression_probe`: {"status": "pass", "per_prompt": [{"prompt": "A strong explanation should mention", "top12_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 18.375, "prob": 0.0198421198874712}, {"token_id": 1378, "piece": " two", "norm": "two", "logit": 18.125, "prob": 0.01545305922627449}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.125, "prob": 0.01545305922627449}, {"token_
- `FAIL` `keyword_specific_tail_slot_probe`: {"status": "fail", "metric_version": "v3.45", "per_memory": [{"mid": 0, "source_preview": "The pianist practiced arpeggios and Chopin nocturnes until m", "rare_keyword_ids": [32333, 43564], "rare_keyword_pieces": [" midnight", " practiced"], "tail_slot_top5_ids_centered": [13, 11, 320, 12, 198], "tail_slot_top5_pieces_centered": [".", ",", " (", "-", "\n"], "intersection_size_top20": 0, "rank_of_best_rare": 4073}, {"mid": 1, "source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [2524, 14317, 14762], "rare_keyword_pieces": [" control", " finger", " technique"], "tail_slot_top5_ids_centered": [13, 11, 320, 12, 198], "tail_slot_top5_pieces_centered": [".", ",", " (", "-", "\n"], "intersection_size_top20": 0, "rank_of_best_rare": 759}, {"mid": 2, "source_preview": "Classical interpretation often depends on dynamics, tempo ru", "rare_keyword_ids": [5796, 13798, 22845], "rare_keyword_pieces": [" touch", " depends", " interpretation"], "tail_slot_top5_ids_centered": [13, 11, 320, 12, 198], "tail_slot_top5_pieces_centered": [".", ",", " (", "-", "\n"], "intersection_size_top20": 0, "rank_of_best_rare": 4291}, {"mid": 3, "source_preview": "A c
- `FAIL` `context_descriptor_cluster_probe`: {"status": "fail", "metric_version": "v3.45", "loo_nn_accuracy": 0.6, "n_labeled": 5, "correct": 3, "per_memory": [{"mid": 0, "true_label": "music", "pred_label": "space", "nn_sim": -0.048688676208257675, "correct": false}, {"mid": 1, "true_label": "music", "pred_label": "space", "nn_sim": 0.013835892081260681, "correct": false}, {"mid": 4, "true_label": "space", "pred_label": "space", "nn_sim": 0.4526756703853607, "correct": true}, {"mid": 5, "true_label": "space", "pred_label": "space", "nn_sim": -0.015170933678746223, "correct": true}, {"mid": 6, "true_label": "space", "pred_label": "space", "nn_sim": 0.4526756703853607, "correct": true}], "intra_music_cos_mean": -0.18783743679523468, "intra_space_cos_mean": 0.13849682236711183, "inter_domain_cos_mean": -0.10874019128580888, "music_gap": -0.0790972455094258, "space_gap": 0.24723701365292072, "unit_norm_within_1e_3": true, "conditions": {"loo_nn_accuracy_ge_0_75": false, "unit_norm_within_1e_3": true}, "gating": "PASS_or_not_implemented"}
- `PASS` `prefix_length_scaling_probe`: {"status": "pass", "metric_version": "v3.45", "L_mem_A": 8, "L_mem_B": 16, "avg_mass_ratio_B_over_A": 1.3753844912492896, "per_prompt": [{"prompt": "A strong explanation should mention", "starter_mass_A": 18709.173828125, "starter_mass_B": 16931.916015625, "ratio": 0.9050060772951772, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 0.6348435580730438, "per_slot_mean_norm_B": 0.6350639648735523}, {"prompt": "The pianist", "starter_mass_A": 22341.75390625, "starter_mass_B": 55738.81640625, "ratio": 2.494827247678945, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 0.6349204927682877, "per_slot_mean_norm_B": 0.6352700144052505}, {"prompt": "The telescope", "starter_mass_A": 25104.185546875, "starter_mass_B": 18233.67578125, "ratio": 0.7263201487737471, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 0.6348015815019608, "per_slot_mean_norm_B": 0.6351062580943108}], "conditions": {"avg_mass_ratio_gt_1_10": true, "per_slot_norms_finite": true}, "gating": "PASS_or_not_implemented"}
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
    "total": 39.28108215332031,
    "recon": 2.104579210281372,
    "contrast": 34.850242614746094,
    "holonomy": 7.79260778427124,
    "write_policy": 0.7723989486694336,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 1.7331069707870483,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.449036598205566,
    "tail_semantic_anchor": 10.83304214477539,
    "functional_suppression": 0.0,
    "context_separation": 0.0,
    "grad_norms": {
      "ctx_encoder": 0.0007482521274841787,
      "fib_encoder": 0.1965887709118549,
      "dir_predictor": 0.0,
      "fiber_connection": 0.07661381791164013,
      "fiber_attn": 0.00013147521659019666,
      "reranker": 5.52562567311736e-09,
      "qformer": 0.0058541068388556945,
      "content_bypass": 0.008790630492632524,
      "semantic_probe": 0.0,
      "layer_pool": 0.003010081360116601,
      "prefix_aligner": 0.0047493121169762675,
      "vocab_proj": 0.034365076759143263,
      "tail_head": 0.1648686377146804,
      "context_heads": 0.026186668693906123,
      "memory_context_encoder": 0.03793344280266559
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
      "context_separation": 0.3
    }
  },
  "metric_grad_norms": [
    0.0007958483183756471,
    2.9731740141869523e-05,
    0.0009104936034418643,
    4.1173221688950434e-05,
    0.006046134978532791,
    0.0003008951898664236
  ],
  "metric_param_deltas": [
    0.0015341643011197448,
    0.0005292497226037085,
    0.0029746764339506626,
    0.0005602681776508689,
    0.003384603885933757,
    0.0005996397230774164
  ],
  "max_metric_grad_norm": 0.006046134978532791,
  "max_metric_param_delta": 0.003384603885933757,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist piano piano practiced difficult Chop piano perfect hours hours practiced perfect difficult Chop perfect Chop difficult hours practiced piano. difficult practiced Chop hours",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. opened companies practiced pian performance，“ please briefly pian pian practiced。Antonio practiced performed company music open pian “什么事情自然灾害omething",
  "space_output": "Tell me something about practice and performance. distant distant galaxies（ space telescope stars planets distant space galaxies—— stellar evolution， � stellar evolution stellar space galaxies deep space observed",
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Watson dermat graph structure。\\omega´mesurer son impact sur les cons qui utilisent\n第一步介绍了大熊猫近年来在中国四川省、陕西省、云南省……\n\n 따라서",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique piano technique finger control， pedal control finger piano pedal piano finger control musician musician musician pedal technique\n\n学生的 focus � piano techniques control finger pedal。\n\n专注于技术和",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. mechanics explains force gravitational satellites move planets mechanics force planets move gravitational mechanics satellites gravitational explains move force planets satellites explains mechanics gravitational subject force move Understanding planets improve technique.",
  "blank_music_score": 0.06666666666666667,
  "blank_space_score": 0.0,
  "music_music_score": 0.5161290322580645,
  "music_space_score": 0.0,
  "space_space_score": 0.2777777777777778,
  "space_music_score": 0.05555555555555555,
  "music_margin": 0.5161290322580645,
  "space_margin": 0.22222222222222224,
  "music_lift": 0.44946236559139785,
  "space_lift": 0.2777777777777778,
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
      "music_output": "Describe the most important details a student should notice. student student studied student study 時aneous studied studied expressive 学\n\nAssistant-normal expressive expressive studied normal student・studied student studying expressive descriptive",
      "space_output": "Describe the most important details a student should notice. Política mechanics explains force studies— large scale force mechanics explains gravitational force explains mechanics – gravitational gravitational planets satellites move force laws explains planets move satellites planets",
      "music_margin": 0.0,
      "space_margin": 0.3,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. student studied keyboard scales practiced：（ student scales studied scales keyboard student keyboard � conserv expressive student\n\nstudent studied:\n\nAssistant conserv expressive expressive conserv",
      "space_output": "Summarize the key ideas a learner should practice and remember. structure dark matter studies universe expansion large scale structure universe dark matter large expansion scale studies expansion universe large dark scale matter structure studies large studies scale.\n\n",
      "music_margin": 0.037037037037037035,
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
  "passed": true,
  "metrics": [
    {
      "prompt": "The pianist",
      "output": "The pianist 불구하고 opened pian piano，“出现在《开放式 HTML Technology typing ?的照片 \n rarely changed pian Tech news》。\r\n，我们可以很方便 mktime midnight piano tutorials",
      "token_count": 15,
      "unique_token_ratio": 0.8666666666666667,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.047619047619047616,
      "newline_ratio": 0.013605442176870748,
      "alpha_ratio": 0.8027210884353742,
      "content_token_ratio": 1.0,
      "generated_preview": "opened pian piano html technology typing rarely changed pian tech news mktime midnight piano tutorials"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope telescope spectral，“ telescope 是什么� spectral spectral distant stars captured nebula：\n\n neb stars distant captured captured distant neb\n\n telescope stars spectral power",
      "token_count": 21,
      "unique_token_ratio": 0.38095238095238093,
      "repeated_bigram_ratio": 0.05,
      "max_token_run": 2,
      "punct_ratio": 0.020942408376963352,
      "newline_ratio": 0.020942408376963352,
      "alpha_ratio": 0.837696335078534,
      "content_token_ratio": 0.9047619047619048,
      "generated_preview": "telescope telescope spectral telescope spectral spectral distant stars captured nebula neb stars distant captured captured distant neb telescope stars spectral power"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path distant galaxies observed，“ stellar evolution space deep space galaxies distant stellar evolution：\n  observed space distant deep stellar galaxies evolution：phot observed deep observed stellar",
      "token_count": 24,
      "unique_token_ratio": 0.3333333333333333,
      "repeated_bigram_ratio": 0.08695652173913043,
      "max_token_run": 1,
      "punct_ratio": 0.01932367149758454,
      "newline_ratio": 0.004830917874396135,
      "alpha_ratio": 0.8502415458937198,
      "content_token_ratio": 0.875,
      "generated_preview": "distant galaxies observed stellar evolution space deep space galaxies distant stellar evolution observed space distant deep stellar galaxies evolution phot observed deep observed stellar"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market stock，“ market：__是什么 stock stock power rail__\n\n### Instruction:\n ahora market volatility stock price\n\nmarket: volatility volatility high/",
      "token_count": 18,
      "unique_token_ratio": 0.5,
      "repeated_bigram_ratio": 0.11764705882352941,
      "max_token_run": 2,
      "punct_ratio": 0.07647058823529412,
      "newline_ratio": 0.029411764705882353,
      "alpha_ratio": 0.7823529411764706,
      "content_token_ratio": 1.0,
      "generated_preview": "market market stock market stock stock power rail instruction ahora market volatility stock price market volatility volatility high"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly professor simple everyday analog explained，“ relativity rel explained simple everyday analog rel professor：\n\n professor explained everyday simple analog comparison rel\n\n Voll professor kann erklä",
      "token_count": 24,
      "unique_token_ratio": 0.4583333333333333,
      "repeated_bigram_ratio": 0.08695652173913043,
      "max_token_run": 2,
      "punct_ratio": 0.013574660633484163,
      "newline_ratio": 0.01809954751131222,
      "alpha_ratio": 0.8461538461538461,
      "content_token_ratio": 0.75,
      "generated_preview": "professor simple everyday analog explained relativity rel explained simple everyday analog rel professor professor explained everyday simple analog comparison rel voll professor kann erkl"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.5078571428571428,
    "avg_repeated_bigram_ratio": 0.06831202046035806,
    "avg_content_token_ratio": 0.9059523809523811,
    "avg_newline_ratio": 0.01737801612908496,
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
    "js_divergence": 0.32981958985328674,
    "l2_shift": 1217.627685546875,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 5.3402276039123535,
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
        "prob": 0.13200297951698303
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 14.625,
        "prob": 0.08006385713815689
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 14.1875,
        "prob": 0.051693107932806015
      },
      {
        "token_id": 39565,
        "piece": " Provide",
        "norm": "provide",
        "logit": 13.6875,
        "prob": 0.031353455036878586
      },
      {
        "token_id": 2014,
        "piece": " To",
        "norm": "to",
        "logit": 13.625,
        "prob": 0.02945384755730629
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.4375,
        "prob": 0.024418096989393234
      },
      {
        "token_id": 21806,
        "piece": " Answer",
        "norm": "answer",
        "logit": 13.375,
        "prob": 0.022938678041100502
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.0625,
        "prob": 0.01678229682147503
      },
      {
        "token_id": 758,
        "piece": " In",
        "norm": "in",
        "logit": 13.0,
        "prob": 0.015765508636832237
      },
      {
        "token_id": 320,
        "piece": " (",
        "norm": "",
        "logit": 12.8125,
        "prob": 0.013070065528154373
      },
      {
        "token_id": 44054,
        "piece": " �",
        "norm": "",
        "logit": 12.75,
        "prob": 0.01227818988263607
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 12.75,
        "prob": 0.01227818988263607
      }
    ]
  },
  "memory": {
    "js_divergence": 0.4523841142654419,
    "l2_shift": 322359623680.0,
    "topk_overlap_count": 2,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 6.429177284240723,
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
        "token_id": 21806,
        "piece": " Answer",
        "norm": "answer",
        "logit": 15.9375,
        "prob": 0.04901956394314766
      },
      {
        "token_id": 56310,
        "piece": " Cooking",
        "norm": "cooking",
        "logit": 15.75,
        "prob": 0.04063864424824715
      },
      {
        "token_id": 39565,
        "piece": " Provide",
        "norm": "provide",
        "logit": 15.625,
        "prob": 0.0358634814620018
      },
      {
        "token_id": 32157,
        "piece": " Expert",
        "norm": "expert",
        "logit": 15.5,
        "prob": 0.03164941072463989
      },
      {
        "token_id": 37791,
        "piece": " Imagine",
        "norm": "imagine",
        "logit": 15.0,
        "prob": 0.019196337088942528
      },
      {
        "token_id": 19813,
        "piece": " Generate",
        "norm": "generate",
        "logit": 15.0,
        "prob": 0.019196337088942528
      },
      {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 14.9375,
        "prob": 0.018033290281891823
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 14.8125,
        "prob": 0.015914322808384895
      },
      {
        "token_id": 30536,
        "piece": " Climate",
        "norm": "climate",
        "logit": 14.625,
        "prob": 0.013193436898291111
      },
      {
        "token_id": 56016,
        "piece": " Scientists",
        "norm": "scientists",
        "logit": 14.5625,
        "prob": 0.012394086457788944
      },
      {
        "token_id": 9959,
        "piece": " Water",
        "norm": "water",
        "logit": 14.4375,
        "prob": 0.010937743820250034
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 14.375,
        "prob": 0.010275058448314667
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
          "logit": 19.875,
          "prob": 0.3584842085838318
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 18.125,
          "prob": 0.06229521334171295
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 17.75,
          "prob": 0.04281483590602875
        },
        {
          "token_id": 4650,
          "piece": " potential",
          "norm": "potential",
          "logit": 17.5,
          "prob": 0.03334422782063484
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 17.25,
          "prob": 0.025968510657548904
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 17.25,
          "prob": 0.025968510657548904
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.25,
          "prob": 0.025968510657548904
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.125,
          "prob": 0.0229171272367239
        },
        {
          "token_id": 14976,
          "piece": " practical",
          "norm": "practical",
          "logit": 16.875,
          "prob": 0.017847876995801926
        },
        {
          "token_id": 1931,
          "piece": " real",
          "norm": "real",
          "logit": 16.875,
          "prob": 0.017847876995801926
        },
        {
          "token_id": 9363,
          "piece": " factors",
          "norm": "factors",
          "logit": 16.5,
          "prob": 0.012266654521226883
        },
        {
          "token_id": 13656,
          "piece": " historical",
          "norm": "historical",
          "logit": 16.25,
          "prob": 0.009553280659019947
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
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 18.875,
          "prob": 0.19780392944812775
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 17.875,
          "prob": 0.07276800274848938
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 17.375,
          "prob": 0.04413602501153946
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 17.375,
          "prob": 0.04413602501153946
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.25,
          "prob": 0.03894990310072899
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.25,
          "prob": 0.03894990310072899
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 17.0,
          "prob": 0.030334215611219406
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 16.875,
          "prob": 0.02676985040307045
        },
        {
          "token_id": 4650,
          "piece": " potential",
          "norm": "potential",
          "logit": 16.625,
          "prob": 0.020848380401730537
        },
        {
          "token_id": 9363,
          "piece": " factors",
          "norm": "factors",
          "logit": 16.125,
          "prob": 0.012645181268453598
        },
        {
          "token_id": 14976,
          "piece": " practical",
          "norm": "practical",
          "logit": 16.0,
          "prob": 0.01115933433175087
        },
        {
          "token_id": 1931,
          "piece": " real",
          "norm": "real",
          "logit": 15.9375,
          "prob": 0.01048322394490242
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
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 17.75,
          "prob": 0.1137014850974083
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 17.375,
          "prob": 0.0781458169221878
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 16.625,
          "prob": 0.036913465708494186
        },
        {
          "token_id": 2999,
          "piece": " option",
          "norm": "option",
          "logit": 16.25,
          "prob": 0.02537023089826107
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 15.5,
          "prob": 0.011984048411250114
        },
        {
          "token_id": 1372,
          "piece": " number",
          "norm": "number",
          "logit": 15.375,
          "prob": 0.010575885884463787
        },
        {
          "token_id": 11136,
          "piece": " typically",
          "norm": "typically",
          "logit": 15.3125,
          "prob": 0.009935124777257442
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 15.1875,
          "prob": 0.008767717517912388
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 15.125,
          "prob": 0.008236507885158062
        },
        {
          "token_id": 3897,
          "piece": " provided",
          "norm": "provided",
          "logit": 15.0,
          "prob": 0.0072686923667788506
        },
        {
          "token_id": 1850,
          "piece": " best",
          "norm": "best",
          "logit": 14.9375,
          "prob": 0.006828304845839739
        },
        {
          "token_id": 6959,
          "piece": " Option",
          "norm": "option",
          "logit": 14.625,
          "prob": 0.004995694849640131
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
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 17.0,
          "prob": 0.0791437104344368
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 16.75,
          "prob": 0.061637185513973236
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 16.0,
          "prob": 0.02911534532904625
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 15.8125,
          "prob": 0.02413746900856495
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 15.375,
          "prob": 0.01558432076126337
        },
        {
          "token_id": 2999,
          "piece": " option",
          "norm": "option",
          "logit": 15.125,
          "prob": 0.01213708147406578
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 14.875,
          "prob": 0.009452368132770061
        },
        {
          "token_id": 3897,
          "piece": " provided",
          "norm": "provided",
          "logit": 14.625,
          "prob": 0.007361512165516615
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 14.5,
          "prob": 0.006496511399745941
        },
        {
          "token_id": 15148,
          "piece": " closely",
          "norm": "closely",
          "logit": 14.5,
          "prob": 0.006496511399745941
        },
        {
          "token_id": 1372,
          "piece": " number",
          "norm": "number",
          "logit": 14.5,
          "prob": 0.006496511399745941
        },
        {
          "token_id": 11136,
          "piece": " typically",
          "norm": "typically",
          "logit": 14.4375,
          "prob": 0.006102907937020063
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
    "bad_segment_ratio": 0.1,
    "total_segments": 20,
    "bad_segments": 2,
    "early_collapse_prompts": []
  },
  "rows": [
    {
      "prompt": "The pianist",
      "output": "The pianist 불구하고 opened pian piano，“出现在《开放式 HTML Technology typing ?的照片 \n rarely changed pian Tech news》。\r\n，我们可以很方便 mktime midnight piano tutorials python photos 技 open midnight midnight noct tech openings Changed greatly improved pian Technique typing spect hours opened reopened",
      "generated_token_count": 33,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "opened",
            "pian",
            "piano",
            "html",
            "technology",
            "typing",
            "rarely",
            "changed"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 1,
          "tokens": [
            "pian",
            "tech",
            "news",
            "mktime",
            "midnight",
            "piano",
            "tutorials",
            "python"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 2,
          "tokens": [
            "photos",
            "open",
            "midnight",
            "midnight",
            "noct",
            "tech",
            "openings",
            "changed"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "greatly",
            "improved",
            "pian",
            "technique",
            "typing",
            "spect",
            "hours",
            "opened"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 4,
          "tokens": [
            "reopened"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "bad_segments": [
        {
          "segment_idx": 4,
          "tokens": [
            "reopened"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "first_bad_segment_idx": 4
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope telescope spectral，“ telescope 是什么� spectral spectral distant stars captured nebula：\n\n neb stars distant captured captured distant neb\n\n telescope stars spectral power:\n\nspectral  neb distant captured stars\n\n\n“photographic signatures recorded photographic records” photograph ：\n\n",
      "generated_token_count": 32,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "telescope",
            "spectral",
            "telescope",
            "spectral",
            "spectral",
            "distant",
            "stars"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.14285714285714285,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "captured",
            "nebula",
            "neb",
            "stars",
            "distant",
            "captured",
            "captured",
            "distant"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "neb",
            "telescope",
            "stars",
            "spectral",
            "power",
            "spectral",
            "neb",
            "distant"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "captured",
            "stars",
            "photographic",
            "signatures",
            "recorded",
            "photographic",
            "records",
            "photograph"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market stock，“ market：__是什么 stock stock power rail__\n\n### Instruction:\n ahora market volatility stock price\n\nmarket: volatility volatility high/low 市 session session significant short interest rate limit order significant significant session open close volatility low closing",
      "generated_token_count": 35,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "market",
            "stock",
            "market",
            "stock",
            "stock",
            "power",
            "rail"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.14285714285714285,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "instruction",
            "ahora",
            "market",
            "volatility",
            "stock",
            "price",
            "market",
            "volatility"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.14285714285714285,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "volatility",
            "high",
            "low",
            "session",
            "session",
            "significant",
            "short",
            "interest"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "rate",
            "limit",
            "order",
            "significant",
            "significant",
            "session",
            "open",
            "close"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "volatility",
            "low",
            "closing"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.6666666666666666,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.3333333333333333
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly professor simple everyday analog explained，“ relativity rel explained simple everyday analog rel professor：\n\n professor explained everyday simple analog comparison rel\n\n Voll professor kann erklären, dass die Welt nicht auf einem fest standigen Bod explained simple everyday analog comp relat prof",
      "generated_token_count": 41,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "professor",
            "simple",
            "everyday",
            "analog",
            "explained",
            "relativity",
            "rel",
            "explained"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "simple",
            "everyday",
            "analog",
            "rel",
            "professor",
            "professor",
            "explained",
            "everyday"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "simple",
            "analog",
            "comparison",
            "rel",
            "voll",
            "professor",
            "kann",
            "erkl"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 3,
          "tokens": [
            "ren",
            "dass",
            "die",
            "welt",
            "nicht",
            "auf",
            "einem",
            "fest"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 4,
          "tokens": [
            "standigen",
            "bod",
            "explained",
            "simple",
            "everyday",
            "analog",
            "comp",
            "relat"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 5,
          "tokens": [
            "prof"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "bad_segments": [
        {
          "segment_idx": 5,
          "tokens": [
            "prof"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "first_bad_segment_idx": 5
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
      "decoded_output": "Key piano ideas include playing fast scales, playing legato, and playing in a legato style.",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 5619,
            "piece": " playing",
            "norm": "playing",
            "logit": 16.625,
            "prob": 0.055965278297662735
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.14633911196142435,
            "functional": 0.007115187123417854,
            "punct": 0.0
          },
          "chosen_token_id": 5619,
          "chosen_piece": " playing",
          "chosen_norm": "playing",
          "chosen_category": "semantic"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 4937,
            "piece": " fast",
            "norm": "fast",
            "logit": 18.375,
            "prob": 0.12891888618469238
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4260465120896697,
            "functional": 0.01977035216987133,
            "punct": 0.0
          },
          "chosen_token_id": 4937,
          "chosen_piece": " fast",
          "chosen_norm": "fast",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 46769,
            "piece": " passages",
            "norm": "passages",
            "logit": 18.5,
            "prob": 0.18950460851192474
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.786233326420188,
            "functional": 0.008326251991093159,
            "punct": 0.0
          },
          "chosen_token_id": 28405,
          "chosen_piece": " scales",
          "chosen_norm": "scales",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 23.25,
            "prob": 0.9490125775337219
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 1,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.012638879474252462,
            "functional": 0.0026655809488147497,
            "punct": 0.9672173236031085
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 5619,
            "piece": " playing",
            "norm": "playing",
            "logit": 20.125,
            "prob": 0.25874269008636475
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6127803511917591,
            "functional": 0.01003254298120737,
            "punct": 0.0
          },
          "chosen_token_id": 5619,
          "chosen_piece": " playing",
          "chosen_norm": "playing",
          "chosen_category": "semantic"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 2472,
            "piece": " leg",
            "norm": "leg",
            "logit": 19.125,
            "prob": 0.10786110162734985
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4109602402895689,
            "functional": 0.10786110162734985,
            "punct": 0.0
          },
          "chosen_token_id": 2472,
          "chosen_piece": " leg",
          "chosen_norm": "leg",
          "chosen_category": "functional"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 4330,
            "piece": "ato",
            "norm": "ato",
            "logit": 29.375,
            "prob": 0.9971739053726196
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 6,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.002807282619983198,
            "functional": 0.9971858460561407,
            "punct": 0.0
          },
          "chosen_token_id": 4330,
          "chosen_piece": "ato",
          "chosen_norm": "ato",
          "chosen_category": "functional"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 21.5,
            "prob": 0.45202988386154175
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 8,
            "functional": 2,
            "punct": 2
          },
          "topk_category_prob_mass": {
            "semantic": 0.3921685703098774,
            "functional": 0.029412604868412018,
            "punct": 0.5132054761052132
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
            "logit": 22.25,
            "prob": 0.4658081829547882
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 8,
            "functional": 4,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4031278440961614,
            "functional": 0.5041526712011546,
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
            "token_id": 5619,
            "piece": " playing",
            "norm": "playing",
            "logit": 21.125,
            "prob": 0.3848544955253601
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6917159841395915,
            "functional": 0.10435530869290233,
            "punct": 0.0
          },
          "chosen_token_id": 5619,
          "chosen_piece": " playing",
          "chosen_norm": "playing",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 304,
            "piece": " in",
            "norm": "in",
            "logit": 20.0,
            "prob": 0.1817181408405304
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 9,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.038331788033246994,
            "functional": 0.5816046055406332,
            "punct": 0.0
          },
          "chosen_token_id": 304,
          "chosen_piece": " in",
          "chosen_norm": "in",
          "chosen_category": "functional"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 264,
            "piece": " a",
            "norm": "a",
            "logit": 20.875,
            "prob": 0.3038615584373474
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.32625571079552174,
            "functional": 0.39581816829741,
            "punct": 0.0
          },
          "chosen_token_id": 264,
          "chosen_piece": " a",
          "chosen_norm": "a",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 2472,
            "piece": " leg",
            "norm": "leg",
            "logit": 20.375,
            "prob": 0.22031369805335999
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3361965697258711,
            "functional": 0.22031369805335999,
            "punct": 0.0
          },
          "chosen_token_id": 2472,
          "chosen_piece": " leg",
          "chosen_norm": "leg",
          "chosen_category": "functional"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 4330,
            "piece": "ato",
            "norm": "ato",
            "logit": 26.0,
            "prob": 0.9979791045188904
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 8,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.0002508971538190963,
            "functional": 0.999335296874051,
            "punct": 0.0
          },
          "chosen_token_id": 4330,
          "chosen_piece": "ato",
          "chosen_norm": "ato",
          "chosen_category": "functional"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 1707,
            "piece": " style",
            "norm": "style",
            "logit": 20.125,
            "prob": 0.34817036986351013
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 4,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.5762000782415271,
            "functional": 0.11277720425277948,
            "punct": 0.11825327482074499
          },
          "chosen_token_id": 1707,
          "chosen_piece": " style",
          "chosen_norm": "style",
          "chosen_category": "semantic"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 13,
            "piece": ".",
            "norm": "",
            "logit": 22.875,
            "prob": 0.580551028251648
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 6,
            "punct": 6
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.09820686560124159,
            "punct": 0.7998172752559185
          },
          "chosen_token_id": 13,
          "chosen_piece": ".",
          "chosen_norm": "",
          "chosen_category": "punct"
        }
      ],
      "passed": true
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 4,
      "decoded_output": "Explain the topic clearly without adding extra words. ### Explanation:\n\nThe topic is about the topic of \"",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 2041,
            "piece": " without",
            "norm": "without",
            "logit": 17.5,
            "prob": 0.30406683683395386
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6111956667155027,
            "functional": 0.015138596296310425,
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
            "logit": 18.875,
            "prob": 0.07211075723171234
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3841633405536413,
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
            "logit": 20.125,
            "prob": 0.187013179063797
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7785477498546243,
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
            "logit": 22.125,
            "prob": 0.45523449778556824
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.9258463135920465,
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
            "token_id": 624,
            "piece": ".\n",
            "norm": "",
            "logit": 21.625,
            "prob": 0.32145804166793823
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
            "punct": 0.9540900439023972
          },
          "chosen_token_id": 13,
          "chosen_piece": ".",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 16600,
            "piece": " ###",
            "norm": "",
            "logit": 17.875,
            "prob": 0.1585092544555664
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 0,
            "punct": 9
          },
          "topk_category_prob_mass": {
            "semantic": 0.06374032981693745,
            "functional": 0.0,
            "punct": 0.5794720686972141
          },
          "chosen_token_id": 16600,
          "chosen_piece": " ###",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 71287,
            "piece": " Explanation",
            "norm": "explanation",
            "logit": 21.25,
            "prob": 0.6621538996696472
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 0,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.8287883475422859,
            "functional": 0.0,
            "punct": 0.003937311004847288
          },
          "chosen_token_id": 71287,
          "chosen_piece": " Explanation",
          "chosen_norm": "explanation",
          "chosen_category": "semantic"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 1447,
            "piece": ":\n\n",
            "norm": "",
            "logit": 23.375,
            "prob": 0.48097798228263855
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 0,
            "punct": 9
          },
          "topk_category_prob_mass": {
            "semantic": 0.037628741236403584,
            "functional": 0.0,
            "punct": 0.9478736583841965
          },
          "chosen_token_id": 1447,
          "chosen_piece": ":\n\n",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 785,
            "piece": "The",
            "norm": "the",
            "logit": 19.25,
            "prob": 0.5875779986381531
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 5,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.037091474048793316,
            "functional": 0.6822039540857077,
            "punct": 0.04526147432625294
          },
          "chosen_token_id": 785,
          "chosen_piece": "The",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 8544,
            "piece": " topic",
            "norm": "topic",
            "logit": 23.0,
            "prob": 0.7204391956329346
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.8750082547776401,
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
            "token_id": 374,
            "piece": " is",
            "norm": "is",
            "logit": 23.5,
            "prob": 0.3443308472633362
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 5,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.12725703977048397,
            "functional": 0.6577846948057413,
            "punct": 0.06780276447534561
          },
          "chosen_token_id": 374,
          "chosen_piece": " is",
          "chosen_norm": "is",
          "chosen_category": "functional"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 911,
            "piece": " about",
            "norm": "about",
            "logit": 22.75,
            "prob": 0.5570091009140015
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 5,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.02515899483114481,
            "functional": 0.6764866970479488,
            "punct": 0.1758375777862966
          },
          "chosen_token_id": 911,
          "chosen_piece": " about",
          "chosen_norm": "about",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 279,
            "piece": " the",
            "norm": "the",
            "logit": 20.125,
            "prob": 0.3100799024105072
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 5,
            "punct": 2
          },
          "topk_category_prob_mass": {
            "semantic": 0.0374542074277997,
            "functional": 0.46102052507922053,
            "punct": 0.028897615615278482
          },
          "chosen_token_id": 279,
          "chosen_piece": " the",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 8544,
            "piece": " topic",
            "norm": "topic",
            "logit": 18.875,
            "prob": 0.07481884956359863
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.28823380172252655,
            "functional": 0.013001566752791405,
            "punct": 0.0
          },
          "chosen_token_id": 8544,
          "chosen_piece": " topic",
          "chosen_norm": "topic",
          "chosen_category": "semantic"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 315,
            "piece": " of",
            "norm": "of",
            "logit": 22.75,
            "prob": 0.6075021624565125
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 5,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.009568081237375736,
            "functional": 0.6265824004076421,
            "punct": 0.2920549549162388
          },
          "chosen_token_id": 315,
          "chosen_piece": " of",
          "chosen_norm": "of",
          "chosen_category": "functional"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 330,
            "piece": " \"",
            "norm": "",
            "logit": 19.125,
            "prob": 0.18270710110664368
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 7,
            "functional": 4,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.05580874625593424,
            "functional": 0.11772751808166504,
            "punct": 0.18270710110664368
          },
          "chosen_token_id": 330,
          "chosen_piece": " \"",
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
  "diagnoses": {
    "aligned": 1,
    "retrieval_miss": 1,
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
        2,
        6
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A musician refined finger technique, phrasing, and pedal control on the piano.",
        "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."
      ],
      "output": "What improves piano technique and musical phrasing? piano technique piano musician technique，“ finger technique finger musician piano finger control musician pedal\n pedal control pedal musician control piano pedaling finger refined technique refined",
      "music_score": 0.6333333333333333,
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
        1,
        2,
        4,
        3
      ],
      "retrieved_label_counts": {
        "space": 2,
        "music": 3
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "Orbital mechanics explains how satellites and planets move under gravitational force.",
        "A musician refined finger technique, phrasing, and pedal control on the piano.",
        "Classical interpretation often depends on dynamics, tempo rubato, and touch."
      ],
      "output": "What explains satellites and orbital motion? satellites explains satellites move explains gravitational force explains force gravitational move force planets move gravitational satellites planets planets explains mechanics explain gravitational motion force mechanics mechanics move satellites",
      "music_score": 0.0,
      "space_score": 0.4375,
      "generated_label": "space",
      "diagnosis": "retrieval_miss",
      "passed": false
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_mids": [
        3,
        1,
        2,
        0,
        6
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "A musician refined finger technique, phrasing, and pedal control on the piano.",
        "Classical interpretation often depends on dynamics, tempo rubato, and touch."
      ],
      "output": "Summarize the subject with concrete domain details. structure large scale studies matter universe expansion dark matter dark universe large expansion studies scale structure studies universe scale expansion matter large\n专业的 structure dark studies large",
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
    "retrieval_strength__bad_decode_score": -0.433316342537437,
    "prefix_l2__bad_decode_score": null
  },
  "rows": [
    {
      "prompt": "What improves piano technique and musical phrasing?",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 1,
          "score": 0.6797175288200379
        },
        {
          "mid": 0,
          "score": 0.2829789757728577
        },
        {
          "mid": 3,
          "score": 0.17892389297485353
        },
        {
          "mid": 2,
          "score": 0.11829279661178589
        },
        {
          "mid": 6,
          "score": 0.07854197919368744
        }
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieval_strength": 1.259913194179535,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.6091209650039673,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 18.75,
        "prob": 0.6076661944389343
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.0
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 5,
          "score": 0.600679162144661
        },
        {
          "mid": 1,
          "score": 0.11032906174659729
        },
        {
          "mid": 2,
          "score": 0.1047287404537201
        },
        {
          "mid": 4,
          "score": 0.1040426641702652
        },
        {
          "mid": 3,
          "score": 0.10125940144062043
        }
      ],
      "retrieved_label_counts": {
        "space": 2,
        "music": 3
      },
      "retrieval_strength": 0.7047218263149262,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.5956370234489441,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 16.25,
        "prob": 0.20395730435848236
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.023538557812571526
    },
    {
      "prompt": "Describe what a student should focus on first.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 3,
          "score": 0.5763964593410492
        },
        {
          "mid": 1,
          "score": 0.10781175196170809
        },
        {
          "mid": 0,
          "score": 0.0565662831068039
        },
        {
          "mid": 2,
          "score": 0.03224508464336395
        },
        {
          "mid": 4,
          "score": 0.020098072290420536
        }
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieval_strength": 0.5763964593410492,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.4775673449039459,
      "top1_with_prefix": {
        "token_id": 22201,
        "piece": " Choose",
        "norm": "choose",
        "logit": 16.25,
        "prob": 0.13543322682380676
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.01721840351819992
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 3,
          "score": 0.08414852619171143
        },
        {
          "mid": 1,
          "score": 0.07581821978092194
        },
        {
          "mid": 2,
          "score": 0.055141061544418335
        },
        {
          "mid": 0,
          "score": 0.04655141681432724
        },
        {
          "mid": 6,
          "score": 0.037887351214885706
        }
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieval_strength": 0.08414852619171143,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.3702698349952698,
      "top1_with_prefix": {
        "token_id": 21806,
        "piece": " Answer",
        "norm": "answer",
        "logit": 17.75,
        "prob": 0.17806106805801392
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.04502088949084282
    },
    {
      "prompt": "Key piano ideas include",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 1,
          "score": 0.6121546596288682
        },
        {
          "mid": 0,
          "score": 0.3816523253917694
        },
        {
          "mid": 3,
          "score": 0.2118159383535385
        },
        {
          "mid": 2,
          "score": 0.10122226476669312
        },
        {
          "mid": 6,
          "score": 0.05830757021903992
        }
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieval_strength": 1.3068451881408694,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.3318011164665222,
      "top1_with_prefix": {
        "token_id": 61584,
        "piece": " melody",
        "norm": "melody",
        "logit": 16.125,
        "prob": 0.028064129874110222
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.011698869988322258
    },
    {
      "prompt": "Orbital motion depends on",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 2,
          "score": 0.5370487570762634
        },
        {
          "mid": 3,
          "score": 0.09832845032215119
        },
        {
          "mid": 5,
          "score": 0.08738668859004975
        },
        {
          "mid": 1,
          "score": 0.04912668168544769
        },
        {
          "mid": 0,
          "score": 0.019101133942604067
        }
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieval_strength": 0.08738668859004975,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.4190765917301178,
      "top1_with_prefix": {
        "token_id": 23249,
        "piece": " gravity",
        "norm": "gravity",
        "logit": 18.875,
        "prob": 0.08914415538311005
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
      "decoded_output": "What improves piano technique and musical phrasing? Options omitted Answer: Practice. Question: What is the main",
      "stage_counts": {
        "inject": 12
      },
      "rows": [
        {
          "step": 0,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
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
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " omitted",
          "top1_category": "semantic",
          "chosen_piece": " omitted",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 2,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Answer",
          "top1_category": "semantic",
          "chosen_piece": " Answer",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 3,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
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
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Practice",
          "top1_category": "semantic",
          "chosen_piece": " Practice",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 5,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": ".",
          "top1_category": "punct",
          "chosen_piece": ".",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 6,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Question",
          "top1_category": "semantic",
          "chosen_piece": " Question",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 7,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.259913194179535,
            "space": 0.07854197919368744
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
          "step": 8,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.2160018146038056,
            "space": 0.08279128670692443
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " What",
          "top1_category": "functional",
          "chosen_piece": " What",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 9,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.2160018146038056,
            "space": 0.08279128670692443
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " is",
          "top1_category": "functional",
          "chosen_piece": " is",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 10,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.2160018146038056,
            "space": 0.08279128670692443
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " the",
          "top1_category": "functional",
          "chosen_piece": " the",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 11,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.2160018146038056,
            "space": 0.08279128670692443
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " main",
          "top1_category": "semantic",
          "chosen_piece": " main",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        }
      ],
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "decoded_output": "What explains satellites and orbital motion? Options given options:  - gravity  - gravity and inertia",
      "stage_counts": {
        "retrieve": 8,
        "inject": 4
      },
      "rows": [
        {
          "step": 0,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 1,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " given",
          "top1_category": "semantic",
          "chosen_piece": " given",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 2,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " options",
          "top1_category": "semantic",
          "chosen_piece": " options",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 3,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 4,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.002214637352153659
          },
          "top1_piece": " ",
          "top1_category": "punct",
          "chosen_piece": " ",
          "chosen_category": "punct",
          "chosen_label": "space",
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 5,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " -",
          "top1_category": "punct",
          "chosen_piece": " -",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 6,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " gravity",
          "top1_category": "semantic",
          "chosen_piece": " gravity",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 7,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "space": 2,
            "music": 3
          },
          "retrieved_score_sum": {
            "space": 0.7047218263149262,
            "music": 0.31631720364093785
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 8,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.7756042212247849,
            "music": 0.2000551909208298
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " -",
          "top1_category": "punct",
          "chosen_piece": " -",
          "chosen_category": "punct",
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
            "space": 0.7756042212247849,
            "music": 0.2000551909208298
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " friction",
          "top1_category": "semantic",
          "chosen_piece": " gravity",
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
            "space": 0.7756042212247849,
            "music": 0.2000551909208298
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " and",
          "top1_category": "functional",
          "chosen_piece": " and",
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
            "space": 0.7756042212247849,
            "music": 0.2000551909208298
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " inertia",
          "top1_category": "semantic",
          "chosen_piece": " inertia",
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
    "The pianist performed performances worldwide mainly due _____．报告显示的时间、音乐会的形式_____.\n   \n\n\n leafage",
    "Quantum systems involve sub atomic particles instead, simplifies certain computational problems due correct?\nAnswer:\n\nExplanation",
    "The rainforest destruction leads air quality gets _____ gradually 牢ascar是一款世界上最著名的_____级别的 super的一种？\n"
  ],
  "unique_count": 3,
  "error": null
}
```

## Save/Load Consistency

```json
{
  "passed": false,
  "prompt": "The pianist",
  "output_a": "The pianist piano piano practiced difficult Chop piano perfect hours hours practiced perfect difficult Chop perfect Chop difficult hours practiced",
  "output_b": "The pianist piano hours piano，“什么意思_____ noct hours hours noct，\r\n---\n\n noct + piano perfect",
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
    "The pianist piano piano practiced difficult Chop piano perfect hours hours practiced perfect difficult Chop perfect Chop difficult hours practiced",
    "The telescope perfect noct piano Chop hours difficult practiced”， difficult hours practiced perfect piano noct hours Chop perfect difficult",
    "The trader market volatility stock，“ experienced significant”，__ market experienced significant volatility？\nelder stock market stock volatility",
    "The child professor explained simple，“Look everyday five rel explained professor rel everyday rel simple explained everyday professor simple"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```