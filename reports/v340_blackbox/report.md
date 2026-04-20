# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1309.4s`
- Passed: `16/26`
- Mode: fully external runner, no reuse of module-internal `test()`
- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 427.7305603027344, "recon": 2.8943073749542236, "contrast": 17888.765625, "holonomy": 5195.59130859375, "write_policy": 1.2801257371902466, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.7805848121643066, "vocab_anchor": -0.0, "semantic_alignment": 9.940794944763184, "tail_semantic_anchor": 10.923386573791504, "functional_suppression": 0.0, "grad_norms": {"ctx_encoder": 4.929302395458125e-12, "fib_encoder": 2.126063947075374e-09, "dir_predictor": 0.0, "fiber_connection": 4.753077606208372e-08, "fiber_attn": 3.575994318826387e-11, "reranker": 9.835962686109762e-14, "qformer": 2.328964943221835e-09, "content_bypass": 4.3704047808950467e-10, "semantic_probe": 0.0, "layer_pool": 1.9814493157355173e-07, "prefix_aligner": 4.5831766809876547e-11, "vocab_proj": 1.00001461006052, "tail_head": 2.193948727677274e-09, "context_heads": 2.8766823293333514e-10, "memory_context_encoder": 4.067382248098239e-10}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy": 0.1, "semantic_probe": 0.3, "dir_diversity": 0.1, "reranker_ranking": 0.2, "vo
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist piano piano key finger music keyboard 첼 plate (tablures) stage curtain キリスト holy\n\nBABIES:____"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. practiced fluent Chinese correctly.A. B: Yes, ______ correct answer：Cantonese______: No.\n\nAssistant: speaker", "space_output": "Tell me something about practice and performance. distant galaxies stellar evolution stars space telescope satellites I don ’  Mrs. Wang: John, do you remember? Xiaolin", "outputs_differ": true}
- `FAIL` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. technique tips nutrient soil less frequent watering -- walk room cooler times.\nless caffeineHuman: Ohio weather experts predict high levels _______ record low temperatures.  Leading", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refers generally either ( )注意力集中在（） ontology ontology: 世界的______ structure world's __structure\n\nattention,ontological,onorganizational", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. explains mechanics move force gravitational planets satellites Explain what someone needs focus 
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. student student squirrel cloud rabbit ㄉRequestMapping annotation describes URL mapping, parameter handling\nstudent.servlet.controller.StudentController class contains methods annotated @GetMapping", "space_output": "Describe the most important details a student should notice. explains large scale structure stars matter universe expansion universe dark energy gravity\nีémentีementีtementีืtentี\n\nSize:\n- Univers", "music_margin": 0.0, "space_margin": 0.045454545454545456, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. practiced student Korean vocabulary related 용합니다. Remember, practicing and memorizing new words involves consistent exposure, repetition, context usage within sentences (", "space_output": "Summarize the key ideas a learner should practice and remember. studies scale large universe matter dark expansion structure universe dark matter gravity.雲\n\nTo summarize, the key ideas lear
- `PASS` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist pian Haz elm tree tyre tyres East el piano musician Turkish piano The\n\n劳动者（ ）\n\nLabour labour turkish east asian eastern Turkey Turks Tur", "token_count": 22, "unique_token_ratio": 0.8181818181818182, "repeated_bigram_ratio": 0.0, "max_token_run": 2, "punct_ratio": 0.013513513513513514, "newline_ratio": 0.02702702702702703, "alpha_ratio": 0.8040540540540541, "content_token_ratio": 0.7727272727272727, "generated_preview": "pian haz elm tree tyre tyres east el piano musician turkish piano the labour labour turkish east asian eastern turkey turks tur"}, {"prompt": "The telescope", "output": "The telescope telescope costs quite high Cbd telescope\". Based entirely upon hearing Austin speak, determine whether \"Rachel likes bats\" based solely reasoning:\n\n * cannot tell", "token_count": 22, "unique_token_ratio": 0.9090909090909091, "repeated_bigram_ratio": 0.0, "max_token_run": 1, "punct_ratio": 0.03977272727272727, "newline_ratio": 0.011363636363636364, "alpha_ratio": 0.8125, "content_token_ratio": 0.9545454545454546, "generated_preview": "telescope costs quite high cbd telescope based entirely upon hearing austin speak
- `PASS` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.359661728143692, "l2_shift": 1056.75732421875, "topk_overlap_count": 3, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 5.285704612731934, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"toke
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.0, "total_segments": 7, "bad_segments": 0, "early_collapse_prompts": []}, "rows": [{"prompt": "The pianist", "output": "The pianist pian piano ruler口琴 pianist pencil piano ピ inset: Students participating ( ) music contests often play _______ instruments. ____\nmusician; musicians’\n\n： 有一种“互联网+”商业模式，被称为（），指的是消费者、", "generated_token_count": 16, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["pian", "piano", "ruler", "pianist", "pencil", "piano", "inset", "students"], "unique_ratio": 0.875, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.25}, {"segment_idx": 1, "tokens": ["participating", "music", "contests", "often", "play", "instruments", "musician", "musicians"], "unique_ratio": 1.0, "content_ratio": 0.875, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.125}], "bad_segments": [], "first_bad_segment_idx": null}, {"prompt": "The telescope", "output": "The telescope telescope corp adalah established in______.iku国贸iq Q.uestions请同学们，你知道ACE国际旅行社（中国国际航空公司旗下的子公司）在中国被称为_____。\nAirport airport\n\n企业在生产经营活动中发生的( )等情况,不属于产品质量违法行为。?", "generated_token_count": 12, "window": 8, "segments": [{"segment_idx": 0, "
- `FAIL` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 0, "decoded_output": "Key piano ideas include key ideas related to key concepts, key themes, key themes, key themes,", "rows": [{"step": 0, "top1": {"token_id": 1376, "piece": " key", "norm": "key", "logit": 13.6875, "prob": 0.01144177932292223}, "top1_category": "functional", "topk_category_counts": {"semantic": 10, "functional": 2, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.05977043369784951, "functional": 0.016846492886543274, "punct": 0.0}, "chosen_token_id": 1376, "chosen_piece": " key", "chosen_norm": "key", "chosen_category": "functional"}, {"step": 1, "top1": {"token_id": 6708, "piece": " ideas", "norm": "ideas", "logit": 13.5625, "prob": 0.03829608112573624}, "top1_category": "semantic", "topk_category_counts": {"semantic": 12, "functional": 0, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.17031287029385567, "functional": 0.0, "punct": 0.0}, "chosen_token_id": 6708, "chosen_piece": " ideas", "chosen_norm": "ideas", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 5435, "piece": " related", "norm": "related", "logit": 13.5625, "prob": 0.10104618221521378}, "top1_category":
- `PASS` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 2, "retrieval_miss": 0, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [1, 0, 3, 6, 2], "retrieved_label_counts": {"music": 4, "space": 1}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A musician refined finger technique, phrasing, and pedal control on the piano.", "The pianist practiced arpeggios and Chopin nocturnes until midnight.", "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."], "output": "What improves piano technique and musical phrasing? piano technique piano or phrasing Barry says that both improve Bart, but he emphasizes the importance of __________.\n______Barbarian Bar", "music_score": 0.23529411764705882, "space_score": 0.0, "generated_label": "music", "diagno
- `PASS` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": null, "retrieval_strength__bad_decode_score": 0.19265715550221066, "prefix_l2__bad_decode_score": null}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 1, "score": 0.5666224956512451}, {"mid": 0, "score": 0.1936155676841736}, {"mid": 3, "score": 0.06319719552993774}, {"mid": 6, "score": 0.02747329771518707}, {"mid": 5, "score": 0.02009677290916443}], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieval_strength": 0.8234352588653564, "prefix_l2_shift": 322359623680.0, "prefix_js_divergence": 0.37274929881095886, "top1_with_prefix": {"token_id": 14566, "piece": " Options", "norm": "options", "logit": 12.3125, "prob": 0.09468633681535721}, "top1_category_with_prefix": "semantic", "topk_non_semantic_prob_mass": 0.0}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": 0.5422837436199188}, {"mid": 4, "score": 0.04626110792160035}, {"mid": 6, "score": 0.04496051967144013}, {"mid": 0, "score": 0.007697209715843201}, {"mid": 1, "score": -0.006330269575119014}], "retrieved_l
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? Options refer correctly. ① Practice ② Listening", "stage_counts": {"inject": 12}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " Options", "top1_category": "semantic", "chosen_piece": " Options", "chosen_category": "semantic", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "logits_label_mass": {"musi
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist Lucy wants distribute \\( ABC$ triangle}\\]Consider $\\omega_-(side)$ denotes circum", "Quantum systems cryptography aims towards computing models running inside computers．____body（交通工具) environments.\"\n \n ", "The rainforest chicken Cass spp），被认为是大熊猫、亚马逊地区的“竞争对手”，但我们都知道，实际上巧克力冰淇淋"], "unique_count": 3}
- `FAIL` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist piano piano donald duck ducks `@don <EMAIL>`⁈disjon⁢tion", "output_b": "The pianist piano piano music finger fingers hands class Chopin Chopins nocturn\n\nAdd links within paragraphs"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist piano concert của piano concerts - Tin tức mới nhất | Vandong.com\nanh love �", "The telescope piano noct hours Chop perfect difficult practiced 想要弹好钢琴，赵老师的建议", "The trader market stock volatility session experienced significant pullbacks yesterday ，但大盘并没有受到影响。这句话是什么类型的", "The child everyday simple professor rel explained � wine said 我有一个好朋友，他是一个教授。填"], "exact_same": false, "prefix_only": false, "too_short": false}
- `PASS` `rerank_stability_probe`: {"status": "pass", "pairs": [{"pair": "music_P1", "prompt_a": "What improves piano technique and musical phrasing?", "prompt_b": "How can one improve piano technique and musical expression?", "top5_a": [1, 0, 6, 5, 7], "top5_b": [1, 0, 3, 6, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9621404708846248, "pair_passed_jaccard_0_6": true}, {"pair": "space_P2", "prompt_a": "What explains satellites and orbital motion?", "prompt_b": "What describes satellites and the motion of planets?", "top5_a": [5, 6, 4, 2, 7], "top5_b": [5, 6, 4, 0, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9999999999998858, "pair_passed_jaccard_0_6": true}], "spearman_best": 0.9999999999998858, "gating": "hard_PASS"}
- `PASS` `decode_repetition_feedback_probe`: {"status": "pass", "per_prompt": [{"prompt": "The telescope", "output": "The telescope telescope Japan telescope news Japanese astronomy 滿世界的 Astronomy News フランスfeatured featured feature カリフォ currently active すべて日本の天文ニュース。Japan Telescope", "max_repeat_per_content_token": 2, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The pianist", "output": "The pianist pian piano pianistes specialised specialisespecialistssommersummersummer\nLEE\n\n```\nlee@localhost:~/Downloads$ ssh lee.ter", "max_repeat_per_content_token": 2, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The market analyst", "output": "The market analyst market analyst market is growing explosively owing optimallyoptimizedoptimized code optimizedcode.optimelyomm onError:mm:onerroronnongatteroom市场分析师市场的", "max_repeat_per_content_token": 2, "first_bigram_repeat_index": null, "trigram_lock_count": 0}], "avg_max_repeat_per_content_token": 2.0, "min_first_bigram_repeat_index": null, "avg_trigram_lock_count": 0.0, "conditions": {"avg_max_repeat_le_3": true, "min_first_bigram_ge_4": true, "avg_trigram_lock_le_1": true}, "gating": "hard_PASS"}
- `FAIL` `functional_token_suppression_probe`: {"status": "fail", "per_prompt": [{"prompt": "A strong explanation should mention", "top12_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 18.375, "prob": 0.0198421198874712}, {"token_id": 1378, "piece": " two", "norm": "two", "logit": 18.125, "prob": 0.01545305922627449}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.125, "prob": 0.01545305922627449}, {"token_
- `FAIL` `keyword_specific_tail_slot_probe`: {"status": "fail", "per_memory": [{"mid": 0, "source_preview": "The pianist practiced arpeggios and Chopin nocturnes until m", "rare_keyword_ids": [43564, 32333], "rare_keyword_pieces": [" practiced", " midnight"], "tail_slot_top3_ids": [44903, 21317, 1482], "tail_slot_top3_pieces": ["-*", "信", " current"], "intersection_size": 0}, {"mid": 1, "source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [26278, 37191, 14762], "rare_keyword_pieces": [" piano", " refined", " technique"], "tail_slot_top3_ids": [21317, 44903, 1482], "tail_slot_top3_pieces": ["信", "-*", " current"], "intersection_size": 0}, {"mid": 2, "source_preview": "Classical interpretation often depends on dynamics, tempo ru", "rare_keyword_ids": [5796, 13798, 29195], "rare_keyword_pieces": [" touch", " depends", " dynamics"], "tail_slot_top3_ids": [21317, 44903, 1482], "tail_slot_top3_pieces": ["信", "-*", " current"], "intersection_size": 0}, {"mid": 3, "source_preview": "A conservatory student studied etudes, scales, and expressiv", "rare_keyword_ids": [77123, 11110, 19476], "rare_keyword_pieces": [" expressive", " conserv", " studied"], "tail_slot_top3_ids": [21317, 44903,
- `FAIL` `context_descriptor_cluster_probe`: {"status": "fail", "intra_music_mean_cos": 0.9241883754730225, "intra_space_mean_cos": 0.862261950969696, "inter_domain_mean_cos": 0.8333071072896322, "gating": "PASS_or_not_implemented"}
- `FAIL` `prefix_length_scaling_probe`: {"status": "fail", "L_mem_A": 8, "L_mem_B": 16, "content_starters_top12_A": 3, "content_starters_top12_B": 2, "per_slot_mean_norm_A": 0.6361142545938492, "per_slot_mean_norm_B": 0.6362451836466789, "slot_norm_ratio_B_over_A": 1.0002058263148235, "top12_A": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 20.875, "prob": 0.46686995029449463}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 19.0, "prob": 0.07159683108329773}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.375, "prob": 0.038323018699884415}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 18.375, "prob": 0.038323018699884415}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 18.25, "prob": 0.03381994739174843}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 18.0, "prob": 0.026339000090956688}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 17.625, "prob": 0.018102511763572693}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 17.625, "prob": 0.018102511763572693}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 17.5, "prob": 0.015975410118699074}, {"token_id": 3807, "piece": " several", "norm": "sever
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
    "total": 427.7305603027344,
    "recon": 2.8943073749542236,
    "contrast": 17888.765625,
    "holonomy": 5195.59130859375,
    "write_policy": 1.2801257371902466,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 3.7805848121643066,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.940794944763184,
    "tail_semantic_anchor": 10.923386573791504,
    "functional_suppression": 0.0,
    "grad_norms": {
      "ctx_encoder": 4.929302395458125e-12,
      "fib_encoder": 2.126063947075374e-09,
      "dir_predictor": 0.0,
      "fiber_connection": 4.753077606208372e-08,
      "fiber_attn": 3.575994318826387e-11,
      "reranker": 9.835962686109762e-14,
      "qformer": 2.328964943221835e-09,
      "content_bypass": 4.3704047808950467e-10,
      "semantic_probe": 0.0,
      "layer_pool": 1.9814493157355173e-07,
      "prefix_aligner": 4.5831766809876547e-11,
      "vocab_proj": 1.00001461006052,
      "tail_head": 2.193948727677274e-09,
      "context_heads": 2.8766823293333514e-10,
      "memory_context_encoder": 4.067382248098239e-10
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
      "functional_suppression": 0.4
    }
  },
  "metric_grad_norms": [
    2.125827291976634e-10,
    5.172642262435412e-12,
    3.414297733428384e-10,
    1.1582393898146304e-11,
    2.0087242980082465e-09,
    1.1372647962248905e-10
  ],
  "metric_param_deltas": [
    4.1310395317850634e-06,
    5.171603945086645e-08,
    6.766081696696347e-06,
    1.1578334380146771e-07,
    1.9677709133247845e-05,
    1.1338809144945117e-06
  ],
  "max_metric_grad_norm": 2.0087242980082465e-09,
  "max_metric_param_delta": 1.9677709133247845e-05,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist piano piano key finger music keyboard 첼 plate (tablures) stage curtain キリスト holy\n\nBABIES:____",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. practiced fluent Chinese correctly.A. B: Yes, ______ correct answer：Cantonese______: No.\n\nAssistant: speaker",
  "space_output": "Tell me something about practice and performance. distant galaxies stellar evolution stars space telescope satellites I don ’  Mrs. Wang: John, do you remember? Xiaolin",
  "outputs_differ": true,
  "error": null
}
```

## Semantic Memory Grounding

```json
{
  "passed": false,
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. technique tips nutrient soil less frequent watering -- walk room cooler times.\nless caffeineHuman: Ohio weather experts predict high levels _______ record low temperatures.  Leading",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refers generally either ( )注意力集中在（） ontology ontology: 世界的______ structure world's __structure\n\nattention,ontological,onorganizational",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. explains mechanics move force gravitational planets satellites Explain what someone needs focus on improving technique Improve Technique Absorb the fundamentals Move, Force and Gravitational pull Focus On Ex",
  "blank_music_score": 0.07142857142857142,
  "blank_space_score": 0.0,
  "music_music_score": 0.15789473684210525,
  "music_space_score": 0.0,
  "space_space_score": 0.1111111111111111,
  "space_music_score": 0.1111111111111111,
  "music_margin": 0.15789473684210525,
  "space_margin": 0.0,
  "music_lift": 0.08646616541353383,
  "space_lift": 0.1111111111111111,
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
      "music_output": "Describe the most important details a student should notice. student student squirrel cloud rabbit ㄉRequestMapping annotation describes URL mapping, parameter handling\nstudent.servlet.controller.StudentController class contains methods annotated @GetMapping",
      "space_output": "Describe the most important details a student should notice. explains large scale structure stars matter universe expansion universe dark energy gravity\nีémentีementีtementีืtentี\n\nSize:\n- Univers",
      "music_margin": 0.0,
      "space_margin": 0.045454545454545456,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. practiced student Korean vocabulary related 용합니다. Remember, practicing and memorizing new words involves consistent exposure, repetition, context usage within sentences (",
      "space_output": "Summarize the key ideas a learner should practice and remember. studies scale large universe matter dark expansion structure universe dark matter gravity.雲\n\nTo summarize, the key ideas learners typically study in cosmology and",
      "music_margin": 0.045454545454545456,
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
      "output": "The pianist pian Haz elm tree tyre tyres East el piano musician Turkish piano The\n\n劳动者（ ）\n\nLabour labour turkish east asian eastern Turkey Turks Tur",
      "token_count": 22,
      "unique_token_ratio": 0.8181818181818182,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 2,
      "punct_ratio": 0.013513513513513514,
      "newline_ratio": 0.02702702702702703,
      "alpha_ratio": 0.8040540540540541,
      "content_token_ratio": 0.7727272727272727,
      "generated_preview": "pian haz elm tree tyre tyres east el piano musician turkish piano the labour labour turkish east asian eastern turkey turks tur"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope costs quite high Cbd telescope\". Based entirely upon hearing Austin speak, determine whether \"Rachel likes bats\" based solely reasoning:\n\n * cannot tell",
      "token_count": 22,
      "unique_token_ratio": 0.9090909090909091,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.03977272727272727,
      "newline_ratio": 0.011363636363636364,
      "alpha_ratio": 0.8125,
      "content_token_ratio": 0.9545454545454546,
      "generated_preview": "telescope costs quite high cbd telescope based entirely upon hearing austin speak determine whether rachel likes bats based solely reasoning cannot tell"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path distant galaxies observed space evolution stellar deep space galaxies　centre【知识点】物理学／自然科学\n在中国科学院举行的“新时代科学家”科技创新座谈会，",
      "token_count": 10,
      "unique_token_ratio": 0.8,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.04580152671755725,
      "newline_ratio": 0.007633587786259542,
      "alpha_ratio": 0.8549618320610687,
      "content_token_ratio": 0.9,
      "generated_preview": "distant galaxies observed space evolution stellar deep space galaxies centre"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market size CBD提取 concentrates worldwide reached US$ XX million in ��2XXX and stamped growth at a CAGR=X% comp. during",
      "token_count": 20,
      "unique_token_ratio": 1.0,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.043795620437956206,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.7956204379562044,
      "content_token_ratio": 0.5,
      "generated_preview": "market size cbd concentrates worldwide reached us xx million in xxx and stamped growth at a cagr x comp during"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained everyday analogies rel professor and student? Sure! Imagine Exeel Ryan as someone dedicated, organized, structured in terms. On average",
      "token_count": 21,
      "unique_token_ratio": 1.0,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.028089887640449437,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8370786516853933,
      "content_token_ratio": 0.6666666666666666,
      "generated_preview": "simple explained everyday analogies rel professor and student sure imagine exeel ryan as someone dedicated organized structured in terms on average"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.9054545454545455,
    "avg_repeated_bigram_ratio": 0.0,
    "avg_content_token_ratio": 0.7587878787878788,
    "avg_newline_ratio": 0.009204850235384587,
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
    "js_divergence": 0.359661728143692,
    "l2_shift": 1056.75732421875,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 5.285704612731934,
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
        "logit": 15.8125,
        "prob": 0.14320825040340424
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 15.0625,
        "prob": 0.06764678657054901
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 14.8125,
        "prob": 0.05268337205052376
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 14.3125,
        "prob": 0.0319540798664093
      },
      {
        "token_id": 4891,
        "piece": " �",
        "norm": "",
        "logit": 14.0,
        "prob": 0.023378103971481323
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.9375,
        "prob": 0.021961696445941925
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.8125,
        "prob": 0.019381128251552582
      },
      {
        "token_id": 2014,
        "piece": " To",
        "norm": "to",
        "logit": 13.8125,
        "prob": 0.019381128251552582
      },
      {
        "token_id": 8908,
        "piece": " �",
        "norm": "",
        "logit": 13.75,
        "prob": 0.018206886947155
      },
      {
        "token_id": 49434,
        "piece": " �",
        "norm": "",
        "logit": 13.5625,
        "prob": 0.015094038099050522
      },
      {
        "token_id": 320,
        "piece": " (",
        "norm": "",
        "logit": 13.4375,
        "prob": 0.013320443220436573
      },
      {
        "token_id": 69162,
        "piece": " 对",
        "norm": "",
        "logit": 13.3125,
        "prob": 0.011755249463021755
      }
    ]
  },
  "memory": {
    "js_divergence": 0.32020100951194763,
    "l2_shift": 322359623680.0,
    "topk_overlap_count": 2,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 7.0924224853515625,
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
        "logit": 14.0625,
        "prob": 0.129104882478714
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.1875,
        "prob": 0.053818922489881516
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 12.625,
        "prob": 0.030665095895528793
      },
      {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 11.8125,
        "prob": 0.013607554137706757
      },
      {
        "token_id": 21806,
        "piece": " Answer",
        "norm": "answer",
        "logit": 11.25,
        "prob": 0.007753350771963596
      },
      {
        "token_id": 9731,
        "piece": " Thank",
        "norm": "thank",
        "logit": 11.125,
        "prob": 0.006842308212071657
      },
      {
        "token_id": 45451,
        "piece": " Understanding",
        "norm": "understanding",
        "logit": 10.875,
        "prob": 0.005328794475644827
      },
      {
        "token_id": 20205,
        "piece": " Based",
        "norm": "based",
        "logit": 10.875,
        "prob": 0.005328794475644827
      },
      {
        "token_id": 39565,
        "piece": " Provide",
        "norm": "provide",
        "logit": 10.8125,
        "prob": 0.005005939397960901
      },
      {
        "token_id": 10548,
        "piece": " According",
        "norm": "according",
        "logit": 10.75,
        "prob": 0.0047026448883116245
      },
      {
        "token_id": 14822,
        "piece": " Step",
        "norm": "step",
        "logit": 10.6875,
        "prob": 0.0044177258387207985
      },
      {
        "token_id": 71287,
        "piece": " Explanation",
        "norm": "explanation",
        "logit": 10.625,
        "prob": 0.004150069784373045
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
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 18.125,
          "prob": 0.13755083084106445
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 18.125,
          "prob": 0.13755083084106445
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.375,
          "prob": 0.06497441232204437
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.25,
          "prob": 0.05733971670269966
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.75,
          "prob": 0.03477829694747925
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 16.5,
          "prob": 0.02708536572754383
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 16.125,
          "prob": 0.0186154805123806
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 16.125,
          "prob": 0.0186154805123806
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.125,
          "prob": 0.0186154805123806
        },
        {
          "token_id": 13064,
          "piece": " facts",
          "norm": "facts",
          "logit": 15.875,
          "prob": 0.01449775043874979
        },
        {
          "token_id": 14175,
          "piece": " concrete",
          "norm": "concrete",
          "logit": 15.625,
          "prob": 0.011290859431028366
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 15.5625,
          "prob": 0.010606781579554081
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
          "logit": 18.375,
          "prob": 0.16248038411140442
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 18.25,
          "prob": 0.14338843524456024
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.5,
          "prob": 0.06773190200328827
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.25,
          "prob": 0.05274965614080429
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.875,
          "prob": 0.03625427559018135
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 16.625,
          "prob": 0.028234858065843582
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.375,
          "prob": 0.021989328786730766
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 16.125,
          "prob": 0.017125306650996208
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 16.125,
          "prob": 0.017125306650996208
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 15.8125,
          "prob": 0.012529142200946808
        },
        {
          "token_id": 14175,
          "piece": " concrete",
          "norm": "concrete",
          "logit": 15.8125,
          "prob": 0.012529142200946808
        },
        {
          "token_id": 13064,
          "piece": " facts",
          "norm": "facts",
          "logit": 15.8125,
          "prob": 0.012529142200946808
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
          "token_id": 2999,
          "piece": " option",
          "norm": "option",
          "logit": 16.25,
          "prob": 0.055518388748168945
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 15.6875,
          "prob": 0.03163342550396919
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 15.6875,
          "prob": 0.03163342550396919
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 15.5625,
          "prob": 0.027916399762034416
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.0625,
          "prob": 0.016932152211666107
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 15.0,
          "prob": 0.015906285494565964
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 14.9375,
          "prob": 0.014942571520805359
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 14.9375,
          "prob": 0.014942571520805359
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 14.9375,
          "prob": 0.014942571520805359
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 14.875,
          "prob": 0.014037246815860271
        },
        {
          "token_id": 10007,
          "piece": " listed",
          "norm": "listed",
          "logit": 14.625,
          "prob": 0.010932219214737415
        },
        {
          "token_id": 6959,
          "piece": " Option",
          "norm": "option",
          "logit": 14.625,
          "prob": 0.010932219214737415
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
          "logit": 16.375,
          "prob": 0.06715331226587296
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 15.8125,
          "prob": 0.038262806832790375
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 15.5,
          "prob": 0.027993664145469666
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 15.375,
          "prob": 0.024704324081540108
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.0,
          "prob": 0.016979016363620758
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 15.0,
          "prob": 0.016979016363620758
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 14.9375,
          "prob": 0.015950309112668037
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 14.875,
          "prob": 0.014983929693698883
        },
        {
          "token_id": 10007,
          "piece": " listed",
          "norm": "listed",
          "logit": 14.875,
          "prob": 0.014983929693698883
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 14.875,
          "prob": 0.014983929693698883
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 14.8125,
          "prob": 0.014076098799705505
        },
        {
          "token_id": 6959,
          "piece": " Option",
          "norm": "option",
          "logit": 14.6875,
          "prob": 0.012422113679349422
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
    "total_segments": 7,
    "bad_segments": 0,
    "early_collapse_prompts": []
  },
  "rows": [
    {
      "prompt": "The pianist",
      "output": "The pianist pian piano ruler口琴 pianist pencil piano ピ inset: Students participating ( ) music contests often play _______ instruments. ____\nmusician; musicians’\n\n： 有一种“互联网+”商业模式，被称为（），指的是消费者、",
      "generated_token_count": 16,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "pian",
            "piano",
            "ruler",
            "pianist",
            "pencil",
            "piano",
            "inset",
            "students"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "participating",
            "music",
            "contests",
            "often",
            "play",
            "instruments",
            "musician",
            "musicians"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope corp adalah established in______.iku国贸iq Q.uestions请同学们，你知道ACE国际旅行社（中国国际航空公司旗下的子公司）在中国被称为_____。\nAirport airport\n\n企业在生产经营活动中发生的( )等情况,不属于产品质量违法行为。?",
      "generated_token_count": 12,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "corp",
            "adalah",
            "established",
            "in",
            "iku",
            "iq",
            "q"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.5,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 1,
          "tokens": [
            "uestions",
            "ace",
            "airport",
            "airport"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.5
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market perspective market advantage Corporate culture：Culture是一种“看不见的东西”，也是一种（ ）\n意识形态\n\n《中华人民共和国安全生产许可证》有效期______。\n不超过( 年)\n\n()，中共中央总书记、 国委书记习近平在全国国有企业党的建设工作会议的重要",
      "generated_token_count": 7,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "perspective",
            "market",
            "advantage",
            "corporate",
            "culture",
            "culture"
          ],
          "unique_ratio": 0.7142857142857143,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.2857142857142857
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple everyday professor explained relativity analogies． Albert Einstein's（伟大的______1）_____ is famous________2) ______ his analogy.\n\n【  physics|for\n\n党支部委员会( )的数量，分公司不得超过：党总支不超过（）、子公司",
      "generated_token_count": 14,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "simple",
            "everyday",
            "professor",
            "explained",
            "relativity",
            "analogies",
            "albert",
            "einstein's"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 1,
          "tokens": [
            "is",
            "famous",
            "his",
            "analogy",
            "physics",
            "for"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.5,
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
  "passed": false,
  "rows": [
    {
      "prompt": "Key piano ideas include",
      "first_bad_step": 0,
      "decoded_output": "Key piano ideas include key ideas related to key concepts, key themes, key themes, key themes,",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 13.6875,
            "prob": 0.01144177932292223
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.05977043369784951,
            "functional": 0.016846492886543274,
            "punct": 0.0
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 6708,
            "piece": " ideas",
            "norm": "ideas",
            "logit": 13.5625,
            "prob": 0.03829608112573624
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.17031287029385567,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 6708,
          "chosen_piece": " ideas",
          "chosen_norm": "ideas",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 5435,
            "piece": " related",
            "norm": "related",
            "logit": 13.5625,
            "prob": 0.10104618221521378
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.20747481239959598,
            "functional": 0.05277250427752733,
            "punct": 0.0
          },
          "chosen_token_id": 5435,
          "chosen_piece": " related",
          "chosen_norm": "related",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 311,
            "piece": " to",
            "norm": "to",
            "logit": 16.490406036376953,
            "prob": 0.13374193012714386
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 3,
            "punct": 6
          },
          "topk_category_prob_mass": {
            "semantic": 0.029574831947684288,
            "functional": 0.19764925632625818,
            "punct": 0.12257594987750053
          },
          "chosen_token_id": 311,
          "chosen_piece": " to",
          "chosen_norm": "to",
          "chosen_category": "functional"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 17.0,
            "prob": 0.06792499125003815
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 1,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.14701253734529018,
            "functional": 0.06792499125003815,
            "punct": 0.020715951919555664
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 18940,
            "piece": " concepts",
            "norm": "concepts",
            "logit": 16.125,
            "prob": 0.07567109167575836
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 0,
            "punct": 2
          },
          "topk_category_prob_mass": {
            "semantic": 0.21485954709351063,
            "functional": 0.0,
            "punct": 0.028214489109814167
          },
          "chosen_token_id": 18940,
          "chosen_piece": " concepts",
          "chosen_norm": "concepts",
          "chosen_category": "semantic"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 19.5,
            "prob": 0.33091938495635986
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 2,
            "punct": 9
          },
          "topk_category_prob_mass": {
            "semantic": 0.05750516802072525,
            "functional": 0.024362975731492043,
            "punct": 0.6987464893609285
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 20.75,
            "prob": 0.5112636685371399
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.14407433522865176,
            "functional": 0.5232874378561974,
            "punct": 0.0
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 21386,
            "piece": " themes",
            "norm": "themes",
            "logit": 19.75,
            "prob": 0.134183868765831
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5604055179283023,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 21386,
          "chosen_piece": " themes",
          "chosen_norm": "themes",
          "chosen_category": "semantic"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 25.0,
            "prob": 0.915492057800293
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 5,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.06431761418934911,
            "punct": 0.9254684791667387
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 22.625,
            "prob": 0.472750186920166
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 6,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.004944321321090683,
            "functional": 0.9652375068690162,
            "punct": 0.0
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 21386,
            "piece": " themes",
            "norm": "themes",
            "logit": 20.375,
            "prob": 0.11783194541931152
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.515857171267271,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 21386,
          "chosen_piece": " themes",
          "chosen_norm": "themes",
          "chosen_category": "semantic"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 21.875,
            "prob": 0.6193236112594604
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 6,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.03493984788656235,
            "functional": 0.19757982157170773,
            "punct": 0.6566741280257702
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 20.75,
            "prob": 0.5771417617797852
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 7,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.013002226362004876,
            "functional": 0.914697001921013,
            "punct": 0.0
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 21386,
            "piece": " themes",
            "norm": "themes",
            "logit": 20.375,
            "prob": 0.24426430463790894
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6958057591691613,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 21386,
          "chosen_piece": " themes",
          "chosen_norm": "themes",
          "chosen_category": "semantic"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 21.5,
            "prob": 0.7340126633644104
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 4,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.010470127686858177,
            "functional": 0.09239586070179939,
            "punct": 0.8071568459272385
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        }
      ],
      "passed": false
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 4,
      "decoded_output": "Explain the topic clearly without adding extra words. 《红楼梦》是清代作家曹雪芹创作",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 2041,
            "piece": " without",
            "norm": "without",
            "logit": 14.3125,
            "prob": 0.10658255219459534
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.42449792567640543,
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
            "logit": 18.875,
            "prob": 0.08944802731275558
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.39074560441076756,
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
            "logit": 19.5,
            "prob": 0.2393154799938202
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7617826932109892,
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
            "prob": 0.6185462474822998
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.9357431754469872,
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
            "logit": 19.625,
            "prob": 0.3538092076778412
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
            "punct": 0.9212240122724324
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
            "logit": 15.5,
            "prob": 0.21086671948432922
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 0,
            "punct": 11
          },
          "topk_category_prob_mass": {
            "semantic": 0.03900642320513725,
            "functional": 0.0,
            "punct": 0.45699948258697987
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 26940,
            "piece": "《",
            "norm": "",
            "logit": 13.6875,
            "prob": 0.08805997669696808
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
            "punct": 0.465955400839448
          },
          "chosen_token_id": 26940,
          "chosen_piece": "《",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 117805,
            "piece": "红楼梦",
            "norm": "",
            "logit": 7.40625,
            "prob": 0.02005736343562603
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 0,
            "punct": 10
          },
          "topk_category_prob_mass": {
            "semantic": 0.0105865728110075,
            "functional": 0.0,
            "punct": 0.09069720190018415
          },
          "chosen_token_id": 117805,
          "chosen_piece": "红楼梦",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 25067,
            "piece": "》",
            "norm": "",
            "logit": 21.875,
            "prob": 0.9929779171943665
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
            "punct": 0.9977547683665762
          },
          "chosen_token_id": 25067,
          "chosen_piece": "》",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 20412,
            "piece": "是",
            "norm": "",
            "logit": 16.875,
            "prob": 0.23572656512260437
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
            "punct": 0.7980509856715798
          },
          "chosen_token_id": 20412,
          "chosen_piece": "是",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 112978,
            "piece": "清代",
            "norm": "",
            "logit": 18.125,
            "prob": 0.613299548625946
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
            "punct": 0.8654689900577068
          },
          "chosen_token_id": 112978,
          "chosen_piece": "清代",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 105022,
            "piece": "作家",
            "norm": "",
            "logit": 19.5,
            "prob": 0.4908621311187744
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
            "punct": 0.9412267287261784
          },
          "chosen_token_id": 105022,
          "chosen_piece": "作家",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 102263,
            "piece": "曹",
            "norm": "",
            "logit": 20.875,
            "prob": 0.9727939963340759
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
            "punct": 0.9884256894001737
          },
          "chosen_token_id": 102263,
          "chosen_piece": "曹",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 100167,
            "piece": "雪",
            "norm": "",
            "logit": 23.5,
            "prob": 0.9990718364715576
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
            "punct": 0.9997917111827519
          },
          "chosen_token_id": 100167,
          "chosen_piece": "雪",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 117539,
            "piece": "芹",
            "norm": "",
            "logit": 25.875,
            "prob": 0.999786913394928
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
            "punct": 0.9999598060654762
          },
          "chosen_token_id": 117539,
          "chosen_piece": "芹",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 104223,
            "piece": "创作",
            "norm": "",
            "logit": 21.75,
            "prob": 0.7125537991523743
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
            "punct": 0.9743024373892695
          },
          "chosen_token_id": 104223,
          "chosen_piece": "创作",
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
        2
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
      "output": "What improves piano technique and musical phrasing? piano technique piano or phrasing Barry says that both improve Bart, but he emphasizes the importance of __________.\n______Barbarian Bar",
      "music_score": 0.23529411764705882,
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
        2,
        1
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
      "output": "What explains satellites and orbital motion? satellites explains sinks sink satellitesWhat explains orbitals motion? orbital explain sions ions\norbital motions orbits\n\n【 】\n\norbitalescies",
      "music_score": 0.0,
      "space_score": 0.4,
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
        7,
        1,
        2
      ],
      "retrieved_label_counts": {
        "space": 2,
        "music": 3
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A telescope captured nebulae, exoplanets, and spectral signatures from distant stars.",
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "Cosmology studies dark matter, expansion, and the large scale structure of the universe."
      ],
      "output": "Summarize the subject with concrete domain details. matter large scale structure universe dark expansion studies matter dark energy survey studies Arch. Matter ARCH.Matter APARCH.archmatter.APArch\n\nwrite down",
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
    "retrieval_strength__bad_decode_score": 0.19265715550221066,
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
          "score": 0.1936155676841736
        },
        {
          "mid": 3,
          "score": 0.06319719552993774
        },
        {
          "mid": 6,
          "score": 0.02747329771518707
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
      "prefix_js_divergence": 0.37274929881095886,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 12.3125,
        "prob": 0.09468633681535721
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
          "score": 0.5422837436199188
        },
        {
          "mid": 4,
          "score": 0.04626110792160035
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
          "score": -0.006330269575119014
        }
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieval_strength": 0.6335053712129592,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.5061731338500977,
      "top1_with_prefix": {
        "token_id": 13177,
        "piece": " Sat",
        "norm": "sat",
        "logit": 11.4375,
        "prob": 0.12010252475738525
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.16614807024598122
    },
    {
      "prompt": "Describe what a student should focus on first.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 3,
          "score": 0.45830298662185676
        },
        {
          "mid": 1,
          "score": -0.007808592915534977
        },
        {
          "mid": 0,
          "score": -0.03504327237606048
        },
        {
          "mid": 7,
          "score": -0.038606351613998405
        },
        {
          "mid": 4,
          "score": -0.04108911752700806
        }
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieval_strength": 0.45830298662185676,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.44606852531433105,
      "top1_with_prefix": {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 11.1875,
        "prob": 0.05965147167444229
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.0
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 7,
          "score": -0.002285179495811463
        },
        {
          "mid": 6,
          "score": -0.010802556574344636
        },
        {
          "mid": 5,
          "score": -0.02638280838727951
        },
        {
          "mid": 3,
          "score": -0.026887077093124392
        },
        {
          "mid": 1,
          "score": -0.033489438891410823
        }
      ],
      "retrieved_label_counts": {
        "space": 3,
        "music": 2
      },
      "retrieval_strength": -0.002285179495811463,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.28323596715927124,
      "top1_with_prefix": {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 12.5,
        "prob": 0.0468447208404541
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.026691319420933723
    },
    {
      "prompt": "Key piano ideas include",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 1,
          "score": 0.5106263399124146
        },
        {
          "mid": 0,
          "score": 0.30423030257225037
        },
        {
          "mid": 3,
          "score": 0.10775353312492371
        },
        {
          "mid": 6,
          "score": 0.021317118406295778
        },
        {
          "mid": 2,
          "score": 0.0047838211059570215
        }
      ],
      "retrieved_label_counts": {
        "music": 4,
        "space": 1
      },
      "retrieval_strength": 0.9273939967155457,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.3519740700721741,
      "top1_with_prefix": {
        "token_id": 5619,
        "piece": " playing",
        "norm": "playing",
        "logit": 14.125,
        "prob": 0.021296756342053413
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.011116607580333948
    },
    {
      "prompt": "Orbital motion depends on",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 2,
          "score": 0.43496288061141974
        },
        {
          "mid": 5,
          "score": 0.04124398231506348
        },
        {
          "mid": 3,
          "score": -0.010372707247734071
        },
        {
          "mid": 6,
          "score": -0.03860478103160858
        },
        {
          "mid": 4,
          "score": -0.04442960172891618
        }
      ],
      "retrieved_label_counts": {
        "music": 2,
        "space": 3
      },
      "retrieval_strength": -0.04179040044546128,
      "prefix_l2_shift": 322359623680.0,
      "prefix_js_divergence": 0.4576057493686676,
      "top1_with_prefix": {
        "token_id": 3807,
        "piece": " several",
        "norm": "several",
        "logit": 16.875,
        "prob": 0.07981263101100922
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
      "decoded_output": "What improves piano technique and musical phrasing? Options refer correctly. ① Practice ② Listening",
      "stage_counts": {
        "inject": 12
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
            "space": 0.22133269011974335
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
            "space": 0.22133269011974335
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
            "space": 0.22133269011974335
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " correctly",
          "top1_category": "semantic",
          "chosen_piece": " correctly",
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
            "space": 0.22133269011974335
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
          "step": 4,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269011974335
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
          "step": 5,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269011974335
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
          "step": 6,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0435107663273813,
            "space": 0.22133269011974335
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
            "space": 0.22133269011974335
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
          "step": 8,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0016044586896897,
            "space": 0.20829569399356843
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
          "step": 9,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0016044586896897,
            "space": 0.20829569399356843
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
          "step": 10,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0016044586896897,
            "space": 0.20829569399356843
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
            "music": 1.0016044586896897,
            "space": 0.20829569399356843
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Listening",
          "top1_category": "semantic",
          "chosen_piece": " Listening",
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
      "decoded_output": "What explains satellites and orbital motion? Explain why satellites move around planets. 1. **Understanding",
      "stage_counts": {
        "inject": 10,
        "decode": 1,
        "aligned": 1
      },
      "rows": [
        {
          "step": 0,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Kepler",
          "top1_category": "semantic",
          "chosen_piece": " Explain",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 1,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.05654364451766014
          },
          "top1_piece": " why",
          "top1_category": "functional",
          "chosen_piece": " why",
          "chosen_category": "functional",
          "chosen_label": "space",
          "diagnosed_stage": "decode"
        },
        {
          "step": 2,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.3982073897495866
          },
          "top1_piece": " satellites",
          "top1_category": "semantic",
          "chosen_piece": " satellites",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 3,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " move",
          "top1_category": "semantic",
          "chosen_piece": " move",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 4,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " around",
          "top1_category": "semantic",
          "chosen_piece": " around",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 5,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " planets",
          "top1_category": "semantic",
          "chosen_piece": " planets",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 6,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
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
          "step": 7,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0372649282217026,
            "music": 0.10249900519847871
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
          "step": 8,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.2179216533899306,
            "music": 0.1195145070552826
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "1",
          "top1_category": "punct",
          "chosen_piece": "1",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 9,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.2179216533899306,
            "music": 0.1195145070552826
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
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.2179216533899306,
            "music": 0.1195145070552826
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " **",
          "top1_category": "punct",
          "chosen_piece": " **",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 11,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.2179216533899306,
            "music": 0.1195145070552826
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "Understanding",
          "top1_category": "semantic",
          "chosen_piece": "Understanding",
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
    "The pianist Lucy wants distribute \\( ABC$ triangle}\\]Consider $\\omega_-(side)$ denotes circum",
    "Quantum systems cryptography aims towards computing models running inside computers．____body（交通工具) environments.\"\n \n ",
    "The rainforest chicken Cass spp），被认为是大熊猫、亚马逊地区的“竞争对手”，但我们都知道，实际上巧克力冰淇淋"
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
  "output_a": "The pianist piano piano donald duck ducks `@don <EMAIL>`⁈disjon⁢tion",
  "output_b": "The pianist piano piano music finger fingers hands class Chopin Chopins nocturn\n\nAdd links within paragraphs",
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
    "The pianist piano concert của piano concerts - Tin tức mới nhất | Vandong.com\nanh love �",
    "The telescope piano noct hours Chop perfect difficult practiced 想要弹好钢琴，赵老师的建议",
    "The trader market stock volatility session experienced significant pullbacks yesterday ，但大盘并没有受到影响。这句话是什么类型的",
    "The child everyday simple professor rel explained � wine said 我有一个好朋友，他是一个教授。填"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```