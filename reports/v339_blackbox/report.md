# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1268.9s`
- Passed: `15/26`
- Mode: fully external runner, no reuse of module-internal `test()`
- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 427.6649169921875, "recon": 3.035787582397461, "contrast": 17888.765625, "holonomy": 5204.912109375, "write_policy": 1.2801257371902466, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.7864537239074707, "vocab_anchor": -0.0, "semantic_alignment": 9.940794944763184, "tail_semantic_anchor": 10.398346900939941, "functional_suppression": 0.0, "grad_norms": {"ctx_encoder": 4.851579219790189e-12, "fib_encoder": 1.95768950227421e-09, "dir_predictor": 0.0, "fiber_connection": 4.765428228630466e-08, "fiber_attn": 5.273471775498173e-11, "reranker": 7.259478509676653e-14, "qformer": 2.7505370092025723e-09, "content_bypass": 5.342654388072696e-10, "semantic_probe": 0.0, "layer_pool": 1.981449031518423e-07, "prefix_aligner": 4.958630397654052e-11, "vocab_proj": 1.00001461006052, "tail_head": 2.1919094034663747e-09, "context_heads": 4.204683201766042e-10, "memory_context_encoder": 5.857283409289072e-10}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy": 0.1, "semantic_probe": 0.3, "dir_diversity": 0.1, "reranker_ranking": 0.2, "vocab_an
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist piano piano hands 클래스 Java Class Marquee Paintbrush red blue green pink orange brush Hands _\\ntagliberty 标"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. practiced practiced dns lookup failed:default=true timeout=alt_dns_client_send alt-headers\npractice tutoring helps individuals continuously improve", "space_output": "Tell me something about practice and performance. signatures captured spectral telescope stars nebula distant\\nمعلومات مهمة حول التعلم自动化 TensorFlow ？\n  \r\n\r\nlua 编", "outputs_differ": true}
- `FAIL` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Blake focused focus areas within mort stock photography? google explain graphite mining career\\nPreviousHuman relations\nCertainly! Improving technique generally means focusing on refining techniques related", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refers referring skills – pacing, tone etc Understandability Tone means clarity- clear language\n对未来职业生涯有什么 plans ?\n对公司：做一个自媒体公司的CEO，怎么样", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. matter structure studies expansion universe dark large scale\\nExamin
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. student coding cloud computing environment, install Docker，install Describe PHP/Node/node-red 插\n 得知识：\nUbuntu 安全自动、", "space_output": "Describe the most important details a student should notice. neb stars telescope captured signatures spectral distant galaxy clusters using a custom telescope – Reuters\n\nAdd compound rooting PHP framework\nMetadata):\nName: DescribeDescribe", "music_margin": 0.0, "space_margin": 0.045454545454545456, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. practiced python file descriptor management?', memory profiling，文件是一种特殊的Python的数据类型的另一种。\ndefining defining statement： 定\r\nary practicing\n\n", "space_output": "Summarize the key ideas a learner should practice and remember. telescope neb signatures captured spectral stars distant from Earth telescope：是一种专门为科学研究星\n  XML+xml supports both human-readable and machine readable formats xml", "music_margin": 0.05882352941
- `PASS` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist pian pian event google chrome dev channel background keyboard 它的意思 expressive.js是什么瀏覽器JavaScript庫？在《王者荣耀》modarkets: [\"", "token_count": 13, "unique_token_ratio": 0.9230769230769231, "repeated_bigram_ratio": 0.0, "max_token_run": 2, "punct_ratio": 0.05384615384615385, "newline_ratio": 0.0, "alpha_ratio": 0.8461538461538461, "content_token_ratio": 0.8461538461538461, "generated_preview": "pian pian event google chrome dev channel background keyboard expressive js javascript modarkets"}, {"prompt": "The telescope", "output": "The telescope telescope javascript library is freeware． Visit https://github.com/enginemakerstudio/android-sdk-builder\nuemino gamer 它是一个高效的、多功能", "token_count": 15, "unique_token_ratio": 1.0, "repeated_bigram_ratio": 0.0, "max_token_run": 1, "punct_ratio": 0.06993006993006994, "newline_ratio": 0.006993006993006993, "alpha_ratio": 0.8531468531468531, "content_token_ratio": 0.8, "generated_preview": "telescope javascript library is freeware visit https github com enginemakerstudio android sdk builder uemino gamer"}, {"prompt": "The forest path", "output": "The forest path space galaxies stellar distant
- `PASS` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.36029958724975586, "l2_shift": 1058.7952880859375, "topk_overlap_count": 3, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 5.282861709594727, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.08333333333333333, "total_segments": 12, "bad_segments": 1, "early_collapse_prompts": ["The pianist"]}, "rows": [{"prompt": "The pianist", "output": "The pianist pian piano beats\\n这个问题看起来包含了以下几个方面的不确定、不确定性，这些问题的核心是如何理解和解决问题。解决方案：\n-validator/java: 首首先要尽可能提供更多相关信息：提供了的信息“<DoubleValidator/>永远不会出现在html的标准java。\n在这种情况下，您可以", "generated_token_count": 9, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["pian", "piano", "beats", "n", "validator", "java", "doublevalidator", "html"], "unique_ratio": 1.0, "content_ratio": 0.875, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.125}, {"segment_idx": 1, "tokens": ["java"], "unique_ratio": 1.0, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 1.0}], "bad_segments": [{"segment_idx": 1, "tokens": ["java"], "unique_ratio": 1.0, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 1.0}], "first_bad_segment_idx": 1}, {"prompt": "The telescope", "output": "The telescope telescope package installs telescope extensions dynamically::yesAccelerationSensorPackagepackage.java\nimport androidx.annotation.Nullable;\r\n    import android.content.Intent;\r\n\
- `PASS` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 3, "decoded_output": "Key piano ideas include piano music played by a single player, piano music played by a group of players", "rows": [{"step": 0, "top1": {"token_id": 26278, "piece": " piano", "norm": "piano", "logit": 14.3125, "prob": 0.01961374282836914}, "top1_category": "semantic", "topk_category_counts": {"semantic": 10, "functional": 2, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.08133493596687913, "functional": 0.01555199222639203, "punct": 0.0}, "chosen_token_id": 26278, "chosen_piece": " piano", "chosen_norm": "piano", "chosen_category": "semantic"}, {"step": 1, "top1": {"token_id": 4627, "piece": " music", "norm": "music", "logit": 16.0, "prob": 0.12476923316717148}, "top1_category": "semantic", "topk_category_counts": {"semantic": 11, "functional": 1, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.3615370336920023, "functional": 0.021681642159819603, "punct": 0.0}, "chosen_token_id": 4627, "chosen_piece": " music", "chosen_norm": "music", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 6342, "piece": " played", "norm": "played", "logit": 16.125, "prob": 0.043799854815
- `PASS` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 2, "retrieval_miss": 0, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [1, 0, 3, 6, 2], "retrieved_label_counts": {"music": 4, "space": 1}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A musician refined finger technique, phrasing, and pedal control on the piano.", "The pianist practiced arpeggios and Chopin nocturnes until midnight.", "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."], "output": "What improves piano technique and musical phrasing? piano technique refers to the technique musician uses when playing a piece of music, includiung finger techniques.\nPaperReferenceImprovementsinthematthew", "music_score": 0.35294117647058826, "space_score": 0.0, "generated_label":
- `FAIL` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": null, "retrieval_strength__bad_decode_score": 0.27825978352296893, "prefix_l2__bad_decode_score": null}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 1, "score": 0.5666224956512451}, {"mid": 0, "score": 0.1936155676841736}, {"mid": 3, "score": 0.06319719552993774}, {"mid": 6, "score": 0.02747329771518707}, {"mid": 5, "score": 0.02009677290916443}], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieval_strength": 0.8234352588653564, "prefix_l2_shift": 322359623680.0, "prefix_js_divergence": 0.4568025469779968, "top1_with_prefix": {"token_id": 14566, "piece": " Options", "norm": "options", "logit": 13.0, "prob": 0.1904912143945694}, "top1_category_with_prefix": "semantic", "topk_non_semantic_prob_mass": 0.0}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": 0.5422837436199188}, {"mid": 4, "score": 0.04626110792160035}, {"mid": 6, "score": 0.04496051967144013}, {"mid": 0, "score": 0.007697209715843201}, {"mid": 1, "score": -0.006330269575119014}], "retrieved_label_
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? Impro vis techniques, such as the use of the thumb,", "stage_counts": {"inject": 5, "decode": 5, "aligned": 2}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " Impro", "top1_category": "semantic", "chosen_piece": " Impro", "chosen_category": "semantic", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist Hannah wants balloons proportional weights totaling $\\( NSS_{players}$ grams combined, placed along number", "Quantum systems cryptography aims towards computing models running inside computers．____body（交通工具) environments.\nembedded\n\n", "The rainforest chicken beetle Halitter concinnipes reproduces ____. consumption method.\nOptions:\\nEnteromy"], "unique_count": 3}
- `FAIL` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist piano piano keys white feet happy singing music yellow purple green plant animal dog cat vehicle cool fast", "output_b": "The pianist piano piano keys white feet happy singing music yellow purple green plant grass red blue pink orange teal"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist piano piano pads perfect Japan Festival 〜未来のロックは「鍵 keyboard>\n`;", "The telescope restaurant wine pair meal course exquisite five served five course meal experience restaurant Bangkok Thailand thai Thai cuisine", "The trader market stock volatility session guide | significantfitness.com\\nSkip to\n\nDK Williams: Volatility", "The child course exquisite five pair wine restaurant meal served wine five The sentence compressor compress\nDKDXDNA"], "exact_same": false, "prefix_only": false, "too_short": false}
- `FAIL` `rerank_stability_probe`: {"status": "fail", "pairs": [{"pair": "music_P1", "prompt_a": "What improves piano technique and musical phrasing?", "prompt_b": "How can one improve piano technique and musical expression?", "top5_a": [1], "top5_b": [1], "jaccard": 1.0, "spearman_shared": 0.0, "pair_passed_jaccard_0_6": true}, {"pair": "space_P2", "prompt_a": "What explains satellites and orbital motion?", "prompt_b": "What describes satellites and the motion of planets?", "top5_a": [5], "top5_b": [5], "jaccard": 1.0, "spearman_shared": 0.0, "pair_passed_jaccard_0_6": true}], "spearman_best": 0.0, "gating": "hard_PASS"}
- `PASS` `decode_repetition_feedback_probe`: {"status": "pass", "per_prompt": [{"prompt": "The telescope", "output": "The telescope telescope jac cbd engine jer 缊enz？ thank anyway妥我就想知道该怎么加入了Google+ 我的名字是什么?\n大家都知道，如果你想在我的个人信息（", "max_repeat_per_content_token": 1, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The pianist", "output": "The pianist pian piano pianESSondersThus Honourable Thushonanteenthonanthesondereson\n　ス・thussonantothon sonanseughterteenthHon", "max_repeat_per_content_token": 2, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The market analyst", "output": "The market analyst market analyst organisations organizations market manager managers Err...</value），Autivity组织发布了全新的organizations)err…”） Autonomy公司的新产品Marketplace MarketplaceErr", "max_repeat_per_content_token": 2, "first_bigram_repeat_index": null, "trigram_lock_count": 0}], "avg_max_repeat_per_content_token": 1.6666666666666667, "min_first_bigram_repeat_index": null, "avg_trigram_lock_count": 0.0, "conditions": {"avg_max_repeat_le_3": true, "min_first_bigram_ge_4": true, "avg_trigram_lock_le_1": true}, "gating": "hard_PASS"}
- `FAIL` `functional_token_suppression_probe`: {"status": "fail", "per_prompt": [{"prompt": "A strong explanation should mention", "top12_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 18.375, "prob": 0.0198421198874712}, {"token_id": 1378, "piece": " two", "norm": "two", "logit": 18.125, "prob": 0.01545305922627449}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.125, "prob": 0.01545305922627449}, {"token_
- `FAIL` `keyword_specific_tail_slot_probe`: {"status": "fail", "per_memory": [{"mid": 0, "source_preview": "The pianist practiced arpeggios and Chopin nocturnes until m", "rare_keyword_ids": [43564, 32333], "rare_keyword_pieces": [" practiced", " midnight"], "tail_slot_top3_ids": [72977, 44903, 18905], "tail_slot_top3_pieces": ["*-", "-*", " Limited"], "intersection_size": 0}, {"mid": 1, "source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [26278, 37191, 14762], "rare_keyword_pieces": [" piano", " refined", " technique"], "tail_slot_top3_ids": [44903, 72977, 26921], "tail_slot_top3_pieces": ["-*", "*-", "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"], "intersection_size": 0}, {"mid": 2, "source_preview": "Classical interpretation often depends on dynamics, tempo ru", "rare_keyword_ids": [5796, 13798, 29195], "rare_keyword_pieces": [" touch", " depends", " dynamics"], "tail_slot_top3_ids": [72977, 18905, 26921], "tail_slot_top3_pieces": ["*-", " Limited", "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"], "intersection_size": 0}, {"mid": 3, "source_preview": "A conservatory student studied etudes, scales, and expressiv", "rare_keyword_ids": [77123, 11110, 19476], "rare_keyword_pieces": [" expressive", "
- `FAIL` `context_descriptor_cluster_probe`: {"status": "fail", "intra_music_mean_cos": 0.897429883480072, "intra_space_mean_cos": 0.845003604888916, "inter_domain_mean_cos": 0.7822778622309366, "gating": "PASS_or_not_implemented"}
- `FAIL` `prefix_length_scaling_probe`: {"status": "fail", "L_mem_A": 8, "L_mem_B": 16, "content_starters_top12_A": 3, "content_starters_top12_B": 2, "per_slot_mean_norm_A": 0.6365652978420258, "per_slot_mean_norm_B": 0.6361488550901413, "slot_norm_ratio_B_over_A": 0.9993457972759492, "top12_A": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 20.875, "prob": 0.4727327227592468}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 19.0, "prob": 0.07249591499567032}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 18.5, "prob": 0.043970994651317596}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.25, "prob": 0.03424464538693428}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 18.0, "prob": 0.026669755578041077}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 17.875, "prob": 0.02353597618639469}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 17.875, "prob": 0.02353597618639469}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 17.5, "prob": 0.016176024451851845}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 17.375, "prob": 0.014275291934609413}, {"token_id": 1378, "piece": " two", "norm": "two", "logit":
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
    "total": 427.6649169921875,
    "recon": 3.035787582397461,
    "contrast": 17888.765625,
    "holonomy": 5204.912109375,
    "write_policy": 1.2801257371902466,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 3.7864537239074707,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.940794944763184,
    "tail_semantic_anchor": 10.398346900939941,
    "functional_suppression": 0.0,
    "grad_norms": {
      "ctx_encoder": 4.851579219790189e-12,
      "fib_encoder": 1.95768950227421e-09,
      "dir_predictor": 0.0,
      "fiber_connection": 4.765428228630466e-08,
      "fiber_attn": 5.273471775498173e-11,
      "reranker": 7.259478509676653e-14,
      "qformer": 2.7505370092025723e-09,
      "content_bypass": 5.342654388072696e-10,
      "semantic_probe": 0.0,
      "layer_pool": 1.981449031518423e-07,
      "prefix_aligner": 4.958630397654052e-11,
      "vocab_proj": 1.00001461006052,
      "tail_head": 2.1919094034663747e-09,
      "context_heads": 4.204683201766042e-10,
      "memory_context_encoder": 5.857283409289072e-10
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
    2.1738640054724812e-10,
    5.3120880090518074e-12,
    3.4456504316437986e-10,
    1.1692245262262535e-11,
    2.033205159790441e-09,
    1.1537703431541146e-10
  ],
  "metric_param_deltas": [
    4.154892849328462e-06,
    5.3109705078213665e-08,
    6.780840976716718e-06,
    1.1688093337625105e-07,
    1.991574390558526e-05,
    1.1503167343107634e-06
  ],
  "max_metric_grad_norm": 2.033205159790441e-09,
  "max_metric_param_delta": 1.991574390558526e-05,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist piano piano hands 클래스 Java Class Marquee Paintbrush red blue green pink orange brush Hands _\\ntagliberty 标",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. practiced practiced dns lookup failed:default=true timeout=alt_dns_client_send alt-headers\npractice tutoring helps individuals continuously improve",
  "space_output": "Tell me something about practice and performance. signatures captured spectral telescope stars nebula distant\\nمعلومات مهمة حول التعلم自动化 TensorFlow ？\n  \r\n\r\nlua 编",
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Blake focused focus areas within mort stock photography? google explain graphite mining career\\nPreviousHuman relations\nCertainly! Improving technique generally means focusing on refining techniques related",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refers referring skills – pacing, tone etc Understandability Tone means clarity- clear language\n对未来职业生涯有什么 plans ?\n对公司：做一个自媒体公司的CEO，怎么样",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. matter structure studies expansion universe dark large scale\\nExamining Explain what someone **someone focus when improving technique? ** When explaining Explain understanding subject Analyise matter",
  "blank_music_score": 0.06666666666666667,
  "blank_space_score": 0.0,
  "music_music_score": 0.15,
  "music_space_score": 0.0,
  "space_space_score": 0.0,
  "space_music_score": 0.08333333333333333,
  "music_margin": 0.15,
  "space_margin": -0.08333333333333333,
  "music_lift": 0.08333333333333333,
  "space_lift": 0.0,
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
      "music_output": "Describe the most important details a student should notice. student coding cloud computing environment, install Docker，install Describe PHP/Node/node-red 插\n 得知识：\nUbuntu 安全自动、",
      "space_output": "Describe the most important details a student should notice. neb stars telescope captured signatures spectral distant galaxy clusters using a custom telescope – Reuters\n\nAdd compound rooting PHP framework\nMetadata):\nName: DescribeDescribe",
      "music_margin": 0.0,
      "space_margin": 0.045454545454545456,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. practiced python file descriptor management?', memory profiling，文件是一种特殊的Python的数据类型的另一种。\ndefining defining statement： 定\r\nary practicing\n\n",
      "space_output": "Summarize the key ideas a learner should practice and remember. telescope neb signatures captured spectral stars distant from Earth telescope：是一种专门为科学研究星\n  XML+xml supports both human-readable and machine readable formats xml",
      "music_margin": 0.058823529411764705,
      "space_margin": 0.05,
      "passed": true
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
      "output": "The pianist pian pian event google chrome dev channel background keyboard 它的意思 expressive.js是什么瀏覽器JavaScript庫？在《王者荣耀》modarkets: [\"",
      "token_count": 13,
      "unique_token_ratio": 0.9230769230769231,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 2,
      "punct_ratio": 0.05384615384615385,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8461538461538461,
      "content_token_ratio": 0.8461538461538461,
      "generated_preview": "pian pian event google chrome dev channel background keyboard expressive js javascript modarkets"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope javascript library is freeware． Visit https://github.com/enginemakerstudio/android-sdk-builder\nuemino gamer 它是一个高效的、多功能",
      "token_count": 15,
      "unique_token_ratio": 1.0,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.06993006993006994,
      "newline_ratio": 0.006993006993006993,
      "alpha_ratio": 0.8531468531468531,
      "content_token_ratio": 0.8,
      "generated_preview": "telescope javascript library is freeware visit https github com enginemakerstudio android sdk builder uemino gamer"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path space galaxies stellar distant observed deep evolution planet evolution galaxies formation observed的内容简介:\\\"ESA宣布，詹姆斯委员会理事会（SCP）已完成$ amount",
      "token_count": 15,
      "unique_token_ratio": 0.8,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.045454545454545456,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8571428571428571,
      "content_token_ratio": 0.8,
      "generated_preview": "space galaxies stellar distant observed deep evolution planet evolution galaxies formation observed esa scp amount"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market analyst market engineer การ -การตลาดออนไลุ-- Google\nzenith plumbing fixtures reviews\n\n```%\r\nIndentifying ${text}%Google Reviews",
      "token_count": 13,
      "unique_token_ratio": 0.7692307692307693,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.0784313725490196,
      "newline_ratio": 0.026143790849673203,
      "alpha_ratio": 0.7973856209150327,
      "content_token_ratio": 1.0,
      "generated_preview": "market analyst market engineer google zenith plumbing fixtures reviews indentifying text google reviews"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained analog rel professor everyday work involves explaining analog circuit design concepts and techniques to Darton, including the challenges faced in his field.\n\n作为一个",
      "token_count": 24,
      "unique_token_ratio": 0.9583333333333334,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.00975609756097561,
      "newline_ratio": 0.00975609756097561,
      "alpha_ratio": 0.848780487804878,
      "content_token_ratio": 0.7083333333333334,
      "generated_preview": "simple explained analog rel professor everyday work involves explaining analog circuit design concepts and techniques to darton including the challenges faced in his field"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.8901282051282052,
    "avg_repeated_bigram_ratio": 0.0,
    "avg_content_token_ratio": 0.8308974358974359,
    "avg_newline_ratio": 0.00857857908073116,
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
    "js_divergence": 0.36029958724975586,
    "l2_shift": 1058.7952880859375,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 5.282861709594727,
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
        "logit": 15.75,
        "prob": 0.14378677308559418
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 15.0,
        "prob": 0.06792005896568298
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 14.75,
        "prob": 0.05289619788527489
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 14.25,
        "prob": 0.032083164900541306
      },
      {
        "token_id": 4891,
        "piece": " �",
        "norm": "",
        "logit": 14.0,
        "prob": 0.024986395612359047
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.8125,
        "prob": 0.020714450627565384
      },
      {
        "token_id": 2014,
        "piece": " To",
        "norm": "to",
        "logit": 13.75,
        "prob": 0.019459422677755356
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.6875,
        "prob": 0.01828043721616268
      },
      {
        "token_id": 8908,
        "piece": " �",
        "norm": "",
        "logit": 13.6875,
        "prob": 0.01828043721616268
      },
      {
        "token_id": 49434,
        "piece": " �",
        "norm": "",
        "logit": 13.4375,
        "prob": 0.014236818999052048
      },
      {
        "token_id": 320,
        "piece": " (",
        "norm": "",
        "logit": 13.4375,
        "prob": 0.014236818999052048
      },
      {
        "token_id": 69162,
        "piece": " 对",
        "norm": "",
        "logit": 13.25,
        "prob": 0.011802736669778824
      }
    ]
  },
  "memory": {
    "js_divergence": 0.28038257360458374,
    "l2_shift": 322359623680.0,
    "topk_overlap_count": 5,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 6.851532459259033,
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
        "logit": 15.0625,
        "prob": 0.14501577615737915
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 13.5625,
        "prob": 0.03235739469528198
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.375,
        "prob": 0.02682522125542164
      },
      {
        "token_id": 30536,
        "piece": " Climate",
        "norm": "climate",
        "logit": 12.5,
        "prob": 0.01118241623044014
      },
      {
        "token_id": 10548,
        "piece": " According",
        "norm": "according",
        "logit": 12.4375,
        "prob": 0.010504907928407192
      },
      {
        "token_id": 17838,
        "piece": " Plant",
        "norm": "plant",
        "logit": 12.25,
        "prob": 0.008708873763680458
      },
      {
        "token_id": 7414,
        "piece": " Yes",
        "norm": "yes",
        "logit": 12.125,
        "prob": 0.007685554679483175
      },
      {
        "token_id": 58194,
        "piece": " Artificial",
        "norm": "artificial",
        "logit": 11.9375,
        "prob": 0.006371548864990473
      },
      {
        "token_id": 11097,
        "piece": " Human",
        "norm": "human",
        "logit": 11.875,
        "prob": 0.005985515657812357
      },
      {
        "token_id": 55313,
        "piece": " Quantum",
        "norm": "quantum",
        "logit": 11.875,
        "prob": 0.005985515657812357
      },
      {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 11.8125,
        "prob": 0.005622871685773134
      },
      {
        "token_id": 45451,
        "piece": " Understanding",
        "norm": "understanding",
        "logit": 11.8125,
        "prob": 0.005622871685773134
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
          "logit": 17.75,
          "prob": 0.12110888957977295
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.75,
          "prob": 0.12110888957977295
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.375,
          "prob": 0.08323684334754944
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.0,
          "prob": 0.05720778554677963
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.625,
          "prob": 0.039318300783634186
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 15.875,
          "prob": 0.018572650849819183
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 15.875,
          "prob": 0.018572650849819183
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 15.75,
          "prob": 0.016390305012464523
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.6875,
          "prob": 0.01539726834744215
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 15.5625,
          "prob": 0.013588041067123413
        },
        {
          "token_id": 13064,
          "piece": " facts",
          "norm": "facts",
          "logit": 15.5,
          "prob": 0.012764782644808292
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 15.25,
          "prob": 0.009941223077476025
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
          "logit": 17.875,
          "prob": 0.12463140487670898
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.875,
          "prob": 0.12463140487670898
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.5,
          "prob": 0.08565782755613327
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.125,
          "prob": 0.05887170508503914
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.75,
          "prob": 0.04046189412474632
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 15.9375,
          "prob": 0.017954858019948006
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 15.875,
          "prob": 0.016867026686668396
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.875,
          "prob": 0.016867026686668396
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 15.8125,
          "prob": 0.015845105051994324
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 15.625,
          "prob": 0.013136053457856178
        },
        {
          "token_id": 13064,
          "piece": " facts",
          "norm": "facts",
          "logit": 15.375,
          "prob": 0.010230368934571743
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 15.3125,
          "prob": 0.009610542096197605
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
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 16.75,
          "prob": 0.04907499626278877
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 16.625,
          "prob": 0.04330853000283241
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 16.625,
          "prob": 0.04330853000283241
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 16.125,
          "prob": 0.02626795321702957
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 16.0,
          "prob": 0.023181386291980743
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 15.75,
          "prob": 0.018053682520985603
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 15.625,
          "prob": 0.015932317823171616
        },
        {
          "token_id": 1850,
          "piece": " best",
          "norm": "best",
          "logit": 15.625,
          "prob": 0.015932317823171616
        },
        {
          "token_id": 10449,
          "piece": " presented",
          "norm": "presented",
          "logit": 15.5,
          "prob": 0.014060222543776035
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.5,
          "prob": 0.014060222543776035
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 15.375,
          "prob": 0.012408101931214333
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 15.3125,
          "prob": 0.01165633276104927
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
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 16.25,
          "prob": 0.03240004926919937
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 16.25,
          "prob": 0.03240004926919937
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 16.25,
          "prob": 0.03240004926919937
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 16.125,
          "prob": 0.028592942282557487
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 16.0,
          "prob": 0.025233183056116104
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.8125,
          "prob": 0.020919043570756912
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 15.6875,
          "prob": 0.018460992723703384
        },
        {
          "token_id": 1850,
          "piece": " best",
          "norm": "best",
          "logit": 15.6875,
          "prob": 0.018460992723703384
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 15.625,
          "prob": 0.017342496663331985
        },
        {
          "token_id": 10449,
          "piece": " presented",
          "norm": "presented",
          "logit": 15.4375,
          "prob": 0.014377434737980366
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 15.4375,
          "prob": 0.014377434737980366
        },
        {
          "token_id": 10007,
          "piece": " listed",
          "norm": "listed",
          "logit": 15.25,
          "prob": 0.011919312179088593
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
    "bad_segment_ratio": 0.08333333333333333,
    "total_segments": 12,
    "bad_segments": 1,
    "early_collapse_prompts": [
      "The pianist"
    ]
  },
  "rows": [
    {
      "prompt": "The pianist",
      "output": "The pianist pian piano beats\\n这个问题看起来包含了以下几个方面的不确定、不确定性，这些问题的核心是如何理解和解决问题。解决方案：\n-validator/java: 首首先要尽可能提供更多相关信息：提供了的信息“<DoubleValidator/>永远不会出现在html的标准java。\n在这种情况下，您可以",
      "generated_token_count": 9,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "pian",
            "piano",
            "beats",
            "n",
            "validator",
            "java",
            "doublevalidator",
            "html"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 1,
          "tokens": [
            "java"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "bad_segments": [
        {
          "segment_idx": 1,
          "tokens": [
            "java"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "first_bad_segment_idx": 1
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope package installs telescope extensions dynamically::yesAccelerationSensorPackagepackage.java\nimport androidx.annotation.Nullable;\r\n    import android.content.Intent;\r\n\r\npublic class acceleration>Yes {\r\n        public Yes() {}\r\n\r\n// Package Telescope\r\n@Override\r\n\r\n``` Certainly, continue sharing questions",
      "generated_token_count": 29,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "package",
            "installs",
            "telescope",
            "extensions",
            "dynamically",
            "yesaccelerationsensorpackagepackage",
            "java"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "import",
            "androidx",
            "annotation",
            "nullable",
            "import",
            "android",
            "content",
            "intent"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "public",
            "class",
            "acceleration",
            "yes",
            "public",
            "yes",
            "package",
            "telescope"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "override",
            "certainly",
            "continue",
            "sharing",
            "questions"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.2
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market analyst haha怎么解决了()。 最简单的市场分析师解决方案通常是：\n\n自动化数据分析\n\n**Step 你可以这样做：**\n\n### 步第一步**\n- **准备工作**: 确明确了项目的具体情况，准备好你需要的数据。如果你还没有Excel或其他",
      "generated_token_count": 5,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "analyst",
            "haha",
            "step",
            "excel"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.2
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained everyday analog rel professor的 response\\n>WelcomeWelcomeDear,\n\nThankfully! Please provide specific questions, equations or concepts related specifically. Providing clear examples and simple explanations often makes responses clearer.\n\nBest regards,<Premature Acceleraterecompile HTML",
      "generated_token_count": 34,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "simple",
            "explained",
            "everyday",
            "analog",
            "rel",
            "professor",
            "response",
            "n"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 1,
          "tokens": [
            "welcomewelcomedear",
            "thankfully",
            "please",
            "provide",
            "specific",
            "questions",
            "equations",
            "or"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 2,
          "tokens": [
            "concepts",
            "related",
            "specifically",
            "providing",
            "clear",
            "examples",
            "and",
            "simple"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 3,
          "tokens": [
            "explanations",
            "often",
            "makes",
            "responses",
            "clearer",
            "best",
            "regards",
            "premature"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 4,
          "tokens": [
            "acceleraterecompile",
            "html"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.5
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
      "decoded_output": "Key piano ideas include piano music played by a single player, piano music played by a group of players",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 14.3125,
            "prob": 0.01961374282836914
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.08133493596687913,
            "functional": 0.01555199222639203,
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
            "token_id": 4627,
            "piece": " music",
            "norm": "music",
            "logit": 16.0,
            "prob": 0.12476923316717148
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3615370336920023,
            "functional": 0.021681642159819603,
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
            "token_id": 6342,
            "piece": " played",
            "norm": "played",
            "logit": 16.125,
            "prob": 0.043799854815006256
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.2849879711866379,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 6342,
          "chosen_piece": " played",
          "chosen_norm": "played",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 553,
            "piece": " by",
            "norm": "by",
            "logit": 21.75,
            "prob": 0.4178875684738159
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 10,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.03894465509802103,
            "functional": 0.8358360398560762,
            "punct": 0.0
          },
          "chosen_token_id": 553,
          "chosen_piece": " by",
          "chosen_norm": "by",
          "chosen_category": "functional"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 264,
            "piece": " a",
            "norm": "a",
            "logit": 19.125,
            "prob": 0.27994075417518616
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.1409524055197835,
            "functional": 0.3802263978868723,
            "punct": 0.0
          },
          "chosen_token_id": 264,
          "chosen_piece": " a",
          "chosen_norm": "a",
          "chosen_category": "functional"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 3175,
            "piece": " single",
            "norm": "single",
            "logit": 17.375,
            "prob": 0.06261477619409561
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3670631628483534,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 3175,
          "chosen_piece": " single",
          "chosen_norm": "single",
          "chosen_category": "semantic"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 2781,
            "piece": " player",
            "norm": "player",
            "logit": 18.75,
            "prob": 0.1256874054670334
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 2,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.7289399281144142,
            "functional": 0.06324775516986847,
            "punct": 0.02474931813776493
          },
          "chosen_token_id": 2781,
          "chosen_piece": " player",
          "chosen_norm": "player",
          "chosen_category": "semantic"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 21.0,
            "prob": 0.3703763484954834
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 7,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.01843995228409767,
            "functional": 0.25822507217526436,
            "punct": 0.4818620551377535
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
            "logit": 19.0,
            "prob": 0.13581982254981995
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 10,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.15013512596488,
            "functional": 0.3152196370065212,
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
            "token_id": 4627,
            "piece": " music",
            "norm": "music",
            "logit": 21.0,
            "prob": 0.4348382353782654
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6424392014741898,
            "functional": 0.05203430540859699,
            "punct": 0.0
          },
          "chosen_token_id": 4627,
          "chosen_piece": " music",
          "chosen_norm": "music",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 6342,
            "piece": " played",
            "norm": "played",
            "logit": 23.875,
            "prob": 0.7752459645271301
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 6,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.8506145437713712,
            "functional": 0.0967525583691895,
            "punct": 0.0
          },
          "chosen_token_id": 6342,
          "chosen_piece": " played",
          "chosen_norm": "played",
          "chosen_category": "semantic"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 553,
            "piece": " by",
            "norm": "by",
            "logit": 26.125,
            "prob": 0.9073792695999146
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 9,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.005645772733259946,
            "functional": 0.9839160371338949,
            "punct": 0.0
          },
          "chosen_token_id": 553,
          "chosen_piece": " by",
          "chosen_norm": "by",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 264,
            "piece": " a",
            "norm": "a",
            "logit": 23.75,
            "prob": 0.4665880799293518
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 5,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.2823089915327728,
            "functional": 0.668856821488589,
            "punct": 0.002448429586365819
          },
          "chosen_token_id": 264,
          "chosen_piece": " a",
          "chosen_norm": "a",
          "chosen_category": "functional"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 1874,
            "piece": " group",
            "norm": "group",
            "logit": 21.375,
            "prob": 0.43949833512306213
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7507203156128526,
            "functional": 0.07890469115227461,
            "punct": 0.0
          },
          "chosen_token_id": 1874,
          "chosen_piece": " group",
          "chosen_norm": "group",
          "chosen_category": "semantic"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 315,
            "piece": " of",
            "norm": "of",
            "logit": 26.5,
            "prob": 0.9128096103668213
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 5,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.0011845091939903796,
            "functional": 0.927078340537264,
            "punct": 0.06882047471299302
          },
          "chosen_token_id": 315,
          "chosen_piece": " of",
          "chosen_norm": "of",
          "chosen_category": "functional"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 4217,
            "piece": " players",
            "norm": "players",
            "logit": 23.5,
            "prob": 0.5860422849655151
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 1,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.9454564405605197,
            "functional": 0.005745355971157551,
            "punct": 0.003948722034692764
          },
          "chosen_token_id": 4217,
          "chosen_piece": " players",
          "chosen_norm": "players",
          "chosen_category": "semantic"
        }
      ],
      "passed": true
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 4,
      "decoded_output": "Explain the topic clearly without adding extra words. 请解释一下这个话题。 请用中文",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 2041,
            "piece": " without",
            "norm": "without",
            "logit": 14.0625,
            "prob": 0.09739350527524948
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4008400971069932,
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
            "token_id": 10018,
            "piece": " changing",
            "norm": "changing",
            "logit": 18.375,
            "prob": 0.06668379157781601
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3837658204138279,
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
            "logit": 19.625,
            "prob": 0.24605980515480042
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7665509339421988,
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
            "prob": 0.607067346572876
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.936631160788238,
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
            "logit": 19.5,
            "prob": 0.26437923312187195
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 2,
            "punct": 10
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.14598949626088142,
            "punct": 0.7033802028745413
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
            "logit": 16.25,
            "prob": 0.14331494271755219
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 2,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.0342999417334795,
            "functional": 0.12657136842608452,
            "punct": 0.2797460984438658
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 14880,
            "piece": "请",
            "norm": "",
            "logit": 18.0,
            "prob": 0.22780393064022064
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
            "punct": 0.5994955552741885
          },
          "chosen_token_id": 14880,
          "chosen_piece": "请",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 104136,
            "piece": "解释",
            "norm": "",
            "logit": 17.25,
            "prob": 0.4806210994720459
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
            "punct": 0.8279070127755404
          },
          "chosen_token_id": 104136,
          "chosen_piece": "解释",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 100158,
            "piece": "一下",
            "norm": "",
            "logit": 16.875,
            "prob": 0.559518039226532
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
            "punct": 0.7913715112954378
          },
          "chosen_token_id": 100158,
          "chosen_piece": "一下",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 99487,
            "piece": "这个",
            "norm": "",
            "logit": 13.875,
            "prob": 0.13088639080524445
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
            "punct": 0.46059692464768887
          },
          "chosen_token_id": 99487,
          "chosen_piece": "这个",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 105167,
            "piece": "话题",
            "norm": "",
            "logit": 16.25,
            "prob": 0.2708602547645569
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
            "punct": 0.6634026970714331
          },
          "chosen_token_id": 105167,
          "chosen_piece": "话题",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 1773,
            "piece": "。",
            "norm": "",
            "logit": 18.875,
            "prob": 0.3294813632965088
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
            "punct": 0.8932928708381951
          },
          "chosen_token_id": 1773,
          "chosen_piece": "。",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 220,
            "piece": " ",
            "norm": "",
            "logit": 14.125,
            "prob": 0.13989514112472534
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 2,
            "punct": 10
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.03908655606210232,
            "punct": 0.3854087367653847
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 14880,
            "piece": "请",
            "norm": "",
            "logit": 18.125,
            "prob": 0.17980630695819855
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
            "punct": 0.6386299394071102
          },
          "chosen_token_id": 14880,
          "chosen_piece": "请",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 11622,
            "piece": "用",
            "norm": "",
            "logit": 17.0,
            "prob": 0.24853232502937317
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
            "punct": 0.6788338348269463
          },
          "chosen_token_id": 11622,
          "chosen_piece": "用",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 104811,
            "piece": "中文",
            "norm": "",
            "logit": 17.875,
            "prob": 0.466651976108551
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
            "punct": 0.864302983507514
          },
          "chosen_token_id": 104811,
          "chosen_piece": "中文",
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
      "output": "What improves piano technique and musical phrasing? piano technique refers to the technique musician uses when playing a piece of music, includiung finger techniques.\nPaperReferenceImprovementsinthematthew",
      "music_score": 0.35294117647058826,
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
      "output": "What explains satellites and orbital motion? satellites explains satellites在自然科学和社会科学研究有什么不同的解释自然界的现象\n发布时间zero Explanation in science zero零Explain科学领域的研究成果的社会社会科学的研究",
      "music_score": 0.0,
      "space_score": 0.5,
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
      "output": "Summarize the subject with concrete domain details. stars signatures captured neb telescope spectral distant galaxies and discovered a new subclass The sentence \"deep observationswith\n》揭示了一些重要的科学研究。\n\nAssistantist",
      "music_score": 0.0,
      "space_score": 0.11764705882352941,
      "generated_label": "space",
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
    "retrieval_strength__bad_decode_score": 0.27825978352296893,
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
      "prefix_js_divergence": 0.4568025469779968,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 13.0,
        "prob": 0.1904912143945694
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
      "prefix_js_divergence": 0.5339603424072266,
      "top1_with_prefix": {
        "token_id": 22201,
        "piece": " Choose",
        "norm": "choose",
        "logit": 10.625,
        "prob": 0.06567448377609253
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.08395162038505077
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
      "prefix_js_divergence": 0.44780248403549194,
      "top1_with_prefix": {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 11.0625,
        "prob": 0.04951095953583717
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
      "prefix_js_divergence": 0.2713753879070282,
      "top1_with_prefix": {
        "token_id": 10869,
        "piece": " Title",
        "norm": "title",
        "logit": 13.0,
        "prob": 0.05659090355038643
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
      "prefix_js_divergence": 0.3746432065963745,
      "top1_with_prefix": {
        "token_id": 5619,
        "piece": " playing",
        "norm": "playing",
        "logit": 13.0625,
        "prob": 0.01023199874907732
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.011350993067026138
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
      "prefix_js_divergence": 0.5071285963058472,
      "top1_with_prefix": {
        "token_id": 64591,
        "piece": " orbital",
        "norm": "orbital",
        "logit": 14.875,
        "prob": 0.03811535984277725
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
      "decoded_output": "What improves piano technique and musical phrasing? Impro vis techniques, such as the use of the thumb,",
      "stage_counts": {
        "inject": 5,
        "decode": 5,
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
            "space": 0.22133269011974335
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Impro",
          "top1_category": "semantic",
          "chosen_piece": " Impro",
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
            "music": 0.005353325977921486,
            "space": 0
          },
          "top1_piece": " vis",
          "top1_category": "functional",
          "chosen_piece": " vis",
          "chosen_category": "functional",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
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
            "music": 0.042543062940239906,
            "space": 0
          },
          "top1_piece": " techniques",
          "top1_category": "semantic",
          "chosen_piece": " techniques",
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
            "music": 1.0435107663273813,
            "space": 0.22133269011974335
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
            "music": 0.019319387152791023,
            "space": 0
          },
          "top1_piece": " such",
          "top1_category": "functional",
          "chosen_piece": " such",
          "chosen_category": "functional",
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
            "space": 0.22133269011974335
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " as",
          "top1_category": "functional",
          "chosen_piece": " as",
          "chosen_category": "functional",
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
            "music": 0.0469416044652462,
            "space": 0
          },
          "top1_piece": " the",
          "top1_category": "functional",
          "chosen_piece": " the",
          "chosen_category": "functional",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
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
            "music": 0.019595855847001076,
            "space": 0
          },
          "top1_piece": " use",
          "top1_category": "functional",
          "chosen_piece": " use",
          "chosen_category": "functional",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
        },
        {
          "step": 8,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0188743397593498,
            "space": 0.21808446049690247
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " of",
          "top1_category": "functional",
          "chosen_piece": " of",
          "chosen_category": "functional",
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
            "music": 1.0188743397593498,
            "space": 0.21808446049690247
          },
          "logits_label_mass": {
            "music": 0.04985291138291359,
            "space": 0
          },
          "top1_piece": " the",
          "top1_category": "functional",
          "chosen_piece": " the",
          "chosen_category": "functional",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
        },
        {
          "step": 10,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0188743397593498,
            "space": 0.21808446049690247
          },
          "logits_label_mass": {
            "music": 0.04726432077586651,
            "space": 0
          },
          "top1_piece": " thumb",
          "top1_category": "semantic",
          "chosen_piece": " thumb",
          "chosen_category": "semantic",
          "chosen_label": "music",
          "diagnosed_stage": "aligned"
        },
        {
          "step": 11,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 1.0188743397593498,
            "space": 0.21808446049690247
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
        }
      ],
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "decoded_output": "What explains satellites and orbital motion? Why don Juan Carlos I of Spain, who was born in",
      "stage_counts": {
        "inject": 11,
        "decode": 1
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
          "top1_piece": " Why",
          "top1_category": "functional",
          "chosen_piece": " Why",
          "chosen_category": "functional",
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
            "space": 0.007812274619936943
          },
          "top1_piece": " don",
          "top1_category": "functional",
          "chosen_piece": " don",
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
            "space": 0
          },
          "top1_piece": " Juan",
          "top1_category": "semantic",
          "chosen_piece": " Juan",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "inject"
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
          "top1_piece": " Carlos",
          "top1_category": "semantic",
          "chosen_piece": " Carlos",
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
          "top1_piece": " I",
          "top1_category": "functional",
          "chosen_piece": " I",
          "chosen_category": "functional",
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
          "top1_piece": " of",
          "top1_category": "functional",
          "chosen_piece": " of",
          "chosen_category": "functional",
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
          "top1_piece": " Spain",
          "top1_category": "semantic",
          "chosen_piece": " Spain",
          "chosen_category": "semantic",
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
          "top1_piece": ",",
          "top1_category": "punct",
          "chosen_piece": ",",
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
            "space": 1.0230449587106705,
            "music": 0.10908970832824708
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " who",
          "top1_category": "functional",
          "chosen_piece": " who",
          "chosen_category": "functional",
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
            "space": 1.0230449587106705,
            "music": 0.10908970832824708
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " was",
          "top1_category": "functional",
          "chosen_piece": " was",
          "chosen_category": "functional",
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
            "space": 1.0230449587106705,
            "music": 0.10908970832824708
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " born",
          "top1_category": "semantic",
          "chosen_piece": " born",
          "chosen_category": "semantic",
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
            "space": 1.0230449587106705,
            "music": 0.10908970832824708
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " in",
          "top1_category": "functional",
          "chosen_piece": " in",
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
    "The pianist Hannah wants balloons proportional weights totaling $\\( NSS_{players}$ grams combined, placed along number",
    "Quantum systems cryptography aims towards computing models running inside computers．____body（交通工具) environments.\nembedded\n\n",
    "The rainforest chicken beetle Halitter concinnipes reproduces ____. consumption method.\nOptions:\\nEnteromy"
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
  "output_a": "The pianist piano piano keys white feet happy singing music yellow purple green plant animal dog cat vehicle cool fast",
  "output_b": "The pianist piano piano keys white feet happy singing music yellow purple green plant grass red blue pink orange teal",
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
    "The pianist piano piano pads perfect Japan Festival 〜未来のロックは「鍵 keyboard>\n`;",
    "The telescope restaurant wine pair meal course exquisite five served five course meal experience restaurant Bangkok Thailand thai Thai cuisine",
    "The trader market stock volatility session guide | significantfitness.com\\nSkip to\n\nDK Williams: Volatility",
    "The child course exquisite five pair wine restaurant meal served wine five The sentence compressor compress\nDKDXDNA"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```