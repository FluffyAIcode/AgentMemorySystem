# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1418.4s`
- Passed: `17/26`
- Mode: fully external runner, no reuse of module-internal `test()`
- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 427.4331359863281, "recon": 2.7746963500976562, "contrast": 17888.765625, "holonomy": 5195.74267578125, "write_policy": 1.2801257371902466, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.658731698989868, "vocab_anchor": -0.0, "semantic_alignment": 9.940794944763184, "tail_semantic_anchor": 10.931857109069824, "functional_suppression": 0.0, "context_separation": 0.0, "grad_norms": {"ctx_encoder": 2.973141144412755e-17, "fib_encoder": 8.320122011585413e-15, "dir_predictor": 0.0, "fiber_connection": 1.530845612437877e-13, "fiber_attn": 6.35668824481696e-16, "reranker": 4.445083779216657e-19, "qformer": 2.545513612118145e-14, "content_bypass": 5.18099176084717e-15, "semantic_probe": 0.0, "layer_pool": 6.347911562717967e-13, "prefix_aligner": 3.9952105687971256e-15, "vocab_proj": 3.2042906851240517e-06, "tail_head": 1.0000022028380702, "context_heads": 3.910230471437595e-15, "memory_context_encoder": 5.605620010167894e-15}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy": 0.1, "semantic_probe": 0.3, "dir_diversity": 0.1, "rer
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist piano piano perfect piano Chopsticks Korean Piano Chop perfect Chop student perfect Baby\n\nMusic makes difficult hours pass quickly．(Rew"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. practiced practiced Cec practiced noct midnight Chop pian Chop midnight pian noct Chop noct pian midnight practiced performed perform act noct acted midnight Act", "space_output": "Tell me something about practice and performance. distant distant galaxies distant space stars nebula galaxies stellar evolution space deep space 备 stellar evolution deep deep galaxies evolution stellar space", "outputs_differ": true}
- `PASS` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Ohio Wesley Malone College Technology.\\document lang=\"ISO-ILESOME explain --ver  okay everyone welcome Welcome Mrs. Jones\n  \nWelcome, class.\n\nTrans", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique control technique finger placement pedal piano piano finger control pedal control finger pedal piano musician technique musician control finger movement pedal action musician piano playing\n\nAccording, �porno", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. explains move mechanics force gravitational planets 
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. student student studied student expressive keyboard skills scales keyboard scales expressive studied expressive scales studied keyboard student rat studied\nKeywords: **Important Details Notice Students Noticed", "space_output": "Describe the most important details a student should notice. structure mechanics large scale matter universe stars planets universe large structure matter scale universe matter large scale structure dark expansion matter studies universe\n studies expansion dark studies", "music_margin": 0.0, "space_margin": 0.034482758620689655, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. student keyboard student practiced scales, studied conserv keyboard studied scales conserv student scales studied keyboard conserv practice keyboard sum\n\npractice playing scales regularly study music theory", "space_output": "Summarize the key ideas a learner should practice and remember. studie
- `PASS` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist pian pian lovely pian Lovely <Speaker: piano musician piano finger pedal piano technique control finger technique pedal control musician finger control technique musician pedal musician piano", "token_count": 26, "unique_token_ratio": 0.34615384615384615, "repeated_bigram_ratio": 0.08, "max_token_run": 2, "punct_ratio": 0.009852216748768473, "newline_ratio": 0.0, "alpha_ratio": 0.8571428571428571, "content_token_ratio": 1.0, "generated_preview": "pian pian lovely pian lovely speaker piano musician piano finger pedal piano technique control finger technique pedal control musician finger control technique musician pedal"}, {"prompt": "The telescope", "output": "The telescope telescope telescope Instruments telescope instruments，这三个ending ending ing stars．____\nscience； stars stars science\n\n| Name | Age Group Number Type（", "token_count": 18, "unique_token_ratio": 0.6111111111111112, "repeated_bigram_ratio": 0.11764705882352941, "max_token_run": 2, "punct_ratio": 0.062111801242236024, "newline_ratio": 0.018633540372670808, "alpha_ratio": 0.8074534161490683, "content_token_ratio": 0.8888888888888888, "generated_preview": "
- `PASS` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.335856556892395, "l2_shift": 1038.77197265625, "topk_overlap_count": 3, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 5.558553218841553, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"toke
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.05, "total_segments": 20, "bad_segments": 1, "early_collapse_prompts": []}, "rows": [{"prompt": "The pianist", "output": "The pianist pian piano pian pianses 您这句话 piano piano difficult perfect practiced 念 Piano perfect difficult practiced perfect � practiced difficult imperfect embarrassed shy\n\nOkay，我知道perfect perfect pian practicing piano practice 懂 Chop midnight nocturn 音\n  \n", "generated_token_count": 29, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["pian", "piano", "pian", "pianses", "piano", "piano", "difficult", "perfect"], "unique_ratio": 0.625, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.375}, {"segment_idx": 1, "tokens": ["practiced", "piano", "perfect", "difficult", "practiced", "perfect", "practiced", "difficult"], "unique_ratio": 0.5, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.375}, {"segment_idx": 2, "tokens": ["imperfect", "embarrassed", "shy", "okay", "perfect", "perfect", "pian", "practicing"], "unique_ratio": 0.875, "content_ratio": 0.875, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.25}, {"segment_idx": 3, "tokens": ["piano", "practi
- `FAIL` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 0, "decoded_output": "Key piano ideas include key changes key signatures key signature key change key change key change key change key change", "rows": [{"step": 0, "top1": {"token_id": 1376, "piece": " key", "norm": "key", "logit": 14.0, "prob": 0.020741742104291916}, "top1_category": "functional", "topk_category_counts": {"semantic": 10, "functional": 2, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.07881386065855622, "functional": 0.02837220299988985, "punct": 0.0}, "chosen_token_id": 1376, "chosen_piece": " key", "chosen_norm": "key", "chosen_category": "functional"}, {"step": 1, "top1": {"token_id": 4344, "piece": " changes", "norm": "changes", "logit": 14.0, "prob": 0.05404780060052872}, "top1_category": "semantic", "topk_category_counts": {"semantic": 12, "functional": 0, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.2345929960720241, "functional": 0.0, "punct": 0.0}, "chosen_token_id": 4344, "chosen_piece": " changes", "chosen_norm": "changes", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 1376, "piece": " key", "norm": "key", "logit": 15.9375, "prob": 0.20206211507320404}
- `PASS` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 2, "retrieval_miss": 0, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [1, 0, 3, 6, 2], "retrieved_label_counts": {"music": 4, "space": 1}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A musician refined finger technique, phrasing, and pedal control on the piano.", "The pianist practiced arpeggios and Chopin nocturnes until midnight.", "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."], "output": "What improves piano technique and musical phrasing? technique piano technique technique musician piano piano finger finger musician finger control musician pedal pedal sustain control pedal control piano musician technique refined finger refined refined touch control", "music_score"
- `PASS` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": null, "retrieval_strength__bad_decode_score": 0.1951077207460111, "prefix_l2__bad_decode_score": null}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 1, "score": 0.5666224956512451}, {"mid": 0, "score": 0.1936155676841736}, {"mid": 3, "score": 0.06319719552993774}, {"mid": 6, "score": 0.02747329771518707}, {"mid": 5, "score": 0.02009677290916443}], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieval_strength": 0.8234352588653564, "prefix_l2_shift": 322359623680.0, "prefix_js_divergence": 0.4339211583137512, "top1_with_prefix": {"token_id": 14566, "piece": " Options", "norm": "options", "logit": 11.4375, "prob": 0.08918561786413193}, "top1_category_with_prefix": "semantic", "topk_non_semantic_prob_mass": 0.007792952004820108}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": 0.5422837436199188}, {"mid": 4, "score": 0.04626110792160035}, {"mid": 6, "score": 0.04496051967144013}, {"mid": 0, "score": 0.007697209715843201}, {"mid": 1, "score": -0.006330269575119014}
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? Options refer correctly. ① Playing with a metron", "stage_counts": {"inject": 10, "decode": 2}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " Options", "top1_category": "semantic", "chosen_piece": " Options", "chosen_category": "semantic", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "logits_label
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist decided balloons online calculator equation？\\Feb  Posts: Unknown Author：admin August-", "Quantum systems exhibit probabil behaviour half periodically occurs**: \\( ABC$ touches circle $\\omega_ⅰ.\n", "The rainforest smoke bill covered Sydney Smith Elementary（森林公园 elementary school）________ brightly lit houses.\nmuriling"], "unique_count": 3}
- `FAIL` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist piano youtube piano perfect piano nocturn difficult difficult perfect noct noct difficult girls ...\\nThe perfect hours", "output_b": "The pianist piano hours piano practiced piano practicing perfect difficult practiced hours perfect difficult difficultgirl hoursath perfect practiced"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist piano piano petit piano perfect ピ cell perfect perfect baby 产品经理·\n「 piano cél", "The telescope difficult piano perfect practiced hours Chop piano noct noct difficult perfect hours practiced Chop perfect piano hours difficult", "The trader market stock volatility session significant market experienced volatility significant stock experienced session market volatility experienced significant session stock", "The child simple everyday rel explained analog professor course restaurant professor simple explained everyday analog rel simple professor explained rel"], "exact_same": false, "prefix_only": false, "too_short": false}
- `PASS` `rerank_stability_probe`: {"status": "pass", "pairs": [{"pair": "music_P1", "prompt_a": "What improves piano technique and musical phrasing?", "prompt_b": "How can one improve piano technique and musical expression?", "top5_a": [1, 0, 6, 5, 7], "top5_b": [1, 0, 3, 6, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9621404708846248, "pair_passed_jaccard_0_6": true}, {"pair": "space_P2", "prompt_a": "What explains satellites and orbital motion?", "prompt_b": "What describes satellites and the motion of planets?", "top5_a": [5, 6, 4, 2, 7], "top5_b": [5, 6, 4, 0, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9999999999998858, "pair_passed_jaccard_0_6": true}], "spearman_best": 0.9999999999998858, "gating": "hard_PASS"}
- `FAIL` `decode_repetition_feedback_probe`: {"status": "fail", "per_prompt": [{"prompt": "The telescope", "output": "The telescope telescope telescope điện thoại telescope telescographief đội ngũ stars team stars nebula galaxy neb stars neb signatures spectral lines spectrum spectro spectral signatures signatures spectral telescope spect", "max_repeat_per_content_token": 3, "first_bigram_repeat_index": 17, "trigram_lock_count": 0}, {"prompt": "The pianist", "output": "The pianist pian pian piano pian perfect piano ヽ perfect perfect piano Chop house Chop House midnight midnight Chop noct midnight noct hours hours night hours noct pianist Chop Piano", "max_repeat_per_content_token": 3, "first_bigram_repeat_index": 9, "trigram_lock_count": 0}, {"prompt": "The market analyst", "output": "The market analyst market market stock market การ ตลาดการ stock stock volatility risk free interest rate LIB volatility volatility model stock price session data session session time market trading hours significant", "max_repeat_per_content_token": 4, "first_bigram_repeat_index": null, "trigram_lock_count": 0}], "avg_max_repeat_per_content_token": 3.3333333333333335, "min_first_bigram_repeat_index": 9, "avg_trigram_lock_count": 0.0, "conditions
- `PASS` `functional_token_suppression_probe`: {"status": "pass", "per_prompt": [{"prompt": "A strong explanation should mention", "top12_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 18.375, "prob": 0.0198421198874712}, {"token_id": 1378, "piece": " two", "norm": "two", "logit": 18.125, "prob": 0.01545305922627449}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.125, "prob": 0.01545305922627449}, {"token_
- `FAIL` `keyword_specific_tail_slot_probe`: {"status": "fail", "per_memory": [{"mid": 0, "source_preview": "The pianist practiced arpeggios and Chopin nocturnes until m", "rare_keyword_ids": [32333, 43564], "rare_keyword_pieces": [" midnight", " practiced"], "tail_slot_top3_ids": [0, 1, 2], "tail_slot_top3_pieces": ["!", "\"", "#"], "intersection_size": 0}, {"mid": 1, "source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [2524, 14317, 14762], "rare_keyword_pieces": [" control", " finger", " technique"], "tail_slot_top3_ids": [0, 1, 2], "tail_slot_top3_pieces": ["!", "\"", "#"], "intersection_size": 0}, {"mid": 2, "source_preview": "Classical interpretation often depends on dynamics, tempo ru", "rare_keyword_ids": [5796, 13798, 22845], "rare_keyword_pieces": [" touch", " depends", " interpretation"], "tail_slot_top3_ids": [0, 1, 2], "tail_slot_top3_pieces": ["!", "\"", "#"], "intersection_size": 0}, {"mid": 3, "source_preview": "A conservatory student studied etudes, scales, and expressiv", "rare_keyword_ids": [11110, 13625, 19476], "rare_keyword_pieces": [" conserv", " keyboard", " studied"], "tail_slot_top3_ids": [0, 1, 2], "tail_slot_top3_pieces": ["!", "\"", "#"], "intersect
- `FAIL` `context_descriptor_cluster_probe`: {"status": "fail", "intra_music_mean_cos": 0.10845918953418732, "intra_space_mean_cos": 0.3435296912988027, "inter_domain_mean_cos": 0.1921672224998474, "gating": "PASS_or_not_implemented"}
- `FAIL` `prefix_length_scaling_probe`: {"status": "fail", "L_mem_A": 8, "L_mem_B": 16, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 0.5570162907242775, "per_slot_mean_norm_B": 0.43695218116045, "slot_norm_ratio_B_over_A": 0.7844513498739678, "top12_A": [{"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.25, "prob": 0.15774640440940857}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 17.875, "prob": 0.10841741412878036}, {"token_id": 3807, "piece": " several", "norm": "several", "logit": 17.5, "prob": 0.07451412081718445}, {"token_id": 3170, "piece": " why", "norm": "why", "logit": 17.375, "prob": 0.06575848162174225}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 17.125, "prob": 0.05121275782585144}, {"token_id": 22845, "piece": " interpretation", "norm": "interpretation", "logit": 16.653064727783203, "prob": 0.03194620832800865}, {"token_id": 7966, "piece": " reasons", "norm": "reasons", "logit": 16.25, "prob": 0.021348653361201286}, {"token_id": 3040, "piece": " four", "norm": "four", "logit": 16.0, "prob": 0.016626348719000816}, {"token_id": 1376, "piece": " key", "norm": "key", "logit": 16.0, "prob": 0.01662634
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
    "total": 427.4331359863281,
    "recon": 2.7746963500976562,
    "contrast": 17888.765625,
    "holonomy": 5195.74267578125,
    "write_policy": 1.2801257371902466,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 3.658731698989868,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.940794944763184,
    "tail_semantic_anchor": 10.931857109069824,
    "functional_suppression": 0.0,
    "context_separation": 0.0,
    "grad_norms": {
      "ctx_encoder": 2.973141144412755e-17,
      "fib_encoder": 8.320122011585413e-15,
      "dir_predictor": 0.0,
      "fiber_connection": 1.530845612437877e-13,
      "fiber_attn": 6.35668824481696e-16,
      "reranker": 4.445083779216657e-19,
      "qformer": 2.545513612118145e-14,
      "content_bypass": 5.18099176084717e-15,
      "semantic_probe": 0.0,
      "layer_pool": 6.347911562717967e-13,
      "prefix_aligner": 3.9952105687971256e-15,
      "vocab_proj": 3.2042906851240517e-06,
      "tail_head": 1.0000022028380702,
      "context_heads": 3.910230471437595e-15,
      "memory_context_encoder": 5.605620010167894e-15
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
    6.901610689942646e-16,
    1.6816964010646052e-17,
    1.1124495566061329e-15,
    3.778151418302894e-17,
    6.573589346063656e-15,
    3.8364994152234046e-16
  ],
  "metric_param_deltas": [
    3.3562223507033195e-06,
    1.681696367447985e-13,
    5.7430720517004374e-06,
    3.778151836525412e-13,
    7.068017566780327e-07,
    3.836499694320761e-12
  ],
  "max_metric_grad_norm": 6.573589346063656e-15,
  "max_metric_param_delta": 5.7430720517004374e-06,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist piano piano perfect piano Chopsticks Korean Piano Chop perfect Chop student perfect Baby\n\nMusic makes difficult hours pass quickly．(Rew",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. practiced practiced Cec practiced noct midnight Chop pian Chop midnight pian noct Chop noct pian midnight practiced performed perform act noct acted midnight Act",
  "space_output": "Tell me something about practice and performance. distant distant galaxies distant space stars nebula galaxies stellar evolution space deep space 备 stellar evolution deep deep galaxies evolution stellar space",
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Ohio Wesley Malone College Technology.\\document lang=\"ISO-ILESOME explain --ver  okay everyone welcome Welcome Mrs. Jones\n  \nWelcome, class.\n\nTrans",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique control technique finger placement pedal piano piano finger control pedal control finger pedal piano musician technique musician control finger movement pedal action musician piano playing\n\nAccording, �porno",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. explains move mechanics force gravitational planets satellites explains force mechanics move planets gravitational satellites force move explains gravitational mechanics planets move satellites gravitational force explains planets mechanics satellites gravitation explain planetary",
  "blank_music_score": 0.043478260869565216,
  "blank_space_score": 0.0,
  "music_music_score": 0.45714285714285713,
  "music_space_score": 0.0,
  "space_space_score": 0.32432432432432434,
  "space_music_score": 0.02702702702702703,
  "music_margin": 0.45714285714285713,
  "space_margin": 0.2972972972972973,
  "music_lift": 0.4136645962732919,
  "space_lift": 0.32432432432432434,
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
      "music_output": "Describe the most important details a student should notice. student student studied student expressive keyboard skills scales keyboard scales expressive studied expressive scales studied keyboard student rat studied\nKeywords: **Important Details Notice Students Noticed",
      "space_output": "Describe the most important details a student should notice. structure mechanics large scale matter universe stars planets universe large structure matter scale universe matter large scale structure dark expansion matter studies universe\n studies expansion dark studies",
      "music_margin": 0.0,
      "space_margin": 0.034482758620689655,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. student keyboard student practiced scales, studied conserv keyboard studied scales conserv student scales studied keyboard conserv practice keyboard sum\n\npractice playing scales regularly study music theory",
      "space_output": "Summarize the key ideas a learner should practice and remember. studies large scale expansion structure matter universe dark matter dark universe studies expansion scale structure large universe matter studies scale dark structure expansion large matter studies universe scale",
      "music_margin": 0.03333333333333333,
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
      "output": "The pianist pian pian lovely pian Lovely <Speaker: piano musician piano finger pedal piano technique control finger technique pedal control musician finger control technique musician pedal musician piano",
      "token_count": 26,
      "unique_token_ratio": 0.34615384615384615,
      "repeated_bigram_ratio": 0.08,
      "max_token_run": 2,
      "punct_ratio": 0.009852216748768473,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8571428571428571,
      "content_token_ratio": 1.0,
      "generated_preview": "pian pian lovely pian lovely speaker piano musician piano finger pedal piano technique control finger technique pedal control musician finger control technique musician pedal"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope telescope Instruments telescope instruments，这三个ending ending ing stars．____\nscience； stars stars science\n\n| Name | Age Group Number Type（",
      "token_count": 18,
      "unique_token_ratio": 0.6111111111111112,
      "repeated_bigram_ratio": 0.11764705882352941,
      "max_token_run": 2,
      "punct_ratio": 0.062111801242236024,
      "newline_ratio": 0.018633540372670808,
      "alpha_ratio": 0.8074534161490683,
      "content_token_ratio": 0.8888888888888888,
      "generated_preview": "telescope telescope instruments telescope instruments ending ending ing stars science stars stars science name age group number type"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path observed evolution distant galaxies ● stellar space deep space stellar evolution deep galaxies distant observed Kent space galaxies stellar observed deep evolution space distant stellar galaxies observed �",
      "token_count": 26,
      "unique_token_ratio": 0.3076923076923077,
      "repeated_bigram_ratio": 0.04,
      "max_token_run": 1,
      "punct_ratio": 0.00904977375565611,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8552036199095022,
      "content_token_ratio": 0.8846153846153846,
      "generated_preview": "observed evolution distant galaxies stellar space deep space stellar evolution deep galaxies distant observed kent space galaxies stellar observed deep evolution space distant stellar"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market session stock market _-_CELL session stock session ant stock ＿cel market volatility cell\n\nABCDE［知识点管理制度 defaultProps volatility volatility stock",
      "token_count": 20,
      "unique_token_ratio": 0.45,
      "repeated_bigram_ratio": 0.05263157894736842,
      "max_token_run": 2,
      "punct_ratio": 0.02824858757062147,
      "newline_ratio": 0.011299435028248588,
      "alpha_ratio": 0.8418079096045198,
      "content_token_ratio": 0.75,
      "generated_preview": "market market session stock market cell session stock session ant stock cel market volatility cell abcde defaultprops volatility volatility stock"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained rel everyday analog professor explained simple rel analog everyday explained professor simple Explain rel analog everyday professor simplified explained analog easy rel professor everyday simple\n\n",
      "token_count": 27,
      "unique_token_ratio": 0.3333333333333333,
      "repeated_bigram_ratio": 0.07692307692307693,
      "max_token_run": 1,
      "punct_ratio": 0.0,
      "newline_ratio": 0.008368200836820083,
      "alpha_ratio": 0.8661087866108786,
      "content_token_ratio": 0.7037037037037037,
      "generated_preview": "simple explained rel everyday analog professor explained simple rel analog everyday explained professor simple explain rel analog everyday professor simplified explained analog easy rel"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.4096581196581196,
    "avg_repeated_bigram_ratio": 0.07344034293879495,
    "avg_content_token_ratio": 0.8454415954415954,
    "avg_newline_ratio": 0.007660235247547896,
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
    "js_divergence": 0.335856556892395,
    "l2_shift": 1038.77197265625,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 5.558553218841553,
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
        "prob": 0.126971036195755
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 14.5625,
        "prob": 0.07234591990709305
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 14.0,
        "prob": 0.041221458464860916
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.625,
        "prob": 0.02833106927573681
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.4375,
        "prob": 0.023487281054258347
      },
      {
        "token_id": 4891,
        "piece": " �",
        "norm": "",
        "logit": 13.375,
        "prob": 0.02206425741314888
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 13.3125,
        "prob": 0.020727451890707016
      },
      {
        "token_id": 49434,
        "piece": " �",
        "norm": "",
        "logit": 13.0625,
        "prob": 0.016142556443810463
      },
      {
        "token_id": 8908,
        "piece": " �",
        "norm": "",
        "logit": 12.9375,
        "prob": 0.014245755970478058
      },
      {
        "token_id": 320,
        "piece": " (",
        "norm": "",
        "logit": 12.9375,
        "prob": 0.014245755970478058
      },
      {
        "token_id": 1084,
        "piece": " It",
        "norm": "it",
        "logit": 12.9375,
        "prob": 0.014245755970478058
      },
      {
        "token_id": 18137,
        "piece": " �",
        "norm": "",
        "logit": 12.75,
        "prob": 0.011810146272182465
      }
    ]
  },
  "memory": {
    "js_divergence": 0.34661346673965454,
    "l2_shift": 322359623680.0,
    "topk_overlap_count": 2,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 6.496068000793457,
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
        "logit": 13.0,
        "prob": 0.0942673534154892
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 12.5625,
        "prob": 0.06086358055472374
      },
      {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 12.5625,
        "prob": 0.06086358055472374
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 11.6875,
        "prob": 0.025371713563799858
      },
      {
        "token_id": 9645,
        "piece": " Write",
        "norm": "write",
        "logit": 11.0625,
        "prob": 0.013580500148236752
      },
      {
        "token_id": 39565,
        "piece": " Provide",
        "norm": "provide",
        "logit": 11.0625,
        "prob": 0.013580500148236752
      },
      {
        "token_id": 14822,
        "piece": " Step",
        "norm": "step",
        "logit": 11.0625,
        "prob": 0.013580500148236752
      },
      {
        "token_id": 32911,
        "piece": " Topic",
        "norm": "topic",
        "logit": 10.9375,
        "prob": 0.0119847496971488
      },
      {
        "token_id": 10548,
        "piece": " According",
        "norm": "according",
        "logit": 10.9375,
        "prob": 0.0119847496971488
      },
      {
        "token_id": 21806,
        "piece": " Answer",
        "norm": "answer",
        "logit": 10.9375,
        "prob": 0.0119847496971488
      },
      {
        "token_id": 10869,
        "piece": " Title",
        "norm": "title",
        "logit": 10.75,
        "prob": 0.009935705922544003
      },
      {
        "token_id": 71287,
        "piece": " Explanation",
        "norm": "explanation",
        "logit": 10.5625,
        "prob": 0.008236989378929138
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
          "prob": 0.15976263582706451
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.5,
          "prob": 0.12442326545715332
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.125,
          "prob": 0.08551478385925293
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 16.625,
          "prob": 0.05186733230948448
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.375,
          "prob": 0.04039432108402252
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.25,
          "prob": 0.03564786538481712
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.75,
          "prob": 0.021621521562337875
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 15.4375,
          "prob": 0.01581864431500435
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 15.3125,
          "prob": 0.01395990327000618
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 15.0625,
          "prob": 0.01087198406457901
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 14.9375,
          "prob": 0.009594491682946682
        },
        {
          "token_id": 14976,
          "piece": " practical",
          "norm": "practical",
          "logit": 14.8125,
          "prob": 0.008467109873890877
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
          "logit": 18.0,
          "prob": 0.16751043498516083
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.625,
          "prob": 0.11512812972068787
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.25,
          "prob": 0.07912632822990417
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.125,
          "prob": 0.06982873380184174
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.875,
          "prob": 0.054382674396038055
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.25,
          "prob": 0.02910894900560379
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 16.125,
          "prob": 0.02568855881690979
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 15.6875,
          "prob": 0.016585780307650566
        },
        {
          "token_id": 13064,
          "piece": " facts",
          "norm": "facts",
          "logit": 15.6875,
          "prob": 0.016585780307650566
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.625,
          "prob": 0.015580898150801659
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 15.5625,
          "prob": 0.014636898413300514
        },
        {
          "token_id": 5257,
          "piece": " various",
          "norm": "various",
          "logit": 15.4375,
          "prob": 0.012917017564177513
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
          "logit": 14.25,
          "prob": 0.04233283922076225
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 14.125,
          "prob": 0.0373586006462574
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 14.0625,
          "prob": 0.035095155239105225
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 13.625,
          "prob": 0.022659137845039368
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 13.625,
          "prob": 0.022659137845039368
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 13.625,
          "prob": 0.022659137845039368
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 13.375,
          "prob": 0.017646951600909233
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 13.375,
          "prob": 0.017646951600909233
        },
        {
          "token_id": 10007,
          "piece": " listed",
          "norm": "listed",
          "logit": 13.3125,
          "prob": 0.01657777838408947
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 13.0,
          "prob": 0.012128561735153198
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 12.9375,
          "prob": 0.011393729597330093
        },
        {
          "token_id": 6959,
          "piece": " Option",
          "norm": "option",
          "logit": 12.8125,
          "prob": 0.010054930113255978
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
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 15.625,
          "prob": 0.07730851322412491
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 15.0625,
          "prob": 0.04404906556010246
        },
        {
          "token_id": 3545,
          "piece": " often",
          "norm": "often",
          "logit": 15.0625,
          "prob": 0.04404906556010246
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 14.375,
          "prob": 0.022149261087179184
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 14.25,
          "prob": 0.01954665407538414
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 13.9375,
          "prob": 0.014300637878477573
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 13.625,
          "prob": 0.010462569072842598
        },
        {
          "token_id": 4396,
          "piece": " correct",
          "norm": "correct",
          "logit": 13.5,
          "prob": 0.009233185090124607
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 13.4375,
          "prob": 0.008673775009810925
        },
        {
          "token_id": 3520,
          "piece": " actually",
          "norm": "actually",
          "logit": 13.3125,
          "prob": 0.007654579356312752
        },
        {
          "token_id": 10449,
          "piece": " presented",
          "norm": "presented",
          "logit": 13.3125,
          "prob": 0.007654579356312752
        },
        {
          "token_id": 4936,
          "piece": " simply",
          "norm": "simply",
          "logit": 13.3125,
          "prob": 0.007654579356312752
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
    "bad_segment_ratio": 0.05,
    "total_segments": 20,
    "bad_segments": 1,
    "early_collapse_prompts": []
  },
  "rows": [
    {
      "prompt": "The pianist",
      "output": "The pianist pian piano pian pianses 您这句话 piano piano difficult perfect practiced 念 Piano perfect difficult practiced perfect � practiced difficult imperfect embarrassed shy\n\nOkay，我知道perfect perfect pian practicing piano practice 懂 Chop midnight nocturn 音\n  \n",
      "generated_token_count": 29,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "pian",
            "piano",
            "pian",
            "pianses",
            "piano",
            "piano",
            "difficult",
            "perfect"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "practiced",
            "piano",
            "perfect",
            "difficult",
            "practiced",
            "perfect",
            "practiced",
            "difficult"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "imperfect",
            "embarrassed",
            "shy",
            "okay",
            "perfect",
            "perfect",
            "pian",
            "practicing"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "piano",
            "practice",
            "chop",
            "midnight",
            "nocturn"
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
      "prompt": "The telescope",
      "output": "The telescope telescope telescope stars telescope 螃 glitches stars stars captured nebula nebulous distant neb captured distant captured\nSpanish translation:\n\n Gad móvil gad rebote dulles telescope tel distant neb mobile vulgar glitch\n\n stars distant star captured far gas giant galaxy",
      "generated_token_count": 38,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "telescope",
            "stars",
            "telescope",
            "glitches",
            "stars",
            "stars",
            "captured"
          ],
          "unique_ratio": 0.5,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "nebula",
            "nebulous",
            "distant",
            "neb",
            "captured",
            "distant",
            "captured",
            "spanish"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "translation",
            "gad",
            "m",
            "vil",
            "gad",
            "rebote",
            "dulles",
            "telescope"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.5,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "tel",
            "distant",
            "neb",
            "mobile",
            "vulgar",
            "glitch",
            "stars",
            "distant"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "star",
            "captured",
            "far",
            "gas",
            "giant",
            "galaxy"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.6666666666666666,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.16666666666666666
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market Cũng như market stock – � stock stock significant som 没有什么 significant significant volatility 经 market experienced experienced volatility volatility experienced significant 有没有 stock � Ebony session volatility experience session volatile experienced � session vol average 持stockhold",
      "generated_token_count": 33,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "market",
            "c",
            "ng",
            "nh",
            "market",
            "stock",
            "stock"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 1,
          "tokens": [
            "stock",
            "significant",
            "som",
            "significant",
            "significant",
            "volatility",
            "market",
            "experienced"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        },
        {
          "segment_idx": 2,
          "tokens": [
            "experienced",
            "volatility",
            "volatility",
            "experienced",
            "significant",
            "stock",
            "ebony",
            "session"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "volatility",
            "experience",
            "session",
            "volatile",
            "experienced",
            "session",
            "vol",
            "average"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "stockhold"
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
            "stockhold"
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
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple everyday professor explained simple rel analog professor analog explained rel everyday explained ali simple analog rel professor everyday simple explained � Appropri professor rel soft analog everyday argali sim prof argument explain easy\nJapanese:\nProfessor explained example analogy relative Professor argued simply relationship",
      "generated_token_count": 43,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "simple",
            "everyday",
            "professor",
            "explained",
            "simple",
            "rel",
            "analog",
            "professor"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "analog",
            "explained",
            "rel",
            "everyday",
            "explained",
            "ali",
            "simple",
            "analog"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "rel",
            "professor",
            "everyday",
            "simple",
            "explained",
            "appropri",
            "professor",
            "rel"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "soft",
            "analog",
            "everyday",
            "argali",
            "sim",
            "prof",
            "argument",
            "explain"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 4,
          "tokens": [
            "easy",
            "japanese",
            "professor",
            "explained",
            "example",
            "analogy",
            "relative",
            "professor"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "argued",
            "simply",
            "relationship"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 1.0,
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
  "passed": false,
  "rows": [
    {
      "prompt": "Key piano ideas include",
      "first_bad_step": 0,
      "decoded_output": "Key piano ideas include key changes key signatures key signature key change key change key change key change key change",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 14.0,
            "prob": 0.020741742104291916
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.07881386065855622,
            "functional": 0.02837220299988985,
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
            "token_id": 4344,
            "piece": " changes",
            "norm": "changes",
            "logit": 14.0,
            "prob": 0.05404780060052872
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.2345929960720241,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 4344,
          "chosen_piece": " changes",
          "chosen_norm": "changes",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 15.9375,
            "prob": 0.20206211507320404
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 7,
            "functional": 5,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.17649920657277107,
            "functional": 0.300684982445091,
            "punct": 0.0
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 32628,
            "piece": " signatures",
            "norm": "signatures",
            "logit": 17.0,
            "prob": 0.13124355673789978
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.35870234202593565,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 32628,
          "chosen_piece": " signatures",
          "chosen_norm": "signatures",
          "chosen_category": "semantic"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 17.75,
            "prob": 0.3088439106941223
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 7,
            "functional": 2,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.09050555247813463,
            "functional": 0.3176051387563348,
            "punct": 0.20805970765650272
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 11957,
            "piece": " signature",
            "norm": "signature",
            "logit": 16.75,
            "prob": 0.05546201765537262
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.23835186567157507,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 11957,
          "chosen_piece": " signature",
          "chosen_norm": "signature",
          "chosen_category": "semantic"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 15.5,
            "prob": 0.24588018655776978
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 2,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.04452964477241039,
            "functional": 0.2597517976537347,
            "punct": 0.2561801830306649
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 2297,
            "piece": " change",
            "norm": "change",
            "logit": 16.625,
            "prob": 0.10650631785392761
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 0,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.2608667379245162,
            "functional": 0.0,
            "punct": 0.01633327268064022
          },
          "chosen_token_id": 2297,
          "chosen_piece": " change",
          "chosen_norm": "change",
          "chosen_category": "semantic"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 18.0,
            "prob": 0.3777182996273041
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 3,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.10207993909716606,
            "functional": 0.4594597755931318,
            "punct": 0.054172796197235584
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 2297,
            "piece": " change",
            "norm": "change",
            "logit": 17.75,
            "prob": 0.2065647542476654
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6627578469924629,
            "functional": 0.01476366026327014,
            "punct": 0.0
          },
          "chosen_token_id": 2297,
          "chosen_piece": " change",
          "chosen_norm": "change",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 17.875,
            "prob": 0.3116839826107025
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 4,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.08832715172320604,
            "functional": 0.4160269293934107,
            "punct": 0.07765554077923298
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 4344,
            "piece": " changes",
            "norm": "changes",
            "logit": 18.5,
            "prob": 0.2974588871002197
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7851861473172903,
            "functional": 0.0069955624639987946,
            "punct": 0.0
          },
          "chosen_token_id": 2297,
          "chosen_piece": " change",
          "chosen_norm": "change",
          "chosen_category": "semantic"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 18.875,
            "prob": 0.6846060752868652
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 6,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.016609814949333668,
            "functional": 0.7576055601239204,
            "punct": 0.06840430246666074
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 2297,
            "piece": " change",
            "norm": "change",
            "logit": 20.75,
            "prob": 0.7129963040351868
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 1,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.9608450214145705,
            "functional": 0.0020026599522680044,
            "punct": 0.0011410812148824334
          },
          "chosen_token_id": 2297,
          "chosen_piece": " change",
          "chosen_norm": "change",
          "chosen_category": "semantic"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 1376,
            "piece": " key",
            "norm": "key",
            "logit": 19.25,
            "prob": 0.8340669274330139
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 7,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.004959543235599995,
            "functional": 0.8862963512074202,
            "punct": 0.04490260942839086
          },
          "chosen_token_id": 1376,
          "chosen_piece": " key",
          "chosen_norm": "key",
          "chosen_category": "functional"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 2297,
            "piece": " change",
            "norm": "change",
            "logit": 22.0,
            "prob": 0.9026045203208923
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 1,
            "punct": 2
          },
          "topk_category_prob_mass": {
            "semantic": 0.9872476307791658,
            "functional": 0.0006410066271200776,
            "punct": 0.001166912610642612
          },
          "chosen_token_id": 2297,
          "chosen_piece": " change",
          "chosen_norm": "change",
          "chosen_category": "semantic"
        }
      ],
      "passed": false
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 4,
      "decoded_output": "Explain the topic clearly without adding extra words. 以下是“00000000",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 2041,
            "piece": " without",
            "norm": "without",
            "logit": 14.0,
            "prob": 0.09850539267063141
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3910854011774063,
            "functional": 0.011764791794121265,
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
            "prob": 0.09000641852617264
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.392568988725543,
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
            "prob": 0.27343612909317017
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.770895641297102,
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
            "prob": 0.6935112476348877
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.9403748787008226,
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
            "logit": 19.75,
            "prob": 0.3122035264968872
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
            "punct": 0.9225383130833507
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
            "logit": 14.875,
            "prob": 0.23103740811347961
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 0,
            "punct": 10
          },
          "topk_category_prob_mass": {
            "semantic": 0.06568178720772266,
            "functional": 0.0,
            "punct": 0.4368665777146816
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
            "logit": 13.4375,
            "prob": 0.1537264734506607
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
            "punct": 0.47032637521624565
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
            "logit": 9.875,
            "prob": 0.053394023329019547
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 0,
            "punct": 11
          },
          "topk_category_prob_mass": {
            "semantic": 0.010513906367123127,
            "functional": 0.0,
            "punct": 0.17506452836096287
          },
          "chosen_token_id": 2073,
          "chosen_piece": "“",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 8.25,
            "prob": 0.014365371316671371
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
            "punct": 0.06999299628660083
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 15.5625,
            "prob": 0.497150719165802
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 4,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.20614299923181534,
            "punct": 0.6485904697328806
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 19.875,
            "prob": 0.9884335398674011
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 1,
            "punct": 11
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.00014713883865624666,
            "punct": 0.997317543515237
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 18.75,
            "prob": 0.9916009902954102
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
            "punct": 0.9966894963290542
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 18.625,
            "prob": 0.9850409626960754
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
            "punct": 0.9947413181798765
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 18.5,
            "prob": 0.9921715259552002
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
            "punct": 0.9963783955245162
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 18.5,
            "prob": 0.9907309412956238
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
            "punct": 0.9968664359912509
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 15,
            "piece": "0",
            "norm": "",
            "logit": 18.5,
            "prob": 0.9886339902877808
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
            "punct": 0.997266473757918
          },
          "chosen_token_id": 15,
          "chosen_piece": "0",
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
      "output": "What improves piano technique and musical phrasing? technique piano technique technique musician piano piano finger finger musician finger control musician pedal pedal sustain control pedal control piano musician technique refined finger refined refined touch control",
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
      "output": "What explains satellites and orbital motion? satellites explains satellites înt explains planets explains gravitational gravitational planets gravitational satellites planets force không batt force force move move satellites sondern môn otherwise move planets sofort phải",
      "music_score": 0.0,
      "space_score": 0.35714285714285715,
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
      "output": "Summarize the subject with concrete domain details. matter structure studies large scale universe expansion dark matter studies dark universe structure large scale expansion matter universe studies structure expansion large dark scale studies matter universe чем",
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
    "retrieval_strength__bad_decode_score": 0.1951077207460111,
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
      "prefix_js_divergence": 0.4339211583137512,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 11.4375,
        "prob": 0.08918561786413193
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.007792952004820108
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
      "prefix_js_divergence": 0.5182103514671326,
      "top1_with_prefix": {
        "token_id": 13177,
        "piece": " Sat",
        "norm": "sat",
        "logit": 9.875,
        "prob": 0.052451349794864655
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.11109275184571743
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
      "prefix_js_divergence": 0.4658929109573364,
      "top1_with_prefix": {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 10.4375,
        "prob": 0.03619079291820526
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
      "prefix_js_divergence": 0.2656633257865906,
      "top1_with_prefix": {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 12.625,
        "prob": 0.040825504809617996
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.021852318197488785
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
      "prefix_js_divergence": 0.38104575872421265,
      "top1_with_prefix": {
        "token_id": 5619,
        "piece": " playing",
        "norm": "playing",
        "logit": 13.1875,
        "prob": 0.009058593772351742
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.005848668050020933
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
      "prefix_js_divergence": 0.459091454744339,
      "top1_with_prefix": {
        "token_id": 64591,
        "piece": " orbital",
        "norm": "orbital",
        "logit": 15.5,
        "prob": 0.04256873577833176
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
      "decoded_output": "What improves piano technique and musical phrasing? Options refer correctly. ① Playing with a metron",
      "stage_counts": {
        "inject": 10,
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
          "top1_piece": " Playing",
          "top1_category": "semantic",
          "chosen_piece": " Playing",
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
            "music": 0.9947564095258714,
            "space": 0.20747083127498628
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
          "step": 9,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.9947564095258714,
            "space": 0.20747083127498628
          },
          "logits_label_mass": {
            "music": 0.07759510725736618,
            "space": 0
          },
          "top1_piece": " a",
          "top1_category": "functional",
          "chosen_piece": " a",
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
            "music": 0.9947564095258714,
            "space": 0.20747083127498628
          },
          "logits_label_mass": {
            "music": 0.06662677228450775,
            "space": 0
          },
          "top1_piece": " met",
          "top1_category": "functional",
          "chosen_piece": " met",
          "chosen_category": "functional",
          "chosen_label": "music",
          "diagnosed_stage": "decode"
        },
        {
          "step": 11,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.9947564095258714,
            "space": 0.20747083127498628
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "ron",
          "top1_category": "functional",
          "chosen_piece": "ron",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        }
      ],
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "decoded_output": "What explains satellites and orbital motion? Why don Juan Carlos,  为什么Juan Carlos会说",
      "stage_counts": {
        "inject": 12
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
            "space": 0
          },
          "top1_piece": " don",
          "top1_category": "functional",
          "chosen_piece": " don",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "inject"
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
          "top1_piece": ",",
          "top1_category": "punct",
          "chosen_piece": ",",
          "chosen_category": "punct",
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
          "top1_piece": " ",
          "top1_category": "punct",
          "chosen_piece": " ",
          "chosen_category": "punct",
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
          "top1_piece": " ",
          "top1_category": "punct",
          "chosen_piece": " ",
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
          "top1_piece": "为什么",
          "top1_category": "punct",
          "chosen_piece": "为什么",
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
            "space": 1.010819947719574,
            "music": 0.10524652898311615
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "Juan",
          "top1_category": "semantic",
          "chosen_piece": "Juan",
          "chosen_category": "semantic",
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
            "space": 1.010819947719574,
            "music": 0.10524652898311615
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
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.010819947719574,
            "music": 0.10524652898311615
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "会",
          "top1_category": "punct",
          "chosen_piece": "会",
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
            "space": 1.010819947719574,
            "music": 0.10524652898311615
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "说",
          "top1_category": "punct",
          "chosen_piece": "说",
          "chosen_category": "punct",
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
    "The pianist decided balloons online calculator equation？\\Feb  Posts: Unknown Author：admin August-",
    "Quantum systems exhibit probabil behaviour half periodically occurs**: \\( ABC$ touches circle $\\omega_ⅰ.\n",
    "The rainforest smoke bill covered Sydney Smith Elementary（森林公园 elementary school）________ brightly lit houses.\nmuriling"
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
  "output_a": "The pianist piano youtube piano perfect piano nocturn difficult difficult perfect noct noct difficult girls ...\\nThe perfect hours",
  "output_b": "The pianist piano hours piano practiced piano practicing perfect difficult practiced hours perfect difficult difficultgirl hoursath perfect practiced",
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
    "The pianist piano piano petit piano perfect ピ cell perfect perfect baby 产品经理·\n「 piano cél",
    "The telescope difficult piano perfect practiced hours Chop piano noct noct difficult perfect hours practiced Chop perfect piano hours difficult",
    "The trader market stock volatility session significant market experienced volatility significant stock experienced session market volatility experienced significant session stock",
    "The child simple everyday rel explained analog professor course restaurant professor simple explained everyday analog rel simple professor explained rel"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```