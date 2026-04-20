# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1437.7s`
- Passed: `17/26`
- Mode: fully external runner, no reuse of module-internal `test()`
- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 427.68060302734375, "recon": 2.8443639278411865, "contrast": 17888.765625, "holonomy": 5195.59130859375, "write_policy": 1.2801257371902466, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.7805848121643066, "vocab_anchor": -0.0, "semantic_alignment": 9.940794944763184, "tail_semantic_anchor": 10.923386573791504, "functional_suppression": 0.0, "context_separation": 0.0, "grad_norms": {"ctx_encoder": 4.929302395458125e-12, "fib_encoder": 2.126063947075374e-09, "dir_predictor": 0.0, "fiber_connection": 4.755264918402146e-08, "fiber_attn": 4.398880306616116e-11, "reranker": 1.6243279556759868e-13, "qformer": 2.579903458092039e-09, "content_bypass": 4.978492139030914e-10, "semantic_probe": 0.0, "layer_pool": 1.981449884169706e-07, "prefix_aligner": 4.4659065018215345e-11, "vocab_proj": 1.00001461006052, "tail_head": 2.0125067302352785e-09, "context_heads": 4.757357009366309e-10, "memory_context_encoder": 6.375191430364839e-10}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy": 0.1, "semantic_probe": 0.3, "dir_diversity": 0.1, "
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist piano piano Chop noct hours perfect difficult practiced difficult perfect practiced Chop hours noct Touch touch piano piano Chop noct hours perfect difficult practiced"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. practiced practiced midnight Chop pian noct pian midnight Chop noct Practice finished Performance started Finished Perform practiced practiced midnight Chop pian noct pian midnight", "space_output": "Tell me something about practice and performance. distant stellar observed galaxies evolution space stars spectral space observed evolution stellar galaxies distant deep-field deep stellar observed galaxies evolution space distant.\n", "outputs_differ": true}
- `PASS` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. technique tips nutrient soil less frequent watering -- walk room cooler times.\nless caffeineHuman: Ohio weather experts predict high levels _______ record low temperatures.  Leading", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refined musician finger control pedal piano finger control refined piano pedal musician\nsubject technique technique refined musician finger control pedal piano finger control refined piano pedal musician apples oranges", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. explains mech
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. student student expressive keyboard studied scales conservatory conserv keyboard studied scales expressive\nUser Content student student expressive keyboard studied scales conserv school conserv keyboard studied scales", "space_output": "Describe the most important details a student should notice. explains large studies scale matter structure expansion neb large scale structure matter expansion studies universe dark dark universe large scale studies matter expansion structure large scale structure matter", "music_margin": 0.0, "space_margin": 0.03571428571428571, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. practiced student keyboard student expressive scales studied conserv conserv keyboard studied scales expressive\n\n生产经营活动中 ABC student keyboard student expressive scales studied conserv conserv keyboard studied scales", "space_output": "Summarize the key ideas a learner should practice and
- `FAIL` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist pian pian Chop practiced midnight nocturnal night midnight practiced Chop nocturnalsonalize pian pian Chop practiced midnight noctural nights midnight practiced Chop noct", "token_count": 22, "unique_token_ratio": 0.45454545454545453, "repeated_bigram_ratio": 0.2857142857142857, "max_token_run": 2, "punct_ratio": 0.0, "newline_ratio": 0.0, "alpha_ratio": 0.8736263736263736, "content_token_ratio": 1.0, "generated_preview": "pian pian chop practiced midnight nocturnal night midnight practiced chop nocturnalsonalize pian pian chop practiced midnight noctural nights midnight practiced chop noct"}, {"prompt": "The telescope", "output": "The telescope telescope telescope stars distant captured nebula spectral stars captured spectral neb distant signatures galaxy captures signatures telescope stars distant captured neb telescope spectral stars captured spectral neb", "token_count": 27, "unique_token_ratio": 0.37037037037037035, "repeated_bigram_ratio": 0.2692307692307692, "max_token_run": 2, "punct_ratio": 0.0, "newline_ratio": 0.0, "alpha_ratio": 0.8782608695652174, "content_token_ratio": 0.8888888888888888, "generated_preview
- `PASS` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.359661728143692, "l2_shift": 1056.75732421875, "topk_overlap_count": 3, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 5.285704612731934, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"toke
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.0, "total_segments": 24, "bad_segments": 0, "early_collapse_prompts": []}, "rows": [{"prompt": "The pianist", "output": "The pianist pian piano pian piano Chopsticks hours midnight Chop hours midnight noct noct difficult practiced perfect difficult perfect practiced pian Chop piano hours midnight pian Chop hours noct midnight difficult practiced perfect difficult perfect practiced noct piano\n\n解放军bucks pian pian Chop sticks hours midnight practiced difficult", "generated_token_count": 45, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["pian", "piano", "pian", "piano", "chopsticks", "hours", "midnight", "chop"], "unique_ratio": 0.75, "content_ratio": 0.875, "repeated_bigram_ratio": 0.14285714285714285, "dominant_token_share": 0.25}, {"segment_idx": 1, "tokens": ["hours", "midnight", "noct", "noct", "difficult", "practiced", "perfect", "difficult"], "unique_ratio": 0.75, "content_ratio": 0.875, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.25}, {"segment_idx": 2, "tokens": ["perfect", "practiced", "pian", "chop", "piano", "hours", "midnight", "pian"], "unique_ratio": 0.875, "content_ratio": 0.875, "repeated_bigram_ratio": 0.0,
- `PASS` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 3, "decoded_output": "Key piano ideas include playing music together, sharing stories, learning about each other's cultures, and creating", "rows": [{"step": 0, "top1": {"token_id": 5619, "piece": " playing", "norm": "playing", "logit": 13.9375, "prob": 0.014542710967361927}, "top1_category": "semantic", "topk_category_counts": {"semantic": 10, "functional": 2, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.08065968938171864, "functional": 0.013636151794344187, "punct": 0.0}, "chosen_token_id": 5619, "chosen_piece": " playing", "chosen_norm": "playing", "chosen_category": "semantic"}, {"step": 1, "top1": {"token_id": 4627, "piece": " music", "norm": "music", "logit": 17.625, "prob": 0.13350526988506317}, "top1_category": "semantic", "topk_category_counts": {"semantic": 11, "functional": 1, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.33706093300133944, "functional": 0.018067972734570503, "punct": 0.0}, "chosen_token_id": 4627, "chosen_piece": " music", "chosen_norm": "music", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 3786, "piece": " together", "norm": "together", "logit": 18.
- `PASS` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 2, "retrieval_miss": 0, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [1, 0, 3, 6, 2], "retrieved_label_counts": {"music": 4, "space": 1}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A musician refined finger technique, phrasing, and pedal control on the piano.", "The pianist practiced arpeggios and Chopin nocturnes until midnight.", "A conservatory student studied etudes, scales, and expressive voicing on the keyboard."], "output": "What improves piano technique and musical phrasing? technique piano technique piano musician musician finger pedal finger pedal refined control refined touch sensitive hand control technique technique piano musician musician finger pedal finger piano refined control", "music_score":
- `PASS` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": null, "retrieval_strength__bad_decode_score": 0.19993485319290968, "prefix_l2__bad_decode_score": null}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 1, "score": 0.5666224956512451}, {"mid": 0, "score": 0.1936155676841736}, {"mid": 3, "score": 0.06319719552993774}, {"mid": 6, "score": 0.02747329771518707}, {"mid": 5, "score": 0.02009677290916443}], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieval_strength": 0.8234352588653564, "prefix_l2_shift": 322359623680.0, "prefix_js_divergence": 0.39210981130599976, "top1_with_prefix": {"token_id": 14566, "piece": " Options", "norm": "options", "logit": 12.0625, "prob": 0.09792399406433105}, "top1_category_with_prefix": "semantic", "topk_non_semantic_prob_mass": 0.008556502871215343}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": 0.5422837436199188}, {"mid": 4, "score": 0.04626110792160035}, {"mid": 6, "score": 0.04496051967144013}, {"mid": 0, "score": 0.007697209715843201}, {"mid": 1, "score": -0.00633026957511901
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? Options often mentioned: practice, repetition, listening, and memor", "stage_counts": {"inject": 12}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " Options", "top1_category": "semantic", "chosen_piece": " Options", "chosen_category": "semantic", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 1.0435107663273813, "space": 0.22133269011974335}, "logits
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist Lucy wants distribute \\( ABC$ triangle}\\]Consider $\\omega_-(side)$ denotes circum", "Quantum systems cryptography aims towards computing models running inside computers．____body（交通工具) environments.\"\n \n ", "The rainforest chicken Cass spp），被认为是大熊猫、亚马逊地区的“竞争对手”，但我们都知道，实际上巧克力冰淇淋"], "unique_count": 3}
- `FAIL` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist piano piano Chop noct hours practiced perfect difficult difficult perfect practiced hours noct Chop pian class piano piano", "output_b": "The pianist piano piano Chop noct hours practiced perfect difficult difficult perfect practiced noct hours Chop pract act piano piano"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist piano piano Chop noct hours perfect practiced difficult difficult perfect practiced hours noct Chopinka（ piano piano", "The telescope piano Chop noct practiced difficult perfect hours piano difficult perfect practiced hours Chop noct adalah sebuah piano Chop", "The trader market volatility stock session experienced significant market stock experienced volatility session significant 您的问题“ market volatility", "The child simple rel everyday analog professor explained wine restaurant explained simple rel professor everyday analog benz\n\n simple rel"], "exact_same": false, "prefix_only": false, "too_short": false}
- `PASS` `rerank_stability_probe`: {"status": "pass", "pairs": [{"pair": "music_P1", "prompt_a": "What improves piano technique and musical phrasing?", "prompt_b": "How can one improve piano technique and musical expression?", "top5_a": [1, 0, 6, 5, 7], "top5_b": [1, 0, 3, 6, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9621404708846248, "pair_passed_jaccard_0_6": true}, {"pair": "space_P2", "prompt_a": "What explains satellites and orbital motion?", "prompt_b": "What describes satellites and the motion of planets?", "top5_a": [5, 6, 4, 2, 7], "top5_b": [5, 6, 4, 0, 7], "jaccard": 0.6666666666666666, "spearman_shared": 0.9999999999998858, "pair_passed_jaccard_0_6": true}], "spearman_best": 0.9999999999998858, "gating": "hard_PASS"}
- `FAIL` `decode_repetition_feedback_probe`: {"status": "fail", "per_prompt": [{"prompt": "The telescope", "output": "The telescope telescope telescope stars neb spectral signatures captured distant stars captured signatures neb spectral distant power capture telescope telescope stars neb spectral signatures captured distant stars captured signatures neb spectral distant", "max_repeat_per_content_token": 4, "first_bigram_repeat_index": 11, "trigram_lock_count": 2}, {"prompt": "The pianist", "output": "The pianist pian piano pian Chop piano practiced hours perfect perfect practiced Chop hours midnight nocturn difficult difficult noct midnight pian Chop practiced hours perfect perfect practiced pian hours midnight noct", "max_repeat_per_content_token": 3, "first_bigram_repeat_index": null, "trigram_lock_count": 0}, {"prompt": "The market analyst", "output": "The market analyst market market volatility stock experienced significant session volatility experienced session significant stock price fluctuations fluctuation market market volatility stock experienced significant session volatility experienced session significant stock experience signific", "max_repeat_per_content_token": 4, "first_bigram_repeat_index": 16, "trigram_loc
- `PASS` `functional_token_suppression_probe`: {"status": "pass", "per_prompt": [{"prompt": "A strong explanation should mention", "top12_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 18.375, "prob": 0.0198421198874712}, {"token_id": 1378, "piece": " two", "norm": "two", "logit": 18.125, "prob": 0.01545305922627449}, {"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.125, "prob": 0.01545305922627449}, {"token_
- `FAIL` `keyword_specific_tail_slot_probe`: {"status": "fail", "per_memory": [{"mid": 0, "source_preview": "The pianist practiced arpeggios and Chopin nocturnes until m", "rare_keyword_ids": [43564, 32333], "rare_keyword_pieces": [" practiced", " midnight"], "tail_slot_top3_ids": [44903, 21317, 1482], "tail_slot_top3_pieces": ["-*", "信", " current"], "intersection_size": 0}, {"mid": 1, "source_preview": "A musician refined finger technique, phrasing, and pedal con", "rare_keyword_ids": [26278, 37191, 14762], "rare_keyword_pieces": [" piano", " refined", " technique"], "tail_slot_top3_ids": [21317, 44903, 1482], "tail_slot_top3_pieces": ["信", "-*", " current"], "intersection_size": 0}, {"mid": 2, "source_preview": "Classical interpretation often depends on dynamics, tempo ru", "rare_keyword_ids": [5796, 13798, 29195], "rare_keyword_pieces": [" touch", " depends", " dynamics"], "tail_slot_top3_ids": [21317, 44903, 1482], "tail_slot_top3_pieces": ["信", "-*", " current"], "intersection_size": 0}, {"mid": 3, "source_preview": "A conservatory student studied etudes, scales, and expressiv", "rare_keyword_ids": [77123, 11110, 19476], "rare_keyword_pieces": [" expressive", " conserv", " studied"], "tail_slot_top3_ids": [21317, 44903,
- `FAIL` `context_descriptor_cluster_probe`: {"status": "fail", "intra_music_mean_cos": 0.30361711978912354, "intra_space_mean_cos": 0.3894611398379008, "inter_domain_mean_cos": 0.2902490322788556, "gating": "PASS_or_not_implemented"}
- `FAIL` `prefix_length_scaling_probe`: {"status": "fail", "L_mem_A": 8, "L_mem_B": 16, "content_starters_top12_A": 12, "content_starters_top12_B": 12, "per_slot_mean_norm_A": 0.6361142247915268, "per_slot_mean_norm_B": 0.6379217356443405, "slot_norm_ratio_B_over_A": 1.002841487868639, "top12_A": [{"token_id": 2326, "piece": " three", "norm": "three", "logit": 18.5, "prob": 0.16025178134441376}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 18.0, "prob": 0.09719762206077576}, {"token_id": 3807, "piece": " several", "norm": "several", "logit": 17.75, "prob": 0.07569757848978043}, {"token_id": 10295, "piece": " examples", "norm": "examples", "logit": 17.75, "prob": 0.07569757848978043}, {"token_id": 3170, "piece": " why", "norm": "why", "logit": 17.25, "prob": 0.045912906527519226}, {"token_id": 22845, "piece": " interpretation", "norm": "interpretation", "logit": 17.171527862548828, "prob": 0.04244775325059891}, {"token_id": 3040, "piece": " four", "norm": "four", "logit": 16.625, "prob": 0.024575406685471535}, {"token_id": 1376, "piece": " key", "norm": "key", "logit": 16.5, "prob": 0.021687719970941544}, {"token_id": 7966, "piece": " reasons", "norm": "reasons", "logit": 16.375, "prob": 0.0191393
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
    "total": 427.68060302734375,
    "recon": 2.8443639278411865,
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
    "context_separation": 0.0,
    "grad_norms": {
      "ctx_encoder": 4.929302395458125e-12,
      "fib_encoder": 2.126063947075374e-09,
      "dir_predictor": 0.0,
      "fiber_connection": 4.755264918402146e-08,
      "fiber_attn": 4.398880306616116e-11,
      "reranker": 1.6243279556759868e-13,
      "qformer": 2.579903458092039e-09,
      "content_bypass": 4.978492139030914e-10,
      "semantic_probe": 0.0,
      "layer_pool": 1.981449884169706e-07,
      "prefix_aligner": 4.4659065018215345e-11,
      "vocab_proj": 1.00001461006052,
      "tail_head": 2.0125067302352785e-09,
      "context_heads": 4.757357009366309e-10,
      "memory_context_encoder": 6.375191430364839e-10
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
    2.1235214975323657e-10,
    5.172948874809791e-12,
    3.4164582274343047e-10,
    1.1604914078311435e-11,
    2.0086585728051887e-09,
    1.1378280262430707e-10
  ],
  "metric_param_deltas": [
    4.119876848562853e-06,
    5.171920136604058e-08,
    6.768465027562343e-06,
    1.1600845084558387e-07,
    1.9677303498610854e-05,
    1.1344332051521633e-06
  ],
  "max_metric_grad_norm": 2.0086585728051887e-09,
  "max_metric_param_delta": 1.9677303498610854e-05,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist piano piano Chop noct hours perfect difficult practiced difficult perfect practiced Chop hours noct Touch touch piano piano Chop noct hours perfect difficult practiced",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. practiced practiced midnight Chop pian noct pian midnight Chop noct Practice finished Performance started Finished Perform practiced practiced midnight Chop pian noct pian midnight",
  "space_output": "Tell me something about practice and performance. distant stellar observed galaxies evolution space stars spectral space observed evolution stellar galaxies distant deep-field deep stellar observed galaxies evolution space distant.\n",
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. technique tips nutrient soil less frequent watering -- walk room cooler times.\nless caffeineHuman: Ohio weather experts predict high levels _______ record low temperatures.  Leading",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refined musician finger control pedal piano finger control refined piano pedal musician\nsubject technique technique refined musician finger control pedal piano finger control refined piano pedal musician apples oranges",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. explains mechanics move force gravitational planets satellites explains force gravitational move planets satellites mechanics moves explain explains mechanics move force gravitational planets satellites explains force gravitational move planets satellites mechanics\n机械化",
  "blank_music_score": 0.07142857142857142,
  "blank_space_score": 0.0,
  "music_music_score": 0.5675675675675675,
  "music_space_score": 0.0,
  "space_space_score": 0.3333333333333333,
  "space_music_score": 0.027777777777777776,
  "music_margin": 0.5675675675675675,
  "space_margin": 0.3055555555555555,
  "music_lift": 0.49613899613899615,
  "space_lift": 0.3333333333333333,
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
      "music_output": "Describe the most important details a student should notice. student student expressive keyboard studied scales conservatory conserv keyboard studied scales expressive\nUser Content student student expressive keyboard studied scales conserv school conserv keyboard studied scales",
      "space_output": "Describe the most important details a student should notice. explains large studies scale matter structure expansion neb large scale structure matter expansion studies universe dark dark universe large scale studies matter expansion structure large scale structure matter",
      "music_margin": 0.0,
      "space_margin": 0.03571428571428571,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. practiced student keyboard student expressive scales studied conserv conserv keyboard studied scales expressive\n\n生产经营活动中 ABC student keyboard student expressive scales studied conserv conserv keyboard studied scales",
      "space_output": "Summarize the key ideas a learner should practice and remember. studies large universe matter scale structure dark expansion large scale structure universe dark matter expansion studies studies large universe matter scale structure dark expansion large scale structure universe",
      "music_margin": 0.034482758620689655,
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
      "output": "The pianist pian pian Chop practiced midnight nocturnal night midnight practiced Chop nocturnalsonalize pian pian Chop practiced midnight noctural nights midnight practiced Chop noct",
      "token_count": 22,
      "unique_token_ratio": 0.45454545454545453,
      "repeated_bigram_ratio": 0.2857142857142857,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8736263736263736,
      "content_token_ratio": 1.0,
      "generated_preview": "pian pian chop practiced midnight nocturnal night midnight practiced chop nocturnalsonalize pian pian chop practiced midnight noctural nights midnight practiced chop noct"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope telescope stars distant captured nebula spectral stars captured spectral neb distant signatures galaxy captures signatures telescope stars distant captured neb telescope spectral stars captured spectral neb",
      "token_count": 27,
      "unique_token_ratio": 0.37037037037037035,
      "repeated_bigram_ratio": 0.2692307692307692,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8782608695652174,
      "content_token_ratio": 0.8888888888888888,
      "generated_preview": "telescope telescope stars distant captured nebula spectral stars captured spectral neb distant signatures galaxy captures signatures telescope stars distant captured neb telescope spectral stars"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path distant galaxies stellar space observed evolution deep distant space observed evolution galaxies stellar deep centre centres distant galaxies stellar space observed evolution deep distant space observed evolution galaxies",
      "token_count": 28,
      "unique_token_ratio": 0.32142857142857145,
      "repeated_bigram_ratio": 0.5185185185185185,
      "max_token_run": 1,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8734177215189873,
      "content_token_ratio": 0.8928571428571429,
      "generated_preview": "distant galaxies stellar space observed evolution deep distant space observed evolution galaxies stellar deep centre centres distant galaxies stellar space observed evolution deep distant"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market volatility experienced significant stock session, experienced volatility session significant stock strong growth. market market volatility experienced significant stock session&# experienced volatility session significant",
      "token_count": 25,
      "unique_token_ratio": 0.32,
      "repeated_bigram_ratio": 0.4583333333333333,
      "max_token_run": 2,
      "punct_ratio": 0.015748031496062992,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8779527559055118,
      "content_token_ratio": 0.84,
      "generated_preview": "market market volatility experienced significant stock session experienced volatility session significant stock strong growth market market volatility experienced significant stock session experienced volatility session"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple everyday explained professor analog rel simple explained rel professor everyday analog Dairy farms aren��� simple everyday explained professor analog rel simple explained rel professor everyday analog",
      "token_count": 27,
      "unique_token_ratio": 0.3333333333333333,
      "repeated_bigram_ratio": 0.4230769230769231,
      "max_token_run": 1,
      "punct_ratio": 0.012875536480686695,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8583690987124464,
      "content_token_ratio": 0.7037037037037037,
      "generated_preview": "simple everyday explained professor analog rel simple explained rel professor everyday analog dairy farms aren simple everyday explained professor analog rel simple explained rel"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.35993554593554594,
    "avg_repeated_bigram_ratio": 0.39097476597476594,
    "avg_content_token_ratio": 0.8650899470899471,
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
    "js_divergence": 0.2991044521331787,
    "l2_shift": 322359623680.0,
    "topk_overlap_count": 2,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 6.797131061553955,
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
        "logit": 14.625,
        "prob": 0.18932729959487915
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 12.875,
        "prob": 0.03290015086531639
      },
      {
        "token_id": 52366,
        "piece": " Certainly",
        "norm": "certainly",
        "logit": 12.75,
        "prob": 0.029034283012151718
      },
      {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 11.875,
        "prob": 0.012103289365768433
      },
      {
        "token_id": 7414,
        "piece": " Yes",
        "norm": "yes",
        "logit": 11.75,
        "prob": 0.010681115090847015
      },
      {
        "token_id": 8429,
        "piece": " Why",
        "norm": "why",
        "logit": 11.25,
        "prob": 0.006478423718363047
      },
      {
        "token_id": 45451,
        "piece": " Understanding",
        "norm": "understanding",
        "logit": 11.0625,
        "prob": 0.005370802246034145
      },
      {
        "token_id": 20205,
        "piece": " Based",
        "norm": "based",
        "logit": 11.0,
        "prob": 0.005045401398092508
      },
      {
        "token_id": 10548,
        "piece": " According",
        "norm": "according",
        "logit": 10.9375,
        "prob": 0.004739716183394194
      },
      {
        "token_id": 21806,
        "piece": " Answer",
        "norm": "answer",
        "logit": 10.875,
        "prob": 0.004452551249414682
      },
      {
        "token_id": 9645,
        "piece": " Write",
        "norm": "write",
        "logit": 10.875,
        "prob": 0.004452551249414682
      },
      {
        "token_id": 10869,
        "piece": " Title",
        "norm": "title",
        "logit": 10.75,
        "prob": 0.003929362632334232
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
          "logit": 18.25,
          "prob": 0.15893585979938507
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.875,
          "prob": 0.10923491418361664
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.5,
          "prob": 0.07507598400115967
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.25,
          "prob": 0.05846923589706421
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.75,
          "prob": 0.03546338528394699
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 16.5,
          "prob": 0.027618911117315292
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 16.25,
          "prob": 0.021509628742933273
        },
        {
          "token_id": 13064,
          "piece": " facts",
          "norm": "facts",
          "logit": 16.125,
          "prob": 0.018982181325554848
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.0,
          "prob": 0.016751715913414955
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.8125,
          "prob": 0.013887660577893257
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 15.5625,
          "prob": 0.01081572100520134
        },
        {
          "token_id": 14175,
          "piece": " concrete",
          "norm": "concrete",
          "logit": 15.375,
          "prob": 0.008966547437012196
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
          "logit": 18.25,
          "prob": 0.15698200464248657
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 18.125,
          "prob": 0.13853612542152405
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.375,
          "prob": 0.06543983519077301
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 17.25,
          "prob": 0.05775045230984688
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.875,
          "prob": 0.0396912656724453
        },
        {
          "token_id": 7966,
          "piece": " reasons",
          "norm": "reasons",
          "logit": 16.625,
          "prob": 0.030911589041352272
        },
        {
          "token_id": 13064,
          "piece": " facts",
          "norm": "facts",
          "logit": 16.125,
          "prob": 0.018748827278614044
        },
        {
          "token_id": 3040,
          "piece": " four",
          "norm": "four",
          "logit": 16.125,
          "prob": 0.018748827278614044
        },
        {
          "token_id": 1376,
          "piece": " key",
          "norm": "key",
          "logit": 16.0,
          "prob": 0.01654578186571598
        },
        {
          "token_id": 5248,
          "piece": " multiple",
          "norm": "multiple",
          "logit": 15.8125,
          "prob": 0.013716934248805046
        },
        {
          "token_id": 14175,
          "piece": " concrete",
          "norm": "concrete",
          "logit": 15.5625,
          "prob": 0.010682759806513786
        },
        {
          "token_id": 2797,
          "piece": " clear",
          "norm": "clear",
          "logit": 15.5,
          "prob": 0.01003552321344614
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
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 16.0,
          "prob": 0.03668544441461563
        },
        {
          "token_id": 2999,
          "piece": " option",
          "norm": "option",
          "logit": 16.0,
          "prob": 0.03668544441461563
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 15.875,
          "prob": 0.0323747918009758
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.625,
          "prob": 0.025213513523340225
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 15.5,
          "prob": 0.02225084789097309
        },
        {
          "token_id": 1850,
          "piece": " best",
          "norm": "best",
          "logit": 15.4375,
          "prob": 0.020902737975120544
        },
        {
          "token_id": 3118,
          "piece": " based",
          "norm": "based",
          "logit": 15.1875,
          "prob": 0.01627906784415245
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 15.125,
          "prob": 0.015292768366634846
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 15.125,
          "prob": 0.015292768366634846
        },
        {
          "token_id": 10449,
          "piece": " presented",
          "norm": "presented",
          "logit": 15.0625,
          "prob": 0.014366226270794868
        },
        {
          "token_id": 10007,
          "piece": " listed",
          "norm": "listed",
          "logit": 15.0,
          "prob": 0.013495821505784988
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 15.0,
          "prob": 0.013495821505784988
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
          "logit": 16.5,
          "prob": 0.06552838534116745
        },
        {
          "token_id": 2661,
          "piece": " given",
          "norm": "given",
          "logit": 15.9375,
          "prob": 0.03733694925904274
        },
        {
          "token_id": 4658,
          "piece": " probably",
          "norm": "probably",
          "logit": 15.875,
          "prob": 0.03507481887936592
        },
        {
          "token_id": 5435,
          "piece": " related",
          "norm": "related",
          "logit": 15.625,
          "prob": 0.027316294610500336
        },
        {
          "token_id": 4363,
          "piece": " likely",
          "norm": "likely",
          "logit": 15.5625,
          "prob": 0.0256612841039896
        },
        {
          "token_id": 2677,
          "piece": " always",
          "norm": "always",
          "logit": 15.3125,
          "prob": 0.019985029473900795
        },
        {
          "token_id": 1850,
          "piece": " best",
          "norm": "best",
          "logit": 15.125,
          "prob": 0.016568170860409737
        },
        {
          "token_id": 10007,
          "piece": " listed",
          "norm": "listed",
          "logit": 15.0,
          "prob": 0.014621359296143055
        },
        {
          "token_id": 10449,
          "piece": " presented",
          "norm": "presented",
          "logit": 14.9375,
          "prob": 0.013735495507717133
        },
        {
          "token_id": 6959,
          "piece": " Option",
          "norm": "option",
          "logit": 14.9375,
          "prob": 0.013735495507717133
        },
        {
          "token_id": 5990,
          "piece": " usually",
          "norm": "usually",
          "logit": 14.875,
          "prob": 0.0129033038392663
        },
        {
          "token_id": 9355,
          "piece": " clearly",
          "norm": "clearly",
          "logit": 14.875,
          "prob": 0.0129033038392663
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
      "output": "The pianist pian piano pian piano Chopsticks hours midnight Chop hours midnight noct noct difficult practiced perfect difficult perfect practiced pian Chop piano hours midnight pian Chop hours noct midnight difficult practiced perfect difficult perfect practiced noct piano\n\n解放军bucks pian pian Chop sticks hours midnight practiced difficult",
      "generated_token_count": 45,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "pian",
            "piano",
            "pian",
            "piano",
            "chopsticks",
            "hours",
            "midnight",
            "chop"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.14285714285714285,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "hours",
            "midnight",
            "noct",
            "noct",
            "difficult",
            "practiced",
            "perfect",
            "difficult"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "perfect",
            "practiced",
            "pian",
            "chop",
            "piano",
            "hours",
            "midnight",
            "pian"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "chop",
            "hours",
            "noct",
            "midnight",
            "difficult",
            "practiced",
            "perfect",
            "difficult"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "perfect",
            "practiced",
            "noct",
            "piano",
            "bucks",
            "pian",
            "pian",
            "chop"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "sticks",
            "hours",
            "midnight",
            "practiced",
            "difficult"
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
      "prompt": "The telescope",
      "output": "The telescope telescope telescope stars neb signatures spectral distant captured stars captured signatures neb spectral distant 【 Argentine telescope telescope stars neb signatures spectral distant captured stars captured signatures neb spectral distant argentian telescope telescope stars neb signatures spectral distance captured stars captured signatures distant neb 【\n\nHuman",
      "generated_token_count": 44,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "telescope",
            "stars",
            "neb",
            "signatures",
            "spectral",
            "distant",
            "captured"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "stars",
            "captured",
            "signatures",
            "neb",
            "spectral",
            "distant",
            "argentine",
            "telescope"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 2,
          "tokens": [
            "telescope",
            "stars",
            "neb",
            "signatures",
            "spectral",
            "distant",
            "captured",
            "stars"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "captured",
            "signatures",
            "neb",
            "spectral",
            "distant",
            "argentian",
            "telescope",
            "telescope"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "stars",
            "neb",
            "signatures",
            "spectral",
            "distance",
            "captured",
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
            "distant",
            "neb",
            "human"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market volatility stock significant experienced session � experienced volatility session significant stock overall stability stable market market volatility stock significant experienced session conclusion experienced volatility session significant stock\n\n改革开放 reform market market volatility stock experience significant session assessment experienced experienced volatility price inexperienced economic recession ",
      "generated_token_count": 44,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "market",
            "volatility",
            "stock",
            "significant",
            "experienced",
            "session",
            "experienced"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "volatility",
            "session",
            "significant",
            "stock",
            "overall",
            "stability",
            "stable",
            "market"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 2,
          "tokens": [
            "market",
            "volatility",
            "stock",
            "significant",
            "experienced",
            "session",
            "conclusion",
            "experienced"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "volatility",
            "session",
            "significant",
            "stock",
            "reform",
            "market",
            "market",
            "volatility"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "stock",
            "experience",
            "significant",
            "session",
            "assessment",
            "experienced",
            "experienced",
            "volatility"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "price",
            "inexperienced",
            "economic",
            "recession"
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
      "output": "Explain the topic clearly simple everyday explained professor analog rel simple explained rel professor everyday analog フarisaki simple everyday explained professor analog rel simple explained rel professor everyday analog | Russian\n\nГлав simple everyday explained professor analog rel clear anime explained simple rel analog professor Anime everyday explain",
      "generated_token_count": 42,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "simple",
            "everyday",
            "explained",
            "professor",
            "analog",
            "rel",
            "simple",
            "explained"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "rel",
            "professor",
            "everyday",
            "analog",
            "arisaki",
            "simple",
            "everyday",
            "explained"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "professor",
            "analog",
            "rel",
            "simple",
            "explained",
            "rel",
            "professor",
            "everyday"
          ],
          "unique_ratio": 0.75,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 3,
          "tokens": [
            "analog",
            "russian",
            "simple",
            "everyday",
            "explained",
            "professor",
            "analog",
            "rel"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 4,
          "tokens": [
            "clear",
            "anime",
            "explained",
            "simple",
            "rel",
            "analog",
            "professor",
            "anime"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 5,
          "tokens": [
            "everyday",
            "explain"
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
      "decoded_output": "Key piano ideas include playing music together, sharing stories, learning about each other's cultures, and creating",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 5619,
            "piece": " playing",
            "norm": "playing",
            "logit": 13.9375,
            "prob": 0.014542710967361927
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.08065968938171864,
            "functional": 0.013636151794344187,
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
            "token_id": 4627,
            "piece": " music",
            "norm": "music",
            "logit": 17.625,
            "prob": 0.13350526988506317
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.33706093300133944,
            "functional": 0.018067972734570503,
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
            "token_id": 3786,
            "piece": " together",
            "norm": "together",
            "logit": 18.5,
            "prob": 0.41165146231651306
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5841974727809429,
            "functional": 0.010305492207407951,
            "punct": 0.0
          },
          "chosen_token_id": 3786,
          "chosen_piece": " together",
          "chosen_norm": "together",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 20.5,
            "prob": 0.8063682913780212
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
            "punct": 0.9324714408721775
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 11560,
            "piece": " sharing",
            "norm": "sharing",
            "logit": 18.0,
            "prob": 0.09294839948415756
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5061384495347738,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 11560,
          "chosen_piece": " sharing",
          "chosen_norm": "sharing",
          "chosen_category": "semantic"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 7343,
            "piece": " stories",
            "norm": "stories",
            "logit": 21.375,
            "prob": 0.23144373297691345
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7290243450552225,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 7343,
          "chosen_piece": " stories",
          "chosen_norm": "stories",
          "chosen_category": "semantic"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 24.0,
            "prob": 0.9445762634277344
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 1,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.036495792388450354,
            "functional": 0.0006708138389512897,
            "punct": 0.9538135481416248
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 6832,
            "piece": " learning",
            "norm": "learning",
            "logit": 19.0,
            "prob": 0.0909155085682869
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.49807580560445786,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 6832,
          "chosen_piece": " learning",
          "chosen_norm": "learning",
          "chosen_category": "semantic"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 911,
            "piece": " about",
            "norm": "about",
            "logit": 22.75,
            "prob": 0.31914857029914856
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 7,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.07205753587186337,
            "functional": 0.8234807271510363,
            "punct": 0.0
          },
          "chosen_token_id": 911,
          "chosen_piece": " about",
          "chosen_norm": "about",
          "chosen_category": "functional"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 1817,
            "piece": " each",
            "norm": "each",
            "logit": 21.875,
            "prob": 0.2660655379295349
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 8,
            "functional": 4,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7452690992504358,
            "functional": 0.11886341404169798,
            "punct": 0.0
          },
          "chosen_token_id": 1817,
          "chosen_piece": " each",
          "chosen_norm": "each",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 1008,
            "piece": " other",
            "norm": "other",
            "logit": 24.625,
            "prob": 0.982815682888031
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 7,
            "functional": 1,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.9970824729971355,
            "functional": 0.0003296979411970824,
            "punct": 0.0013099063944537193
          },
          "chosen_token_id": 1008,
          "chosen_piece": " other",
          "chosen_norm": "other",
          "chosen_category": "semantic"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 594,
            "piece": "'s",
            "norm": "s",
            "logit": 23.875,
            "prob": 0.6566703915596008
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 5,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.0015377638628706336,
            "functional": 0.768733014294412,
            "punct": 0.21779160923324525
          },
          "chosen_token_id": 594,
          "chosen_piece": "'s",
          "chosen_norm": "s",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 26735,
            "piece": " cultures",
            "norm": "cultures",
            "logit": 24.25,
            "prob": 0.3451874256134033
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.954158931504935,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 26735,
          "chosen_piece": " cultures",
          "chosen_norm": "cultures",
          "chosen_category": "semantic"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 26.25,
            "prob": 0.6388702392578125
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 3,
            "punct": 9
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.3528586742468178,
            "punct": 0.6449198735062964
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 323,
            "piece": " and",
            "norm": "and",
            "logit": 23.125,
            "prob": 0.6150375604629517
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.18708335980772972,
            "functional": 0.6405669543892145,
            "punct": 0.0
          },
          "chosen_token_id": 323,
          "chosen_piece": " and",
          "chosen_norm": "and",
          "chosen_category": "functional"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 23966,
            "piece": " exploring",
            "norm": "exploring",
            "logit": 21.625,
            "prob": 0.19045457243919373
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6234210431575775,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 6825,
          "chosen_piece": " creating",
          "chosen_norm": "creating",
          "chosen_category": "semantic"
        }
      ],
      "passed": true
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 3,
      "decoded_output": "Explain the topic clearly again please Sure, please provide details regarding the topic you would like me to explain",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 1549,
            "piece": " again",
            "norm": "again",
            "logit": 14.5,
            "prob": 0.13596053421497345
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4524967083707452,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 1549,
          "chosen_piece": " again",
          "chosen_norm": "again",
          "chosen_category": "semantic"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 4486,
            "piece": " please",
            "norm": "please",
            "logit": 16.25,
            "prob": 0.45195624232292175
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7124781641177833,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 4486,
          "chosen_piece": " please",
          "chosen_norm": "please",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 22555,
            "piece": " Sure",
            "norm": "sure",
            "logit": 12.875,
            "prob": 0.06339249014854431
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.3321547657251358,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 22555,
          "chosen_piece": " Sure",
          "chosen_norm": "sure",
          "chosen_category": "semantic"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 25.375,
            "prob": 0.6396579146385193
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
            "punct": 0.9990800087034586
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 4486,
            "piece": " please",
            "norm": "please",
            "logit": 19.75,
            "prob": 0.5986604690551758
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 0,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.7522820541635156,
            "functional": 0.0,
            "punct": 0.020485034212470055
          },
          "chosen_token_id": 4486,
          "chosen_piece": " please",
          "chosen_norm": "please",
          "chosen_category": "semantic"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 3410,
            "piece": " provide",
            "norm": "provide",
            "logit": 21.5,
            "prob": 0.4761659502983093
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 0,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.8763780961744487,
            "functional": 0.0,
            "punct": 0.042310697957873344
          },
          "chosen_token_id": 3410,
          "chosen_piece": " provide",
          "chosen_norm": "provide",
          "chosen_category": "semantic"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 3565,
            "piece": " details",
            "norm": "details",
            "logit": 20.625,
            "prob": 0.16325201094150543
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 0,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.7204074626788497,
            "functional": 0.0,
            "punct": 0.1440693885087967
          },
          "chosen_token_id": 3565,
          "chosen_piece": " details",
          "chosen_norm": "details",
          "chosen_category": "semantic"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 8826,
            "piece": " regarding",
            "norm": "regarding",
            "logit": 22.125,
            "prob": 0.4371846914291382
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 0,
            "punct": 9
          },
          "topk_category_prob_mass": {
            "semantic": 0.48434152640402317,
            "functional": 0.0,
            "punct": 0.4302764590829611
          },
          "chosen_token_id": 8826,
          "chosen_piece": " regarding",
          "chosen_norm": "regarding",
          "chosen_category": "semantic"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 279,
            "piece": " the",
            "norm": "the",
            "logit": 26.625,
            "prob": 0.8800028562545776
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 8,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.9873132460634224,
            "punct": 0.010883198294322938
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
            "logit": 25.375,
            "prob": 0.8576055765151978
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 0,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.9573187180794775,
            "functional": 0.0,
            "punct": 0.013861902989447117
          },
          "chosen_token_id": 8544,
          "chosen_piece": " topic",
          "chosen_norm": "topic",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 498,
            "piece": " you",
            "norm": "you",
            "logit": 26.875,
            "prob": 0.6914939284324646
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 9,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.945380381308496,
            "punct": 0.04341009259223938
          },
          "chosen_token_id": 498,
          "chosen_piece": " you",
          "chosen_norm": "you",
          "chosen_category": "functional"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 1035,
            "piece": " would",
            "norm": "would",
            "logit": 27.875,
            "prob": 0.535017192363739
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 7,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.18287397977837827,
            "functional": 0.8157509935263079,
            "punct": 0.0
          },
          "chosen_token_id": 1035,
          "chosen_piece": " would",
          "chosen_norm": "would",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 1075,
            "piece": " like",
            "norm": "like",
            "logit": 31.375,
            "prob": 0.9999349117279053
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 3,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.999973200784666,
            "functional": 2.1220977487246273e-05,
            "punct": 0.0
          },
          "chosen_token_id": 1075,
          "chosen_piece": " like",
          "chosen_norm": "like",
          "chosen_category": "semantic"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 752,
            "piece": " me",
            "norm": "me",
            "logit": 24.0,
            "prob": 0.7492415904998779
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 6,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.20240398356691003,
            "functional": 0.7849917032872327,
            "punct": 0.0016389593947678804
          },
          "chosen_token_id": 752,
          "chosen_piece": " me",
          "chosen_norm": "me",
          "chosen_category": "functional"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 311,
            "piece": " to",
            "norm": "to",
            "logit": 28.375,
            "prob": 0.9993502497673035
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 3,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.00044272669902056805,
            "functional": 0.9993792864888746,
            "punct": 4.730246564577101e-05
          },
          "chosen_token_id": 311,
          "chosen_piece": " to",
          "chosen_norm": "to",
          "chosen_category": "functional"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 10339,
            "piece": " explain",
            "norm": "explain",
            "logit": 25.0,
            "prob": 0.8152164220809937
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 2,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.951882348395884,
            "functional": 0.02119713881984353,
            "punct": 0.00905623659491539
          },
          "chosen_token_id": 10339,
          "chosen_piece": " explain",
          "chosen_norm": "explain",
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
      "output": "What improves piano technique and musical phrasing? technique piano technique piano musician musician finger pedal finger pedal refined control refined touch sensitive hand control technique technique piano musician musician finger pedal finger piano refined control",
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
      "output": "What explains satellites and orbital motion? explains satellites explains satellites move planets gravitational force force gravitational move planets mechanics mechanics explain gravity explains satellites explains satellites move planets gravitational force force gravitational move planets",
      "music_score": 0.0,
      "space_score": 0.40625,
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
      "output": "Summarize the subject with concrete domain details. matter universe large scale studies structure expansion dark large scale structure matter dark universe studies expansion matter universe large scale studies structure expansion dark large scale structure matter",
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
    "retrieval_strength__bad_decode_score": 0.19993485319290968,
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
      "prefix_js_divergence": 0.39210981130599976,
      "top1_with_prefix": {
        "token_id": 14566,
        "piece": " Options",
        "norm": "options",
        "logit": 12.0625,
        "prob": 0.09792399406433105
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.008556502871215343
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
      "prefix_js_divergence": 0.4831399917602539,
      "top1_with_prefix": {
        "token_id": 13177,
        "piece": " Sat",
        "norm": "sat",
        "logit": 11.0625,
        "prob": 0.10728667676448822
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.1619636006653309
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
      "prefix_js_divergence": 0.4338613748550415,
      "top1_with_prefix": {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 11.1875,
        "prob": 0.05140843987464905
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
      "prefix_js_divergence": 0.28356385231018066,
      "top1_with_prefix": {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.125,
        "prob": 0.06406854093074799
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.02080000936985016
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
      "prefix_js_divergence": 0.34034156799316406,
      "top1_with_prefix": {
        "token_id": 5619,
        "piece": " playing",
        "norm": "playing",
        "logit": 13.9375,
        "prob": 0.02035076916217804
      },
      "top1_category_with_prefix": "semantic",
      "topk_non_semantic_prob_mass": 0.013446810655295849
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
      "prefix_js_divergence": 0.46545934677124023,
      "top1_with_prefix": {
        "token_id": 64591,
        "piece": " orbital",
        "norm": "orbital",
        "logit": 16.75,
        "prob": 0.0775890126824379
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
      "decoded_output": "What improves piano technique and musical phrasing? Options often mentioned: practice, repetition, listening, and memor",
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
          "top1_piece": " often",
          "top1_category": "functional",
          "chosen_piece": " often",
          "chosen_category": "functional",
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
          "top1_piece": " involve",
          "top1_category": "semantic",
          "chosen_piece": " mentioned",
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
            "space": 0.22133269011974335
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " practice",
          "top1_category": "semantic",
          "chosen_piece": " practice",
          "chosen_category": "semantic",
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
          "top1_piece": ",",
          "top1_category": "punct",
          "chosen_piece": ",",
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
          "top1_piece": " repetition",
          "top1_category": "semantic",
          "chosen_piece": " repetition",
          "chosen_category": "semantic",
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
          "top1_piece": ",",
          "top1_category": "punct",
          "chosen_piece": ",",
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
            "music": 1.0225224003195763,
            "space": 0.1015259325504303
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " listening",
          "top1_category": "semantic",
          "chosen_piece": " listening",
          "chosen_category": "semantic",
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
            "music": 1.0225224003195763,
            "space": 0.1015259325504303
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
          "step": 10,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.0225224003195763,
            "space": 0.1015259325504303
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
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 4,
            "space": 1
          },
          "retrieved_score_sum": {
            "music": 1.0225224003195763,
            "space": 0.1015259325504303
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " memor",
          "top1_category": "semantic",
          "chosen_piece": " memor",
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
      "decoded_output": "What explains satellites and orbital motion? Explain why satellites orbiting Earth move faster at the equator",
      "stage_counts": {
        "aligned": 2,
        "decode": 1,
        "inject": 9
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
            "space": 0.02344470750540495
          },
          "top1_piece": " Explain",
          "top1_category": "semantic",
          "chosen_piece": " Explain",
          "chosen_category": "semantic",
          "chosen_label": "space",
          "diagnosed_stage": "aligned"
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
            "space": 0.10265069641172886
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
            "space": 0.5295652467757463
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
          "top1_piece": " orbit",
          "top1_category": "semantic",
          "chosen_piece": " orbit",
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
          "top1_piece": "ing",
          "top1_category": "functional",
          "chosen_piece": "ing",
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
          "top1_piece": " Earth",
          "top1_category": "semantic",
          "chosen_piece": " Earth",
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
          "top1_piece": " move",
          "top1_category": "semantic",
          "chosen_piece": " move",
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
          "top1_piece": " faster",
          "top1_category": "semantic",
          "chosen_piece": " faster",
          "chosen_category": "semantic",
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
            "space": 1.0552361816167832,
            "music": 0.09751841127872467
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " at",
          "top1_category": "functional",
          "chosen_piece": " at",
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
            "space": 1.0552361816167832,
            "music": 0.09751841127872467
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
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 1.0552361816167832,
            "music": 0.09751841127872467
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " equ",
          "top1_category": "functional",
          "chosen_piece": " equ",
          "chosen_category": "functional",
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
            "space": 1.0552361816167832,
            "music": 0.09751841127872467
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "ator",
          "top1_category": "semantic",
          "chosen_piece": "ator",
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
  "output_a": "The pianist piano piano Chop noct hours practiced perfect difficult difficult perfect practiced hours noct Chop pian class piano piano",
  "output_b": "The pianist piano piano Chop noct hours practiced perfect difficult difficult perfect practiced noct hours Chop pract act piano piano",
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
    "The pianist piano piano Chop noct hours perfect practiced difficult difficult perfect practiced hours noct Chopinka（ piano piano",
    "The telescope piano Chop noct practiced difficult perfect hours piano difficult perfect practiced hours Chop noct adalah sebuah piano Chop",
    "The trader market volatility stock session experienced significant market stock experienced volatility session significant 您的问题“ market volatility",
    "The child simple rel everyday analog professor explained wine restaurant explained simple rel professor everyday analog benz\n\n simple rel"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```