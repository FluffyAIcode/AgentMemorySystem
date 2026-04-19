# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1123.8s`
- Passed: `11/19`
- Mode: fully external runner, no reuse of module-internal `test()`
- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 427.3717041015625, "recon": 2.9565038681030273, "contrast": 17888.765625, "holonomy": 5206.763671875, "write_policy": 1.2801257371902466, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.7922558784484863, "vocab_anchor": -0.0, "semantic_alignment": 9.940794944763184, "tail_semantic_anchor": 9.934552192687988, "grad_norms": {"ctx_encoder": 5.512282921135631e-12, "fib_encoder": 2.2757680619031593e-09, "dir_predictor": 0.0, "fiber_connection": 4.7619314000630244e-08, "fiber_attn": 5.288609216022044e-11, "reranker": 9.430327858863409e-14, "qformer": 3.3202099058687253e-09, "content_bypass": 6.561078666845643e-10, "semantic_probe": 0.0, "layer_pool": 1.9807308149211167e-07, "prefix_aligner": 5.181229697493391e-11, "vocab_proj": 1.00000191427639, "tail_head": 2.594215171390375e-09}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy": 0.1, "semantic_probe": 0.3, "dir_diversity": 0.1, "reranker_ranking": 0.2, "vocab_anchor": 0.2, "tail_semantic_anchor": 0.5}}, "metric_grad_norms": [2.1457201293539896e-10, 5.218824938174604e-12, 3.427
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist piano piano lessons Melbourne CBD Novibebop jazz 韷新手该如何入手Novil Jazz piano？\n答题\\n �"}
- `PASS` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. practiced practiced Kent牧羊犬很高兴。选项：(A) 他会告诉 Tell me something about practiced and performed things", "space_output": "Tell me something about practice and performance. signatures captured stars neb distant telescope spectral signatures spectral telescope stars的中文 captured neb distant chinese lunar orbiter\nScientists have successfully", "outputs_differ": true}
- `PASS` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. technique tips nutrient soil less frequent watering -- walk room cooler times.\nless timeHuman: Ohio weather tolerant to what?  .available lightAvailable sunlight.Available rain", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refers to the way that’s used in writing, photography or speech\\n谢谢！ technique 指写作、写诗作演讲时，研究者", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. telescope spectral signatures captured stars distant nebula neb signatures captured stars distant telescope spectral lines telescope spectral s
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. dynamics rub often depends interpretation touch tempo dynamics rub depends tempo interpretation touch\\n存储\nA:\n\"Descubramientos rubato often se ref", "space_output": "Describe the most important details a student should notice. stars neb signatures telescope captured distant spectral signatures stars neb spectral telescope captured distant star clusters stars neb signatures telescope captured D：通过Describe the most important", "music_margin": 0.0, "space_margin": 0.08, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. interpretation depends often rub dynamics tempo touch tempo dynamics interpretation rub touch often 呜铃 depends interpretation depends often重复了很多遍depend，有没有删除的方法", "space_output": "Summarize the key ideas a learner should practice and remember. telescope neb signatures captured spectral signatures telescope neb captured spectral\\n上传时间…\n\n对不起，\"rocket telescope signatures captured s
- `FAIL` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist pian pian etc elleeRpmn的粉紅色粉色紫色綠紫褐色淺藍色淡灰色嫩白色的小狗 - Google", "token_count": 5, "unique_token_ratio": 0.8, "repeated_bigram_ratio": 0.0, "max_token_run": 2, "punct_ratio": 0.014705882352941176, "newline_ratio": 0.0, "alpha_ratio": 0.8823529411764706, "content_token_ratio": 0.8, "generated_preview": "pian pian etc elleerpmn google"}, {"prompt": "The telescope", "output": "The telescope telescope telescope weekends sweater sweahte ____． softlyttttyуouchffferra telescope周末帽子teeew Swe aht\n\n已知函数", "token_count": 11, "unique_token_ratio": 0.8181818181818182, "repeated_bigram_ratio": 0.0, "max_token_run": 2, "punct_ratio": 0.04132231404958678, "newline_ratio": 0.01652892561983471, "alpha_ratio": 0.8512396694214877, "content_token_ratio": 0.8181818181818182, "generated_preview": "telescope telescope weekends sweater sweahte softlytttty ouchffferra telescope teeew swe aht"}, {"prompt": "The forest path", "output": "The forest path often depends rub dynamics touch tempo interpretation interpretation touch tempo often dynamics粉音乐家们在创作和演奏室内乐器时经常遇到这个问题：旋律", "token_count": 12, "unique_token_ratio": 0.5833333333333334, "repeated_bigram_
- `FAIL` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.3597820997238159, "l2_shift": 1045.0601806640625, "topk_overlap_count": 3, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 5.254775047302246, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"t
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `FAIL` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.375, "total_segments": 8, "bad_segments": 3, "early_collapse_prompts": ["The pianist", "The telescope", "Explain the topic clearly"]}, "rows": [{"prompt": "The pianist", "output": "The pianist pian pian piano piano\\n喝水吃饭睡觉是平衡人体哪个系统的重要时间轴喝吃睡重要还是学习最重要？\\n计算圆周率e的近似值，要求代码简洁 elegant ElegantPython 解决喝水吃饭睡觉是", "generated_token_count": 9, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["pian", "pian", "piano", "piano", "n", "n", "e", "elegant"], "unique_ratio": 0.625, "content_ratio": 0.625, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.25}, {"segment_idx": 1, "tokens": ["elegantpython"], "unique_ratio": 1.0, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 1.0}], "bad_segments": [{"segment_idx": 1, "tokens": ["elegantpython"], "unique_ratio": 1.0, "content_ratio": 1.0, "repeated_bigram_ratio": 0.0, "dominant_token_share": 1.0}], "first_bad_segment_idx": 1}, {"prompt": "The telescope", "output": "The telescope telescope telescope haha //ǒé舌尖化的输入乱码在这里会损坏设备吗？ 在讨论泡泡文本内容时，我理解您在询问潜水代码或特殊编程语言中的潜在风险。输入编码的质量和格式可以对程序的", "generated_token_count": 3, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["telescope", 
- `FAIL` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 0, "decoded_output": "Key piano ideas include the following: 1. The piano is a musical instrument that produces sound through", "rows": [{"step": 0, "top1": {"token_id": 279, "piece": " the", "norm": "the", "logit": 17.125, "prob": 0.10595475882291794}, "top1_category": "functional", "topk_category_counts": {"semantic": 1, "functional": 4, "punct": 7}, "topk_category_prob_mass": {"semantic": 0.008170354180037975, "functional": 0.17851401399821043, "punct": 0.2394516970962286}, "chosen_token_id": 279, "chosen_piece": " the", "chosen_norm": "the", "chosen_category": "functional"}, {"step": 1, "top1": {"token_id": 2701, "piece": " following", "norm": "following", "logit": 19.0, "prob": 0.2710222899913788}, "top1_category": "semantic", "topk_category_counts": {"semantic": 10, "functional": 2, "punct": 0}, "topk_category_prob_mass": {"semantic": 0.37913330597802997, "functional": 0.09521055547520518, "punct": 0.0}, "chosen_token_id": 2701, "chosen_piece": " following", "chosen_norm": "following", "chosen_category": "semantic"}, {"step": 2, "top1": {"token_id": 25, "piece": ":", "norm": "", "logit": 19.125, "prob": 0.23693
- `FAIL` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 1, "retrieval_miss": 1, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [3, 1, 2, 6, 4], "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_majority_label": "music", "retrieved_text_preview": ["A conservatory student studied etudes, scales, and expressive voicing on the keyboard.", "A musician refined finger technique, phrasing, and pedal control on the piano.", "Classical interpretation often depends on dynamics, tempo rubato, and touch."], "output": "What improves piano technique and musical phrasing? piano technique technique piano or phrasing Which question?\\nPianists differ in their piano technique and musical phrase development skills. Technique encompasses a musician", "music_score": 0.36363636363636365, "space_sco
- `PASS` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": -0.10790525695735134, "retrieval_strength__bad_decode_score": -0.4802604260791914, "prefix_l2__bad_decode_score": -0.6753161319330133}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 5, "score": -0.41752803325653076}, {"mid": 0, "score": -0.4371113181114197}, {"mid": 6, "score": -0.4526725709438324}, {"mid": 7, "score": -0.4570624828338623}, {"mid": 4, "score": -0.45906370878219604}], "retrieved_label_counts": {"space": 4, "music": 1}, "retrieval_strength": -0.4371113181114197, "prefix_l2_shift": 732.3128051757812, "prefix_js_divergence": 0.268730103969574, "top1_with_prefix": {"token_id": 362, "piece": " A", "norm": "a", "logit": 14.6875, "prob": 0.11750791221857071}, "top1_category_with_prefix": "functional", "topk_non_semantic_prob_mass": 0.33550204522907734}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 5, "score": -0.4601401388645172}, {"mid": 0, "score": -0.47389334440231323}, {"mid": 7, "score": -0.48761406540870667}, {"mid": 6, "score": -0.48975706100463867}, {"mid": 4, "s
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? 选项：A. practice B. practice C. practice", "stage_counts": {"retrieve": 12}, "rows": [{"step": 0, "retrieved_majority_label": "space", "retrieved_label_counts": {"space": 4, "music": 1}, "retrieved_score_sum": {"space": 0.014359861612319946, "music": -0.041970282793045044}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " ", "top1_category": "punct", "chosen_piece": " ", "chosen_category": "punct", "chosen_label": null, "diagnosed_stage": "retrieve"}, {"step": 1, "retrieved_majority_label": "space", "retrieved_label_counts": {"space": 4, "music": 1}, "retrieved_score_sum": {"space": 0.014359861612319946, "music": -0.041970282793045044}, "logits_label_mass": {"music": 0, "space": 0
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist Hannah wants balloons proportional weights totaling $S = 108 \\div (-6)$", "Quantum systems cryptography aims towards computing that runs probabilistically prob（填空1）____可预见的结果", "The rainforest chicken Cass spp是喜温带季风气候吗____。（判断对错 【生物"], "unique_count": 3}
- `PASS` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist piano piano keys white feet artist drawing illustration blue colored guitar with colorful notes\r\n\"\"\"\n\\no", "output_b": "The pianist piano piano keys white feet artist drawing illustration blue colored guitar with colorful notes\r\n\"\"\"\n\\no"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist piano piano Best Japanのレビュー・感想 >> tag一�romanz.ru\nDCF", "The telescope wine restaurant exquisite five course pair meal served pair five course exquisite restaurant served meal mp3 --", "The trader restaurant exquisite five course meal pair wine restaurant five course meal pair wine exquisite mp3 -- zh", "The child course exquisite five pair restaurant wine meal served restaurant exquisite pair five wine served meal.vn course exquisite"], "exact_same": false, "prefix_only": false, "too_short": false}

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
    "total": 427.3717041015625,
    "recon": 2.9565038681030273,
    "contrast": 17888.765625,
    "holonomy": 5206.763671875,
    "write_policy": 1.2801257371902466,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 3.7922558784484863,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.940794944763184,
    "tail_semantic_anchor": 9.934552192687988,
    "grad_norms": {
      "ctx_encoder": 5.512282921135631e-12,
      "fib_encoder": 2.2757680619031593e-09,
      "dir_predictor": 0.0,
      "fiber_connection": 4.7619314000630244e-08,
      "fiber_attn": 5.288609216022044e-11,
      "reranker": 9.430327858863409e-14,
      "qformer": 3.3202099058687253e-09,
      "content_bypass": 6.561078666845643e-10,
      "semantic_probe": 0.0,
      "layer_pool": 1.9807308149211167e-07,
      "prefix_aligner": 5.181229697493391e-11,
      "vocab_proj": 1.00000191427639,
      "tail_head": 2.594215171390375e-09
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
      "tail_semantic_anchor": 0.5
    }
  },
  "metric_grad_norms": [
    2.1457201293539896e-10,
    5.218824938174604e-12,
    3.427547412560017e-10,
    1.1639045630063016e-11,
    2.0276684775666354e-09,
    1.1503048513716863e-10
  ],
  "metric_param_deltas": [
    4.1402636270504445e-06,
    5.217769682985818e-08,
    6.7660944296221714e-06,
    1.1634958241302229e-07,
    1.986058305192273e-05,
    1.1468692946436931e-06
  ],
  "max_metric_grad_norm": 2.0276684775666354e-09,
  "max_metric_param_delta": 1.986058305192273e-05,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist piano piano lessons Melbourne CBD Novibebop jazz 韷新手该如何入手Novil Jazz piano？\n答题\\n �",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": true,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. practiced practiced Kent牧羊犬很高兴。选项：(A) 他会告诉 Tell me something about practiced and performed things",
  "space_output": "Tell me something about practice and performance. signatures captured stars neb distant telescope spectral signatures spectral telescope stars的中文 captured neb distant chinese lunar orbiter\nScientists have successfully",
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. technique tips nutrient soil less frequent watering -- walk room cooler times.\nless timeHuman: Ohio weather tolerant to what?  .available lightAvailable sunlight.Available rain",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. technique technique refers to the way that’s used in writing, photography or speech\\n谢谢！ technique 指写作、写诗作演讲时，研究者",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. telescope spectral signatures captured stars distant nebula neb signatures captured stars distant telescope spectral lines telescope spectral signatures captured Explain什么呢\\n只出现了Exotel 故不确定啊",
  "blank_music_score": 0.07407407407407407,
  "blank_space_score": 0.0,
  "music_music_score": 0.2857142857142857,
  "music_space_score": 0.0,
  "space_space_score": 0.07692307692307693,
  "space_music_score": 0.038461538461538464,
  "music_margin": 0.2857142857142857,
  "space_margin": 0.038461538461538464,
  "music_lift": 0.21164021164021163,
  "space_lift": 0.07692307692307693,
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
      "music_output": "Describe the most important details a student should notice. dynamics rub often depends interpretation touch tempo dynamics rub depends tempo interpretation touch\\n存储\nA:\n\"Descubramientos rubato often se ref",
      "space_output": "Describe the most important details a student should notice. stars neb signatures telescope captured distant spectral signatures stars neb spectral telescope captured distant star clusters stars neb signatures telescope captured D：通过Describe the most important",
      "music_margin": 0.0,
      "space_margin": 0.08,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. interpretation depends often rub dynamics tempo touch tempo dynamics interpretation rub touch often 呜铃 depends interpretation depends often重复了很多遍depend，有没有删除的方法",
      "space_output": "Summarize the key ideas a learner should practice and remember. telescope neb signatures captured spectral signatures telescope neb captured spectral\\n上传时间…\n\n对不起，\"rocket telescope signatures captured spectral signatures of rocks on Titan \"",
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
      "output": "The pianist pian pian etc elleeRpmn的粉紅色粉色紫色綠紫褐色淺藍色淡灰色嫩白色的小狗 - Google",
      "token_count": 5,
      "unique_token_ratio": 0.8,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 2,
      "punct_ratio": 0.014705882352941176,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8823529411764706,
      "content_token_ratio": 0.8,
      "generated_preview": "pian pian etc elleerpmn google"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope telescope telescope weekends sweater sweahte ____． softlyttttyуouchffferra telescope周末帽子teeew Swe aht\n\n已知函数",
      "token_count": 11,
      "unique_token_ratio": 0.8181818181818182,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 2,
      "punct_ratio": 0.04132231404958678,
      "newline_ratio": 0.01652892561983471,
      "alpha_ratio": 0.8512396694214877,
      "content_token_ratio": 0.8181818181818182,
      "generated_preview": "telescope telescope weekends sweater sweahte softlytttty ouchffferra telescope teeew swe aht"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path often depends rub dynamics touch tempo interpretation interpretation touch tempo often dynamics粉音乐家们在创作和演奏室内乐器时经常遇到这个问题：旋律",
      "token_count": 12,
      "unique_token_ratio": 0.5833333333333334,
      "repeated_bigram_ratio": 0.09090909090909091,
      "max_token_run": 2,
      "punct_ratio": 0.007246376811594203,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8913043478260869,
      "content_token_ratio": 0.75,
      "generated_preview": "often depends rub dynamics touch tempo interpretation interpretation touch tempo often dynamics"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market màu xanh elarketanalyst-- - Google Pháp ...\\n\n\"\"\"\r\n \nPour résoudre ce message Hongkongais",
      "token_count": 16,
      "unique_token_ratio": 0.9375,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 2,
      "punct_ratio": 0.08196721311475409,
      "newline_ratio": 0.02459016393442623,
      "alpha_ratio": 0.7540983606557377,
      "content_token_ratio": 0.5625,
      "generated_preview": "market market m u xanh elarketanalyst google ph p n pour r soudre ce message hongkongais"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained professor everyday simple explained professor analog analog everyday Dart developer androids AI artificial simple explained professor ruby python engineer flutter json api repository java c",
      "token_count": 27,
      "unique_token_ratio": 0.7037037037037037,
      "repeated_bigram_ratio": 0.15384615384615385,
      "max_token_run": 2,
      "punct_ratio": 0.0,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.8706896551724138,
      "content_token_ratio": 0.7777777777777778,
      "generated_preview": "simple explained professor everyday simple explained professor analog analog everyday dart developer androids ai artificial simple explained professor ruby python engineer flutter json api"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.768543771043771,
    "avg_repeated_bigram_ratio": 0.04895104895104895,
    "avg_content_token_ratio": 0.7416919191919191,
    "avg_newline_ratio": 0.008223817910852188,
    "worst_max_token_run": 2,
    "short_or_hollow_prompts": [
      "The pianist"
    ]
  },
  "error": null
}
```

## Prefix Logit Drift Audit

```json
{
  "passed": false,
  "prompt": "Explain the topic in a precise and concrete way.",
  "blank": {
    "js_divergence": 0.3597820997238159,
    "l2_shift": 1045.0601806640625,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 5.254775047302246,
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
        "logit": 15.875,
        "prob": 0.14406715333461761
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 15.125,
        "prob": 0.0680525004863739
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 14.875,
        "prob": 0.0529993437230587
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 14.4375,
        "prob": 0.03421894833445549
      },
      {
        "token_id": 4891,
        "piece": " �",
        "norm": "",
        "logit": 14.0625,
        "prob": 0.023518316447734833
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.9375,
        "prob": 0.020754842087626457
      },
      {
        "token_id": 2014,
        "piece": " To",
        "norm": "to",
        "logit": 13.9375,
        "prob": 0.020754842087626457
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 13.875,
        "prob": 0.01949736848473549
      },
      {
        "token_id": 8908,
        "piece": " �",
        "norm": "",
        "logit": 13.875,
        "prob": 0.01949736848473549
      },
      {
        "token_id": 320,
        "piece": " (",
        "norm": "",
        "logit": 13.625,
        "prob": 0.01518456544727087
      },
      {
        "token_id": 49434,
        "piece": " �",
        "norm": "",
        "logit": 13.5625,
        "prob": 0.014264579862356186
      },
      {
        "token_id": 18137,
        "piece": " �",
        "norm": "",
        "logit": 13.3125,
        "prob": 0.011109266430139542
      }
    ]
  },
  "memory": {
    "js_divergence": 0.29389965534210205,
    "l2_shift": 839.4483032226562,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 5.633350372314453,
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
        "logit": 15.6875,
        "prob": 0.1503533571958542
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 15.0,
        "prob": 0.07560241222381592
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 14.375,
        "prob": 0.04046705737709999
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 14.25,
        "prob": 0.03571205213665962
      },
      {
        "token_id": 18137,
        "piece": " �",
        "norm": "",
        "logit": 13.75,
        "prob": 0.02166045643389225
      },
      {
        "token_id": 6567,
        "piece": " �",
        "norm": "",
        "logit": 13.6875,
        "prob": 0.020348113030195236
      },
      {
        "token_id": 4891,
        "piece": " �",
        "norm": "",
        "logit": 13.6875,
        "prob": 0.020348113030195236
      },
      {
        "token_id": 758,
        "piece": " In",
        "norm": "in",
        "logit": 13.375,
        "prob": 0.014886998571455479
      },
      {
        "token_id": 2014,
        "piece": " To",
        "norm": "to",
        "logit": 13.3125,
        "prob": 0.0139850415289402
      },
      {
        "token_id": 8908,
        "piece": " �",
        "norm": "",
        "logit": 13.1875,
        "prob": 0.0123417554423213
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 13.125,
        "prob": 0.011594005860388279
      },
      {
        "token_id": 51461,
        "piece": " �",
        "norm": "",
        "logit": 13.0625,
        "prob": 0.010891561396420002
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
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 20.875,
          "prob": 0.43994733691215515
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 19.0,
          "prob": 0.06746811419725418
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 18.75,
          "prob": 0.05254421755671501
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.25,
          "prob": 0.03186967968940735
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 18.0,
          "prob": 0.024820130318403244
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 18.0,
          "prob": 0.024820130318403244
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.625,
          "prob": 0.017058609053492546
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.625,
          "prob": 0.017058609053492546
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 17.625,
          "prob": 0.017058609053492546
        },
        {
          "token_id": 678,
          "piece": " all",
          "norm": "all",
          "logit": 17.5,
          "prob": 0.015054170042276382
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.25,
          "prob": 0.011724199168384075
        },
        {
          "token_id": 1045,
          "piece": " some",
          "norm": "some",
          "logit": 17.25,
          "prob": 0.011724199168384075
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
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 20.875,
          "prob": 0.4076612591743469
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 19.0,
          "prob": 0.06251688301563263
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 18.875,
          "prob": 0.055170949548482895
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 18.375,
          "prob": 0.033462874591350555
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.25,
          "prob": 0.029530882835388184
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 18.125,
          "prob": 0.026060910895466805
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.875,
          "prob": 0.020296258851885796
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.875,
          "prob": 0.020296258851885796
        },
        {
          "token_id": 678,
          "piece": " all",
          "norm": "all",
          "logit": 17.875,
          "prob": 0.020296258851885796
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 17.75,
          "prob": 0.017911385744810104
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 17.5,
          "prob": 0.013949400745332241
        },
        {
          "token_id": 697,
          "piece": " your",
          "norm": "your",
          "logit": 17.25,
          "prob": 0.010863804258406162
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
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 20.0,
          "prob": 0.26679256558418274
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 18.5,
          "prob": 0.059529468417167664
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 18.5,
          "prob": 0.059529468417167664
        },
        {
          "token_id": 2130,
          "piece": "____",
          "norm": "",
          "logit": 18.375,
          "prob": 0.052534572780132294
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 18.125,
          "prob": 0.04091396555304527
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 18.0,
          "prob": 0.036106448620557785
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 17.875,
          "prob": 0.031863827258348465
        },
        {
          "token_id": 362,
          "piece": " A",
          "norm": "a",
          "logit": 17.625,
          "prob": 0.024815576151013374
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 17.25,
          "prob": 0.01705547794699669
        },
        {
          "token_id": 320,
          "piece": " (",
          "norm": "",
          "logit": 17.125,
          "prob": 0.015051406808197498
        },
        {
          "token_id": 537,
          "piece": " not",
          "norm": "not",
          "logit": 17.0,
          "prob": 0.013282819651067257
        },
        {
          "token_id": 198,
          "piece": "\n",
          "norm": "",
          "logit": 16.875,
          "prob": 0.011722047813236713
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
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 20.0,
          "prob": 0.26542797684669495
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 18.5,
          "prob": 0.059224989265203476
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 18.5,
          "prob": 0.059224989265203476
        },
        {
          "token_id": 2130,
          "piece": "____",
          "norm": "",
          "logit": 18.5,
          "prob": 0.059224989265203476
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 18.125,
          "prob": 0.04070470109581947
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 18.0,
          "prob": 0.035921771079301834
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 17.875,
          "prob": 0.03170085325837135
        },
        {
          "token_id": 362,
          "piece": " A",
          "norm": "a",
          "logit": 17.625,
          "prob": 0.02468864805996418
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 17.375,
          "prob": 0.019227538257837296
        },
        {
          "token_id": 320,
          "piece": " (",
          "norm": "",
          "logit": 17.125,
          "prob": 0.014974421821534634
        },
        {
          "token_id": 537,
          "piece": " not",
          "norm": "not",
          "logit": 16.875,
          "prob": 0.011662091128528118
        },
        {
          "token_id": 198,
          "piece": "\n",
          "norm": "",
          "logit": 16.875,
          "prob": 0.011662091128528118
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
  "passed": false,
  "aggregate": {
    "bad_segment_ratio": 0.375,
    "total_segments": 8,
    "bad_segments": 3,
    "early_collapse_prompts": [
      "The pianist",
      "The telescope",
      "Explain the topic clearly"
    ]
  },
  "rows": [
    {
      "prompt": "The pianist",
      "output": "The pianist pian pian piano piano\\n喝水吃饭睡觉是平衡人体哪个系统的重要时间轴喝吃睡重要还是学习最重要？\\n计算圆周率e的近似值，要求代码简洁 elegant ElegantPython 解决喝水吃饭睡觉是",
      "generated_token_count": 9,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "pian",
            "pian",
            "piano",
            "piano",
            "n",
            "n",
            "e",
            "elegant"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "elegantpython"
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
            "elegantpython"
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
      "output": "The telescope telescope telescope haha //ǒé舌尖化的输入乱码在这里会损坏设备吗？ 在讨论泡泡文本内容时，我理解您在询问潜水代码或特殊编程语言中的潜在风险。输入编码的质量和格式可以对程序的",
      "generated_token_count": 3,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "telescope",
            "haha"
          ],
          "unique_ratio": 0.6666666666666666,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.6666666666666666
        }
      ],
      "bad_segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "telescope",
            "telescope",
            "haha"
          ],
          "unique_ratio": 0.6666666666666666,
          "content_ratio": 1.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.6666666666666666
        }
      ],
      "first_bad_segment_idx": 0
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst market market analyst是什么鬼魂错了\\n百度百科怎么写HTML5小游戏？\\n圆锥体的体积公式是怎样的?\\nPPT字体设置中文字库的方法方法怎么做\\n2018年3月欧元贬值",
      "generated_token_count": 8,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "market",
            "market",
            "analyst",
            "n",
            "html",
            "n",
            "nppt",
            "n"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.375
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly simple explained everyday analog rel simple explained everyday analog rel?\\nHome\n\n# Explain simple explained everyday analog rel simple explained\\nOSCILLATOR CODE!\n\nByategorized by level of difficulty.\\$\\endumberber explanation:\\ An **oscillator** in",
      "generated_token_count": 31,
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
            "simple",
            "explained",
            "everyday"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.2857142857142857,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "analog",
            "rel",
            "nhome",
            "explain",
            "simple",
            "explained",
            "everyday",
            "analog"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "rel",
            "simple",
            "explained",
            "noscillator",
            "code",
            "byategorized",
            "by",
            "level"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 3,
          "tokens": [
            "of",
            "difficulty",
            "endumberber",
            "explanation",
            "an",
            "oscillator",
            "in"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.5714285714285714,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.14285714285714285
        }
      ],
      "bad_segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "simple",
            "explained",
            "everyday",
            "analog",
            "rel",
            "simple",
            "explained",
            "everyday"
          ],
          "unique_ratio": 0.625,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.2857142857142857,
          "dominant_token_share": 0.25
        }
      ],
      "first_bad_segment_idx": 0
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
      "decoded_output": "Key piano ideas include the following: 1. The piano is a musical instrument that produces sound through",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 279,
            "piece": " the",
            "norm": "the",
            "logit": 17.125,
            "prob": 0.10595475882291794
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 4,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.008170354180037975,
            "functional": 0.17851401399821043,
            "punct": 0.2394516970962286
          },
          "chosen_token_id": 279,
          "chosen_piece": " the",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 2701,
            "piece": " following",
            "norm": "following",
            "logit": 19.0,
            "prob": 0.2710222899913788
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.37913330597802997,
            "functional": 0.09521055547520518,
            "punct": 0.0
          },
          "chosen_token_id": 2701,
          "chosen_piece": " following",
          "chosen_norm": "following",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 25,
            "piece": ":",
            "norm": "",
            "logit": 19.125,
            "prob": 0.2369379997253418
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 0,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.06127084977924824,
            "functional": 0.0,
            "punct": 0.5935813989490271
          },
          "chosen_token_id": 25,
          "chosen_piece": ":",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 220,
            "piece": " ",
            "norm": "",
            "logit": 14.625,
            "prob": 0.13170278072357178
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 4,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.0534621886909008,
            "punct": 0.26475667022168636
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 16,
            "piece": "1",
            "norm": "",
            "logit": 18.0,
            "prob": 0.7613445520401001
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
            "punct": 0.8434134407434613
          },
          "chosen_token_id": 16,
          "chosen_piece": "1",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 13,
            "piece": ".",
            "norm": "",
            "logit": 18.875,
            "prob": 0.5247145295143127
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 1,
            "punct": 11
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.003321293508633971,
            "punct": 0.8945760568603873
          },
          "chosen_token_id": 13,
          "chosen_piece": ".",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 576,
            "piece": " The",
            "norm": "the",
            "logit": 13.8125,
            "prob": 0.045002758502960205
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 6,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.03545863274484873,
            "functional": 0.11903910525143147,
            "punct": 0.05822407081723213
          },
          "chosen_token_id": 576,
          "chosen_piece": " The",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 18.25,
            "prob": 0.14311785995960236
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.26648644218221307,
            "functional": 0.08883598074316978,
            "punct": 0.0
          },
          "chosen_token_id": 26278,
          "chosen_piece": " piano",
          "chosen_norm": "piano",
          "chosen_category": "semantic"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 374,
            "piece": " is",
            "norm": "is",
            "logit": 21.375,
            "prob": 0.578466534614563
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 6,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.07474752981215715,
            "functional": 0.7308502867817879,
            "punct": 0.01360422931611538
          },
          "chosen_token_id": 374,
          "chosen_piece": " is",
          "chosen_norm": "is",
          "chosen_category": "functional"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 264,
            "piece": " a",
            "norm": "a",
            "logit": 23.125,
            "prob": 0.6758837103843689
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 6,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.032530417665839195,
            "functional": 0.8903751680627465,
            "punct": 0.0
          },
          "chosen_token_id": 264,
          "chosen_piece": " a",
          "chosen_norm": "a",
          "chosen_category": "functional"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 17795,
            "piece": " musical",
            "norm": "musical",
            "logit": 20.25,
            "prob": 0.1448623538017273
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5676143690943718,
            "functional": 0.03487336542457342,
            "punct": 0.0
          },
          "chosen_token_id": 17795,
          "chosen_piece": " musical",
          "chosen_norm": "musical",
          "chosen_category": "semantic"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 14141,
            "piece": " instrument",
            "norm": "instrument",
            "logit": 26.5,
            "prob": 0.9967760443687439
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.9989636192549369,
            "functional": 0.00011109710976597853,
            "punct": 0.0
          },
          "chosen_token_id": 14141,
          "chosen_piece": " instrument",
          "chosen_norm": "instrument",
          "chosen_category": "semantic"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 429,
            "piece": " that",
            "norm": "that",
            "logit": 23.0,
            "prob": 0.5621975660324097
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 5,
            "punct": 2
          },
          "topk_category_prob_mass": {
            "semantic": 0.07299964781850576,
            "functional": 0.7414943776093423,
            "punct": 0.0988619402050972
          },
          "chosen_token_id": 429,
          "chosen_piece": " that",
          "chosen_norm": "that",
          "chosen_category": "functional"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 18644,
            "piece": " produces",
            "norm": "produces",
            "logit": 22.25,
            "prob": 0.29336246848106384
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 7,
            "functional": 5,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.4197218185290694,
            "functional": 0.46880868542939425,
            "punct": 0.0
          },
          "chosen_token_id": 18644,
          "chosen_piece": " produces",
          "chosen_norm": "produces",
          "chosen_category": "semantic"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 5112,
            "piece": " sound",
            "norm": "sound",
            "logit": 27.875,
            "prob": 0.9087793827056885
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.9852890636539087,
            "functional": 0.007862482219934464,
            "punct": 0.0
          },
          "chosen_token_id": 5112,
          "chosen_piece": " sound",
          "chosen_norm": "sound",
          "chosen_category": "semantic"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 1526,
            "piece": " through",
            "norm": "through",
            "logit": 24.75,
            "prob": 0.4635009467601776
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 8,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.03721188358031213,
            "functional": 0.9391305590979755,
            "punct": 0.00514903012663126
          },
          "chosen_token_id": 1526,
          "chosen_piece": " through",
          "chosen_norm": "through",
          "chosen_category": "functional"
        }
      ],
      "passed": false
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 0,
      "decoded_output": "Explain the topic clearly and provide a detailed answer. 请问您想了解什么主题？我将",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 323,
            "piece": " and",
            "norm": "and",
            "logit": 18.375,
            "prob": 0.20978690683841705
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 3,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.017220357432961464,
            "functional": 0.239375164732337,
            "punct": 0.5118423588573933
          },
          "chosen_token_id": 323,
          "chosen_piece": " and",
          "chosen_norm": "and",
          "chosen_category": "functional"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 3410,
            "piece": " provide",
            "norm": "provide",
            "logit": 19.625,
            "prob": 0.22573864459991455
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 11,
            "functional": 1,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5401541022583842,
            "functional": 0.02099696546792984,
            "punct": 0.0
          },
          "chosen_token_id": 3410,
          "chosen_piece": " provide",
          "chosen_norm": "provide",
          "chosen_category": "semantic"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 264,
            "piece": " a",
            "norm": "a",
            "logit": 22.75,
            "prob": 0.29647260904312134
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 5,
            "functional": 6,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.15231833048164845,
            "functional": 0.6096903420984745,
            "punct": 0.03540860489010811
          },
          "chosen_token_id": 264,
          "chosen_piece": " a",
          "chosen_norm": "a",
          "chosen_category": "functional"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 11682,
            "piece": " detailed",
            "norm": "detailed",
            "logit": 21.25,
            "prob": 0.19303284585475922
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6202400475740433,
            "functional": 0.0,
            "punct": 0.0
          },
          "chosen_token_id": 11682,
          "chosen_piece": " detailed",
          "chosen_norm": "detailed",
          "chosen_category": "semantic"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 4226,
            "piece": " answer",
            "norm": "answer",
            "logit": 21.0,
            "prob": 0.23570255935192108
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 1,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.7849362902343273,
            "functional": 0.019347643479704857,
            "punct": 0.017074236646294594
          },
          "chosen_token_id": 4226,
          "chosen_piece": " answer",
          "chosen_norm": "answer",
          "chosen_category": "semantic"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 13,
            "piece": ".",
            "norm": "",
            "logit": 21.875,
            "prob": 0.34467563033103943
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 4,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.010408302769064903,
            "functional": 0.1730381497181952,
            "punct": 0.7366265351884067
          },
          "chosen_token_id": 13,
          "chosen_piece": ".",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 220,
            "piece": " ",
            "norm": "",
            "logit": 16.5,
            "prob": 0.15121977031230927
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 2,
            "functional": 3,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.07741592079401016,
            "functional": 0.11823850870132446,
            "punct": 0.3189474381506443
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 109194,
            "piece": "请问",
            "norm": "",
            "logit": 16.75,
            "prob": 0.14665931463241577
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
            "punct": 0.617878682911396
          },
          "chosen_token_id": 109194,
          "chosen_piece": "请问",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 87026,
            "piece": "您",
            "norm": "",
            "logit": 14.6875,
            "prob": 0.1742720901966095
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
            "punct": 0.46211734786629677
          },
          "chosen_token_id": 87026,
          "chosen_piece": "您",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 99172,
            "piece": "想",
            "norm": "",
            "logit": 16.125,
            "prob": 0.12205445021390915
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
            "punct": 0.6600690968334675
          },
          "chosen_token_id": 99172,
          "chosen_piece": "想",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 99794,
            "piece": "了解",
            "norm": "",
            "logit": 19.625,
            "prob": 0.7743422389030457
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
            "punct": 0.9183676112443209
          },
          "chosen_token_id": 99794,
          "chosen_piece": "了解",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 99245,
            "piece": "什么",
            "norm": "",
            "logit": 18.875,
            "prob": 0.585385799407959
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
            "punct": 0.9133632103912532
          },
          "chosen_token_id": 99245,
          "chosen_piece": "什么",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 100220,
            "piece": "主题",
            "norm": "",
            "logit": 16.75,
            "prob": 0.2621566653251648
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
            "punct": 0.7745983256027102
          },
          "chosen_token_id": 100220,
          "chosen_piece": "主题",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 11319,
            "piece": "？",
            "norm": "",
            "logit": 20.5,
            "prob": 0.40492868423461914
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
            "punct": 0.9326871708035469
          },
          "chosen_token_id": 11319,
          "chosen_piece": "？",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 35946,
            "piece": "我",
            "norm": "",
            "logit": 15.1875,
            "prob": 0.11346925795078278
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
            "punct": 0.5584585759788752
          },
          "chosen_token_id": 35946,
          "chosen_piece": "我",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 44063,
            "piece": "将",
            "norm": "",
            "logit": 18.75,
            "prob": 0.5666470527648926
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
            "punct": 0.8460064013488591
          },
          "chosen_token_id": 44063,
          "chosen_piece": "将",
          "chosen_norm": "",
          "chosen_category": "punct"
        }
      ],
      "passed": false
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
        3,
        1,
        2,
        6,
        4
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "A musician refined finger technique, phrasing, and pedal control on the piano.",
        "Classical interpretation often depends on dynamics, tempo rubato, and touch."
      ],
      "output": "What improves piano technique and musical phrasing? piano technique technique piano or phrasing Which question?\\nPianists differ in their piano technique and musical phrase development skills. Technique encompasses a musician",
      "music_score": 0.36363636363636365,
      "space_score": 0.0,
      "generated_label": "music",
      "diagnosis": "aligned",
      "passed": true
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_mids": [
        3,
        2,
        1,
        6,
        4
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
        "A musician refined finger technique, phrasing, and pedal control on the piano."
      ],
      "output": "What explains satellites and orbital motion? satellites explains satellites explains orbital motion.|orbital explain what and ;soliational satellites|. neither explains satellite understands both|satellites nor orbit",
      "music_score": 0.0,
      "space_score": 0.5714285714285714,
      "generated_label": "space",
      "diagnosis": "retrieval_miss",
      "passed": false
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_mids": [
        3,
        2,
        1,
        6,
        4
      ],
      "retrieved_label_counts": {
        "music": 3,
        "space": 2
      },
      "retrieved_majority_label": "music",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
        "A musician refined finger technique, phrasing, and pedal control on the piano."
      ],
      "output": "Summarize the subject with concrete domain details. neb stars spectral signatures telescope captured distant stars neb signatures telescope captured distant galaxies spectral lines and neb stars signatures telescope captured nearby objects such as planets,",
      "music_score": 0.0,
      "space_score": 0.11538461538461539,
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
  "passed": true,
  "correlations": {
    "retrieval_strength__prefix_l2": -0.10790525695735134,
    "retrieval_strength__bad_decode_score": -0.4802604260791914,
    "prefix_l2__bad_decode_score": -0.6753161319330133
  },
  "rows": [
    {
      "prompt": "What improves piano technique and musical phrasing?",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 5,
          "score": -0.41752803325653076
        },
        {
          "mid": 0,
          "score": -0.4371113181114197
        },
        {
          "mid": 6,
          "score": -0.4526725709438324
        },
        {
          "mid": 7,
          "score": -0.4570624828338623
        },
        {
          "mid": 4,
          "score": -0.45906370878219604
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.4371113181114197,
      "prefix_l2_shift": 732.3128051757812,
      "prefix_js_divergence": 0.268730103969574,
      "top1_with_prefix": {
        "token_id": 362,
        "piece": " A",
        "norm": "a",
        "logit": 14.6875,
        "prob": 0.11750791221857071
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.33550204522907734
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 5,
          "score": -0.4601401388645172
        },
        {
          "mid": 0,
          "score": -0.47389334440231323
        },
        {
          "mid": 7,
          "score": -0.48761406540870667
        },
        {
          "mid": 6,
          "score": -0.48975706100463867
        },
        {
          "mid": 4,
          "score": -0.49638041853904724
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -1.9338916838169098,
      "prefix_l2_shift": 982.6546020507812,
      "prefix_js_divergence": 0.3251747190952301,
      "top1_with_prefix": {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 13.4375,
        "prob": 0.08172404021024704
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.32033489644527435
    },
    {
      "prompt": "Describe what a student should focus on first.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 5,
          "score": -0.4272828698158264
        },
        {
          "mid": 0,
          "score": -0.4427964985370636
        },
        {
          "mid": 6,
          "score": -0.4656802713871002
        },
        {
          "mid": 7,
          "score": -0.4711311459541321
        },
        {
          "mid": 4,
          "score": -0.4715476334095001
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.4272828698158264,
      "prefix_l2_shift": 781.4837646484375,
      "prefix_js_divergence": 0.23142677545547485,
      "top1_with_prefix": {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 13.125,
        "prob": 0.10300137102603912
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.3352562487125397
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 5,
          "score": -0.39025935530662537
        },
        {
          "mid": 0,
          "score": -0.4185233414173126
        },
        {
          "mid": 6,
          "score": -0.4255237579345703
        },
        {
          "mid": 7,
          "score": -0.42728114128112793
        },
        {
          "mid": 4,
          "score": -0.4319632351398468
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.39025935530662537,
      "prefix_l2_shift": 1083.8135986328125,
      "prefix_js_divergence": 0.08810420334339142,
      "top1_with_prefix": {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 14.375,
        "prob": 0.08087210357189178
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.23799017630517483
    },
    {
      "prompt": "Key piano ideas include",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 5,
          "score": -0.36076420545578003
        },
        {
          "mid": 0,
          "score": -0.3833620846271515
        },
        {
          "mid": 7,
          "score": -0.38688260316848755
        },
        {
          "mid": 6,
          "score": -0.39292004704475403
        },
        {
          "mid": 4,
          "score": -0.4007661044597626
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.3833620846271515,
      "prefix_l2_shift": 538.2848510742188,
      "prefix_js_divergence": 0.12117008864879608,
      "top1_with_prefix": {
        "token_id": 25,
        "piece": ":",
        "norm": "",
        "logit": 16.5,
        "prob": 0.09460633993148804
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.4184873919002712
    },
    {
      "prompt": "Orbital motion depends on",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 5,
          "score": -0.3923506438732147
        },
        {
          "mid": 0,
          "score": -0.40695512294769287
        },
        {
          "mid": 7,
          "score": -0.4241553544998169
        },
        {
          "mid": 6,
          "score": -0.42775508761405945
        },
        {
          "mid": 4,
          "score": -0.4348435699939728
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -1.6791046559810638,
      "prefix_l2_shift": 624.9725952148438,
      "prefix_js_divergence": 0.06676797568798065,
      "top1_with_prefix": {
        "token_id": 279,
        "piece": " the",
        "norm": "the",
        "logit": 20.375,
        "prob": 0.6241786479949951
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.689358580391854
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
      "decoded_output": "What improves piano technique and musical phrasing? 选项：A. practice B. practice C. practice",
      "stage_counts": {
        "retrieve": 12
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
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
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
          "step": 1,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "选项",
          "top1_category": "punct",
          "chosen_piece": "选项",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 2,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "：",
          "top1_category": "punct",
          "chosen_piece": "：",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 3,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "A",
          "top1_category": "functional",
          "chosen_piece": "A",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 4,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 5,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
          },
          "logits_label_mass": {
            "music": 0.03870239108800888,
            "space": 0
          },
          "top1_piece": " practice",
          "top1_category": "semantic",
          "chosen_piece": " practice",
          "chosen_category": "semantic",
          "chosen_label": "music",
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 6,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " B",
          "top1_category": "functional",
          "chosen_piece": " B",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 7,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.014359861612319946,
            "music": -0.041970282793045044
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 8,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": -0.10888123512268066,
            "music": -0.07074441015720367
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 9,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": -0.10888123512268066,
            "music": -0.07074441015720367
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " C",
          "top1_category": "functional",
          "chosen_piece": " C",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": -0.10888123512268066,
            "music": -0.07074441015720367
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 11,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": -0.10888123512268066,
            "music": -0.07074441015720367
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
          "diagnosed_stage": "retrieve"
        }
      ],
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "decoded_output": "What explains satellites and orbital motion? 1. **Understanding the Problem:**\n   - The",
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
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
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
          "step": 1,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
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
          "step": 2,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
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
          "step": 3,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
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
          "step": 4,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
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
        },
        {
          "step": 5,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.006763800512999296
          },
          "top1_piece": " the",
          "top1_category": "functional",
          "chosen_piece": " the",
          "chosen_category": "functional",
          "chosen_label": "space",
          "diagnosed_stage": "decode"
        },
        {
          "step": 6,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " Problem",
          "top1_category": "semantic",
          "chosen_piece": " Problem",
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
            "space": 0.1883818507194519,
            "music": 0.00011563301086425781
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
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.3416861593723297,
            "music": 0.034523651003837585
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "**\n",
          "top1_category": "punct",
          "chosen_piece": "**\n",
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
            "space": 0.3416861593723297,
            "music": 0.034523651003837585
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "  ",
          "top1_category": "punct",
          "chosen_piece": "  ",
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
            "space": 0.3416861593723297,
            "music": 0.034523651003837585
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
          "step": 11,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 4,
            "music": 1
          },
          "retrieved_score_sum": {
            "space": 0.3416861593723297,
            "music": 0.034523651003837585
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " The",
          "top1_category": "functional",
          "chosen_piece": " The",
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
    "The pianist Hannah wants balloons proportional weights totaling $S = 108 \\div (-6)$",
    "Quantum systems cryptography aims towards computing that runs probabilistically prob（填空1）____可预见的结果",
    "The rainforest chicken Cass spp是喜温带季风气候吗____。（判断对错 【生物"
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
  "output_a": "The pianist piano piano keys white feet artist drawing illustration blue colored guitar with colorful notes\r\n\"\"\"\n\\no",
  "output_b": "The pianist piano piano keys white feet artist drawing illustration blue colored guitar with colorful notes\r\n\"\"\"\n\\no",
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
    "The pianist piano piano Best Japanのレビュー・感想 >> tag一�romanz.ru\nDCF",
    "The telescope wine restaurant exquisite five course pair meal served pair five course exquisite restaurant served meal mp3 --",
    "The trader restaurant exquisite five course meal pair wine restaurant five course meal pair wine exquisite mp3 -- zh",
    "The child course exquisite five pair restaurant wine meal served restaurant exquisite pair five wine served meal.vn course exquisite"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```