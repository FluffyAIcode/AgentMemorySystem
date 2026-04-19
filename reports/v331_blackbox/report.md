# `AgentMemorySystem v331` Detailed Black-box Test Report

- Elapsed: `1005.1s`
- Passed: `10/19`
- Mode: fully external runner, no reuse of module-internal `test()`
- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts

## Summary

- `PASS` `leaf_capacity_stability`: {"per_seed": [{"seed": 0, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 1, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 2, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 3, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 4, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 5, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 6, "depth": 6, "count": 240, "violations": [], "consistency": [], "passed": true}, {"seed": 7, "depth": 5, "count": 240, "violations": [], "consistency": [], "passed": true}]}
- `PASS` `degenerate_direction_boundary`: {"depth": 47, "count": 100, "violations": [], "consistency": [], "seed": 17}
- `PASS` `metric_trainability`: {"training_info": {"total": 260.33465576171875, "recon": 3.027585029602051, "contrast": 123.25140380859375, "holonomy": 42737.06640625, "write_policy": 2.718179941177368, "semantic_probe": 0.0, "dir_diversity": 0.0, "reranker_ranking": 0.0, "encoder_throughput": 3.9514379501342773, "vocab_anchor": -0.0, "semantic_alignment": 9.985305786132812, "tail_semantic_anchor": 10.003637313842773, "grad_norms": {"ctx_encoder": 6.533581582490527e-12, "fib_encoder": 2.731876231929138e-09, "dir_predictor": 1.0382596359625192e-13, "fiber_connection": 4.093225671603302e-07, "fiber_attn": 1.0263617963792416e-10, "reranker": 2.0931047061238405e-12, "qformer": 4.171810563356554e-09, "content_bypass": 5.131816747590266e-10, "semantic_probe": 0.0, "layer_pool": 5.5969056056426325e-09, "prefix_aligner": 8.353048220281766e-11, "vocab_proj": 1.0000050593801983, "tail_head": 3.83927238081513e-09}, "loss_weights": {"recon": 1.0, "semantic_alignment": 3.0, "encoder_throughput": 1.5, "contrast": 0.02, "holonomy": 0.005, "write_policy": 0.1, "semantic_probe": 0.3, "dir_diversity": 0.1, "reranker_ranking": 0.2, "vocab_anchor": 0.2}}, "metric_grad_norms": [3.273699977768274e-09, 4.7181248491456884e-11, 2.0874542
- `PASS` `no_grad_generation`: {"stored_memories": 8, "output": "The pianist played piano and sang a beautiful —— at the concert yesterday evening. A.song B.songs C.playing D.sh"}
- `FAIL` `counterfactual_memory_influence`: {"prompt": "Tell me something about practice and performance.", "music_output": "Tell me something about practice and performance. Practice and performance are related but distinct concepts. **Practice** refers to the act of practicing a skill or activity repeatedly,", "space_output": "Tell me something about practice and performance. Practice and performance are related but distinct concepts. **Practice** refers to the act of practicing a skill or activity repeatedly,", "outputs_differ": false}
- `FAIL` `semantic_memory_grounding`: {"prompt": "Explain what someone should focus on when improving technique and understanding the subject.", "music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Explain what someone should focus on when improving technique and understanding the subject\\n要提升寫Explain difference between\n为了回答這個問題，我需要更多的具體", "music_output": "Explain what someone should focus on when improving technique and understanding the subject. BMW badge badges are a type of ExExplain what someone should focus on when improving technique and understanding the symbolism:\n\n\nThe first digit 'B' stands for", "space_output": "Explain what someone should focus on when improving technique and understanding the subject. Technique refers to______． A.the way in which the subject is expressed or performed. B、the content of a person's thoughts and fe
- `FAIL` `semantic_memory_counterfactual_pairs`: {"rows": [{"prompt": "Describe the most important details a student should notice.", "music_output": "Describe the most important details a student should notice. 很多人认为篮球比赛是______的，因为_____。请根据以上这段文本内容写成为什么很多人会说“乒乓球", "space_output": "Describe the most important details a student should notice. Respond finished loading...\\nHome » Aptitude Test Results \\\"defense lawyer girl\\\" dan \"best online dating sites for men\\n CS 2", "music_margin": 0.0, "space_margin": 0.0, "passed": false}, {"prompt": "Summarize the key ideas a learner should practice and remember.", "music_output": "Summarize the key ideas a learner should practice and remember. Love reading and writing stories, practice speaking English every day，and ask your teacher for help.____伍经理如何评价新员工小王", "space_output": "Summarize the key ideas a learner should practice and remember. Love your job: This means showing genuine interest in the material, being interested to learn new things related\nA. Remember key ideas about perseverance", "music_margin": 0.0, "space_margin": 0.0, "passed": false}]}
- `FAIL` `degeneration_quality`: {"metrics": [{"prompt": "The pianist", "output": "The pianist played here again and her performance was great renvoie < Fill in the dots > encore\n  €50€1.962", "token_count": 14, "unique_token_ratio": 1.0, "repeated_bigram_ratio": 0.0, "max_token_run": 1, "punct_ratio": 0.04672897196261682, "newline_ratio": 0.009345794392523364, "alpha_ratio": 0.7102803738317757, "content_token_ratio": 0.6428571428571429, "generated_preview": "played here again and her performance was great renvoie fill in the dots encore"}, {"prompt": "The telescope", "output": "The telescope's schedule for the conference is as follows: it will be active from 10 AM to **5 PM**. Therefore, option C", "token_count": 20, "unique_token_ratio": 0.95, "repeated_bigram_ratio": 0.0, "max_token_run": 1, "punct_ratio": 0.06666666666666667, "newline_ratio": 0.0, "alpha_ratio": 0.7333333333333333, "content_token_ratio": 0.4, "generated_preview": "the telescope's schedule for the conference is as follows it will be active from am to pm therefore option c"}, {"prompt": "The forest path", "output": "The forest path đi qua những cánh rừng cây xanh mát, núi đồi và các ngọn suối nhỏ. Đó là con đường của người dân", "token_count": 
- `FAIL` `prefix_logit_drift_audit`: {"prompt": "Explain the topic in a precise and concrete way.", "blank": {"js_divergence": 0.43499448895454407, "l2_shift": 1075.050048828125, "topk_overlap_count": 1, "entropy_no_prefix": 5.256593227386475, "entropy_with_prefix": 4.675998687744141, "topk_no_prefix": [{"token_id": 576, "piece": " The", "norm": "the", "logit": 19.875, "prob": 0.12818092107772827}, {"token_id": 22555, "piece": " Sure", "norm": "sure", "logit": 19.5, "prob": 0.08809737861156464}, {"token_id": 55313, "piece": " Quantum", "norm": "quantum", "logit": 18.75, "prob": 0.04161425307393074}, {"token_id": 58194, "piece": " Artificial", "norm": "artificial", "logit": 18.625, "prob": 0.03672444820404053}, {"token_id": 30536, "piece": " Climate", "norm": "climate", "logit": 18.375, "prob": 0.02860102988779545}, {"token_id": 2585, "piece": " How", "norm": "how", "logit": 18.25, "prob": 0.025240320712327957}, {"token_id": 3555, "piece": " What", "norm": "what", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 12960, "piece": " Machine", "norm": "machine", "logit": 18.125, "prob": 0.022274503484368324}, {"token_id": 2885, "piece": " Data", "norm": "data", "logit": 17.875, "prob": 0.01734740100800991}, {"t
- `FAIL` `retrieval_topk_semantic_shift`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "rows": [{"prompt": "A strong explanation should mention", "music_no_prefix": [{"token_id": 279, "piece": " the", "norm": "the", "logit": 21.125, "prob": 0.31038299202919006}, {"token_id": 518, "piece": " at", "norm": "at", "logit": 19.5, "prob": 0.06111803650856018}, {"token_id": 264, "piece": " a", "norm": "a", "logit": 19.375, "prob": 0.05393647775053978}, {"token_id": 2176, "piece": " both", "norm": "both", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 3151, "piece": " specific", "norm": "specific", "logit": 19.0, "prob": 0.03706996142864227}, {"token_id": 429, "piece": " that", "norm": "that", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 1246, "piece": " how", "norm": "how", "logit": 18.625, "prob": 0.025477787479758263}, {"token_id": 678, "piece": " all", "norm": "all", "logit": 18.5, "prob": 0.0224840696901083}, {"token_id": 1029
- `PASS` `repetition_segment_audit`: {"aggregate": {"bad_segment_ratio": 0.09090909090909091, "total_segments": 11, "bad_segments": 1, "early_collapse_prompts": []}, "rows": [{"prompt": "The pianist", "output": "The pianist will arrive at 7:15. 德国人德国人澳大利亚人都喜欢吃披萨。 The Germans, the Danes and Australians all like pizza . 敂�将在电影院等我 You will wait for me at a cinema. �", "generated_token_count": 20, "window": 8, "segments": [{"segment_idx": 0, "tokens": ["will", "arrive", "at", "the", "germans", "the", "danes", "and"], "unique_ratio": 0.875, "content_ratio": 0.5, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.25}, {"segment_idx": 1, "tokens": ["australians", "all", "like", "pizza", "you", "will", "wait", "for"], "unique_ratio": 1.0, "content_ratio": 0.625, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.125}, {"segment_idx": 2, "tokens": ["me", "at", "a", "cinema"], "unique_ratio": 1.0, "content_ratio": 0.25, "repeated_bigram_ratio": 0.0, "dominant_token_share": 0.25}], "bad_segments": [], "first_bad_segment_idx": null}, {"prompt": "The telescope", "output": "The telescope you lent me is mine now．（同义句型转换） The telescope _____. Mozart的唱片已经售罄。（完成完形ifnfo/Mozart oper album has been sold out. 20名学生参加了这次", "genera
- `FAIL` `prefix_stepwise_drift_trajectory`: {"rows": [{"prompt": "Key piano ideas include", "first_bad_step": 0, "decoded_output": "Key piano ideas include: 1. The use of a grand piano for its rich, resonant", "rows": [{"step": 0, "top1": {"token_id": 25, "piece": ":", "norm": "", "logit": 17.625, "prob": 0.12786318361759186}, "top1_category": "punct", "topk_category_counts": {"semantic": 0, "functional": 3, "punct": 9}, "topk_category_prob_mass": {"semantic": 0.0, "functional": 0.18933850806206465, "punct": 0.33162419497966766}, "chosen_token_id": 25, "chosen_piece": ":", "chosen_norm": "", "chosen_category": "punct"}, {"step": 1, "top1": {"token_id": 220, "piece": " ", "norm": "", "logit": 15.625, "prob": 0.10892719775438309}, "top1_category": "punct", "topk_category_counts": {"semantic": 1, "functional": 4, "punct": 7}, "topk_category_prob_mass": {"semantic": 0.007412588689476252, "functional": 0.08527705539017916, "punct": 0.21729147899895906}, "chosen_token_id": 220, "chosen_piece": " ", "chosen_norm": "", "chosen_category": "punct"}, {"step": 2, "top1": {"token_id": 16, "piece": "1", "norm": "", "logit": 18.375, "prob": 0.6832453608512878}, "top1_category": "punct", "topk_category_counts": {"semantic": 0, "functional":
- `FAIL` `retrieval_generation_alignment_audit`: {"music_keywords": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space_keywords": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"], "diagnoses": {"aligned": 1, "retrieval_miss": 1, "bridge_unused": 1, "unknown": 0}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_mids": [3, 5, 1, 6, 7], "retrieved_label_counts": {"music": 2, "space": 3}, "retrieved_majority_label": "space", "retrieved_text_preview": ["A conservatory student studied etudes, scales, and expressive voicing on the keyboard.", "Orbital mechanics explains how satellites and planets move under gravitational force.", "A musician refined finger technique, phrasing, and pedal control on the piano."], "output": "What improves piano technique and musical phrasing? Options:a．Practiceb. technique practicec .practice techniquesd.dont play too fast\\nScience Chemistry - chemistry of life help please!!!", "music_score": 0.15, "space_score": 0.0, "generated_label": "music", "diagn
- `PASS` `retrieval_prefix_decode_correlation_audit`: {"correlations": {"retrieval_strength__prefix_l2": 0.47129727854893727, "retrieval_strength__bad_decode_score": -0.5153013642952196, "prefix_l2__bad_decode_score": -0.8119768806119197}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "retrieved_scored": [{"mid": 6, "score": -0.46566978096961975}, {"mid": 4, "score": -0.5127263069152832}, {"mid": 7, "score": -0.6141006350517273}, {"mid": 5, "score": -0.6516022086143494}, {"mid": 2, "score": -0.7098392248153687}], "retrieved_label_counts": {"space": 4, "music": 1}, "retrieval_strength": -0.7098392248153687, "prefix_l2_shift": 1036.929443359375, "prefix_js_divergence": 0.3171480894088745, "top1_with_prefix": {"token_id": 220, "piece": " ", "norm": "", "logit": 12.0, "prob": 0.05946128070354462}, "top1_category_with_prefix": "punct", "topk_non_semantic_prob_mass": 0.2697937758639455}, {"prompt": "What explains satellites and orbital motion?", "expected_label": "space", "retrieved_scored": [{"mid": 6, "score": -0.496165931224823}, {"mid": 4, "score": -0.5487112402915955}, {"mid": 7, "score": -0.6517217755317688}, {"mid": 5, "score": -0.684630811214447}, {"mid": 2, "score": -0.7428811
- `FAIL` `stepwise_label_mass_alignment_audit`: {"label_keywords": {"music": ["pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight", "musician", "refined", "finger", "technique", "phrasing", "pedal"], "space": ["distant", "astronomers", "observed", "galaxies", "quasars", "stellar", "evolution", "space", "orbital", "mechanics", "explains", "satellites"]}, "rows": [{"prompt": "What improves piano technique and musical phrasing?", "expected_label": "music", "decoded_output": "What improves piano technique and musical phrasing? 1. **Understanding the Question**: The question asks about", "stage_counts": {"inject": 8, "retrieve": 4}, "rows": [{"step": 0, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 0.19865716993808746, "space": 0.1326199248433113}, "logits_label_mass": {"music": 0, "space": 0}, "top1_piece": " ", "top1_category": "punct", "chosen_piece": " ", "chosen_category": "punct", "chosen_label": null, "diagnosed_stage": "inject"}, {"step": 1, "retrieved_majority_label": "music", "retrieved_label_counts": {"music": 3, "space": 2}, "retrieved_score_sum": {"music": 0.19865716993808746, "space": 0.1326199248433113}, "logits_label_mass": {
- `PASS` `prompt_diversity_without_memory`: {"prompts": ["The pianist", "Quantum systems", "The rainforest"], "outputs": ["The pianist-ios版是一款什么类型的软件？\nA. 游戏<\\/p> <-{amic", "Quantum systems系列产品有哪些产品系列？\nABCDE ABCD 解析：华为云量子计算品牌“澜", "The rainforest公司将推出自家操作系统后，以下哪个平台提供每日、每周或每月发布净水行业"], "unique_count": 3}
- `PASS` `save_load_consistency`: {"prompt": "The pianist", "output_a": "The pianist的官方预告片！视频中可以看到Dragonboat Festival乐队主唱Peter KnighterPhoenix", "output_b": "The pianist的官方预告片！视频中可以看到Dragonboat Festival乐队主唱Peter KnighterPhoenix"}
- `PASS` `training_cache_isolation`: {"changed": [], "memory_count": 8}
- `PASS` `cheating_heuristics`: {"outputs": ["The pianist played a piano piece called \"The Beatles Theme from A Hard Day's Night,\" which is", "The telescope is on Tommy ________ desk.________ he be Tom？ A：a； could B", "The trader bought a pair of trainers for Kate last year at 0.65 euros and sold", "The child _____ in the classroom suddenly fell off his _____. Fill blanks: graduation______. A falls"], "exact_same": false, "prefix_only": false, "too_short": false}

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
    "total": 260.33465576171875,
    "recon": 3.027585029602051,
    "contrast": 123.25140380859375,
    "holonomy": 42737.06640625,
    "write_policy": 2.718179941177368,
    "semantic_probe": 0.0,
    "dir_diversity": 0.0,
    "reranker_ranking": 0.0,
    "encoder_throughput": 3.9514379501342773,
    "vocab_anchor": -0.0,
    "semantic_alignment": 9.985305786132812,
    "tail_semantic_anchor": 10.003637313842773,
    "grad_norms": {
      "ctx_encoder": 6.533581582490527e-12,
      "fib_encoder": 2.731876231929138e-09,
      "dir_predictor": 1.0382596359625192e-13,
      "fiber_connection": 4.093225671603302e-07,
      "fiber_attn": 1.0263617963792416e-10,
      "reranker": 2.0931047061238405e-12,
      "qformer": 4.171810563356554e-09,
      "content_bypass": 5.131816747590266e-10,
      "semantic_probe": 0.0,
      "layer_pool": 5.5969056056426325e-09,
      "prefix_aligner": 8.353048220281766e-11,
      "vocab_proj": 1.0000050593801983,
      "tail_head": 3.83927238081513e-09
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
      "vocab_anchor": 0.2
    }
  },
  "metric_grad_norms": [
    3.273699977768274e-09,
    4.7181248491456884e-11,
    2.087454209487305e-09,
    5.082130549727282e-11,
    1.5883928838889005e-08,
    4.730260072527415e-10
  ],
  "metric_param_deltas": [
    3.108839882770553e-05,
    4.711087058240082e-07,
    2.0694133127108216e-05,
    5.07268111960002e-07,
    0.00014018247020430863,
    4.674165211326908e-06
  ],
  "max_metric_grad_norm": 1.5883928838889005e-08,
  "max_metric_param_delta": 0.00014018247020430863,
  "error": null
}
```

## No-Grad Generation

```json
{
  "passed": true,
  "stored_memories": 8,
  "output": "The pianist played piano and sang a beautiful —— at the concert yesterday evening. A.song B.songs C.playing D.sh",
  "error": null
}
```

## Counterfactual Memory Influence

```json
{
  "passed": false,
  "prompt": "Tell me something about practice and performance.",
  "music_output": "Tell me something about practice and performance. Practice and performance are related but distinct concepts. **Practice** refers to the act of practicing a skill or activity repeatedly,",
  "space_output": "Tell me something about practice and performance. Practice and performance are related but distinct concepts. **Practice** refers to the act of practicing a skill or activity repeatedly,",
  "outputs_differ": false,
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
  "blank_output": "Explain what someone should focus on when improving technique and understanding the subject. Explain what someone should focus on when improving technique and understanding the subject\\n要提升寫Explain difference between\n为了回答這個問題，我需要更多的具體",
  "music_output": "Explain what someone should focus on when improving technique and understanding the subject. BMW badge badges are a type of ExExplain what someone should focus on when improving technique and understanding the symbolism:\n\n\nThe first digit 'B' stands for",
  "space_output": "Explain what someone should focus on when improving technique and understanding the subject. Technique refers to______． A.the way in which the subject is expressed or performed. B、the content of a person's thoughts and feelings C、how",
  "blank_music_score": 0.13333333333333333,
  "blank_space_score": 0.0,
  "music_music_score": 0.1111111111111111,
  "music_space_score": 0.0,
  "space_space_score": 0.0,
  "space_music_score": 0.13333333333333333,
  "music_margin": 0.1111111111111111,
  "space_margin": -0.13333333333333333,
  "music_lift": -0.022222222222222227,
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
      "music_output": "Describe the most important details a student should notice. 很多人认为篮球比赛是______的，因为_____。请根据以上这段文本内容写成为什么很多人会说“乒乓球",
      "space_output": "Describe the most important details a student should notice. Respond finished loading...\\nHome » Aptitude Test Results \\\"defense lawyer girl\\\" dan \"best online dating sites for men\\n CS 2",
      "music_margin": 0.0,
      "space_margin": 0.0,
      "passed": false
    },
    {
      "prompt": "Summarize the key ideas a learner should practice and remember.",
      "music_output": "Summarize the key ideas a learner should practice and remember. Love reading and writing stories, practice speaking English every day，and ask your teacher for help.____伍经理如何评价新员工小王",
      "space_output": "Summarize the key ideas a learner should practice and remember. Love your job: This means showing genuine interest in the material, being interested to learn new things related\nA. Remember key ideas about perseverance",
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
      "output": "The pianist played here again and her performance was great renvoie < Fill in the dots > encore\n  €50€1.962",
      "token_count": 14,
      "unique_token_ratio": 1.0,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.04672897196261682,
      "newline_ratio": 0.009345794392523364,
      "alpha_ratio": 0.7102803738317757,
      "content_token_ratio": 0.6428571428571429,
      "generated_preview": "played here again and her performance was great renvoie fill in the dots encore"
    },
    {
      "prompt": "The telescope",
      "output": "The telescope's schedule for the conference is as follows: it will be active from 10 AM to **5 PM**. Therefore, option C",
      "token_count": 20,
      "unique_token_ratio": 0.95,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.06666666666666667,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.7333333333333333,
      "content_token_ratio": 0.4,
      "generated_preview": "the telescope's schedule for the conference is as follows it will be active from am to pm therefore option c"
    },
    {
      "prompt": "The forest path",
      "output": "The forest path đi qua những cánh rừng cây xanh mát, núi đồi và các ngọn suối nhỏ. Đó là con đường của người dân",
      "token_count": 33,
      "unique_token_ratio": 0.5151515151515151,
      "repeated_bigram_ratio": 0.0625,
      "max_token_run": 2,
      "punct_ratio": 0.017857142857142856,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.7678571428571429,
      "content_token_ratio": 0.030303030303030304,
      "generated_preview": "i qua nh ng c nh r ng c y xanh m t n i i v c c ng n su i nh"
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst said the market for electric vehicles is expected to grow at a compound annual growth rate of 15% over\n答案:\n\nAssistant: sad",
      "token_count": 20,
      "unique_token_ratio": 1.0,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.02127659574468085,
      "newline_ratio": 0.02127659574468085,
      "alpha_ratio": 0.7872340425531915,
      "content_token_ratio": 0.55,
      "generated_preview": "said the market for electric vehicles is expected to grow at a compound annual growth rate of over assistant sad"
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly\\nprevious Next -->#1/2 + 3 =.5+0, explain the ExplanationsCompelling Essays on this",
      "token_count": 8,
      "unique_token_ratio": 1.0,
      "repeated_bigram_ratio": 0.0,
      "max_token_run": 1,
      "punct_ratio": 0.10091743119266056,
      "newline_ratio": 0.0,
      "alpha_ratio": 0.7247706422018348,
      "content_token_ratio": 0.625,
      "generated_preview": "nprevious next explain the explanationscompelling essays on this"
    }
  ],
  "aggregate": {
    "avg_unique_token_ratio": 0.893030303030303,
    "avg_repeated_bigram_ratio": 0.0125,
    "avg_content_token_ratio": 0.44963203463203466,
    "avg_newline_ratio": 0.006124478027440842,
    "worst_max_token_run": 2,
    "short_or_hollow_prompts": [
      "The forest path"
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
    "js_divergence": 0.43499448895454407,
    "l2_shift": 1075.050048828125,
    "topk_overlap_count": 1,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 4.675998687744141,
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
        "logit": 16.5,
        "prob": 0.17788897454738617
      },
      {
        "token_id": 10236,
        "piece": " �",
        "norm": "",
        "logit": 15.5625,
        "prob": 0.06966232508420944
      },
      {
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 15.4375,
        "prob": 0.06147678941488266
      },
      {
        "token_id": 4891,
        "piece": " �",
        "norm": "",
        "logit": 15.0,
        "prob": 0.03969239816069603
      },
      {
        "token_id": 320,
        "piece": " (",
        "norm": "",
        "logit": 14.875,
        "prob": 0.03502841666340828
      },
      {
        "token_id": 330,
        "piece": " \"",
        "norm": "",
        "logit": 14.875,
        "prob": 0.03502841666340828
      },
      {
        "token_id": 49434,
        "piece": " �",
        "norm": "",
        "logit": 14.3125,
        "prob": 0.01995858922600746
      },
      {
        "token_id": 6567,
        "piece": " �",
        "norm": "",
        "logit": 14.25,
        "prob": 0.018749359995126724
      },
      {
        "token_id": 69162,
        "piece": " 对",
        "norm": "",
        "logit": 14.25,
        "prob": 0.018749359995126724
      },
      {
        "token_id": 73562,
        "piece": " 在",
        "norm": "",
        "logit": 14.125,
        "prob": 0.016546253114938736
      },
      {
        "token_id": 51461,
        "piece": " �",
        "norm": "",
        "logit": 14.0625,
        "prob": 0.01554376445710659
      },
      {
        "token_id": 32181,
        "piece": " �",
        "norm": "",
        "logit": 14.0625,
        "prob": 0.01554376445710659
      }
    ]
  },
  "memory": {
    "js_divergence": 0.22554050385951996,
    "l2_shift": 704.2648315429688,
    "topk_overlap_count": 3,
    "entropy_no_prefix": 5.256593227386475,
    "entropy_with_prefix": 6.019233703613281,
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
        "token_id": 576,
        "piece": " The",
        "norm": "the",
        "logit": 16.375,
        "prob": 0.09373567998409271
      },
      {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 16.0,
        "prob": 0.06442353129386902
      },
      {
        "token_id": 22555,
        "piece": " Sure",
        "norm": "sure",
        "logit": 15.75,
        "prob": 0.05017309635877609
      },
      {
        "token_id": 5209,
        "piece": " Please",
        "norm": "please",
        "logit": 15.0625,
        "prob": 0.02522861585021019
      },
      {
        "token_id": 2014,
        "piece": " To",
        "norm": "to",
        "logit": 15.0,
        "prob": 0.023700091987848282
      },
      {
        "token_id": 358,
        "piece": " I",
        "norm": "i",
        "logit": 14.9375,
        "prob": 0.02226417511701584
      },
      {
        "token_id": 1096,
        "piece": " This",
        "norm": "this",
        "logit": 14.875,
        "prob": 0.020915258675813675
      },
      {
        "token_id": 758,
        "piece": " In",
        "norm": "in",
        "logit": 14.875,
        "prob": 0.020915258675813675
      },
      {
        "token_id": 3070,
        "piece": " **",
        "norm": "",
        "logit": 14.6875,
        "prob": 0.01733935810625553
      },
      {
        "token_id": 715,
        "piece": " \n",
        "norm": "",
        "logit": 14.6875,
        "prob": 0.01733935810625553
      },
      {
        "token_id": 21806,
        "piece": " Answer",
        "norm": "answer",
        "logit": 14.6875,
        "prob": 0.01733935810625553
      },
      {
        "token_id": 81917,
        "piece": " Explain",
        "norm": "explain",
        "logit": 14.5625,
        "prob": 0.015301928855478764
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
          "logit": 21.125,
          "prob": 0.5324241518974304
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 19.625,
          "prob": 0.11879988759756088
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.5,
          "prob": 0.03856867924332619
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 18.375,
          "prob": 0.03403673693537712
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 18.125,
          "prob": 0.02650783769786358
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 17.75,
          "prob": 0.018218552693724632
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.375,
          "prob": 0.012521415948867798
        },
        {
          "token_id": 1045,
          "piece": " some",
          "norm": "some",
          "logit": 17.25,
          "prob": 0.011050110682845116
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 17.0,
          "prob": 0.008605835027992725
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 17.0,
          "prob": 0.008605835027992725
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 16.875,
          "prob": 0.007594622205942869
        },
        {
          "token_id": 3807,
          "piece": " several",
          "norm": "several",
          "logit": 16.75,
          "prob": 0.006702231243252754
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
          "logit": 21.25,
          "prob": 0.5463313460350037
        },
        {
          "token_id": 429,
          "piece": " that",
          "norm": "that",
          "logit": 19.75,
          "prob": 0.12190300226211548
        },
        {
          "token_id": 1246,
          "piece": " how",
          "norm": "how",
          "logit": 18.625,
          "prob": 0.03957611322402954
        },
        {
          "token_id": 264,
          "piece": " a",
          "norm": "a",
          "logit": 18.375,
          "prob": 0.03082190454006195
        },
        {
          "token_id": 518,
          "piece": " at",
          "norm": "at",
          "logit": 18.125,
          "prob": 0.024004124104976654
        },
        {
          "token_id": 2176,
          "piece": " both",
          "norm": "both",
          "logit": 17.75,
          "prob": 0.01649777777493
        },
        {
          "token_id": 3151,
          "piece": " specific",
          "norm": "specific",
          "logit": 17.5,
          "prob": 0.012848482467234135
        },
        {
          "token_id": 1045,
          "piece": " some",
          "norm": "some",
          "logit": 17.375,
          "prob": 0.01133874524384737
        },
        {
          "token_id": 1378,
          "piece": " two",
          "norm": "two",
          "logit": 17.0,
          "prob": 0.007792997639626265
        },
        {
          "token_id": 10295,
          "piece": " examples",
          "norm": "examples",
          "logit": 17.0,
          "prob": 0.007792997639626265
        },
        {
          "token_id": 3170,
          "piece": " why",
          "norm": "why",
          "logit": 16.875,
          "prob": 0.006877296604216099
        },
        {
          "token_id": 2326,
          "piece": " three",
          "norm": "three",
          "logit": 16.875,
          "prob": 0.006877296604216099
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
          "logit": 17.625,
          "prob": 0.2709504961967468
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 16.125,
          "prob": 0.06045722961425781
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 16.0,
          "prob": 0.05335331708192825
        },
        {
          "token_id": 2130,
          "piece": "____",
          "norm": "",
          "logit": 15.8125,
          "prob": 0.04423145204782486
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 15.5625,
          "prob": 0.03444749116897583
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 15.5,
          "prob": 0.03236042335629463
        },
        {
          "token_id": 362,
          "piece": " A",
          "norm": "a",
          "logit": 15.1875,
          "prob": 0.02367538958787918
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 15.0625,
          "prob": 0.020893458276987076
        },
        {
          "token_id": 220,
          "piece": " ",
          "norm": "",
          "logit": 14.8125,
          "prob": 0.016271842643618584
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 14.75,
          "prob": 0.015285980887711048
        },
        {
          "token_id": 320,
          "piece": " (",
          "norm": "",
          "logit": 14.75,
          "prob": 0.015285980887711048
        },
        {
          "token_id": 5122,
          "piece": "：",
          "norm": "",
          "logit": 14.625,
          "prob": 0.013489830307662487
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
          "logit": 16.375,
          "prob": 0.2082545906305313
        },
        {
          "token_id": 25,
          "piece": ":",
          "norm": "",
          "logit": 15.25,
          "prob": 0.06761036813259125
        },
        {
          "token_id": 279,
          "piece": " the",
          "norm": "the",
          "logit": 15.0625,
          "prob": 0.05605096369981766
        },
        {
          "token_id": 2130,
          "piece": "____",
          "norm": "",
          "logit": 14.75,
          "prob": 0.04100776091217995
        },
        {
          "token_id": 311,
          "piece": " to",
          "norm": "to",
          "logit": 14.6875,
          "prob": 0.03852322697639465
        },
        {
          "token_id": 32671,
          "piece": " ______",
          "norm": "",
          "logit": 14.3125,
          "prob": 0.026476601138710976
        },
        {
          "token_id": 30743,
          "piece": " ____",
          "norm": "",
          "logit": 14.25,
          "prob": 0.024872465059161186
        },
        {
          "token_id": 362,
          "piece": " A",
          "norm": "a",
          "logit": 14.25,
          "prob": 0.024872465059161186
        },
        {
          "token_id": 220,
          "piece": " ",
          "norm": "",
          "logit": 14.25,
          "prob": 0.024872465059161186
        },
        {
          "token_id": 320,
          "piece": " (",
          "norm": "",
          "logit": 14.0,
          "prob": 0.01937069557607174
        },
        {
          "token_id": 1304,
          "piece": " __",
          "norm": "",
          "logit": 13.75,
          "prob": 0.015085912309587002
        },
        {
          "token_id": 198,
          "piece": "\n",
          "norm": "",
          "logit": 13.75,
          "prob": 0.015085912309587002
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
    "bad_segment_ratio": 0.09090909090909091,
    "total_segments": 11,
    "bad_segments": 1,
    "early_collapse_prompts": []
  },
  "rows": [
    {
      "prompt": "The pianist",
      "output": "The pianist will arrive at 7:15. 德国人德国人澳大利亚人都喜欢吃披萨。 The Germans, the Danes and Australians all like pizza . 敂�将在电影院等我 You will wait for me at a cinema. �",
      "generated_token_count": 20,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "will",
            "arrive",
            "at",
            "the",
            "germans",
            "the",
            "danes",
            "and"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.5,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 1,
          "tokens": [
            "australians",
            "all",
            "like",
            "pizza",
            "you",
            "will",
            "wait",
            "for"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 2,
          "tokens": [
            "me",
            "at",
            "a",
            "cinema"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.25,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "The telescope",
      "output": "The telescope you lent me is mine now．（同义句型转换） The telescope _____. Mozart的唱片已经售罄。（完成完形ifnfo/Mozart oper album has been sold out. 20名学生参加了这次",
      "generated_token_count": 17,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "you",
            "lent",
            "me",
            "is",
            "mine",
            "now",
            "the",
            "telescope"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.375,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 1,
          "tokens": [
            "mozart",
            "ifnfo",
            "mozart",
            "oper",
            "album",
            "has",
            "been",
            "sold"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "out"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "bad_segments": [
        {
          "segment_idx": 2,
          "tokens": [
            "out"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.0,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 1.0
        }
      ],
      "first_bad_segment_idx": 2
    },
    {
      "prompt": "The market analyst",
      "output": "The market analyst role is for you!Do not hesitate． We offer： ● Competitive salary and competitive benefits● A ________ chance to work with China's top universities' students， teachers, professionals in the fashion industry.____\nA．unique ● Competitive",
      "generated_token_count": 32,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "role",
            "is",
            "for",
            "you",
            "do",
            "not",
            "hesitate",
            "we"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.25,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 1,
          "tokens": [
            "offer",
            "competitive",
            "salary",
            "and",
            "competitive",
            "benefits",
            "a",
            "chance"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.75,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
        },
        {
          "segment_idx": 2,
          "tokens": [
            "to",
            "work",
            "with",
            "china's",
            "top",
            "universities'",
            "students",
            "teachers"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        },
        {
          "segment_idx": 3,
          "tokens": [
            "professionals",
            "in",
            "the",
            "fashion",
            "industry",
            "a",
            "unique",
            "competitive"
          ],
          "unique_ratio": 1.0,
          "content_ratio": 0.625,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.125
        }
      ],
      "bad_segments": [],
      "first_bad_segment_idx": null
    },
    {
      "prompt": "Explain the topic clearly",
      "output": "Explain the topic clearly is crucial because______(确保她相信学习化学的重要性)． [smoke)—Explain###induce#ensure【解析】because引导原因状语从句，根据汉语提示“确保她相信学习化学的重要性”可知用",
      "generated_token_count": 8,
      "window": 8,
      "segments": [
        {
          "segment_idx": 0,
          "tokens": [
            "is",
            "crucial",
            "because",
            "smoke",
            "explain",
            "induce",
            "ensure",
            "because"
          ],
          "unique_ratio": 0.875,
          "content_ratio": 0.875,
          "repeated_bigram_ratio": 0.0,
          "dominant_token_share": 0.25
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
      "decoded_output": "Key piano ideas include: 1. The use of a grand piano for its rich, resonant",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 25,
            "piece": ":",
            "norm": "",
            "logit": 17.625,
            "prob": 0.12786318361759186
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 3,
            "punct": 9
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.18933850806206465,
            "punct": 0.33162419497966766
          },
          "chosen_token_id": 25,
          "chosen_piece": ":",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 1,
          "top1": {
            "token_id": 220,
            "piece": " ",
            "norm": "",
            "logit": 15.625,
            "prob": 0.10892719775438309
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 4,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.007412588689476252,
            "functional": 0.08527705539017916,
            "punct": 0.21729147899895906
          },
          "chosen_token_id": 220,
          "chosen_piece": " ",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 2,
          "top1": {
            "token_id": 16,
            "piece": "1",
            "norm": "",
            "logit": 18.375,
            "prob": 0.6832453608512878
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
            "punct": 0.8175897472538054
          },
          "chosen_token_id": 16,
          "chosen_piece": "1",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 3,
          "top1": {
            "token_id": 13,
            "piece": ".",
            "norm": "",
            "logit": 19.25,
            "prob": 0.4662233889102936
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
            "punct": 0.9262601176742464
          },
          "chosen_token_id": 13,
          "chosen_piece": ".",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 4,
          "top1": {
            "token_id": 576,
            "piece": " The",
            "norm": "the",
            "logit": 15.0,
            "prob": 0.09447456151247025
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 6,
            "punct": 5
          },
          "topk_category_prob_mass": {
            "semantic": 0.014488143846392632,
            "functional": 0.2122769709676504,
            "punct": 0.08204689994454384
          },
          "chosen_token_id": 576,
          "chosen_piece": " The",
          "chosen_norm": "the",
          "chosen_category": "functional"
        },
        {
          "step": 5,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 17.625,
            "prob": 0.0675923079252243
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 9,
            "functional": 2,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.24230931047350168,
            "functional": 0.1272423081099987,
            "punct": 0.007583647035062313
          },
          "chosen_token_id": 990,
          "chosen_piece": " use",
          "chosen_norm": "use",
          "chosen_category": "functional"
        },
        {
          "step": 6,
          "top1": {
            "token_id": 315,
            "piece": " of",
            "norm": "of",
            "logit": 25.75,
            "prob": 0.9977940320968628
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 9,
            "punct": 3
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.9987860523579002,
            "punct": 0.0002597432230686536
          },
          "chosen_token_id": 315,
          "chosen_piece": " of",
          "chosen_norm": "of",
          "chosen_category": "functional"
        },
        {
          "step": 7,
          "top1": {
            "token_id": 264,
            "piece": " a",
            "norm": "a",
            "logit": 19.125,
            "prob": 0.10830729454755783
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.17387889418751,
            "functional": 0.19265709817409515,
            "punct": 0.0
          },
          "chosen_token_id": 264,
          "chosen_piece": " a",
          "chosen_norm": "a",
          "chosen_category": "functional"
        },
        {
          "step": 8,
          "top1": {
            "token_id": 6662,
            "piece": " grand",
            "norm": "grand",
            "logit": 18.375,
            "prob": 0.08683046698570251
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.33784899953752756,
            "functional": 0.05276703368872404,
            "punct": 0.0
          },
          "chosen_token_id": 6662,
          "chosen_piece": " grand",
          "chosen_norm": "grand",
          "chosen_category": "semantic"
        },
        {
          "step": 9,
          "top1": {
            "token_id": 26278,
            "piece": " piano",
            "norm": "piano",
            "logit": 22.875,
            "prob": 0.9613589644432068
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 4,
            "punct": 2
          },
          "topk_category_prob_mass": {
            "semantic": 0.9714400037191808,
            "functional": 0.015464706753846258,
            "punct": 0.0022699576220475137
          },
          "chosen_token_id": 26278,
          "chosen_piece": " piano",
          "chosen_norm": "piano",
          "chosen_category": "semantic"
        },
        {
          "step": 10,
          "top1": {
            "token_id": 369,
            "piece": " for",
            "norm": "for",
            "logit": 20.375,
            "prob": 0.19226950407028198
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 0,
            "functional": 8,
            "punct": 4
          },
          "topk_category_prob_mass": {
            "semantic": 0.0,
            "functional": 0.6617485322058201,
            "punct": 0.13422521948814392
          },
          "chosen_token_id": 369,
          "chosen_piece": " for",
          "chosen_norm": "for",
          "chosen_category": "functional"
        },
        {
          "step": 11,
          "top1": {
            "token_id": 1181,
            "piece": " its",
            "norm": "its",
            "logit": 20.75,
            "prob": 0.23610667884349823
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 8,
            "functional": 4,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.16647457890212536,
            "functional": 0.5596857713535428,
            "punct": 0.0
          },
          "chosen_token_id": 1181,
          "chosen_piece": " its",
          "chosen_norm": "its",
          "chosen_category": "functional"
        },
        {
          "step": 12,
          "top1": {
            "token_id": 9080,
            "piece": " rich",
            "norm": "rich",
            "logit": 22.375,
            "prob": 0.40728428959846497
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.7111744908615947,
            "functional": 0.06896425969898701,
            "punct": 0.0
          },
          "chosen_token_id": 9080,
          "chosen_piece": " rich",
          "chosen_norm": "rich",
          "chosen_category": "semantic"
        },
        {
          "step": 13,
          "top1": {
            "token_id": 11,
            "piece": ",",
            "norm": "",
            "logit": 24.375,
            "prob": 0.45172712206840515
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 6,
            "functional": 5,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.1752969198860228,
            "functional": 0.35670430050231516,
            "punct": 0.45172712206840515
          },
          "chosen_token_id": 11,
          "chosen_piece": ",",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 28180,
            "piece": " reson",
            "norm": "reson",
            "logit": 22.5,
            "prob": 0.33740657567977905
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.8690649289637804,
            "functional": 0.05047586001455784,
            "punct": 0.0
          },
          "chosen_token_id": 28180,
          "chosen_piece": " reson",
          "chosen_norm": "reson",
          "chosen_category": "semantic"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 517,
            "piece": "ant",
            "norm": "ant",
            "logit": 28.0,
            "prob": 0.9888412356376648
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.011117235548226745,
            "functional": 0.9888481202515322,
            "punct": 0.0
          },
          "chosen_token_id": 517,
          "chosen_piece": "ant",
          "chosen_norm": "ant",
          "chosen_category": "functional"
        }
      ],
      "passed": false
    },
    {
      "prompt": "Explain the topic clearly",
      "first_bad_step": 0,
      "decoded_output": "Explain the topic clearly and provide a detailed answer. 请问您想了解什么主题？请提供",
      "rows": [
        {
          "step": 0,
          "top1": {
            "token_id": 323,
            "piece": " and",
            "norm": "and",
            "logit": 19.75,
            "prob": 0.29089707136154175
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 4,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.021072514355182648,
            "functional": 0.33393036108464,
            "punct": 0.48121924325823784
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
            "logit": 20.5,
            "prob": 0.3041878044605255
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 2,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.5836828779429197,
            "functional": 0.03540037106722593,
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
            "logit": 23.125,
            "prob": 0.3568190634250641
          },
          "top1_category": "functional",
          "topk_category_counts": {
            "semantic": 4,
            "functional": 7,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.1275522243231535,
            "functional": 0.6550241876393557,
            "punct": 0.025847887620329857
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
            "logit": 21.625,
            "prob": 0.21949423849582672
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 12,
            "functional": 0,
            "punct": 0
          },
          "topk_category_prob_mass": {
            "semantic": 0.6207721829414368,
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
            "token_id": 16148,
            "piece": " explanation",
            "norm": "explanation",
            "logit": 21.375,
            "prob": 0.23082181811332703
          },
          "top1_category": "semantic",
          "topk_category_counts": {
            "semantic": 10,
            "functional": 1,
            "punct": 1
          },
          "topk_category_prob_mass": {
            "semantic": 0.8106414331123233,
            "functional": 0.014755944721400738,
            "punct": 0.01672067679464817
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
            "logit": 22.375,
            "prob": 0.347540944814682
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 1,
            "functional": 3,
            "punct": 8
          },
          "topk_category_prob_mass": {
            "semantic": 0.017303043976426125,
            "functional": 0.07726758439093828,
            "punct": 0.8144159568473697
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
            "logit": 17.25,
            "prob": 0.18577441573143005
          },
          "top1_category": "punct",
          "topk_category_counts": {
            "semantic": 3,
            "functional": 2,
            "punct": 7
          },
          "topk_category_prob_mass": {
            "semantic": 0.06556552834808826,
            "functional": 0.047606656327843666,
            "punct": 0.3611793518066406
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
            "logit": 18.75,
            "prob": 0.2687368392944336
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
            "punct": 0.7193908896297216
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
            "logit": 15.4375,
            "prob": 0.23672355711460114
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
            "punct": 0.5552087984979153
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
            "logit": 18.625,
            "prob": 0.25641879439353943
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
            "punct": 0.8415695177391171
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
            "logit": 19.75,
            "prob": 0.4153730869293213
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
            "punct": 0.8661790871992707
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
            "logit": 21.25,
            "prob": 0.7669149041175842
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
            "punct": 0.9736726651899517
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
            "logit": 17.625,
            "prob": 0.20356985926628113
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
            "punct": 0.8066401928663254
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
            "logit": 22.375,
            "prob": 0.5806931257247925
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
            "punct": 0.9654297647066414
          },
          "chosen_token_id": 11319,
          "chosen_piece": "？",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 14,
          "top1": {
            "token_id": 14880,
            "piece": "请",
            "norm": "",
            "logit": 17.25,
            "prob": 0.3568967878818512
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
            "punct": 0.6588917188346386
          },
          "chosen_token_id": 14880,
          "chosen_piece": "请",
          "chosen_norm": "",
          "chosen_category": "punct"
        },
        {
          "step": 15,
          "top1": {
            "token_id": 99553,
            "piece": "提供",
            "norm": "",
            "logit": 22.5,
            "prob": 0.43682757019996643
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
            "punct": 0.975669561419636
          },
          "chosen_token_id": 99553,
          "chosen_piece": "提供",
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
        5,
        1,
        6,
        7
      ],
      "retrieved_label_counts": {
        "music": 2,
        "space": 3
      },
      "retrieved_majority_label": "space",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "Orbital mechanics explains how satellites and planets move under gravitational force.",
        "A musician refined finger technique, phrasing, and pedal control on the piano."
      ],
      "output": "What improves piano technique and musical phrasing? Options:a．Practiceb. technique practicec .practice techniquesd.dont play too fast\\nScience Chemistry - chemistry of life help please!!!",
      "music_score": 0.15,
      "space_score": 0.0,
      "generated_label": "music",
      "diagnosis": "retrieval_miss",
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_mids": [
        3,
        5,
        1,
        6,
        7
      ],
      "retrieved_label_counts": {
        "music": 2,
        "space": 3
      },
      "retrieved_majority_label": "space",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "Orbital mechanics explains how satellites and planets move under gravitational force.",
        "A musician refined finger technique, phrasing, and pedal control on the piano."
      ],
      "output": "What explains satellites and orbital motion? satellites的解释) - Wikipedia中文版 is available at <https://en.wikipedia.org/wiki/Satellite>．\nAnswer all parts of",
      "music_score": 0.0,
      "space_score": 0.3076923076923077,
      "generated_label": "space",
      "diagnosis": "aligned",
      "passed": true
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_mids": [
        3,
        5,
        1,
        6,
        7
      ],
      "retrieved_label_counts": {
        "music": 2,
        "space": 3
      },
      "retrieved_majority_label": "space",
      "retrieved_text_preview": [
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
        "Orbital mechanics explains how satellites and planets move under gravitational force.",
        "A musician refined finger technique, phrasing, and pedal control on the piano."
      ],
      "output": "Summarize the subject with concrete domain details. data analytics healthcare consulting company dbdbdbc Webb, located in San Francisco. Can you provide a precise answer?\nAnswer:\n\nAssistantFraction of loans",
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
    "retrieval_strength__prefix_l2": 0.47129727854893727,
    "retrieval_strength__bad_decode_score": -0.5153013642952196,
    "prefix_l2__bad_decode_score": -0.8119768806119197
  },
  "rows": [
    {
      "prompt": "What improves piano technique and musical phrasing?",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 6,
          "score": -0.46566978096961975
        },
        {
          "mid": 4,
          "score": -0.5127263069152832
        },
        {
          "mid": 7,
          "score": -0.6141006350517273
        },
        {
          "mid": 5,
          "score": -0.6516022086143494
        },
        {
          "mid": 2,
          "score": -0.7098392248153687
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.7098392248153687,
      "prefix_l2_shift": 1036.929443359375,
      "prefix_js_divergence": 0.3171480894088745,
      "top1_with_prefix": {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 12.0,
        "prob": 0.05946128070354462
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.2697937758639455
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 6,
          "score": -0.496165931224823
        },
        {
          "mid": 4,
          "score": -0.5487112402915955
        },
        {
          "mid": 7,
          "score": -0.6517217755317688
        },
        {
          "mid": 5,
          "score": -0.684630811214447
        },
        {
          "mid": 2,
          "score": -0.742881178855896
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -2.3812297582626343,
      "prefix_l2_shift": 966.638916015625,
      "prefix_js_divergence": 0.27688848972320557,
      "top1_with_prefix": {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 13.0625,
        "prob": 0.06195751950144768
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.2790477816015482
    },
    {
      "prompt": "Describe what a student should focus on first.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 6,
          "score": -0.4532322883605957
        },
        {
          "mid": 4,
          "score": -0.4944304823875427
        },
        {
          "mid": 7,
          "score": -0.5890504717826843
        },
        {
          "mid": 5,
          "score": -0.635491132736206
        },
        {
          "mid": 2,
          "score": -0.6935202479362488
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.4532322883605957,
      "prefix_l2_shift": 847.4472045898438,
      "prefix_js_divergence": 0.2715299725532532,
      "top1_with_prefix": {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 13.5,
        "prob": 0.12762686610221863
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.35419856663793325
    },
    {
      "prompt": "Summarize the subject with concrete domain details.",
      "expected_label": null,
      "retrieved_scored": [
        {
          "mid": 6,
          "score": -0.4866049885749817
        },
        {
          "mid": 4,
          "score": -0.5152117013931274
        },
        {
          "mid": 7,
          "score": -0.5996290445327759
        },
        {
          "mid": 5,
          "score": -0.6369448900222778
        },
        {
          "mid": 2,
          "score": -0.692505419254303
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.4866049885749817,
      "prefix_l2_shift": 1529.544921875,
      "prefix_js_divergence": 0.2122175395488739,
      "top1_with_prefix": {
        "token_id": 220,
        "piece": " ",
        "norm": "",
        "logit": 12.625,
        "prob": 0.0805383250117302
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.2604658892378211
    },
    {
      "prompt": "Key piano ideas include",
      "expected_label": "music",
      "retrieved_scored": [
        {
          "mid": 6,
          "score": -0.4803183376789093
        },
        {
          "mid": 4,
          "score": -0.529201865196228
        },
        {
          "mid": 7,
          "score": -0.6350346207618713
        },
        {
          "mid": 5,
          "score": -0.6713391542434692
        },
        {
          "mid": 2,
          "score": -0.7322195768356323
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -0.7322195768356323,
      "prefix_l2_shift": 619.2012939453125,
      "prefix_js_divergence": 0.16191813349723816,
      "top1_with_prefix": {
        "token_id": 25,
        "piece": ":",
        "norm": "",
        "logit": 16.625,
        "prob": 0.11665172874927521
      },
      "top1_category_with_prefix": "punct",
      "topk_non_semantic_prob_mass": 0.4488343568518758
    },
    {
      "prompt": "Orbital motion depends on",
      "expected_label": "space",
      "retrieved_scored": [
        {
          "mid": 6,
          "score": -0.5201330184936523
        },
        {
          "mid": 4,
          "score": -0.5670214891433716
        },
        {
          "mid": 7,
          "score": -0.6661254167556763
        },
        {
          "mid": 5,
          "score": -0.698891282081604
        },
        {
          "mid": 3,
          "score": -0.7582919597625732
        }
      ],
      "retrieved_label_counts": {
        "space": 4,
        "music": 1
      },
      "retrieval_strength": -2.452171206474304,
      "prefix_l2_shift": 440.25799560546875,
      "prefix_js_divergence": 0.052666906267404556,
      "top1_with_prefix": {
        "token_id": 279,
        "piece": " the",
        "norm": "the",
        "logit": 20.875,
        "prob": 0.649355411529541
      },
      "top1_category_with_prefix": "functional",
      "topk_non_semantic_prob_mass": 0.7203306462615728
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
      "decoded_output": "What improves piano technique and musical phrasing? 1. **Understanding the Question**: The question asks about",
      "stage_counts": {
        "inject": 8,
        "retrieve": 4
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
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
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
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
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
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
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
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
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
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
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
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
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
          "step": 6,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
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
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.19865716993808746,
            "space": 0.1326199248433113
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "**:",
          "top1_category": "punct",
          "chosen_piece": "**:",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "inject"
        },
        {
          "step": 8,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "music": 2,
            "space": 3
          },
          "retrieved_score_sum": {
            "music": 0.18539325147867203,
            "space": 0.276333287358284
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 9,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "music": 2,
            "space": 3
          },
          "retrieved_score_sum": {
            "music": 0.18539325147867203,
            "space": 0.276333287358284
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " question",
          "top1_category": "semantic",
          "chosen_piece": " question",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "music": 2,
            "space": 3
          },
          "retrieved_score_sum": {
            "music": 0.18539325147867203,
            "space": 0.276333287358284
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " asks",
          "top1_category": "semantic",
          "chosen_piece": " asks",
          "chosen_category": "semantic",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 11,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "music": 2,
            "space": 3
          },
          "retrieved_score_sum": {
            "music": 0.18539325147867203,
            "space": 0.276333287358284
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " about",
          "top1_category": "functional",
          "chosen_piece": " about",
          "chosen_category": "functional",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        }
      ],
      "passed": false
    },
    {
      "prompt": "What explains satellites and orbital motion?",
      "expected_label": "space",
      "decoded_output": "What explains satellites and orbital motion? 为什么卫星和轨道运动？\nA. The gravitational force",
      "stage_counts": {
        "retrieve": 8,
        "inject": 3,
        "aligned": 1
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
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
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
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
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
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 2,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "卫星",
          "top1_category": "punct",
          "chosen_piece": "卫星",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 3,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "和",
          "top1_category": "punct",
          "chosen_piece": "和",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 4,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "轨道",
          "top1_category": "punct",
          "chosen_piece": "轨道",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 5,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "运动",
          "top1_category": "punct",
          "chosen_piece": "运动",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 6,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": "？\n",
          "top1_category": "punct",
          "chosen_piece": "？\n",
          "chosen_category": "punct",
          "chosen_label": null,
          "diagnosed_stage": "retrieve"
        },
        {
          "step": 7,
          "retrieved_majority_label": "music",
          "retrieved_label_counts": {
            "music": 3,
            "space": 2
          },
          "retrieved_score_sum": {
            "music": 0.18264970928430557,
            "space": 0.1216234415769577
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
          "step": 8,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.3073362410068512,
            "music": 0.20129191130399704
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
          "step": 9,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.3073362410068512,
            "music": 0.20129191130399704
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
        },
        {
          "step": 10,
          "retrieved_majority_label": "space",
          "retrieved_label_counts": {
            "space": 3,
            "music": 2
          },
          "retrieved_score_sum": {
            "space": 0.3073362410068512,
            "music": 0.20129191130399704
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0.04752008616924286
          },
          "top1_piece": " gravitational",
          "top1_category": "semantic",
          "chosen_piece": " gravitational",
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
            "space": 0.3073362410068512,
            "music": 0.20129191130399704
          },
          "logits_label_mass": {
            "music": 0,
            "space": 0
          },
          "top1_piece": " force",
          "top1_category": "semantic",
          "chosen_piece": " force",
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
    "The pianist-ios版是一款什么类型的软件？\nA. 游戏<\\/p> <-{amic",
    "Quantum systems系列产品有哪些产品系列？\nABCDE ABCD 解析：华为云量子计算品牌“澜",
    "The rainforest公司将推出自家操作系统后，以下哪个平台提供每日、每周或每月发布净水行业"
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
  "output_a": "The pianist的官方预告片！视频中可以看到Dragonboat Festival乐队主唱Peter KnighterPhoenix",
  "output_b": "The pianist的官方预告片！视频中可以看到Dragonboat Festival乐队主唱Peter KnighterPhoenix",
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
    "The pianist played a piano piece called \"The Beatles Theme from A Hard Day's Night,\" which is",
    "The telescope is on Tommy ________ desk.________ he be Tom？ A：a； could B",
    "The trader bought a pair of trainers for Kate last year at 0.65 euros and sold",
    "The child _____ in the classroom suddenly fell off his _____. Fill blanks: graduation______. A falls"
  ],
  "exact_same": false,
  "prefix_only": false,
  "too_short": false,
  "error": null
}
```