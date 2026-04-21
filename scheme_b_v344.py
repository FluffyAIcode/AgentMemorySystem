#!/usr/bin/env python3
"""
嵌入级方案B · v3.44
═══════════════════════════════════════════════════════════════════════════
针对 v3.43 遗留 FAIL(4.7/4.11/4.13/4.16/4.19/4.23/4.24) 的收敛修复:

[A] MemoryContextEncoder 注意力池化
    Q = learnable Parameter(d_ctx)
    K,V = Linear(d_LLM, d_ctx)(hidden_states)
    + residual shortcut via orthogonal proj_wte(wte_centroid)
    write() 路径传入 content-token hidden states + mask

[B] 尾槽 slot_1 残差主导分解
    slot_1 = α × rare_keyword_residual + β × LN(head_output)
    β = Parameter(init=0.3),可学习但受 L2 正则约束
    新增 slot_residual_alignment_loss: relu(0.5 - cos(slot_1, residual))

[C] inter-domain margin + 检索 crowding
    Trainer.inter_domain_margin_loss: KMeans on semantic_emb → 弱标签
      same-domain cos ≥ 0.6, cross-domain cos ≤ 0.3, margin = 0.3
    DirectionTree 存储 _mem_cluster_id(每次写入后 re-cluster)
    retrieve_multi rerank 扣 λ × inter_domain_crowding

[D] save/load 确定性
    sorted(state['store'].keys()) 写入
    sorted(set(...)) 替 list(set(...)) on content_token_ids union
    _refresh_rare_keyword_indices 按 mid 排序
    PrefixAligner._calibrated flag 保护重复校准
    每个 MemEntry 写入 SHA256 指纹,load 校验

[E] content_bias top-1 专属 + top-k 兜底
    b_bias = 0.7 × build(top1, floor=0.5) + 0.3 × build(rest, floor=0.2)

[F] mixture gate circuit breaker
    generate 步记录 -log P(chosen),baseline = 前 3 步均值
    连续 3 步超 1.5 × baseline → 临时 ceiling = 0.3(hysteresis 5 步)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math, time, os, hashlib
from typing import Dict, List, Tuple, Optional, NamedTuple, FrozenSet, Set
from dataclasses import dataclass, field

@dataclass
class Cfg:
    llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_dtype: str = "bf16"
    use_chat_template_for_gen: bool = False
    d_LLM: int = 1536
    vocab_size: int = 151936
    d_M: int = 8; d_F: int = 32
    L_mem: int = 8; n_heads_fiber: int = 4
    bridge_heads: int = 4; bridge_layers: int = 2
    n_geo_pts: int = 8; geo_max_steps: int = 80
    geo_tol: float = 1e-5; geo_lr: float = 0.02
    tree_K: int = 8; tree_max_leaf: int = 20
    tau: float = 0.07
    write_gate_threshold: float = 0.4
    retention_gc_threshold: float = 0.15
    consol_dist: float = 0.3; consol_conflict_ratio: float = 0.5
    retrieval_topk: int = 8; retrieval_beam: int = 5
    retrieval_interval: int = 8
    retrieval_recall_factor: float = 2.0
    flat_scan_threshold_factor: int = 3
    gen_top_p: float = 0.9; gen_temp: float = 0.8
    norm_correction_interval: int = 4
    write_update_alpha: float = 0.3
    dir_diversity_tau: float = 0.5
    bypass_init_gate_bias: float = 0.5
    degen_min_tokens: int = 5; degen_repeat_penalty: float = 1.4
    degen_max_consec_punct: int = 2
    probe_contrastive_tau: float = 0.1
    contrast_tau: float = 0.5
    prefix_init_scale: float = 0.5
    degen_early_punct_penalty: float = 6.0
    degen_early_newline_penalty: float = 6.0
    early_content_steps: int = 5
    use_early_content_starter_hard_mask: bool = True
    early_starter_hard_mask_steps: int = 3
    use_fwd_path_hard_mask: bool = True
    fwd_path_hard_mask_value: float = -1e9
    use_no_repeat_bigram: bool = True
    no_repeat_bigram_penalty: float = 5.0
    use_fwd_path_content_bias: bool = True
    fwd_path_bias_dampen: float = 1.0
    apply_content_bias_symmetric_cfg: bool = True
    shape_step_applies_content_bias: bool = False
    shape_step_applies_suppression_bias: bool = False
    guidance_min_memory_weight: float = 1e-6
    content_bias_scale: float = 6.0
    use_adaptive_content_bias_scale: bool = True
    content_bias_std_multiplier: float = 1.5
    content_bias_decay: float = 0.02
    content_bias_floor: float = 0.5
    generated_token_decay: float = 0.2
    content_repeat_penalty: float = 2.5
    content_repeat_exponent: float = 1.0
    cyclic_content_window: int = 15
    cyclic_content_max_count: int = 5
    content_bias_relevance_floor: float = 0.30
    content_bias_concentration: float = 1.5
    # [E] top-1 exclusive content bias
    # [v3.46] Disabled.  v3.44-rewrite enabled this (=True) to concentrate
    # content_bias onto top-1 memory's keywords.  Under v3.44-rewrite this
    # over-concentrated the bias to ~+22 logit on ~8 tokens, which then
    # races against content_repeat_penalty=2.5 -- the penalty only wins
    # at k>=10 while cyclic_content_max_count=5 hard-masks at k=5, so
    # 4.7 / 4.8 / 4.21 collapsed into repetition.  [C] cluster-crowding
    # (independent Cfg path) already delivered 4.16; [E] was not its
    # cause.  Reverting to the aggregated (pre-v3.44-rewrite) path.
    # v3.48 baseline under this config had 4.7 / 4.8 / 4.21 all PASS.
    use_top1_exclusive_content_bias: bool = False
    top1_content_bias_weight: float = 0.7
    rest_content_bias_weight: float = 0.3
    top1_relevance_floor: float = 0.5
    rest_relevance_floor: float = 0.2
    retrieval_use_expanded_ids: bool = True
    use_memory_guided_suppression: bool = True
    suppression_bias_scale: float = 4.0
    suppression_std_multiplier: float = 1.0
    suppression_decay: float = 0.03
    suppression_floor: float = 0.3
    use_mean_centered_scoring: bool = True
    mc_keep_margin: float = 0.0
    mc_min_keep: int = 3
    mc_require_min_candidates: int = 2
    use_hungarian_fwd: bool = True
    hungarian_max_n: int = 24
    use_cfg_decoding: bool = True
    use_contrastive_memory_cfg: bool = True
    cfg_scale: float = 3.5
    cfg_decay_steps: int = 0
    use_content_semantic_tail: bool = True
    content_tail_slots: int = 2
    tail_head_hidden: int = 1024
    tail_head_tied_extra: bool = True
    tail_head_zero_init_tied: bool = True
    # [B] tail slot_1 residual-dominant decomposition
    # [v3.45] Off.  In v3.44-rewrite (true=on) combine_with_residual produced
    # slot_1 = alpha*residual (L2=1.07) + beta*LN(head_out) (L2=11.76) and
    # LN(head_out) direction dominated.  On a fresh init with zero-init
    # slot_heads[1], LN(0) reduces to LayerNorm gamma direction (uniform),
    # which is far from every rare-keyword WTE direction -> 4.23 median rank
    # went to 1402 (v3.48 baseline 1089).  Reverted to naive additive path:
    #   slot_1 = tail_head(fiber) + alpha * residual
    # which in fresh init = alpha * residual (because slot_heads[1] is zero)
    # and has slot direction = rare_keyword_centroid direction (by construction).
    tail_slot_residual_dominant: bool = False
    tail_slot_beta_init: float = 0.3
    tail_slot_cos_alignment_floor: float = 0.5
    ret_centroid_weight: float = 0.30
    ret_sem_weight: float = 0.10
    ret_bidi_min_weight: float = 0.25
    ret_forward_maxsim_weight: float = 0.35
    ret_dir_weight: float = 0.00
    reranker_clip: float = 0.2
    fwd_coherence_ratio: float = 0.55
    score_keep_ratio: float = 0.80
    retrieval_weight_temperature: float = 0.05
    consol_maxsim_min: float = 0.40
    gate_sem_ratio: float = 0.65
    gate_bidi_ratio: float = 0.70
    gate_sem_floor: float = 0.10
    gate_bidi_floor: float = 0.10
    gate_bidi_hard_min: float = 0.12
    gate_sem_weight: float = 0.50
    gate_bidi_weight: float = 0.50
    bidi_absolute_gap: float = 0.15
    use_tfidf_weighting: bool = True
    tfidf_smoothing: float = 1.0
    use_idf_retrieval: bool = True
    idf_floor: float = 0.1
    use_idf_centroid: bool = True
    use_word_starter_filter: bool = True
    bpe_echo_window: int = 3
    bpe_echo_penalty: float = 3.0
    post_starter_nonstarter_penalty: float = 2.0
    use_strict_content_starter: bool = True
    strict_starter_min_decoded_len: int = 5
    use_upstream_semantic_gate: bool = True
    upstream_gate_fwd_idf_floor: float = 0.12
    upstream_gate_sem_floor: float = 0.15
    upstream_gate_min_keep: int = 1
    use_strict_content_overlap_gate: bool = True
    strict_overlap_sim_threshold: float = 0.32
    strict_overlap_min_matches: int = 1
    strict_overlap_min_keep: int = 1
    retrieval_min_keep_for_rerank: int = 5
    use_ngram_repeat_block: bool = True
    ngram_repeat_penalty: float = 10.0
    ngram_repeat_max_n: int = 4
    use_cyclic_content_hard_mask: bool = True
    use_content_gated_newline: bool = True
    min_content_tokens_before_newline: int = 8
    late_newline_penalty: float = 20.0
    use_newline_hard_gate: bool = True
    newline_hard_gate_min_step: int = 12
    newline_hard_gate_min_content: int = 6
    use_eos_hard_mask: bool = True
    eos_hard_mask_steps: int = 10
    use_filler_direction_projection: bool = True
    filler_projection_last_slots: int = 2
    use_slot_norm_renormalize: bool = True
    use_prefix_norm_clamp: bool = False
    prefix_norm_clamp_ratio: float = 1.0
    semantic_boost_scale: float = 0.5
    semantic_boost_decay: float = 0.06
    semantic_boost_floor: float = 0.2
    semantic_align_temp: float = 0.3
    wte_neighbor_k: int = 5
    wte_neighbor_threshold: float = 0.5
    wte_neighbor_max_vocab: int = 60000
    stopwords_override: Optional[FrozenSet[str]] = None
    filler_words_override: Optional[FrozenSet[str]] = None
    stopwords_extra: FrozenSet[str] = field(default_factory=frozenset)
    filler_words_extra: FrozenSet[str] = field(default_factory=frozenset)
    dedup_filler_from_stop: bool = False
    use_idf_content_bias: bool = True
    idf_bias_max_boost: float = 3.0
    use_tree_semantic_rerank: bool = True
    tree_rerank_dir_weight: float = 0.2
    tree_rerank_centroid_weight: float = 0.4
    tree_rerank_forward_weight: float = 0.4
    use_functional_suppression: bool = True
    functional_suppression_margin: float = 2.0
    use_keyword_tail_slot: bool = True
    keyword_tail_top_k: int = 8
    keyword_tail_weight: float = 1.0
    use_context_descriptor: bool = True
    context_slot_enabled: bool = True
    use_content_bias_history_decay: bool = True
    content_bias_history_decay_rate: float = 0.5
    content_bias_history_floor: float = 0.1
    use_degeneration_detector: bool = True
    degen_detector_window: int = 8
    degen_detector_unique_ratio: float = 0.4
    degen_detector_bias_dampen: float = 0.3
    use_memory_context_encoder: bool = True
    d_ctx: int = 128
    context_encoder_hidden: int = 256
    context_encoder_hybrid: bool = True
    context_hybrid_hidden_weight: float = 0.8
    # [A] attention pool ctx encoder
    context_encoder_use_attention_pool: bool = True
    context_encoder_residual_weight: float = 0.3
    context_encoder_attn_dropout: float = 0.0
    use_decode_functional_suppression: bool = True
    decode_fs_margin: float = 1.5
    decode_fs_scale: float = 4.0
    decode_fs_decay: float = 0.04
    decode_fs_floor: float = 0.3
    decode_fs_topk_eval: int = 20
    use_fwd_function_suppression: bool = True
    fwd_function_suppression_scale: float = 5.0
    fwd_function_suppression_decay: float = 0.04
    fwd_function_suppression_floor: float = 0.3
    fwd_function_suppression_apply_dampen: bool = False
    use_wte_residual_tail: bool = True
    wte_residual_alpha: float = 1.5
    wte_residual_post_aligner: bool = True
    wte_residual_centered: bool = True
    wte_residual_exclude_generated: bool = True
    scale_tail_with_L_mem: bool = True
    tail_L_mem_base: int = 8
    tail_L_mem_step: int = 2
    ctx_L_mem_threshold: int = 12
    use_mixture_decoding: bool = False
    mixture_gate_floor: float = 0.0
    mixture_gate_ceiling: float = 0.7
    mixture_gate_hidden: int = 256
    # [F] circuit breaker
    use_circuit_breaker: bool = True
    circuit_breaker_baseline_steps: int = 3
    circuit_breaker_threshold_ratio: float = 1.5
    circuit_breaker_consec_steps: int = 3
    circuit_breaker_hysteresis: int = 5
    circuit_breaker_clamp_ceiling: float = 0.3
    # [C] inter-domain margin
    use_inter_domain_margin: bool = True
    inter_domain_same_cos_target: float = 0.6
    inter_domain_cross_cos_target: float = 0.3
    inter_domain_margin: float = 0.3
    inter_domain_kmeans_k: int = 2
    inter_domain_kmeans_iters: int = 20
    retrieval_crowding_lambda: float = 0.15
    mem_recluster_every_writes: int = 4
    context_encoder_source: str = "wte_strict_starter"
    context_encoder_fp32: bool = True
    warmup_steps_ctx_sep: int = 10
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'recon': 1.0, 'semantic_alignment': 3.0,
        'encoder_throughput': 1.5, 'contrast': 0.02,
        'holonomy': 0.005, 'write_policy': 0.1,
        'semantic_probe': 0.3, 'dir_diversity': 0.1,
        'reranker_ranking': 0.2, 'vocab_anchor': 0.2,
        'tail_semantic_anchor': 0.5,
        'functional_suppression': 0.4,
        'context_separation': 0.3,
        'slot_residual_alignment': 0.0,  # [B] v3.45 disabled (see Cfg)
        'inter_domain_margin': 0.2})     # [C]
    warmup_steps_probe: int = 5; warmup_steps_dd: int = 5
    warmup_steps_rr: int = 5; warmup_steps_va: int = 5
    warmup_steps_sa: int = 0
    warmup_steps_tsa: int = 0
    warmup_steps_fs: int = 3
    warmup_steps_sra: int = 3   # [B]
    warmup_steps_idm: int = 5   # [C]
    uw_clamp_lo: float = -4.0; uw_clamp_hi: float = 4.0
    vocab_anchor_topk: int = 5; content_min_len: int = 3
    refresh_memories_every: int = 1
    content_inject_scale: float = 1.0

    def effective_tail_slots(self) -> int:
        base = self.content_tail_slots
        if self.scale_tail_with_L_mem and self.tail_L_mem_step > 0:
            extra = max(0, (self.L_mem - self.tail_L_mem_base) // self.tail_L_mem_step)
            return base + extra
        return base

    def effective_ctx_slots(self) -> int:
        if not (self.use_context_descriptor and self.context_slot_enabled):
            return 0
        base = 1
        if self.scale_tail_with_L_mem and self.L_mem >= self.ctx_L_mem_threshold:
            base = 2
        return base

    def __post_init__(self):
        assert self.d_F % self.n_heads_fiber == 0
        assert self.n_geo_pts >= 2 and 0 < self.tau < 1
        w_sum = (self.ret_centroid_weight + self.ret_sem_weight
                 + self.ret_bidi_min_weight + self.ret_forward_maxsim_weight
                 + self.ret_dir_weight)
        assert 0.8 < w_sum < 1.2
        assert self.cfg_scale >= 0
        assert self.content_tail_slots >= 0
        assert self.llm_dtype in ("bf16", "fp16", "fp32")
        assert 0.0 <= self.fwd_path_bias_dampen <= 1.0
        assert self.guidance_min_memory_weight > 0
        assert self.idf_bias_max_boost >= 1.0
        rr = (self.tree_rerank_dir_weight + self.tree_rerank_centroid_weight
              + self.tree_rerank_forward_weight)
        assert 0.8 < rr < 1.2
        tail_eff = self.effective_tail_slots()
        ctx_eff = self.effective_ctx_slots()
        used = tail_eff + ctx_eff
        assert used < self.L_mem, f"tail({tail_eff})+ctx({ctx_eff})={used} must be < L_mem={self.L_mem}"
        assert self.keyword_tail_top_k >= 1
        assert 0.0 < self.content_bias_history_decay_rate <= 1.0
        assert 0.0 < self.content_bias_history_floor <= 1.0
        assert self.degen_detector_window >= 2
        assert 0.0 < self.degen_detector_unique_ratio <= 1.0
        assert 0.0 <= self.degen_detector_bias_dampen <= 1.0
        assert self.d_ctx >= 16
        assert 0.0 <= self.wte_residual_alpha <= 3.0
        assert 0.0 <= self.mixture_gate_floor <= self.mixture_gate_ceiling <= 1.0
        assert self.retrieval_min_keep_for_rerank >= 1
        assert self.context_encoder_source in ("wte_strict_starter", "hidden_mean")
        assert self.content_bias_relevance_floor >= 0.0
        assert self.cyclic_content_max_count >= 1
        assert 0.0 <= self.context_hybrid_hidden_weight <= 2.0
        assert 0.0 <= self.top1_content_bias_weight <= 1.0
        assert abs(self.top1_content_bias_weight + self.rest_content_bias_weight - 1.0) < 1e-6
        assert 0.0 <= self.top1_relevance_floor <= 1.0
        assert 0.0 <= self.rest_relevance_floor <= 1.0
        assert self.inter_domain_kmeans_k >= 2
        assert 0.0 < self.inter_domain_same_cos_target <= 1.0
        assert 0.0 <= self.inter_domain_cross_cos_target < self.inter_domain_same_cos_target
        assert 0.0 < self.circuit_breaker_clamp_ceiling <= self.mixture_gate_ceiling or not self.use_mixture_decoding
        assert 0.0 <= self.tail_slot_cos_alignment_floor <= 1.0
        assert 0.0 <= self.tail_slot_beta_init <= 2.0

def _dev(ref): return dict(device=ref.device, dtype=ref.dtype)
def _resolve_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]

def _sorted_set(iterable):
    """[D] 确定性版本的 list(set(x)),适用于整数 token id。"""
    return sorted(set(int(x) for x in iterable))

def _simple_kmeans(x, k, n_iter=20, seed=0):
    """[C] 确定性 KMeans(torch-only),用于弱域标签。"""
    N, D = x.shape
    if N <= k:
        return torch.arange(N, device=x.device), x.clone()
    g = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(N, generator=g).to(x.device)
    centers = x[perm[:k]].clone()
    assign_prev = None
    for _ in range(n_iter):
        d = torch.cdist(x, centers)
        assign = d.argmin(dim=1)
        if assign_prev is not None and torch.equal(assign, assign_prev):
            break
        new_centers = centers.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = x[mask].mean(0)
        centers = new_centers
        assign_prev = assign
    return assign, centers

@dataclass
class DecodeState:
    generated_ids: List[int] = field(default_factory=list)
    generated_content_counts: Dict[int, int] = field(default_factory=dict)
    content_history: List[Tuple[int, int]] = field(default_factory=list)
    recent_starters: List[Tuple[int, int]] = field(default_factory=list)
    # [F] circuit breaker signals
    token_nll_history: List[float] = field(default_factory=list)

    def update(self, nxt_id, step, cc, bpe_echo_window, cyclic_content_window, nll=None):
        self.generated_ids.append(nxt_id)
        if cc is not None and nxt_id in cc.content_ids:
            self.generated_content_counts[nxt_id] = self.generated_content_counts.get(nxt_id, 0) + 1
            self.content_history.append((step, nxt_id))
            if nxt_id in cc.word_starter_ids:
                self.recent_starters.append((nxt_id, step))
        self.recent_starters = [(t, s) for (t, s) in self.recent_starters
                                if (step - s) < bpe_echo_window]
        if len(self.content_history) > 2 * cyclic_content_window:
            self.content_history = self.content_history[-cyclic_content_window:]
        if nll is not None:
            self.token_nll_history.append(float(nll))

class CircuitBreaker:
    """[F] 监控生成过程 -log P,高于 baseline × threshold → 临时压低 mixture ceiling。"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.baseline = None
        self.active = False
        self.hysteresis_left = 0
    def set_baseline_from(self, nll_list: List[float]):
        if len(nll_list) >= self.cfg.circuit_breaker_baseline_steps:
            base_slice = nll_list[:self.cfg.circuit_breaker_baseline_steps]
            self.baseline = sum(base_slice) / len(base_slice)
    def update(self, nll_list: List[float]):
        if self.baseline is None:
            self.set_baseline_from(nll_list); return
        K = self.cfg.circuit_breaker_consec_steps
        if len(nll_list) < K: return
        tail = nll_list[-K:]
        thresh = self.baseline * self.cfg.circuit_breaker_threshold_ratio
        if all(x > thresh for x in tail):
            self.active = True
            self.hysteresis_left = self.cfg.circuit_breaker_hysteresis
        elif self.active:
            self.hysteresis_left -= 1
            if self.hysteresis_left <= 0:
                self.active = False
    def effective_ceiling(self, default_ceiling: float) -> float:
        if self.active:
            return min(default_ceiling, self.cfg.circuit_breaker_clamp_ceiling)
        return default_ceiling

class LLMBackbone(nn.Module):
    def __init__(self, name, dtype_name="bf16"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name; self._dtype = _resolve_dtype(dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else: raise ValueError(f"Tokenizer for {name} has no pad/eos")
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=self._dtype, trust_remote_code=True)
        for p in self.model.parameters(): p.requires_grad_(False)
        self.model.eval()
        cfg = self.model.config
        self.d_model = cfg.hidden_size; self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.has_chat_template = getattr(self.tokenizer, 'chat_template', None) is not None
        with torch.no_grad():
            self._wte_fp32 = self.model.get_input_embeddings().weight.detach().float().clone()

    def input_embedding_weight(self): return self._wte_fp32
    def embed_tokens(self, ids): return self.model.get_input_embeddings()(ids)
    @property
    def device(self): return next(self.model.parameters()).device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for arg in args:
            if isinstance(arg, torch.device) or (isinstance(arg, str) and arg in ("cuda","cpu")):
                self._wte_fp32 = self._wte_fp32.to(arg)
        if 'device' in kwargs: self._wte_fp32 = self._wte_fp32.to(kwargs['device'])
        return self

    def forward(self, ids, attention_mask, prefix=None):
        te = self.embed_tokens(ids)
        if prefix is not None:
            prefix_cast = prefix.to(te.dtype)
            inputs_embeds = torch.cat([prefix_cast, te], dim=1)
            B, P = prefix_cast.shape[:2]
            pm = torch.ones(B, P, device=ids.device, dtype=attention_mask.dtype)
            ext_mask = torch.cat([pm, attention_mask], dim=1); pl = P
        else:
            inputs_embeds = te; ext_mask = attention_mask; pl = 0
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=ext_mask,
                         output_hidden_states=True, use_cache=False, return_dict=True)
        hs_list = [h.float() for h in out.hidden_states]
        logits = out.logits.float()
        return {'logits': logits, 'hs': hs_list, 'pl': pl, 'mask': ext_mask}

    def build_chat_text(self, user_text):
        if not self.has_chat_template: return user_text
        msgs = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

def hungarian_max_assignment(sim):
    device = sim.device; n_rows, n_cols = sim.shape
    if n_rows == 0 or n_cols == 0:
        return torch.empty(0, 2, dtype=torch.long, device=device), 0.0
    transposed = False
    if n_rows > n_cols:
        sim = sim.T; n_rows, n_cols = n_cols, n_rows; transposed = True
    import numpy as np
    cost = (-sim).detach().cpu().numpy().astype('float64')
    INF = float('inf')
    u = np.zeros(n_rows + 1); v = np.zeros(n_cols + 1)
    p = np.zeros(n_cols + 1, dtype=int); way = np.zeros(n_cols + 1, dtype=int)
    for i in range(1, n_rows + 1):
        p[0] = i; j0 = 0
        minv = np.full(n_cols + 1, INF); used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = INF; j1 = -1
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]: minv[j] = cur; way[j] = j0
                    if minv[j] < delta: delta = minv[j]; j1 = j
            for j in range(n_cols + 1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    pairs = []
    for j in range(1, n_cols + 1):
        i = p[j]
        if i > 0 and i <= n_rows:
            if transposed: pairs.append((j - 1, i - 1))
            else: pairs.append((i - 1, j - 1))
    if not pairs:
        return torch.empty(0,2,dtype=torch.long,device=device), 0.0
    pairs_t = torch.tensor(pairs, dtype=torch.long, device=device)
    total = float(sim[pairs_t[:,0], pairs_t[:,1]].sum().item()) if not transposed \
        else float(sim[pairs_t[:,1], pairs_t[:,0]].sum().item())
    return pairs_t, total

class RiemannianMetric(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d
        n_tri = d*(d+1)//2
        self.net = nn.Sequential(nn.Linear(d,4*d), nn.SiLU(),
                                 nn.Linear(4*d,4*d), nn.SiLU(),
                                 nn.Linear(4*d, n_tri))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.normal_(self.net[-1].weight, std=0.02); nn.init.zeros_(self.net[-1].bias)
        r,c=[],[]
        for i in range(d):
            for j in range(i+1): r.append(i); c.append(j)
        self.register_buffer('_r', torch.tensor(r)); self.register_buffer('_c', torch.tensor(c))
    def forward(self, x):
        B=x.shape[0]; d=self.d; v=self.net(x)
        L=x.new_zeros(B,d,d); L[:,self._r,self._c]=v
        di=torch.arange(d,device=x.device); L[:,di,di]=F.softplus(L[:,di,di])+1e-3
        return L@L.transpose(1,2)
    def christoffel(self, x):
        d=self.d; B=x.shape[0]
        xv=x.detach().clone().requires_grad_(True)
        g=self.forward(xv); g_inv=torch.linalg.inv(g.detach())
        dg=x.new_zeros(B,d,d,d)
        for i in range(d):
            for j in range(i,d):
                gr=torch.autograd.grad(g[:,i,j].sum(),xv,retain_graph=True)[0]
                dg[:,i,j,:]=gr
                if i!=j: dg[:,j,i,:]=gr
        term=dg.permute(0,3,1,2)+dg.permute(0,1,3,2)-dg
        return (0.5*torch.einsum('bkl,bijl->bkij',g_inv,term)).detach()
    def midpoint_approx_distance(self, x, y):
        diff=x-y; mid=(x+y)/2
        with torch.no_grad(): g=self.forward(mid)
        return torch.einsum('bi,bij,bj->b',diff,g,diff).clamp(min=0).sqrt()

class GeodesicResult(NamedTuple):
    path: torch.Tensor; energy: float; converged: bool; iterations: int

class GeodesicSolver:
    def __init__(self, metric, cfg): self.metric=metric; self.cfg=cfg
    def solve(self, xs, xe):
        B,d=xs.shape; N=self.cfg.n_geo_pts; dev=xs.device
        t=torch.linspace(0,1,N+2,device=dev)[1:-1]
        ps={n:p.requires_grad for n,p in self.metric.named_parameters()}
        for p in self.metric.parameters(): p.requires_grad_(False)
        with torch.enable_grad():
            interior=(xs.detach().unsqueeze(1)*(1-t[None,:,None])
                      +xe.detach().unsqueeze(1)*t[None,:,None]).detach().clone().requires_grad_(True)
            opt=torch.optim.Adam([interior],lr=self.cfg.geo_lr)
            prev=float('inf'); converged=False; iters=0; cur=prev
            for it in range(self.cfg.geo_max_steps):
                opt.zero_grad()
                path=torch.cat([xs.detach().unsqueeze(1),interior,xe.detach().unsqueeze(1)],1)
                dx=path[:,1:]-path[:,:-1]; mid=(path[:,1:]+path[:,:-1])/2
                g=self.metric(mid.reshape(-1,d)).reshape(B,N+1,d,d)
                energy=torch.einsum('bni,bnij,bnj->',dx,g,dx)
                if energy.item()!=energy.item():
                    t_full=torch.linspace(0,1,N+2,device=dev).view(1,-1,1)
                    lin=xs.unsqueeze(1)*(1-t_full)+xe.unsqueeze(1)*t_full
                    for n,p in self.metric.named_parameters(): p.requires_grad_(ps[n])
                    return GeodesicResult(lin,float('inf'),False,it)
                energy.backward(); opt.step(); iters=it+1; cur=energy.item()
                if abs(prev-cur)/(abs(prev)+1e-10)<self.cfg.geo_tol:
                    converged=True; break
                prev=cur
        for n,p in self.metric.named_parameters(): p.requires_grad_(ps[n])
        final=torch.cat([xs.unsqueeze(1),interior.detach(),xe.unsqueeze(1)],1)
        return GeodesicResult(final,cur,converged,iters)

class FiberConnection(nn.Module):
    def __init__(self, d_M, d_F, metric, grad_coupling=True):
        super().__init__(); self.d_F=d_F; self.metric=metric; self.grad_coupling=grad_coupling
        d_g=d_M*(d_M+1)//2
        self.net=nn.Sequential(nn.Linear(2*d_M+d_g,4*d_F),nn.SiLU(),
                               nn.Linear(4*d_F,4*d_F),nn.SiLU(),nn.Linear(4*d_F,d_F*d_F))
        nn.init.normal_(self.net[-1].weight,std=0.01); nn.init.normal_(self.net[-1].bias,std=0.01)
    def forward(self, x, v):
        g=self.metric(x); d=g.shape[-1]; idx=torch.triu_indices(d,d,device=x.device)
        gf=g[:,idx[0],idx[1]]
        if not self.grad_coupling: gf=gf.detach()
        raw=self.net(torch.cat([x,v,gf],-1)).reshape(-1,self.d_F,self.d_F)
        return (raw-raw.transpose(1,2))/2

class FiberTransporter(nn.Module):
    def __init__(self, conn, cfg): super().__init__(); self.conn=conn; self.cfg=cfg
    def forward(self, fiber, path):
        f=fiber; n0=fiber.norm(dim=-1,keepdim=True).clamp(min=1e-8)
        nci=self.cfg.norm_correction_interval
        for k in range(path.shape[1]-1):
            p0,p1=path[:,k],path[:,k+1]; v=p1-p0; mid=(p0+p1)/2
            k1=-(self.conn(p0,v)@f.unsqueeze(-1)).squeeze(-1)
            k2=-(self.conn(mid,v)@(f+.5*k1).unsqueeze(-1)).squeeze(-1)
            k3=-(self.conn(mid,v)@(f+.5*k2).unsqueeze(-1)).squeeze(-1)
            k4=-(self.conn(p1,v)@(f+k3).unsqueeze(-1)).squeeze(-1)
            f=f+(k1+2*k2+2*k3+k4)/6
            if (k+1)%nci==0: f=f*(n0/f.norm(dim=-1,keepdim=True).clamp(min=1e-8))
        return f

class CtxEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f1=nn.Linear(c.d_LLM,4*c.d_M); self.f2=nn.Linear(4*c.d_M,4*c.d_M)
        self.f3=nn.Linear(4*c.d_M,c.d_M); self.skip=nn.Linear(c.d_LLM,c.d_M)
        self.n1=nn.LayerNorm(4*c.d_M); self.n2=nn.LayerNorm(4*c.d_M); self.no=nn.LayerNorm(c.d_M)
    def forward(self, h):
        x=F.silu(self.n1(self.f1(h))); x=F.silu(self.n2(self.f2(x)))
        return self.no(self.f3(x)+self.skip(h))

class FibEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(c.d_LLM+c.d_M,4*c.d_F),nn.SiLU(),nn.LayerNorm(4*c.d_F),
                               nn.Linear(4*c.d_F,4*c.d_F),nn.SiLU(),nn.LayerNorm(4*c.d_F),
                               nn.Linear(4*c.d_F,c.d_F))
        self.sg=nn.Sequential(nn.Linear(1,c.d_F),nn.SiLU(),nn.Linear(c.d_F,c.d_F),nn.Sigmoid())
    def forward(self, h, x, surprise=None):
        f=self.enc(torch.cat([h,x],-1))
        if surprise is not None:
            s=surprise.view(-1,1) if surprise.dim()>=1 else surprise.unsqueeze(0).unsqueeze(0)
            if s.shape[0]!=f.shape[0]: s=s.expand(f.shape[0],-1)
            f=f*self.sg(s)
        return f

class DirectionPredictor(nn.Module):
    def __init__(self, d_M, d_F):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_M+d_F,4*d_M),nn.SiLU(),
                               nn.LayerNorm(4*d_M),nn.Linear(4*d_M,d_M))
    def forward(self, x, f):
        return F.normalize(self.net(torch.cat([x,f],-1)),dim=-1,eps=1e-8)

class EmptyStateNet(nn.Module):
    def __init__(self, d_M, d_F):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_M+d_F,2*d_F),nn.SiLU(),nn.LayerNorm(2*d_F),
                               nn.Linear(2*d_F,d_F))
    def forward(self, xq, fq): return self.net(torch.cat([xq,fq],-1))

class WriteGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(c.d_LLM+1,c.d_LLM//4),nn.SiLU(),nn.Linear(c.d_LLM//4,1))
    def forward(self, h, surprise):
        s=surprise.view(-1,1) if surprise.dim()>=1 else surprise.unsqueeze(0).unsqueeze(0)
        if s.shape[0]!=h.shape[0]: s=s[:h.shape[0]]
        return torch.sigmoid(self.net(torch.cat([h,s],-1)).squeeze(-1))

class RetentionScorer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(c.d_M+c.d_F+3,64),nn.SiLU(),
                               nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, base, fiber, surprise, dt, cnt):
        return self.net(torch.cat([base,fiber,
            surprise.unsqueeze(-1) if surprise.dim()==1 else surprise,
            dt.unsqueeze(-1) if dt.dim()==1 else dt,
            cnt.float().unsqueeze(-1) if cnt.dim()==1 else cnt.float()],-1)).squeeze(-1)

class RetrievalReranker(nn.Module):
    def __init__(self, d_M, d_F, clip=0.2):
        super().__init__(); self.clip=clip
        inp=2*d_M+2*d_F+1
        self.net=nn.Sequential(nn.Linear(inp,128),nn.SiLU(),nn.LayerNorm(128),
                               nn.Linear(128,64),nn.SiLU(),nn.LayerNorm(64),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, xq, fq, xc, fc, dir_sim):
        B,C=xc.shape[:2]
        xq_e=xq.unsqueeze(1).expand(-1,C,-1); fq_e=fq.unsqueeze(1).expand(-1,C,-1)
        inp=torch.cat([xq_e,fq_e,xc,fc,dir_sim.unsqueeze(-1)],-1)
        correction=self.net(inp).squeeze(-1)
        return dir_sim + correction.clamp(-self.clip, self.clip)

class ContentBypass(nn.Module):
    def __init__(self, d_F, d_LLM, gate_bias=0.5):
        super().__init__()
        self.proj=nn.Sequential(
            nn.Linear(d_F,2*d_LLM),nn.SiLU(),nn.LayerNorm(2*d_LLM),
            nn.Linear(2*d_LLM,d_LLM),nn.LayerNorm(d_LLM))
        self.gate_net=nn.Sequential(nn.Linear(d_F+d_LLM,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.constant_(self.gate_net[-1].bias,gate_bias)
        nn.init.normal_(self.proj[3].weight,std=0.02); nn.init.zeros_(self.proj[3].bias)
        self._last_gate=None
    def forward(self, fiber_summary, qformer_context):
        projected=self.proj(fiber_summary)
        gate_in=torch.cat([fiber_summary,qformer_context],-1)
        g=torch.sigmoid(self.gate_net(gate_in)); self._last_gate=g.detach()
        return projected*g

class PrefixSemanticProbe(nn.Module):
    def __init__(self, d_LLM, L_mem, d_F):
        super().__init__()
        self.attn_pool=nn.Linear(d_LLM,1)
        self.fiber_decode=nn.Sequential(
            nn.Linear(d_LLM,2*d_F),nn.SiLU(),nn.LayerNorm(2*d_F),nn.Linear(2*d_F,d_F))
    def forward(self, prefix):
        w=F.softmax(self.attn_pool(prefix).squeeze(-1),dim=1)
        pooled=(w.unsqueeze(-1)*prefix).sum(1)
        return self.fiber_decode(pooled)

class PrefixAligner(nn.Module):
    def __init__(self, d_LLM, init_scale=0.5):
        super().__init__()
        self.ln=nn.LayerNorm(d_LLM)
        self.scale_logit=nn.Parameter(torch.tensor(init_scale))
        self.register_buffer('_target_std',torch.tensor(1.0))
        # [D] _calibrated flag 确保一次性校准,load_memory 不会重算
        self.register_buffer('_calibrated', torch.tensor(False))
    def calibrate(self, wte_fp32):
        if bool(self._calibrated.item()):
            return  # [D] idempotent
        with torch.no_grad():
            V = wte_fp32.shape[0]
            stride = max(1, V // 5000)
            idx = torch.arange(0, V, stride, device=wte_fp32.device)[:5000]
            sample = wte_fp32[idx]
            self._target_std.fill_(float(sample.std().item()))
            self._calibrated.fill_(True)
    def forward(self, prefix):
        normed=self.ln(prefix)
        scale=torch.sigmoid(self.scale_logit)*self._target_std
        return normed*scale
    def effective_scale(self) -> float:
        return float(torch.sigmoid(self.scale_logit).item() * self._target_std.item())

class ContentSemanticTailHead(nn.Module):
    """
    [B] slot_1 分解: α × residual + β × LN(head_out),β 可学习。
    head_out 由 MLP 产出,LN 限定其量级 ≈ sqrt(d_LLM)。
    """
    def __init__(self, d_F, d_LLM, n_slots, hidden=1024, tied_extra=True,
                 zero_init_tied=True, residual_dominant=True, beta_init=0.3):
        super().__init__()
        self.n_slots = n_slots; self.d_LLM = d_LLM; self.tied_extra = tied_extra
        self.zero_init_tied = zero_init_tied
        self.residual_dominant = residual_dominant
        if n_slots == 0: return
        self.shared = nn.Sequential(
            nn.Linear(d_F, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        n_distinct = min(n_slots, 2) if tied_extra else n_slots
        self.slot_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, d_LLM), nn.LayerNorm(d_LLM))
            for _ in range(n_distinct)])
        for i, head in enumerate(self.slot_heads):
            if tied_extra and zero_init_tied and i == 1:
                nn.init.zeros_(head[0].weight); nn.init.zeros_(head[0].bias)
            else:
                nn.init.normal_(head[0].weight, std=0.02); nn.init.zeros_(head[0].bias)
        self._n_distinct = n_distinct
        # [B] 每个 rare-keyword slot (s>=1) 一个 β
        if residual_dominant and n_slots >= 2:
            self.residual_beta = nn.Parameter(
                torch.full((n_slots - 1,), float(beta_init)))
            self.residual_ln = nn.LayerNorm(d_LLM)
        else:
            self.residual_beta = None
            self.residual_ln = None

    def _head_for_slot(self, s: int):
        if self.tied_extra:
            return self.slot_heads[0] if s == 0 else self.slot_heads[min(1, self._n_distinct - 1)]
        return self.slot_heads[s]

    def forward(self, fiber_summary):
        if self.n_slots == 0: return None
        h = self.shared(fiber_summary)
        slots = [self._head_for_slot(s)(h) for s in range(self.n_slots)]
        return torch.stack(slots, dim=1)

    def combine_with_residual(self, head_out, residual, alpha: float):
        """
        [B] slot_1..n-1 = α × residual + β × LN(head_out)
            slot_0 保持纯 head_out(general content)
        head_out: [B, n_slots, d_LLM]
        residual: [B, n_slots, d_LLM] (residual[:,0] 为零)
        """
        if residual is None or self.residual_beta is None:
            return head_out + (alpha * residual if residual is not None else 0.0)
        out = head_out.clone()
        for s in range(1, self.n_slots):
            beta_s = self.residual_beta[s - 1]
            out[:, s, :] = alpha * residual[:, s, :] + beta_s * self.residual_ln(head_out[:, s, :])
        return out

class ContextHead(nn.Module):
    def __init__(self, d_LLM):
        super().__init__()
        self.ln = nn.LayerNorm(d_LLM)
        self.proj = nn.Linear(d_LLM, d_LLM)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        return self.proj(self.ln(x))

class MemoryContextEncoder(nn.Module):
    """
    [A] Attention-pooled context encoder.
      Q = learnable Parameter(1, d_ctx)
      K,V = Linear(d_LLM, d_ctx)(hidden_states[content_positions])
      ctx = attn(Q, K, V) + residual_weight × W_ortho(wte_centroid)
    兜底:若 hidden_states 为 None 或无 content position,仅走 residual(orthogonal proj)。
    """
    def __init__(self, d_LLM, d_ctx, hidden=256, hybrid=True, hidden_weight=0.8,
                 use_attention_pool=True, residual_weight=0.3, attn_dropout=0.0):
        super().__init__()
        self.d_ctx = d_ctx
        self.d_LLM = d_LLM
        self.hybrid = hybrid
        self.hidden_weight = hidden_weight
        self.use_attention_pool = use_attention_pool
        self.residual_weight = residual_weight
        # Residual shortcut(JL-like 正交投影,始终保留),用于 fallback 与几何先验
        self.proj_wte = nn.Linear(d_LLM, d_ctx, bias=False)
        nn.init.orthogonal_(self.proj_wte.weight, gain=1.0)
        if use_attention_pool:
            self.attn_kv = nn.Linear(d_LLM, 2 * d_ctx, bias=False)
            self.attn_ln = nn.LayerNorm(d_LLM)
            self.attn_q = nn.Parameter(torch.randn(1, d_ctx) * (1.0 / math.sqrt(d_ctx)))
            self.attn_out = nn.Linear(d_ctx, d_ctx, bias=False)
            nn.init.orthogonal_(self.attn_out.weight, gain=1.0)
            self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None
        if hybrid and not use_attention_pool:
            self.proj_hid = nn.Linear(d_LLM, d_ctx, bias=False)
            nn.init.orthogonal_(self.proj_hid.weight, gain=1.0)
        self.back_proj = nn.Linear(d_ctx, d_LLM)
        nn.init.normal_(self.back_proj.weight, std=0.02)
        nn.init.zeros_(self.back_proj.bias)

    def _attention_pool(self, hidden_states, mask=None):
        """hidden_states: [T, d_LLM]; mask: [T] bool; returns [d_ctx]."""
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        h = self.attn_ln(hidden_states.float())
        kv = self.attn_kv(h)
        k, v = kv.chunk(2, dim=-1)
        q = self.attn_q
        scores = (q @ k.T) / math.sqrt(self.d_ctx)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)
        pooled = attn @ v
        out = self.attn_out(pooled).squeeze(0)
        return out

    def encode_with_hidden(self, wte_centroid, hidden_states=None,
                           hidden_mask=None):
        """
        [A] 主路径:
          主分量 = attention_pool(hidden_states, mask)
          残差分量 = proj_wte(wte_centroid)
          ctx = normalize(main + residual_weight × residual)
        """
        dev = next(self.parameters()).device
        residual = self.proj_wte(wte_centroid.float().to(dev))
        if (self.use_attention_pool and hidden_states is not None
                and hidden_states.numel() > 0):
            main = self._attention_pool(hidden_states.to(dev), mask=hidden_mask)
            out = main + self.residual_weight * residual
        elif self.hybrid and hasattr(self, 'proj_hid'):
            if hidden_states is not None and hidden_states.numel() > 0:
                hid_mean = hidden_states.float().mean(0)
                out = self.proj_hid(hid_mean) * self.hidden_weight + residual
            else:
                out = residual
        else:
            out = residual
        return F.normalize(out, dim=-1, eps=1e-8)

    def encode(self, wte_centroid, hidden_mean=None):
        if hidden_mean is not None and hidden_mean.numel() > 0:
            return self.encode_with_hidden(
                wte_centroid, hidden_states=hidden_mean.unsqueeze(0))
        return self.encode_with_hidden(wte_centroid, hidden_states=None)

    def encode_from_tokens(self, content_token_ids, wte, hidden_states=None,
                           hidden_mask=None):
        if not content_token_ids or wte is None: return None
        V = wte.shape[0]
        valid = [t for t in content_token_ids if 0 <= t < V]
        if not valid: return None
        idx = torch.tensor(sorted(valid), device=wte.device, dtype=torch.long)
        centroid = wte.index_select(0, idx).float().mean(0)
        return self.encode_with_hidden(
            centroid, hidden_states=hidden_states, hidden_mask=hidden_mask
        ).detach().contiguous()

    def encode_from_hidden(self, hidden_mean):
        return self.encode_with_hidden(
            torch.zeros(self.d_LLM, device=next(self.parameters()).device),
            hidden_states=hidden_mean.unsqueeze(0) if hidden_mean.dim() == 1 else hidden_mean)

    def decode(self, ctx_vec):
        return self.back_proj(ctx_vec)

class MixtureGateHead(nn.Module):
    def __init__(self, d_F, floor=0.0, ceiling=0.7, hidden=256):
        super().__init__()
        self.floor = floor; self.ceiling = ceiling
        self.net = nn.Sequential(
            nn.Linear(d_F, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, fiber_summary, override_ceiling: Optional[float] = None):
        raw = torch.sigmoid(self.net(fiber_summary)).squeeze(-1)
        ceil = override_ceiling if override_ceiling is not None else self.ceiling
        return self.floor + (ceil - self.floor) * raw

class ContentTokenClassifier:
    DEFAULT_STOPWORDS = frozenset({
        'the','a','an','is','are','was','were','be','been','being',
        'have','has','had','having','do','does','did','doing',
        'will','would','could','should','may','might','can','shall',
        'and','but','or','nor','for','yet','so',
        'in','on','at','to','of','by','with','from','as','into','through',
        'during','before','after','above','below','between','under','over',
        'that','this','these','those','it','its',
        'he','she','they','we','you','me','him','her','them','us',
        'his','her','their','our','your','my','mine','yours',
        'not','no','if','then','than','when','where','what','which','who',
        'how','all','each','every','both','few','more','most','some','any',
        'also','just','about','very','really','only','even','still','already',
        'up','down','out','off','away','back','here','there','now',
        'too','much','many','such','own','other','another',
        'because','since','while','although','though','until','unless',
        'however','therefore','moreover','furthermore','nevertheless',
        'like','get','got','go','went','gone','come','came',
        'make','made','take','took','give','gave','see','saw','know','knew',
        'think','thought','say','said','tell','told','want','need',
        'use','used','find','found','put','keep','kept','let',
        'seem','become','became','leave','left','call','called',
        'try','tried','ask','asked','work','worked','well','way',
        'thing','things','something','anything','nothing','everything',
        'one','two','first','new','old','good','bad','big','small',
        'long','little','right','same','different','last','next',
        'part','being','going','using','getting','making','looking',
        'coming','taking','having','doing','saying','working','trying',
        'include','includes','including','included'})
    DEFAULT_FILLER_WORDS = frozenset({
        'include','includes','including','included',
        'also','just','however','moreover','furthermore',
        'nevertheless','therefore','thus','hence','accordingly',
        'meanwhile','instead','rather','otherwise','additionally',
        'basically','essentially','actually','obviously','clearly',
        'simply','certainly','indeed','probably','perhaps',
        'apparently','presumably','supposedly','regardless',
        'nonetheless','conversely','alternatively','specifically',
        'generally','typically','usually','often','sometimes',
        'particularly','especially','notably',
        'various','several','many','multiple','different','diverse','varied',
        'certain','particular','specific','general','overall','whole','entire',
        'aspect','aspects','feature','features','element','elements',
        'factor','factors','component','components','quality','qualities',
        'example','examples','instance','instances','case','cases',
        'method','methods','approach','approaches','technique_generic',
        'process','processes','system','systems','part','parts',
        'kind','kinds','type','types','sort','sorts',
        'people','person','someone','anyone','everyone',
        'matter','matters','issue','issues','point','points',
        'number','numbers','amount','amounts','level','levels',
        'particularly','especially','notably',
        'student','students','practice','practicing',
        'action','actions','role','roles','purpose','purposes',
        'nature','natures','character','characters','condition','conditions',
        'state','states','status','statuses','fact','facts',
        'substance','substances','material','materials','content','contents',
        'context','contexts','task','tasks','duty','duties',
        'operation','operations','performance','performances',
        'activity','activities','topic','topics','subject','subjects',
        'concept','concepts','idea','ideas','notion','notions',
        'result','results','outcome','outcomes','effect','effects',
        'area','areas','region','regions','range','ranges',
        'degree','degrees','extent','extents','period','periods',
        'moment','moments','detail','details','information',
        'piece','pieces','group','groups','set','sets',
        'form','forms','style','styles','mode','modes','version','versions',
        'manner','manners','fashion','fashions','attribute','attributes',
        'property','properties','trait','traits','characteristic','characteristics',
        'place','places','way','ways'})

    def __init__(self, tokenizer, cfg=None, vocab_size=None, min_len=None, strict_min_len=None):
        if cfg is None: cfg = Cfg()
        self.cfg = cfg
        _min_len = min_len if isinstance(min_len, int) else cfg.content_min_len
        _strict_min_len = (strict_min_len if isinstance(strict_min_len, int)
                           else cfg.strict_starter_min_decoded_len)
        self.STOPWORDS = (cfg.stopwords_override if cfg.stopwords_override is not None
                          else self.DEFAULT_STOPWORDS | cfg.stopwords_extra)
        self.FILLER_WORDS = (cfg.filler_words_override if cfg.filler_words_override is not None
                             else self.DEFAULT_FILLER_WORDS | cfg.filler_words_extra)
        if cfg.dedup_filler_from_stop:
            self.FILLER_WORDS = self.FILLER_WORDS - self.STOPWORDS
        self.content_ids = set(); self.function_ids = set()
        self.punct_ids = set(); self.newline_ids = set()
        self.filler_ids = set(); self.word_starter_ids = set()
        self.content_starter_ids = set(); self.strict_content_starter_ids = set()
        V = int(vocab_size) if vocab_size is not None else int(getattr(tokenizer, 'vocab_size', 50257))
        self._V = V
        for i in range(V):
            try: tok_text = tokenizer.decode([i])
            except Exception:
                self.function_ids.add(i); continue
            if not isinstance(tok_text, str): self.function_ids.add(i); continue
            is_word_starter = len(tok_text) > 0 and tok_text[0] in (' ', '\t')
            stripped = tok_text.strip().lower()
            cleaned = ''.join(c for c in stripped if c.isalpha())
            if is_word_starter: self.word_starter_ids.add(i)
            if '\n' in tok_text:
                self.newline_ids.add(i); self.function_ids.add(i)
            elif stripped == '' or all(not c.isalnum() for c in stripped):
                self.punct_ids.add(i); self.function_ids.add(i)
            elif len(cleaned) >= _min_len and cleaned not in self.STOPWORDS:
                self.content_ids.add(i)
                if is_word_starter:
                    self.content_starter_ids.add(i)
                    if (stripped == cleaned and len(stripped) >= _strict_min_len
                            and stripped not in self.STOPWORDS
                            and stripped not in self.FILLER_WORDS):
                        self.strict_content_starter_ids.add(i)
            else: self.function_ids.add(i)
            if cleaned in self.FILLER_WORDS: self.filler_ids.add(i)
        self._content_tensor = None; self._content_starter_tensor = None
        self._strict_content_starter_tensor = None; self._filler_tensor = None
        self._function_tensor = None
        self._pure_function_tensor = None
        # [D] 冻结分类器指纹
        self._fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self):
        h = hashlib.sha256()
        for name, s in [
                ('content', self.content_ids), ('content_starter', self.content_starter_ids),
                ('strict', self.strict_content_starter_ids),
                ('filler', self.filler_ids), ('function', self.function_ids),
                ('newline', self.newline_ids), ('punct', self.punct_ids),
                ('word_starter', self.word_starter_ids)]:
            h.update(name.encode()); h.update(b'|')
            for t in sorted(s): h.update(str(int(t)).encode()); h.update(b',')
            h.update(b';')
        return h.hexdigest()

    def _mask_size(self): return int(self._V)
    def content_mask(self, device):
        if self._content_tensor is None or self._content_tensor.device != device:
            V = self._mask_size(); m = torch.zeros(V, device=device)
            for i in self.content_ids:
                if i < V: m[i] = 1.0
            self._content_tensor = m
        return self._content_tensor
    def content_starter_mask(self, device):
        if self._content_starter_tensor is None or self._content_starter_tensor.device != device:
            V = self._mask_size(); m = torch.zeros(V, device=device)
            for i in self.content_starter_ids:
                if i < V: m[i] = 1.0
            self._content_starter_tensor = m
        return self._content_starter_tensor
    def strict_content_starter_mask(self, device):
        if (self._strict_content_starter_tensor is None
                or self._strict_content_starter_tensor.device != device):
            V = self._mask_size(); m = torch.zeros(V, device=device)
            for i in self.strict_content_starter_ids:
                if i < V: m[i] = 1.0
            self._strict_content_starter_tensor = m
        return self._strict_content_starter_tensor
    def filler_mask(self, device):
        if self._filler_tensor is None or self._filler_tensor.device != device:
            V = self._mask_size(); m = torch.zeros(V, device=device)
            for i in self.filler_ids:
                if i < V: m[i] = 1.0
            self._filler_tensor = m
        return self._filler_tensor
    def function_mask(self, device):
        if self._function_tensor is None or self._function_tensor.device != device:
            V = self._mask_size(); m = torch.zeros(V, device=device)
            for i in self.function_ids:
                if i < V: m[i] = 1.0
            self._function_tensor = m
        return self._function_tensor
    def pure_function_mask(self, device, eos_id=None):
        cache_key = (device, eos_id)
        if (self._pure_function_tensor is None
                or getattr(self, '_pf_key', None) != cache_key):
            V = self._mask_size(); m = torch.zeros(V, device=device)
            exclude = set(self.newline_ids) | set(self.punct_ids)
            if eos_id is not None: exclude.add(int(eos_id))
            for i in self.function_ids:
                if i < V and i not in exclude: m[i] = 1.0
            self._pure_function_tensor = m
            self._pf_key = cache_key
        return self._pure_function_tensor
    def get_content_ids_from_tokens(self, token_ids):
        return [t for t in token_ids if t in self.content_ids]
    def get_strict_starter_ids_from_tokens(self, token_ids):
        return [t for t in token_ids if t in self.strict_content_starter_ids]

class MemoryVocabProjector(nn.Module):
    def __init__(self, d_F, d_LLM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_F, 4*d_LLM), nn.SiLU(), nn.LayerNorm(4*d_LLM),
            nn.Linear(4*d_LLM, 2*d_LLM), nn.SiLU(), nn.LayerNorm(2*d_LLM),
            nn.Linear(2*d_LLM, d_LLM))
        nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)
    def forward(self, fiber_summary, wte_weight):
        mem_emb = self.proj(fiber_summary)
        mem_n = F.normalize(mem_emb, dim=-1, eps=1e-8)
        wte_n = F.normalize(wte_weight, dim=-1, eps=1e-8)
        return mem_n @ wte_n.T

@dataclass
class MemEntry:
    mid: int; base: torch.Tensor; fiber: torch.Tensor; dirn: torch.Tensor
    surprise: float; ts: float; last: float; cnt: int = 0; version: int = 0
    source_text: str = ""
    content_token_ids: List[int] = field(default_factory=list)
    semantic_emb: Optional[torch.Tensor] = None
    expanded_content_ids: List[int] = field(default_factory=list)
    rare_keyword_ids: List[int] = field(default_factory=list)
    context_descriptor: Optional[torch.Tensor] = None
    strict_starter_ids: List[int] = field(default_factory=list)
    # [C] cluster id from offline KMeans(inter-domain)
    cluster_id: int = -1

class _Node:
    __slots__=('leaf','ids','children','centers','depth')
    def __init__(self,d=0):
        self.depth=d; self.leaf=True; self.ids=[]; self.children=[]; self.centers=None
    def count(self):
        return len(self.ids) if self.leaf else sum(c.count() for c in self.children)

class DirectionTree:
    def __init__(self, c, amm_ref=None):
        self.c=c; self.root=_Node(); self.store={}; self.nid=0
        self._amm_ref = amm_ref
    def insert(self, m):
        self.store[m.mid]=m; self._ins(self.root,m)
    def _ins(self, nd, m):
        if nd.leaf:
            nd.ids.append(m.mid)
            if len(nd.ids)>self.c.tree_max_leaf: self._split(nd)
        else:
            best=self._best(nd,m.dirn); self._ins(nd.children[best],m); self._update_centers(nd)
    def update(self, mid, new_base=None, new_fiber=None, new_dirn=None):
        if mid not in self.store: return
        m=self.store[mid]; dc=False
        if new_base is not None: m.base=new_base.detach().clone()
        if new_fiber is not None: m.fiber=new_fiber.detach().clone()
        if new_dirn is not None: dc=True; m.dirn=new_dirn.detach().clone()
        m.version+=1
        if dc: self._rm(self.root,mid); self._ins(self.root,m); self._rebalance(self.root)
    def _split(self, nd):
        ids=nd.ids
        if len(ids)<2: return
        K=min(self.c.tree_K,len(ids))
        if K<2: return
        dirs=torch.stack([self.store[i].dirn for i in ids])
        centered=dirs-dirs.mean(0)
        try: _,_,Vh=torch.linalg.svd(centered,full_matrices=False)
        except: return
        n_comp=min(K,dirs.shape[1]); proj=centered@Vh[:n_comp].T
        asgn=self._farthest_kmeans(proj,K)
        children=[]
        for k in range(K):
            ch=_Node(nd.depth+1); ch.ids=[ids[i] for i in range(len(ids)) if asgn[i]==k]
            if ch.ids: children.append(ch)
        if len(children)<=1: return
        nd.leaf=False; nd.children=children; nd.ids=[]; self._update_centers(nd)
        for ch in nd.children:
            if ch.leaf and len(ch.ids)>self.c.tree_max_leaf: self._split(ch)
    @staticmethod
    def _farthest_kmeans(data, K, max_iter=50):
        N=data.shape[0]; K=min(K,N)
        if K<=0: return torch.zeros(N,dtype=torch.long,device=data.device)
        ctrs=[data[0].clone()]
        for _ in range(K-1):
            d2=torch.cdist(data,torch.stack(ctrs)).min(1)[0].pow(2)
            ctrs.append(data[d2.argmax()].clone())
        ctrs=torch.stack(ctrs); asgn=torch.zeros(N,dtype=torch.long,device=data.device)
        for _ in range(max_iter):
            dists=torch.cdist(data,ctrs); new=dists.argmin(1)
            if (new==asgn).all(): break
            asgn=new
            for k in range(K):
                mk=asgn==k
                if mk.any(): ctrs[k]=data[mk].mean(0)
                else:
                    far=dists.min(1)[0].argmax(); ctrs[k]=data[far].clone(); asgn[far]=k
        return asgn
    def _best(self, nd, d):
        if nd.centers is None or len(nd.children)==0: return 0
        return (nd.centers@d).argmax().item()
    def _beam_retrieve(self, qdir, bw):
        beams=[(self.root,0.)]; results={}
        while beams:
            nb=[]
            for nd,sc in beams:
                if nd.leaf:
                    for mid in nd.ids:
                        if mid in self.store:
                            s=(qdir@self.store[mid].dirn).item()+sc
                            if mid not in results or s>results[mid]: results[mid]=s
                elif nd.centers is not None:
                    sims=nd.centers@qdir; tk=min(bw,len(nd.children)); _,idxs=sims.topk(tk)
                    for i in idxs: nb.append((nd.children[i.item()],sc+sims[i.item()].item()))
                else:
                    for ch in nd.children: nb.append((ch,sc))
            nb.sort(key=lambda x:-x[1]); beams=nb[:bw]
        # 确定性:mid 升序 tie-break
        return sorted(results.items(), key=lambda x: (-x[1], x[0]))
    def retrieve(self, qdir, bw=3):
        raw = self._beam_retrieve(qdir, bw)
        amm = self._amm_ref
        if amm is None: return raw
        if not getattr(amm.c, 'use_tree_semantic_rerank', False): return raw
        if amm.training: return raw
        cc = getattr(amm, '_content_classifier', None)
        wte_n = getattr(amm, 'wte_normed', None)
        q_ids = getattr(amm, '_last_query_ids', None)
        if cc is None or wte_n is None or q_ids is None: return raw
        try:
            q_tokens = q_ids[0].tolist() if q_ids.dim() > 1 else q_ids.tolist()
        except Exception:
            return raw
        q_content = [t for t in q_tokens if t in cc.content_ids]
        if not q_content: return raw
        V_wte = wte_n.shape[0]
        q_content = [t for t in q_content if t < V_wte]
        if not q_content: return raw
        corpus_idf = amm._compute_corpus_idf(cc)
        idf_floor = amm.c.idf_floor
        q_centroid = AMM._compute_idf_weighted_centroid(
            q_content, wte_n, corpus_idf, idf_floor)
        if q_centroid is None: return raw
        a_d = amm.c.tree_rerank_dir_weight
        a_c = amm.c.tree_rerank_centroid_weight
        a_f = amm.c.tree_rerank_forward_weight
        # [C] crowding: 先算每个 mid 的综合 score,再按 top-1 cluster 扣分
        scored = []
        for mid, dir_score in raw:
            mem = self.store.get(mid)
            if mem is None:
                scored.append((mid, float(dir_score), -1)); continue
            m_ids = amm._get_mem_scoring_ids(mem)
            m_ids = [t for t in m_ids if t < V_wte]
            if not m_ids:
                scored.append((mid, a_d * max(-1.0, min(1.0, float(dir_score))),
                               mem.cluster_id))
                continue
            m_centroid = AMM._compute_idf_weighted_centroid(
                m_ids, wte_n, corpus_idf, idf_floor)
            cen_sim = float((q_centroid @ m_centroid).item()) if m_centroid is not None else 0.0
            fwd_sim = AMM._compute_forward_maxsim(
                q_content, m_ids, wte_n, corpus_idf, idf_floor)
            dir_clamped = max(-1.0, min(1.0, float(dir_score)))
            combined = a_d * dir_clamped + a_c * cen_sim + a_f * fwd_sim
            scored.append((mid, combined, mem.cluster_id))
        # [C] crowding penalty:与 top-1 同 cluster 的条目保留分数,异 cluster 扣 λ
        if amm.c.use_inter_domain_margin and scored:
            top_cluster = None
            for mid, sc, cid in scored:
                if cid >= 0:
                    top_cluster = cid; break
            if top_cluster is not None:
                lam = amm.c.retrieval_crowding_lambda
                scored = [
                    (mid, sc - lam if (cid >= 0 and cid != top_cluster) else sc, cid)
                    for mid, sc, cid in scored
                ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(mid, sc) for mid, sc, _ in scored]
    def remove(self, mid):
        if mid not in self.store: return
        del self.store[mid]; self._rm(self.root,mid); self._rebalance(self.root)
    def _rm(self, nd, mid):
        if nd.leaf:
            if mid in nd.ids: nd.ids.remove(mid); return True
            return False
        return any(self._rm(c,mid) for c in nd.children)
    def _rebalance(self, nd):
        if nd.leaf: return
        for c in nd.children: self._rebalance(c)
        nd.children=[c for c in nd.children if c.count()>0]
        if not nd.children: nd.leaf=True; nd.ids=[]; nd.centers=None
        elif len(nd.children)==1:
            ch=nd.children[0]; nd.leaf=ch.leaf; nd.ids=ch.ids; nd.children=ch.children; nd.centers=ch.centers
        else: self._update_centers(nd)
    def _update_centers(self, nd):
        cs=[]
        for c in nd.children:
            ids=self._collect(c); dirs=[self.store[i].dirn for i in ids if i in self.store]
            if not dirs: continue
            cs.append(F.normalize(torch.stack(dirs).mean(0),dim=0))
        nd.centers=torch.stack(cs) if cs else None
    def _collect(self, nd):
        if nd.leaf: return list(nd.ids)
        return [i for c in nd.children for i in self._collect(c)]
    def rebuild(self):
        ms=list(self.store.values()); self.root=_Node()
        for m in ms: self._ins(self.root,m)
    def verify_consistency(self):
        errs=[]; ti=set(self._collect(self.root)); si=set(self.store.keys())
        if ti!=si: errs.append(f"tree≠store: tree_only={ti-si}, store_only={si-ti}")
        if self.root.count()!=len(self.store): errs.append(f"count mismatch")
        return errs
    def max_depth(self) -> int:
        def _d(nd):
            if nd.leaf: return nd.depth
            if not nd.children: return nd.depth
            return max(_d(c) for c in nd.children)
        return _d(self.root)
    def leaf_size_violations(self) -> List[Tuple[int, int]]:
        viols = []
        def _check(nd):
            if nd.leaf:
                if len(nd.ids) > self.c.tree_max_leaf:
                    viols.append((nd.depth, len(nd.ids)))
            else:
                for c in nd.children: _check(c)
        _check(self.root)
        return viols
    def recluster_semantic(self, K):
        """[C] 对所有 memory 的 semantic_emb 做 KMeans,赋予 cluster_id。"""
        mids = sorted(self.store.keys())
        mems = [self.store[mid] for mid in mids]
        embs = []
        valid_mids = []
        for mem in mems:
            if mem.semantic_emb is not None:
                v = mem.semantic_emb.detach().flatten().float()
                if v.numel() > 0 and torch.isfinite(v).all():
                    embs.append(v); valid_mids.append(mem.mid)
        if len(embs) < K:
            for mid in mids: self.store[mid].cluster_id = 0
            return
        X = torch.stack(embs, dim=0)
        X_n = F.normalize(X, dim=-1, eps=1e-8)
        assign, _ = _simple_kmeans(X_n, k=K, n_iter=20, seed=42)
        assign_list = assign.detach().cpu().tolist()
        seen = set()
        for mid, a in zip(valid_mids, assign_list):
            self.store[mid].cluster_id = int(a); seen.add(mid)
        for mid in mids:
            if mid not in seen: self.store[mid].cluster_id = 0

class FiberAttn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh=c.n_heads_fiber; self.hd=c.d_F//c.n_heads_fiber
        self.Wq=nn.Linear(c.d_F,c.d_F,bias=False); self.Wk=nn.Linear(c.d_F,c.d_F,bias=False)
        self.Wv=nn.Linear(c.d_F,c.d_F,bias=False); self.Wo=nn.Linear(c.d_F,c.d_F,bias=False)
        self.n1=nn.LayerNorm(c.d_F)
        self.ff=nn.Sequential(nn.Linear(c.d_F,2*c.d_F),nn.GELU(),nn.Linear(2*c.d_F,c.d_F))
        self.n2=nn.LayerNorm(c.d_F)
    def forward(self, qf, mf, mem_mask=None, dir_bias=None):
        B,C,d=mf.shape; nh=self.nh; hd=self.hd; S=1+C
        seq=torch.cat([qf.unsqueeze(1),mf],1)
        Q=self.Wq(seq).reshape(B,S,nh,hd).permute(0,2,1,3)
        K=self.Wk(seq).reshape(B,S,nh,hd).permute(0,2,1,3)
        V=self.Wv(seq).reshape(B,S,nh,hd).permute(0,2,1,3)
        a=(Q@K.transpose(-2,-1))/math.sqrt(hd)
        if dir_bias is not None:
            db=dir_bias.unsqueeze(1).unsqueeze(2)
            pad=torch.zeros(B,1,1,1,**_dev(a)); a=a+torch.cat([pad,db],-1)
        if mem_mask is not None:
            qm=torch.ones(B,1,**_dev(mem_mask)); full=torch.cat([qm,mem_mask],1)
            a=a.masked_fill(full.unsqueeze(1).unsqueeze(2)==0,-1e9)
        a=F.softmax(a,-1); out=(a@V).permute(0,2,1,3).reshape(B,S,d)
        out=self.n1(seq+self.Wo(out)); out=self.n2(out+self.ff(out))
        return out[:,1:]

class QFormerLayer(nn.Module):
    def __init__(self, c):
        super().__init__(); d=c.d_LLM; nh=c.bridge_heads
        self.sa=nn.MultiheadAttention(d,nh,batch_first=True)
        self.ca=nn.MultiheadAttention(d,nh,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d); self.n3=nn.LayerNorm(d)
    def forward(self, q, k, v, kv_mask=None):
        h=self.n1(q); q=q+self.sa(h,h,h)[0]; h=self.n2(q)
        kpm=None
        if kv_mask is not None:
            kpm=(kv_mask==0); all_m=kpm.all(dim=-1)
            if all_m.any(): kpm=kpm.clone(); kpm[all_m]=False
        q=q+self.ca(h,k,v,key_padding_mask=kpm)[0]
        return q+self.ff(self.n3(q))

class QFormerProj(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q=nn.Parameter(torch.randn(c.L_mem,c.d_LLM)*0.02)
        self.fkv=nn.Linear(c.d_F,c.d_LLM*2)
        self.layers=nn.ModuleList([QFormerLayer(c) for _ in range(c.bridge_layers)])
        self.norm=nn.LayerNorm(c.d_LLM)
    def forward(self, fibers, mem_mask=None):
        B=fibers.shape[0]; kv=self.fkv(fibers); k,v=kv.chunk(2,-1)
        q=self.q.unsqueeze(0).expand(B,-1,-1)
        for l in self.layers: q=l(q,k,v,kv_mask=mem_mask)
        return self.norm(q)

class AdaptiveLayerPool(nn.Module):
    def __init__(self, n, d):
        super().__init__(); self.w=nn.Parameter(torch.linspace(-2,2,n))
    def forward(self, hs):
        w=F.softmax(self.w,0); return sum(w[i]*h for i,h in enumerate(hs))
    def weight_dist(self): return F.softmax(self.w.detach(),0)

class StateExtractor(nn.Module):
    def __init__(self, c):
        super().__init__(); pos_dim=5
        self.sc=nn.Sequential(nn.Linear(c.d_LLM+pos_dim,c.d_LLM//4),nn.Tanh(),
                              nn.Linear(c.d_LLM//4,1))
        self.tb=nn.Linear(c.d_LLM,c.d_M); self.tf=nn.Linear(c.d_LLM,c.d_F)
    def _pos_feat(self, T, ref):
        pos=torch.linspace(0,1,T,**_dev(ref))
        return torch.stack([pos,torch.sin(pos*math.pi),torch.cos(pos*math.pi),
                           torch.sin(2*pos*math.pi),torch.cos(2*pos*math.pi)],-1)
    def forward(self, h, mask=None):
        B,T,_=h.shape; pf=self._pos_feat(T,h).unsqueeze(0).expand(B,-1,-1)
        s=self.sc(torch.cat([h,pf],-1)).squeeze(-1)
        if mask is not None and mask.shape[1]==T:
            s=s.masked_fill(mask==0,-1e9)
        w=F.softmax(s,-1); p=(w.unsqueeze(-1)*h).sum(1)
        return self.tb(p), self.tf(p)

class EmbBridge(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c=c
        self.proj=QFormerProj(c); self.ext=StateExtractor(c)
        self.pe=nn.Parameter(torch.randn(c.L_mem,c.d_LLM)*0.02)
        self.bypass=ContentBypass(c.d_F,c.d_LLM,gate_bias=c.bypass_init_gate_bias)
        self.aligner=PrefixAligner(c.d_LLM,c.prefix_init_scale)
        self._effective_tail_slots = (c.effective_tail_slots()
                                      if c.use_content_semantic_tail else 0)
        self.tail_head = ContentSemanticTailHead(
            c.d_F, c.d_LLM,
            n_slots=self._effective_tail_slots,
            hidden=c.tail_head_hidden,
            tied_extra=c.tail_head_tied_extra,
            zero_init_tied=c.tail_head_zero_init_tied,
            residual_dominant=c.tail_slot_residual_dominant,
            beta_init=c.tail_slot_beta_init)
        self._effective_ctx_slots = c.effective_ctx_slots()
        if self._effective_ctx_slots > 0:
            self.context_heads = nn.ModuleList([
                ContextHead(c.d_LLM) for _ in range(self._effective_ctx_slots)])
        else:
            self.context_heads = None
        self._last_inject_diag={}
        self._last_fiber_summary=None
        self._last_tail_slots=None
        self._last_context_slot=None
        self._last_tail_pre_renorm = None   # [B] 测试探针
        self._last_residual = None
        # [v3.45 cond-buffer] Separate mirrors that are updated ONLY when
        # inject is called with is_cond_path=True.  Audit probes that want
        # the cond-path tensors (e.g. 4.23 keyword_specific_tail_slot_probe)
        # MUST read these instead of _last_*, because
        # prepare_decode_context's second inject call (uncond contrastive)
        # overwrites _last_* with uncond values.
        self._last_cond_fiber_summary = None
        self._last_cond_tail_slots = None
        self._last_cond_context_slot = None
        self._last_cond_tail_pre_renorm = None
        self._last_cond_residual = None
        self._last_cond_inject_diag = {}

    def _build_body_prefix(self, fibers, mem_mask, fiber_summary):
        qf_out = self.proj(fibers, mem_mask) + self.pe.unsqueeze(0)
        bp_out = None; gate_val = None
        if fiber_summary is not None:
            qf_context = qf_out.mean(1)
            bp_out = self.bypass(fiber_summary, qf_context)
            gate_val = self.bypass._last_gate
            qf_out = qf_out + bp_out.unsqueeze(1)
        qf_out = self.aligner(qf_out)
        return qf_out, bp_out, gate_val

    def _apply_filler_projection_and_renorm(self, qf_out, filler_centroid):
        L = qf_out.shape[1]; filler_dir_used = False
        if self.c.use_filler_direction_projection and filler_centroid is not None:
            n_proj = min(self.c.filler_projection_last_slots, L)
            fd = filler_centroid.view(1, 1, -1)
            mask_slot = torch.zeros(L, device=qf_out.device)
            mask_slot[L - n_proj:] = 1.0
            mask_slot = mask_slot.view(1, -1, 1)
            comp = (qf_out * fd).sum(-1, keepdim=True)
            qf_out = qf_out - comp * fd * mask_slot
            filler_dir_used = True
        if self.c.use_slot_norm_renormalize:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            cur_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            qf_out = qf_out * (target_norm / cur_norms)
        elif self.c.use_prefix_norm_clamp:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            max_allowed = target_norm * self.c.prefix_norm_clamp_ratio
            slot_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(max_allowed / slot_norms, max=1.0)
            qf_out = qf_out * scale
        return qf_out, filler_dir_used

    def inject(self, fibers, mem_mask=None, fiber_summary=None,
               filler_centroid=None, context_descriptors_d_llm=None,
               rare_keyword_wte_residual=None, is_cond_path: bool = True):
        qf_out, bp_out, gate_val = self._build_body_prefix(fibers, mem_mask, fiber_summary)
        L_total = qf_out.shape[1]
        tail_slots_used = 0
        ctx_slots_used = 0

        pieces = []
        use_ctx = (self._effective_ctx_slots > 0
                   and context_descriptors_d_llm is not None
                   and len(context_descriptors_d_llm) > 0)
        if use_ctx:
            ctx_pieces = []
            for i, ctx_vec in enumerate(context_descriptors_d_llm):
                if i >= self._effective_ctx_slots: break
                if ctx_vec is None: continue
                head = self.context_heads[i]
                ctx_emb = head(ctx_vec)
                ctx_aligned = self.aligner(ctx_emb.unsqueeze(1))
                ctx_pieces.append(ctx_aligned)
            if ctx_pieces:
                ctx_all = torch.cat(ctx_pieces, dim=1)
                pieces.append(ctx_all)
                ctx_slots_used = ctx_all.shape[1]
                self._last_context_slot = ctx_all.detach()
            else:
                self._last_context_slot = None
        else:
            self._last_context_slot = None

        if (self._effective_tail_slots > 0 and fiber_summary is not None):
            tail_raw = self.tail_head(fiber_summary)
            tail_aligned = self.aligner(tail_raw)
            self._last_tail_pre_renorm = tail_aligned.detach()
            if (self.c.wte_residual_post_aligner
                    and rare_keyword_wte_residual is not None):
                alpha = self.c.wte_residual_alpha
                # [B] 用 combine_with_residual 替代朴素加法
                if self.c.tail_slot_residual_dominant:
                    tail_aligned = self.tail_head.combine_with_residual(
                        tail_aligned, rare_keyword_wte_residual, alpha)
                else:
                    tail_aligned = tail_aligned + alpha * rare_keyword_wte_residual
                self._last_residual = rare_keyword_wte_residual.detach()
            else:
                self._last_residual = None
            pieces.append(tail_aligned)
            tail_slots_used = self._effective_tail_slots
        else:
            self._last_residual = None

        n_replace = ctx_slots_used + tail_slots_used
        if n_replace > 0 and n_replace <= L_total:
            replacement = torch.cat(pieces, dim=1)
            qf_out = torch.cat([qf_out[:, :L_total - n_replace, :], replacement], dim=1)

        qf_out, filler_dir_used = self._apply_filler_projection_and_renorm(qf_out, filler_centroid)

        if tail_slots_used > 0:
            tail_start = L_total - tail_slots_used
            self._last_tail_slots = qf_out[:, tail_start:L_total].detach()
        else:
            self._last_tail_slots = None

        self._last_fiber_summary = (fiber_summary.detach()
                                    if fiber_summary is not None else None)
        self._last_inject_diag = {
            'bypass_gate': gate_val.mean().item() if gate_val is not None else None,
            'qf_norm': qf_out.norm().item(),
            'bypass_norm': bp_out.norm().item() if bp_out is not None else 0.0,
            'aligner_scale': self.aligner.effective_scale(),
            'last_slot_norm_per_b': qf_out[:, -1].norm(dim=-1).mean().item(),
            'tail_slots_used': tail_slots_used,
            'ctx_slot_used': ctx_slots_used,
            'filler_dir_projected': filler_dir_used,
            'is_cond_path': bool(is_cond_path)}
        # [v3.45 cond-buffer] Mirror to cond-only buffers so that a later
        # uncond inject in prepare_decode_context does not clobber what
        # audit probes need.  Nothing else on this path reads these mirrors;
        # reads happen only from audit code (and from new
        # slot_residual_alignment_loss if enabled, see below).
        if is_cond_path:
            self._last_cond_fiber_summary = self._last_fiber_summary
            self._last_cond_tail_slots = self._last_tail_slots
            self._last_cond_context_slot = self._last_context_slot
            self._last_cond_tail_pre_renorm = self._last_tail_pre_renorm
            self._last_cond_residual = self._last_residual
            self._last_cond_inject_diag = dict(self._last_inject_diag)
        return qf_out

    def build_neutral_prefix(self, B, device):
        qf_out = self.pe.unsqueeze(0).expand(B, -1, -1).contiguous()
        qf_out = self.aligner(qf_out)
        if self.c.use_slot_norm_renormalize:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            cur_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            qf_out = qf_out * (target_norm / cur_norms)
        elif self.c.use_prefix_norm_clamp:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            max_allowed = target_norm * self.c.prefix_norm_clamp_ratio
            slot_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(max_allowed / slot_norms, max=1.0)
            qf_out = qf_out * scale
        return qf_out

class LossWarmup:
    def __init__(self, schedules): self.schedules=schedules; self.step_count=0
    def weight(self, name):
        ws=self.schedules.get(name,0)
        if ws<=0: return 1.0
        return min(1.0, self.step_count/max(ws,1))
    def advance(self): self.step_count+=1

class GradientMonitor:
    def __init__(self): self._groups={}
    def register(self, name, mod): self._groups[name]=mod
    def snapshot(self):
        norms={}
        for name,mod in self._groups.items():
            total=0.0; cnt=0
            for p in mod.parameters():
                if p.grad is not None: total+=p.grad.norm().item()**2; cnt+=1
            norms[name]=math.sqrt(total) if cnt>0 else 0.0
        return norms

class DegenerationGuard:
    def __init__(self, tok, cfg, content_classifier=None):
        self.tok=tok; self.cfg=cfg; self.cc=content_classifier
    def process(self, logits, generated_ids, step):
        punct_ids = self.cc.punct_ids if self.cc else set()
        newline_ids = self.cc.newline_ids if self.cc else set()
        V = logits.shape[-1]
        if step < self.cfg.early_content_steps:
            pen_p = self.cfg.degen_early_punct_penalty
            pen_n = self.cfg.degen_early_newline_penalty
            for pid in punct_ids:
                if pid < V: logits[0, pid] -= pen_p
            for nid in newline_ids:
                if nid < V: logits[0, nid] -= pen_n
        if step < self.cfg.degen_min_tokens and self.tok.eos_token_id is not None:
            if self.tok.eos_token_id < V:
                logits[0, self.tok.eos_token_id] = -float('inf')
        seen = set(generated_ids[-30:]) if generated_ids else set()
        for tid in seen:
            if tid < V:
                if logits[0, tid] > 0: logits[0, tid] /= self.cfg.degen_repeat_penalty
                else: logits[0, tid] *= self.cfg.degen_repeat_penalty
        mc = self.cfg.degen_max_consec_punct
        if len(generated_ids) >= mc:
            recent = generated_ids[-mc:]
            if all(t in punct_ids for t in recent):
                for pid in punct_ids:
                    if pid < V: logits[0, pid] -= 10.0
        return logits

@dataclass
class RetrievalDiag:
    was_flat_scan: bool = False
    recall_count: int = 0
    reranker_delta_mean: float = 0.0
    fiber_summary_norm: float = 0.0
    top_reranker_score: float = 0.0
    top_dir_sim: float = 0.0; top_sem_sim: float = 0.0
    top_forward_maxsim: float = 0.0; top_backward_maxsim: float = 0.0
    top_bidi_min: float = 0.0; top_gate_affinity: float = 0.0; gate_threshold: float = 0.0
    n_gate_pass: int = 0; n_candidates_initial: int = 0
    n_after_strict_overlap_gate: int = 0; n_after_upstream_semantic_gate: int = 0
    n_after_hard_filter: int = 0; n_after_score_filter: int = 0
    n_after_coherence_filter: int = 0; n_after_bidi_gap_filter: int = 0
    n_after_mean_center: int = 0
    mean_center_applied: bool = False
    mean_center_dropped_ids: List[int] = field(default_factory=list)
    mean_center_raw_scores: Dict[int, float] = field(default_factory=dict)
    mean_center_final_scores: Dict[int, float] = field(default_factory=dict)
    hungarian_used: bool = False
    batch_mem_weights: List[List[Tuple[int, float]]] = field(default_factory=list)
    per_memory_forward_maxsim: Dict[int, float] = field(default_factory=dict)
    per_memory_bidi_min: Dict[int, float] = field(default_factory=dict)
    per_memory_sem_sim: Dict[int, float] = field(default_factory=dict)
    per_memory_gate_affinity: Dict[int, float] = field(default_factory=dict)
    per_memory_strict_overlap: Dict[int, int] = field(default_factory=dict)
    dominant_per_batch: List[Optional[int]] = field(default_factory=list)
    dominant_memory_id: Optional[int] = None
    non_dominant_per_batch: List[List[int]] = field(default_factory=list)
    non_dominant_weights_per_batch: List[Dict[int, float]] = field(default_factory=list)
    idf_applied: bool = False; centroid_applied: bool = False
    top_centroid_cosine: float = 0.0
    per_memory_centroid_cosine: Dict[int, float] = field(default_factory=dict)
    upstream_semantic_gate_applied: bool = False
    upstream_gate_dropped_ids: List[int] = field(default_factory=list)
    strict_overlap_gate_applied: bool = False
    strict_overlap_dropped_ids: List[int] = field(default_factory=list)
    n_candidates_for_rerank: int = 0
    min_keep_enforcements: int = 0

class AMM(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c=c
        self.metric=RiemannianMetric(c.d_M)
        self.geo=GeodesicSolver(self.metric,c)
        self.conn=FiberConnection(c.d_M,c.d_F,self.metric,grad_coupling=True)
        self.trans=FiberTransporter(self.conn,c)
        self.ctx=CtxEncoder(c); self.fib=FibEncoder(c)
        self.dir_pred=DirectionPredictor(c.d_M,c.d_F)
        self.write_gate=WriteGate(c); self.retention=RetentionScorer(c)
        self.attn=FiberAttn(c); self.empty_state=EmptyStateNet(c.d_M,c.d_F)
        self.contrast_proj_f=nn.Linear(c.d_F,c.d_M,bias=False)
        self.contrast_proj_x=nn.Linear(c.d_M,c.d_M,bias=False)
        nn.init.eye_(self.contrast_proj_x.weight)
        self.reranker=RetrievalReranker(c.d_M,c.d_F,clip=c.reranker_clip)
        self.tree=DirectionTree(c, amm_ref=self); self.time=0.
        self.wte_normed = None
        self._last_query_ids = None
        self._last_query_mask = None
        self._content_classifier = None
        self._writes_since_recluster = 0

    def surprise_proxy(self, logits, tgt):
        nll=-F.log_softmax(logits,-1).gather(2,tgt.unsqueeze(-1)).squeeze(-1)
        T=nll.shape[1]
        if T==0: return logits.new_zeros(logits.shape[0])
        w=torch.linspace(0.5,1.5,T,**_dev(nll)); w=w/w.sum()*T
        return (nll*w.unsqueeze(0)).mean(-1)

    def _compute_dirn(self, base, fiber):
        with torch.no_grad():
            return self.dir_pred(base.unsqueeze(0),fiber.unsqueeze(0)).squeeze(0)

    def _get_mem_scoring_ids(self, mem):
        if self.c.retrieval_use_expanded_ids and mem.expanded_content_ids:
            return mem.expanded_content_ids
        return mem.content_token_ids

    def _compute_corpus_idf(self, content_classifier):
        s = self.c.tfidf_smoothing
        N = len(self.tree.store)
        if N == 0: return {}
        df = {}
        # [D] 确定性:按 mid 排序迭代
        for mid in sorted(self.tree.store.keys()):
            mem = self.tree.store[mid]
            label_set = (set(t for t in mem.content_token_ids
                             if t in content_classifier.content_starter_ids)
                         if content_classifier is not None else set(mem.content_token_ids))
            for t in label_set: df[t] = df.get(t, 0) + 1
        return {t: math.log((N + s) / (d + s)) + 1.0 for t, d in df.items()}

    @staticmethod
    def _compute_idf_weighted_centroid(token_ids, wte_normed, corpus_idf, idf_floor=0.1):
        if not token_ids or wte_normed is None: return None
        V = wte_normed.shape[0]
        valid = sorted([t for t in token_ids if t < V])  # [D] 排序
        if not valid: return None
        if corpus_idf is not None and len(corpus_idf) > 0:
            weights = torch.tensor(
                [max(corpus_idf.get(t, idf_floor), idf_floor) for t in valid],
                device=wte_normed.device, dtype=wte_normed.dtype)
        else:
            weights = torch.ones(len(valid), device=wte_normed.device, dtype=wte_normed.dtype)
        vecs = wte_normed[valid]
        centroid = (vecs * weights.unsqueeze(1)).sum(0) / weights.sum().clamp(min=1e-8)
        return F.normalize(centroid, dim=-1, eps=1e-8)

    def _compute_forward_hungarian(self, query_ids, mem_ids, wte_normed, query_idf=None, idf_floor=0.1):
        if not query_ids or not mem_ids: return 0.0
        V = wte_normed.shape[0]
        q_valid = sorted([q for q in query_ids if q < V])
        m_valid = sorted([m for m in mem_ids if m < V])
        if not q_valid or not m_valid: return 0.0
        n_q, n_m = len(q_valid), len(m_valid)
        q_vecs = wte_normed[q_valid]; m_vecs = wte_normed[m_valid]
        sim = q_vecs @ m_vecs.T
        if max(n_q, n_m) > self.c.hungarian_max_n:
            max_per_q = sim.max(dim=1).values
            if query_idf is not None:
                w = torch.tensor(
                    [max(query_idf.get(q, idf_floor), idf_floor) for q in q_valid],
                    device=wte_normed.device, dtype=sim.dtype)
                return ((max_per_q * w).sum() / w.sum().clamp(min=1e-8)).item()
            return max_per_q.mean().item()
        pairs, _ = hungarian_max_assignment(sim)
        if pairs.numel() == 0: return 0.0
        matched_sims = sim[pairs[:, 0], pairs[:, 1]]
        if query_idf is not None:
            q_ids_for_pairs = [q_valid[int(r.item())] for r in pairs[:, 0]]
            w = torch.tensor(
                [max(query_idf.get(q, idf_floor), idf_floor) for q in q_ids_for_pairs],
                device=wte_normed.device, dtype=matched_sims.dtype)
            return ((matched_sims * w).sum() / w.sum().clamp(min=1e-8)).item()
        return matched_sims.mean().item()

    @staticmethod
    def _compute_forward_maxsim(query_ids, mem_ids, wte_normed, query_idf=None, idf_floor=0.1):
        if not query_ids or not mem_ids: return 0.0
        V = wte_normed.shape[0]
        q_valid = sorted([q for q in query_ids if q < V])
        m_valid = sorted([m for m in mem_ids if m < V])
        if not q_valid or not m_valid: return 0.0
        q_vecs = wte_normed[q_valid]; m_vecs = wte_normed[m_valid]
        sim = q_vecs @ m_vecs.T
        max_per_q = sim.max(dim=1).values
        if query_idf is not None:
            weights = torch.tensor(
                [max(query_idf.get(q, idf_floor), idf_floor) for q in q_valid],
                device=wte_normed.device, dtype=sim.dtype)
            total = weights.sum().clamp(min=1e-8)
            return ((max_per_q * weights).sum() / total).item()
        return max_per_q.mean().item()

    @staticmethod
    def _compute_backward_maxsim(query_ids, mem_ids, wte_normed, query_idf=None, idf_floor=0.1):
        if not query_ids or not mem_ids: return 0.0
        V = wte_normed.shape[0]
        q_valid = sorted([q for q in query_ids if q < V])
        m_valid = sorted([m for m in mem_ids if m < V])
        if not q_valid or not m_valid: return 0.0
        q_vecs = wte_normed[q_valid]; m_vecs = wte_normed[m_valid]
        sim = q_vecs @ m_vecs.T
        max_per_m_vals, max_per_m_idx = sim.max(dim=0)
        if query_idf is not None:
            q_weights = torch.tensor(
                [max(query_idf.get(q, idf_floor), idf_floor) for q in q_valid],
                device=wte_normed.device, dtype=sim.dtype)
            matched_weights = q_weights[max_per_m_idx]
            total = matched_weights.sum().clamp(min=1e-8)
            return ((max_per_m_vals * matched_weights).sum() / total).item()
        return max_per_m_vals.mean().item()

    def _compute_bidi_min(self, q_ids, m_ids, wte_normed, query_idf, idf_floor):
        fwd = (self._compute_forward_hungarian(q_ids, m_ids, wte_normed, query_idf, idf_floor)
               if self.c.use_hungarian_fwd
               else self._compute_forward_maxsim(q_ids, m_ids, wte_normed, query_idf, idf_floor))
        bwd = self._compute_backward_maxsim(q_ids, m_ids, wte_normed, query_idf, idf_floor)
        return fwd, bwd, min(fwd, bwd)

    @staticmethod
    def _count_strict_overlap_matches(q_strict_ids, m_strict_ids, wte_normed, sim_threshold):
        if not q_strict_ids or not m_strict_ids or wte_normed is None: return 0
        V = wte_normed.shape[0]
        q_valid = sorted([t for t in q_strict_ids if t < V])
        m_valid = sorted([t for t in m_strict_ids if t < V])
        if not q_valid or not m_valid: return 0
        dev = wte_normed.device
        q_vecs = wte_normed[torch.tensor(q_valid, device=dev)]
        m_vecs = wte_normed[torch.tensor(m_valid, device=dev)]
        sim = q_vecs @ m_vecs.T
        has_match = (sim >= sim_threshold).any(dim=1)
        return int(has_match.sum().item())

    def _check_consolidation_compatible(self, existing_content_ids, new_content_ids):
        if not existing_content_ids or not new_content_ids: return True
        if self.wte_normed is None: return True
        _, _, m = self._compute_bidi_min(existing_content_ids, new_content_ids,
                                         self.wte_normed, None, self.c.idf_floor)
        return m >= self.c.consol_maxsim_min

    def store_mem(self, h, surp, training_mode=False, source_text="",
                  content_token_ids=None, content_semantic_emb=None,
                  expanded_content_ids=None, context_descriptor=None,
                  strict_starter_ids=None):
        dev=h.device; h2=h.unsqueeze(0)
        x=self.ctx(h2).squeeze(0).detach()
        s=surp if isinstance(surp,torch.Tensor) else torch.tensor(surp,**_dev(h))
        sv=s.view(1) if s.dim()<=1 else s
        f=self.fib(h2,x.unsqueeze(0),sv).squeeze(0).detach()
        d=self._compute_dirn(x,f)
        sem_emb=content_semantic_emb if content_semantic_emb is not None else h.detach().clone()
        ct_ids=content_token_ids or []; exp_ids=expanded_content_ids or []
        strict_ids=strict_starter_ids or []
        if self.tree.store:
            scored=self.tree.retrieve(d.detach(),bw=1)[:5]
            for mid,_ in scored:
                if mid in self.tree.store:
                    ex=self.tree.store[mid]
                    dist=self.metric.midpoint_approx_distance(
                        x.unsqueeze(0),ex.base.unsqueeze(0).to(dev)).item()
                    if dist<self.c.consol_dist*self.c.consol_conflict_ratio:
                        if not self._check_consolidation_compatible(
                                ex.content_token_ids, ct_ids): continue
                        alpha=self.c.write_update_alpha
                        nb=((1-alpha)*ex.fiber+alpha*f).detach().clone()
                        nba=((1-alpha)*ex.base+alpha*x).detach().clone()
                        nd=self._compute_dirn(nba,nb)
                        self.tree.update(mid,new_base=nba,new_fiber=nb,new_dirn=nd)
                        ex.surprise=max(ex.surprise,s.item()); ex.last=self.time; ex.cnt+=1
                        if source_text: ex.source_text=source_text
                        if sem_emb is not None:
                            if ex.semantic_emb is not None:
                                ex.semantic_emb=((1-alpha)*ex.semantic_emb.to(dev)+alpha*sem_emb).detach().clone()
                            else: ex.semantic_emb=sem_emb.detach().clone()
                        # [D] 确定性合并
                        if ct_ids: ex.content_token_ids=_sorted_set(ex.content_token_ids+ct_ids)
                        if exp_ids: ex.expanded_content_ids=_sorted_set(ex.expanded_content_ids+exp_ids)
                        if strict_ids: ex.strict_starter_ids=_sorted_set(ex.strict_starter_ids+strict_ids)
                        if context_descriptor is not None:
                            ex.context_descriptor = context_descriptor.detach().clone().contiguous()
                        ex.rare_keyword_ids = []
                        self.time+=1
                        self._writes_since_recluster += 1
                        return ex
        m=MemEntry(mid=self.tree.nid,
                   base=x.detach().clone().contiguous(),
                   fiber=f.detach().clone().contiguous(),
                   dirn=d.detach().clone().contiguous(),
                   surprise=s.item(),ts=self.time,last=self.time,
                   source_text=source_text,content_token_ids=_sorted_set(ct_ids),
                   semantic_emb=(sem_emb.detach().clone().contiguous()
                                  if sem_emb is not None else None),
                   expanded_content_ids=_sorted_set(exp_ids),
                   rare_keyword_ids=[],
                   strict_starter_ids=_sorted_set(strict_ids),
                   context_descriptor=(context_descriptor.detach().clone().contiguous()
                                        if context_descriptor is not None else None))
        self.tree.nid+=1; self.tree.insert(m); self.time+=1
        self._writes_since_recluster += 1
        return m

    def maybe_recluster(self, force=False):
        """[C] 批量写入后 re-cluster semantic_emb。"""
        if not self.c.use_inter_domain_margin: return
        K = self.c.inter_domain_kmeans_k
        if force or self._writes_since_recluster >= self.c.mem_recluster_every_writes:
            if len(self.tree.store) >= K:
                self.tree.recluster_semantic(K)
            self._writes_since_recluster = 0

    def _preserve_min_keep(self, pass_mask, scores, min_keep, diag):
        total = pass_mask.numel()
        effective = min(min_keep, total)
        n_pass = int(pass_mask.sum().item())
        if n_pass >= effective:
            return pass_mask
        keep_n = effective
        top_idx = scores.topk(min(keep_n, total)).indices
        add_mask = torch.zeros_like(pass_mask)
        add_mask[top_idx] = True
        new_mask = pass_mask | add_mask
        if new_mask.sum().item() > n_pass:
            diag.min_keep_enforcements += 1
        return new_mask

    def retrieve_multi(self, xq, fq, topk=None, bw=None, update_stats=True,
                       query_semantic_emb=None, query_content_ids_per_batch=None,
                       wte_normed=None, content_classifier=None):
        B=xq.shape[0]; dev=xq.device
        topk=topk or self.c.retrieval_topk; bw=bw or self.c.retrieval_beam
        recall_k=int(topk*self.c.retrieval_recall_factor)
        flat_thresh=self.c.flat_scan_threshold_factor*topk
        qdir=self.dir_pred(xq,fq)
        diag=RetrievalDiag()
        corpus_idf = (self._compute_corpus_idf(content_classifier)
                      if self.c.use_idf_retrieval else None)
        diag.idf_applied = corpus_idf is not None
        diag.centroid_applied = self.c.use_idf_centroid
        diag.hungarian_used = self.c.use_hungarian_fwd
        idf_floor = self.c.idf_floor
        min_keep_global = self.c.retrieval_min_keep_for_rerank
        if not self.tree.store:
            empty=self.empty_state(xq,fq); mask=torch.ones(B,1,**_dev(xq))
            summary=empty.mean(1) if empty.dim()==3 else empty
            diag.fiber_summary_norm=summary.norm().item()
            diag.batch_mem_weights=[[] for _ in range(B)]
            diag.dominant_per_batch=[None for _ in range(B)]
            diag.non_dominant_per_batch=[[] for _ in range(B)]
            diag.non_dominant_weights_per_batch=[{} for _ in range(B)]
            return empty.unsqueeze(1),mask,summary,diag
        all_results=[]; all_masks=[]; all_biases=[]; all_summaries=[]; all_batch_mw=[]
        all_dominant=[]; all_non_dominant=[]; all_non_dom_weights=[]
        wn=wte_normed if wte_normed is not None else self.wte_normed
        for b in range(B):
            n_store=len(self.tree.store)
            if n_store<=flat_thresh:
                mids=sorted(self.tree.store.keys())  # [D] 确定性
                diag.was_flat_scan=True
            else:
                scored=self.tree.retrieve(qdir[b].detach(),bw)
                mids=[s[0] for s in scored[:recall_k]]
            mems=[self.tree.store[i] for i in mids if i in self.tree.store]
            diag.recall_count=len(mems); diag.n_candidates_initial=len(mems)
            if not mems:
                empty=self.empty_state(xq[b:b+1],fq[b:b+1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1,**_dev(xq)))
                all_biases.append(torch.zeros(1,**_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([]); all_dominant.append(None)
                all_non_dominant.append([]); all_non_dom_weights.append({})
                continue
            q_content_ids=(query_content_ids_per_batch[b]
                           if query_content_ids_per_batch and b<len(query_content_ids_per_batch)
                           else [])
            q_strict = []
            if content_classifier is not None:
                q_strict = [t for t in q_content_ids
                            if t in content_classifier.strict_content_starter_ids
                            and wn is not None and t < wn.shape[0]]
            if (self.c.use_strict_content_overlap_gate
                    and q_strict and wn is not None and content_classifier is not None):
                overlap_counts = torch.zeros(len(mems), dtype=torch.long, device=dev)
                for mi, mem in enumerate(mems):
                    m_strict = [t for t in mem.content_token_ids
                                if t in content_classifier.strict_content_starter_ids
                                and t < wn.shape[0]]
                    cnt = self._count_strict_overlap_matches(
                        q_strict, m_strict, wn, self.c.strict_overlap_sim_threshold)
                    overlap_counts[mi] = cnt
                    diag.per_memory_strict_overlap[mem.mid] = cnt
                pass_mask = overlap_counts >= self.c.strict_overlap_min_matches
                pass_mask = self._preserve_min_keep(
                    pass_mask, overlap_counts.float(),
                    max(self.c.strict_overlap_min_keep, min_keep_global), diag)
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                diag.strict_overlap_dropped_ids = [mems[i].mid for i in dropped_local]
                diag.strict_overlap_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < len(mems):
                    mems = [mems[i] for i in keep_local.tolist()]
            diag.n_after_strict_overlap_gate = len(mems)
            C_init = len(mems)
            if C_init == 0:
                empty=self.empty_state(xq[b:b+1],fq[b:b+1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1,**_dev(xq)))
                all_biases.append(torch.zeros(1,**_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([]); all_dominant.append(None)
                all_non_dominant.append([]); all_non_dom_weights.append({})
                continue
            sb_all=torch.stack([m.base.to(dev) for m in mems])
            sf_all=torch.stack([m.fiber.to(dev) for m in mems])
            md_all=torch.stack([m.dirn.to(dev) for m in mems])
            sem_sim_all=torch.zeros(C_init, device=dev)
            if query_semantic_emb is not None:
                for mi, mem in enumerate(mems):
                    if mem.semantic_emb is not None:
                        sem_sim_all[mi] = F.cosine_similarity(
                            query_semantic_emb[b:b+1],
                            mem.semantic_emb.unsqueeze(0).to(dev),dim=-1).squeeze()
            forward_all=torch.zeros(C_init, device=dev)
            backward_all=torch.zeros(C_init, device=dev)
            bidi_min_all=torch.zeros(C_init, device=dev)
            if q_content_ids and wn is not None:
                for mi, mem in enumerate(mems):
                    scoring_ids = self._get_mem_scoring_ids(mem)
                    fwd, bwd, bmin = self._compute_bidi_min(
                        q_content_ids, scoring_ids, wn, corpus_idf, idf_floor)
                    forward_all[mi] = fwd; backward_all[mi] = bwd; bidi_min_all[mi] = bmin
            if self.c.use_upstream_semantic_gate and q_content_ids and wn is not None:
                fwd_pass = forward_all >= self.c.upstream_gate_fwd_idf_floor
                sem_pass = sem_sim_all >= self.c.upstream_gate_sem_floor
                pass_mask = fwd_pass & sem_pass
                composite_score = 0.5 * forward_all + 0.5 * sem_sim_all
                pass_mask = self._preserve_min_keep(
                    pass_mask, composite_score,
                    max(self.c.upstream_gate_min_keep, min_keep_global), diag)
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                if dropped_local:
                    diag.upstream_gate_dropped_ids = [mems[i].mid for i in dropped_local]
                diag.upstream_semantic_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C_init:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb_all = sb_all[keep_local]; sf_all = sf_all[keep_local]
                    md_all = md_all[keep_local]; sem_sim_all = sem_sim_all[keep_local]
                    forward_all = forward_all[keep_local]
                    backward_all = backward_all[keep_local]
                    bidi_min_all = bidi_min_all[keep_local]
                    C_init = len(mems)
            diag.n_after_upstream_semantic_gate = C_init
            diag.n_candidates_for_rerank = C_init
            sb = sb_all; sf = sf_all
            sem_sim_t = sem_sim_all; forward_t = forward_all; bidi_min_t = bidi_min_all
            raw_dir_sim = torch.einsum('d,cd->c', qdir[b], md_all)
            diag.top_dir_sim = raw_dir_sim.max().item() if C_init > 0 else 0.0
            diag.top_sem_sim = sem_sim_t.max().item() if C_init > 0 else 0.0
            diag.top_forward_maxsim = forward_t.max().item() if C_init > 0 else 0.0
            diag.top_backward_maxsim = backward_all.max().item() if C_init > 0 else 0.0
            diag.top_bidi_min = bidi_min_t.max().item() if C_init > 0 else 0.0
            centroid_scores = torch.zeros(C_init, device=dev)
            if self.c.use_idf_centroid and q_content_ids and wn is not None:
                q_centroid = self._compute_idf_weighted_centroid(q_content_ids, wn, corpus_idf, idf_floor)
                if q_centroid is not None:
                    for mi, mem in enumerate(mems):
                        m_scoring_ids = self._get_mem_scoring_ids(mem)
                        m_centroid = self._compute_idf_weighted_centroid(
                            m_scoring_ids, wn, corpus_idf, idf_floor)
                        if m_centroid is not None:
                            centroid_scores[mi] = (q_centroid @ m_centroid).item()
                diag.top_centroid_cosine = centroid_scores.max().item() if C_init > 0 else 0.0
            combined_sim = (self.c.ret_centroid_weight * centroid_scores
                            + self.c.ret_sem_weight * sem_sim_t
                            + self.c.ret_bidi_min_weight * bidi_min_t
                            + self.c.ret_forward_maxsim_weight * forward_t
                            + self.c.ret_dir_weight * raw_dir_sim)
            C = C_init
            top_sem  = sem_sim_t.max().item()  if C > 0 else 0.0
            top_bidi = bidi_min_t.max().item() if C > 0 else 0.0
            sem_thresh  = max(self.c.gate_sem_floor,  top_sem  * self.c.gate_sem_ratio)
            bidi_thresh = max(self.c.gate_bidi_floor, top_bidi * self.c.gate_bidi_ratio,
                              self.c.gate_bidi_hard_min)
            hard_mask = (sem_sim_t >= sem_thresh) & (bidi_min_t >= bidi_thresh)
            gate_affinity = (self.c.gate_sem_weight  * sem_sim_t
                             + self.c.gate_bidi_weight * bidi_min_t)
            diag.top_gate_affinity = gate_affinity.max().item() if C > 0 else 0.0
            diag.gate_threshold    = max(sem_thresh, bidi_thresh)
            diag.n_gate_pass       = int(hard_mask.sum().item())
            hard_mask = self._preserve_min_keep(
                hard_mask, combined_sim, min_keep_global, diag)
            diag.n_after_hard_filter = int(hard_mask.sum().item())
            for mi, mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid] = gate_affinity[mi].item()
            keep_indices = hard_mask.nonzero(as_tuple=True)[0]
            if keep_indices.numel() > 0 and keep_indices.numel() < C:
                mems = [mems[i] for i in keep_indices.tolist()]
                sb = sb[keep_indices]; sf = sf[keep_indices]
                combined_sim = combined_sim[keep_indices]
                raw_dir_sim = raw_dir_sim[keep_indices]
                forward_t = forward_t[keep_indices]; bidi_min_t = bidi_min_t[keep_indices]
                sem_sim_t = sem_sim_t[keep_indices]; centroid_scores = centroid_scores[keep_indices]
                C = len(mems)
            rerank_scores = self.reranker(
                xq[b:b+1], fq[b:b+1], sb.unsqueeze(0), sf.unsqueeze(0),
                combined_sim.unsqueeze(0)).squeeze(0)
            diag.reranker_delta_mean = (rerank_scores - combined_sim).abs().mean().item()
            diag.top_reranker_score = rerank_scores.max().item() if C > 0 else 0.0
            if C > 1:
                top_score = rerank_scores.max()
                score_mask = rerank_scores >= top_score * self.c.score_keep_ratio
                score_mask = self._preserve_min_keep(
                    score_mask, rerank_scores, min_keep_global, diag)
                score_keep = score_mask.nonzero(as_tuple=True)[0]
                diag.n_after_score_filter = score_keep.numel()
                if score_keep.numel() < C:
                    mems = [mems[i] for i in score_keep.tolist()]
                    sb = sb[score_keep]; sf = sf[score_keep]
                    rerank_scores = rerank_scores[score_keep]
                    forward_t = forward_t[score_keep]; bidi_min_t = bidi_min_t[score_keep]
                    sem_sim_t = sem_sim_t[score_keep]; centroid_scores = centroid_scores[score_keep]
                    C = len(mems)
            else: diag.n_after_score_filter = C
            if C > 1 and forward_t.max().item() > 0:
                top_fwd_here = forward_t.max()
                coherence_mask = forward_t >= top_fwd_here * self.c.fwd_coherence_ratio
                coherence_mask = self._preserve_min_keep(
                    coherence_mask, forward_t, min_keep_global, diag)
                coherence_keep = coherence_mask.nonzero(as_tuple=True)[0]
                diag.n_after_coherence_filter = coherence_keep.numel()
                if coherence_keep.numel() < C:
                    mems = [mems[i] for i in coherence_keep.tolist()]
                    sb = sb[coherence_keep]; sf = sf[coherence_keep]
                    rerank_scores = rerank_scores[coherence_keep]
                    forward_t = forward_t[coherence_keep]; bidi_min_t = bidi_min_t[coherence_keep]
                    sem_sim_t = sem_sim_t[coherence_keep]; centroid_scores = centroid_scores[coherence_keep]
                    C = len(mems)
            else: diag.n_after_coherence_filter = C
            if C > 1 and bidi_min_t.max().item() > 0:
                top_bidi_here = bidi_min_t.max().item()
                gap_mask = bidi_min_t >= (top_bidi_here - self.c.bidi_absolute_gap)
                gap_mask = self._preserve_min_keep(
                    gap_mask, bidi_min_t, min_keep_global, diag)
                gap_keep = gap_mask.nonzero(as_tuple=True)[0]
                diag.n_after_bidi_gap_filter = gap_keep.numel()
                if gap_keep.numel() < C:
                    mems = [mems[i] for i in gap_keep.tolist()]
                    sb = sb[gap_keep]; sf = sf[gap_keep]
                    rerank_scores = rerank_scores[gap_keep]
                    forward_t = forward_t[gap_keep]; bidi_min_t = bidi_min_t[gap_keep]
                    sem_sim_t = sem_sim_t[gap_keep]; centroid_scores = centroid_scores[gap_keep]
                    C = len(mems)
            else: diag.n_after_bidi_gap_filter = C
            raw_composite = (0.4 * centroid_scores + 0.4 * forward_t
                             + 0.15 * bidi_min_t + 0.05 * sem_sim_t.clamp(min=0))
            if self.c.use_mean_centered_scoring and C >= self.c.mc_require_min_candidates:
                C_f = float(C); sum_raw = raw_composite.sum()
                centered = (C_f / (C_f - 1.0)) * raw_composite - sum_raw / (C_f - 1.0)
                for mi, mem in enumerate(mems):
                    diag.mean_center_raw_scores[mem.mid] = raw_composite[mi].item()
                    diag.mean_center_final_scores[mem.mid] = centered[mi].item()
                keep_mask = centered > self.c.mc_keep_margin
                keep_mask = self._preserve_min_keep(
                    keep_mask, centered,
                    max(self.c.mc_min_keep, min_keep_global), diag)
                dropped_local = (~keep_mask).nonzero(as_tuple=True)[0].tolist()
                if dropped_local:
                    diag.mean_center_applied = True
                    diag.mean_center_dropped_ids = [mems[i].mid for i in dropped_local]
                keep_local = keep_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb = sb[keep_local]; sf = sf[keep_local]
                    rerank_scores = rerank_scores[keep_local]
                    forward_t = forward_t[keep_local]; bidi_min_t = bidi_min_t[keep_local]
                    sem_sim_t = sem_sim_t[keep_local]; centroid_scores = centroid_scores[keep_local]
                    raw_composite = raw_composite[keep_local]
                    C = len(mems)
            diag.n_after_mean_center = C
            # [C] cluster-crowding 二次 rerank
            cluster_adjust = torch.zeros(C, device=dev)
            if self.c.use_inter_domain_margin and C >= 2:
                dom_local = int(rerank_scores.argmax().item())
                dom_cluster = mems[dom_local].cluster_id
                if dom_cluster >= 0:
                    for mi, mem in enumerate(mems):
                        if mi == dom_local: continue
                        if mem.cluster_id >= 0 and mem.cluster_id != dom_cluster:
                            cluster_adjust[mi] = -self.c.retrieval_crowding_lambda
            dominant_mid = None; non_dominant_mids = []; non_dom_weights = {}
            if C >= 1:
                final_rank = (0.4 * rerank_scores + 0.4 * centroid_scores
                              + 0.2 * forward_t + cluster_adjust)
                dom_idx = int(final_rank.argmax().item())
                dominant_mid = mems[dom_idx].mid
                if C > 1:
                    nd_idx = torch.tensor([i for i in range(C) if i != dom_idx], device=dev)
                    nd_scores = final_rank[nd_idx]
                    nd_w = F.softmax(nd_scores / self.c.retrieval_weight_temperature, dim=0)
                    for j, idx in enumerate(nd_idx.tolist()):
                        mid_j = mems[idx].mid
                        non_dominant_mids.append(mid_j)
                        non_dom_weights[mid_j] = nd_w[j].item()
            if not self.training and C > topk:
                _, top_idx = rerank_scores.topk(topk)
                mems = [mems[i] for i in top_idx.cpu().tolist()]
                sb = sb[top_idx]; sf = sf[top_idx]
                rerank_scores = rerank_scores[top_idx]
                forward_t = forward_t[top_idx]; bidi_min_t = bidi_min_t[top_idx]
                sem_sim_t = sem_sim_t[top_idx]; centroid_scores = centroid_scores[top_idx]
                cluster_adjust = cluster_adjust[top_idx] if cluster_adjust.numel() > 0 else cluster_adjust
                C = topk
            for mi, mem in enumerate(mems):
                diag.per_memory_forward_maxsim[mem.mid] = forward_t[mi].item()
                diag.per_memory_bidi_min[mem.mid] = bidi_min_t[mi].item()
                diag.per_memory_sem_sim[mem.mid] = sem_sim_t[mi].item()
                diag.per_memory_centroid_cosine[mem.mid] = centroid_scores[mi].item()
            qp = xq[b].unsqueeze(0).expand(C, -1)
            geo_r = self.geo.solve(sb, qp)
            transported = self.trans(sf, geo_r.path)
            if self.training:
                ret_s = self.retention(sb, sf,
                    torch.tensor([m.surprise for m in mems], **_dev(xq)),
                    torch.tensor([self.time - m.last for m in mems], **_dev(xq)),
                    torch.tensor([m.cnt for m in mems], **_dev(xq)))
                transported = transported * ret_s.unsqueeze(-1)
            if update_stats:
                for m in mems: m.last = self.time; m.cnt += 1
            final_scores = (0.4 * rerank_scores + 0.4 * centroid_scores
                            + 0.2 * forward_t + cluster_adjust)
            w = F.softmax(final_scores / self.c.retrieval_weight_temperature, dim=0)
            fs = (transported * w.unsqueeze(-1)).sum(0)
            batch_mw = [(m.mid, w[mi].item()) for mi, m in enumerate(mems)]
            all_batch_mw.append(batch_mw)
            all_dominant.append(dominant_mid); all_non_dominant.append(non_dominant_mids)
            all_non_dom_weights.append(non_dom_weights)
            all_results.append(transported); all_masks.append(torch.ones(C, **_dev(xq)))
            all_biases.append(final_scores / self.c.tau); all_summaries.append(fs)
        maxC = max(r.shape[0] for r in all_results)
        padded = []; pm = []; pd = []
        for bi in range(B):
            r, mk, db = all_results[bi], all_masks[bi], all_biases[bi]; gap = maxC - r.shape[0]
            if gap > 0:
                pr = self.empty_state(xq[bi:bi+1], fq[bi:bi+1]).expand(gap, -1)
                r = torch.cat([r, pr if self.training else pr.detach()], 0)
                mk = torch.cat([mk, torch.zeros(gap, **_dev(xq))])
                db = torch.cat([db, torch.full((gap,), -1e9, **_dev(xq))])
            padded.append(r); pm.append(mk); pd.append(db)
        mf = torch.stack(padded); mem_mask = torch.stack(pm); dir_bias = torch.stack(pd)
        fiber_summary = torch.stack(all_summaries)
        diag.fiber_summary_norm = fiber_summary.norm().item()
        diag.batch_mem_weights = all_batch_mw
        diag.dominant_per_batch = all_dominant
        diag.non_dominant_per_batch = all_non_dominant
        diag.non_dominant_weights_per_batch = all_non_dom_weights
        if diag.dominant_per_batch and diag.dominant_per_batch[0] is not None:
            diag.dominant_memory_id = diag.dominant_per_batch[0]
        refined = self.attn(fq, mf, mem_mask=mem_mask, dir_bias=dir_bias)
        return refined, mem_mask, fiber_summary, diag

    def decay(self):
        rm = []
        for mid in sorted(self.tree.store.keys()):
            m = self.tree.store[mid]
            dt = torch.tensor([self.time - m.last], **_dev(m.base))
            cnt = torch.tensor([m.cnt], **_dev(m.base))
            with torch.no_grad():
                sc = self.retention(m.base.unsqueeze(0), m.fiber.unsqueeze(0),
                    torch.tensor([m.surprise], **_dev(m.base)), dt, cnt).item()
            if sc < self.c.retention_gc_threshold: rm.append(mid)
        for i in rm: self.tree.remove(i)
        return len(rm)

    def consolidate(self):
        ms = [self.tree.store[mid] for mid in sorted(self.tree.store.keys())]
        if len(ms) < 2: return 0
        merged = set()
        for i in range(len(ms)):
            if ms[i].mid in merged: continue
            for j in range(i+1, len(ms)):
                if ms[j].mid in merged: continue
                d = self.metric.midpoint_approx_distance(
                    ms[i].base.unsqueeze(0), ms[j].base.unsqueeze(0)).item()
                if d < self.c.consol_dist:
                    if not self._check_consolidation_compatible(
                            ms[i].content_token_ids, ms[j].content_token_ids): continue
                    wi, wj = ms[i].cnt+1, ms[j].cnt+1; t = wi+wj
                    nb = (ms[i].base*wi + ms[j].base*wj) / t
                    nf = (ms[i].fiber*wi + ms[j].fiber*wj) / t
                    nd = self._compute_dirn(nb, nf)
                    ms[i].base = nb.detach().clone().contiguous()
                    ms[i].fiber = nf.detach().clone().contiguous()
                    ms[i].dirn = nd.detach().clone().contiguous()
                    ms[i].cnt += ms[j].cnt
                    ms[i].surprise = max(ms[i].surprise, ms[j].surprise); ms[i].version += 1
                    if ms[j].source_text and not ms[i].source_text:
                        ms[i].source_text = ms[j].source_text
                    ms[i].content_token_ids = _sorted_set(ms[i].content_token_ids + ms[j].content_token_ids)
                    ms[i].expanded_content_ids = _sorted_set(ms[i].expanded_content_ids + ms[j].expanded_content_ids)
                    ms[i].strict_starter_ids = _sorted_set(ms[i].strict_starter_ids + ms[j].strict_starter_ids)
                    if ms[i].semantic_emb is not None and ms[j].semantic_emb is not None:
                        ms[i].semantic_emb = ((ms[i].semantic_emb*wi + ms[j].semantic_emb*wj) / t).detach().clone().contiguous()
                    elif ms[j].semantic_emb is not None:
                        ms[i].semantic_emb = ms[j].semantic_emb.clone().contiguous()
                    ms[i].rare_keyword_ids = []
                    merged.add(ms[j].mid)
        for mid in merged: del self.tree.store[mid]
        if merged: self.tree.rebuild()
        return len(merged)

@dataclass
class DecodeContext:
    prefix_cond: torch.Tensor
    prefix_uncond: Optional[torch.Tensor]
    fiber_summary: torch.Tensor
    diag: RetrievalDiag
    content_bias: torch.Tensor
    suppression_bias: torch.Tensor
    vocab_bias: Optional[torch.Tensor]
    mixture_gate: Optional[torch.Tensor] = None
    memory_logit_bias: Optional[torch.Tensor] = None

_PREFIX_META_ATTR = "_mem_decode_prompt_len"
_PREFIX_GUIDANCE_ACTIVE_ATTR = "_mem_guidance_active"
_PREFIX_CONTENT_BIAS_ATTR = "_mem_content_bias"
_PREFIX_SUPPRESSION_BIAS_ATTR = "_mem_suppression_bias"

def _set_prefix_meta(prefix_tensor, prompt_len):
    try: setattr(prefix_tensor, _PREFIX_META_ATTR, int(prompt_len))
    except Exception: pass
def _get_prefix_meta(prefix_tensor):
    return getattr(prefix_tensor, _PREFIX_META_ATTR, None)
def _set_prefix_guidance(prefix_tensor, active: bool):
    try: setattr(prefix_tensor, _PREFIX_GUIDANCE_ACTIVE_ATTR, bool(active))
    except Exception: pass
def _get_prefix_guidance(prefix_tensor):
    return getattr(prefix_tensor, _PREFIX_GUIDANCE_ACTIVE_ATTR, False)
def _set_prefix_biases(prefix_tensor, content_bias, suppression_bias):
    try:
        setattr(prefix_tensor, _PREFIX_CONTENT_BIAS_ATTR, content_bias)
        setattr(prefix_tensor, _PREFIX_SUPPRESSION_BIAS_ATTR, suppression_bias)
    except Exception: pass

class MemLLM(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        self.amm = AMM(c); self.bridge = EmbBridge(c)
        self.semantic_probe = PrefixSemanticProbe(c.d_LLM, c.L_mem, c.d_F)
        self.vocab_proj = MemoryVocabProjector(c.d_F, c.d_LLM)
        if c.use_memory_context_encoder:
            self.memory_context_encoder = MemoryContextEncoder(
                c.d_LLM, c.d_ctx, hidden=c.context_encoder_hidden,
                hybrid=c.context_encoder_hybrid,
                hidden_weight=c.context_hybrid_hidden_weight,
                use_attention_pool=c.context_encoder_use_attention_pool,
                residual_weight=c.context_encoder_residual_weight,
                attn_dropout=c.context_encoder_attn_dropout)
        else:
            self.memory_context_encoder = None
        if c.use_mixture_decoding:
            self.mixture_gate_head = MixtureGateHead(
                c.d_F, floor=c.mixture_gate_floor, ceiling=c.mixture_gate_ceiling,
                hidden=c.mixture_gate_hidden)
        else:
            self.mixture_gate_head = None
        self.layer_pool = None; self.backbone = None
        self.tok = None; self._degen_guard = None; self.content_classifier = None
        self._wte_neighbor_cache = None
        self._wte_normed = None
        self._wte_mean_fp32 = None
        self._filler_centroid = None
        self._classifier_fingerprint = None  # [D]

    def load(self, name=None, dtype_name=None):
        name = name or self.c.llm_name
        dtype_name = dtype_name or self.c.llm_dtype
        self.backbone = LLMBackbone(name, dtype_name=dtype_name)
        self.tok = self.backbone.tokenizer
        self.c.d_LLM = self.backbone.d_model
        self.c.vocab_size = self.backbone.vocab_size
        dev = next(self.parameters()).device
        if self.bridge.proj.fkv.out_features != 2 * self.c.d_LLM:
            self.bridge = EmbBridge(self.c).to(dev)
            self.semantic_probe = PrefixSemanticProbe(self.c.d_LLM, self.c.L_mem, self.c.d_F).to(dev)
            self.vocab_proj = MemoryVocabProjector(self.c.d_F, self.c.d_LLM).to(dev)
            if self.c.use_memory_context_encoder:
                self.memory_context_encoder = MemoryContextEncoder(
                    self.c.d_LLM, self.c.d_ctx,
                    hidden=self.c.context_encoder_hidden,
                    hybrid=self.c.context_encoder_hybrid,
                    hidden_weight=self.c.context_hybrid_hidden_weight,
                    use_attention_pool=self.c.context_encoder_use_attention_pool,
                    residual_weight=self.c.context_encoder_residual_weight,
                    attn_dropout=self.c.context_encoder_attn_dropout).to(dev)
        self.layer_pool = AdaptiveLayerPool(self.backbone.n_layers + 1, self.c.d_LLM).to(dev)
        self.content_classifier = ContentTokenClassifier(
            self.tok, self.c, vocab_size=self.backbone.vocab_size)
        self._classifier_fingerprint = self.content_classifier._fingerprint
        self._degen_guard = DegenerationGuard(self.tok, self.c, self.content_classifier)
        wte_fp32 = self.backbone.input_embedding_weight().to(dev)
        self.bridge.aligner.calibrate(wte_fp32)  # [D] idempotent
        self._wte_normed = F.normalize(wte_fp32.detach(), dim=-1, eps=1e-8)
        self._wte_mean_fp32 = wte_fp32.mean(0).detach().contiguous()
        self.amm.wte_normed = self._wte_normed
        self.amm._content_classifier = self.content_classifier
        amm_ref = self.amm
        def _capture_query_ids(module, args):
            if len(args) >= 1 and isinstance(args[0], torch.Tensor):
                try: amm_ref._last_query_ids = args[0].detach()
                except Exception: amm_ref._last_query_ids = None
            if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                try: amm_ref._last_query_mask = args[1].detach()
                except Exception: amm_ref._last_query_mask = None
        self.backbone.register_forward_pre_hook(_capture_query_ids)
        self._build_wte_neighbor_cache()
        self._compute_filler_centroid()
        self._maybe_load_trained_weights()
        return self

    def _maybe_load_trained_weights(self):
        """Optional hook: if env AMS_TRAINED_WEIGHTS points to a checkpoint written by
        train_v346.py (or any sibling trainer), load non-backbone params/buffers with
        strict=False. Backbone is intentionally excluded — trainer only saves trainables
        + non-backbone buffers (see train_v346.py §5.3). Missing/unexpected keys are
        logged but not fatal, so a partial-shape ckpt fails loud only on shape mismatch.
        """
        path = os.environ.get("AMS_TRAINED_WEIGHTS", "").strip()
        if not path: return
        if not os.path.exists(path):
            print(f"  [AMS_TRAINED_WEIGHTS] file not found: {path} — skipping")
            return
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [AMS_TRAINED_WEIGHTS] torch.load failed: {type(e).__name__}: {e}")
            return
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        if not isinstance(sd, dict):
            print(f"  [AMS_TRAINED_WEIGHTS] unexpected format (no 'state_dict' mapping) — skipping")
            return
        dev = next(self.parameters()).device
        own_params = dict(self.named_parameters())
        own_buffers = dict(self.named_buffers())
        loaded, skipped = 0, 0
        shape_errs = []
        with torch.no_grad():
            for n, t in sd.items():
                if n.startswith("backbone"): skipped += 1; continue
                if n in own_params:
                    p = own_params[n]
                    if p.shape != t.shape:
                        shape_errs.append((n, tuple(p.shape), tuple(t.shape))); continue
                    p.data.copy_(t.to(dev, dtype=p.dtype))
                    loaded += 1
                elif n in own_buffers:
                    b = own_buffers[n]
                    if b.shape != t.shape:
                        shape_errs.append((n, tuple(b.shape), tuple(t.shape))); continue
                    b.data.copy_(t.to(dev, dtype=b.dtype))
                    loaded += 1
                else:
                    skipped += 1
        prov = blob.get("provenance", "?") if isinstance(blob, dict) else "?"
        print(f"  [AMS_TRAINED_WEIGHTS] loaded={loaded} skipped={skipped} "
              f"shape_errs={len(shape_errs)}  path={path}  provenance={prov}")
        if shape_errs:
            for n, s_model, s_ckpt in shape_errs[:5]:
                print(f"    ! shape mismatch {n}: model={s_model} ckpt={s_ckpt}")
            raise RuntimeError(
                f"AMS_TRAINED_WEIGHTS shape mismatch on {len(shape_errs)} tensor(s); "
                f"ckpt not compatible with current SUT shapes")

    def _compute_filler_centroid(self):
        if self.content_classifier is None or self.backbone is None:
            self._filler_centroid = None; return
        wte = self.backbone.input_embedding_weight().to(next(self.parameters()).device)
        V = wte.shape[0]
        filler_ids = sorted(self.content_classifier.filler_ids)
        valid = [t for t in filler_ids if t < V]
        if len(valid) < 3:
            self._filler_centroid = None; return
        filler_vecs = wte[torch.tensor(valid, device=wte.device)]
        centroid = filler_vecs.mean(0)
        self._filler_centroid = F.normalize(centroid, dim=-1, eps=1e-8)

    def _build_wte_neighbor_cache(self):
        if self.backbone is None or self.content_classifier is None: return
        V = self.backbone.vocab_size
        if V > self.c.wte_neighbor_max_vocab:
            self._wte_neighbor_cache = {}
            print(f"  [neighbor cache] vocab_size={V} > {self.c.wte_neighbor_max_vocab}, skip")
            return
        wte_n = self._wte_normed; cc = self.content_classifier
        content_list = sorted(cc.content_ids)
        valid = [t for t in content_list if t < wte_n.shape[0]]
        self._wte_neighbor_cache = {}
        K = self.c.wte_neighbor_k; thresh = self.c.wte_neighbor_threshold
        batch_size = 500
        for start in range(0, len(valid), batch_size):
            batch_ids = valid[start:start+batch_size]
            batch_t = torch.tensor(batch_ids, device=wte_n.device)
            batch_vecs = wte_n[batch_t]
            sims = batch_vecs @ wte_n.T
            topk_vals, topk_ids = sims.topk(K+1, dim=-1)
            for i, tid in enumerate(batch_ids):
                neighbors = []
                for v_val, nid in zip(topk_vals[i], topk_ids[i]):
                    nid_int = nid.item()
                    if nid_int == tid: continue
                    if v_val.item() >= thresh and nid_int in cc.content_ids:
                        neighbors.append(nid_int)
                self._wte_neighbor_cache[tid] = neighbors

    def _expand_content_ids(self, content_ids):
        if not self._wte_neighbor_cache: return content_ids
        expanded = set(content_ids)
        for tid in content_ids:
            neighbors = self._wte_neighbor_cache.get(tid, [])
            expanded.update(neighbors)
        return sorted(expanded)  # [D]

    def _compute_rare_keyword_ids(self, mem, corpus_idf):
        if not corpus_idf: return []
        cc = self.content_classifier
        if cc is None: return []
        candidates = [t for t in mem.content_token_ids
                      if t in cc.strict_content_starter_ids]
        if not candidates:
            candidates = [t for t in mem.content_token_ids if t in cc.content_ids]
        if not candidates: return []
        # [D] 确定性排序
        ranked = sorted(candidates,
                        key=lambda t: (-corpus_idf.get(t, self.c.idf_floor), t))
        return ranked[:self.c.keyword_tail_top_k]

    def _refresh_rare_keyword_indices(self):
        if not self.amm.tree.store: return
        corpus_idf = self.amm._compute_corpus_idf(self.content_classifier)
        if not corpus_idf: return
        for mid in sorted(self.amm.tree.store.keys()):
            mem = self.amm.tree.store[mid]
            mem.rare_keyword_ids = self._compute_rare_keyword_ids(mem, corpus_idf)

    def _check_guidance_active(self, diag) -> bool:
        thresh = self.c.guidance_min_memory_weight
        if not diag or not diag.batch_mem_weights: return False
        for mem_weights in diag.batch_mem_weights:
            for mid, w in mem_weights:
                if w > thresh and mid in self.amm.tree.store:
                    return True
        return False

    def _compute_aggregated_context_descriptors_d_llm(self, diag):
        if not diag or not diag.batch_mem_weights: return None
        K = self.bridge._effective_ctx_slots
        if K == 0: return None
        B = len(diag.batch_mem_weights)
        dev = next(self.parameters()).device
        out_slots = [[] for _ in range(K)]
        any_populated = False
        for b in range(B):
            mw = diag.batch_mem_weights[b]
            mw_sorted = [(mid, w) for mid, w in mw if w > 0
                         and mid in self.amm.tree.store]
            mw_sorted.sort(key=lambda x: (-x[1], x[0]))
            ctx_sum_d_llm = torch.zeros(self.c.d_LLM, device=dev)
            w_sum = 0.0
            for mid, w in mw_sorted:
                mem = self.amm.tree.store[mid]
                if (mem.context_descriptor is not None
                        and self.memory_context_encoder is not None):
                    d_llm_vec = self.memory_context_encoder.decode(
                        mem.context_descriptor.to(dev).float())
                elif mem.semantic_emb is not None:
                    d_llm_vec = mem.semantic_emb.to(dev).float()
                else:
                    continue
                ctx_sum_d_llm = ctx_sum_d_llm + w * d_llm_vec
                w_sum += w
            if w_sum > 1e-6:
                out_slots[0].append(ctx_sum_d_llm / w_sum)
                any_populated = True
            else:
                out_slots[0].append(torch.zeros(self.c.d_LLM, device=dev))
            for k in range(1, K):
                if k < len(mw_sorted):
                    mid, _ = mw_sorted[k]
                elif mw_sorted:
                    mid, _ = mw_sorted[0]
                else:
                    out_slots[k].append(torch.zeros(self.c.d_LLM, device=dev))
                    continue
                mem = self.amm.tree.store[mid]
                if (mem.context_descriptor is not None
                        and self.memory_context_encoder is not None):
                    vec = self.memory_context_encoder.decode(
                        mem.context_descriptor.to(dev).float())
                elif mem.semantic_emb is not None:
                    vec = mem.semantic_emb.to(dev).float()
                else:
                    vec = torch.zeros(self.c.d_LLM, device=dev)
                out_slots[k].append(vec)
        if not any_populated: return None
        return [torch.stack(slot_list) for slot_list in out_slots]

    def _compute_rare_keyword_wte_residual(self, diag, exclude_token_ids: Optional[Set[int]] = None):
        if not self.c.use_wte_residual_tail:
            return None
        n_slots = self.bridge._effective_tail_slots
        if n_slots < 2: return None
        if not diag or not diag.batch_mem_weights: return None
        B = len(diag.batch_mem_weights)
        dev = next(self.parameters()).device
        wte_fp32 = self.backbone.input_embedding_weight().to(dev)
        V_wte = wte_fp32.shape[0]
        wte_mean = (self._wte_mean_fp32.to(dev)
                    if (self.c.wte_residual_centered and self._wte_mean_fp32 is not None)
                    else None)
        exclude = exclude_token_ids if exclude_token_ids else set()
        residual = torch.zeros(B, n_slots, self.c.d_LLM, device=dev)
        any_nonzero = False
        for b in range(B):
            mw = diag.batch_mem_weights[b]
            for slot_idx in range(1, n_slots):
                kw_rank = slot_idx - 1
                kw_weights: Dict[int, float] = {}
                for mid, w in mw:
                    if w <= 0 or mid not in self.amm.tree.store: continue
                    mem = self.amm.tree.store[mid]
                    available = [t for t in mem.rare_keyword_ids
                                 if t not in exclude and t < V_wte]
                    if len(available) > kw_rank:
                        tid = available[kw_rank]
                        kw_weights[tid] = kw_weights.get(tid, 0.0) + w
                if not kw_weights: continue
                ids_sorted = sorted(kw_weights.keys())
                weights = torch.tensor(
                    [kw_weights[t] for t in ids_sorted],
                    device=dev, dtype=wte_fp32.dtype)
                weights = weights / weights.sum().clamp(min=1e-8)
                vecs = wte_fp32[torch.tensor(ids_sorted, device=dev)]
                centroid = (vecs * weights.unsqueeze(1)).sum(0)
                if wte_mean is not None:
                    centroid = centroid - wte_mean
                residual[b, slot_idx, :] = centroid
                any_nonzero = True
        if not any_nonzero: return None
        return residual

    def _compute_mixture_memory_logit(self, fiber_summary, diag, ids, mask):
        if fiber_summary is None: return None
        dev = next(self.parameters()).device
        wte = self.backbone.input_embedding_weight().to(dev)
        base = self.vocab_proj(fiber_summary, wte)
        B = fiber_summary.shape[0]; V = wte.shape[0]
        boost = torch.zeros(B, V, device=dev)
        for b in range(B):
            if b >= len(diag.batch_mem_weights): continue
            for mid, w in diag.batch_mem_weights[b]:
                if w <= 0 or mid not in self.amm.tree.store: continue
                mem = self.amm.tree.store[mid]
                for tid in mem.rare_keyword_ids + mem.content_token_ids[:20]:
                    if tid < V:
                        boost[b, tid] += w
        b_max = boost.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        boost = boost / b_max
        logits_std_base = base.std().clamp(min=1e-3)
        logit_mem = base + boost * logits_std_base * 6.0
        return logit_mem

    def fwd(self, ids, mask, prefix=None):
        out = self.backbone(ids, mask, prefix=prefix)
        if (prefix is None or self.training or self.content_classifier is None):
            return out
        prompt_len = _get_prefix_meta(prefix)
        if prompt_len is None: return out
        step = int(ids.shape[1]) - int(prompt_len)
        if step < 0: return out

        guidance_active = _get_prefix_guidance(prefix)
        content_bias = getattr(prefix, _PREFIX_CONTENT_BIAS_ATTR, None)
        suppression_bias = getattr(prefix, _PREFIX_SUPPRESSION_BIAS_ATTR, None)
        has_biases = (content_bias is not None) or (suppression_bias is not None)

        if not guidance_active and not has_biases:
            return out

        logits = out['logits']; dev = logits.device
        V_lg = logits.shape[-1]
        last = logits[:, -1:, :].clone()
        mod_last = False
        cc = self.content_classifier

        if guidance_active:
            if (self.c.use_fwd_path_hard_mask
                    and self.c.use_early_content_starter_hard_mask
                    and step < self.c.early_starter_hard_mask_steps):
                starter_mask = cc.content_starter_mask(dev)
                V = min(V_lg, starter_mask.shape[0])
                mask_val = float(self.c.fwd_path_hard_mask_value)
                mask_bool = starter_mask[:V].bool().view(1, 1, V)
                last_V = last[:, :, :V]
                last[:, :, :V] = torch.where(
                    mask_bool, last_V, torch.full_like(last_V, mask_val))
                mod_last = True

            if self.c.use_fwd_function_suppression and cc is not None:
                logits_std_fs = logits.std().item()
                eos_id = self.tok.eos_token_id
                fn_mask = cc.pure_function_mask(dev, eos_id=eos_id)
                V_fn = min(V_lg, fn_mask.shape[0])
                step_scale_fn = max(self.c.fwd_function_suppression_floor,
                                    1.0 - step * self.c.fwd_function_suppression_decay)
                unit_fn = (logits_std_fs * self.c.content_bias_std_multiplier
                           if self.c.use_adaptive_content_bias_scale else 1.0)
                fs_dampen = (self.c.fwd_path_bias_dampen
                             if self.c.fwd_function_suppression_apply_dampen else 1.0)
                scale_fn = (unit_fn * self.c.fwd_function_suppression_scale
                            * step_scale_fn * fs_dampen)
                last[:, 0, :V_fn] = last[:, 0, :V_fn] - fn_mask[:V_fn].to(dev) * scale_fn
                mod_last = True

            if self.c.use_no_repeat_bigram and step >= 2:
                B = ids.shape[0]
                pen = self.c.no_repeat_bigram_penalty
                for b in range(B):
                    gen_ids_b = ids[b, int(prompt_len):].tolist()
                    if len(gen_ids_b) < 2: continue
                    last_tok = gen_ids_b[-1]
                    penalize_nexts = set()
                    for i in range(len(gen_ids_b) - 1):
                        if gen_ids_b[i] == last_tok:
                            penalize_nexts.add(gen_ids_b[i + 1])
                    if penalize_nexts:
                        pen_ids = [t for t in penalize_nexts if 0 <= t < V_lg]
                        if pen_ids:
                            pen_t = torch.tensor(pen_ids, device=dev, dtype=torch.long)
                            last[b, 0, pen_t] = last[b, 0, pen_t] - pen
                            mod_last = True

        if self.c.use_fwd_path_content_bias and has_biases:
            logits_std = logits.std().item()
            dampen = self.c.fwd_path_bias_dampen
            if content_bias is not None:
                step_scale = max(self.c.content_bias_floor,
                                 1.0 - step * self.c.content_bias_decay)
                unit = (logits_std * self.c.content_bias_std_multiplier
                        if self.c.use_adaptive_content_bias_scale else 1.0)
                V = min(V_lg, content_bias.shape[-1])
                cb = content_bias[:, :V].to(dev)
                scale = unit * self.c.content_bias_scale * step_scale * dampen
                last[:, 0, :V] = last[:, 0, :V] + cb * scale
                mod_last = True
            if suppression_bias is not None and self.c.use_memory_guided_suppression:
                step_scale_sup = max(self.c.suppression_floor,
                                     1.0 - step * self.c.suppression_decay)
                unit_sup = (logits_std * self.c.suppression_std_multiplier
                            if self.c.use_adaptive_content_bias_scale else 1.0)
                V = min(V_lg, suppression_bias.shape[-1])
                sb = suppression_bias[:, :V].to(dev)
                scale_sup = unit_sup * self.c.suppression_bias_scale * step_scale_sup * dampen
                last[:, 0, :V] = last[:, 0, :V] - sb * scale_sup
                mod_last = True

        if mod_last:
            new_logits = logits.clone()
            new_logits[:, -1:, :] = last
            out['logits'] = new_logits
        return out

    def _compute_content_semantic_emb(self, hidden_states, ids, mask):
        B, T, D = hidden_states.shape
        cc = self.content_classifier
        result = []
        for b in range(B):
            content_positions = []
            T_valid = min(T, ids.shape[1]) if ids is not None else T
            for pos in range(T_valid):
                if mask is not None and mask.shape[1] > pos and mask[b, pos].item() == 0:
                    continue
                if ids is not None:
                    tid = ids[b, pos].item()
                    if cc is not None and tid in cc.content_ids:
                        content_positions.append(min(pos, T-1))
            if content_positions:
                pos_t = torch.tensor(content_positions, device=hidden_states.device)
                content_hs = hidden_states[b, pos_t]
                result.append(content_hs.mean(0))
            else:
                if mask is not None:
                    valid_len = min(int(mask[b].sum().item()), T); valid_len = max(valid_len, 1)
                    result.append(hidden_states[b, :valid_len].mean(0))
                else: result.append(hidden_states[b].mean(0))
        return torch.stack(result)

    def _extract_content_hidden_per_b(self, hidden_states, ids, mask):
        """[A] 为 MemoryContextEncoder 提供逐条 content-position hidden states。"""
        B, T, D = hidden_states.shape
        cc = self.content_classifier
        out_list = []
        for b in range(B):
            positions = []
            T_valid = min(T, ids.shape[1]) if ids is not None else T
            for pos in range(T_valid):
                if mask is not None and mask.shape[1] > pos and mask[b, pos].item() == 0:
                    continue
                if ids is not None:
                    tid = ids[b, pos].item()
                    if cc is not None and tid in cc.content_ids:
                        positions.append(min(pos, T - 1))
            if not positions:
                if mask is not None:
                    valid_len = min(int(mask[b].sum().item()), T); valid_len = max(valid_len, 1)
                    out_list.append(hidden_states[b, :valid_len])
                else:
                    out_list.append(hidden_states[b])
            else:
                pos_t = torch.tensor(positions, device=hidden_states.device)
                out_list.append(hidden_states[b, pos_t])
        return out_list

    def extract_state(self, hs, mask=None, pl=0):
        pooled = self.layer_pool(hs)
        if pl > 0: pooled = pooled[:, pl:]
        m = mask[:, pl:] if mask is not None and pl > 0 else mask
        if m is not None and m.shape[1] != pooled.shape[1]: m = None
        xq, fq = self.bridge.ext(pooled, m)
        return pooled, xq, fq

    def _build_token_bias_from_memories(self, mem_weight_list, q_content_ids,
                                        corpus_idf=None, relevance_floor=None):
        """[E] 允许按 call site 指定 relevance_floor(top-1 vs rest)。"""
        V = self.c.vocab_size; dev = next(self.parameters()).device
        cc = self.content_classifier; wte_n = self._wte_normed
        floor = (relevance_floor if relevance_floor is not None
                 else self.c.content_bias_relevance_floor)
        concentration = self.c.content_bias_concentration
        bias = torch.zeros(V, device=dev)
        q_valid = sorted([i for i in q_content_ids if i < wte_n.shape[0]])
        q_vecs = wte_n[q_valid] if q_valid else None
        use_idf = (self.c.use_idf_content_bias and corpus_idf is not None
                   and len(corpus_idf) > 0)
        max_boost = self.c.idf_bias_max_boost
        idf_floor = self.c.idf_floor
        for mid, weight in mem_weight_list:
            if mid not in self.amm.tree.store or weight <= 0: continue
            mem = self.amm.tree.store[mid]
            scoring_ids = self.amm._get_mem_scoring_ids(mem)
            if cc is not None and self.c.use_word_starter_filter:
                valid_ids = sorted([t for t in scoring_ids if t < V and t < wte_n.shape[0]
                                    and t in cc.content_starter_ids])
            elif cc is not None:
                valid_ids = sorted([t for t in scoring_ids if t < V and t < wte_n.shape[0]
                                    and t in cc.content_ids])
            else: valid_ids = []
            if not valid_ids: continue
            if q_valid and q_vecs is not None:
                m_vecs = wte_n[valid_ids]; sim = m_vecs @ q_vecs.T
                relevance = sim.max(dim=1).values.clamp(min=0)
                relevance = relevance.pow(concentration)
                relevance = relevance * (1.0 - floor) + floor
                for i, tid in enumerate(valid_ids):
                    idf_val = (max(idf_floor, min(max_boost, corpus_idf.get(tid, idf_floor)))
                               if use_idf else 1.0)
                    bias[tid] += weight * relevance[i].item() * idf_val
            else:
                for tid in valid_ids:
                    idf_val = (max(idf_floor, min(max_boost, corpus_idf.get(tid, idf_floor)))
                               if use_idf else 1.0)
                    bias[tid] += weight * idf_val
        return bias

    def _build_content_bias(self, diag, query_content_ids_per_batch):
        """[E] top-1 专属 + top-k 兜底。"""
        V = self.c.vocab_size; dev = next(self.parameters()).device
        B = len(diag.batch_mem_weights)
        bias = torch.zeros(B, V, device=dev)
        cc = self.content_classifier
        corpus_idf = None
        if self.c.use_idf_content_bias and cc is not None:
            corpus_idf = self.amm._compute_corpus_idf(cc)
        use_top1 = self.c.use_top1_exclusive_content_bias
        for b, mem_weights in enumerate(diag.batch_mem_weights):
            q_ids = (query_content_ids_per_batch[b]
                     if query_content_ids_per_batch and b < len(query_content_ids_per_batch)
                     else [])
            reweighted = [(mid, w * (diag.per_memory_bidi_min.get(mid, 0.5) ** 2))
                          for mid, w in mem_weights]
            if not reweighted:
                continue
            if use_top1 and len(reweighted) >= 1:
                reweighted.sort(key=lambda x: (-x[1], x[0]))
                top1 = reweighted[:1]
                rest = reweighted[1:]
                b1 = self._build_token_bias_from_memories(
                    top1, q_ids, corpus_idf,
                    relevance_floor=self.c.top1_relevance_floor)
                if rest:
                    br = self._build_token_bias_from_memories(
                        rest, q_ids, corpus_idf,
                        relevance_floor=self.c.rest_relevance_floor)
                else:
                    br = torch.zeros_like(b1)
                b_bias = (self.c.top1_content_bias_weight * b1
                          + self.c.rest_content_bias_weight * br)
            else:
                b_bias = self._build_token_bias_from_memories(
                    reweighted, q_ids, corpus_idf)
            bmax = b_bias.max()
            if bmax > 1e-8: bias[b] = b_bias / bmax
        return bias

    def _build_suppression_bias(self, diag, query_content_ids_per_batch):
        V = self.c.vocab_size; dev = next(self.parameters()).device
        B = len(diag.batch_mem_weights)
        suppression = torch.zeros(B, V, device=dev)
        cc = self.content_classifier
        if cc is None: return suppression
        corpus_idf = None
        if self.c.use_idf_content_bias:
            corpus_idf = self.amm._compute_corpus_idf(cc)
        for b in range(B):
            dom_mid = diag.dominant_per_batch[b] if b < len(diag.dominant_per_batch) else None
            nd_mids = (diag.non_dominant_per_batch[b]
                       if b < len(diag.non_dominant_per_batch) else [])
            nd_weights = (diag.non_dominant_weights_per_batch[b]
                          if b < len(diag.non_dominant_weights_per_batch) else {})
            if not nd_mids: continue
            dom_token_set = set()
            if dom_mid is not None and dom_mid in self.amm.tree.store:
                dom_mem = self.amm.tree.store[dom_mid]
                for t in self.amm._get_mem_scoring_ids(dom_mem):
                    if t in cc.content_ids: dom_token_set.add(t)
            q_ids = (query_content_ids_per_batch[b]
                     if query_content_ids_per_batch and b < len(query_content_ids_per_batch)
                     else [])
            nd_mem_weights = sorted(
                [(mid, nd_weights.get(mid, 0.0)) for mid in nd_mids],
                key=lambda x: x[0])
            nd_bias = self._build_token_bias_from_memories(nd_mem_weights, q_ids, corpus_idf)
            for t in dom_token_set:
                if 0 <= t < V: nd_bias[t] = 0.0
            nmax = nd_bias.max()
            if nmax > 1e-8: suppression[b] = nd_bias / nmax
        return suppression

    def _get_prefix(self, hs, mask=None, pl=0, update_stats=True, return_extra=False, ids=None,
                    exclude_token_ids: Optional[Set[int]] = None):
        pooled, xq, fq = self.extract_state(hs, mask, pl)
        trimmed_mask = mask[:, pl:] if mask is not None and pl > 0 else mask
        if trimmed_mask is not None and pooled.shape[1] != trimmed_mask.shape[1]:
            trimmed_mask = None
        query_content_ids_per_batch = []
        if ids is not None and self.content_classifier is not None:
            for b in range(ids.shape[0]):
                b_ids = ids[b].tolist()
                b_exact = sorted(set(self.content_classifier.get_content_ids_from_tokens(b_ids)))
                query_content_ids_per_batch.append(b_exact)
        query_sem = (self._compute_content_semantic_emb(pooled, ids, trimmed_mask)
                     if ids is not None and self.content_classifier is not None
                     else pooled.mean(1))
        wte_n = self._wte_normed
        fibers, mem_mask, fiber_summary, diag = self.amm.retrieve_multi(
            xq, fq, update_stats=update_stats,
            query_semantic_emb=query_sem,
            query_content_ids_per_batch=query_content_ids_per_batch,
            wte_normed=wte_n, content_classifier=self.content_classifier)

        ctx_descriptors_d_llm = (self._compute_aggregated_context_descriptors_d_llm(diag)
                                 if self.c.use_context_descriptor else None)
        rare_residual = self._compute_rare_keyword_wte_residual(
            diag, exclude_token_ids=exclude_token_ids)

        prefix = self.bridge.inject(
            fibers, mem_mask, fiber_summary=fiber_summary,
            filler_centroid=self._filler_centroid,
            context_descriptors_d_llm=ctx_descriptors_d_llm,
            rare_keyword_wte_residual=rare_residual)

        prompt_len_for_meta = (mask.shape[1] if mask is not None
                               else (ids.shape[1] if ids is not None else hs.shape[1]))
        _set_prefix_meta(prefix, prompt_len_for_meta)

        if not self.training:
            guidance = self._check_guidance_active(diag)
            _set_prefix_guidance(prefix, guidance)
        else:
            guidance = False
            _set_prefix_guidance(prefix, False)

        if return_extra:
            content_bias = self._build_content_bias(diag, query_content_ids_per_batch)
            suppression_bias = (self._build_suppression_bias(diag, query_content_ids_per_batch)
                                if self.c.use_memory_guided_suppression
                                else torch.zeros_like(content_bias))
            if self.c.use_fwd_path_content_bias and guidance:
                _set_prefix_biases(prefix, content_bias, suppression_bias)
            return prefix, fiber_summary, diag, content_bias, suppression_bias

        if not self.training and guidance and self.c.use_fwd_path_content_bias:
            with torch.no_grad():
                cb = self._build_content_bias(diag, query_content_ids_per_batch)
                sb = (self._build_suppression_bias(diag, query_content_ids_per_batch)
                      if self.c.use_memory_guided_suppression else None)
            _set_prefix_biases(prefix, cb, sb)
        return prefix

    def _build_contrastive_uncond_prefix(self, diag, prefix_cond, prompt_len_for_meta=None,
                                          content_bias=None, suppression_bias=None):
        dev = prefix_cond.device; B = prefix_cond.shape[0]
        non_dom_fibers = []; have_contrast = []
        for b in range(B):
            mids = diag.non_dominant_per_batch[b] if b < len(diag.non_dominant_per_batch) else []
            mids = [m for m in mids if m in self.amm.tree.store]
            if mids:
                mids = sorted(mids)
                fvecs = torch.stack([self.amm.tree.store[m].fiber.to(dev) for m in mids])
                non_dom_fibers.append(fvecs.mean(0)); have_contrast.append(True)
            else:
                non_dom_fibers.append(torch.zeros(self.c.d_F, device=dev)); have_contrast.append(False)
        non_dom_fibers_t = torch.stack(non_dom_fibers, dim=0)
        uncond_prefix = torch.zeros_like(prefix_cond)
        for b in range(B):
            if have_contrast[b]:
                single = non_dom_fibers_t[b:b+1].unsqueeze(1)
                mask_one = torch.ones(1, 1, device=dev)
                pref_b = self.bridge.inject(
                    single, mask_one, fiber_summary=non_dom_fibers_t[b:b+1],
                    filler_centroid=self._filler_centroid,
                    context_descriptors_d_llm=None,
                    rare_keyword_wte_residual=None,
                    is_cond_path=False)
                uncond_prefix[b:b+1] = pref_b
            else:
                uncond_prefix[b:b+1] = self.bridge.build_neutral_prefix(1, dev)
        if prompt_len_for_meta is not None:
            _set_prefix_meta(uncond_prefix, prompt_len_for_meta)
        _set_prefix_guidance(uncond_prefix, False)
        if content_bias is not None:
            _set_prefix_biases(uncond_prefix, content_bias, suppression_bias)
        return uncond_prefix

    def _compute_vocab_bias(self, fiber_summary):
        if fiber_summary is None: return None
        wte = self.backbone.input_embedding_weight().to(fiber_summary.device)
        return self.vocab_proj(fiber_summary, wte)

    def _collect_exclude_ids(self, ids):
        if self.content_classifier is None: return set()
        exclude = set()
        for b in range(ids.shape[0]):
            b_ids = ids[b].tolist()
            b_content = self.content_classifier.get_content_ids_from_tokens(b_ids)
            exclude.update(b_content)
        return exclude

    def prepare_decode_context(self, ids, mask, update_stats=False,
                                mixture_ceiling_override: Optional[float] = None):
        prompt_len = ids.shape[1]
        exclude_ids = self._collect_exclude_ids(ids)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            prefix_cond, fs, diag, cb, sb = self._get_prefix(
                o['hs'], mask, update_stats=update_stats, return_extra=True, ids=ids,
                exclude_token_ids=exclude_ids)
            vb = self._compute_vocab_bias(fs)
            if self.c.use_cfg_decoding:
                if self.c.use_contrastive_memory_cfg:
                    sym_cb = cb if self.c.apply_content_bias_symmetric_cfg else None
                    sym_sb = sb if self.c.apply_content_bias_symmetric_cfg else None
                    prefix_uncond = self._build_contrastive_uncond_prefix(
                        diag, prefix_cond, prompt_len_for_meta=prompt_len,
                        content_bias=sym_cb, suppression_bias=sym_sb)
                else:
                    B = prefix_cond.shape[0]
                    prefix_uncond = self.bridge.build_neutral_prefix(B, prefix_cond.device)
                    _set_prefix_meta(prefix_uncond, prompt_len)
                    _set_prefix_guidance(prefix_uncond, False)
                    if self.c.apply_content_bias_symmetric_cfg:
                        _set_prefix_biases(prefix_uncond, cb, sb)
            else:
                prefix_uncond = None
            mixture_gate = None; memory_logit_bias = None
            if self.c.use_mixture_decoding and self.mixture_gate_head is not None:
                mixture_gate = self.mixture_gate_head(
                    fs, override_ceiling=mixture_ceiling_override)
                memory_logit_bias = self._compute_mixture_memory_logit(fs, diag, ids, mask)
        return DecodeContext(
            prefix_cond=prefix_cond, prefix_uncond=prefix_uncond,
            fiber_summary=fs, diag=diag,
            content_bias=cb, suppression_bias=sb, vocab_bias=vb,
            mixture_gate=mixture_gate, memory_logit_bias=memory_logit_bias)

    def shape_step_logits(self, logits_cond, logits_uncond, step,
                          content_bias, suppression_bias, vocab_bias, state,
                          mixture_gate=None, memory_logit_bias=None):
        c = self.c; dev = logits_cond.device; cc = self.content_classifier
        HARD_MASK = -1e9

        if (c.use_mixture_decoding and mixture_gate is not None
                and memory_logit_bias is not None):
            V_mem = memory_logit_bias.shape[-1]
            V_cond = logits_cond.shape[-1]
            V_min = min(V_mem, V_cond)
            g = mixture_gate.view(-1, 1)
            mixed = logits_cond.clone()
            mixed[:, :V_min] = ((1.0 - g) * logits_cond[:, :V_min]
                                + g * memory_logit_bias[:, :V_min])
            lg_base = mixed
        else:
            lg_base = logits_cond

        if c.use_cfg_decoding and logits_uncond is not None:
            alpha = c.cfg_scale
            if c.cfg_decay_steps > 0:
                alpha *= max(0.0, 1.0 - step / c.cfg_decay_steps)
            lg = lg_base + alpha * (lg_base - logits_uncond)
        else:
            lg = lg_base.clone()

        V_lg = lg.shape[-1]
        step_scale_learned = max(c.semantic_boost_floor, 1.0 - step * c.semantic_boost_decay)
        if vocab_bias is not None:
            V2 = min(V_lg, vocab_bias.shape[-1])
            lg[:, :V2] = lg[:, :V2] + vocab_bias[:, :V2] * c.semantic_boost_scale * step_scale_learned

        if c.use_decode_functional_suppression and cc is not None:
            eos_id = self.tok.eos_token_id
            pure_func_mask = cc.pure_function_mask(dev, eos_id=eos_id)
            V_pf = min(V_lg, pure_func_mask.shape[0])
            starter_mask = cc.content_starter_mask(dev)
            V_sm = min(V_lg, starter_mask.shape[0])
            step_scale_fs = max(c.decode_fs_floor, 1.0 - step * c.decode_fs_decay)
            pf_bool = pure_func_mask[:V_pf].bool()
            sm_bool = starter_mask[:V_sm].bool()
            B_lg = lg.shape[0]
            for b in range(B_lg):
                row = lg[b, :V_pf]
                sm_row = lg[b, :V_sm]
                func_vals = torch.where(pf_bool, row, torch.full_like(row, -1e9))
                star_vals = torch.where(sm_bool, sm_row, torch.full_like(sm_row, -1e9))
                top_func = func_vals.max().item()
                top_star = star_vals.max().item()
                if top_func > -1e8 and top_star > -1e8:
                    deficit = top_func - top_star + c.decode_fs_margin
                    if deficit > 0:
                        penalty = c.decode_fs_scale * step_scale_fs * deficit
                        lg[b, :V_pf] = torch.where(
                            pf_bool, lg[b, :V_pf] - penalty, lg[b, :V_pf])

        if cc:
            for tid, count in state.generated_content_counts.items():
                if tid in cc.content_ids and tid < V_lg:
                    scaled_count = count ** c.content_repeat_exponent
                    lg[0, tid] -= c.content_repeat_penalty * scaled_count

        if c.use_cyclic_content_hard_mask and cc is not None:
            window = c.cyclic_content_window; max_cnt = c.cyclic_content_max_count
            window_counts = {}; cutoff_step = step - window
            for (step_idx, tid) in state.content_history:
                if step_idx >= cutoff_step:
                    window_counts[tid] = window_counts.get(tid, 0) + 1
            for tid, cnt in window_counts.items():
                if cnt >= max_cnt and 0 <= tid < V_lg:
                    lg[0, tid] = HARD_MASK
        if c.use_ngram_repeat_block and len(state.generated_ids) >= 4:
            max_n = min(c.ngram_repeat_max_n, len(state.generated_ids) // 2)
            for n in range(2, max_n + 1):
                if len(state.generated_ids) >= 2 * n:
                    tail = state.generated_ids[-n:]
                    prev = state.generated_ids[-2 * n:-n]
                    if tail == prev:
                        expected_next = state.generated_ids[-n]
                        if 0 <= expected_next < V_lg:
                            lg[0, expected_next] -= c.ngram_repeat_penalty
        if c.use_no_repeat_bigram and len(state.generated_ids) >= 2:
            last_tok = state.generated_ids[-1]
            penalize_nexts = set()
            for i in range(len(state.generated_ids) - 1):
                if state.generated_ids[i] == last_tok:
                    penalize_nexts.add(state.generated_ids[i + 1])
            for next_tok in penalize_nexts:
                if 0 <= next_tok < V_lg:
                    lg[0, next_tok] -= c.no_repeat_bigram_penalty
        if cc and self._wte_neighbor_cache and state.recent_starters:
            for prev_tid, _ in state.recent_starters:
                neighbors = self._wte_neighbor_cache.get(prev_tid, [])
                for nid in neighbors:
                    if nid in cc.word_starter_ids: continue
                    if nid < V_lg: lg[0, nid] -= c.bpe_echo_penalty
        if cc and state.generated_ids and state.generated_ids[-1] in cc.content_starter_ids:
            for tid in cc.content_ids:
                if tid not in cc.word_starter_ids and tid < V_lg:
                    lg[0, tid] -= c.post_starter_nonstarter_penalty
        newline_ids_set = cc.newline_ids if cc is not None else set()
        if c.use_newline_hard_gate and cc is not None:
            content_count_so_far = sum(state.generated_content_counts.values())
            hard_gate_active = (step < c.newline_hard_gate_min_step
                                or content_count_so_far < c.newline_hard_gate_min_content)
            if hard_gate_active:
                for nid in newline_ids_set:
                    if nid < V_lg: lg[0, nid] = HARD_MASK
        eos_token_id = self.tok.eos_token_id
        if (c.use_eos_hard_mask and eos_token_id is not None
                and step < c.eos_hard_mask_steps and eos_token_id < V_lg):
            lg[0, eos_token_id] = HARD_MASK
        if c.use_content_gated_newline and cc is not None:
            content_count_so_far = sum(state.generated_content_counts.values())
            if content_count_so_far < c.min_content_tokens_before_newline:
                for nid in newline_ids_set:
                    if nid < V_lg: lg[0, nid] -= c.late_newline_penalty
        if (c.use_early_content_starter_hard_mask and cc is not None
                and step < c.early_starter_hard_mask_steps):
            starter_mask = cc.content_starter_mask(dev)[:V_lg]
            lg[0, :V_lg] = torch.where(
                starter_mask.bool(), lg[0, :V_lg],
                torch.full_like(lg[0, :V_lg], HARD_MASK))
        if self._degen_guard is not None:
            lg = self._degen_guard.process(lg, state.generated_ids, step)
        return lg

    def write(self, text, training_mode=False):
        tk = self.tok(text, return_tensors='pt', padding=True, truncation=True)
        ids, mask = tk['input_ids'], tk['attention_mask']
        dev = next(self.parameters()).device; ids, mask = ids.to(dev), mask.to(dev)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            hs_pooled = self.layer_pool(o['hs'])
        surp = self.amm.surprise_proxy(o['logits'][:, :-1], ids[:, 1:])
        pooled_mean = hs_pooled.mean(1)
        content_sem = self._compute_content_semantic_emb(hs_pooled, ids, mask)
        # [A] 为 attention-pool encoder 准备逐条 content hidden
        content_hidden_per_b = self._extract_content_hidden_per_b(hs_pooled, ids, mask)
        raw_ids = self.tok.encode(text); cc = self.content_classifier
        content_ids = sorted(set(cc.get_content_ids_from_tokens(raw_ids))) if cc else []
        strict_ids = sorted(set(cc.get_strict_starter_ids_from_tokens(raw_ids))) if cc else []
        expanded_ids = sorted(self._expand_content_ids(content_ids))
        wte_fp32 = self.backbone.input_embedding_weight().to(dev)
        stored = 0; gate_vals = []
        for b in range(ids.shape[0]):
            with torch.no_grad():
                gate = self.amm.write_gate(pooled_mean[b:b+1], surp[b:b+1]).item()
            gate_vals.append(gate)
            if training_mode or gate >= self.c.write_gate_threshold:
                ctx_desc = None
                if self.memory_context_encoder is not None:
                    with torch.no_grad():
                        hidden_states_b = content_hidden_per_b[b]
                        if self.c.context_encoder_source == "wte_strict_starter":
                            src_ids = strict_ids if strict_ids else content_ids
                            ctx_desc = self.memory_context_encoder.encode_from_tokens(
                                src_ids, wte_fp32,
                                hidden_states=hidden_states_b, hidden_mask=None)
                        else:
                            ctx_desc = self.memory_context_encoder.encode_from_hidden(
                                content_sem[b]).detach().contiguous()
                self.amm.store_mem(pooled_mean[b], surp[b], training_mode,
                    source_text=text, content_token_ids=content_ids,
                    content_semantic_emb=content_sem[b],
                    expanded_content_ids=expanded_ids,
                    context_descriptor=ctx_desc,
                    strict_starter_ids=strict_ids)
                stored += 1
        # [C] 触发 re-cluster
        self.amm.maybe_recluster(force=False)
        # [D v3.45] rare_keyword_ids 的生成时机统一:write() 末尾刷新,与
        # load_memory() 的行为对齐.  修正 4.13 的已识别不等价源 --- fresh
        # path 和 load path 在此时机上产生的 rare_keyword_ids 必须完全相同,
        # 否则 _compute_rare_keyword_wte_residual 一侧返回 None,
        # 另一侧返回非零张量,generate() 输出字面值不同.
        if stored > 0:
            self._refresh_rare_keyword_indices()
        return stored, gate_vals

    def _refresh_all_memories(self):
        entries = [self.amm.tree.store[mid] for mid in sorted(self.amm.tree.store.keys())]
        texts = [e.source_text for e in entries if e.source_text]
        if not texts: return 0
        unique_texts = list(dict.fromkeys(texts))
        self.amm.tree.store.clear()
        self.amm.tree.root = _Node()
        self.amm.tree.nid = 0; self.amm.time = 0
        self.amm._writes_since_recluster = 0
        for text in unique_texts: self.write(text, training_mode=True)
        self.amm.maybe_recluster(force=True)
        self._refresh_rare_keyword_indices()
        return len(unique_texts)

    def _prep_prompt_ids(self, prompt):
        if self.c.use_chat_template_for_gen and self.backbone.has_chat_template:
            prompt = self.backbone.build_chat_text(prompt)
        tk = self.tok(prompt, return_tensors='pt')
        return tk['input_ids'], tk['attention_mask']

    def generate(self, prompt, mt=50, greedy=False):
        ids, mask = self._prep_prompt_ids(prompt)
        dev = next(self.parameters()).device
        ids = ids.to(dev); mask = mask.to(dev)
        # [F] circuit breaker
        breaker = CircuitBreaker(self.c) if self.c.use_circuit_breaker else None
        ctx = self.prepare_decode_context(
            ids, mask, update_stats=False,
            mixture_ceiling_override=None)
        state = DecodeState(); prompt_len = ids.shape[1]
        for i in range(mt):
            if i > 0 and i % self.c.retrieval_interval == 0:
                override = (breaker.effective_ceiling(self.c.mixture_gate_ceiling)
                            if breaker is not None and breaker.active else None)
                ctx = self.prepare_decode_context(
                    ids, mask, update_stats=False,
                    mixture_ceiling_override=override)
            with torch.no_grad():
                o_cond = self.fwd(ids, mask, ctx.prefix_cond)
                lg_cond = o_cond['logits'][:, -1:].squeeze(1)
                if self.c.use_cfg_decoding and ctx.prefix_uncond is not None:
                    o_uncond = self.fwd(ids, mask, ctx.prefix_uncond)
                    lg_uncond = o_uncond['logits'][:, -1:].squeeze(1)
                else:
                    lg_uncond = None
                lg = self.shape_step_logits(
                    lg_cond, lg_uncond, i,
                    ctx.content_bias, ctx.suppression_bias, ctx.vocab_bias, state,
                    mixture_gate=ctx.mixture_gate,
                    memory_logit_bias=ctx.memory_logit_bias)
                if greedy:
                    nxt = lg.argmax(-1, keepdim=True)
                else:
                    lg_t = lg / self.c.gen_temp; p = F.softmax(lg_t, -1)
                    sp, si = torch.sort(p, descending=True); cs = torch.cumsum(sp, -1)
                    rm = cs - sp > self.c.gen_top_p; sp[rm] = 0
                    total = sp.sum(-1, keepdim=True)
                    if (total < 1e-10).any(): sp[:, 0] = 1.0; total = sp.sum(-1, keepdim=True)
                    sp = sp / total; nxt = si.gather(-1, torch.multinomial(sp, 1))
                # [F] 记录 -log P(chosen)
                chosen_id = int(nxt.item())
                log_probs = F.log_softmax(lg, dim=-1)
                chosen_nll = float(-log_probs[0, chosen_id].item())
            nxt_id = chosen_id
            if nxt_id == self.tok.eos_token_id and len(state.generated_ids) >= self.c.degen_min_tokens:
                break
            state.update(nxt_id, i, self.content_classifier,
                         self.c.bpe_echo_window, self.c.cyclic_content_window,
                         nll=chosen_nll)
            if breaker is not None:
                if breaker.baseline is None:
                    breaker.set_baseline_from(state.token_nll_history)
                else:
                    breaker.update(state.token_nll_history)
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)
        new_ids = ids[0, prompt_len:].tolist()
        gen_text = self.tok.decode(new_ids, skip_special_tokens=True)
        return prompt + gen_text if not self.c.use_chat_template_for_gen else gen_text

    def save_memory(self, path):
        """[D] 确定性序列化:mid 排序,字段顺序稳定,SHA256 指纹。"""
        data = {'store': {}, 'nid': self.amm.tree.nid, 'time': self.amm.time,
                'classifier_fingerprint': self._classifier_fingerprint,
                'aligner_target_std': float(self.bridge.aligner._target_std.item()),
                'aligner_scale_logit': float(self.bridge.aligner.scale_logit.item())}
        def _ser(t):
            if t is None: return None
            return t.detach().cpu().clone().contiguous()
        sorted_mids = sorted(self.amm.tree.store.keys())
        ordered = {}
        h = hashlib.sha256()
        for mid in sorted_mids:
            m = self.amm.tree.store[mid]
            entry = {
                'base': _ser(m.base), 'fiber': _ser(m.fiber), 'dirn': _ser(m.dirn),
                'surprise': m.surprise, 'ts': m.ts, 'last': m.last,
                'cnt': m.cnt, 'version': m.version,
                'source_text': m.source_text,
                'content_token_ids': sorted(list(m.content_token_ids)),
                'expanded_content_ids': sorted(list(m.expanded_content_ids)),
                'rare_keyword_ids': list(m.rare_keyword_ids),
                'strict_starter_ids': sorted(list(m.strict_starter_ids)),
                'semantic_emb': _ser(m.semantic_emb),
                'context_descriptor': _ser(m.context_descriptor),
                'cluster_id': int(m.cluster_id)}
            ordered[mid] = entry
            h.update(str(mid).encode()); h.update(b'|')
            h.update(m.source_text.encode()); h.update(b'|')
            for k_in in ['content_token_ids', 'strict_starter_ids']:
                for t in entry[k_in]:
                    h.update(str(int(t)).encode()); h.update(b',')
                h.update(b';')
        data['store'] = ordered
        data['fingerprint'] = h.hexdigest()
        torch.save(data, path)

    def load_memory(self, path):
        data = torch.load(path, weights_only=False)
        # [D] 验证分类器指纹
        saved_fp = data.get('classifier_fingerprint', None)
        if (saved_fp is not None and self._classifier_fingerprint is not None
                and saved_fp != self._classifier_fingerprint):
            raise RuntimeError(
                f"ContentTokenClassifier fingerprint mismatch: "
                f"saved={saved_fp[:16]} cur={self._classifier_fingerprint[:16]}. "
                f"Tokenizer/vocab changed.")
        self.amm.tree.store.clear(); self.amm.tree.root = _Node()
        self.amm.tree.nid = data['nid']; self.amm.time = data['time']
        self.amm._writes_since_recluster = 0
        dev = next(self.parameters()).device
        def _load(t):
            if t is None: return None
            return t.detach().to(dev).clone().contiguous()
        sorted_mids = sorted(data['store'].keys())
        for mid in sorted_mids:
            d = data['store'][mid]
            m = MemEntry(mid=mid,
                base=_load(d['base']), fiber=_load(d['fiber']), dirn=_load(d['dirn']),
                surprise=d['surprise'], ts=d['ts'],
                last=d['last'], cnt=d['cnt'], version=d['version'],
                source_text=d.get('source_text', ''),
                content_token_ids=sorted(list(d.get('content_token_ids', []))),
                expanded_content_ids=sorted(list(d.get('expanded_content_ids', []))),
                rare_keyword_ids=list(d.get('rare_keyword_ids', [])),
                strict_starter_ids=sorted(list(d.get('strict_starter_ids', []))),
                semantic_emb=_load(d.get('semantic_emb', None)),
                context_descriptor=_load(d.get('context_descriptor', None)),
                cluster_id=int(d.get('cluster_id', -1)))
            self.amm.tree.insert(m)
        self._refresh_rare_keyword_indices()

class Trainer:
    def __init__(self, m, c):
        self.m = m; self.c = c
        ps = [p for n, p in m.named_parameters() if p.requires_grad and 'backbone' not in n]
        self.opt = torch.optim.AdamW(ps, lr=1e-4, weight_decay=0.01)
        self.warmup = LossWarmup({
            'semantic_probe': c.warmup_steps_probe, 'dir_diversity': c.warmup_steps_dd,
            'reranker_ranking': c.warmup_steps_rr, 'vocab_anchor': c.warmup_steps_va,
            'semantic_alignment': c.warmup_steps_sa,
            'tail_semantic_anchor': c.warmup_steps_tsa,
            'functional_suppression': c.warmup_steps_fs,
            'context_separation': c.warmup_steps_ctx_sep,
            'slot_residual_alignment': c.warmup_steps_sra,
            'inter_domain_margin': c.warmup_steps_idm})
        self.grad_monitor = GradientMonitor()
        self.grad_monitor.register('ctx_encoder', m.amm.ctx)
        self.grad_monitor.register('fib_encoder', m.amm.fib)
        self.grad_monitor.register('dir_predictor', m.amm.dir_pred)
        self.grad_monitor.register('fiber_connection', m.amm.conn)
        self.grad_monitor.register('fiber_attn', m.amm.attn)
        self.grad_monitor.register('reranker', m.amm.reranker)
        self.grad_monitor.register('qformer', m.bridge.proj)
        self.grad_monitor.register('content_bypass', m.bridge.bypass)
        self.grad_monitor.register('semantic_probe', m.semantic_probe)
        self.grad_monitor.register('layer_pool', m.layer_pool)
        self.grad_monitor.register('prefix_aligner', m.bridge.aligner)
        self.grad_monitor.register('vocab_proj', m.vocab_proj)
        if m.bridge._effective_tail_slots > 0:
            self.grad_monitor.register('tail_head', m.bridge.tail_head)
        if m.bridge.context_heads is not None:
            self.grad_monitor.register('context_heads', m.bridge.context_heads)
        if m.memory_context_encoder is not None:
            self.grad_monitor.register('memory_context_encoder', m.memory_context_encoder)
        if m.mixture_gate_head is not None:
            self.grad_monitor.register('mixture_gate_head', m.mixture_gate_head)
        self.layer_weight_history = []; self._step_count = 0

    def _encode_with_grad(self, texts):
        tk = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
        dev = next(self.m.parameters()).device
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad():
            o = self.m.fwd(ids, mask)
            surp = self.m.amm.surprise_proxy(o['logits'][:, :-1], ids[:, 1:])
        pooled = self.m.layer_pool(o['hs']); pooled_mean = pooled.mean(1)
        base = self.m.amm.ctx(pooled_mean)
        fiber = self.m.amm.fib(pooled_mean, base, surp)
        _ = self.m.amm.dir_pred(base, fiber)
        return ids, mask, base, fiber, surp, pooled_mean

    def encoder_throughput_loss(self, ids, mask, fiber):
        B = ids.shape[0]; dev = ids.device
        fiber_unsq = fiber.unsqueeze(1); mem_mask_ones = torch.ones(B, 1, device=dev)
        prefix = self.m.bridge.inject(fiber_unsq, mem_mask_ones, fiber_summary=fiber,
                                      context_descriptors_d_llm=None,
                                      rare_keyword_wte_residual=None)
        o2 = self.m.fwd(ids, mask, prefix)
        lg = o2['logits'][:, o2['pl']:-1]; tg = ids[:, 1:]
        ml = min(lg.shape[1], tg.shape[1])
        if ml == 0: return torch.tensor(0.0, device=dev, requires_grad=True)
        return F.cross_entropy(lg[:, :ml].reshape(-1, lg.shape[-1]), tg[:, :ml].reshape(-1))

    def semantic_alignment_loss(self, fiber, target_ids, target_mask):
        dev = fiber.device
        wte = self.m.backbone.input_embedding_weight().to(dev)
        vocab_logits = self.m.vocab_proj(fiber, wte)
        B, V = vocab_logits.shape; cc = self.m.content_classifier
        if cc is None: return torch.tensor(0.0, device=dev, requires_grad=True)
        target = torch.zeros(B, V, device=dev); valid_count = 0
        for b in range(B):
            valid = target_ids[b][target_mask[b].bool()].tolist()
            content_ids = cc.get_content_ids_from_tokens(valid)
            if content_ids:
                uids = sorted(set(content_ids)); uids = [uid for uid in uids if uid < V]
                if uids: target[b, uids] = 1.0 / len(uids); valid_count += 1
        if valid_count == 0: return torch.tensor(0.0, device=dev, requires_grad=True)
        log_probs = F.log_softmax(vocab_logits / self.c.semantic_align_temp, dim=-1)
        kl = F.kl_div(log_probs, target, reduction='none').sum(-1)
        return kl.mean()

    def vocab_anchor_loss(self, prefix):
        dev = prefix.device
        wte = self.m.backbone.input_embedding_weight().to(dev)
        pn = F.normalize(prefix.reshape(-1, prefix.shape[-1]), dim=-1)
        wn = F.normalize(wte, dim=-1)
        sim = pn @ wn.T; topk_sim = sim.topk(self.c.vocab_anchor_topk, dim=-1).values
        return -topk_sim.mean()

    def tail_semantic_anchor_loss(self, fiber, ids, mask):
        if self.m.bridge._effective_tail_slots == 0:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        tail = self.m.bridge.tail_head(fiber)
        if tail is None:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        dev = fiber.device
        wte = self.m.backbone.input_embedding_weight().to(dev)
        B, n_slots, _ = tail.shape; V = wte.shape[0]
        cc = self.m.content_classifier
        if cc is None: return torch.tensor(0.0, device=dev, requires_grad=True)
        tn = F.normalize(tail, dim=-1); wn = F.normalize(wte, dim=-1)
        corpus_idf = self.m.amm._compute_corpus_idf(cc)
        use_rare = (self.c.use_keyword_tail_slot and n_slots >= 2
                    and corpus_idf and len(corpus_idf) > 0)
        losses = []
        for b in range(B):
            valid = ids[b][mask[b].bool()].tolist()
            content_tids = sorted(set(cc.get_content_ids_from_tokens(valid)))
            content_tids = [t for t in content_tids if t < V]
            if not content_tids: continue
            target_general = torch.zeros(V, device=dev)
            target_general[content_tids] = 1.0 / len(content_tids)
            slot0_logits = tn[b, 0] @ wn.T / 0.3
            log_p0 = F.log_softmax(slot0_logits, dim=-1)
            losses.append(F.kl_div(log_p0.unsqueeze(0), target_general.unsqueeze(0),
                                   reduction='none').sum(-1).mean())
            if use_rare:
                strict_starters = [t for t in content_tids
                                   if t in cc.strict_content_starter_ids]
                pool = strict_starters if strict_starters else content_tids
                ranked_rare = sorted(pool,
                                     key=lambda t: (-corpus_idf.get(t, self.c.idf_floor), t))
                for s in range(1, n_slots):
                    kw_rank = s - 1
                    if kw_rank < len(ranked_rare):
                        rare_tid = ranked_rare[kw_rank]
                        target_s = torch.zeros(V, device=dev)
                        target_s[rare_tid] = 1.0
                        slot_s_logits = tn[b, s] @ wn.T / 0.3
                        log_ps = F.log_softmax(slot_s_logits, dim=-1)
                        losses.append(self.c.keyword_tail_weight *
                                      F.kl_div(log_ps.unsqueeze(0),
                                               target_s.unsqueeze(0),
                                               reduction='none').sum(-1).mean())
                    else:
                        slot_s_logits = tn[b, s] @ wn.T / 0.3
                        log_ps = F.log_softmax(slot_s_logits, dim=-1)
                        losses.append(F.kl_div(log_ps.unsqueeze(0),
                                               target_general.unsqueeze(0),
                                               reduction='none').sum(-1).mean())
        if not losses:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        return torch.stack(losses).mean()

    def slot_residual_alignment_loss(self):
        """[B] 惩罚 slot_1+ 方向与 residual 方向夹角过大(cos < floor)。"""
        dev = next(self.m.parameters()).device
        tail_post = self.m.bridge._last_tail_pre_renorm
        residual = self.m.bridge._last_residual
        if tail_post is None or residual is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        B, n_slots, D = tail_post.shape
        if n_slots < 2:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        losses = []
        for s in range(1, n_slots):
            r_s = residual[:, s, :]
            t_s = tail_post[:, s, :]
            r_norm = r_s.norm(dim=-1, keepdim=True)
            valid = (r_norm.squeeze(-1) > 1e-6)
            if valid.sum() == 0: continue
            r_n = F.normalize(r_s[valid], dim=-1, eps=1e-8)
            t_n = F.normalize(t_s[valid], dim=-1, eps=1e-8)
            cos = (r_n * t_n).sum(dim=-1)
            floor = self.c.tail_slot_cos_alignment_floor
            violation = F.relu(floor - cos)
            losses.append(violation.mean())
        if not losses:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        return torch.stack(losses).mean()

    def inter_domain_margin_loss(self, texts):
        """[C] 用 semantic_emb KMeans 弱标签,约束 fiber dir 的同/跨域 cos。"""
        dev = next(self.m.parameters()).device
        if len(texts) < 4 or not self.c.use_inter_domain_margin:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        tk = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad(): o = self.m.fwd(ids, mask)
        _, xq, fq = self.m.extract_state(o['hs'], mask)
        with torch.no_grad():
            pooled = self.m.layer_pool(o['hs'])
            sem_emb = self.m._compute_content_semantic_emb(pooled, ids, mask)
            sem_n = F.normalize(sem_emb.float(), dim=-1, eps=1e-8)
            K = min(self.c.inter_domain_kmeans_k, len(texts) // 2)
            if K < 2:
                return torch.tensor(0.0, device=dev, requires_grad=True)
            assign, _ = _simple_kmeans(
                sem_n, k=K, n_iter=self.c.inter_domain_kmeans_iters, seed=0)
        dirs = F.normalize(self.m.amm.dir_pred(xq, fq), dim=-1, eps=1e-8)
        sim = dirs @ dirs.T
        N = dirs.shape[0]
        same_mask = (assign.unsqueeze(0) == assign.unsqueeze(1))
        same_mask = same_mask & (~torch.eye(N, dtype=torch.bool, device=dev))
        cross_mask = ~same_mask & (~torch.eye(N, dtype=torch.bool, device=dev))
        same_target = self.c.inter_domain_same_cos_target
        cross_target = self.c.inter_domain_cross_cos_target
        margin = self.c.inter_domain_margin
        loss = torch.tensor(0.0, device=dev)
        if same_mask.any():
            same_cos = sim[same_mask]
            loss = loss + F.relu(same_target - same_cos + margin * 0.5).mean()
        if cross_mask.any():
            cross_cos = sim[cross_mask]
            loss = loss + F.relu(cross_cos - cross_target + margin * 0.5).mean()
        return loss

    def functional_suppression_loss(self, prefix, ids, mask):
        o = self.m.fwd(ids, mask, prefix)
        last_logits = o['logits'][:, -1, :]
        cc = self.m.content_classifier
        if cc is None:
            return torch.tensor(0.0, device=last_logits.device, requires_grad=True)
        dev = last_logits.device
        V_cur = last_logits.shape[-1]
        starter_mask = cc.content_starter_mask(dev)[:V_cur].bool()
        eos_id = self.m.tok.eos_token_id
        func_mask = cc.pure_function_mask(dev, eos_id=eos_id)[:V_cur].bool()
        B = last_logits.shape[0]
        starter_bool = starter_mask.unsqueeze(0).expand(B, -1)
        func_bool = func_mask.unsqueeze(0).expand(B, -1)
        NEG = last_logits.new_full((), -1e9)
        top_starter = torch.where(starter_bool, last_logits, NEG).max(-1).values
        top_func = torch.where(func_bool, last_logits, NEG).max(-1).values
        margin = self.c.functional_suppression_margin
        violation = top_func - top_starter + margin
        return F.relu(violation).mean()

    def context_separation_loss(self, texts):
        if self.m.memory_context_encoder is None or len(texts) < 2:
            dev = next(self.m.parameters()).device
            return torch.tensor(0.0, device=dev, requires_grad=True)
        dev = next(self.m.parameters()).device
        wte = self.m.backbone.input_embedding_weight().to(dev)
        cc = self.m.content_classifier
        tk = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
        ids_b, mask_b = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad():
            o = self.m.fwd(ids_b, mask_b)
            hs_pooled = self.m.layer_pool(o['hs'])
        content_hidden_per_b = self.m._extract_content_hidden_per_b(hs_pooled, ids_b, mask_b)
        per_text_strict_ids = []
        for t in texts:
            raw_ids = self.m.tok.encode(t)
            ss = cc.get_strict_starter_ids_from_tokens(raw_ids) if cc else []
            per_text_strict_ids.append(sorted(set(ss)))
        descs = []
        for i, ss in enumerate(per_text_strict_ids):
            if not ss: continue
            idx = torch.tensor(sorted([t for t in ss if t < wte.shape[0]]),
                               device=dev, dtype=torch.long)
            if idx.numel() == 0: continue
            centroid = wte.index_select(0, idx).float().mean(0)
            d = self.m.memory_context_encoder.encode_with_hidden(
                centroid, hidden_states=content_hidden_per_b[i])
            descs.append(d)
        if len(descs) < 2:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        D = torch.stack(descs, dim=0)
        sim = D @ D.T
        N = D.shape[0]
        off_mask = ~torch.eye(N, dtype=torch.bool, device=dev)
        off_sim = sim[off_mask]
        return off_sim.clamp(min=0.0).mean()

    def _recon_forward(self, text):
        tk = self.m.tok(text, return_tensors='pt', padding=True, truncation=True)
        dev = next(self.m.parameters()).device
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad(): bo = self.m.fwd(ids, mask)
        prefix = self.m._get_prefix(bo['hs'], mask, update_stats=False, ids=ids)
        o = self.m.fwd(ids, mask, prefix)
        lg = o['logits'][:, o['pl']:-1]; tg = ids[:, 1:]
        ml = min(lg.shape[1], tg.shape[1])
        if ml == 0:
            zero = ids.new_tensor(0.0, dtype=torch.float, requires_grad=True)
            return zero, prefix, self.m.bridge._last_fiber_summary, ids, mask
        l_r = F.cross_entropy(lg[:, :ml].reshape(-1, lg.shape[-1]), tg[:, :ml].reshape(-1))
        fs = self.m.bridge._last_fiber_summary
        if fs is None: fs = torch.zeros(1, self.c.d_F, device=dev)
        return l_r, prefix, fs, ids, mask

    def recon(self, text):
        loss, prefix, fs, ids, mask = self._recon_forward(text)
        return {'loss': loss, 'prefix': prefix, 'fiber_summary': fs,
                'ids': ids, 'mask': mask}

    def _semantic_probe_loss(self, prefix_batch, fs_batch):
        pred = self.m.semantic_probe(prefix_batch)
        l_mse = F.mse_loss(pred, fs_batch.detach())
        if prefix_batch.shape[0] >= 2:
            pn = F.normalize(pred, dim=-1); tn = F.normalize(fs_batch.detach(), dim=-1)
            sim = pn @ tn.T / self.c.probe_contrastive_tau
            lb = torch.arange(prefix_batch.shape[0], device=prefix_batch.device)
            l_ctr = F.cross_entropy(sim, lb)
            return l_mse + 0.5 * l_ctr
        return l_mse

    def contrast(self, texts):
        tk = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
        dev = next(self.m.parameters()).device
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad(): o = self.m.fwd(ids, mask)
        _, xq, fq = self.m.extract_state(o['hs'], mask)
        x = F.normalize(self.m.amm.contrast_proj_x(xq), -1)
        f = F.normalize(self.m.amm.contrast_proj_f(fq), -1)
        sxf = x @ f.T / self.c.contrast_tau; sfx = f @ x.T / self.c.contrast_tau
        lb = torch.arange(len(texts), device=dev)
        return (F.cross_entropy(sxf, lb) + F.cross_entropy(sfx, lb)) / 2

    def holonomy_proxy(self, x, f):
        sz = 0.05; v1 = torch.randn_like(x) * sz; v2 = torch.randn_like(x) * sz
        loop = torch.stack([x, x+v1, x+v1+v2, x+v2, x], 1)
        return (self.m.amm.trans(f, loop) - f).pow(2).sum(-1).mean()

    def write_policy_loss(self, texts):
        tk = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
        dev = next(self.m.parameters()).device
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad():
            o = self.m.fwd(ids, mask)
            surp = self.m.amm.surprise_proxy(o['logits'][:, :-1], ids[:, 1:])
        pooled = self.m.layer_pool(o['hs']).mean(1)
        gates = self.m.amm.write_gate(pooled, surp)
        labels = (surp > surp.median()).float()
        return F.binary_cross_entropy(gates, labels)

    def direction_diversity_loss(self, texts):
        tk = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
        dev = next(self.m.parameters()).device
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad(): o = self.m.fwd(ids, mask)
        _, xq, fq = self.m.extract_state(o['hs'], mask)
        dirs = F.normalize(self.m.amm.dir_pred(xq, fq), dim=-1, eps=1e-8)
        dir_sim = (dirs @ dirs.T).clamp(-1.0, 1.0)
        with torch.no_grad():
            fn = F.normalize(fq, dim=-1, eps=1e-8); fiber_sim = (fn @ fn.T).clamp(-1.0, 1.0)
        tau = self.c.dir_diversity_tau
        dir_prob = torch.sigmoid(dir_sim / tau); fiber_prob = torch.sigmoid(fiber_sim / tau)
        B = len(texts); mask_off = ~torch.eye(B, dtype=torch.bool, device=dev)
        return F.binary_cross_entropy(dir_prob[mask_off], fiber_prob[mask_off].detach())

    def reranker_ranking_loss(self, texts):
        store = self.m.amm.tree.store
        if len(store) < 2:
            dev = next(self.m.parameters()).device
            return torch.tensor(0.0, device=dev, requires_grad=True)
        tk = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
        dev = next(self.m.parameters()).device
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad(): o = self.m.fwd(ids, mask)
        _, xq, fq = self.m.extract_state(o['hs'], mask)
        mids = sorted(store.keys())
        cb = torch.stack([store[m].base.to(dev) for m in mids])
        cf = torch.stack([store[m].fiber.to(dev) for m in mids])
        cd = torch.stack([store[m].dirn.to(dev) for m in mids])
        B = xq.shape[0]; qdir = self.m.amm.dir_pred(xq, fq)
        dir_sims = torch.einsum('bd,cd->bc', qdir, cd)
        cb_e = cb.unsqueeze(0).expand(B, -1, -1); cf_e = cf.unsqueeze(0).expand(B, -1, -1)
        scores = self.m.amm.reranker(xq, fq, cb_e, cf_e, dir_sims)
        with torch.no_grad():
            fqn = F.normalize(fq, dim=-1); cfn = F.normalize(cf, dim=-1)
            relevance = torch.einsum('bd,cd->bc', fqn, cfn)
        s_mean = scores.mean(-1, keepdim=True); s_std = scores.std(-1, keepdim=True).clamp(min=1e-6)
        r_mean = relevance.mean(-1, keepdim=True); r_std = relevance.std(-1, keepdim=True).clamp(min=1e-6)
        sn = (scores - s_mean) / s_std; rn = (relevance - r_mean) / r_std
        return F.mse_loss(sn, rn.detach())

    def step(self, texts):
        self.m.train(); self.opt.zero_grad()
        dev = next(self.m.parameters()).device; W = self.c.loss_weights
        ids_enc, mask_enc, base, fiber, surp, pooled_mean = self._encode_with_grad(texts)
        l_et = self.encoder_throughput_loss(ids_enc, mask_enc, fiber)
        w_sa = self.warmup.weight('semantic_alignment')
        l_sa = self.semantic_alignment_loss(fiber, ids_enc, mask_enc) * w_sa
        w_tsa = self.warmup.weight('tail_semantic_anchor')
        l_tsa = self.tail_semantic_anchor_loss(fiber, ids_enc, mask_enc) * w_tsa
        all_lr, all_pf, all_fs, all_ids, all_mask = [], [], [], [], []
        for t in texts:
            l_r_t, pf_t, fs_t, ids_t, mask_t = self._recon_forward(t)
            all_lr.append(l_r_t); all_pf.append(pf_t)
            all_fs.append(fs_t if fs_t is not None else torch.zeros(1, self.c.d_F, device=dev))
            all_ids.append(ids_t); all_mask.append(mask_t)
        l_r = sum(all_lr) / len(texts)
        pf_batch = torch.cat(all_pf, 0); fs_batch = torch.cat(all_fs, 0)
        w_sp = self.warmup.weight('semantic_probe')
        l_sp = self._semantic_probe_loss(pf_batch, fs_batch) * w_sp
        w_va = self.warmup.weight('vocab_anchor')
        l_va = self.vocab_anchor_loss(pf_batch) * w_va
        if self.c.use_functional_suppression:
            w_fs = self.warmup.weight('functional_suppression')
            l_fs_list = [
                self.functional_suppression_loss(all_pf[i], all_ids[i], all_mask[i])
                for i in range(len(texts))]
            l_fs = (sum(l_fs_list) / len(l_fs_list)) * w_fs
        else:
            l_fs = torch.tensor(0.0, device=dev)
        w_cs = self.warmup.weight('context_separation')
        l_cs = self.context_separation_loss(texts) * w_cs
        w_sra = self.warmup.weight('slot_residual_alignment')
        l_sra = self.slot_residual_alignment_loss() * w_sra
        w_idm = self.warmup.weight('inter_domain_margin')
        l_idm = self.inter_domain_margin_loss(texts) * w_idm
        l_c = self.contrast(texts) if len(texts) >= 2 else torch.tensor(0.0, device=dev)
        with torch.no_grad():
            tk2 = self.m.tok(texts, return_tensors='pt', padding=True, truncation=True)
            ids2, mask2 = tk2['input_ids'].to(dev), tk2['attention_mask'].to(dev)
            o2 = self.m.fwd(ids2, mask2)
        _, xq2, fq2 = self.m.extract_state(o2['hs'], mask2)
        l_h = self.holonomy_proxy(xq2, fq2)
        l_w = self.write_policy_loss(texts)
        w_dd = self.warmup.weight('dir_diversity')
        l_dd = (self.direction_diversity_loss(texts) if len(texts) >= 2
                else torch.tensor(0.0, device=dev)) * w_dd
        w_rr = self.warmup.weight('reranker_ranking')
        l_rr = self.reranker_ranking_loss(texts) * w_rr
        loss = (W['recon']*l_r + W['semantic_alignment']*l_sa +
                W['encoder_throughput']*l_et + W['contrast']*l_c +
                W['holonomy']*l_h + W['write_policy']*l_w +
                W['semantic_probe']*l_sp + W['dir_diversity']*l_dd +
                W['reranker_ranking']*l_rr + W['vocab_anchor']*l_va +
                W.get('tail_semantic_anchor', 0.5)*l_tsa +
                W.get('functional_suppression', 0.4)*l_fs +
                W.get('context_separation', 0.3)*l_cs +
                W.get('slot_residual_alignment', 0.3)*l_sra +
                W.get('inter_domain_margin', 0.2)*l_idm)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for n, p in self.m.named_parameters()
             if p.requires_grad and 'backbone' not in n], 1.)
        self.opt.step(); self.warmup.advance(); self._step_count += 1
        grad_norms = self.grad_monitor.snapshot()
        self.layer_weight_history.append(self.m.layer_pool.weight_dist().cpu().numpy().copy())
        if self._step_count % self.c.refresh_memories_every == 0:
            self.m.eval()
            with torch.no_grad(): self.m._refresh_all_memories()
            self.m.train()
        self.m.eval()
        return {'total': loss.item(), 'recon': l_r.item(), 'contrast': l_c.item(),
                'holonomy': l_h.item(), 'write_policy': l_w.item(),
                'semantic_probe': l_sp.item(), 'dir_diversity': l_dd.item(),
                'reranker_ranking': l_rr.item(), 'encoder_throughput': l_et.item(),
                'vocab_anchor': l_va.item(), 'semantic_alignment': l_sa.item(),
                'tail_semantic_anchor': l_tsa.item(),
                'functional_suppression': l_fs.item(),
                'context_separation': l_cs.item(),
                'slot_residual_alignment': l_sra.item(),
                'inter_domain_margin': l_idm.item(),
                'grad_norms': grad_norms, 'loss_weights': W}
