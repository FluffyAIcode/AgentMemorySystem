from scheme_b_v323 import *
import scheme_b_v323 as v323

_dev = v323._dev
_Node = v323._Node


@dataclass
class Cfg(v323.Cfg):
    use_quadruple_consensus: bool = True
    consensus_token_vote_topk: int = 3
    consensus_token_vote_threshold: float = 0.5
    ret_centroid_weight: float = 0.25
    ret_sem_weight: float = 0.10
    ret_bidi_min_weight: float = 0.20
    ret_forward_maxsim_weight: float = 0.40
    ret_dir_weight: float = 0.05
    consensus_vote_weight: float = 0.6
    use_sustained_filler: bool = True
    sustained_filler_penalty: float = 15.0
    sustained_filler_steps: int = 10
    sustained_filler_decay: float = 0.12
    content_repeat_exponent: float = 1.5
    use_strict_anchor_boost: bool = True
    strict_anchor_boost_topk: int = 6
    strict_anchor_boost_scale: float = 8.0
    strict_anchor_boost_steps: int = 12
    strict_anchor_boost_decay: float = 0.06
    strict_anchor_boost_floor: float = 0.2
    stopwords_override: Optional[FrozenSet[str]] = None
    filler_words_override: Optional[FrozenSet[str]] = None
    stopwords_extra: FrozenSet[str] = field(default_factory=frozenset)
    filler_words_extra: FrozenSet[str] = field(default_factory=frozenset)
    dedup_filler_from_stop: bool = False
    use_cluster_vote_aggregation: bool = True
    cluster_vote_jaccard_threshold: float = 0.15
    use_ngram_repeat_block: bool = True
    ngram_repeat_penalty: float = 10.0
    ngram_repeat_max_n: int = 4
    use_content_gated_newline: bool = True
    min_content_tokens_before_newline: int = 8
    late_newline_penalty: float = 50.0
    use_upstream_semantic_gate: bool = True
    upstream_gate_fwd_idf_floor: float = 0.12
    upstream_gate_sem_floor: float = 0.15
    use_strict_content_overlap_gate: bool = True
    strict_overlap_sim_threshold: float = 0.45
    strict_overlap_min_matches: int = 1
    strict_overlap_min_keep: int = 1
    upstream_gate_require_both: bool = True
    upstream_gate_min_keep: int = 1
    use_adaptive_consensus_threshold: bool = True
    consensus_threshold_query_size_ref: int = 4
    consensus_threshold_min_ratio: float = 0.65
    use_domain_conflict_resolver: bool = True
    domain_conflict_jaccard_threshold: float = 0.15
    domain_conflict_min_clusters: int = 2
    domain_conflict_score_min_ratio: float = 1.05
    use_cyclic_content_hard_mask: bool = True
    cyclic_content_window: int = 15
    cyclic_content_max_count: int = 2
    use_early_bigram_hard_mask: bool = True
    early_bigram_min_content_token: bool = True
    use_newline_hard_gate: bool = True
    use_prefix_norm_clamp: bool = True
    prefix_norm_clamp_ratio: float = 1.0
    use_eos_hard_mask: bool = True
    eos_hard_mask_steps: int = 15
    newline_hard_gate_min_step: int = 20
    newline_hard_gate_min_content: int = 10
    use_strict_avg_maxsim_gate: bool = True
    strict_avg_maxsim_threshold: float = 0.28
    strict_avg_maxsim_min_keep: int = 1
    domain_conflict_use_match_rate_weight: bool = True
    use_post_gate_fwd_idf_floor: bool = True
    post_gate_fwd_idf_floor: float = 0.15
    post_gate_fwd_idf_min_keep: int = 1
    use_filler_direction_projection: bool = True
    filler_projection_last_slots: int = 2
    use_step0_strict_hard_restrict: bool = True
    step0_strict_fallback_threshold: float = -50.0
    use_early_non_strict_hard_penalty: bool = True
    early_non_strict_hard_penalty: float = 15.0
    early_non_strict_hard_penalty_steps: int = 12
    use_strict_avg_maxsim_relative_floor: bool = True
    strict_avg_maxsim_relative_ratio: float = 0.5
    strict_avg_maxsim_relative_min_top: float = 0.30
    strict_avg_maxsim_relative_min_keep: int = 1
    use_fwd_idf_relative_floor: bool = True
    fwd_idf_relative_ratio: float = 0.55
    fwd_idf_relative_min_top: float = 0.18
    fwd_idf_relative_min_keep: int = 1
    use_final_domain_purge: bool = True
    final_domain_purge_margin: float = 1.08
    final_domain_purge_jaccard: float = 0.12
    extended_strict_restrict_steps: int = 3
    extended_strict_fallback_threshold: float = -50.0
    use_early_punct_hard_mask: bool = True
    early_punct_hard_mask_steps: int = 6
    use_early_function_hard_mask: bool = True
    early_function_hard_mask_steps: int = 4


class ContentTokenClassifier(v323.ContentTokenClassifier):
    DEFAULT_STOPWORDS = v323.ContentTokenClassifier.STOPWORDS
    DEFAULT_FILLER_WORDS = v323.ContentTokenClassifier.FILLER_WORDS | frozenset(
        {
            "various",
            "several",
            "many",
            "multiple",
            "different",
            "diverse",
            "varied",
            "certain",
            "particular",
            "specific",
            "general",
            "overall",
            "whole",
            "entire",
            "aspect",
            "aspects",
            "feature",
            "features",
            "element",
            "elements",
            "factor",
            "factors",
            "component",
            "components",
            "quality",
            "qualities",
            "example",
            "examples",
            "instance",
            "instances",
            "case",
            "cases",
            "method",
            "methods",
            "approach",
            "approaches",
            "process",
            "processes",
            "system",
            "systems",
            "part",
            "parts",
            "kind",
            "kinds",
            "type",
            "types",
            "sort",
            "sorts",
            "people",
            "person",
            "someone",
            "anyone",
            "everyone",
            "matter",
            "matters",
            "issue",
            "issues",
            "point",
            "points",
            "number",
            "numbers",
            "amount",
            "amounts",
            "level",
            "levels",
            "student",
            "students",
            "practice",
            "practicing",
            "action",
            "actions",
            "role",
            "roles",
            "purpose",
            "purposes",
            "nature",
            "natures",
            "character",
            "characters",
            "condition",
            "conditions",
            "state",
            "states",
            "status",
            "statuses",
            "fact",
            "facts",
            "substance",
            "substances",
            "material",
            "materials",
            "content",
            "contents",
            "context",
            "contexts",
            "task",
            "tasks",
            "duty",
            "duties",
            "operation",
            "operations",
            "performance",
            "performances",
            "activity",
            "activities",
            "topic",
            "topics",
            "subject",
            "subjects",
            "concept",
            "concepts",
            "idea",
            "ideas",
            "notion",
            "notions",
            "result",
            "results",
            "outcome",
            "outcomes",
            "effect",
            "effects",
            "area",
            "areas",
            "region",
            "regions",
            "range",
            "ranges",
            "degree",
            "degrees",
            "extent",
            "extents",
            "period",
            "periods",
            "moment",
            "moments",
            "detail",
            "details",
            "information",
            "piece",
            "pieces",
            "group",
            "groups",
            "set",
            "sets",
            "form",
            "forms",
            "style",
            "styles",
            "mode",
            "modes",
            "version",
            "versions",
            "manner",
            "manners",
            "fashion",
            "fashions",
            "attribute",
            "attributes",
            "property",
            "properties",
            "trait",
            "traits",
            "characteristic",
            "characteristics",
            "place",
            "places",
            "way",
            "ways",
        }
    )

    def __init__(self, tokenizer, cfg=None, min_len=None, strict_min_len=None):
        if isinstance(cfg, int):
            legacy_min = cfg
            legacy_strict = min_len if isinstance(min_len, int) else strict_min_len
            cfg = Cfg()
            min_len = legacy_min
            if legacy_strict is not None:
                strict_min_len = legacy_strict
        if cfg is None:
            cfg = Cfg()
        self.cfg = cfg
        min_len = min_len if isinstance(min_len, int) else cfg.content_min_len
        strict_min_len = (
            strict_min_len if isinstance(strict_min_len, int) else cfg.strict_starter_min_decoded_len
        )
        if cfg.stopwords_override is not None:
            self.STOPWORDS = cfg.stopwords_override
        else:
            self.STOPWORDS = self.DEFAULT_STOPWORDS | cfg.stopwords_extra
        if cfg.filler_words_override is not None:
            self.FILLER_WORDS = cfg.filler_words_override
        else:
            self.FILLER_WORDS = self.DEFAULT_FILLER_WORDS | cfg.filler_words_extra
        if cfg.dedup_filler_from_stop:
            self.FILLER_WORDS = self.FILLER_WORDS - self.STOPWORDS
        raw_vocab_size = getattr(tokenizer, "vocab_size", 50257)
        self._scan_upper = min(int(raw_vocab_size), 50300)
        self._V: int = self._scan_upper
        super().__init__(tokenizer, min_len=min_len, strict_min_len=strict_min_len)
        self._filler_tensor = None
        self._function_tensor = None
        self._punct_tensor = None

    def _vocab_size(self) -> int:
        return int(getattr(self, "_V", 50300))

    def _mask_size(self) -> int:
        return int(getattr(self, "_V", 50300))

    def content_mask(self, device):
        if self._content_tensor is None or self._content_tensor.device != device:
            V = self._mask_size()
            m = torch.zeros(V, device=device)
            for i in self.content_ids:
                if i < V:
                    m[i] = 1.0
            self._content_tensor = m
        return self._content_tensor

    def content_starter_mask(self, device):
        if self._content_starter_tensor is None or self._content_starter_tensor.device != device:
            V = self._mask_size()
            m = torch.zeros(V, device=device)
            for i in self.content_starter_ids:
                if i < V:
                    m[i] = 1.0
            self._content_starter_tensor = m
        return self._content_starter_tensor

    def strict_content_starter_mask(self, device):
        if self._strict_content_starter_tensor is None or self._strict_content_starter_tensor.device != device:
            V = self._mask_size()
            m = torch.zeros(V, device=device)
            for i in self.strict_content_starter_ids:
                if i < V:
                    m[i] = 1.0
            self._strict_content_starter_tensor = m
        return self._strict_content_starter_tensor

    def non_strict_content_mask(self, device):
        if self._non_strict_content_tensor is None or self._non_strict_content_tensor.device != device:
            cm = self.content_mask(device)
            sm = self.strict_content_starter_mask(device)
            V = min(cm.shape[0], sm.shape[0])
            m = torch.zeros(cm.shape[0], device=device)
            m[:V] = cm[:V] * (1.0 - sm[:V])
            self._non_strict_content_tensor = m
        return self._non_strict_content_tensor

    def filler_mask(self, device):
        if self._filler_tensor is None or self._filler_tensor.device != device:
            V = self._mask_size()
            m = torch.zeros(V, device=device)
            for i in self.filler_ids:
                if i < V:
                    m[i] = 1.0
            self._filler_tensor = m
        return self._filler_tensor

    def punct_mask(self, device):
        if self._punct_tensor is None or self._punct_tensor.device != device:
            V = self._mask_size()
            m = torch.zeros(V, device=device)
            for i in self.punct_ids:
                if i < V:
                    m[i] = 1.0
            self._punct_tensor = m
        return self._punct_tensor

    def function_mask(self, device):
        if self._function_tensor is None or self._function_tensor.device != device:
            V = self._mask_size()
            m = torch.zeros(V, device=device)
            for i in self.function_ids:
                if i < V:
                    m[i] = 1.0
            self._function_tensor = m
        return self._function_tensor

    def get_strict_content_ids_from_tokens(self, token_ids):
        return [t for t in token_ids if t in self.strict_content_starter_ids]


class EmbBridge(v323.EmbBridge):
    def inject(
        self,
        fibers,
        mem_mask=None,
        fiber_summary=None,
        content_wte_mean=None,
        content_target_wte=None,
        hard_wte_last_slots=None,
        filler_centroid=None,
    ):
        qf_out = super().inject(
            fibers,
            mem_mask=mem_mask,
            fiber_summary=fiber_summary,
            content_wte_mean=content_wte_mean,
            content_target_wte=content_target_wte,
            hard_wte_last_slots=hard_wte_last_slots,
        )
        filler_dir_used = self.c.use_filler_direction_projection and filler_centroid is not None
        filler_proj_comp_max = 0.0
        if filler_dir_used:
            n_proj = min(self.c.filler_projection_last_slots, qf_out.shape[1])
            fd = filler_centroid.view(1, 1, -1)
            slot_mask = torch.zeros(qf_out.shape[1], device=qf_out.device).view(1, -1, 1)
            slot_mask[:, -n_proj:, :] = 1.0
            comp = (qf_out * fd).sum(dim=-1, keepdim=True)
            filler_proj_comp_max = comp.abs().max().item()
            qf_out = qf_out - comp * fd * slot_mask
        pre_clamp_norm_max = qf_out.norm(dim=-1).max().item()
        clamp_applied_count = 0
        target_norm_used = 0.0
        max_allowed_used = 0.0
        if self.c.use_prefix_norm_clamp:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            max_allowed = target_norm * self.c.prefix_norm_clamp_ratio
            slot_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            exceed_mask = slot_norms.squeeze(-1) > max_allowed
            clamp_applied_count = int(exceed_mask.sum().item())
            scale = torch.clamp(max_allowed / slot_norms, max=1.0)
            qf_out = qf_out * scale
            target_norm_used = target_norm
            max_allowed_used = max_allowed
        post_clamp_norm_max = qf_out.norm(dim=-1).max().item()
        self._last_inject_diag = {
            **self._last_inject_diag,
            "qf_norm": qf_out.norm().item(),
            "last_slot_norm_per_b": qf_out[:, -1].norm(dim=-1).mean().item(),
            "second_last_slot_norm_per_b": (qf_out[:, -2].norm(dim=-1).mean().item() if qf_out.shape[1] >= 2 else 0.0),
            "pre_clamp_max_slot_norm": pre_clamp_norm_max,
            "post_clamp_max_slot_norm": post_clamp_norm_max,
            "clamp_applied_slots": clamp_applied_count,
            "target_norm": target_norm_used,
            "max_allowed_norm": max_allowed_used,
            "filler_dir_projected": filler_dir_used,
            "filler_proj_comp_max": filler_proj_comp_max,
        }
        return qf_out


@dataclass
class RetrievalDiag(v323.RetrievalDiag):
    per_memory_vote_ratio: Dict[int, float] = field(default_factory=dict)
    consensus_top1_vote_ratio: float = 0.0
    consensus_vote_reassigned: bool = False
    consensus_combined_margin: float = 0.0
    per_memory_cluster_vote_ratio: Dict[int, float] = field(default_factory=dict)
    consensus_top1_cluster_vote_ratio: float = 0.0
    cluster_vote_aggregation_applied: bool = False
    n_after_upstream_semantic_gate: int = 0
    upstream_semantic_gate_applied: bool = False
    upstream_gate_dropped_ids: List[int] = field(default_factory=list)
    consensus_effective_threshold: float = 0.5
    consensus_query_strict_size: int = 0
    n_after_strict_overlap_gate: int = 0
    n_after_strict_avg_maxsim_gate: int = 0
    n_after_strict_avg_maxsim_relative_floor: int = 0
    per_memory_strict_overlap: Dict[int, int] = field(default_factory=dict)
    per_memory_strict_avg_maxsim: Dict[int, float] = field(default_factory=dict)
    strict_overlap_gate_applied: bool = False
    strict_overlap_dropped_ids: List[int] = field(default_factory=list)
    strict_avg_maxsim_gate_applied: bool = False
    strict_avg_maxsim_dropped_ids: List[int] = field(default_factory=list)
    strict_avg_maxsim_relative_floor_applied: bool = False
    strict_avg_maxsim_relative_dropped_ids: List[int] = field(default_factory=list)
    domain_conflict_resolver_applied: bool = False
    domain_conflict_cluster_count: int = 0
    domain_conflict_top_cluster_size: int = 0
    domain_conflict_dropped_ids: List[int] = field(default_factory=list)
    n_after_domain_conflict_resolver: int = 0
    domain_conflict_top_score: float = 0.0
    domain_conflict_second_score: float = 0.0
    n_after_post_gate_fwd_idf_floor: int = 0
    n_after_fwd_idf_relative_floor: int = 0
    n_after_final_domain_purge: int = 0
    post_gate_fwd_idf_floor_applied: bool = False
    post_gate_fwd_idf_dropped_ids: List[int] = field(default_factory=list)
    fwd_idf_relative_floor_applied: bool = False
    fwd_idf_relative_dropped_ids: List[int] = field(default_factory=list)
    final_domain_purge_applied: bool = False
    final_domain_purge_dropped_ids: List[int] = field(default_factory=list)
    final_domain_purge_top_score: float = 0.0
    final_domain_purge_second_score: float = 0.0


class AMM(v323.AMM):
    def _compute_token_majority_votes(
        self,
        query_content_ids,
        candidate_mems,
        wte_normed,
        corpus_idf,
        content_classifier,
        topk,
        idf_floor,
    ):
        C = len(candidate_mems)
        dev = wte_normed.device
        if C == 0 or not query_content_ids:
            return torch.zeros(C, device=dev)
        q_with_idf = (
            [(t, corpus_idf.get(t, idf_floor)) for t in query_content_ids if t < wte_normed.shape[0]]
            if corpus_idf
            else [(t, 1.0) for t in query_content_ids if t < wte_normed.shape[0]]
        )
        q_with_idf.sort(key=lambda x: -x[1])
        top_q_tokens = [t for t, _ in q_with_idf[:topk]]
        if not top_q_tokens:
            return torch.zeros(C, device=dev)
        mem_vecs = []
        for mem in candidate_mems:
            strict_ids = []
            if content_classifier is not None:
                strict_ids = [
                    t
                    for t in mem.content_token_ids
                    if t in content_classifier.strict_content_starter_ids and t < wte_normed.shape[0]
                ]
            if not strict_ids:
                strict_ids = [t for t in self._get_mem_scoring_ids(mem) if t < wte_normed.shape[0]]
            mem_vecs.append(wte_normed[torch.tensor(strict_ids, device=dev)] if strict_ids else None)
        votes = torch.zeros(C, device=dev)
        for q_tok in top_q_tokens:
            q_vec = wte_normed[q_tok]
            best_sim = -1e9
            best_idx = -1
            for ci, mvec in enumerate(mem_vecs):
                if mvec is None:
                    continue
                s = (mvec @ q_vec).max().item()
                if s > best_sim:
                    best_sim = s
                    best_idx = ci
            if best_idx >= 0:
                votes[best_idx] += 1.0
        return votes / votes.sum().clamp(min=1.0)

    def _compute_cluster_votes(self, votes, mems, content_classifier, jaccard_threshold):
        cluster_votes = votes.clone()
        if content_classifier is None or len(mems) < 2:
            return cluster_votes
        strict_sets = [self._mem_strict_label_set(mem, content_classifier) for mem in mems]
        for i in range(len(mems)):
            for j in range(len(mems)):
                if i == j:
                    continue
                if self._jaccard(strict_sets[i], strict_sets[j]) >= jaccard_threshold:
                    cluster_votes[i] = cluster_votes[i] + votes[j]
        return cluster_votes.clamp(max=1.0)

    @staticmethod
    def _count_strict_overlap_matches(q_strict_ids, m_strict_ids, wte_normed, sim_threshold):
        if not q_strict_ids or not m_strict_ids or wte_normed is None:
            return 0
        V = wte_normed.shape[0]
        q_valid = [t for t in q_strict_ids if t < V]
        m_valid = [t for t in m_strict_ids if t < V]
        if not q_valid or not m_valid:
            return 0
        dev = wte_normed.device
        q_vecs = wte_normed[torch.tensor(q_valid, device=dev)]
        m_vecs = wte_normed[torch.tensor(m_valid, device=dev)]
        sim = q_vecs @ m_vecs.T
        has_match = (sim >= sim_threshold).any(dim=1)
        return int(has_match.sum().item())

    @staticmethod
    def _compute_strict_avg_maxsim(q_strict_ids, m_strict_ids, wte_normed):
        if not q_strict_ids or not m_strict_ids or wte_normed is None:
            return 0.0
        V = wte_normed.shape[0]
        q_valid = [t for t in q_strict_ids if t < V]
        m_valid = [t for t in m_strict_ids if t < V]
        if not q_valid or not m_valid:
            return 0.0
        dev = wte_normed.device
        q_vecs = wte_normed[torch.tensor(q_valid, device=dev)]
        m_vecs = wte_normed[torch.tensor(m_valid, device=dev)]
        sim = q_vecs @ m_vecs.T
        return sim.max(dim=1).values.mean().item()

    def _resolve_domain_conflict(
        self,
        mems,
        forward_idf_t,
        strict_avg_t,
        content_classifier,
        jaccard_threshold,
        min_ratio=None,
    ):
        C = len(mems)
        if C < 2 or content_classifier is None:
            return list(range(C)), 1, [], C, 0.0, 0.0
        if min_ratio is None:
            min_ratio = self.c.domain_conflict_score_min_ratio
        strict_sets = [self._mem_strict_label_set(m, content_classifier) for m in mems]
        parent = list(range(C))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(C):
            for j in range(i + 1, C):
                if self._jaccard(strict_sets[i], strict_sets[j]) >= jaccard_threshold:
                    union(i, j)
        clusters: Dict[int, List[int]] = {}
        for i in range(C):
            clusters.setdefault(find(i), []).append(i)
        if len(clusters) < self.c.domain_conflict_min_clusters:
            return list(range(C)), len(clusters), [], C, 0.0, 0.0
        cluster_list = list(clusters.values())
        if self.c.domain_conflict_use_match_rate_weight:
            cluster_scores = [
                sum(forward_idf_t[i].item() * (1.0 + strict_avg_t[i].item()) for i in cl)
                for cl in cluster_list
            ]
        else:
            cluster_scores = [sum(forward_idf_t[i].item() for i in cl) for cl in cluster_list]
        top_cluster_idx = max(range(len(cluster_list)), key=lambda i: cluster_scores[i])
        top_cluster = cluster_list[top_cluster_idx]
        top_score = cluster_scores[top_cluster_idx]
        other_scores = [cluster_scores[i] for i in range(len(cluster_list)) if i != top_cluster_idx]
        max_other = max(other_scores) if other_scores else 0.0
        if max_other > 0 and top_score < max_other * min_ratio:
            return list(range(C)), len(clusters), [], C, top_score, max_other
        dropped_local = [i for i in range(C) if i not in top_cluster]
        return sorted(top_cluster), len(clusters), dropped_local, len(top_cluster), top_score, max_other

    def retrieve_multi(
        self,
        xq,
        fq,
        topk=None,
        bw=None,
        update_stats=True,
        query_semantic_emb=None,
        query_content_ids_per_batch=None,
        wte_normed=None,
        content_classifier=None,
    ):
        B = xq.shape[0]
        dev = xq.device
        topk = topk or self.c.retrieval_topk
        bw = bw or self.c.retrieval_beam
        recall_k = int(topk * self.c.retrieval_recall_factor)
        flat_thresh = self.c.flat_scan_threshold_factor * topk
        qdir = self.dir_pred(xq, fq)
        diag = RetrievalDiag()
        corpus_idf = self._compute_corpus_idf(content_classifier) if self.c.use_idf_retrieval else None
        diag.idf_applied = corpus_idf is not None
        diag.centroid_applied = self.c.use_idf_centroid
        idf_floor = self.c.idf_floor

        if not self.tree.store:
            empty = self.empty_state(xq, fq)
            mask = torch.ones(B, 1, **_dev(xq))
            summary = empty.mean(1) if empty.dim() == 3 else empty
            diag.fiber_summary_norm = summary.norm().item()
            diag.batch_mem_weights = [[] for _ in range(B)]
            diag.dominant_per_batch = [None for _ in range(B)]
            return empty.unsqueeze(1), mask, summary, diag

        all_results = []
        all_masks = []
        all_biases = []
        all_summaries = []
        all_batch_mw = []
        all_dominant = []
        wn = wte_normed if wte_normed is not None else self.wte_normed

        for b in range(B):
            n_store = len(self.tree.store)
            if n_store <= flat_thresh:
                mids = list(self.tree.store.keys())
                diag.was_flat_scan = True
            else:
                scored = self.tree.retrieve(qdir[b].detach(), bw)
                mids = [s[0] for s in scored[:recall_k]]
            mems = [self.tree.store[i] for i in mids if i in self.tree.store]
            diag.recall_count = len(mems)
            diag.n_candidates_initial = len(mems)
            if not mems:
                empty = self.empty_state(xq[b : b + 1], fq[b : b + 1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1, **_dev(xq)))
                all_biases.append(torch.zeros(1, **_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([])
                all_dominant.append(None)
                continue

            q_content_ids = query_content_ids_per_batch[b] if query_content_ids_per_batch and b < len(query_content_ids_per_batch) else []
            q_strict = []
            if content_classifier is not None:
                q_strict = [
                    t
                    for t in q_content_ids
                    if t in content_classifier.strict_content_starter_ids and wn is not None and t < wn.shape[0]
                ]
            if self.c.use_strict_content_overlap_gate and q_strict and wn is not None and content_classifier is not None:
                overlap_counts = torch.zeros(len(mems), dtype=torch.long, device=dev)
                for mi, mem in enumerate(mems):
                    m_strict = [
                        t
                        for t in mem.content_token_ids
                        if t in content_classifier.strict_content_starter_ids and t < wn.shape[0]
                    ]
                    cnt = self._count_strict_overlap_matches(
                        q_strict, m_strict, wn, self.c.strict_overlap_sim_threshold
                    )
                    overlap_counts[mi] = cnt
                    diag.per_memory_strict_overlap[mem.mid] = cnt
                pass_mask = overlap_counts >= self.c.strict_overlap_min_matches
                n_pass = int(pass_mask.sum().item())
                if n_pass < self.c.strict_overlap_min_keep:
                    keep_n = max(self.c.strict_overlap_min_keep, 1)
                    _, top_keep = overlap_counts.topk(min(keep_n, len(mems)))
                    pass_mask = torch.zeros(len(mems), dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                diag.strict_overlap_dropped_ids = [mems[i].mid for i in dropped_local]
                diag.strict_overlap_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < len(mems):
                    mems = [mems[i] for i in keep_local.tolist()]
            diag.n_after_strict_overlap_gate = len(mems)
            if self.c.use_strict_avg_maxsim_gate and q_strict and wn is not None and content_classifier is not None:
                strict_avg_scores = torch.zeros(len(mems), device=dev)
                for mi, mem in enumerate(mems):
                    m_strict = [
                        t
                        for t in mem.content_token_ids
                        if t in content_classifier.strict_content_starter_ids and t < wn.shape[0]
                    ]
                    score = self._compute_strict_avg_maxsim(q_strict, m_strict, wn)
                    strict_avg_scores[mi] = score
                    diag.per_memory_strict_avg_maxsim[mem.mid] = score
                pass_mask = strict_avg_scores >= self.c.strict_avg_maxsim_threshold
                n_pass = int(pass_mask.sum().item())
                if n_pass < self.c.strict_avg_maxsim_min_keep:
                    keep_n = max(self.c.strict_avg_maxsim_min_keep, 1)
                    _, top_keep = strict_avg_scores.topk(min(keep_n, len(mems)))
                    pass_mask = torch.zeros(len(mems), dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                diag.strict_avg_maxsim_dropped_ids = [mems[i].mid for i in dropped_local]
                diag.strict_avg_maxsim_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < len(mems):
                    mems = [mems[i] for i in keep_local.tolist()]
            diag.n_after_strict_avg_maxsim_gate = len(mems)
            if (
                self.c.use_strict_avg_maxsim_relative_floor
                and q_strict
                and wn is not None
                and content_classifier is not None
                and len(mems) >= 2
            ):
                cur_avg = torch.tensor(
                    [diag.per_memory_strict_avg_maxsim.get(mem.mid, 0.0) for mem in mems],
                    device=dev,
                )
                top_avg = cur_avg.max().item()
                if top_avg >= self.c.strict_avg_maxsim_relative_min_top:
                    threshold = max(
                        self.c.strict_avg_maxsim_threshold,
                        top_avg * self.c.strict_avg_maxsim_relative_ratio,
                    )
                    pass_mask = cur_avg >= threshold
                    n_pass = int(pass_mask.sum().item())
                    if n_pass < self.c.strict_avg_maxsim_relative_min_keep:
                        keep_n = max(self.c.strict_avg_maxsim_relative_min_keep, 1)
                        _, top_keep = cur_avg.topk(min(keep_n, len(mems)))
                        pass_mask = torch.zeros(len(mems), dtype=torch.bool, device=dev)
                        pass_mask[top_keep] = True
                    dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                    if dropped_local:
                        diag.strict_avg_maxsim_relative_floor_applied = True
                        diag.strict_avg_maxsim_relative_dropped_ids = [mems[i].mid for i in dropped_local]
                        keep_local = pass_mask.nonzero(as_tuple=True)[0]
                        mems = [mems[i] for i in keep_local.tolist()]
            diag.n_after_strict_avg_maxsim_relative_floor = len(mems)
            C_init = len(mems)
            sb_all = torch.stack([m.base.to(dev) for m in mems])
            sf_all = torch.stack([m.fiber.to(dev) for m in mems])
            md_all = torch.stack([m.dirn.to(dev) for m in mems])

            sem_sim_all = torch.zeros(C_init, device=dev)
            if query_semantic_emb is not None:
                for mi, mem in enumerate(mems):
                    if mem.semantic_emb is not None:
                        sem_sim_all[mi] = F.cosine_similarity(
                            query_semantic_emb[b : b + 1], mem.semantic_emb.unsqueeze(0).to(dev), dim=-1
                        ).squeeze()

            forward_idf_all = torch.zeros(C_init, device=dev)
            bidi_min_all = torch.zeros(C_init, device=dev)
            forward_all = torch.zeros(C_init, device=dev)
            backward_all = torch.zeros(C_init, device=dev)
            strict_avg_all = torch.zeros(C_init, device=dev)
            if q_content_ids and wn is not None:
                for mi, mem in enumerate(mems):
                    scoring_ids = self._get_mem_scoring_ids(mem)
                    fwd_idf = self._compute_forward_maxsim(
                        q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor
                    )
                    bwd_idf = self._compute_backward_maxsim(
                        q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor
                    )
                    forward_all[mi] = fwd_idf
                    backward_all[mi] = bwd_idf
                    forward_idf_all[mi] = fwd_idf
                    bidi_min_all[mi] = min(fwd_idf, bwd_idf)
                    if q_strict and content_classifier is not None:
                        m_strict = [
                            t
                            for t in mem.content_token_ids
                            if t in content_classifier.strict_content_starter_ids and t < wn.shape[0]
                        ]
                        strict_avg_all[mi] = self._compute_strict_avg_maxsim(q_strict, m_strict, wn)

            if self.c.use_upstream_semantic_gate and q_content_ids and wn is not None:
                fwd_pass = forward_idf_all >= self.c.upstream_gate_fwd_idf_floor
                sem_pass = sem_sim_all >= self.c.upstream_gate_sem_floor
                pass_mask = (fwd_pass & sem_pass) if self.c.upstream_gate_require_both else (fwd_pass | sem_pass)
                n_pass = int(pass_mask.sum().item())
                if n_pass < self.c.upstream_gate_min_keep:
                    keep_n = max(self.c.upstream_gate_min_keep, 1)
                    top_keep = forward_idf_all.topk(min(keep_n, C_init)).indices
                    pass_mask = torch.zeros(C_init, dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                if dropped_local:
                    diag.upstream_gate_dropped_ids = [mems[i].mid for i in dropped_local]
                diag.upstream_semantic_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C_init:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb_all = sb_all[keep_local]
                    sf_all = sf_all[keep_local]
                    md_all = md_all[keep_local]
                    sem_sim_all = sem_sim_all[keep_local]
                    forward_all = forward_all[keep_local]
                    backward_all = backward_all[keep_local]
                    forward_idf_all = forward_idf_all[keep_local]
                    bidi_min_all = bidi_min_all[keep_local]
                    strict_avg_all = strict_avg_all[keep_local]
                    C_init = len(mems)
            diag.n_after_upstream_semantic_gate = C_init

            sb = sb_all
            sf = sf_all
            sem_sim_t = sem_sim_all
            forward_t = forward_all
            backward_t = backward_all
            forward_idf_t = forward_idf_all
            bidi_min_t = bidi_min_all
            strict_avg_t = strict_avg_all
            raw_dir_sim = torch.einsum("d,cd->c", qdir[b], md_all)
            diag.top_dir_sim = raw_dir_sim.max().item() if C_init > 0 else 0.0
            diag.top_sem_sim = sem_sim_t.max().item() if C_init > 0 else 0.0
            diag.top_forward_maxsim = forward_t.max().item() if C_init > 0 else 0.0
            diag.top_backward_maxsim = backward_t.max().item() if C_init > 0 else 0.0
            diag.top_bidi_min = bidi_min_t.max().item() if C_init > 0 else 0.0
            diag.top_forward_maxsim_idf = forward_idf_t.max().item() if C_init > 0 else 0.0
            diag.top_bidi_min_idf = bidi_min_t.max().item() if C_init > 0 else 0.0

            centroid_scores = torch.zeros(C_init, device=dev)
            if self.c.use_idf_centroid and q_content_ids and wn is not None:
                q_centroid = self._compute_idf_weighted_centroid(q_content_ids, wn, corpus_idf, idf_floor)
                if q_centroid is not None:
                    for mi, mem in enumerate(mems):
                        m_scoring_ids = self._get_mem_scoring_ids(mem)
                        m_centroid = self._compute_idf_weighted_centroid(m_scoring_ids, wn, corpus_idf, idf_floor)
                        centroid_scores[mi] = self._compute_centroid_cosine(q_centroid, m_centroid)
                diag.top_centroid_cosine = centroid_scores.max().item() if C_init > 0 else 0.0

            combined_sim = (
                self.c.ret_centroid_weight * centroid_scores
                + self.c.ret_sem_weight * sem_sim_t
                + self.c.ret_bidi_min_weight * bidi_min_t
                + self.c.ret_forward_maxsim_weight * forward_t
                + self.c.ret_dir_weight * raw_dir_sim
            )
            C = C_init

            top_sem = sem_sim_t.max().item() if C > 0 else 0.0
            top_bidi = bidi_min_t.max().item() if C > 0 else 0.0
            sem_thresh = max(self.c.gate_sem_floor, top_sem * self.c.gate_sem_ratio)
            bidi_thresh = max(
                self.c.gate_bidi_floor, top_bidi * self.c.gate_bidi_ratio, self.c.gate_bidi_hard_min
            )
            hard_mask = (sem_sim_t >= sem_thresh) & (bidi_min_t >= bidi_thresh)
            gate_affinity = self.c.gate_sem_weight * sem_sim_t + self.c.gate_bidi_weight * bidi_min_t
            diag.top_gate_affinity = gate_affinity.max().item() if C > 0 else 0.0
            diag.gate_threshold = max(sem_thresh, bidi_thresh)
            diag.n_gate_pass = int(hard_mask.sum().item())
            if hard_mask.sum().item() == 0 and C > 0:
                and_score = torch.minimum(sem_sim_t, bidi_min_t)
                hard_mask[and_score.argmax()] = True
            diag.n_after_hard_filter = int(hard_mask.sum().item())
            for mi, mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid] = gate_affinity[mi].item()
            keep_indices = hard_mask.nonzero(as_tuple=True)[0]
            if keep_indices.numel() > 0 and keep_indices.numel() < C:
                mems = [mems[i] for i in keep_indices.tolist()]
                sb = sb[keep_indices]
                sf = sf[keep_indices]
                combined_sim = combined_sim[keep_indices]
                raw_dir_sim = raw_dir_sim[keep_indices]
                forward_t = forward_t[keep_indices]
                bidi_min_t = bidi_min_t[keep_indices]
                sem_sim_t = sem_sim_t[keep_indices]
                forward_idf_t = forward_idf_t[keep_indices]
                centroid_scores = centroid_scores[keep_indices]
                strict_avg_t = strict_avg_t[keep_indices]
                C = len(mems)

            rerank_scores = self.reranker(
                xq[b : b + 1], fq[b : b + 1], sb.unsqueeze(0), sf.unsqueeze(0), combined_sim.unsqueeze(0)
            ).squeeze(0)
            diag.reranker_delta_mean = (rerank_scores - combined_sim).abs().mean().item()
            diag.top_reranker_score = rerank_scores.max().item() if C > 0 else 0.0

            if C > 1:
                top_score = rerank_scores.max()
                score_thresh = top_score * self.c.score_keep_ratio
                score_mask = rerank_scores >= score_thresh
                if score_mask.sum().item() < 1:
                    score_mask[rerank_scores.argmax()] = True
                score_keep = score_mask.nonzero(as_tuple=True)[0]
                diag.n_after_score_filter = score_keep.numel()
                if score_keep.numel() < C:
                    mems = [mems[i] for i in score_keep.tolist()]
                    sb = sb[score_keep]
                    sf = sf[score_keep]
                    rerank_scores = rerank_scores[score_keep]
                    forward_t = forward_t[score_keep]
                    bidi_min_t = bidi_min_t[score_keep]
                    sem_sim_t = sem_sim_t[score_keep]
                    forward_idf_t = forward_idf_t[score_keep]
                    centroid_scores = centroid_scores[score_keep]
                    strict_avg_t = strict_avg_t[score_keep]
                    C = len(mems)
            else:
                diag.n_after_score_filter = C

            if C > 1 and forward_t.max().item() > 0:
                top_fwd_here = forward_t.max()
                coherence_mask = forward_t >= top_fwd_here * self.c.fwd_coherence_ratio
                if coherence_mask.sum() >= 1:
                    coherence_keep = coherence_mask.nonzero(as_tuple=True)[0]
                    diag.n_after_coherence_filter = coherence_keep.numel()
                    if coherence_keep.numel() < C:
                        mems = [mems[i] for i in coherence_keep.tolist()]
                        sb = sb[coherence_keep]
                        sf = sf[coherence_keep]
                        rerank_scores = rerank_scores[coherence_keep]
                        forward_t = forward_t[coherence_keep]
                        bidi_min_t = bidi_min_t[coherence_keep]
                        sem_sim_t = sem_sim_t[coherence_keep]
                        forward_idf_t = forward_idf_t[coherence_keep]
                        centroid_scores = centroid_scores[coherence_keep]
                        strict_avg_t = strict_avg_t[coherence_keep]
                        C = len(mems)
                else:
                    diag.n_after_coherence_filter = C
            else:
                diag.n_after_coherence_filter = C

            if C > 1 and bidi_min_t.max().item() > 0:
                top_bidi_here = bidi_min_t.max().item()
                gap_mask = bidi_min_t >= (top_bidi_here - self.c.bidi_absolute_gap)
                if gap_mask.sum() >= 1:
                    gap_keep = gap_mask.nonzero(as_tuple=True)[0]
                    diag.n_after_bidi_gap_filter = gap_keep.numel()
                    if gap_keep.numel() < C:
                        mems = [mems[i] for i in gap_keep.tolist()]
                        sb = sb[gap_keep]
                        sf = sf[gap_keep]
                        rerank_scores = rerank_scores[gap_keep]
                        forward_t = forward_t[gap_keep]
                        bidi_min_t = bidi_min_t[gap_keep]
                        sem_sim_t = sem_sim_t[gap_keep]
                        forward_idf_t = forward_idf_t[gap_keep]
                        centroid_scores = centroid_scores[gap_keep]
                        strict_avg_t = strict_avg_t[gap_keep]
                        C = len(mems)
                else:
                    diag.n_after_bidi_gap_filter = C
            else:
                diag.n_after_bidi_gap_filter = C

            if self.c.use_domain_conflict_resolver and C >= 2 and content_classifier is not None:
                (
                    top_cluster_indices,
                    n_clusters,
                    dropped_local,
                    top_cluster_size,
                    top_score,
                    second_score,
                ) = self._resolve_domain_conflict(
                    mems, forward_idf_t, strict_avg_t, content_classifier, self.c.domain_conflict_jaccard_threshold
                )
                diag.domain_conflict_cluster_count = n_clusters
                diag.domain_conflict_top_cluster_size = top_cluster_size
                diag.domain_conflict_top_score = top_score
                diag.domain_conflict_second_score = second_score
                if dropped_local:
                    diag.domain_conflict_resolver_applied = True
                    diag.domain_conflict_dropped_ids = [mems[i].mid for i in dropped_local]
                    keep_t = torch.tensor(top_cluster_indices, device=dev, dtype=torch.long)
                    mems = [mems[i] for i in top_cluster_indices]
                    sb = sb[keep_t]
                    sf = sf[keep_t]
                    rerank_scores = rerank_scores[keep_t]
                    forward_t = forward_t[keep_t]
                    bidi_min_t = bidi_min_t[keep_t]
                    sem_sim_t = sem_sim_t[keep_t]
                    forward_idf_t = forward_idf_t[keep_t]
                    centroid_scores = centroid_scores[keep_t]
                    strict_avg_t = strict_avg_t[keep_t]
                    C = len(mems)
            diag.n_after_domain_conflict_resolver = C

            if self.c.use_post_gate_fwd_idf_floor and C > 0:
                pass_mask = forward_idf_t >= self.c.post_gate_fwd_idf_floor
                n_pass = int(pass_mask.sum().item())
                if n_pass < self.c.post_gate_fwd_idf_min_keep:
                    keep_n = max(self.c.post_gate_fwd_idf_min_keep, 1)
                    _, top_keep = forward_idf_t.topk(min(keep_n, C))
                    pass_mask = torch.zeros(C, dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                if dropped_local:
                    diag.post_gate_fwd_idf_floor_applied = True
                    diag.post_gate_fwd_idf_dropped_ids = [mems[i].mid for i in dropped_local]
                    keep_local = pass_mask.nonzero(as_tuple=True)[0]
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb = sb[keep_local]
                    sf = sf[keep_local]
                    rerank_scores = rerank_scores[keep_local]
                    forward_t = forward_t[keep_local]
                    bidi_min_t = bidi_min_t[keep_local]
                    sem_sim_t = sem_sim_t[keep_local]
                    forward_idf_t = forward_idf_t[keep_local]
                    centroid_scores = centroid_scores[keep_local]
                    strict_avg_t = strict_avg_t[keep_local]
                    C = len(mems)
            diag.n_after_post_gate_fwd_idf_floor = C
            if self.c.use_fwd_idf_relative_floor and C >= 2:
                top_fwd = forward_idf_t.max().item()
                if top_fwd >= self.c.fwd_idf_relative_min_top:
                    threshold = max(
                        self.c.post_gate_fwd_idf_floor,
                        top_fwd * self.c.fwd_idf_relative_ratio,
                    )
                    pass_mask = forward_idf_t >= threshold
                    n_pass = int(pass_mask.sum().item())
                    if n_pass < self.c.fwd_idf_relative_min_keep:
                        keep_n = max(self.c.fwd_idf_relative_min_keep, 1)
                        _, top_keep = forward_idf_t.topk(min(keep_n, C))
                        pass_mask = torch.zeros(C, dtype=torch.bool, device=dev)
                        pass_mask[top_keep] = True
                    dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                    if dropped_local:
                        diag.fwd_idf_relative_floor_applied = True
                        diag.fwd_idf_relative_dropped_ids = [mems[i].mid for i in dropped_local]
                        keep_local = pass_mask.nonzero(as_tuple=True)[0]
                        mems = [mems[i] for i in keep_local.tolist()]
                        sb = sb[keep_local]
                        sf = sf[keep_local]
                        rerank_scores = rerank_scores[keep_local]
                        forward_t = forward_t[keep_local]
                        bidi_min_t = bidi_min_t[keep_local]
                        sem_sim_t = sem_sim_t[keep_local]
                        forward_idf_t = forward_idf_t[keep_local]
                        centroid_scores = centroid_scores[keep_local]
                        strict_avg_t = strict_avg_t[keep_local]
                        C = len(mems)
            diag.n_after_fwd_idf_relative_floor = C

            dominant_mid = None
            if self.c.use_centroid_dominance and C >= 2 and centroid_scores.max().item() > 0:
                if self.c.use_quadruple_consensus and q_content_ids and wn is not None:
                    votes = self._compute_token_majority_votes(
                        q_content_ids,
                        mems,
                        wn,
                        corpus_idf,
                        content_classifier=content_classifier,
                        topk=self.c.consensus_token_vote_topk,
                        idf_floor=idf_floor,
                    )
                else:
                    votes = torch.zeros(C, device=dev)
                if self.c.use_cluster_vote_aggregation and self.c.use_quadruple_consensus and content_classifier is not None:
                    cluster_votes = self._compute_cluster_votes(
                        votes, mems, content_classifier, self.c.cluster_vote_jaccard_threshold
                    )
                    diag.cluster_vote_aggregation_applied = True
                else:
                    cluster_votes = votes

                combined_dom_scores = centroid_scores + self.c.consensus_vote_weight * cluster_votes
                comb_sorted, comb_idx = torch.sort(combined_dom_scores, descending=True)
                top1_c_idx = comb_idx[0].item()
                pure_cent_top1 = centroid_scores.argmax().item()
                diag.consensus_vote_reassigned = top1_c_idx != pure_cent_top1
                top1_c = comb_sorted[0].item()
                top2_c = comb_sorted[1].item() if C >= 2 else 0.0
                cent_margin = top1_c / max(top2_c, 1e-6) if top2_c > 0 else float("inf")
                diag.dominance_centroid_margin_observed = cent_margin
                diag.consensus_combined_margin = cent_margin
                top1_raw_centroid = centroid_scores[top1_c_idx].item()
                centroid_cond = (
                    top1_raw_centroid >= self.c.dominance_centroid_top1_floor
                    and cent_margin >= self.c.dominance_centroid_margin
                )

                consensus_cond = True
                if self.c.use_triple_consensus_dominance and centroid_cond:
                    if forward_idf_t.max().item() > 0:
                        fwd_ranks = torch.argsort(forward_idf_t, descending=True)
                        pos = (fwd_ranks == top1_c_idx).nonzero(as_tuple=True)[0]
                        if pos.numel() > 0:
                            diag.consensus_fwd_rank = int(pos[0].item())
                            if pos[0].item() >= self.c.consensus_fwd_rank_max:
                                consensus_cond = False
                        else:
                            diag.consensus_fwd_rank = -1
                            consensus_cond = False
                    else:
                        consensus_cond = False
                    if consensus_cond and content_classifier is not None:
                        top1_mem = mems[top1_c_idx]
                        strict_label = self._mem_strict_label_set(top1_mem, content_classifier)
                        diag.consensus_label_size = len(strict_label)
                        if len(strict_label) < self.c.consensus_label_size_min:
                            consensus_cond = False

                vote_cond = True
                top1_raw_vote = votes[top1_c_idx].item() if votes.max() > 0 else 0.0
                top1_cluster_vote = cluster_votes[top1_c_idx].item() if cluster_votes.max() > 0 else 0.0
                diag.consensus_top1_vote_ratio = top1_raw_vote
                diag.consensus_top1_cluster_vote_ratio = top1_cluster_vote
                for mi, mem in enumerate(mems):
                    diag.per_memory_vote_ratio[mem.mid] = votes[mi].item()
                    diag.per_memory_cluster_vote_ratio[mem.mid] = cluster_votes[mi].item()

                n_q_strict = 0
                if content_classifier is not None:
                    n_q_strict = sum(1 for t in q_content_ids if t in content_classifier.strict_content_starter_ids)
                diag.consensus_query_strict_size = n_q_strict
                if self.c.use_adaptive_consensus_threshold:
                    ref = max(self.c.consensus_threshold_query_size_ref, 1)
                    ratio = min(1.0, max(n_q_strict, 0) / ref)
                    ratio = max(ratio, self.c.consensus_threshold_min_ratio)
                    effective_threshold = self.c.consensus_token_vote_threshold * ratio
                else:
                    effective_threshold = self.c.consensus_token_vote_threshold
                diag.consensus_effective_threshold = effective_threshold
                if self.c.use_quadruple_consensus and top1_cluster_vote < effective_threshold:
                    vote_cond = False

                diag.consensus_passed = centroid_cond and consensus_cond and vote_cond
                if diag.consensus_passed:
                    diag.dominance_triggered = True
                    diag.centroid_dominance_triggered = True
                    dominant_mid = mems[top1_c_idx].mid
                    keep_thresh = top1_c * self.c.consensus_strict_keep_ratio
                    keep_mask = combined_dom_scores >= keep_thresh
                    keep_mask[top1_c_idx] = True
                    keep_local = keep_mask.nonzero(as_tuple=True)[0]
                    if keep_local.numel() < C:
                        mems = [mems[i] for i in keep_local.tolist()]
                        sb = sb[keep_local]
                        sf = sf[keep_local]
                        rerank_scores = rerank_scores[keep_local]
                        forward_t = forward_t[keep_local]
                        bidi_min_t = bidi_min_t[keep_local]
                        sem_sim_t = sem_sim_t[keep_local]
                        forward_idf_t = forward_idf_t[keep_local]
                        centroid_scores = centroid_scores[keep_local]
                        strict_avg_t = strict_avg_t[keep_local]
                        C = len(mems)

            if self.c.use_idf_dominance and C >= 2 and forward_idf_t.max().item() > 0:
                fwd_sorted, fwd_sort_idx = torch.sort(forward_idf_t, descending=True)
                top1_idx = fwd_sort_idx[0].item()
                top1_fwd = fwd_sorted[0].item()
                top2_fwd = fwd_sorted[1].item()
                idf_margin = top1_fwd / max(top2_fwd, 1e-6)
                diag.dominance_idf_margin_observed = idf_margin
                if top1_fwd >= self.c.dominance_idf_top1_floor and idf_margin >= self.c.dominance_idf_margin:
                    diag.dominance_triggered = True
                    if dominant_mid is None:
                        dominant_mid = mems[top1_idx].mid
                    keep_thresh = top1_fwd / self.c.dominance_idf_margin
                    keep_mask = forward_idf_t >= keep_thresh
                    keep_mask[top1_idx] = True
                    keep_local = keep_mask.nonzero(as_tuple=True)[0]
                    if keep_local.numel() < C:
                        mems = [mems[i] for i in keep_local.tolist()]
                        sb = sb[keep_local]
                        sf = sf[keep_local]
                        rerank_scores = rerank_scores[keep_local]
                        forward_t = forward_t[keep_local]
                        bidi_min_t = bidi_min_t[keep_local]
                        sem_sim_t = sem_sim_t[keep_local]
                        forward_idf_t = forward_idf_t[keep_local]
                        centroid_scores = centroid_scores[keep_local]
                        strict_avg_t = strict_avg_t[keep_local]
                        C = len(mems)

            diag.n_after_dominance_filter = C
            if self.c.use_final_domain_purge and C >= 2 and content_classifier is not None:
                (
                    top_cluster_indices,
                    _n_clusters,
                    dropped_local,
                    _top_cluster_size,
                    top_score,
                    second_score,
                ) = self._resolve_domain_conflict(
                    mems,
                    forward_idf_t,
                    strict_avg_t,
                    content_classifier,
                    self.c.final_domain_purge_jaccard,
                    min_ratio=self.c.final_domain_purge_margin,
                )
                diag.final_domain_purge_top_score = top_score
                diag.final_domain_purge_second_score = second_score
                if dropped_local:
                    diag.final_domain_purge_applied = True
                    diag.final_domain_purge_dropped_ids = [mems[i].mid for i in dropped_local]
                    keep_t = torch.tensor(top_cluster_indices, device=dev, dtype=torch.long)
                    mems = [mems[i] for i in top_cluster_indices]
                    sb = sb[keep_t]
                    sf = sf[keep_t]
                    rerank_scores = rerank_scores[keep_t]
                    forward_t = forward_t[keep_t]
                    bidi_min_t = bidi_min_t[keep_t]
                    sem_sim_t = sem_sim_t[keep_t]
                    forward_idf_t = forward_idf_t[keep_t]
                    centroid_scores = centroid_scores[keep_t]
                    strict_avg_t = strict_avg_t[keep_t]
                    C = len(mems)
            diag.n_after_final_domain_purge = C

            if not self.training and C > topk:
                _, top_idx = rerank_scores.topk(topk)
                mems = [mems[i] for i in top_idx.cpu().tolist()]
                sb = sb[top_idx]
                sf = sf[top_idx]
                rerank_scores = rerank_scores[top_idx]
                forward_t = forward_t[top_idx]
                bidi_min_t = bidi_min_t[top_idx]
                sem_sim_t = sem_sim_t[top_idx]
                forward_idf_t = forward_idf_t[top_idx]
                centroid_scores = centroid_scores[top_idx]
                strict_avg_t = strict_avg_t[top_idx]
                C = topk

            for mi, mem in enumerate(mems):
                diag.per_memory_forward_maxsim[mem.mid] = forward_t[mi].item()
                diag.per_memory_bidi_min[mem.mid] = bidi_min_t[mi].item()
                diag.per_memory_sem_sim[mem.mid] = sem_sim_t[mi].item()
                diag.per_memory_forward_maxsim_idf[mem.mid] = forward_idf_t[mi].item()
                diag.per_memory_centroid_cosine[mem.mid] = centroid_scores[mi].item()

            qp = xq[b].unsqueeze(0).expand(C, -1)
            geo_r = self.geo.solve(sb, qp)
            transported = self.trans(sf, geo_r.path)
            if self.training:
                ret_s = self.retention(
                    sb,
                    sf,
                    torch.tensor([m.surprise for m in mems], **_dev(xq)),
                    torch.tensor([self.time - m.last for m in mems], **_dev(xq)),
                    torch.tensor([m.cnt for m in mems], **_dev(xq)),
                )
                transported = transported * ret_s.unsqueeze(-1)
            if update_stats:
                for m in mems:
                    m.last = self.time
                    m.cnt += 1

            if self.c.use_idf_centroid and centroid_scores.max().item() > 0:
                final_scores = 0.4 * rerank_scores + 0.4 * centroid_scores + 0.2 * forward_idf_t
            elif self.c.use_idf_retrieval and forward_idf_t.max().item() > 0:
                final_scores = 0.5 * rerank_scores + 0.5 * forward_idf_t
            else:
                final_scores = rerank_scores
            w = F.softmax(final_scores / self.c.retrieval_weight_temperature, dim=0)
            fs = (transported * w.unsqueeze(-1)).sum(0)
            batch_mw = [(m.mid, w[mi].item()) for mi, m in enumerate(mems)]
            all_batch_mw.append(batch_mw)
            all_dominant.append(dominant_mid)
            all_results.append(transported)
            all_masks.append(torch.ones(C, **_dev(xq)))
            all_biases.append(final_scores / self.c.tau)
            all_summaries.append(fs)

        maxC = max(r.shape[0] for r in all_results)
        padded = []
        pm = []
        pd = []
        for bi in range(B):
            r, mk, db = all_results[bi], all_masks[bi], all_biases[bi]
            gap = maxC - r.shape[0]
            if gap > 0:
                pr = self.empty_state(xq[bi : bi + 1], fq[bi : bi + 1]).expand(gap, -1)
                r = torch.cat([r, pr if self.training else pr.detach()], 0)
                mk = torch.cat([mk, torch.zeros(gap, **_dev(xq))])
                db = torch.cat([db, torch.full((gap,), -1e9, **_dev(xq))])
            padded.append(r)
            pm.append(mk)
            pd.append(db)
        mf = torch.stack(padded)
        mem_mask = torch.stack(pm)
        dir_bias = torch.stack(pd)
        fiber_summary = torch.stack(all_summaries)
        diag.fiber_summary_norm = fiber_summary.norm().item()
        diag.batch_mem_weights = all_batch_mw
        diag.dominant_per_batch = all_dominant
        if diag.dominant_per_batch and diag.dominant_per_batch[0] is not None:
            diag.dominant_memory_id = diag.dominant_per_batch[0]
        refined = self.attn(fq, mf, mem_mask=mem_mask, dir_bias=dir_bias)
        return refined, mem_mask, fiber_summary, diag


class MemLLM(v323.MemLLM):
    def __init__(self, c):
        super().__init__(c)
        self.amm = AMM(c)
        self.bridge = EmbBridge(c)
        self._filler_centroid = None

    def load(self, name="gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tok = GPT2Tokenizer.from_pretrained(name)
        self.llm = GPT2LMHeadModel.from_pretrained(name)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layer_pool = AdaptiveLayerPool(self.llm.config.n_layer + 1, self.c.d_LLM)
        self.content_classifier = ContentTokenClassifier(self.tok, self.c)
        self._degen_guard = DegenerationGuard(self.tok, self.c, self.content_classifier)
        self.bridge.aligner.calibrate(self.llm)
        self.c.vocab_size = self.llm.config.vocab_size
        self._wte_normed = F.normalize(self.llm.transformer.wte.weight.detach(), dim=-1, eps=1e-8)
        self.amm.wte_normed = self._wte_normed
        self._build_wte_neighbor_cache()
        self._compute_filler_centroid()

    def _compute_filler_centroid(self):
        if self.content_classifier is None or self.llm is None:
            self._filler_centroid = None
            return
        wte = self.llm.transformer.wte.weight.detach()
        valid = [tid for tid in sorted(self.content_classifier.filler_ids) if tid < wte.shape[0]]
        if len(valid) < 3:
            self._filler_centroid = None
            return
        filler_vecs = wte[torch.tensor(valid, device=wte.device)]
        self._filler_centroid = F.normalize(filler_vecs.mean(0), dim=-1, eps=1e-8)

    def _compute_strict_anchor_boost(self, diag, query_content_ids_per_batch):
        V = self.c.vocab_size
        dev = next(self.parameters()).device
        cc = self.content_classifier
        if cc is None or not self.c.use_strict_anchor_boost or not diag.batch_mem_weights:
            return torch.zeros(len(diag.batch_mem_weights), V, device=dev)
        idf = self._compute_tfidf_idf() if self.c.use_tfidf_weighting else {}
        boost = torch.zeros(len(diag.batch_mem_weights), V, device=dev)
        for b in range(len(diag.batch_mem_weights)):
            dom_mid = diag.dominant_per_batch[b] if b < len(diag.dominant_per_batch) else None
            if dom_mid is None or dom_mid not in self.amm.tree.store:
                continue
            mem = self.amm.tree.store[dom_mid]
            strict_ids = [
                t
                for t in self.amm._get_mem_scoring_ids(mem)
                if t in cc.strict_content_starter_ids and t < V and t < self._wte_normed.shape[0]
            ]
            if not strict_ids:
                continue
            vals = torch.tensor([idf.get(t, 1.0) for t in strict_ids], device=dev)
            vals, idx = vals.topk(min(self.c.strict_anchor_boost_topk, len(strict_ids)))
            vals = vals / vals.max().clamp(min=1e-8)
            for i in range(len(idx)):
                boost[b, strict_ids[idx[i].item()]] = vals[i].item()
        return boost

    def _get_prefix(self, hs, mask=None, pl=0, update_stats=True, return_extra=False, ids=None):
        pooled, xq, fq = self.extract_state(hs, mask, pl)
        trimmed_mask = mask[:, pl:] if mask is not None and pl > 0 else mask
        if trimmed_mask is not None and pooled.shape[1] != trimmed_mask.shape[1]:
            trimmed_mask = None
        query_content_ids_per_batch = []
        if ids is not None and self.content_classifier is not None:
            for b in range(ids.shape[0]):
                query_content_ids_per_batch.append(
                    list(set(self.content_classifier.get_content_ids_from_tokens(ids[b].tolist())))
                )
        if ids is not None and self.content_classifier is not None:
            query_sem = self._compute_content_semantic_emb(pooled, ids, trimmed_mask)
        else:
            query_sem = pooled.mean(1)
        fibers, mem_mask, fiber_summary, diag = self.amm.retrieve_multi(
            xq,
            fq,
            update_stats=update_stats,
            query_semantic_emb=query_sem,
            query_content_ids_per_batch=query_content_ids_per_batch,
            wte_normed=self._wte_normed,
            content_classifier=self.content_classifier,
        )
        hard_wte_last, hard_mask_list, injected_tids = self._build_hard_wte_last_slots(
            diag, query_content_ids_per_batch
        )
        all_triggered = (
            hard_wte_last is not None and hard_mask_list is not None and all(hard_mask_list)
        )
        self._last_hard_injected_tids = injected_tids if all_triggered else None
        content_wte_mean, content_target_wte = self._compute_content_wte_topk(
            diag, query_content_ids_per_batch
        )
        has_cwm = content_wte_mean.abs().max().item() > 1e-6
        has_tgt = content_target_wte.abs().max().item() > 1e-6
        prefix = self.bridge.inject(
            fibers,
            mem_mask,
            fiber_summary=fiber_summary,
            content_wte_mean=content_wte_mean if has_cwm else None,
            content_target_wte=content_target_wte if has_tgt else None,
            hard_wte_last_slots=hard_wte_last if all_triggered else None,
            filler_centroid=self._filler_centroid,
        )
        content_bias = self._build_content_bias(diag, query_content_ids_per_batch)
        first_step_bias = self._build_first_step_lexical_bias(diag, query_content_ids_per_batch)
        strict_anchor_boost = self._compute_strict_anchor_boost(diag, query_content_ids_per_batch)
        if return_extra:
            return prefix, fiber_summary, diag, content_bias, first_step_bias, strict_anchor_boost
        return prefix

    def generate(self, prompt, mt=50, greedy=False):
        tk = self.tok(prompt, return_tensors="pt")
        dev = next(self.parameters()).device
        ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            prefix, fiber_summary, _, content_bias, first_step_bias, strict_anchor_boost = self._get_prefix(
                o["hs"], mask, update_stats=True, return_extra=True, ids=ids
            )
            vocab_bias = self._compute_vocab_bias(fiber_summary)
        cc = self.content_classifier
        hard_injected_tids: Set[int] = set()
        hard_inject_start_step = 0
        if self._last_hard_injected_tids is not None and self._last_hard_injected_tids:
            hard_injected_tids = set(self._last_hard_injected_tids[0])
        has_content = content_bias is not None and content_bias.abs().max().item() > 0.01
        domain_anchors = self._compute_domain_anchors(content_bias) if has_content else [[]]
        anchors_for_b0 = set(domain_anchors[0]) if domain_anchors else set()
        generated_anchors = set()
        filler_mask_vec = cc.filler_mask(dev) if cc is not None else None
        generated_ids = []
        generated_content_counts: Dict[int, int] = {}
        consecutive_content = 0
        recent_starters: List[Tuple[int, int]] = []
        newline_ids_set = cc.newline_ids if cc is not None else set()
        content_history: List[Tuple[int, int]] = []
        HARD_MASK = -1e9
        eos_token_id = self.tok.eos_token_id
        strict_mask_vec = cc.strict_content_starter_mask(dev) if cc is not None else None
        non_strict_content_mask_vec = cc.non_strict_content_mask(dev) if cc is not None else None
        punct_mask_vec = cc.punct_mask(dev) if cc is not None else None
        function_mask_vec = cc.function_mask(dev) if cc is not None else None
        for i in range(mt):
            if i > 0 and i % self.c.retrieval_interval == 0:
                with torch.no_grad():
                    o = self.fwd(ids, mask, prefix)
                    pl = o["pl"]
                    prefix, fiber_summary, _, content_bias, first_step_bias, strict_anchor_boost = self._get_prefix(
                        o["hs"], o["mask"], pl, update_stats=True, return_extra=True, ids=ids
                    )
                    vocab_bias = self._compute_vocab_bias(fiber_summary)
                    has_content = content_bias is not None and content_bias.abs().max().item() > 0.01
                    if has_content:
                        domain_anchors = self._compute_domain_anchors(content_bias)
                        anchors_for_b0 = set(domain_anchors[0]) if domain_anchors else set()
                    if self._last_hard_injected_tids is not None and self._last_hard_injected_tids:
                        hard_injected_tids = set(self._last_hard_injected_tids[0])
                        hard_inject_start_step = i
                    else:
                        hard_injected_tids = set()
            with torch.no_grad():
                o = self.fwd(ids, mask, prefix)
                lg = o["logits"][:, -1:].squeeze(1).clone()
                if first_step_bias is not None and i < self.c.first_step_lexical_decay_steps:
                    V = min(lg.shape[-1], first_step_bias.shape[-1])
                    lg[:, :V] += first_step_bias[:, :V] * self.c.first_step_lexical_scale
                if content_bias is not None:
                    V = min(lg.shape[-1], content_bias.shape[-1])
                    lg[:, :V] += content_bias[:, :V] * self.c.content_bias_scale
                if strict_anchor_boost is not None and i < self.c.strict_anchor_boost_steps:
                    V = min(lg.shape[-1], strict_anchor_boost.shape[-1])
                    scale = max(1.0 - i * self.c.strict_anchor_boost_decay, self.c.strict_anchor_boost_floor)
                    lg[:, :V] += strict_anchor_boost[:, :V] * self.c.strict_anchor_boost_scale * scale
                if vocab_bias is not None:
                    V = min(lg.shape[-1], vocab_bias.shape[-1])
                    lg[:, :V] += vocab_bias[:, :V] * self.c.semantic_boost_scale
                if i >= self.c.domain_anchor_start_step and anchors_for_b0 and has_content:
                    coverage = len(generated_anchors) / max(len(anchors_for_b0), 1)
                    if coverage < self.c.domain_anchor_coverage_threshold:
                        for tid in anchors_for_b0 - generated_anchors:
                            if tid < lg.shape[-1]:
                                lg[0, tid] += self.c.domain_anchor_boost
                if cc:
                    for tid, count in generated_content_counts.items():
                        if tid in cc.content_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.content_repeat_penalty * (count ** self.c.content_repeat_exponent)
                if self.c.use_cyclic_content_hard_mask and cc is not None:
                    window_counts: Dict[int, int] = {}
                    cutoff_step = i - self.c.cyclic_content_window
                    for step_idx, tid in content_history:
                        if step_idx >= cutoff_step:
                            window_counts[tid] = window_counts.get(tid, 0) + 1
                    for tid, cnt in window_counts.items():
                        if cnt >= self.c.cyclic_content_max_count and 0 <= tid < lg.shape[-1]:
                            lg[0, tid] = HARD_MASK
                if self.c.use_early_bigram_hard_mask and len(generated_ids) >= 2:
                    x_prev = generated_ids[-2]
                    y_prev = generated_ids[-1]
                    x_is_content = cc is not None and x_prev in cc.content_ids
                    if (not self.c.early_bigram_min_content_token) or x_is_content:
                        y_is_function = cc is not None and (y_prev in cc.function_ids or y_prev not in cc.content_ids)
                        if y_is_function and 0 <= x_prev < lg.shape[-1]:
                            lg[0, x_prev] = HARD_MASK
                if self.c.use_ngram_repeat_block and len(generated_ids) >= 4:
                    max_n = min(self.c.ngram_repeat_max_n, len(generated_ids) // 2)
                    for n in range(2, max_n + 1):
                        if generated_ids[-n:] == generated_ids[-2 * n : -n]:
                            expected_next = generated_ids[-n]
                            if 0 <= expected_next < lg.shape[-1]:
                                lg[0, expected_next] -= self.c.ngram_repeat_penalty
                if cc and self._wte_neighbor_cache is not None and recent_starters:
                    for prev_tid, _prev_step in recent_starters:
                        neighbors = self._wte_neighbor_cache.get(prev_tid, [])
                        for nid in neighbors:
                            if nid in cc.word_starter_ids:
                                continue
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.bpe_echo_penalty
                if cc and generated_ids and generated_ids[-1] in cc.content_starter_ids:
                    for tid in cc.content_ids:
                        if tid not in cc.word_starter_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.post_starter_nonstarter_penalty
                if (
                    self.c.use_post_inject_suppress
                    and hard_injected_tids
                    and (i - hard_inject_start_step) < self.c.post_inject_suppress_steps
                ):
                    local_step = i - hard_inject_start_step
                    decay_factor = 1.0 - local_step / max(self.c.post_inject_suppress_steps, 1)
                    pen = self.c.post_inject_suppress_penalty * decay_factor
                    for tid in hard_injected_tids:
                        if tid < lg.shape[-1]:
                            lg[0, tid] -= pen
                if self.c.use_strict_or_continuation and cc is not None and i < self.c.strict_or_cont_steps:
                    prev_is_strict_starter = len(generated_ids) > 0 and generated_ids[-1] in cc.strict_content_starter_ids
                    if not prev_is_strict_starter:
                        nsc_mask = cc.non_strict_content_mask(dev)
                        V = min(lg.shape[-1], nsc_mask.shape[0])
                        lg[0, :V] -= nsc_mask[:V] * self.c.strict_or_cont_penalty
                if (
                    self.c.use_early_non_strict_hard_penalty
                    and cc is not None
                    and i < self.c.early_non_strict_hard_penalty_steps
                    and non_strict_content_mask_vec is not None
                ):
                    V = min(lg.shape[-1], non_strict_content_mask_vec.shape[0])
                    lg[0, :V] -= non_strict_content_mask_vec[:V] * self.c.early_non_strict_hard_penalty
                if self.c.use_sustained_filler and filler_mask_vec is not None and i < self.c.sustained_filler_steps:
                    V = min(lg.shape[-1], filler_mask_vec.shape[0])
                    filler_decay = max(1.0 - i * self.c.sustained_filler_decay, 0.0)
                    lg[0, :V] -= filler_mask_vec[:V] * self.c.sustained_filler_penalty * filler_decay
                if (
                    self.c.use_early_punct_hard_mask
                    and cc is not None
                    and i < self.c.early_punct_hard_mask_steps
                    and punct_mask_vec is not None
                ):
                    V = min(lg.shape[-1], punct_mask_vec.shape[0])
                    lg[0, :V] = torch.where(
                        punct_mask_vec[:V] > 0.5,
                        torch.full_like(lg[0, :V], HARD_MASK),
                        lg[0, :V],
                    )
                if (
                    self.c.use_early_function_hard_mask
                    and cc is not None
                    and i < self.c.early_function_hard_mask_steps
                    and function_mask_vec is not None
                ):
                    V = min(lg.shape[-1], function_mask_vec.shape[0])
                    lg[0, :V] = torch.where(
                        function_mask_vec[:V] > 0.5,
                        torch.full_like(lg[0, :V], HARD_MASK),
                        lg[0, :V],
                    )
                if self.c.use_newline_hard_gate and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if i < self.c.newline_hard_gate_min_step or content_count_so_far < self.c.newline_hard_gate_min_content:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] = HARD_MASK
                if (
                    self.c.use_eos_hard_mask
                    and eos_token_id is not None
                    and i < self.c.eos_hard_mask_steps
                    and eos_token_id < lg.shape[-1]
                ):
                    lg[0, eos_token_id] = HARD_MASK
                if (
                    cc is not None
                    and i < self.c.extended_strict_restrict_steps
                    and strict_mask_vec is not None
                ):
                    V = min(lg.shape[-1], strict_mask_vec.shape[0])
                    strict_logits = lg[0, :V].clone()
                    strict_logits[strict_mask_vec[:V] < 0.5] = HARD_MASK
                    if strict_logits.max().item() > self.c.extended_strict_fallback_threshold:
                        lg[0, :V] = torch.where(
                            strict_mask_vec[:V] < 0.5,
                            torch.full_like(lg[0, :V], HARD_MASK),
                            lg[0, :V],
                        )
                    else:
                        cs_mask = cc.content_starter_mask(dev)
                        V2 = min(V, cs_mask.shape[0])
                        lg[0, :V2] = torch.where(
                            cs_mask[:V2] < 0.5,
                            torch.full_like(lg[0, :V2], HARD_MASK),
                            lg[0, :V2],
                        )
                if self.c.use_content_gated_newline and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if content_count_so_far < self.c.min_content_tokens_before_newline:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.late_newline_penalty
                if self._degen_guard is not None:
                    lg = self._degen_guard.process(lg, generated_ids, i)
                if greedy:
                    nxt = lg.argmax(-1, keepdim=True)
                else:
                    lg = lg / self.c.gen_temp
                    p = F.softmax(lg, -1)
                    sp, si = torch.sort(p, descending=True)
                    cs = torch.cumsum(sp, -1)
                    rm = cs - sp > self.c.gen_top_p
                    sp[rm] = 0
                    total = sp.sum(-1, keepdim=True)
                    if (total < 1e-10).any():
                        sp[:, 0] = 1.0
                        total = sp.sum(-1, keepdim=True)
                    sp = sp / total
                    nxt = si.gather(-1, torch.multinomial(sp, 1))
            nxt_id = nxt.item()
            if nxt_id == self.tok.eos_token_id and len(generated_ids) >= self.c.degen_min_tokens:
                break
            generated_ids.append(nxt_id)
            if cc and nxt_id in cc.content_ids:
                generated_content_counts[nxt_id] = generated_content_counts.get(nxt_id, 0) + 1
                consecutive_content += 1
                content_history.append((i, nxt_id))
                if nxt_id in anchors_for_b0:
                    generated_anchors.add(nxt_id)
                if nxt_id in cc.word_starter_ids:
                    recent_starters.append((nxt_id, i))
            else:
                consecutive_content = 0
            recent_starters = [(t, s) for (t, s) in recent_starters if (i - s) < self.c.bpe_echo_window]
            if len(content_history) > 2 * self.c.cyclic_content_window:
                content_history = content_history[-self.c.cyclic_content_window :]
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)
        return self.tok.decode(ids[0], skip_special_tokens=True)


def hungarian_max_assignment(sim: torch.Tensor) -> Tuple[torch.Tensor, float]:
    device = sim.device
    n_rows, n_cols = sim.shape
    if n_rows == 0 or n_cols == 0:
        return torch.empty(0, 2, dtype=torch.long, device=device), 0.0
    transposed = False
    original_sim = sim
    if n_rows > n_cols:
        sim = sim.T
        n_rows, n_cols = sim.shape
        transposed = True
    cost = (-sim).detach().cpu().numpy().astype("float64")
    import numpy as np

    INF = float("inf")
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, INF)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = []
    total = 0.0
    for j in range(1, n_cols + 1):
        i = p[j]
        if i > 0 and i <= n_rows:
            if transposed:
                pairs.append((j - 1, i - 1))
                total += original_sim[j - 1, i - 1].item()
            else:
                pairs.append((i - 1, j - 1))
                total += original_sim[i - 1, j - 1].item()
    pairs_tensor = torch.tensor(pairs, dtype=torch.long, device=device) if pairs else torch.empty(0, 2, dtype=torch.long, device=device)
    return pairs_tensor, total


@dataclass
class Cfg(Cfg):
    degen_early_punct_penalty: float = 8.0
    degen_early_newline_penalty: float = 8.0
    content_bias_scale: float = 6.0

    use_mean_centered_scoring: bool = True
    mc_keep_margin: float = 0.0
    mc_min_keep: int = 1
    mc_require_min_candidates: int = 2

    use_hungarian_fwd: bool = True
    hungarian_max_n: int = 24

    use_cfg_decoding: bool = True
    use_contrastive_memory_cfg: bool = True
    cfg_scale: float = 2.5
    cfg_decay_steps: int = 0

    use_content_semantic_tail: bool = True
    content_tail_slots: int = 2
    tail_head_hidden: int = 512

    def __post_init__(self):
        super().__post_init__()
        assert self.content_tail_slots >= 0
        assert self.content_tail_slots < self.L_mem


@dataclass
class RetrievalDiag(RetrievalDiag):
    n_after_mean_center: int = 0
    mean_center_applied: bool = False
    mean_center_dropped_ids: List[int] = field(default_factory=list)
    mean_center_raw_scores: Dict[int, float] = field(default_factory=dict)
    mean_center_final_scores: Dict[int, float] = field(default_factory=dict)
    hungarian_used: bool = False
    non_dominant_per_batch: List[List[int]] = field(default_factory=list)


class ContentSemanticTailHead(nn.Module):
    def __init__(self, d_F: int, d_LLM: int, n_slots: int, hidden: int = 512):
        super().__init__()
        self.n_slots = n_slots
        self.d_LLM = d_LLM
        if n_slots == 0:
            self.shared = None
            self.slot_heads = nn.ModuleList([])
            return
        self.shared = nn.Sequential(
            nn.Linear(d_F, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.slot_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, d_LLM), nn.LayerNorm(d_LLM))
            for _ in range(n_slots)
        ])
        for head in self.slot_heads:
            nn.init.normal_(head[0].weight, std=0.02)
            nn.init.zeros_(head[0].bias)

    def forward(self, fiber_summary: torch.Tensor) -> Optional[torch.Tensor]:
        if self.n_slots == 0 or self.shared is None:
            return None
        h = self.shared(fiber_summary)
        return torch.stack([head(h) for head in self.slot_heads], dim=1)


class EmbBridge(EmbBridge):
    def __init__(self, c):
        nn.Module.__init__(self)
        self.c = c
        self.proj = QFormerProj(c)
        self.ext = StateExtractor(c)
        self.pe = nn.Parameter(torch.randn(c.L_mem, c.d_LLM) * 0.02)
        self.bypass = ContentBypass(c.d_F, c.d_LLM, gate_bias=c.bypass_init_gate_bias)
        self.aligner = PrefixAligner(c.d_LLM, c.prefix_init_scale)
        self.tail_head = ContentSemanticTailHead(
            c.d_F, c.d_LLM,
            n_slots=c.content_tail_slots if c.use_content_semantic_tail else 0,
            hidden=c.tail_head_hidden,
        )
        self._last_inject_diag = {}
        self._last_fiber_summary = None
        self._last_tail_slots = None
        self._filler_centroid = None

    def _build_body_prefix(self, fibers, mem_mask, fiber_summary):
        qf_out = self.proj(fibers, mem_mask) + self.pe.unsqueeze(0)
        bp_out = None
        gate_val = None
        if fiber_summary is not None:
            qf_context = qf_out.mean(1)
            bp_out = self.bypass(fiber_summary, qf_context)
            gate_val = self.bypass._last_gate
            qf_out = qf_out + bp_out.unsqueeze(1)
        qf_out = self.aligner(qf_out)
        return qf_out, bp_out, gate_val

    def _apply_filler_projection_and_clamp(self, qf_out, filler_centroid):
        L = qf_out.shape[1]
        filler_dir_used = False
        if self.c.use_filler_direction_projection and filler_centroid is not None:
            n_proj = min(self.c.filler_projection_last_slots, L)
            fd = filler_centroid.view(1, 1, -1)
            mask_slot = torch.zeros(L, device=qf_out.device)
            mask_slot[L - n_proj :] = 1.0
            mask_slot = mask_slot.view(1, -1, 1)
            comp = (qf_out * fd).sum(-1, keepdim=True)
            qf_out = qf_out - comp * fd * mask_slot
            filler_dir_used = True
        if self.c.use_prefix_norm_clamp:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            max_allowed = target_norm * self.c.prefix_norm_clamp_ratio
            slot_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(max_allowed / slot_norms, max=1.0)
            qf_out = qf_out * scale
        return qf_out, filler_dir_used

    def inject(self, fibers, mem_mask=None, fiber_summary=None, filler_centroid=None, **_ignored):
        qf_out, bp_out, gate_val = self._build_body_prefix(fibers, mem_mask, fiber_summary)
        tail_slots_used = 0
        if self.c.use_content_semantic_tail and self.c.content_tail_slots > 0 and fiber_summary is not None:
            tail = self.tail_head(fiber_summary)
            if tail is not None:
                tail = self.aligner(tail)
                n = self.c.content_tail_slots
                qf_out = torch.cat([qf_out[:, :-n, :], tail], dim=1)
                tail_slots_used = n
                self._last_tail_slots = tail.detach()
        else:
            self._last_tail_slots = None
        qf_out, filler_dir_used = self._apply_filler_projection_and_clamp(qf_out, filler_centroid)
        self._last_fiber_summary = fiber_summary.detach() if fiber_summary is not None else None
        self._last_inject_diag = {
            "bypass_gate": gate_val.mean().item() if gate_val is not None else None,
            "qf_norm": qf_out.norm().item(),
            "bypass_norm": bp_out.norm().item() if bp_out is not None else 0.0,
            "aligner_scale": torch.sigmoid(self.aligner.scale_logit).item() * self.aligner._target_std.item(),
            "last_slot_norm_per_b": qf_out[:, -1].norm(dim=-1).mean().item(),
            "tail_slots_used": tail_slots_used,
            "filler_dir_projected": filler_dir_used,
        }
        return qf_out


class AMM(AMM):
    def _compute_forward_hungarian(self, query_ids, mem_ids, wte_normed, query_idf=None, idf_floor=0.1):
        if not query_ids or not mem_ids:
            return 0.0
        V = wte_normed.shape[0]
        q_valid = [q for q in query_ids if q < V]
        m_valid = [m for m in mem_ids if m < V]
        if not q_valid or not m_valid:
            return 0.0
        if max(len(q_valid), len(m_valid)) > self.c.hungarian_max_n:
            return self._compute_forward_maxsim(q_valid, m_valid, wte_normed, query_idf, idf_floor)
        q_vecs = wte_normed[q_valid]
        m_vecs = wte_normed[m_valid]
        sim = q_vecs @ m_vecs.T
        pairs, _ = hungarian_max_assignment(sim)
        if pairs.numel() == 0:
            return 0.0
        matched_sims = sim[pairs[:, 0], pairs[:, 1]]
        if query_idf is not None:
            q_ids_for_pairs = [q_valid[int(r.item())] for r in pairs[:, 0]]
            w = torch.tensor([max(query_idf.get(q, idf_floor), idf_floor) for q in q_ids_for_pairs], device=wte_normed.device, dtype=matched_sims.dtype)
            return ((matched_sims * w).sum() / w.sum().clamp(min=1e-8)).item()
        return matched_sims.mean().item()

    def _compute_bidi_min(self, q_ids, m_ids, wte_normed, query_idf, idf_floor):
        fwd = self._compute_forward_hungarian(q_ids, m_ids, wte_normed, query_idf, idf_floor) if self.c.use_hungarian_fwd else self._compute_forward_maxsim(q_ids, m_ids, wte_normed, query_idf, idf_floor)
        bwd = self._compute_backward_maxsim(q_ids, m_ids, wte_normed, query_idf, idf_floor)
        return fwd, bwd, min(fwd, bwd)

    def _check_consolidation_compatible(self, existing_content_ids, new_content_ids):
        if not existing_content_ids or not new_content_ids:
            return True
        if self.wte_normed is None:
            return True
        _, _, m = self._compute_bidi_min(existing_content_ids, new_content_ids, self.wte_normed, None, self.c.idf_floor)
        return m >= self.c.consol_maxsim_min

    def retrieve_multi(self, xq, fq, topk=None, bw=None, update_stats=True, query_semantic_emb=None, query_content_ids_per_batch=None, wte_normed=None, content_classifier=None):
        B = xq.shape[0]
        dev = xq.device
        topk = topk or self.c.retrieval_topk
        bw = bw or self.c.retrieval_beam
        recall_k = int(topk * self.c.retrieval_recall_factor)
        flat_thresh = self.c.flat_scan_threshold_factor * topk
        qdir = self.dir_pred(xq, fq)
        diag = RetrievalDiag()
        corpus_idf = self._compute_corpus_idf(content_classifier) if self.c.use_idf_retrieval else None
        diag.idf_applied = corpus_idf is not None
        diag.centroid_applied = self.c.use_idf_centroid
        diag.hungarian_used = self.c.use_hungarian_fwd
        idf_floor = self.c.idf_floor
        if not self.tree.store:
            empty = self.empty_state(xq, fq)
            mask = torch.ones(B, 1, **_dev(xq))
            summary = empty.mean(1) if empty.dim() == 3 else empty
            diag.fiber_summary_norm = summary.norm().item()
            diag.batch_mem_weights = [[] for _ in range(B)]
            diag.dominant_per_batch = [None for _ in range(B)]
            diag.non_dominant_per_batch = [[] for _ in range(B)]
            return empty.unsqueeze(1), mask, summary, diag
        all_results, all_masks, all_biases, all_summaries = [], [], [], []
        all_batch_mw, all_dominant, all_non_dominant = [], [], []
        wn = wte_normed if wte_normed is not None else self.wte_normed
        for b in range(B):
            n_store = len(self.tree.store)
            if n_store <= flat_thresh:
                mids = list(self.tree.store.keys())
                diag.was_flat_scan = True
            else:
                scored = self.tree.retrieve(qdir[b].detach(), bw)
                mids = [s[0] for s in scored[:recall_k]]
            mems = [self.tree.store[i] for i in mids if i in self.tree.store]
            diag.recall_count = len(mems)
            diag.n_candidates_initial = len(mems)
            if not mems:
                empty = self.empty_state(xq[b:b+1], fq[b:b+1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1, **_dev(xq)))
                all_biases.append(torch.zeros(1, **_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([])
                all_dominant.append(None)
                all_non_dominant.append([])
                continue
            q_content_ids = query_content_ids_per_batch[b] if query_content_ids_per_batch and b < len(query_content_ids_per_batch) else []
            q_strict = []
            if content_classifier is not None:
                q_strict = [t for t in q_content_ids if t in content_classifier.strict_content_starter_ids and wn is not None and t < wn.shape[0]]
            if self.c.use_strict_content_overlap_gate and q_strict and wn is not None and content_classifier is not None:
                overlap_counts = torch.zeros(len(mems), dtype=torch.long, device=dev)
                for mi, mem in enumerate(mems):
                    m_strict = [t for t in mem.content_token_ids if t in content_classifier.strict_content_starter_ids and t < wn.shape[0]]
                    cnt = self._count_strict_overlap_matches(q_strict, m_strict, wn, self.c.strict_overlap_sim_threshold)
                    overlap_counts[mi] = cnt
                    diag.per_memory_strict_overlap[mem.mid] = cnt
                pass_mask = overlap_counts >= self.c.strict_overlap_min_matches
                if int(pass_mask.sum().item()) < self.c.strict_overlap_min_keep:
                    _, top_keep = overlap_counts.topk(min(max(self.c.strict_overlap_min_keep, 1), len(mems)))
                    pass_mask = torch.zeros(len(mems), dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                diag.strict_overlap_dropped_ids = [mems[i].mid for i in (~pass_mask).nonzero(as_tuple=True)[0].tolist()]
                diag.strict_overlap_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < len(mems):
                    mems = [mems[i] for i in keep_local.tolist()]
            diag.n_after_strict_overlap_gate = len(mems)
            C_init = len(mems)
            if C_init == 0:
                empty = self.empty_state(xq[b:b+1], fq[b:b+1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1, **_dev(xq)))
                all_biases.append(torch.zeros(1, **_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([])
                all_dominant.append(None)
                all_non_dominant.append([])
                continue
            sb = torch.stack([m.base.to(dev) for m in mems])
            sf = torch.stack([m.fiber.to(dev) for m in mems])
            md = torch.stack([m.dirn.to(dev) for m in mems])
            sem_sim_t = torch.zeros(C_init, device=dev)
            if query_semantic_emb is not None:
                for mi, mem in enumerate(mems):
                    if mem.semantic_emb is not None:
                        sem_sim_t[mi] = F.cosine_similarity(query_semantic_emb[b:b+1], mem.semantic_emb.unsqueeze(0).to(dev), dim=-1).squeeze()
            forward_t = torch.zeros(C_init, device=dev)
            backward_t = torch.zeros(C_init, device=dev)
            bidi_min_t = torch.zeros(C_init, device=dev)
            if q_content_ids and wn is not None:
                for mi, mem in enumerate(mems):
                    scoring_ids = self._get_mem_scoring_ids(mem)
                    fwd, bwd, bmin = self._compute_bidi_min(q_content_ids, scoring_ids, wn, corpus_idf, idf_floor)
                    forward_t[mi] = fwd
                    backward_t[mi] = bwd
                    bidi_min_t[mi] = bmin
            if self.c.use_upstream_semantic_gate and q_content_ids and wn is not None:
                fwd_pass = forward_t >= self.c.upstream_gate_fwd_idf_floor
                sem_pass = sem_sim_t >= self.c.upstream_gate_sem_floor
                pass_mask = (fwd_pass & sem_pass) if self.c.upstream_gate_require_both else (fwd_pass | sem_pass)
                if int(pass_mask.sum().item()) < self.c.upstream_gate_min_keep:
                    top_keep = forward_t.topk(min(max(self.c.upstream_gate_min_keep, 1), C_init)).indices
                    pass_mask = torch.zeros(C_init, dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                diag.upstream_gate_dropped_ids = [mems[i].mid for i in (~pass_mask).nonzero(as_tuple=True)[0].tolist()]
                diag.upstream_semantic_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C_init:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb = sb[keep_local]
                    sf = sf[keep_local]
                    md = md[keep_local]
                    sem_sim_t = sem_sim_t[keep_local]
                    forward_t = forward_t[keep_local]
                    backward_t = backward_t[keep_local]
                    bidi_min_t = bidi_min_t[keep_local]
                    C_init = len(mems)
            diag.n_after_upstream_semantic_gate = C_init
            raw_dir_sim = torch.einsum("d,cd->c", qdir[b], md)
            diag.top_dir_sim = raw_dir_sim.max().item() if C_init > 0 else 0.0
            diag.top_sem_sim = sem_sim_t.max().item() if C_init > 0 else 0.0
            diag.top_forward_maxsim = forward_t.max().item() if C_init > 0 else 0.0
            diag.top_backward_maxsim = backward_t.max().item() if C_init > 0 else 0.0
            diag.top_bidi_min = bidi_min_t.max().item() if C_init > 0 else 0.0
            centroid_scores = torch.zeros(C_init, device=dev)
            if self.c.use_idf_centroid and q_content_ids and wn is not None:
                q_centroid = self._compute_idf_weighted_centroid(q_content_ids, wn, corpus_idf, idf_floor)
                if q_centroid is not None:
                    for mi, mem in enumerate(mems):
                        m_centroid = self._compute_idf_weighted_centroid(self._get_mem_scoring_ids(mem), wn, corpus_idf, idf_floor)
                        if m_centroid is not None:
                            centroid_scores[mi] = (q_centroid @ m_centroid).item()
                diag.top_centroid_cosine = centroid_scores.max().item() if C_init > 0 else 0.0
            combined_sim = self.c.ret_centroid_weight * centroid_scores + self.c.ret_sem_weight * sem_sim_t + self.c.ret_bidi_min_weight * bidi_min_t + self.c.ret_forward_maxsim_weight * forward_t + self.c.ret_dir_weight * raw_dir_sim
            C = C_init
            sem_thresh = max(self.c.gate_sem_floor, sem_sim_t.max().item() * self.c.gate_sem_ratio) if C > 0 else self.c.gate_sem_floor
            bidi_thresh = max(self.c.gate_bidi_floor, bidi_min_t.max().item() * self.c.gate_bidi_ratio if C > 0 else 0.0, self.c.gate_bidi_hard_min)
            hard_mask = (sem_sim_t >= sem_thresh) & (bidi_min_t >= bidi_thresh)
            gate_affinity = self.c.gate_sem_weight * sem_sim_t + self.c.gate_bidi_weight * bidi_min_t
            diag.top_gate_affinity = gate_affinity.max().item() if C > 0 else 0.0
            diag.gate_threshold = max(sem_thresh, bidi_thresh)
            diag.n_gate_pass = int(hard_mask.sum().item())
            if hard_mask.sum().item() == 0 and C > 0:
                hard_mask[torch.minimum(sem_sim_t, bidi_min_t).argmax()] = True
            diag.n_after_hard_filter = int(hard_mask.sum().item())
            for mi, mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid] = gate_affinity[mi].item()
            keep_indices = hard_mask.nonzero(as_tuple=True)[0]
            if keep_indices.numel() > 0 and keep_indices.numel() < C:
                mems = [mems[i] for i in keep_indices.tolist()]
                sb = sb[keep_indices]; sf = sf[keep_indices]
                combined_sim = combined_sim[keep_indices]
                raw_dir_sim = raw_dir_sim[keep_indices]
                forward_t = forward_t[keep_indices]
                bidi_min_t = bidi_min_t[keep_indices]
                sem_sim_t = sem_sim_t[keep_indices]
                centroid_scores = centroid_scores[keep_indices]
                C = len(mems)
            rerank_scores = self.reranker(xq[b:b+1], fq[b:b+1], sb.unsqueeze(0), sf.unsqueeze(0), combined_sim.unsqueeze(0)).squeeze(0)
            diag.reranker_delta_mean = (rerank_scores - combined_sim).abs().mean().item()
            diag.top_reranker_score = rerank_scores.max().item() if C > 0 else 0.0
            if C > 1:
                score_mask = rerank_scores >= rerank_scores.max() * self.c.score_keep_ratio
                if score_mask.sum().item() < 1:
                    score_mask[rerank_scores.argmax()] = True
                score_keep = score_mask.nonzero(as_tuple=True)[0]
                diag.n_after_score_filter = score_keep.numel()
                if score_keep.numel() < C:
                    mems = [mems[i] for i in score_keep.tolist()]
                    sb = sb[score_keep]; sf = sf[score_keep]
                    rerank_scores = rerank_scores[score_keep]
                    forward_t = forward_t[score_keep]
                    bidi_min_t = bidi_min_t[score_keep]
                    sem_sim_t = sem_sim_t[score_keep]
                    centroid_scores = centroid_scores[score_keep]
                    C = len(mems)
            else:
                diag.n_after_score_filter = C
            if C > 1 and forward_t.max().item() > 0:
                coherence_keep = (forward_t >= forward_t.max() * self.c.fwd_coherence_ratio).nonzero(as_tuple=True)[0]
                diag.n_after_coherence_filter = coherence_keep.numel()
                if coherence_keep.numel() >= 1 and coherence_keep.numel() < C:
                    mems = [mems[i] for i in coherence_keep.tolist()]
                    sb = sb[coherence_keep]; sf = sf[coherence_keep]
                    rerank_scores = rerank_scores[coherence_keep]
                    forward_t = forward_t[coherence_keep]
                    bidi_min_t = bidi_min_t[coherence_keep]
                    sem_sim_t = sem_sim_t[coherence_keep]
                    centroid_scores = centroid_scores[coherence_keep]
                    C = len(mems)
            else:
                diag.n_after_coherence_filter = C
            if C > 1 and bidi_min_t.max().item() > 0:
                gap_keep = (bidi_min_t >= (bidi_min_t.max().item() - self.c.bidi_absolute_gap)).nonzero(as_tuple=True)[0]
                diag.n_after_bidi_gap_filter = gap_keep.numel()
                if gap_keep.numel() >= 1 and gap_keep.numel() < C:
                    mems = [mems[i] for i in gap_keep.tolist()]
                    sb = sb[gap_keep]; sf = sf[gap_keep]
                    rerank_scores = rerank_scores[gap_keep]
                    forward_t = forward_t[gap_keep]
                    bidi_min_t = bidi_min_t[gap_keep]
                    sem_sim_t = sem_sim_t[gap_keep]
                    centroid_scores = centroid_scores[gap_keep]
                    C = len(mems)
            else:
                diag.n_after_bidi_gap_filter = C
            raw_composite = 0.4 * centroid_scores + 0.4 * forward_t + 0.15 * bidi_min_t + 0.05 * sem_sim_t.clamp(min=0)
            if self.c.use_mean_centered_scoring and C >= self.c.mc_require_min_candidates:
                C_f = float(C)
                sum_raw = raw_composite.sum()
                centered = (C_f / (C_f - 1.0)) * raw_composite - sum_raw / (C_f - 1.0)
                for mi, mem in enumerate(mems):
                    diag.mean_center_raw_scores[mem.mid] = raw_composite[mi].item()
                    diag.mean_center_final_scores[mem.mid] = centered[mi].item()
                keep_mask = centered > self.c.mc_keep_margin
                if int(keep_mask.sum().item()) < self.c.mc_min_keep:
                    top_keep = centered.topk(min(max(self.c.mc_min_keep, 1), C)).indices
                    keep_mask = torch.zeros(C, dtype=torch.bool, device=dev)
                    keep_mask[top_keep] = True
                if (~keep_mask).any():
                    diag.mean_center_applied = True
                    diag.mean_center_dropped_ids = [mems[i].mid for i in (~keep_mask).nonzero(as_tuple=True)[0].tolist()]
                keep_local = keep_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb = sb[keep_local]; sf = sf[keep_local]
                    rerank_scores = rerank_scores[keep_local]
                    forward_t = forward_t[keep_local]
                    bidi_min_t = bidi_min_t[keep_local]
                    sem_sim_t = sem_sim_t[keep_local]
                    centroid_scores = centroid_scores[keep_local]
                    C = len(mems)
            diag.n_after_mean_center = C
            dominant_mid = None
            non_dominant_mids = []
            if C >= 1:
                final_rank = 0.4 * rerank_scores + 0.4 * centroid_scores + 0.2 * forward_t
                dom_idx = int(final_rank.argmax().item())
                dominant_mid = mems[dom_idx].mid
                non_dominant_mids = [mems[i].mid for i in range(C) if i != dom_idx]
            if not self.training and C > topk:
                _, top_idx = rerank_scores.topk(topk)
                mems = [mems[i] for i in top_idx.cpu().tolist()]
                sb = sb[top_idx]; sf = sf[top_idx]
                rerank_scores = rerank_scores[top_idx]
                forward_t = forward_t[top_idx]
                bidi_min_t = bidi_min_t[top_idx]
                sem_sim_t = sem_sim_t[top_idx]
                centroid_scores = centroid_scores[top_idx]
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
                ret_s = self.retention(sb, sf, torch.tensor([m.surprise for m in mems], **_dev(xq)), torch.tensor([self.time - m.last for m in mems], **_dev(xq)), torch.tensor([m.cnt for m in mems], **_dev(xq)))
                transported = transported * ret_s.unsqueeze(-1)
            if update_stats:
                for m in mems:
                    m.last = self.time
                    m.cnt += 1
            final_scores = 0.4 * rerank_scores + 0.4 * centroid_scores + 0.2 * forward_t
            w = F.softmax(final_scores / self.c.retrieval_weight_temperature, dim=0)
            fs = (transported * w.unsqueeze(-1)).sum(0)
            all_batch_mw.append([(m.mid, w[mi].item()) for mi, m in enumerate(mems)])
            all_dominant.append(dominant_mid)
            all_non_dominant.append(non_dominant_mids)
            all_results.append(transported)
            all_masks.append(torch.ones(C, **_dev(xq)))
            all_biases.append(final_scores / self.c.tau)
            all_summaries.append(fs)
        maxC = max(r.shape[0] for r in all_results)
        padded, pm, pd = [], [], []
        for bi in range(B):
            r, mk, db = all_results[bi], all_masks[bi], all_biases[bi]
            gap = maxC - r.shape[0]
            if gap > 0:
                pr = self.empty_state(xq[bi:bi+1], fq[bi:bi+1]).expand(gap, -1)
                r = torch.cat([r, pr if self.training else pr.detach()], 0)
                mk = torch.cat([mk, torch.zeros(gap, **_dev(xq))])
                db = torch.cat([db, torch.full((gap,), -1e9, **_dev(xq))])
            padded.append(r)
            pm.append(mk)
            pd.append(db)
        mf = torch.stack(padded)
        mem_mask = torch.stack(pm)
        dir_bias = torch.stack(pd)
        fiber_summary = torch.stack(all_summaries)
        diag.fiber_summary_norm = fiber_summary.norm().item()
        diag.batch_mem_weights = all_batch_mw
        diag.dominant_per_batch = all_dominant
        diag.non_dominant_per_batch = all_non_dominant
        if diag.dominant_per_batch and diag.dominant_per_batch[0] is not None:
            diag.dominant_memory_id = diag.dominant_per_batch[0]
        refined = self.attn(fq, mf, mem_mask=mem_mask, dir_bias=dir_bias)
        return refined, mem_mask, fiber_summary, diag


class MemLLM(MemLLM):
    def __init__(self, c):
        super().__init__(c)
        self.amm = AMM(c)
        self.bridge = EmbBridge(c)
        self._filler_centroid = None

    def _build_contrastive_uncond_prefix(self, diag, prefix_cond):
        dev = prefix_cond.device
        B = prefix_cond.shape[0]
        uncond_prefix = torch.zeros_like(prefix_cond)
        for b in range(B):
            mids = diag.non_dominant_per_batch[b] if b < len(diag.non_dominant_per_batch) else []
            mids = [m for m in mids if m in self.amm.tree.store]
            if mids:
                fvecs = torch.stack([self.amm.tree.store[m].fiber.to(dev) for m in mids])
                non_dom = fvecs.mean(0, keepdim=True)
                pref_b = self.bridge.inject(
                    non_dom.unsqueeze(1),
                    torch.ones(1, 1, device=dev),
                    fiber_summary=non_dom,
                    filler_centroid=self._filler_centroid,
                )
                uncond_prefix[b:b+1] = pref_b
            else:
                uncond_prefix[b:b+1] = self.bridge.build_neutral_prefix(1, dev)
        return uncond_prefix

    def generate(self, prompt, mt=50, greedy=False):
        tk = self.tok(prompt, return_tensors="pt")
        dev = next(self.parameters()).device
        ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            prefix_cond, fiber_summary, diag, content_bias = self._get_prefix(
                o["hs"], mask, update_stats=True, return_extra=True, ids=ids
            )
            vocab_bias = self._compute_vocab_bias(fiber_summary)
            if self.c.use_cfg_decoding:
                prefix_uncond = self._build_contrastive_uncond_prefix(diag, prefix_cond) if self.c.use_contrastive_memory_cfg else self.bridge.build_neutral_prefix(prefix_cond.shape[0], dev)
            else:
                prefix_uncond = None
        generated_ids = []
        generated_content_counts: Dict[int, int] = {}
        content_history: List[Tuple[int, int]] = []
        recent_starters: List[Tuple[int, int]] = []
        cc = self.content_classifier
        newline_ids_set = cc.newline_ids if cc is not None else set()
        HARD_MASK = -1e9
        eos_token_id = self.tok.eos_token_id
        for i in range(mt):
            if i > 0 and i % self.c.retrieval_interval == 0:
                with torch.no_grad():
                    o = self.fwd(ids, mask, prefix_cond)
                    pl = o["pl"]
                    prefix_cond, fiber_summary, diag, content_bias = self._get_prefix(
                        o["hs"], o["mask"], pl, update_stats=True, return_extra=True, ids=ids
                    )
                    vocab_bias = self._compute_vocab_bias(fiber_summary)
                    if self.c.use_cfg_decoding:
                        prefix_uncond = self._build_contrastive_uncond_prefix(diag, prefix_cond) if self.c.use_contrastive_memory_cfg else self.bridge.build_neutral_prefix(prefix_cond.shape[0], dev)
            with torch.no_grad():
                o_cond = self.fwd(ids, mask, prefix_cond)
                lg_cond = o_cond["logits"][:, -1:].squeeze(1)
                if self.c.use_cfg_decoding and prefix_uncond is not None:
                    o_uncond = self.fwd(ids, mask, prefix_uncond)
                    lg_uncond = o_uncond["logits"][:, -1:].squeeze(1)
                    alpha = self.c.cfg_scale
                    if self.c.cfg_decay_steps > 0:
                        alpha *= max(0.0, 1.0 - i / self.c.cfg_decay_steps)
                    lg = lg_cond + alpha * (lg_cond - lg_uncond)
                else:
                    lg = lg_cond.clone()
                step_scale_content = max(self.c.content_bias_floor, 1.0 - i * self.c.content_bias_decay)
                if content_bias is not None and content_bias.abs().max().item() > 0.01:
                    V = min(lg.shape[-1], content_bias.shape[-1])
                    lg[:, :V] = lg[:, :V] + content_bias[:, :V] * self.c.content_bias_scale * step_scale_content
                step_scale_learned = max(self.c.semantic_boost_floor, 1.0 - i * self.c.semantic_boost_decay)
                if vocab_bias is not None:
                    V2 = min(lg.shape[-1], vocab_bias.shape[-1])
                    lg[:, :V2] = lg[:, :V2] + vocab_bias[:, :V2] * self.c.semantic_boost_scale * step_scale_learned
                if cc:
                    for tid, count in generated_content_counts.items():
                        if tid in cc.content_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.content_repeat_penalty * (count ** self.c.content_repeat_exponent)
                if self.c.use_cyclic_content_hard_mask and cc is not None:
                    window_counts: Dict[int, int] = {}
                    cutoff_step = i - self.c.cyclic_content_window
                    for step_idx, tid in content_history:
                        if step_idx >= cutoff_step:
                            window_counts[tid] = window_counts.get(tid, 0) + 1
                    for tid, cnt in window_counts.items():
                        if cnt >= self.c.cyclic_content_max_count and 0 <= tid < lg.shape[-1]:
                            lg[0, tid] = HARD_MASK
                if self.c.use_ngram_repeat_block and len(generated_ids) >= 4:
                    max_n = min(self.c.ngram_repeat_max_n, len(generated_ids) // 2)
                    for n in range(2, max_n + 1):
                        if len(generated_ids) >= 2 * n and generated_ids[-n:] == generated_ids[-2 * n : -n]:
                            expected_next = generated_ids[-n]
                            if 0 <= expected_next < lg.shape[-1]:
                                lg[0, expected_next] -= self.c.ngram_repeat_penalty
                if cc and self._wte_neighbor_cache is not None and recent_starters:
                    for prev_tid, _ in recent_starters:
                        for nid in self._wte_neighbor_cache.get(prev_tid, []):
                            if nid in cc.word_starter_ids:
                                continue
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.bpe_echo_penalty
                if cc and generated_ids and generated_ids[-1] in cc.content_starter_ids:
                    for tid in cc.content_ids:
                        if tid not in cc.word_starter_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.post_starter_nonstarter_penalty
                if self.c.use_newline_hard_gate and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if i < self.c.newline_hard_gate_min_step or content_count_so_far < self.c.newline_hard_gate_min_content:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] = HARD_MASK
                if self.c.use_eos_hard_mask and eos_token_id is not None and i < self.c.eos_hard_mask_steps and eos_token_id < lg.shape[-1]:
                    lg[0, eos_token_id] = HARD_MASK
                if self.c.use_content_gated_newline and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if content_count_so_far < self.c.min_content_tokens_before_newline:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.late_newline_penalty
                if self._degen_guard is not None:
                    lg = self._degen_guard.process(lg, generated_ids, i)
                if greedy:
                    nxt = lg.argmax(-1, keepdim=True)
                else:
                    lg_t = lg / self.c.gen_temp
                    p = F.softmax(lg_t, -1)
                    sp, si = torch.sort(p, descending=True)
                    cs = torch.cumsum(sp, -1)
                    rm = cs - sp > self.c.gen_top_p
                    sp[rm] = 0
                    total = sp.sum(-1, keepdim=True)
                    if (total < 1e-10).any():
                        sp[:, 0] = 1.0
                        total = sp.sum(-1, keepdim=True)
                    sp = sp / total
                    nxt = si.gather(-1, torch.multinomial(sp, 1))
            nxt_id = nxt.item()
            if nxt_id == self.tok.eos_token_id and len(generated_ids) >= self.c.degen_min_tokens:
                break
            generated_ids.append(nxt_id)
            if cc and nxt_id in cc.content_ids:
                generated_content_counts[nxt_id] = generated_content_counts.get(nxt_id, 0) + 1
                content_history.append((i, nxt_id))
                if nxt_id in cc.word_starter_ids:
                    recent_starters.append((nxt_id, i))
            recent_starters = [(t, s) for (t, s) in recent_starters if (i - s) < self.c.bpe_echo_window]
            if len(content_history) > 2 * self.c.cyclic_content_window:
                content_history = content_history[-self.c.cyclic_content_window :]
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)
        return self.tok.decode(ids[0], skip_special_tokens=True)


class Trainer(Trainer):
    def __init__(self, m, c):
        super().__init__(m, c)
        if c.use_content_semantic_tail and c.content_tail_slots > 0:
            self.grad_monitor.register("tail_head", m.bridge.tail_head)

    def tail_semantic_anchor_loss(self, fiber, ids, mask):
        if not (self.c.use_content_semantic_tail and self.c.content_tail_slots > 0):
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        tail = self.m.bridge.tail_head(fiber)
        if tail is None:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        wte = self.m.llm.transformer.wte.weight.detach()
        cc = self.m.content_classifier
        if cc is None:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        tn = F.normalize(tail, dim=-1)
        wn = F.normalize(wte, dim=-1)
        losses = []
        V = wte.shape[0]
        for b in range(tail.shape[0]):
            valid = ids[b][mask[b].bool()].tolist()
            content_tids = [t for t in set(cc.get_content_ids_from_tokens(valid)) if t < V]
            if not content_tids:
                continue
            target = torch.zeros(V, device=tail.device)
            target[content_tids] = 1.0 / len(content_tids)
            slot_logits = tn[b] @ wn.T / 0.3
            log_probs = F.log_softmax(slot_logits, dim=-1)
            kl = F.kl_div(log_probs, target.unsqueeze(0).expand_as(log_probs), reduction="none").sum(-1).mean()
            losses.append(kl)
        if not losses:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        return torch.stack(losses).mean()

    def step(self, texts):
        self.m.train()
        self.opt.zero_grad()
        dev = next(self.m.parameters()).device
        W = self.c.loss_weights
        ids_enc, mask_enc, base, fiber, surp, pooled_mean = self._encode_with_grad(texts)
        l_et = self.encoder_throughput_loss(ids_enc, mask_enc, fiber)
        w_sa = self.warmup.weight("semantic_alignment")
        l_sa = self.semantic_alignment_loss(fiber, ids_enc, mask_enc) * w_sa
        w_tsa = self.warmup.weight("tail_semantic_anchor")
        l_tsa = self.tail_semantic_anchor_loss(fiber, ids_enc, mask_enc) * w_tsa
        all_lr, all_pf, all_fs = [], [], []
        for t in texts:
            lr, pf, fs = self._recon_forward(t)
            all_lr.append(lr)
            all_pf.append(pf)
            all_fs.append(fs if fs is not None else torch.zeros(1, self.c.d_F, device=dev))
        l_r = sum(all_lr) / len(texts)
        pf_batch = torch.cat(all_pf, 0)
        fs_batch = torch.cat(all_fs, 0)
        w_sp = self.warmup.weight("semantic_probe")
        l_sp = self._semantic_probe_loss(pf_batch, fs_batch) * w_sp
        w_va = self.warmup.weight("vocab_anchor")
        l_va = self.vocab_anchor_loss(pf_batch) * w_va
        l_c = self.contrast(texts) if len(texts) >= 2 else torch.tensor(0.0, device=dev)
        with torch.no_grad():
            tk2 = self.m.tok(texts, return_tensors="pt", padding=True, truncation=True)
            ids2, mask2 = tk2["input_ids"].to(dev), tk2["attention_mask"].to(dev)
            o2 = self.m.fwd(ids2, mask2)
        _, xq2, fq2 = self.m.extract_state(o2["hs"], mask2)
        l_h = self.holonomy_proxy(xq2, fq2)
        l_w = self.write_policy_loss(texts)
        w_dd = self.warmup.weight("dir_diversity")
        l_dd = (self.direction_diversity_loss(texts) if len(texts) >= 2 else torch.tensor(0.0, device=dev)) * w_dd
        w_rr = self.warmup.weight("reranker_ranking")
        l_rr = self.reranker_ranking_loss(texts) * w_rr
        loss = (
            W["recon"] * l_r
            + W["semantic_alignment"] * l_sa
            + W["encoder_throughput"] * l_et
            + W["contrast"] * l_c
            + W["holonomy"] * l_h
            + W["write_policy"] * l_w
            + W["semantic_probe"] * l_sp
            + W["dir_diversity"] * l_dd
            + W["reranker_ranking"] * l_rr
            + W["vocab_anchor"] * l_va
            + W.get("tail_semantic_anchor", 0.5) * l_tsa
        )
        loss.backward()
        nn.utils.clip_grad_norm_([p for n, p in self.m.named_parameters() if p.requires_grad and "llm" not in n], 1.0)
        self.opt.step()
        self.warmup.advance()
        self._step_count += 1
        grad_norms = self.grad_monitor.snapshot()
        self.layer_weight_history.append(self.m.layer_pool.weight_dist().cpu().numpy().copy())
        if self._step_count % self.c.refresh_memories_every == 0:
            self.m.eval()
            with torch.no_grad():
                self.m._refresh_all_memories()
            self.m.train()
        self.m.eval()
        return {
            "total": loss.item(),
            "recon": l_r.item(),
            "contrast": l_c.item(),
            "holonomy": l_h.item(),
            "write_policy": l_w.item(),
            "semantic_probe": l_sp.item(),
            "dir_diversity": l_dd.item(),
            "reranker_ranking": l_rr.item(),
            "encoder_throughput": l_et.item(),
            "vocab_anchor": l_va.item(),
            "semantic_alignment": l_sa.item(),
            "tail_semantic_anchor": l_tsa.item(),
            "grad_norms": grad_norms,
            "loss_weights": W,
        }


@dataclass
class Cfg(Cfg):
    early_content_steps: int = 3
    content_bias_scale: float = 8.0
    content_bias_decay: float = 0.04
    content_bias_floor: float = 0.3
    use_cfg_decoding: bool = True
    cfg_scale: float = 1.5
    cfg_decay_steps: int = 0
    use_gap_cut: bool = True
    gap_outlier_ratio: float = 2.0
    gap_log_shift_eps: float = 1e-6
    gap_min_keep: int = 1
    gap_min_candidates: int = 3
    degen_early_punct_penalty: float = 10.0
    degen_early_newline_penalty: float = 10.0
    late_newline_penalty: float = 30.0
    semantic_boost_scale: float = 0.5
    semantic_boost_decay: float = 0.06
    semantic_boost_floor: float = 0.2
    use_strict_anchor_boost: bool = False
    use_strict_avg_maxsim_relative_floor: bool = False
    use_fwd_idf_relative_floor: bool = False
    use_final_domain_purge: bool = False
    use_early_punct_hard_mask: bool = False
    use_early_function_hard_mask: bool = False
    use_step0_strict_hard_restrict: bool = False
    extended_strict_restrict_steps: int = 0
    use_early_non_strict_hard_penalty: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.cfg_scale >= 0.0
        assert self.gap_outlier_ratio >= 1.0


@dataclass
class RetrievalDiag(RetrievalDiag):
    n_after_gap_cut: int = 0
    gap_cut_applied: bool = False
    gap_cut_max_gap: float = 0.0
    gap_cut_second_gap: float = 0.0
    gap_cut_dropped_ids: List[int] = field(default_factory=list)


class EmbBridge(EmbBridge):
    def __init__(self, c):
        nn.Module.__init__(self)
        self.c = c
        self.proj = QFormerProj(c)
        self.ext = StateExtractor(c)
        self.pe = nn.Parameter(torch.randn(c.L_mem, c.d_LLM) * 0.02)
        self.bypass = ContentBypass(c.d_F, c.d_LLM, gate_bias=c.bypass_init_gate_bias)
        self.aligner = PrefixAligner(c.d_LLM, c.prefix_init_scale)
        self.content_inject_scale = c.content_inject_scale
        self.inject_mode = "both"
        self._last_inject_diag = {}
        self._last_fiber_summary = None
        self._filler_centroid = None

    def inject(self, fibers, mem_mask=None, fiber_summary=None, filler_centroid=None, **_ignored):
        qf_out = self.proj(fibers, mem_mask) + self.pe.unsqueeze(0)
        bp_out = None
        gate_val = None
        if fiber_summary is not None:
            qf_context = qf_out.mean(1)
            bp_out = self.bypass(fiber_summary, qf_context)
            gate_val = self.bypass._last_gate
            qf_out = qf_out + bp_out.unsqueeze(1)
        qf_out = self.aligner(qf_out)
        L = qf_out.shape[1]
        filler_dir_used = self.c.use_filler_direction_projection and filler_centroid is not None
        filler_proj_comp_max = 0.0
        if filler_dir_used:
            n_proj = min(self.c.filler_projection_last_slots, L)
            fd = filler_centroid.view(1, 1, -1)
            mask_slot = torch.zeros(L, device=qf_out.device)
            mask_slot[L - n_proj :] = 1.0
            mask_slot = mask_slot.view(1, -1, 1)
            comp = (qf_out * fd).sum(-1, keepdim=True)
            filler_proj_comp_max = comp.abs().max().item()
            qf_out = qf_out - comp * fd * mask_slot
        pre_clamp_norm_max = qf_out.norm(dim=-1).max().item()
        clamp_applied_count = 0
        if self.c.use_prefix_norm_clamp:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            max_allowed = target_norm * self.c.prefix_norm_clamp_ratio
            slot_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            exceed_mask = slot_norms.squeeze(-1) > max_allowed
            clamp_applied_count = int(exceed_mask.sum().item())
            scale = torch.clamp(max_allowed / slot_norms, max=1.0)
            qf_out = qf_out * scale
        post_clamp_norm_max = qf_out.norm(dim=-1).max().item()
        self._last_fiber_summary = fiber_summary.detach() if fiber_summary is not None else None
        self._last_inject_diag = {
            "bypass_gate": gate_val.mean().item() if gate_val is not None else None,
            "qf_norm": qf_out.norm().item(),
            "bypass_norm": bp_out.norm().item() if bp_out is not None else 0.0,
            "aligner_scale": torch.sigmoid(self.aligner.scale_logit).item() * self.aligner._target_std.item(),
            "last_slot_norm_per_b": qf_out[:, -1].norm(dim=-1).mean().item(),
            "pre_clamp_max_slot_norm": pre_clamp_norm_max,
            "post_clamp_max_slot_norm": post_clamp_norm_max,
            "clamp_applied_slots": clamp_applied_count,
            "filler_dir_projected": filler_dir_used,
            "filler_proj_comp_max": filler_proj_comp_max,
        }
        return qf_out

    def build_neutral_prefix(self, B, device):
        qf_out = self.pe.unsqueeze(0).expand(B, -1, -1).contiguous()
        qf_out = self.aligner(qf_out)
        if self.c.use_prefix_norm_clamp:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            max_allowed = target_norm * self.c.prefix_norm_clamp_ratio
            slot_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(max_allowed / slot_norms, max=1.0)
            qf_out = qf_out * scale
        return qf_out


class AMM(AMM):
    @staticmethod
    def _gap_cut(scores: torch.Tensor, min_keep: int = 1, outlier_ratio: float = 2.0,
                 log_shift_eps: float = 1e-6, min_candidates: int = 3):
        n = scores.numel()
        dev = scores.device
        all_idx = torch.arange(n, device=dev, dtype=torch.long)
        if n < min_candidates or n <= min_keep:
            empty = torch.empty(0, device=dev, dtype=torch.long)
            return all_idx, empty, 0.0, 0.0, False
        sorted_scores, sorted_idx = scores.sort(descending=True)
        min_val = sorted_scores.min().item()
        shift = max(0.0, -min_val) + log_shift_eps
        log_scores = torch.log(sorted_scores + shift)
        gaps = log_scores[:-1] - log_scores[1:]
        if gaps.numel() < 2:
            empty = torch.empty(0, device=dev, dtype=torch.long)
            return all_idx, empty, 0.0, 0.0, False
        gaps_sorted, _ = gaps.sort(descending=True)
        top_gap = gaps_sorted[0].item()
        second_gap = gaps_sorted[1].item()
        if top_gap < outlier_ratio * max(second_gap, log_shift_eps):
            empty = torch.empty(0, device=dev, dtype=torch.long)
            return all_idx, empty, top_gap, second_gap, False
        cut_positions = (gaps == gaps_sorted[0]).nonzero(as_tuple=True)[0]
        cut_at = int(cut_positions[0].item())
        keep_n = max(cut_at + 1, min_keep)
        if keep_n >= n:
            empty = torch.empty(0, device=dev, dtype=torch.long)
            return all_idx, empty, top_gap, second_gap, False
        kept_sorted = sorted_idx[:keep_n]
        dropped_sorted = sorted_idx[keep_n:]
        return kept_sorted.sort().values, dropped_sorted.sort().values, top_gap, second_gap, True

    def retrieve_multi(self, xq, fq, topk=None, bw=None, update_stats=True,
                       query_semantic_emb=None, query_content_ids_per_batch=None,
                       wte_normed=None, content_classifier=None):
        B = xq.shape[0]
        dev = xq.device
        topk = topk or self.c.retrieval_topk
        bw = bw or self.c.retrieval_beam
        recall_k = int(topk * self.c.retrieval_recall_factor)
        flat_thresh = self.c.flat_scan_threshold_factor * topk
        qdir = self.dir_pred(xq, fq)
        diag = RetrievalDiag()
        corpus_idf = self._compute_corpus_idf(content_classifier) if self.c.use_idf_retrieval else None
        diag.idf_applied = corpus_idf is not None
        diag.centroid_applied = self.c.use_idf_centroid
        idf_floor = self.c.idf_floor

        if not self.tree.store:
            empty = self.empty_state(xq, fq)
            mask = torch.ones(B, 1, **_dev(xq))
            summary = empty.mean(1) if empty.dim() == 3 else empty
            diag.fiber_summary_norm = summary.norm().item()
            diag.batch_mem_weights = [[] for _ in range(B)]
            diag.dominant_per_batch = [None for _ in range(B)]
            return empty.unsqueeze(1), mask, summary, diag

        all_results, all_masks, all_biases, all_summaries = [], [], [], []
        all_batch_mw, all_dominant = [], []
        wn = wte_normed if wte_normed is not None else self.wte_normed

        for b in range(B):
            n_store = len(self.tree.store)
            if n_store <= flat_thresh:
                mids = list(self.tree.store.keys())
                diag.was_flat_scan = True
            else:
                scored = self.tree.retrieve(qdir[b].detach(), bw)
                mids = [s[0] for s in scored[:recall_k]]
            mems = [self.tree.store[i] for i in mids if i in self.tree.store]
            diag.recall_count = len(mems)
            diag.n_candidates_initial = len(mems)
            if not mems:
                empty = self.empty_state(xq[b : b + 1], fq[b : b + 1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1, **_dev(xq)))
                all_biases.append(torch.zeros(1, **_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([])
                all_dominant.append(None)
                continue

            q_content_ids = query_content_ids_per_batch[b] if query_content_ids_per_batch and b < len(query_content_ids_per_batch) else []
            q_strict = []
            if content_classifier is not None:
                q_strict = [t for t in q_content_ids if t in content_classifier.strict_content_starter_ids and wn is not None and t < wn.shape[0]]

            if self.c.use_strict_content_overlap_gate and q_strict and wn is not None and content_classifier is not None:
                overlap_counts = torch.zeros(len(mems), dtype=torch.long, device=dev)
                for mi, mem in enumerate(mems):
                    m_strict = [t for t in mem.content_token_ids if t in content_classifier.strict_content_starter_ids and t < wn.shape[0]]
                    cnt = self._count_strict_overlap_matches(q_strict, m_strict, wn, self.c.strict_overlap_sim_threshold)
                    overlap_counts[mi] = cnt
                    diag.per_memory_strict_overlap[mem.mid] = cnt
                pass_mask = overlap_counts >= self.c.strict_overlap_min_matches
                if int(pass_mask.sum().item()) < self.c.strict_overlap_min_keep:
                    keep_n = max(self.c.strict_overlap_min_keep, 1)
                    _, top_keep = overlap_counts.topk(min(keep_n, len(mems)))
                    pass_mask = torch.zeros(len(mems), dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                diag.strict_overlap_dropped_ids = [mems[i].mid for i in dropped_local]
                diag.strict_overlap_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < len(mems):
                    mems = [mems[i] for i in keep_local.tolist()]
            diag.n_after_strict_overlap_gate = len(mems)

            C_init = len(mems)
            if C_init == 0:
                empty = self.empty_state(xq[b : b + 1], fq[b : b + 1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1, **_dev(xq)))
                all_biases.append(torch.zeros(1, **_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([])
                all_dominant.append(None)
                continue

            sb = torch.stack([m.base.to(dev) for m in mems])
            sf = torch.stack([m.fiber.to(dev) for m in mems])
            md_all = torch.stack([m.dirn.to(dev) for m in mems])
            sem_sim_t = torch.zeros(C_init, device=dev)
            if query_semantic_emb is not None:
                for mi, mem in enumerate(mems):
                    if mem.semantic_emb is not None:
                        sem_sim_t[mi] = F.cosine_similarity(query_semantic_emb[b : b + 1], mem.semantic_emb.unsqueeze(0).to(dev), dim=-1).squeeze()

            forward_t = torch.zeros(C_init, device=dev)
            backward_all = torch.zeros(C_init, device=dev)
            bidi_min_t = torch.zeros(C_init, device=dev)
            if q_content_ids and wn is not None:
                for mi, mem in enumerate(mems):
                    scoring_ids = self._get_mem_scoring_ids(mem)
                    fwd = self._compute_forward_maxsim(q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor)
                    bwd = self._compute_backward_maxsim(q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor)
                    forward_t[mi] = fwd
                    backward_all[mi] = bwd
                    bidi_min_t[mi] = min(fwd, bwd)

            if self.c.use_upstream_semantic_gate and q_content_ids and wn is not None:
                fwd_pass = forward_t >= self.c.upstream_gate_fwd_idf_floor
                sem_pass = sem_sim_t >= self.c.upstream_gate_sem_floor
                pass_mask = (fwd_pass & sem_pass) if self.c.upstream_gate_require_both else (fwd_pass | sem_pass)
                if int(pass_mask.sum().item()) < self.c.upstream_gate_min_keep:
                    keep_n = max(self.c.upstream_gate_min_keep, 1)
                    top_keep = forward_t.topk(min(keep_n, C_init)).indices
                    pass_mask = torch.zeros(C_init, dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                dropped_local = (~pass_mask).nonzero(as_tuple=True)[0].tolist()
                if dropped_local:
                    diag.upstream_gate_dropped_ids = [mems[i].mid for i in dropped_local]
                diag.upstream_semantic_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C_init:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb = sb[keep_local]
                    sf = sf[keep_local]
                    md_all = md_all[keep_local]
                    sem_sim_t = sem_sim_t[keep_local]
                    forward_t = forward_t[keep_local]
                    backward_all = backward_all[keep_local]
                    bidi_min_t = bidi_min_t[keep_local]
                    C_init = len(mems)
            diag.n_after_upstream_semantic_gate = C_init

            raw_dir_sim = torch.einsum("d,cd->c", qdir[b], md_all)
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
                        m_centroid = self._compute_idf_weighted_centroid(m_scoring_ids, wn, corpus_idf, idf_floor)
                        centroid_scores[mi] = self._compute_centroid_cosine(q_centroid, m_centroid)
                diag.top_centroid_cosine = centroid_scores.max().item() if C_init > 0 else 0.0

            combined_sim = (
                self.c.ret_centroid_weight * centroid_scores
                + self.c.ret_sem_weight * sem_sim_t
                + self.c.ret_bidi_min_weight * bidi_min_t
                + self.c.ret_forward_maxsim_weight * forward_t
                + self.c.ret_dir_weight * raw_dir_sim
            )
            C = C_init
            top_sem = sem_sim_t.max().item() if C > 0 else 0.0
            top_bidi = bidi_min_t.max().item() if C > 0 else 0.0
            sem_thresh = max(self.c.gate_sem_floor, top_sem * self.c.gate_sem_ratio)
            bidi_thresh = max(self.c.gate_bidi_floor, top_bidi * self.c.gate_bidi_ratio, self.c.gate_bidi_hard_min)
            hard_mask = (sem_sim_t >= sem_thresh) & (bidi_min_t >= bidi_thresh)
            gate_affinity = self.c.gate_sem_weight * sem_sim_t + self.c.gate_bidi_weight * bidi_min_t
            diag.top_gate_affinity = gate_affinity.max().item() if C > 0 else 0.0
            diag.gate_threshold = max(sem_thresh, bidi_thresh)
            diag.n_gate_pass = int(hard_mask.sum().item())
            if hard_mask.sum().item() == 0 and C > 0:
                hard_mask[torch.minimum(sem_sim_t, bidi_min_t).argmax()] = True
            diag.n_after_hard_filter = int(hard_mask.sum().item())
            for mi, mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid] = gate_affinity[mi].item()
            keep_indices = hard_mask.nonzero(as_tuple=True)[0]
            if keep_indices.numel() > 0 and keep_indices.numel() < C:
                mems = [mems[i] for i in keep_indices.tolist()]
                sb = sb[keep_indices]
                sf = sf[keep_indices]
                combined_sim = combined_sim[keep_indices]
                raw_dir_sim = raw_dir_sim[keep_indices]
                forward_t = forward_t[keep_indices]
                bidi_min_t = bidi_min_t[keep_indices]
                sem_sim_t = sem_sim_t[keep_indices]
                centroid_scores = centroid_scores[keep_indices]
                C = len(mems)

            rerank_scores = self.reranker(xq[b : b + 1], fq[b : b + 1], sb.unsqueeze(0), sf.unsqueeze(0), combined_sim.unsqueeze(0)).squeeze(0)
            diag.reranker_delta_mean = (rerank_scores - combined_sim).abs().mean().item()
            diag.top_reranker_score = rerank_scores.max().item() if C > 0 else 0.0

            if C > 1:
                score_mask = rerank_scores >= rerank_scores.max() * self.c.score_keep_ratio
                if score_mask.sum().item() < 1:
                    score_mask[rerank_scores.argmax()] = True
                score_keep = score_mask.nonzero(as_tuple=True)[0]
                diag.n_after_score_filter = score_keep.numel()
                if score_keep.numel() < C:
                    mems = [mems[i] for i in score_keep.tolist()]
                    sb = sb[score_keep]
                    sf = sf[score_keep]
                    rerank_scores = rerank_scores[score_keep]
                    forward_t = forward_t[score_keep]
                    bidi_min_t = bidi_min_t[score_keep]
                    sem_sim_t = sem_sim_t[score_keep]
                    centroid_scores = centroid_scores[score_keep]
                    C = len(mems)
            else:
                diag.n_after_score_filter = C

            if C > 1 and forward_t.max().item() > 0:
                coherence_keep = (forward_t >= forward_t.max() * self.c.fwd_coherence_ratio).nonzero(as_tuple=True)[0]
                diag.n_after_coherence_filter = coherence_keep.numel()
                if coherence_keep.numel() >= 1 and coherence_keep.numel() < C:
                    mems = [mems[i] for i in coherence_keep.tolist()]
                    sb = sb[coherence_keep]
                    sf = sf[coherence_keep]
                    rerank_scores = rerank_scores[coherence_keep]
                    forward_t = forward_t[coherence_keep]
                    bidi_min_t = bidi_min_t[coherence_keep]
                    sem_sim_t = sem_sim_t[coherence_keep]
                    centroid_scores = centroid_scores[coherence_keep]
                    C = len(mems)
            else:
                diag.n_after_coherence_filter = C

            if C > 1 and bidi_min_t.max().item() > 0:
                gap_keep = (bidi_min_t >= (bidi_min_t.max().item() - self.c.bidi_absolute_gap)).nonzero(as_tuple=True)[0]
                diag.n_after_bidi_gap_filter = gap_keep.numel()
                if gap_keep.numel() >= 1 and gap_keep.numel() < C:
                    mems = [mems[i] for i in gap_keep.tolist()]
                    sb = sb[gap_keep]
                    sf = sf[gap_keep]
                    rerank_scores = rerank_scores[gap_keep]
                    forward_t = forward_t[gap_keep]
                    bidi_min_t = bidi_min_t[gap_keep]
                    sem_sim_t = sem_sim_t[gap_keep]
                    centroid_scores = centroid_scores[gap_keep]
                    C = len(mems)
            else:
                diag.n_after_bidi_gap_filter = C

            if self.c.use_gap_cut and C >= self.c.gap_min_candidates:
                composite = 0.4 * centroid_scores + 0.4 * forward_t + 0.15 * bidi_min_t + 0.05 * sem_sim_t.clamp(min=0)
                keep_idx, drop_idx, max_gap, second_gap, applied = self._gap_cut(
                    composite,
                    min_keep=self.c.gap_min_keep,
                    outlier_ratio=self.c.gap_outlier_ratio,
                    log_shift_eps=self.c.gap_log_shift_eps,
                    min_candidates=self.c.gap_min_candidates,
                )
                diag.gap_cut_max_gap = max_gap
                diag.gap_cut_second_gap = second_gap
                if applied:
                    diag.gap_cut_applied = True
                    diag.gap_cut_dropped_ids = [mems[int(i)].mid for i in drop_idx.tolist()]
                    mems = [mems[int(i)] for i in keep_idx.tolist()]
                    sb = sb[keep_idx]
                    sf = sf[keep_idx]
                    rerank_scores = rerank_scores[keep_idx]
                    forward_t = forward_t[keep_idx]
                    bidi_min_t = bidi_min_t[keep_idx]
                    sem_sim_t = sem_sim_t[keep_idx]
                    centroid_scores = centroid_scores[keep_idx]
                    C = len(mems)
            diag.n_after_gap_cut = C

            dominant_mid = None
            if C >= 1:
                composite = 0.4 * centroid_scores + 0.4 * forward_t + 0.15 * bidi_min_t + 0.05 * sem_sim_t.clamp(min=0)
                dominant_mid = mems[int(composite.argmax().item())].mid
            if not self.training and C > topk:
                _, top_idx = rerank_scores.topk(topk)
                mems = [mems[i] for i in top_idx.cpu().tolist()]
                sb = sb[top_idx]
                sf = sf[top_idx]
                rerank_scores = rerank_scores[top_idx]
                forward_t = forward_t[top_idx]
                bidi_min_t = bidi_min_t[top_idx]
                sem_sim_t = sem_sim_t[top_idx]
                centroid_scores = centroid_scores[top_idx]
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
                ret_s = self.retention(
                    sb, sf,
                    torch.tensor([m.surprise for m in mems], **_dev(xq)),
                    torch.tensor([self.time - m.last for m in mems], **_dev(xq)),
                    torch.tensor([m.cnt for m in mems], **_dev(xq)),
                )
                transported = transported * ret_s.unsqueeze(-1)
            if update_stats:
                for m in mems:
                    m.last = self.time
                    m.cnt += 1

            final_scores = 0.4 * rerank_scores + 0.4 * centroid_scores + 0.2 * forward_t
            w = F.softmax(final_scores / self.c.retrieval_weight_temperature, dim=0)
            fs = (transported * w.unsqueeze(-1)).sum(0)
            all_batch_mw.append([(m.mid, w[mi].item()) for mi, m in enumerate(mems)])
            all_dominant.append(dominant_mid)
            all_results.append(transported)
            all_masks.append(torch.ones(C, **_dev(xq)))
            all_biases.append(final_scores / self.c.tau)
            all_summaries.append(fs)

        maxC = max(r.shape[0] for r in all_results)
        padded, pm, pd = [], [], []
        for bi in range(B):
            r, mk, db = all_results[bi], all_masks[bi], all_biases[bi]
            gap = maxC - r.shape[0]
            if gap > 0:
                pr = self.empty_state(xq[bi : bi + 1], fq[bi : bi + 1]).expand(gap, -1)
                r = torch.cat([r, pr if self.training else pr.detach()], 0)
                mk = torch.cat([mk, torch.zeros(gap, **_dev(xq))])
                db = torch.cat([db, torch.full((gap,), -1e9, **_dev(xq))])
            padded.append(r)
            pm.append(mk)
            pd.append(db)
        mf = torch.stack(padded)
        mem_mask = torch.stack(pm)
        dir_bias = torch.stack(pd)
        fiber_summary = torch.stack(all_summaries)
        diag.fiber_summary_norm = fiber_summary.norm().item()
        diag.batch_mem_weights = all_batch_mw
        diag.dominant_per_batch = all_dominant
        if diag.dominant_per_batch and diag.dominant_per_batch[0] is not None:
            diag.dominant_memory_id = diag.dominant_per_batch[0]
        refined = self.attn(fq, mf, mem_mask=mem_mask, dir_bias=dir_bias)
        return refined, mem_mask, fiber_summary, diag


class MemLLM(MemLLM):
    def __init__(self, c):
        super().__init__(c)
        self.amm = AMM(c)
        self.bridge = EmbBridge(c)
        self._filler_centroid = None

    def load(self, name="gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tok = GPT2Tokenizer.from_pretrained(name)
        self.llm = GPT2LMHeadModel.from_pretrained(name)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layer_pool = AdaptiveLayerPool(self.llm.config.n_layer + 1, self.c.d_LLM)
        self.content_classifier = ContentTokenClassifier(self.tok, self.c)
        self._degen_guard = DegenerationGuard(self.tok, self.c, self.content_classifier)
        self.bridge.aligner.calibrate(self.llm)
        self.c.vocab_size = self.llm.config.vocab_size
        self._wte_normed = F.normalize(self.llm.transformer.wte.weight.detach(), dim=-1, eps=1e-8)
        self.amm.wte_normed = self._wte_normed
        self._build_wte_neighbor_cache()
        self._compute_filler_centroid()

    def _compute_filler_centroid(self):
        if self.content_classifier is None or self.llm is None:
            self._filler_centroid = None
            return
        wte = self.llm.transformer.wte.weight.detach()
        valid = [tid for tid in sorted(self.content_classifier.filler_ids) if tid < wte.shape[0]]
        if len(valid) < 3:
            self._filler_centroid = None
            return
        filler_vecs = wte[torch.tensor(valid, device=wte.device)]
        self._filler_centroid = F.normalize(filler_vecs.mean(0), dim=-1, eps=1e-8)

    def _get_prefix(self, hs, mask=None, pl=0, update_stats=True, return_extra=False, ids=None):
        pooled, xq, fq = self.extract_state(hs, mask, pl)
        trimmed_mask = mask[:, pl:] if mask is not None and pl > 0 else mask
        if trimmed_mask is not None and pooled.shape[1] != trimmed_mask.shape[1]:
            trimmed_mask = None
        query_content_ids_per_batch = []
        if ids is not None and self.content_classifier is not None:
            for b in range(ids.shape[0]):
                q_ids = list(set(self.content_classifier.get_content_ids_from_tokens(ids[b].tolist())))
                query_content_ids_per_batch.append(q_ids)
        if ids is not None and self.content_classifier is not None:
            query_sem = self._compute_content_semantic_emb(pooled, ids, trimmed_mask)
        else:
            query_sem = pooled.mean(1)
        fibers, mem_mask, fiber_summary, diag = self.amm.retrieve_multi(
            xq,
            fq,
            update_stats=update_stats,
            query_semantic_emb=query_sem,
            query_content_ids_per_batch=query_content_ids_per_batch,
            wte_normed=self._wte_normed,
            content_classifier=self.content_classifier,
        )
        prefix = self.bridge.inject(
            fibers,
            mem_mask,
            fiber_summary=fiber_summary,
            filler_centroid=self._filler_centroid,
        )
        if return_extra:
            content_bias = self._build_content_bias(diag, query_content_ids_per_batch)
            return prefix, fiber_summary, diag, content_bias
        return prefix

    def generate(self, prompt, mt=50, greedy=False):
        tk = self.tok(prompt, return_tensors="pt")
        dev = next(self.parameters()).device
        ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            prefix_cond, fiber_summary, _, content_bias = self._get_prefix(
                o["hs"], mask, update_stats=True, return_extra=True, ids=ids
            )
            vocab_bias = self._compute_vocab_bias(fiber_summary)
            prefix_uncond = self.bridge.build_neutral_prefix(prefix_cond.shape[0], dev) if self.c.use_cfg_decoding else None

        cc = self.content_classifier
        filler_mask_vec = cc.filler_mask(dev) if cc is not None else None
        generated_ids = []
        generated_content_counts: Dict[int, int] = {}
        recent_starters: List[Tuple[int, int]] = []
        newline_ids_set = cc.newline_ids if cc is not None else set()
        content_history: List[Tuple[int, int]] = []
        HARD_MASK = -1e9
        eos_token_id = self.tok.eos_token_id

        for i in range(mt):
            if i > 0 and i % self.c.retrieval_interval == 0:
                with torch.no_grad():
                    o = self.fwd(ids, mask, prefix_cond)
                    pl = o["pl"]
                    prefix_cond, fiber_summary, _, content_bias = self._get_prefix(
                        o["hs"], o["mask"], pl, update_stats=True, return_extra=True, ids=ids
                    )
                    vocab_bias = self._compute_vocab_bias(fiber_summary)
                    if self.c.use_cfg_decoding:
                        prefix_uncond = self.bridge.build_neutral_prefix(prefix_cond.shape[0], dev)

            with torch.no_grad():
                o_cond = self.fwd(ids, mask, prefix_cond)
                lg_cond = o_cond["logits"][:, -1:].squeeze(1)
                if self.c.use_cfg_decoding and prefix_uncond is not None:
                    o_uncond = self.fwd(ids, mask, prefix_uncond)
                    lg_uncond = o_uncond["logits"][:, -1:].squeeze(1)
                    alpha = self.c.cfg_scale
                    if self.c.cfg_decay_steps > 0:
                        alpha *= max(0.0, 1.0 - i / self.c.cfg_decay_steps)
                    lg = lg_cond + alpha * (lg_cond - lg_uncond)
                else:
                    lg = lg_cond.clone()

                step_scale_content = max(self.c.content_bias_floor, 1.0 - i * self.c.content_bias_decay)
                if content_bias is not None and content_bias.abs().max().item() > 0.01:
                    V = min(lg.shape[-1], content_bias.shape[-1])
                    lg[:, :V] = lg[:, :V] + content_bias[:, :V] * self.c.content_bias_scale * step_scale_content

                step_scale_learned = max(self.c.semantic_boost_floor, 1.0 - i * self.c.semantic_boost_decay)
                if vocab_bias is not None:
                    V2 = min(lg.shape[-1], vocab_bias.shape[-1])
                    lg[:, :V2] = lg[:, :V2] + vocab_bias[:, :V2] * self.c.semantic_boost_scale * step_scale_learned

                if cc:
                    for tid, count in generated_content_counts.items():
                        if tid in cc.content_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.content_repeat_penalty * (count ** self.c.content_repeat_exponent)

                if self.c.use_cyclic_content_hard_mask and cc is not None:
                    window_counts: Dict[int, int] = {}
                    cutoff_step = i - self.c.cyclic_content_window
                    for step_idx, tid in content_history:
                        if step_idx >= cutoff_step:
                            window_counts[tid] = window_counts.get(tid, 0) + 1
                    for tid, cnt in window_counts.items():
                        if cnt >= self.c.cyclic_content_max_count and 0 <= tid < lg.shape[-1]:
                            lg[0, tid] = HARD_MASK

                if self.c.use_ngram_repeat_block and len(generated_ids) >= 4:
                    max_n = min(self.c.ngram_repeat_max_n, len(generated_ids) // 2)
                    for n in range(2, max_n + 1):
                        if len(generated_ids) >= 2 * n and generated_ids[-n:] == generated_ids[-2 * n : -n]:
                            expected_next = generated_ids[-n]
                            if 0 <= expected_next < lg.shape[-1]:
                                lg[0, expected_next] -= self.c.ngram_repeat_penalty

                if cc and self._wte_neighbor_cache is not None and recent_starters:
                    for prev_tid, _prev_step in recent_starters:
                        for nid in self._wte_neighbor_cache.get(prev_tid, []):
                            if nid in cc.word_starter_ids:
                                continue
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.bpe_echo_penalty
                if cc and generated_ids and generated_ids[-1] in cc.content_starter_ids:
                    for tid in cc.content_ids:
                        if tid not in cc.word_starter_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.post_starter_nonstarter_penalty

                if self.c.use_newline_hard_gate and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if i < self.c.newline_hard_gate_min_step or content_count_so_far < self.c.newline_hard_gate_min_content:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] = HARD_MASK
                if self.c.use_eos_hard_mask and eos_token_id is not None and i < self.c.eos_hard_mask_steps and eos_token_id < lg.shape[-1]:
                    lg[0, eos_token_id] = HARD_MASK

                if self.c.use_content_gated_newline and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if content_count_so_far < self.c.min_content_tokens_before_newline:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.late_newline_penalty

                if self.c.use_sustained_filler and filler_mask_vec is not None and i < self.c.sustained_filler_steps:
                    V = min(lg.shape[-1], filler_mask_vec.shape[0])
                    filler_decay = max(1.0 - i * self.c.sustained_filler_decay, 0.0)
                    lg[0, :V] -= filler_mask_vec[:V] * self.c.sustained_filler_penalty * filler_decay

                if self._degen_guard is not None:
                    lg = self._degen_guard.process(lg, generated_ids, i)

                if greedy:
                    nxt = lg.argmax(-1, keepdim=True)
                else:
                    lg_t = lg / self.c.gen_temp
                    p = F.softmax(lg_t, -1)
                    sp, si = torch.sort(p, descending=True)
                    cs = torch.cumsum(sp, -1)
                    rm = cs - sp > self.c.gen_top_p
                    sp[rm] = 0
                    total = sp.sum(-1, keepdim=True)
                    if (total < 1e-10).any():
                        sp[:, 0] = 1.0
                        total = sp.sum(-1, keepdim=True)
                    sp = sp / total
                    nxt = si.gather(-1, torch.multinomial(sp, 1))

            nxt_id = nxt.item()
            if nxt_id == self.tok.eos_token_id and len(generated_ids) >= self.c.degen_min_tokens:
                break
            generated_ids.append(nxt_id)
            if cc and nxt_id in cc.content_ids:
                generated_content_counts[nxt_id] = generated_content_counts.get(nxt_id, 0) + 1
                content_history.append((i, nxt_id))
                if nxt_id in cc.word_starter_ids:
                    recent_starters.append((nxt_id, i))
            recent_starters = [(t, s) for (t, s) in recent_starters if (i - s) < self.c.bpe_echo_window]
            if len(content_history) > 2 * self.c.cyclic_content_window:
                content_history = content_history[-self.c.cyclic_content_window :]
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)
        return self.tok.decode(ids[0], skip_special_tokens=True)


def hungarian_max_assignment(sim: torch.Tensor) -> Tuple[torch.Tensor, float]:
    device = sim.device
    n_rows, n_cols = sim.shape
    if n_rows == 0 or n_cols == 0:
        return torch.empty(0, 2, dtype=torch.long, device=device), 0.0
    transposed = False
    original_sim = sim
    if n_rows > n_cols:
        sim = sim.T
        n_rows, n_cols = sim.shape
        transposed = True
    cost = (-sim).detach().cpu().numpy().astype("float64")
    import numpy as np

    INF = float("inf")
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, INF)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = []
    total = 0.0
    for j in range(1, n_cols + 1):
        i = p[j]
        if i > 0 and i <= n_rows:
            if transposed:
                pairs.append((j - 1, i - 1))
                total += original_sim[j - 1, i - 1].item()
            else:
                pairs.append((i - 1, j - 1))
                total += original_sim[i - 1, j - 1].item()
    pairs_tensor = torch.tensor(pairs, dtype=torch.long, device=device) if pairs else torch.empty(0, 2, dtype=torch.long, device=device)
    return pairs_tensor, total


@dataclass
class Cfg(Cfg):
    degen_early_punct_penalty: float = 8.0
    degen_early_newline_penalty: float = 8.0
    content_bias_scale: float = 6.0

    use_mean_centered_scoring: bool = True
    mc_keep_margin: float = 0.0
    mc_min_keep: int = 1
    mc_require_min_candidates: int = 2

    use_hungarian_fwd: bool = True
    hungarian_max_n: int = 24

    use_cfg_decoding: bool = True
    use_contrastive_memory_cfg: bool = True
    cfg_scale: float = 2.5
    cfg_decay_steps: int = 0

    use_content_semantic_tail: bool = True
    content_tail_slots: int = 2
    tail_head_hidden: int = 512

    def __post_init__(self):
        super().__post_init__()
        assert self.content_tail_slots >= 0
        assert self.content_tail_slots < self.L_mem


@dataclass
class RetrievalDiag(RetrievalDiag):
    n_after_mean_center: int = 0
    mean_center_applied: bool = False
    mean_center_dropped_ids: List[int] = field(default_factory=list)
    mean_center_raw_scores: Dict[int, float] = field(default_factory=dict)
    mean_center_final_scores: Dict[int, float] = field(default_factory=dict)
    hungarian_used: bool = False
    non_dominant_per_batch: List[List[int]] = field(default_factory=list)


class ContentSemanticTailHead(nn.Module):
    def __init__(self, d_F: int, d_LLM: int, n_slots: int, hidden: int = 512):
        super().__init__()
        self.n_slots = n_slots
        self.d_LLM = d_LLM
        if n_slots == 0:
            self.shared = None
            self.slot_heads = nn.ModuleList([])
            return
        self.shared = nn.Sequential(
            nn.Linear(d_F, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
        )
        self.slot_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, d_LLM), nn.LayerNorm(d_LLM))
            for _ in range(n_slots)
        ])
        for head in self.slot_heads:
            nn.init.normal_(head[0].weight, std=0.02)
            nn.init.zeros_(head[0].bias)

    def forward(self, fiber_summary: torch.Tensor) -> Optional[torch.Tensor]:
        if self.n_slots == 0 or self.shared is None:
            return None
        h = self.shared(fiber_summary)
        return torch.stack([head(h) for head in self.slot_heads], dim=1)


class EmbBridge(EmbBridge):
    def __init__(self, c):
        nn.Module.__init__(self)
        self.c = c
        self.proj = QFormerProj(c)
        self.ext = StateExtractor(c)
        self.pe = nn.Parameter(torch.randn(c.L_mem, c.d_LLM) * 0.02)
        self.bypass = ContentBypass(c.d_F, c.d_LLM, gate_bias=c.bypass_init_gate_bias)
        self.aligner = PrefixAligner(c.d_LLM, c.prefix_init_scale)
        self.tail_head = ContentSemanticTailHead(
            c.d_F, c.d_LLM,
            n_slots=c.content_tail_slots if c.use_content_semantic_tail else 0,
            hidden=c.tail_head_hidden,
        )
        self._last_inject_diag = {}
        self._last_fiber_summary = None
        self._last_tail_slots = None
        self._filler_centroid = None

    def _build_body_prefix(self, fibers, mem_mask, fiber_summary):
        qf_out = self.proj(fibers, mem_mask) + self.pe.unsqueeze(0)
        bp_out = None
        gate_val = None
        if fiber_summary is not None:
            qf_context = qf_out.mean(1)
            bp_out = self.bypass(fiber_summary, qf_context)
            gate_val = self.bypass._last_gate
            qf_out = qf_out + bp_out.unsqueeze(1)
        qf_out = self.aligner(qf_out)
        return qf_out, bp_out, gate_val

    def _apply_filler_projection_and_clamp(self, qf_out, filler_centroid):
        L = qf_out.shape[1]
        filler_dir_used = False
        if self.c.use_filler_direction_projection and filler_centroid is not None:
            n_proj = min(self.c.filler_projection_last_slots, L)
            fd = filler_centroid.view(1, 1, -1)
            mask_slot = torch.zeros(L, device=qf_out.device)
            mask_slot[L - n_proj :] = 1.0
            mask_slot = mask_slot.view(1, -1, 1)
            comp = (qf_out * fd).sum(-1, keepdim=True)
            qf_out = qf_out - comp * fd * mask_slot
            filler_dir_used = True
        if self.c.use_prefix_norm_clamp:
            target_std = self.aligner._target_std.item()
            target_norm = target_std * math.sqrt(self.c.d_LLM)
            max_allowed = target_norm * self.c.prefix_norm_clamp_ratio
            slot_norms = qf_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(max_allowed / slot_norms, max=1.0)
            qf_out = qf_out * scale
        return qf_out, filler_dir_used

    def inject(self, fibers, mem_mask=None, fiber_summary=None, filler_centroid=None, **_ignored):
        qf_out, bp_out, gate_val = self._build_body_prefix(fibers, mem_mask, fiber_summary)
        tail_slots_used = 0
        if self.c.use_content_semantic_tail and self.c.content_tail_slots > 0 and fiber_summary is not None:
            tail = self.tail_head(fiber_summary)
            if tail is not None:
                tail = self.aligner(tail)
                n = self.c.content_tail_slots
                qf_out = torch.cat([qf_out[:, :-n, :], tail], dim=1)
                tail_slots_used = n
                self._last_tail_slots = tail.detach()
        else:
            self._last_tail_slots = None
        qf_out, filler_dir_used = self._apply_filler_projection_and_clamp(qf_out, filler_centroid)
        self._last_fiber_summary = fiber_summary.detach() if fiber_summary is not None else None
        self._last_inject_diag = {
            "bypass_gate": gate_val.mean().item() if gate_val is not None else None,
            "qf_norm": qf_out.norm().item(),
            "bypass_norm": bp_out.norm().item() if bp_out is not None else 0.0,
            "aligner_scale": torch.sigmoid(self.aligner.scale_logit).item() * self.aligner._target_std.item(),
            "last_slot_norm_per_b": qf_out[:, -1].norm(dim=-1).mean().item(),
            "tail_slots_used": tail_slots_used,
            "filler_dir_projected": filler_dir_used,
        }
        return qf_out


class AMM(AMM):
    def _compute_forward_hungarian(self, query_ids, mem_ids, wte_normed, query_idf=None, idf_floor=0.1):
        if not query_ids or not mem_ids:
            return 0.0
        V = wte_normed.shape[0]
        q_valid = [q for q in query_ids if q < V]
        m_valid = [m for m in mem_ids if m < V]
        if not q_valid or not m_valid:
            return 0.0
        if max(len(q_valid), len(m_valid)) > self.c.hungarian_max_n:
            return self._compute_forward_maxsim(q_valid, m_valid, wte_normed, query_idf, idf_floor)
        q_vecs = wte_normed[q_valid]
        m_vecs = wte_normed[m_valid]
        sim = q_vecs @ m_vecs.T
        pairs, _ = hungarian_max_assignment(sim)
        if pairs.numel() == 0:
            return 0.0
        matched_sims = sim[pairs[:, 0], pairs[:, 1]]
        if query_idf is not None:
            q_ids_for_pairs = [q_valid[int(r.item())] for r in pairs[:, 0]]
            w = torch.tensor([max(query_idf.get(q, idf_floor), idf_floor) for q in q_ids_for_pairs], device=wte_normed.device, dtype=matched_sims.dtype)
            return ((matched_sims * w).sum() / w.sum().clamp(min=1e-8)).item()
        return matched_sims.mean().item()

    def _compute_bidi_min(self, q_ids, m_ids, wte_normed, query_idf, idf_floor):
        fwd = self._compute_forward_hungarian(q_ids, m_ids, wte_normed, query_idf, idf_floor) if self.c.use_hungarian_fwd else self._compute_forward_maxsim(q_ids, m_ids, wte_normed, query_idf, idf_floor)
        bwd = self._compute_backward_maxsim(q_ids, m_ids, wte_normed, query_idf, idf_floor)
        return fwd, bwd, min(fwd, bwd)

    def _check_consolidation_compatible(self, existing_content_ids, new_content_ids):
        if not existing_content_ids or not new_content_ids:
            return True
        if self.wte_normed is None:
            return True
        _, _, m = self._compute_bidi_min(existing_content_ids, new_content_ids, self.wte_normed, None, self.c.idf_floor)
        return m >= self.c.consol_maxsim_min

    def retrieve_multi(self, xq, fq, topk=None, bw=None, update_stats=True, query_semantic_emb=None, query_content_ids_per_batch=None, wte_normed=None, content_classifier=None):
        B = xq.shape[0]
        dev = xq.device
        topk = topk or self.c.retrieval_topk
        bw = bw or self.c.retrieval_beam
        recall_k = int(topk * self.c.retrieval_recall_factor)
        flat_thresh = self.c.flat_scan_threshold_factor * topk
        qdir = self.dir_pred(xq, fq)
        diag = RetrievalDiag()
        corpus_idf = self._compute_corpus_idf(content_classifier) if self.c.use_idf_retrieval else None
        diag.idf_applied = corpus_idf is not None
        diag.centroid_applied = self.c.use_idf_centroid
        diag.hungarian_used = self.c.use_hungarian_fwd
        idf_floor = self.c.idf_floor
        if not self.tree.store:
            empty = self.empty_state(xq, fq)
            mask = torch.ones(B, 1, **_dev(xq))
            summary = empty.mean(1) if empty.dim() == 3 else empty
            diag.fiber_summary_norm = summary.norm().item()
            diag.batch_mem_weights = [[] for _ in range(B)]
            diag.dominant_per_batch = [None for _ in range(B)]
            diag.non_dominant_per_batch = [[] for _ in range(B)]
            return empty.unsqueeze(1), mask, summary, diag
        all_results, all_masks, all_biases, all_summaries = [], [], [], []
        all_batch_mw, all_dominant, all_non_dominant = [], [], []
        wn = wte_normed if wte_normed is not None else self.wte_normed
        for b in range(B):
            n_store = len(self.tree.store)
            if n_store <= flat_thresh:
                mids = list(self.tree.store.keys())
                diag.was_flat_scan = True
            else:
                scored = self.tree.retrieve(qdir[b].detach(), bw)
                mids = [s[0] for s in scored[:recall_k]]
            mems = [self.tree.store[i] for i in mids if i in self.tree.store]
            diag.recall_count = len(mems)
            diag.n_candidates_initial = len(mems)
            if not mems:
                empty = self.empty_state(xq[b:b+1], fq[b:b+1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1, **_dev(xq)))
                all_biases.append(torch.zeros(1, **_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([])
                all_dominant.append(None)
                all_non_dominant.append([])
                continue
            q_content_ids = query_content_ids_per_batch[b] if query_content_ids_per_batch and b < len(query_content_ids_per_batch) else []
            q_strict = []
            if content_classifier is not None:
                q_strict = [t for t in q_content_ids if t in content_classifier.strict_content_starter_ids and wn is not None and t < wn.shape[0]]
            if self.c.use_strict_content_overlap_gate and q_strict and wn is not None and content_classifier is not None:
                overlap_counts = torch.zeros(len(mems), dtype=torch.long, device=dev)
                for mi, mem in enumerate(mems):
                    m_strict = [t for t in mem.content_token_ids if t in content_classifier.strict_content_starter_ids and t < wn.shape[0]]
                    cnt = self._count_strict_overlap_matches(q_strict, m_strict, wn, self.c.strict_overlap_sim_threshold)
                    overlap_counts[mi] = cnt
                    diag.per_memory_strict_overlap[mem.mid] = cnt
                pass_mask = overlap_counts >= self.c.strict_overlap_min_matches
                if int(pass_mask.sum().item()) < self.c.strict_overlap_min_keep:
                    _, top_keep = overlap_counts.topk(min(max(self.c.strict_overlap_min_keep, 1), len(mems)))
                    pass_mask = torch.zeros(len(mems), dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                diag.strict_overlap_dropped_ids = [mems[i].mid for i in (~pass_mask).nonzero(as_tuple=True)[0].tolist()]
                diag.strict_overlap_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < len(mems):
                    mems = [mems[i] for i in keep_local.tolist()]
            diag.n_after_strict_overlap_gate = len(mems)
            C_init = len(mems)
            if C_init == 0:
                empty = self.empty_state(xq[b:b+1], fq[b:b+1])
                all_results.append(empty.squeeze(0).unsqueeze(0))
                all_masks.append(torch.ones(1, **_dev(xq)))
                all_biases.append(torch.zeros(1, **_dev(xq)))
                all_summaries.append(empty.squeeze(0))
                all_batch_mw.append([])
                all_dominant.append(None)
                all_non_dominant.append([])
                continue
            sb = torch.stack([m.base.to(dev) for m in mems])
            sf = torch.stack([m.fiber.to(dev) for m in mems])
            md = torch.stack([m.dirn.to(dev) for m in mems])
            sem_sim_t = torch.zeros(C_init, device=dev)
            if query_semantic_emb is not None:
                for mi, mem in enumerate(mems):
                    if mem.semantic_emb is not None:
                        sem_sim_t[mi] = F.cosine_similarity(query_semantic_emb[b:b+1], mem.semantic_emb.unsqueeze(0).to(dev), dim=-1).squeeze()
            forward_t = torch.zeros(C_init, device=dev)
            backward_t = torch.zeros(C_init, device=dev)
            bidi_min_t = torch.zeros(C_init, device=dev)
            if q_content_ids and wn is not None:
                for mi, mem in enumerate(mems):
                    scoring_ids = self._get_mem_scoring_ids(mem)
                    fwd, bwd, bmin = self._compute_bidi_min(q_content_ids, scoring_ids, wn, corpus_idf, idf_floor)
                    forward_t[mi] = fwd
                    backward_t[mi] = bwd
                    bidi_min_t[mi] = bmin
            if self.c.use_upstream_semantic_gate and q_content_ids and wn is not None:
                fwd_pass = forward_t >= self.c.upstream_gate_fwd_idf_floor
                sem_pass = sem_sim_t >= self.c.upstream_gate_sem_floor
                pass_mask = (fwd_pass & sem_pass) if self.c.upstream_gate_require_both else (fwd_pass | sem_pass)
                if int(pass_mask.sum().item()) < self.c.upstream_gate_min_keep:
                    top_keep = forward_t.topk(min(max(self.c.upstream_gate_min_keep, 1), C_init)).indices
                    pass_mask = torch.zeros(C_init, dtype=torch.bool, device=dev)
                    pass_mask[top_keep] = True
                diag.upstream_gate_dropped_ids = [mems[i].mid for i in (~pass_mask).nonzero(as_tuple=True)[0].tolist()]
                diag.upstream_semantic_gate_applied = True
                keep_local = pass_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C_init:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb = sb[keep_local]
                    sf = sf[keep_local]
                    md = md[keep_local]
                    sem_sim_t = sem_sim_t[keep_local]
                    forward_t = forward_t[keep_local]
                    backward_t = backward_t[keep_local]
                    bidi_min_t = bidi_min_t[keep_local]
                    C_init = len(mems)
            diag.n_after_upstream_semantic_gate = C_init
            raw_dir_sim = torch.einsum("d,cd->c", qdir[b], md)
            diag.top_dir_sim = raw_dir_sim.max().item() if C_init > 0 else 0.0
            diag.top_sem_sim = sem_sim_t.max().item() if C_init > 0 else 0.0
            diag.top_forward_maxsim = forward_t.max().item() if C_init > 0 else 0.0
            diag.top_backward_maxsim = backward_t.max().item() if C_init > 0 else 0.0
            diag.top_bidi_min = bidi_min_t.max().item() if C_init > 0 else 0.0
            centroid_scores = torch.zeros(C_init, device=dev)
            if self.c.use_idf_centroid and q_content_ids and wn is not None:
                q_centroid = self._compute_idf_weighted_centroid(q_content_ids, wn, corpus_idf, idf_floor)
                if q_centroid is not None:
                    for mi, mem in enumerate(mems):
                        m_centroid = self._compute_idf_weighted_centroid(self._get_mem_scoring_ids(mem), wn, corpus_idf, idf_floor)
                        if m_centroid is not None:
                            centroid_scores[mi] = (q_centroid @ m_centroid).item()
                diag.top_centroid_cosine = centroid_scores.max().item() if C_init > 0 else 0.0
            combined_sim = self.c.ret_centroid_weight * centroid_scores + self.c.ret_sem_weight * sem_sim_t + self.c.ret_bidi_min_weight * bidi_min_t + self.c.ret_forward_maxsim_weight * forward_t + self.c.ret_dir_weight * raw_dir_sim
            C = C_init
            sem_thresh = max(self.c.gate_sem_floor, sem_sim_t.max().item() * self.c.gate_sem_ratio) if C > 0 else self.c.gate_sem_floor
            bidi_thresh = max(self.c.gate_bidi_floor, bidi_min_t.max().item() * self.c.gate_bidi_ratio if C > 0 else 0.0, self.c.gate_bidi_hard_min)
            hard_mask = (sem_sim_t >= sem_thresh) & (bidi_min_t >= bidi_thresh)
            gate_affinity = self.c.gate_sem_weight * sem_sim_t + self.c.gate_bidi_weight * bidi_min_t
            diag.top_gate_affinity = gate_affinity.max().item() if C > 0 else 0.0
            diag.gate_threshold = max(sem_thresh, bidi_thresh)
            diag.n_gate_pass = int(hard_mask.sum().item())
            if hard_mask.sum().item() == 0 and C > 0:
                hard_mask[torch.minimum(sem_sim_t, bidi_min_t).argmax()] = True
            diag.n_after_hard_filter = int(hard_mask.sum().item())
            for mi, mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid] = gate_affinity[mi].item()
            keep_indices = hard_mask.nonzero(as_tuple=True)[0]
            if keep_indices.numel() > 0 and keep_indices.numel() < C:
                mems = [mems[i] for i in keep_indices.tolist()]
                sb = sb[keep_indices]; sf = sf[keep_indices]
                combined_sim = combined_sim[keep_indices]
                raw_dir_sim = raw_dir_sim[keep_indices]
                forward_t = forward_t[keep_indices]
                bidi_min_t = bidi_min_t[keep_indices]
                sem_sim_t = sem_sim_t[keep_indices]
                centroid_scores = centroid_scores[keep_indices]
                C = len(mems)
            rerank_scores = self.reranker(xq[b:b+1], fq[b:b+1], sb.unsqueeze(0), sf.unsqueeze(0), combined_sim.unsqueeze(0)).squeeze(0)
            diag.reranker_delta_mean = (rerank_scores - combined_sim).abs().mean().item()
            diag.top_reranker_score = rerank_scores.max().item() if C > 0 else 0.0
            if C > 1:
                score_mask = rerank_scores >= rerank_scores.max() * self.c.score_keep_ratio
                if score_mask.sum().item() < 1:
                    score_mask[rerank_scores.argmax()] = True
                score_keep = score_mask.nonzero(as_tuple=True)[0]
                diag.n_after_score_filter = score_keep.numel()
                if score_keep.numel() < C:
                    mems = [mems[i] for i in score_keep.tolist()]
                    sb = sb[score_keep]; sf = sf[score_keep]
                    rerank_scores = rerank_scores[score_keep]
                    forward_t = forward_t[score_keep]
                    bidi_min_t = bidi_min_t[score_keep]
                    sem_sim_t = sem_sim_t[score_keep]
                    centroid_scores = centroid_scores[score_keep]
                    C = len(mems)
            else:
                diag.n_after_score_filter = C
            if C > 1 and forward_t.max().item() > 0:
                coherence_keep = (forward_t >= forward_t.max() * self.c.fwd_coherence_ratio).nonzero(as_tuple=True)[0]
                diag.n_after_coherence_filter = coherence_keep.numel()
                if coherence_keep.numel() >= 1 and coherence_keep.numel() < C:
                    mems = [mems[i] for i in coherence_keep.tolist()]
                    sb = sb[coherence_keep]; sf = sf[coherence_keep]
                    rerank_scores = rerank_scores[coherence_keep]
                    forward_t = forward_t[coherence_keep]
                    bidi_min_t = bidi_min_t[coherence_keep]
                    sem_sim_t = sem_sim_t[coherence_keep]
                    centroid_scores = centroid_scores[coherence_keep]
                    C = len(mems)
            else:
                diag.n_after_coherence_filter = C
            if C > 1 and bidi_min_t.max().item() > 0:
                gap_keep = (bidi_min_t >= (bidi_min_t.max().item() - self.c.bidi_absolute_gap)).nonzero(as_tuple=True)[0]
                diag.n_after_bidi_gap_filter = gap_keep.numel()
                if gap_keep.numel() >= 1 and gap_keep.numel() < C:
                    mems = [mems[i] for i in gap_keep.tolist()]
                    sb = sb[gap_keep]; sf = sf[gap_keep]
                    rerank_scores = rerank_scores[gap_keep]
                    forward_t = forward_t[gap_keep]
                    bidi_min_t = bidi_min_t[gap_keep]
                    sem_sim_t = sem_sim_t[gap_keep]
                    centroid_scores = centroid_scores[gap_keep]
                    C = len(mems)
            else:
                diag.n_after_bidi_gap_filter = C
            raw_composite = 0.4 * centroid_scores + 0.4 * forward_t + 0.15 * bidi_min_t + 0.05 * sem_sim_t.clamp(min=0)
            if self.c.use_mean_centered_scoring and C >= self.c.mc_require_min_candidates:
                C_f = float(C)
                sum_raw = raw_composite.sum()
                centered = (C_f / (C_f - 1.0)) * raw_composite - sum_raw / (C_f - 1.0)
                for mi, mem in enumerate(mems):
                    diag.mean_center_raw_scores[mem.mid] = raw_composite[mi].item()
                    diag.mean_center_final_scores[mem.mid] = centered[mi].item()
                keep_mask = centered > self.c.mc_keep_margin
                if int(keep_mask.sum().item()) < self.c.mc_min_keep:
                    top_keep = centered.topk(min(max(self.c.mc_min_keep, 1), C)).indices
                    keep_mask = torch.zeros(C, dtype=torch.bool, device=dev)
                    keep_mask[top_keep] = True
                if (~keep_mask).any():
                    diag.mean_center_applied = True
                    diag.mean_center_dropped_ids = [mems[i].mid for i in (~keep_mask).nonzero(as_tuple=True)[0].tolist()]
                keep_local = keep_mask.nonzero(as_tuple=True)[0]
                if keep_local.numel() < C:
                    mems = [mems[i] for i in keep_local.tolist()]
                    sb = sb[keep_local]; sf = sf[keep_local]
                    rerank_scores = rerank_scores[keep_local]
                    forward_t = forward_t[keep_local]
                    bidi_min_t = bidi_min_t[keep_local]
                    sem_sim_t = sem_sim_t[keep_local]
                    centroid_scores = centroid_scores[keep_local]
                    C = len(mems)
            diag.n_after_mean_center = C
            dominant_mid = None
            non_dominant_mids = []
            if C >= 1:
                final_rank = 0.4 * rerank_scores + 0.4 * centroid_scores + 0.2 * forward_t
                dom_idx = int(final_rank.argmax().item())
                dominant_mid = mems[dom_idx].mid
                non_dominant_mids = [mems[i].mid for i in range(C) if i != dom_idx]
            if not self.training and C > topk:
                _, top_idx = rerank_scores.topk(topk)
                mems = [mems[i] for i in top_idx.cpu().tolist()]
                sb = sb[top_idx]; sf = sf[top_idx]
                rerank_scores = rerank_scores[top_idx]
                forward_t = forward_t[top_idx]
                bidi_min_t = bidi_min_t[top_idx]
                sem_sim_t = sem_sim_t[top_idx]
                centroid_scores = centroid_scores[top_idx]
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
                ret_s = self.retention(sb, sf, torch.tensor([m.surprise for m in mems], **_dev(xq)), torch.tensor([self.time - m.last for m in mems], **_dev(xq)), torch.tensor([m.cnt for m in mems], **_dev(xq)))
                transported = transported * ret_s.unsqueeze(-1)
            if update_stats:
                for m in mems:
                    m.last = self.time
                    m.cnt += 1
            final_scores = 0.4 * rerank_scores + 0.4 * centroid_scores + 0.2 * forward_t
            w = F.softmax(final_scores / self.c.retrieval_weight_temperature, dim=0)
            fs = (transported * w.unsqueeze(-1)).sum(0)
            all_batch_mw.append([(m.mid, w[mi].item()) for mi, m in enumerate(mems)])
            all_dominant.append(dominant_mid)
            all_non_dominant.append(non_dominant_mids)
            all_results.append(transported)
            all_masks.append(torch.ones(C, **_dev(xq)))
            all_biases.append(final_scores / self.c.tau)
            all_summaries.append(fs)
        maxC = max(r.shape[0] for r in all_results)
        padded, pm, pd = [], [], []
        for bi in range(B):
            r, mk, db = all_results[bi], all_masks[bi], all_biases[bi]
            gap = maxC - r.shape[0]
            if gap > 0:
                pr = self.empty_state(xq[bi:bi+1], fq[bi:bi+1]).expand(gap, -1)
                r = torch.cat([r, pr if self.training else pr.detach()], 0)
                mk = torch.cat([mk, torch.zeros(gap, **_dev(xq))])
                db = torch.cat([db, torch.full((gap,), -1e9, **_dev(xq))])
            padded.append(r)
            pm.append(mk)
            pd.append(db)
        mf = torch.stack(padded)
        mem_mask = torch.stack(pm)
        dir_bias = torch.stack(pd)
        fiber_summary = torch.stack(all_summaries)
        diag.fiber_summary_norm = fiber_summary.norm().item()
        diag.batch_mem_weights = all_batch_mw
        diag.dominant_per_batch = all_dominant
        diag.non_dominant_per_batch = all_non_dominant
        if diag.dominant_per_batch and diag.dominant_per_batch[0] is not None:
            diag.dominant_memory_id = diag.dominant_per_batch[0]
        refined = self.attn(fq, mf, mem_mask=mem_mask, dir_bias=dir_bias)
        return refined, mem_mask, fiber_summary, diag


class MemLLM(MemLLM):
    def __init__(self, c):
        super().__init__(c)
        self.amm = AMM(c)
        self.bridge = EmbBridge(c)
        self._filler_centroid = None

    def _build_contrastive_uncond_prefix(self, diag, prefix_cond):
        dev = prefix_cond.device
        B = prefix_cond.shape[0]
        uncond_prefix = torch.zeros_like(prefix_cond)
        for b in range(B):
            mids = diag.non_dominant_per_batch[b] if b < len(diag.non_dominant_per_batch) else []
            mids = [m for m in mids if m in self.amm.tree.store]
            if mids:
                fvecs = torch.stack([self.amm.tree.store[m].fiber.to(dev) for m in mids])
                non_dom = fvecs.mean(0, keepdim=True)
                pref_b = self.bridge.inject(
                    non_dom.unsqueeze(1),
                    torch.ones(1, 1, device=dev),
                    fiber_summary=non_dom,
                    filler_centroid=self._filler_centroid,
                )
                uncond_prefix[b:b+1] = pref_b
            else:
                uncond_prefix[b:b+1] = self.bridge.build_neutral_prefix(1, dev)
        return uncond_prefix

    def generate(self, prompt, mt=50, greedy=False):
        tk = self.tok(prompt, return_tensors="pt")
        dev = next(self.parameters()).device
        ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            prefix_cond, fiber_summary, diag, content_bias = self._get_prefix(
                o["hs"], mask, update_stats=True, return_extra=True, ids=ids
            )
            vocab_bias = self._compute_vocab_bias(fiber_summary)
            if self.c.use_cfg_decoding:
                prefix_uncond = self._build_contrastive_uncond_prefix(diag, prefix_cond) if self.c.use_contrastive_memory_cfg else self.bridge.build_neutral_prefix(prefix_cond.shape[0], dev)
            else:
                prefix_uncond = None
        generated_ids = []
        generated_content_counts: Dict[int, int] = {}
        content_history: List[Tuple[int, int]] = []
        recent_starters: List[Tuple[int, int]] = []
        cc = self.content_classifier
        newline_ids_set = cc.newline_ids if cc is not None else set()
        HARD_MASK = -1e9
        eos_token_id = self.tok.eos_token_id
        for i in range(mt):
            if i > 0 and i % self.c.retrieval_interval == 0:
                with torch.no_grad():
                    o = self.fwd(ids, mask, prefix_cond)
                    pl = o["pl"]
                    prefix_cond, fiber_summary, diag, content_bias = self._get_prefix(
                        o["hs"], o["mask"], pl, update_stats=True, return_extra=True, ids=ids
                    )
                    vocab_bias = self._compute_vocab_bias(fiber_summary)
                    if self.c.use_cfg_decoding:
                        prefix_uncond = self._build_contrastive_uncond_prefix(diag, prefix_cond) if self.c.use_contrastive_memory_cfg else self.bridge.build_neutral_prefix(prefix_cond.shape[0], dev)
            with torch.no_grad():
                o_cond = self.fwd(ids, mask, prefix_cond)
                lg_cond = o_cond["logits"][:, -1:].squeeze(1)
                if self.c.use_cfg_decoding and prefix_uncond is not None:
                    o_uncond = self.fwd(ids, mask, prefix_uncond)
                    lg_uncond = o_uncond["logits"][:, -1:].squeeze(1)
                    alpha = self.c.cfg_scale
                    if self.c.cfg_decay_steps > 0:
                        alpha *= max(0.0, 1.0 - i / self.c.cfg_decay_steps)
                    lg = lg_cond + alpha * (lg_cond - lg_uncond)
                else:
                    lg = lg_cond.clone()
                step_scale_content = max(self.c.content_bias_floor, 1.0 - i * self.c.content_bias_decay)
                if content_bias is not None and content_bias.abs().max().item() > 0.01:
                    V = min(lg.shape[-1], content_bias.shape[-1])
                    lg[:, :V] = lg[:, :V] + content_bias[:, :V] * self.c.content_bias_scale * step_scale_content
                step_scale_learned = max(self.c.semantic_boost_floor, 1.0 - i * self.c.semantic_boost_decay)
                if vocab_bias is not None:
                    V2 = min(lg.shape[-1], vocab_bias.shape[-1])
                    lg[:, :V2] = lg[:, :V2] + vocab_bias[:, :V2] * self.c.semantic_boost_scale * step_scale_learned
                if cc:
                    for tid, count in generated_content_counts.items():
                        if tid in cc.content_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.content_repeat_penalty * (count ** self.c.content_repeat_exponent)
                if self.c.use_cyclic_content_hard_mask and cc is not None:
                    window_counts: Dict[int, int] = {}
                    cutoff_step = i - self.c.cyclic_content_window
                    for step_idx, tid in content_history:
                        if step_idx >= cutoff_step:
                            window_counts[tid] = window_counts.get(tid, 0) + 1
                    for tid, cnt in window_counts.items():
                        if cnt >= self.c.cyclic_content_max_count and 0 <= tid < lg.shape[-1]:
                            lg[0, tid] = HARD_MASK
                if self.c.use_ngram_repeat_block and len(generated_ids) >= 4:
                    max_n = min(self.c.ngram_repeat_max_n, len(generated_ids) // 2)
                    for n in range(2, max_n + 1):
                        if len(generated_ids) >= 2 * n and generated_ids[-n:] == generated_ids[-2 * n : -n]:
                            expected_next = generated_ids[-n]
                            if 0 <= expected_next < lg.shape[-1]:
                                lg[0, expected_next] -= self.c.ngram_repeat_penalty
                if cc and self._wte_neighbor_cache is not None and recent_starters:
                    for prev_tid, _ in recent_starters:
                        for nid in self._wte_neighbor_cache.get(prev_tid, []):
                            if nid in cc.word_starter_ids:
                                continue
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.bpe_echo_penalty
                if cc and generated_ids and generated_ids[-1] in cc.content_starter_ids:
                    for tid in cc.content_ids:
                        if tid not in cc.word_starter_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.post_starter_nonstarter_penalty
                if self.c.use_newline_hard_gate and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if i < self.c.newline_hard_gate_min_step or content_count_so_far < self.c.newline_hard_gate_min_content:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] = HARD_MASK
                if self.c.use_eos_hard_mask and eos_token_id is not None and i < self.c.eos_hard_mask_steps and eos_token_id < lg.shape[-1]:
                    lg[0, eos_token_id] = HARD_MASK
                if self.c.use_content_gated_newline and cc is not None:
                    content_count_so_far = sum(generated_content_counts.values())
                    if content_count_so_far < self.c.min_content_tokens_before_newline:
                        for nid in newline_ids_set:
                            if nid < lg.shape[-1]:
                                lg[0, nid] -= self.c.late_newline_penalty
                if self._degen_guard is not None:
                    lg = self._degen_guard.process(lg, generated_ids, i)
                if greedy:
                    nxt = lg.argmax(-1, keepdim=True)
                else:
                    lg_t = lg / self.c.gen_temp
                    p = F.softmax(lg_t, -1)
                    sp, si = torch.sort(p, descending=True)
                    cs = torch.cumsum(sp, -1)
                    rm = cs - sp > self.c.gen_top_p
                    sp[rm] = 0
                    total = sp.sum(-1, keepdim=True)
                    if (total < 1e-10).any():
                        sp[:, 0] = 1.0
                        total = sp.sum(-1, keepdim=True)
                    sp = sp / total
                    nxt = si.gather(-1, torch.multinomial(sp, 1))
            nxt_id = nxt.item()
            if nxt_id == self.tok.eos_token_id and len(generated_ids) >= self.c.degen_min_tokens:
                break
            generated_ids.append(nxt_id)
            if cc and nxt_id in cc.content_ids:
                generated_content_counts[nxt_id] = generated_content_counts.get(nxt_id, 0) + 1
                content_history.append((i, nxt_id))
                if nxt_id in cc.word_starter_ids:
                    recent_starters.append((nxt_id, i))
            recent_starters = [(t, s) for (t, s) in recent_starters if (i - s) < self.c.bpe_echo_window]
            if len(content_history) > 2 * self.c.cyclic_content_window:
                content_history = content_history[-self.c.cyclic_content_window :]
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)
        return self.tok.decode(ids[0], skip_special_tokens=True)


class Trainer(Trainer):
    def __init__(self, m, c):
        super().__init__(m, c)
        if c.use_content_semantic_tail and c.content_tail_slots > 0:
            self.grad_monitor.register("tail_head", m.bridge.tail_head)

    def tail_semantic_anchor_loss(self, fiber, ids, mask):
        if not (self.c.use_content_semantic_tail and self.c.content_tail_slots > 0):
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        tail = self.m.bridge.tail_head(fiber)
        if tail is None:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        wte = self.m.llm.transformer.wte.weight.detach()
        cc = self.m.content_classifier
        if cc is None:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        tn = F.normalize(tail, dim=-1)
        wn = F.normalize(wte, dim=-1)
        losses = []
        V = wte.shape[0]
        for b in range(tail.shape[0]):
            valid = ids[b][mask[b].bool()].tolist()
            content_tids = [t for t in set(cc.get_content_ids_from_tokens(valid)) if t < V]
            if not content_tids:
                continue
            target = torch.zeros(V, device=tail.device)
            target[content_tids] = 1.0 / len(content_tids)
            slot_logits = tn[b] @ wn.T / 0.3
            log_probs = F.log_softmax(slot_logits, dim=-1)
            kl = F.kl_div(log_probs, target.unsqueeze(0).expand_as(log_probs), reduction="none").sum(-1).mean()
            losses.append(kl)
        if not losses:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        return torch.stack(losses).mean()

    def step(self, texts):
        self.m.train()
        self.opt.zero_grad()
        dev = next(self.m.parameters()).device
        W = self.c.loss_weights
        ids_enc, mask_enc, base, fiber, surp, pooled_mean = self._encode_with_grad(texts)
        l_et = self.encoder_throughput_loss(ids_enc, mask_enc, fiber)
        w_sa = self.warmup.weight("semantic_alignment")
        l_sa = self.semantic_alignment_loss(fiber, ids_enc, mask_enc) * w_sa
        w_tsa = self.warmup.weight("tail_semantic_anchor")
        l_tsa = self.tail_semantic_anchor_loss(fiber, ids_enc, mask_enc) * w_tsa
        all_lr, all_pf, all_fs = [], [], []
        for t in texts:
            lr, pf, fs = self._recon_forward(t)
            all_lr.append(lr)
            all_pf.append(pf)
            all_fs.append(fs if fs is not None else torch.zeros(1, self.c.d_F, device=dev))
        l_r = sum(all_lr) / len(texts)
        pf_batch = torch.cat(all_pf, 0)
        fs_batch = torch.cat(all_fs, 0)
        w_sp = self.warmup.weight("semantic_probe")
        l_sp = self._semantic_probe_loss(pf_batch, fs_batch) * w_sp
        w_va = self.warmup.weight("vocab_anchor")
        l_va = self.vocab_anchor_loss(pf_batch) * w_va
        l_c = self.contrast(texts) if len(texts) >= 2 else torch.tensor(0.0, device=dev)
        with torch.no_grad():
            tk2 = self.m.tok(texts, return_tensors="pt", padding=True, truncation=True)
            ids2, mask2 = tk2["input_ids"].to(dev), tk2["attention_mask"].to(dev)
            o2 = self.m.fwd(ids2, mask2)
        _, xq2, fq2 = self.m.extract_state(o2["hs"], mask2)
        l_h = self.holonomy_proxy(xq2, fq2)
        l_w = self.write_policy_loss(texts)
        w_dd = self.warmup.weight("dir_diversity")
        l_dd = (self.direction_diversity_loss(texts) if len(texts) >= 2 else torch.tensor(0.0, device=dev)) * w_dd
        w_rr = self.warmup.weight("reranker_ranking")
        l_rr = self.reranker_ranking_loss(texts) * w_rr
        loss = (
            W["recon"] * l_r
            + W["semantic_alignment"] * l_sa
            + W["encoder_throughput"] * l_et
            + W["contrast"] * l_c
            + W["holonomy"] * l_h
            + W["write_policy"] * l_w
            + W["semantic_probe"] * l_sp
            + W["dir_diversity"] * l_dd
            + W["reranker_ranking"] * l_rr
            + W["vocab_anchor"] * l_va
            + W.get("tail_semantic_anchor", 0.5) * l_tsa
        )
        loss.backward()
        nn.utils.clip_grad_norm_([p for n, p in self.m.named_parameters() if p.requires_grad and "llm" not in n], 1.0)
        self.opt.step()
        self.warmup.advance()
        self._step_count += 1
        grad_norms = self.grad_monitor.snapshot()
        self.layer_weight_history.append(self.m.layer_pool.weight_dist().cpu().numpy().copy())
        if self._step_count % self.c.refresh_memories_every == 0:
            self.m.eval()
            with torch.no_grad():
                self.m._refresh_all_memories()
            self.m.train()
        self.m.eval()
        return {
            "total": loss.item(),
            "recon": l_r.item(),
            "contrast": l_c.item(),
            "holonomy": l_h.item(),
            "write_policy": l_w.item(),
            "semantic_probe": l_sp.item(),
            "dir_diversity": l_dd.item(),
            "reranker_ranking": l_rr.item(),
            "encoder_throughput": l_et.item(),
            "vocab_anchor": l_va.item(),
            "semantic_alignment": l_sa.item(),
            "tail_semantic_anchor": l_tsa.item(),
            "grad_norms": grad_norms,
            "loss_weights": W,
        }
