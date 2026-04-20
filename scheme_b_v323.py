#!/usr/bin/env python3
"""
Delta module for scheme_b_v3.22.

Implements the v3.22 runtime changes on top of scheme_b_v321 without
changing the external black-box auditor.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, FrozenSet

import torch
import torch.nn.functional as F

import scheme_b_v321 as v321
from scheme_b_v321 import *  # noqa: F401,F403

_dev = v321._dev
_Node = v321._Node


@dataclass
class Cfg(v321.Cfg):
    ret_centroid_weight: float = 0.50
    ret_sem_weight: float = 0.20
    ret_bidi_min_weight: float = 0.15
    ret_forward_maxsim_weight: float = 0.10
    ret_dir_weight: float = 0.05
    use_idf_centroid: bool = True
    use_centroid_dominance: bool = True
    dominance_centroid_margin: float = 1.4
    dominance_centroid_top1_floor: float = 0.25
    use_dominant_hard_prefix: bool = True
    prefix_hard_anchor_scale: float = 1.0
    prefix_hard_pe_scale: float = 1.0
    use_strict_content_starter: bool = True
    strict_starter_min_decoded_len: int = 5


class ContentTokenClassifier(v321.ContentTokenClassifier):
    def __init__(self, tokenizer, min_len=3, strict_min_len=5):
        super().__init__(tokenizer, min_len=min_len)
        self.strict_content_starter_ids: Set[int] = set()
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
        for i in range(min(vocab_size, 50300)):
            try:
                tok_text = tokenizer.decode([i])
                stripped = tok_text.strip().lower()
                cleaned = "".join(c for c in stripped if c.isalpha())
                is_word_starter = len(tok_text) > 0 and tok_text[0] in (" ", "\t")
                if (
                    is_word_starter
                    and i in self.content_starter_ids
                    and stripped == cleaned
                    and len(stripped) >= strict_min_len
                    and stripped not in self.STOPWORDS
                ):
                    self.strict_content_starter_ids.add(i)
            except Exception:
                pass
        self._strict_content_starter_tensor = None

    def strict_content_starter_mask(self, device):
        if (
            self._strict_content_starter_tensor is None
            or self._strict_content_starter_tensor.device != device
        ):
            V = (
                max(
                    max(self.content_ids, default=0),
                    max(self.function_ids, default=0),
                    max(self.punct_ids, default=0),
                    max(self.newline_ids, default=0),
                )
                + 1
            )
            m = torch.zeros(V, device=device)
            for i in self.strict_content_starter_ids:
                if i < V:
                    m[i] = 1.0
            self._strict_content_starter_tensor = m
        return self._strict_content_starter_tensor


class EmbBridge(v321.EmbBridge):
    def inject(
        self,
        fibers,
        mem_mask=None,
        fiber_summary=None,
        content_wte_mean=None,
        content_target_wte=None,
        hard_prefix_wte=None,
    ):
        B = fibers.shape[0]
        if hard_prefix_wte is not None:
            hard_prefix = (
                hard_prefix_wte * self.c.prefix_hard_anchor_scale
                + self.pe.unsqueeze(0) * self.c.prefix_hard_pe_scale
            )
            self._last_fiber_summary = (
                fiber_summary.detach() if fiber_summary is not None else None
            )
            self._last_inject_diag = {
                "hard_prefix_mode": True,
                "hard_prefix_norm": hard_prefix.norm().item(),
                "hard_prefix_per_slot_norm": hard_prefix.norm(dim=-1).mean().item(),
                "bypass_gate": None,
                "qf_norm": 0.0,
                "bypass_norm": 0.0,
                "aligner_scale": torch.sigmoid(self.aligner.scale_logit).item()
                * self.aligner._target_std.item(),
                "cwm_applied": False,
                "target_applied": False,
                "anchor_replace": False,
                "anchor_norm": 0.0,
            }
            return hard_prefix

        return super().inject(
            fibers,
            mem_mask=mem_mask,
            fiber_summary=fiber_summary,
            content_wte_mean=content_wte_mean,
            content_target_wte=content_target_wte,
        )


@dataclass
class RetrievalDiag(v321.RetrievalDiag):
    centroid_applied: bool = False
    top_centroid_cosine: float = 0.0
    per_memory_centroid_cosine: Dict[int, float] = field(default_factory=dict)
    dominance_centroid_margin_observed: float = 0.0
    centroid_dominance_triggered: bool = False


class AMM(v321.AMM):
    @staticmethod
    def _compute_idf_weighted_centroid(token_ids, wte_normed, corpus_idf, idf_floor=0.1):
        if not token_ids or wte_normed is None:
            return None
        V = wte_normed.shape[0]
        valid = [t for t in token_ids if t < V]
        if not valid:
            return None
        if corpus_idf:
            weights = torch.tensor(
                [max(corpus_idf.get(t, idf_floor), idf_floor) for t in valid],
                device=wte_normed.device,
                dtype=wte_normed.dtype,
            )
        else:
            weights = torch.ones(len(valid), device=wte_normed.device, dtype=wte_normed.dtype)
        vecs = wte_normed[valid]
        centroid = (vecs * weights.unsqueeze(1)).sum(0) / weights.sum().clamp(min=1e-8)
        return F.normalize(centroid, dim=-1, eps=1e-8)

    @staticmethod
    def _compute_centroid_cosine(q_centroid, m_centroid):
        if q_centroid is None or m_centroid is None:
            return 0.0
        return (q_centroid @ m_centroid).item()

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

            C = len(mems)
            sb = torch.stack([m.base.to(dev) for m in mems])
            sf = torch.stack([m.fiber.to(dev) for m in mems])
            md = torch.stack([m.dirn.to(dev) for m in mems])
            raw_dir_sim = torch.einsum("d,cd->c", qdir[b], md)
            diag.top_dir_sim = raw_dir_sim.max().item()

            sem_sims = []
            if query_semantic_emb is not None:
                for mem in mems:
                    if mem.semantic_emb is not None:
                        s = F.cosine_similarity(
                            query_semantic_emb[b : b + 1],
                            mem.semantic_emb.unsqueeze(0).to(dev),
                            dim=-1,
                        ).squeeze()
                        sem_sims.append(s)
                    else:
                        sem_sims.append(raw_dir_sim.new_tensor(0.0))
                sem_sim_t = torch.stack(sem_sims)
                diag.top_sem_sim = sem_sim_t.max().item()
            else:
                sem_sim_t = torch.zeros(C, device=dev)

            q_content_ids = (
                query_content_ids_per_batch[b]
                if query_content_ids_per_batch and b < len(query_content_ids_per_batch)
                else []
            )

            centroid_scores = torch.zeros(C, device=dev)
            if self.c.use_idf_centroid and q_content_ids and wn is not None:
                q_centroid = self._compute_idf_weighted_centroid(q_content_ids, wn, corpus_idf, idf_floor)
                if q_centroid is not None:
                    for mi, mem in enumerate(mems):
                        m_scoring_ids = self._get_mem_scoring_ids(mem)
                        m_centroid = self._compute_idf_weighted_centroid(
                            m_scoring_ids, wn, corpus_idf, idf_floor
                        )
                        centroid_scores[mi] = self._compute_centroid_cosine(q_centroid, m_centroid)
                diag.top_centroid_cosine = centroid_scores.max().item() if C > 0 else 0.0

            if q_content_ids and wn is not None:
                forward_scores = []
                backward_scores = []
                for mem in mems:
                    scoring_ids = self._get_mem_scoring_ids(mem)
                    fwd_idf = self._compute_forward_maxsim(
                        q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor
                    )
                    bwd_idf = self._compute_backward_maxsim(
                        q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor
                    )
                    forward_scores.append(fwd_idf)
                    backward_scores.append(bwd_idf)
                forward_t = torch.tensor(forward_scores, device=dev)
                backward_t = torch.tensor(backward_scores, device=dev)
                bidi_min_t = torch.minimum(forward_t, backward_t)
                forward_idf_t = forward_t.clone()
                diag.top_forward_maxsim = forward_t.max().item()
                diag.top_backward_maxsim = backward_t.max().item()
                diag.top_bidi_min = bidi_min_t.max().item()
                diag.top_forward_maxsim_idf = forward_idf_t.max().item()
                diag.top_bidi_min_idf = bidi_min_t.max().item()
            else:
                forward_t = torch.zeros(C, device=dev)
                backward_t = torch.zeros(C, device=dev)
                bidi_min_t = torch.zeros(C, device=dev)
                forward_idf_t = torch.zeros(C, device=dev)

            combined_sim = (
                self.c.ret_centroid_weight * centroid_scores
                + self.c.ret_sem_weight * sem_sim_t
                + self.c.ret_bidi_min_weight * bidi_min_t
                + self.c.ret_forward_maxsim_weight * forward_t
                + self.c.ret_dir_weight * raw_dir_sim
            )

            top_sem = sem_sim_t.max().item() if C > 0 else 0.0
            top_bidi = bidi_min_t.max().item() if C > 0 else 0.0
            sem_thresh = max(self.c.gate_sem_floor, top_sem * self.c.gate_sem_ratio)
            bidi_thresh = max(
                self.c.gate_bidi_floor,
                top_bidi * self.c.gate_bidi_ratio,
                self.c.gate_bidi_hard_min,
            )
            hard_mask = (sem_sim_t >= sem_thresh) & (bidi_min_t >= bidi_thresh)
            gate_affinity = self.c.gate_sem_weight * sem_sim_t + self.c.gate_bidi_weight * bidi_min_t
            diag.top_gate_affinity = gate_affinity.max().item() if C > 0 else 0.0
            diag.gate_threshold = max(sem_thresh, bidi_thresh)
            diag.n_gate_pass = int(hard_mask.sum().item())
            if hard_mask.sum().item() == 0:
                hard_mask[torch.minimum(sem_sim_t, bidi_min_t).argmax()] = True
            diag.n_after_hard_filter = int(hard_mask.sum().item())
            for mi, mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid] = gate_affinity[mi].item()

            keep_indices = hard_mask.nonzero(as_tuple=True)[0]
            if 0 < keep_indices.numel() < C:
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
                C = len(mems)

            rerank_scores = self.reranker(
                xq[b : b + 1], fq[b : b + 1], sb.unsqueeze(0), sf.unsqueeze(0), combined_sim.unsqueeze(0)
            ).squeeze(0)
            diag.reranker_delta_mean = (rerank_scores - combined_sim).abs().mean().item()
            diag.top_reranker_score = rerank_scores.max().item()

            if C > 1:
                top_score = rerank_scores.max()
                score_mask = rerank_scores >= top_score * self.c.score_keep_ratio
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
                    C = len(mems)
            else:
                diag.n_after_score_filter = C

            if C > 1 and forward_t.max().item() > 0:
                coherence_mask = forward_t >= forward_t.max() * self.c.fwd_coherence_ratio
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
                        C = len(mems)
                else:
                    diag.n_after_coherence_filter = C
            else:
                diag.n_after_coherence_filter = C

            if C > 1 and bidi_min_t.max().item() > 0:
                gap_mask = bidi_min_t >= (bidi_min_t.max().item() - self.c.bidi_absolute_gap)
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
                        C = len(mems)
                else:
                    diag.n_after_bidi_gap_filter = C
            else:
                diag.n_after_bidi_gap_filter = C

            dominant_mid = None
            if self.c.use_centroid_dominance and C >= 2 and centroid_scores.max().item() > 0:
                c_sorted, c_idx = torch.sort(centroid_scores, descending=True)
                top1_c = c_sorted[0].item()
                top2_c = c_sorted[1].item()
                cent_margin = top1_c / max(top2_c, 1e-6) if top2_c > 0 else float("inf")
                diag.dominance_centroid_margin_observed = cent_margin
                if (
                    top1_c >= self.c.dominance_centroid_top1_floor
                    and cent_margin >= self.c.dominance_centroid_margin
                ):
                    diag.dominance_triggered = True
                    diag.centroid_dominance_triggered = True
                    top1_idx = c_idx[0].item()
                    dominant_mid = mems[top1_idx].mid
                    keep_thresh = top1_c / self.c.dominance_centroid_margin
                    keep_mask = centroid_scores >= keep_thresh
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
                        C = len(mems)

            if self.c.use_idf_dominance and C >= 2 and forward_idf_t.max().item() > 0:
                fwd_sorted, fwd_sort_idx = torch.sort(forward_idf_t, descending=True)
                top1_fwd = fwd_sorted[0].item()
                top2_fwd = fwd_sorted[1].item()
                idf_margin = top1_fwd / max(top2_fwd, 1e-6)
                diag.dominance_idf_margin_observed = idf_margin
                if (
                    top1_fwd >= self.c.dominance_idf_top1_floor
                    and idf_margin >= self.c.dominance_idf_margin
                ):
                    diag.dominance_triggered = True
                    if dominant_mid is None:
                        dominant_mid = mems[fwd_sort_idx[0].item()].mid
                    keep_thresh = top1_fwd / self.c.dominance_idf_margin
                    keep_mask = forward_idf_t >= keep_thresh
                    keep_mask[fwd_sort_idx[0].item()] = True
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
                        C = len(mems)

            if self.c.use_dominance_filter and C >= 2 and content_classifier is not None:
                dominance_scores = forward_idf_t if forward_idf_t.max().item() > 0 else rerank_scores
                sorted_idx = torch.argsort(dominance_scores, descending=True)
                top1_local = sorted_idx[0].item()
                top2_local = sorted_idx[1].item()
                top1_score = dominance_scores[top1_local].item()
                top2_score = dominance_scores[top2_local].item()
                margin = top1_score / max(abs(top2_score), 1e-6) if top2_score > 0 else float("inf")
                diag.dominance_margin_observed = margin
                top1_sem = sem_sim_t[top1_local].item()
                top1_mem = mems[top1_local]
                top1_label = self._mem_label_set(top1_mem, content_classifier)
                if (
                    len(top1_label) >= self.c.dominance_min_label_size
                    and top1_sem >= self.c.dominance_sem_floor
                    and margin >= self.c.dominance_margin
                ):
                    diag.dominance_triggered = True
                    if dominant_mid is None:
                        dominant_mid = top1_mem.mid
                    keep_local = []
                    for i, mem in enumerate(mems):
                        if i == top1_local:
                            keep_local.append(i)
                            continue
                        mem_label = self._mem_label_set(mem, content_classifier)
                        if self._jaccard(top1_label, mem_label) >= self.c.dominance_jaccard_threshold:
                            keep_local.append(i)
                    if len(keep_local) < C:
                        kt = torch.tensor(keep_local, device=dev, dtype=torch.long)
                        mems = [mems[i] for i in keep_local]
                        sb = sb[kt]
                        sf = sf[kt]
                        rerank_scores = rerank_scores[kt]
                        forward_t = forward_t[kt]
                        bidi_min_t = bidi_min_t[kt]
                        sem_sim_t = sem_sim_t[kt]
                        forward_idf_t = forward_idf_t[kt]
                        centroid_scores = centroid_scores[kt]
                        C = len(mems)
            diag.n_after_dominance_filter = C

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


class MemLLM(v321.MemLLM):
    def __init__(self, c):
        super().__init__(c)
        self.amm = AMM(c)
        self.bridge = EmbBridge(c)

    def load(self, name="gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tok = GPT2Tokenizer.from_pretrained(name)
        self.llm = GPT2LMHeadModel.from_pretrained(name)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layer_pool = AdaptiveLayerPool(self.llm.config.n_layer + 1, self.c.d_LLM)
        self.content_classifier = ContentTokenClassifier(
            self.tok,
            self.c.content_min_len,
            strict_min_len=self.c.strict_starter_min_decoded_len,
        )
        self._degen_guard = DegenerationGuard(self.tok, self.c, self.content_classifier)
        self.bridge.aligner.calibrate(self.llm)
        self.c.vocab_size = self.llm.config.vocab_size
        self._wte_normed = F.normalize(self.llm.transformer.wte.weight.detach(), dim=-1, eps=1e-8)
        self.amm.wte_normed = self._wte_normed
        self._build_wte_neighbor_cache()

    def _compute_tfidf_idf(self) -> Dict[int, float]:
        if self.content_classifier is None:
            return {}
        return self.amm._compute_corpus_idf(self.content_classifier)

    def _compute_content_wte_topk(self, diag, query_content_ids_per_batch):
        dev = next(self.parameters()).device
        wte = self.llm.transformer.wte.weight.detach()
        wte_n = self._wte_normed
        cc = self.content_classifier
        floor = self.c.content_bias_relevance_floor
        concentration = self.c.content_bias_concentration
        use_strict = self.c.use_strict_content_starter
        use_starter = self.c.use_word_starter_filter
        K = self.c.content_wte_topk_for_inject
        B = len(diag.batch_mem_weights)
        idf = self._compute_tfidf_idf() if self.c.use_tfidf_weighting else {}
        mean_list = []
        target_list = []

        for b in range(B):
            q_ids = (
                query_content_ids_per_batch[b]
                if query_content_ids_per_batch and b < len(query_content_ids_per_batch)
                else []
            )
            q_valid = [i for i in q_ids if i < wte_n.shape[0]]
            dom_mid = (
                diag.dominant_per_batch[b]
                if diag.dominant_per_batch and b < len(diag.dominant_per_batch)
                else None
            )
            weight_map: Dict[int, float] = {}
            if dom_mid is not None and dom_mid in self.amm.tree.store:
                mem = self.amm.tree.store[dom_mid]
                scoring_ids = self.amm._get_mem_scoring_ids(mem)
                strict_set = (
                    cc.strict_content_starter_ids
                    if use_strict and cc is not None
                    else (cc.content_starter_ids if cc is not None else set())
                )
                for tid in scoring_ids:
                    if tid >= wte.shape[0] or cc is None:
                        continue
                    if use_strict and tid not in strict_set:
                        continue
                    if (not use_strict) and use_starter and tid not in cc.content_starter_ids:
                        continue
                    if (not use_strict) and (not use_starter) and tid not in cc.content_ids:
                        continue
                    weight_map[tid] = weight_map.get(tid, 0.0) + 1.0
            elif b < len(diag.batch_mem_weights):
                for mid, w in diag.batch_mem_weights[b]:
                    if mid not in self.amm.tree.store:
                        continue
                    mem = self.amm.tree.store[mid]
                    bidi_w = diag.per_memory_bidi_min.get(mid, 0.5)
                    adjusted_w = w * (bidi_w ** 2)
                    scoring_ids = self.amm._get_mem_scoring_ids(mem)
                    for tid in scoring_ids:
                        if tid >= wte.shape[0] or cc is None:
                            continue
                        if use_starter and tid not in cc.content_starter_ids:
                            continue
                        if (not use_starter) and tid not in cc.content_ids:
                            continue
                        weight_map[tid] = weight_map.get(tid, 0.0) + adjusted_w

            if not weight_map:
                zero = torch.zeros(self.c.d_LLM, device=dev)
                mean_list.append(zero)
                target_list.append(zero.clone())
                continue

            tids = list(weight_map.keys())
            tids_t = torch.tensor(tids, device=dev)
            base_weights = torch.tensor([weight_map[t] for t in tids], device=dev)
            idf_weights = torch.tensor([idf.get(t, 1.0) for t in tids], device=dev)
            if q_valid:
                q_centroid = self.amm._compute_idf_weighted_centroid(q_valid, wte_n, idf, self.c.idf_floor)
                if q_centroid is not None:
                    m_vecs_n = wte_n[tids_t]
                    relevance = (m_vecs_n @ q_centroid).clamp(min=0)
                    relevance = relevance.pow(concentration)
                    relevance = relevance * (1.0 - floor) + floor
                    final_weights = base_weights * relevance * idf_weights
                else:
                    final_weights = base_weights * idf_weights
            else:
                final_weights = base_weights * idf_weights

            K_eff = min(K, len(tids))
            topk_vals, topk_idx = final_weights.topk(K_eff)
            topk_tids = tids_t[topk_idx]
            topk_wte = wte[topk_tids]
            total = topk_vals.sum()
            mean_vec = (topk_wte * topk_vals.unsqueeze(1)).sum(0) / total if total > 1e-8 else topk_wte.mean(0)
            mean_list.append(mean_vec)
            target_list.append(wte[tids_t[final_weights.argmax()]])

        return torch.stack(mean_list), torch.stack(target_list)

    def _build_dominant_hard_prefix_wte(self, diag, query_content_ids_per_batch):
        if not self.c.use_dominant_hard_prefix:
            return None, None
        dev = next(self.parameters()).device
        wte = self.llm.transformer.wte.weight.detach()
        wte_n = self._wte_normed
        cc = self.content_classifier
        if cc is None:
            return None, None
        idf = self._compute_tfidf_idf() if self.c.use_tfidf_weighting else {}
        L = self.c.L_mem
        D = self.c.d_LLM
        B = len(diag.batch_mem_weights) if diag.batch_mem_weights else 0
        if B == 0:
            return None, None
        hard_wte = torch.zeros(B, L, D, device=dev)
        triggered_mask = [False] * B
        strict_set = cc.strict_content_starter_ids if self.c.use_strict_content_starter else cc.content_starter_ids

        for b in range(B):
            dom_mid = diag.dominant_per_batch[b] if b < len(diag.dominant_per_batch) else None
            if dom_mid is None or dom_mid not in self.amm.tree.store:
                continue
            mem = self.amm.tree.store[dom_mid]
            valid_ids = [tid for tid in self.amm._get_mem_scoring_ids(mem) if tid < wte.shape[0] and tid in strict_set]
            if not valid_ids:
                continue

            idf_vals = torch.tensor([idf.get(t, 1.0) for t in valid_ids], device=dev)
            q_ids = query_content_ids_per_batch[b] if b < len(query_content_ids_per_batch) else []
            q_valid = [i for i in q_ids if i < wte_n.shape[0]]
            if q_valid:
                q_centroid = self.amm._compute_idf_weighted_centroid(q_valid, wte_n, idf, self.c.idf_floor)
                if q_centroid is not None:
                    v_tensor = torch.tensor(valid_ids, device=dev)
                    rel = (wte_n[v_tensor] @ q_centroid).clamp(min=0)
                    scores = idf_vals * (rel + self.c.content_bias_relevance_floor)
                else:
                    scores = idf_vals
            else:
                scores = idf_vals

            K = min(L, len(valid_ids))
            _, top_idx = scores.topk(K)
            top_tids = [valid_ids[i.item()] for i in top_idx]
            for si in range(K):
                hard_wte[b, si] = wte[top_tids[si]]
            if K < L:
                top_vals = scores[top_idx]
                mean_w = top_vals / top_vals.sum().clamp(min=1e-8)
                mean_vec = torch.zeros(D, device=dev)
                for i in range(K):
                    mean_vec = mean_vec + wte[top_tids[i]] * mean_w[i].item()
                for si in range(K, L):
                    hard_wte[b, si] = mean_vec
            triggered_mask[b] = True

        if not any(triggered_mask):
            return None, None
        return hard_wte, triggered_mask

    def _get_prefix(self, hs, mask=None, pl=0, update_stats=True, return_extra=False, ids=None):
        pooled, xq, fq = self.extract_state(hs, mask, pl)
        trimmed_mask = mask[:, pl:] if mask is not None and pl > 0 else mask
        if trimmed_mask is not None and pooled.shape[1] != trimmed_mask.shape[1]:
            trimmed_mask = None
        query_content_ids_per_batch = []
        if ids is not None and self.content_classifier is not None:
            for b in range(ids.shape[0]):
                b_ids = ids[b].tolist()
                query_content_ids_per_batch.append(list(set(self.content_classifier.get_content_ids_from_tokens(b_ids))))
        query_sem = self._compute_content_semantic_emb(pooled, ids, trimmed_mask) if ids is not None and self.content_classifier is not None else pooled.mean(1)
        fibers, mem_mask, fiber_summary, diag = self.amm.retrieve_multi(
            xq,
            fq,
            update_stats=update_stats,
            query_semantic_emb=query_sem,
            query_content_ids_per_batch=query_content_ids_per_batch,
            wte_normed=self._wte_normed,
            content_classifier=self.content_classifier,
        )

        hard_wte, hard_mask = self._build_dominant_hard_prefix_wte(diag, query_content_ids_per_batch)
        all_triggered = hard_mask is not None and all(hard_mask)
        if all_triggered:
            prefix = self.bridge.inject(
                fibers,
                mem_mask,
                fiber_summary=fiber_summary,
                hard_prefix_wte=hard_wte,
            )
        else:
            content_wte_mean, content_target_wte = self._compute_content_wte_topk(diag, query_content_ids_per_batch)
            has_cwm = content_wte_mean.abs().max().item() > 1e-6
            has_tgt = content_target_wte.abs().max().item() > 1e-6
            prefix = self.bridge.inject(
                fibers,
                mem_mask,
                fiber_summary=fiber_summary,
                content_wte_mean=content_wte_mean if has_cwm else None,
                content_target_wte=content_target_wte if has_tgt else None,
            )

        if return_extra:
            content_bias = self._build_content_bias(diag, query_content_ids_per_batch)
            first_step_bias = self._build_first_step_lexical_bias(diag, query_content_ids_per_batch)
            return prefix, fiber_summary, diag, content_bias, first_step_bias
        return prefix

    def generate(self, prompt, mt=50, greedy=False):
        tk = self.tok(prompt, return_tensors="pt")
        dev = next(self.parameters()).device
        ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            prefix, fiber_summary, _, content_bias, first_step_bias = self._get_prefix(
                o["hs"], mask, update_stats=True, return_extra=True, ids=ids
            )
            vocab_bias = self._compute_vocab_bias(fiber_summary)
        has_content = content_bias is not None and content_bias.abs().max().item() > 0.01
        has_first_step = first_step_bias is not None and first_step_bias.abs().max().item() > 1e-6
        cc = self.content_classifier
        domain_anchors = self._compute_domain_anchors(content_bias) if has_content else [[]]
        anchors_for_b0 = set(domain_anchors[0]) if domain_anchors else set()
        generated_anchors = set()
        generated_ids = []
        generated_content_counts: Dict[int, int] = {}
        consecutive_content = 0
        recent_starters: List[Tuple[int, int]] = []

        for i in range(mt):
            if i > 0 and i % self.c.retrieval_interval == 0:
                with torch.no_grad():
                    o = self.fwd(ids, mask, prefix)
                    pl = o["pl"]
                    prefix, fiber_summary, _, content_bias, first_step_bias = self._get_prefix(
                        o["hs"], o["mask"], pl, update_stats=True, return_extra=True, ids=ids
                    )
                    vocab_bias = self._compute_vocab_bias(fiber_summary)
                    has_content = content_bias is not None and content_bias.abs().max().item() > 0.01
                    if has_content:
                        domain_anchors = self._compute_domain_anchors(content_bias)
                        anchors_for_b0 = set(domain_anchors[0]) if domain_anchors else set()

            with torch.no_grad():
                o = self.fwd(ids, mask, prefix)
                lg = o["logits"][:, -1:].squeeze(1).clone()
                step_scale_content = max(self.c.content_bias_floor, 1.0 - i * self.c.content_bias_decay)
                step_scale_learned = max(self.c.semantic_boost_floor, 1.0 - i * self.c.semantic_boost_decay)
                if i == 0:
                    effective_content_scale = step_scale_content * self.c.first_step_content_multiplier
                elif consecutive_content >= self.c.structural_rhythm_threshold:
                    effective_content_scale = step_scale_content * 0.25
                    if cc:
                        for fid in list(cc.function_ids)[:5000]:
                            if fid < lg.shape[-1]:
                                lg[0, fid] += self.c.structural_boost
                else:
                    effective_content_scale = step_scale_content

                if has_first_step and i < self.c.first_step_lexical_decay_steps:
                    V_fs = min(lg.shape[-1], first_step_bias.shape[-1])
                    lg[:, :V_fs] = lg[:, :V_fs] + first_step_bias[:, :V_fs] * self.c.first_step_lexical_scale
                if has_content:
                    cb_adjusted = content_bias.clone()
                    for tid, count in generated_content_counts.items():
                        if tid < cb_adjusted.shape[-1]:
                            cb_adjusted[0, tid] *= self.c.generated_token_decay ** count
                    V = min(lg.shape[-1], cb_adjusted.shape[-1])
                    lg[:, :V] = lg[:, :V] + cb_adjusted[:, :V] * self.c.content_bias_scale * effective_content_scale
                if vocab_bias is not None:
                    V2 = min(lg.shape[-1], vocab_bias.shape[-1])
                    lg[:, :V2] = lg[:, :V2] + vocab_bias[:, :V2] * self.c.semantic_boost_scale * step_scale_learned

                if i == 0 and cc is not None:
                    if self.c.use_strict_content_starter:
                        cmask = cc.strict_content_starter_mask(dev)
                    elif self.c.use_word_starter_filter:
                        cmask = cc.content_starter_mask(dev)
                    else:
                        cmask = cc.content_mask(dev)
                    V3 = min(lg.shape[-1], cmask.shape[0])
                    lg[0, :V3] = lg[0, :V3] + cmask[:V3] * self.c.universal_content_boost
                elif i < self.c.universal_content_boost_steps and cc is not None and has_content:
                    cmask = cc.content_starter_mask(dev) if self.c.use_word_starter_filter else cc.content_mask(dev)
                    V3 = min(lg.shape[-1], cmask.shape[0])
                    boost_scale = 1.0 - i / self.c.universal_content_boost_steps
                    lg[0, :V3] = lg[0, :V3] + cmask[:V3] * self.c.universal_content_boost * boost_scale

                if i >= self.c.domain_anchor_start_step and anchors_for_b0 and has_content:
                    coverage = len(generated_anchors) / max(len(anchors_for_b0), 1)
                    if coverage < self.c.domain_anchor_coverage_threshold:
                        for tid in anchors_for_b0 - generated_anchors:
                            if tid < lg.shape[-1]:
                                lg[0, tid] += self.c.domain_anchor_boost

                if cc:
                    for tid, count in generated_content_counts.items():
                        if tid in cc.content_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.content_repeat_penalty * count
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
                if self._degen_guard is not None:
                    lg = self._degen_guard.process(
                        lg,
                        generated_ids,
                        i,
                        first_step_penalty_mult=self.c.first_step_penalty_multiplier if i == 0 else 1.0,
                    )
                if i < self.c.early_content_steps and cc is not None:
                    for pid in cc.punct_ids:
                        if pid < lg.shape[-1]:
                            lg[0, pid] = -float("inf")
                    for nid in cc.newline_ids:
                        if nid < lg.shape[-1]:
                            lg[0, nid] = -float("inf")
                if i == 0 and cc is not None:
                    for fid in cc.filler_ids:
                        if fid < lg.shape[-1]:
                            lg[0, fid] -= self.c.step0_filler_penalty

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
                if nxt_id in anchors_for_b0:
                    generated_anchors.add(nxt_id)
                if nxt_id in cc.word_starter_ids:
                    recent_starters.append((nxt_id, i))
            else:
                consecutive_content = 0
            recent_starters = [(t, s) for (t, s) in recent_starters if (i - s) < self.c.bpe_echo_window]
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)

        return self.tok.decode(ids[0], skip_special_tokens=True)


import scheme_b_v322 as v322

_dev = v322._dev
_Node = v322._Node


@dataclass
class Cfg(v322.Cfg):
    use_triple_consensus_dominance: bool = True
    consensus_fwd_rank_max: int = 2
    consensus_label_size_min: int = 3
    consensus_strict_keep_ratio: float = 0.85
    hard_prefix_last_slots: int = 2
    use_post_inject_suppress: bool = True
    post_inject_suppress_steps: int = 5
    post_inject_suppress_penalty: float = 8.0
    use_strict_or_continuation: bool = True
    strict_or_cont_penalty: float = 4.0
    strict_or_cont_steps: int = 8

    def __post_init__(self):
        super().__post_init__()
        assert self.hard_prefix_last_slots >= 1
        assert self.hard_prefix_last_slots < self.L_mem


class ContentTokenClassifier(v322.ContentTokenClassifier):
    def __init__(self, tokenizer, min_len=3, strict_min_len=5):
        super().__init__(tokenizer, min_len=min_len, strict_min_len=strict_min_len)
        self._non_strict_content_tensor = None

    def non_strict_content_mask(self, device):
        if (
            self._non_strict_content_tensor is None
            or self._non_strict_content_tensor.device != device
        ):
            cm = self.content_mask(device)
            sm = self.strict_content_starter_mask(device)
            V = min(cm.shape[0], sm.shape[0])
            m = torch.zeros(cm.shape[0], device=device)
            m[:V] = cm[:V] * (1.0 - sm[:V])
            self._non_strict_content_tensor = m
        return self._non_strict_content_tensor


class EmbBridge(v322.EmbBridge):
    def inject(
        self,
        fibers,
        mem_mask=None,
        fiber_summary=None,
        content_wte_mean=None,
        content_target_wte=None,
        hard_wte_last_slots=None,
    ):
        B = fibers.shape[0]
        if self.inject_mode in ("both", "qformer_only"):
            qf_out = self.proj(fibers, mem_mask) + self.pe.unsqueeze(0)
        else:
            qf_out = self.pe.unsqueeze(0).expand(B, -1, -1)

        bp_out = None
        gate_val = None
        if fiber_summary is not None and self.inject_mode in ("both", "bypass_only"):
            qf_context = qf_out.mean(1)
            bp_out = self.bypass(fiber_summary, qf_context)
            gate_val = self.bypass._last_gate
            qf_out = qf_out + bp_out.unsqueeze(1)
        qf_out = self.aligner(qf_out)
        L = qf_out.shape[1]

        hard_last_n = 0
        if hard_wte_last_slots is not None:
            hard_last_n = hard_wte_last_slots.shape[1]
            assert 1 <= hard_last_n < L

        anchor_replace = (
            self.c.prefix_anchor_replace
            and content_target_wte is not None
            and content_target_wte.abs().max().item() > 1e-6
            and hard_last_n == 0
        )

        cwm_applied = False
        if content_wte_mean is not None:
            cwm = content_wte_mean
            if cwm.dim() == 2:
                cwm = cwm.unsqueeze(1)
            n_last = max(1, int(L * self.prefix_inject_last_ratio))
            pos_scale = torch.ones(L, device=qf_out.device)
            pos_scale[: L - n_last] = self.prefix_inject_other_multiplier
            pos_scale[L - n_last :] = self.prefix_inject_last_multiplier
            if hard_last_n > 0:
                pos_scale[L - hard_last_n :] = 0.0
            elif anchor_replace:
                pos_scale[-1] = 0.0
            pos_scale = pos_scale.view(1, -1, 1)
            qf_out = qf_out + cwm * self.content_inject_scale * pos_scale
            cwm_applied = True

        tgt_applied = False
        anchor_norm_val = 0.0
        hybrid_hard_applied = False

        if hard_last_n > 0:
            hard_block = (
                hard_wte_last_slots * self.c.prefix_hard_anchor_scale
                + self.pe[L - hard_last_n :].unsqueeze(0) * self.c.prefix_hard_pe_scale
            )
            qf_out = torch.cat([qf_out[:, : L - hard_last_n], hard_block], dim=1)
            hybrid_hard_applied = True
            tgt_applied = True
            anchor_norm_val = hard_block.norm(dim=-1).mean().item()
        elif anchor_replace:
            ctw = content_target_wte
            anchor_slot = ctw * self.c.prefix_anchor_scale
            if self.c.prefix_anchor_use_pe:
                anchor_slot = anchor_slot + self.pe[-1].unsqueeze(0)
            qf_out = torch.cat([qf_out[:, :-1, :], anchor_slot.unsqueeze(1)], dim=1)
            tgt_applied = True
            anchor_norm_val = anchor_slot.norm(dim=-1).mean().item()
        elif content_target_wte is not None:
            ctw = content_target_wte
            if ctw.dim() == 2:
                ctw = ctw.unsqueeze(1)
            tgt_scale = torch.zeros(L, device=qf_out.device)
            tgt_scale[-1] = self.prefix_target_multiplier
            qf_out = qf_out + ctw * tgt_scale.view(1, -1, 1)
            tgt_applied = True

        self._last_fiber_summary = fiber_summary.detach() if fiber_summary is not None else None
        self._last_inject_diag = {
            "hybrid_hard_applied": hybrid_hard_applied,
            "hard_last_n": hard_last_n,
            "bypass_gate": gate_val.mean().item() if gate_val is not None else None,
            "qf_norm": qf_out.norm().item(),
            "bypass_norm": bp_out.norm().item() if bp_out is not None else 0.0,
            "aligner_scale": torch.sigmoid(self.aligner.scale_logit).item()
            * self.aligner._target_std.item(),
            "cwm_applied": cwm_applied,
            "target_applied": tgt_applied,
            "anchor_replace": anchor_replace,
            "anchor_norm": anchor_norm_val,
            "last_slot_norm_per_b": qf_out[:, -1].norm(dim=-1).mean().item(),
            "second_last_slot_norm_per_b": (
                qf_out[:, -2].norm(dim=-1).mean().item() if L >= 2 else 0.0
            ),
        }
        return qf_out


@dataclass
class RetrievalDiag(v322.RetrievalDiag):
    consensus_fwd_rank: int = -1
    consensus_label_size: int = 0
    consensus_passed: bool = False


class AMM(v322.AMM):
    @staticmethod
    def _mem_strict_label_set(mem, content_classifier) -> FrozenSet[int]:
        if content_classifier is None:
            return frozenset(mem.content_token_ids)
        return frozenset(
            t for t in mem.content_token_ids if t in content_classifier.strict_content_starter_ids
        )

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

            C = len(mems)
            sb = torch.stack([m.base.to(dev) for m in mems])
            sf = torch.stack([m.fiber.to(dev) for m in mems])
            md = torch.stack([m.dirn.to(dev) for m in mems])
            raw_dir_sim = torch.einsum("d,cd->c", qdir[b], md)
            diag.top_dir_sim = raw_dir_sim.max().item()

            sem_sims = []
            if query_semantic_emb is not None:
                for mem in mems:
                    if mem.semantic_emb is not None:
                        s = F.cosine_similarity(
                            query_semantic_emb[b : b + 1],
                            mem.semantic_emb.unsqueeze(0).to(dev),
                            dim=-1,
                        ).squeeze()
                        sem_sims.append(s)
                    else:
                        sem_sims.append(raw_dir_sim.new_tensor(0.0))
                sem_sim_t = torch.stack(sem_sims)
                diag.top_sem_sim = sem_sim_t.max().item()
            else:
                sem_sim_t = torch.zeros(C, device=dev)

            q_content_ids = (
                query_content_ids_per_batch[b]
                if query_content_ids_per_batch and b < len(query_content_ids_per_batch)
                else []
            )

            centroid_scores = torch.zeros(C, device=dev)
            if self.c.use_idf_centroid and q_content_ids and wn is not None:
                q_centroid = self._compute_idf_weighted_centroid(q_content_ids, wn, corpus_idf, idf_floor)
                if q_centroid is not None:
                    for mi, mem in enumerate(mems):
                        m_scoring_ids = self._get_mem_scoring_ids(mem)
                        m_centroid = self._compute_idf_weighted_centroid(
                            m_scoring_ids, wn, corpus_idf, idf_floor
                        )
                        centroid_scores[mi] = self._compute_centroid_cosine(q_centroid, m_centroid)
                diag.top_centroid_cosine = centroid_scores.max().item() if C > 0 else 0.0

            if q_content_ids and wn is not None:
                forward_scores = []
                backward_scores = []
                for mem in mems:
                    scoring_ids = self._get_mem_scoring_ids(mem)
                    fwd_idf = self._compute_forward_maxsim(
                        q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor
                    )
                    bwd_idf = self._compute_backward_maxsim(
                        q_content_ids, scoring_ids, wn, query_idf=corpus_idf, idf_floor=idf_floor
                    )
                    forward_scores.append(fwd_idf)
                    backward_scores.append(bwd_idf)
                forward_t = torch.tensor(forward_scores, device=dev)
                backward_t = torch.tensor(backward_scores, device=dev)
                bidi_min_t = torch.minimum(forward_t, backward_t)
                forward_idf_t = forward_t.clone()
                diag.top_forward_maxsim = forward_t.max().item()
                diag.top_backward_maxsim = backward_t.max().item()
                diag.top_bidi_min = bidi_min_t.max().item()
                diag.top_forward_maxsim_idf = forward_idf_t.max().item()
                diag.top_bidi_min_idf = bidi_min_t.max().item()
            else:
                forward_t = torch.zeros(C, device=dev)
                backward_t = torch.zeros(C, device=dev)
                bidi_min_t = torch.zeros(C, device=dev)
                forward_idf_t = torch.zeros(C, device=dev)

            combined_sim = (
                self.c.ret_centroid_weight * centroid_scores
                + self.c.ret_sem_weight * sem_sim_t
                + self.c.ret_bidi_min_weight * bidi_min_t
                + self.c.ret_forward_maxsim_weight * forward_t
                + self.c.ret_dir_weight * raw_dir_sim
            )

            top_sem = sem_sim_t.max().item() if C > 0 else 0.0
            top_bidi = bidi_min_t.max().item() if C > 0 else 0.0
            sem_thresh = max(self.c.gate_sem_floor, top_sem * self.c.gate_sem_ratio)
            bidi_thresh = max(
                self.c.gate_bidi_floor,
                top_bidi * self.c.gate_bidi_ratio,
                self.c.gate_bidi_hard_min,
            )
            hard_mask = (sem_sim_t >= sem_thresh) & (bidi_min_t >= bidi_thresh)
            gate_affinity = self.c.gate_sem_weight * sem_sim_t + self.c.gate_bidi_weight * bidi_min_t
            diag.top_gate_affinity = gate_affinity.max().item() if C > 0 else 0.0
            diag.gate_threshold = max(sem_thresh, bidi_thresh)
            diag.n_gate_pass = int(hard_mask.sum().item())
            if hard_mask.sum().item() == 0:
                hard_mask[torch.minimum(sem_sim_t, bidi_min_t).argmax()] = True
            diag.n_after_hard_filter = int(hard_mask.sum().item())
            for mi, mem in enumerate(mems):
                diag.per_memory_gate_affinity[mem.mid] = gate_affinity[mi].item()

            keep_indices = hard_mask.nonzero(as_tuple=True)[0]
            if 0 < keep_indices.numel() < C:
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
                C = len(mems)

            rerank_scores = self.reranker(
                xq[b : b + 1], fq[b : b + 1], sb.unsqueeze(0), sf.unsqueeze(0), combined_sim.unsqueeze(0)
            ).squeeze(0)
            diag.reranker_delta_mean = (rerank_scores - combined_sim).abs().mean().item()
            diag.top_reranker_score = rerank_scores.max().item()

            if C > 1:
                top_score = rerank_scores.max()
                score_mask = rerank_scores >= top_score * self.c.score_keep_ratio
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
                    C = len(mems)
            else:
                diag.n_after_score_filter = C

            if C > 1 and forward_t.max().item() > 0:
                coherence_mask = forward_t >= forward_t.max() * self.c.fwd_coherence_ratio
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
                        C = len(mems)
                else:
                    diag.n_after_coherence_filter = C
            else:
                diag.n_after_coherence_filter = C

            if C > 1 and bidi_min_t.max().item() > 0:
                gap_mask = bidi_min_t >= (bidi_min_t.max().item() - self.c.bidi_absolute_gap)
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
                        C = len(mems)
                else:
                    diag.n_after_bidi_gap_filter = C
            else:
                diag.n_after_bidi_gap_filter = C

            dominant_mid = None
            if self.c.use_centroid_dominance and C >= 2 and centroid_scores.max().item() > 0:
                c_sorted, c_idx = torch.sort(centroid_scores, descending=True)
                top1_c = c_sorted[0].item()
                top2_c = c_sorted[1].item()
                cent_margin = top1_c / max(top2_c, 1e-6) if top2_c > 0 else float("inf")
                diag.dominance_centroid_margin_observed = cent_margin
                centroid_cond = (
                    top1_c >= self.c.dominance_centroid_top1_floor
                    and cent_margin >= self.c.dominance_centroid_margin
                )

                consensus_cond = True
                top1_c_idx = c_idx[0].item()
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

                diag.consensus_passed = centroid_cond and consensus_cond
                if centroid_cond and consensus_cond:
                    diag.dominance_triggered = True
                    diag.centroid_dominance_triggered = True
                    dominant_mid = mems[top1_c_idx].mid
                    keep_thresh = top1_c * self.c.consensus_strict_keep_ratio
                    keep_mask = centroid_scores >= keep_thresh
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
                        C = len(mems)

            if self.c.use_idf_dominance and C >= 2 and forward_idf_t.max().item() > 0:
                fwd_sorted, fwd_sort_idx = torch.sort(forward_idf_t, descending=True)
                top1_fwd = fwd_sorted[0].item()
                top2_fwd = fwd_sorted[1].item()
                idf_margin = top1_fwd / max(top2_fwd, 1e-6)
                diag.dominance_idf_margin_observed = idf_margin
                if (
                    top1_fwd >= self.c.dominance_idf_top1_floor
                    and idf_margin >= self.c.dominance_idf_margin
                ):
                    diag.dominance_triggered = True
                    if dominant_mid is None:
                        dominant_mid = mems[fwd_sort_idx[0].item()].mid
                    keep_thresh = top1_fwd / self.c.dominance_idf_margin
                    keep_mask = forward_idf_t >= keep_thresh
                    keep_mask[fwd_sort_idx[0].item()] = True
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
                        C = len(mems)

            if self.c.use_dominance_filter and C >= 2 and content_classifier is not None:
                dominance_scores = forward_idf_t if forward_idf_t.max().item() > 0 else rerank_scores
                sorted_idx = torch.argsort(dominance_scores, descending=True)
                top1_local = sorted_idx[0].item()
                top2_local = sorted_idx[1].item()
                top1_score = dominance_scores[top1_local].item()
                top2_score = dominance_scores[top2_local].item()
                margin = top1_score / max(abs(top2_score), 1e-6) if top2_score > 0 else float("inf")
                diag.dominance_margin_observed = margin
                top1_sem = sem_sim_t[top1_local].item()
                top1_mem = mems[top1_local]
                top1_label = self._mem_label_set(top1_mem, content_classifier)
                if (
                    len(top1_label) >= self.c.dominance_min_label_size
                    and top1_sem >= self.c.dominance_sem_floor
                    and margin >= self.c.dominance_margin
                ):
                    diag.dominance_triggered = True
                    if dominant_mid is None:
                        dominant_mid = top1_mem.mid
                    keep_local = []
                    for i, mem in enumerate(mems):
                        if i == top1_local:
                            keep_local.append(i)
                            continue
                        mem_label = self._mem_label_set(mem, content_classifier)
                        if self._jaccard(top1_label, mem_label) >= self.c.dominance_jaccard_threshold:
                            keep_local.append(i)
                    if len(keep_local) < C:
                        kt = torch.tensor(keep_local, device=dev, dtype=torch.long)
                        mems = [mems[i] for i in keep_local]
                        sb = sb[kt]
                        sf = sf[kt]
                        rerank_scores = rerank_scores[kt]
                        forward_t = forward_t[kt]
                        bidi_min_t = bidi_min_t[kt]
                        sem_sim_t = sem_sim_t[kt]
                        forward_idf_t = forward_idf_t[kt]
                        centroid_scores = centroid_scores[kt]
                        C = len(mems)
            diag.n_after_dominance_filter = C

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


class MemLLM(v322.MemLLM):
    def __init__(self, c):
        super().__init__(c)
        self.amm = AMM(c)
        self.bridge = EmbBridge(c)
        self._last_hard_injected_tids = None

    def load(self, name="gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tok = GPT2Tokenizer.from_pretrained(name)
        self.llm = GPT2LMHeadModel.from_pretrained(name)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layer_pool = AdaptiveLayerPool(self.llm.config.n_layer + 1, self.c.d_LLM)
        self.content_classifier = ContentTokenClassifier(
            self.tok,
            self.c.content_min_len,
            strict_min_len=self.c.strict_starter_min_decoded_len,
        )
        self._degen_guard = DegenerationGuard(self.tok, self.c, self.content_classifier)
        self.bridge.aligner.calibrate(self.llm)
        self.c.vocab_size = self.llm.config.vocab_size
        self._wte_normed = F.normalize(self.llm.transformer.wte.weight.detach(), dim=-1, eps=1e-8)
        self.amm.wte_normed = self._wte_normed
        self._build_wte_neighbor_cache()

    def _compute_tfidf_idf(self) -> Dict[int, float]:
        if self.content_classifier is None:
            return {}
        return self.amm._compute_corpus_idf(self.content_classifier)

    def _build_hard_wte_last_slots(self, diag, query_content_ids_per_batch):
        if not self.c.use_dominant_hard_prefix:
            return None, None, None
        dev = next(self.parameters()).device
        wte = self.llm.transformer.wte.weight.detach()
        wte_n = self._wte_normed
        cc = self.content_classifier
        idf = self._compute_tfidf_idf() if self.c.use_tfidf_weighting else {}
        hard_last_n = self.c.hard_prefix_last_slots
        D = self.c.d_LLM
        B = len(diag.batch_mem_weights) if diag.batch_mem_weights else 0
        if B == 0 or cc is None:
            return None, None, None

        hard_wte_last = torch.zeros(B, hard_last_n, D, device=dev)
        triggered_mask = [False] * B
        injected_tids_per_batch = [[] for _ in range(B)]
        strict_set = (
            cc.strict_content_starter_ids if self.c.use_strict_content_starter else cc.content_starter_ids
        )

        for b in range(B):
            dom_mid = (
                diag.dominant_per_batch[b]
                if diag.dominant_per_batch and b < len(diag.dominant_per_batch)
                else None
            )
            if dom_mid is None or dom_mid not in self.amm.tree.store:
                continue
            mem = self.amm.tree.store[dom_mid]
            valid_ids = []
            for tid in self.amm._get_mem_scoring_ids(mem):
                if tid >= wte.shape[0]:
                    continue
                if tid not in strict_set:
                    continue
                valid_ids.append(tid)
            if not valid_ids:
                continue

            idf_vals = torch.tensor([idf.get(t, 1.0) for t in valid_ids], device=dev)
            q_ids = query_content_ids_per_batch[b] if b < len(query_content_ids_per_batch) else []
            q_valid = [i for i in q_ids if i < wte_n.shape[0]]
            if q_valid:
                q_centroid = self.amm._compute_idf_weighted_centroid(q_valid, wte_n, idf, self.c.idf_floor)
                if q_centroid is not None:
                    v_tensor = torch.tensor(valid_ids, device=dev)
                    rel = (wte_n[v_tensor] @ q_centroid).clamp(min=0)
                    scores = idf_vals * (rel + self.c.content_bias_relevance_floor)
                else:
                    scores = idf_vals
            else:
                scores = idf_vals

            K = min(hard_last_n, len(valid_ids))
            _, top_idx = scores.topk(K)
            top_tids_ranked = [valid_ids[top_idx[i].item()] for i in range(K)]
            injected_tids_per_batch[b] = top_tids_ranked
            for slot_pos in range(hard_last_n):
                rank = hard_last_n - 1 - slot_pos
                if rank < K:
                    tid = top_tids_ranked[rank]
                    hard_wte_last[b, slot_pos] = wte[tid]
                else:
                    hard_wte_last[b, slot_pos] = wte[top_tids_ranked[0]]
            triggered_mask[b] = True

        if not any(triggered_mask):
            return None, None, None
        return hard_wte_last, triggered_mask, injected_tids_per_batch

    def _get_prefix(self, hs, mask=None, pl=0, update_stats=True, return_extra=False, ids=None):
        pooled, xq, fq = self.extract_state(hs, mask, pl)
        trimmed_mask = mask[:, pl:] if mask is not None and pl > 0 else mask
        if trimmed_mask is not None and pooled.shape[1] != trimmed_mask.shape[1]:
            trimmed_mask = None
        query_content_ids_per_batch = []
        if ids is not None and self.content_classifier is not None:
            for b in range(ids.shape[0]):
                b_ids = ids[b].tolist()
                b_exact = list(set(self.content_classifier.get_content_ids_from_tokens(b_ids)))
                query_content_ids_per_batch.append(b_exact)
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
        )

        if return_extra:
            content_bias = self._build_content_bias(diag, query_content_ids_per_batch)
            first_step_bias = self._build_first_step_lexical_bias(diag, query_content_ids_per_batch)
            return prefix, fiber_summary, diag, content_bias, first_step_bias
        return prefix

    def generate(self, prompt, mt=50, greedy=False):
        tk = self.tok(prompt, return_tensors="pt")
        dev = next(self.parameters()).device
        ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
        with torch.no_grad():
            o = self.fwd(ids, mask)
            prefix, fiber_summary, _, content_bias, first_step_bias = self._get_prefix(
                o["hs"], mask, update_stats=True, return_extra=True, ids=ids
            )
            vocab_bias = self._compute_vocab_bias(fiber_summary)
        has_content = content_bias is not None and content_bias.abs().max().item() > 0.01
        has_first_step = first_step_bias is not None and first_step_bias.abs().max().item() > 1e-6
        cc = self.content_classifier

        hard_injected_tids = set()
        hard_inject_start_step = 0
        if self._last_hard_injected_tids is not None and self._last_hard_injected_tids:
            hard_injected_tids = set(self._last_hard_injected_tids[0])

        domain_anchors = self._compute_domain_anchors(content_bias) if has_content else [[]]
        anchors_for_b0 = set(domain_anchors[0]) if domain_anchors else set()
        generated_anchors = set()
        generated_ids = []
        generated_content_counts: Dict[int, int] = {}
        consecutive_content = 0
        recent_starters: List[Tuple[int, int]] = []

        for i in range(mt):
            if i > 0 and i % self.c.retrieval_interval == 0:
                with torch.no_grad():
                    o = self.fwd(ids, mask, prefix)
                    pl = o["pl"]
                    prefix, fiber_summary, _, content_bias, first_step_bias = self._get_prefix(
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
                step_scale_content = max(self.c.content_bias_floor, 1.0 - i * self.c.content_bias_decay)
                step_scale_learned = max(self.c.semantic_boost_floor, 1.0 - i * self.c.semantic_boost_decay)
                if i == 0:
                    effective_content_scale = step_scale_content * self.c.first_step_content_multiplier
                elif consecutive_content >= self.c.structural_rhythm_threshold:
                    effective_content_scale = step_scale_content * 0.25
                    if cc:
                        for fid in list(cc.function_ids)[:5000]:
                            if fid < lg.shape[-1]:
                                lg[0, fid] += self.c.structural_boost
                else:
                    effective_content_scale = step_scale_content
                if has_first_step and i < self.c.first_step_lexical_decay_steps:
                    V_fs = min(lg.shape[-1], first_step_bias.shape[-1])
                    lg[:, :V_fs] = lg[:, :V_fs] + first_step_bias[:, :V_fs] * self.c.first_step_lexical_scale
                if has_content:
                    cb_adjusted = content_bias.clone()
                    for tid, count in generated_content_counts.items():
                        if tid < cb_adjusted.shape[-1]:
                            cb_adjusted[0, tid] *= self.c.generated_token_decay ** count
                    V = min(lg.shape[-1], cb_adjusted.shape[-1])
                    lg[:, :V] = lg[:, :V] + cb_adjusted[:, :V] * self.c.content_bias_scale * effective_content_scale
                if vocab_bias is not None:
                    V2 = min(lg.shape[-1], vocab_bias.shape[-1])
                    lg[:, :V2] = lg[:, :V2] + vocab_bias[:, :V2] * self.c.semantic_boost_scale * step_scale_learned
                if i == 0 and cc is not None:
                    if self.c.use_strict_content_starter:
                        cmask = cc.strict_content_starter_mask(dev)
                    elif self.c.use_word_starter_filter:
                        cmask = cc.content_starter_mask(dev)
                    else:
                        cmask = cc.content_mask(dev)
                    V3 = min(lg.shape[-1], cmask.shape[0])
                    lg[0, :V3] = lg[0, :V3] + cmask[:V3] * self.c.universal_content_boost
                elif i < self.c.universal_content_boost_steps and cc is not None and has_content:
                    cmask = cc.content_starter_mask(dev) if self.c.use_word_starter_filter else cc.content_mask(dev)
                    V3 = min(lg.shape[-1], cmask.shape[0])
                    boost_scale = 1.0 - i / self.c.universal_content_boost_steps
                    lg[0, :V3] = lg[0, :V3] + cmask[:V3] * self.c.universal_content_boost * boost_scale
                if i >= self.c.domain_anchor_start_step and anchors_for_b0 and has_content:
                    coverage = len(generated_anchors) / max(len(anchors_for_b0), 1)
                    if coverage < self.c.domain_anchor_coverage_threshold:
                        for tid in anchors_for_b0 - generated_anchors:
                            if tid < lg.shape[-1]:
                                lg[0, tid] += self.c.domain_anchor_boost
                if cc:
                    for tid, count in generated_content_counts.items():
                        if tid in cc.content_ids and tid < lg.shape[-1]:
                            lg[0, tid] -= self.c.content_repeat_penalty * count
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
                if (
                    self.c.use_strict_or_continuation
                    and cc is not None
                    and i < self.c.strict_or_cont_steps
                ):
                    prev_is_word_starter_content = (
                        len(generated_ids) > 0
                        and generated_ids[-1] in cc.word_starter_ids
                        and generated_ids[-1] in cc.content_ids
                    )
                    if not prev_is_word_starter_content:
                        nsc_mask = cc.non_strict_content_mask(dev)
                        V4 = min(lg.shape[-1], nsc_mask.shape[0])
                        lg[0, :V4] = lg[0, :V4] - nsc_mask[:V4] * self.c.strict_or_cont_penalty
                if self._degen_guard is not None:
                    lg = self._degen_guard.process(
                        lg,
                        generated_ids,
                        i,
                        first_step_penalty_mult=self.c.first_step_penalty_multiplier if i == 0 else 1.0,
                    )
                if i < self.c.early_content_steps and cc is not None:
                    for pid in cc.punct_ids:
                        if pid < lg.shape[-1]:
                            lg[0, pid] = -float("inf")
                    for nid in cc.newline_ids:
                        if nid < lg.shape[-1]:
                            lg[0, nid] = -float("inf")
                if i == 0 and cc is not None:
                    for fid in cc.filler_ids:
                        if fid < lg.shape[-1]:
                            lg[0, fid] -= self.c.step0_filler_penalty

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
                if nxt_id in anchors_for_b0:
                    generated_anchors.add(nxt_id)
                if nxt_id in cc.word_starter_ids:
                    recent_starters.append((nxt_id, i))
            else:
                consecutive_content = 0
            recent_starters = [(t, s) for (t, s) in recent_starters if (i - s) < self.c.bpe_echo_window]
            ids = torch.cat([ids, nxt], 1)
            mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)

        return self.tok.decode(ids[0], skip_special_tokens=True)
