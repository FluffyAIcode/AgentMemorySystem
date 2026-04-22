"""MemLLM4 — top-level model.

Composes: LLMBackbone4, TemporalBundle, TopicBundle, ContextBundle,
KakeyaRegistry, CrossBundleAttention, EmbBridge4, MemStore.

Public surface (write / prepare_decode_context / generate) intentionally
mirrors v3.46 MemLLM so later benchmark adapters can swap one for the other.
Internally everything is v4's design.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ams_v4.attention.cross_bundle import CrossBundleAttention
from ams_v4.bridge.backbone import LLMBackbone4
from ams_v4.bundles.context import ContextBundle
from ams_v4.bundles.temporal import TemporalBundle
from ams_v4.bundles.topic import TopicBundle
from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import MemEntry
from ams_v4.core.mem_store import MemStore
from ams_v4.core.types import Tensor
from ams_v4.kakeya.registry import KakeyaRegistry
from ams_v4.projection.bridge import EmbBridge4


@dataclass
class DecodeContext4:
    prefix: Tensor        # (B, L_mem, d_LLM)
    n_memories: int


class MemLLM4(nn.Module):
    """End-to-end memory LM.

    Usage:
      cfg = Cfg4()
      m = MemLLM4(cfg)
      m.load()                      # loads backbone
      m.write("fact 1")             # encode + insert
      m.write("fact 2")
      # ...
      out = m.generate("question", mt=30)
    """

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self.backbone = LLMBackbone4(cfg)
        self.bundle_time = TemporalBundle(cfg)
        self.bundle_topic = TopicBundle(cfg)
        self.bundle_ctx = ContextBundle(cfg)
        self.cross_attn = CrossBundleAttention(cfg)
        self.bridge = EmbBridge4(cfg)
        self.kakeya = KakeyaRegistry(cfg)
        self.store = MemStore(cfg)

        # Running session summary (exponential moving average over write() hidden states)
        self._session_summary: Optional[Tensor] = None
        self._session_ema = 0.3  # weight on new hidden

        # Cached wte_normed for TopicEncoder (rebuilt lazily on first write after load)
        self._wte_normed_cache: Optional[Tensor] = None

    # ─── Load ────────────────────────────────────────────────────────────

    def load(self, name: Optional[str] = None,
             device: Optional[torch.device] = None) -> None:
        self.backbone.load(name=name, device=device)
        # Move v4 modules to backbone's device (these are the trainable parts)
        self.to(self.backbone.device)
        # Cache normalized word-token embeddings for TopicEncoder
        self._wte_normed_cache = self._build_wte_normed()

    def _build_wte_normed(self) -> Tensor:
        """L2-normalized wte weight; used as the content-token embedding table."""
        wte = self.backbone.wte
        # Embedding weight: (V, d_LLM) — float32 regardless of backbone dtype
        W = wte.weight.detach().float()
        return F.normalize(W, dim=-1, eps=1e-8)

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _tokenize(self, text: str):
        return self.backbone.tokenize(text)

    def _hidden_pooled(self, text: str):
        """Forward the text through the backbone and return (pooled_hidden,
        ids, mask). pooled_hidden is the mean over tokens: (1, d_LLM)."""
        ids, mask = self._tokenize(text)
        hs = self.backbone.hidden_states(ids, mask)       # (1, T, d_LLM)
        # Masked mean over tokens
        m = mask.unsqueeze(-1).to(hs.dtype)               # (1, T, 1)
        pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        return pooled.float(), ids, mask                  # encoders use float32

    def _update_session_summary(self, hidden: Tensor) -> Tensor:
        """Running EMA of pooled hidden states. Returns current summary (B, d_LLM)."""
        if self._session_summary is None:
            self._session_summary = hidden.detach().clone()
        else:
            self._session_summary = (
                (1 - self._session_ema) * self._session_summary
                + self._session_ema * hidden.detach()
            )
        return self._session_summary.to(device=hidden.device, dtype=hidden.dtype)

    def _time_scalars(self) -> Tensor:
        """(1, 3) tensor: (absolute_time, recency, cnt). v4.5 uses store's
        internal clock (== number of writes so far)."""
        ts = float(self.store._next_mid)
        recency = 0.0     # time since last access; new write has recency 0
        cnt = 0.0
        device = self.backbone.device
        return torch.tensor([[ts, recency, cnt]], device=device, dtype=torch.float32)

    def _extract_large_fields(
        self, hidden: Tensor, content_token_ids: List[int], ctx_base: Tensor,
    ) -> Dict[str, Tensor]:
        """Assemble the three large fields that the KakeyaRegistry compresses.

          - semantic_emb:       pooled hidden state, (d_LLM,)
          - content_wte_mean:   IDF-uniform mean of wte_normed over content tokens
          - context_descriptor: projected context_base to d_LLM via repeat+pad
                                (simple; proper projection is a v4.6 concern)
        """
        hidden_flat = hidden.squeeze(0).detach().cpu().float()  # (d_LLM,)
        W = self._wte_normed_cache  # (V, d_LLM) on backbone device
        if W is None:
            W = self._build_wte_normed()
        V, d = W.shape
        ids = [t for t in content_token_ids if 0 <= int(t) < V]
        if ids:
            idx = torch.tensor(ids, dtype=torch.long, device=W.device)
            content_wte_mean = W[idx].mean(dim=0).detach().cpu().float()
        else:
            content_wte_mean = torch.zeros(d, dtype=torch.float32)

        # context_descriptor: simplest "carry to d_LLM" projection — tile + zero-pad
        ctx_flat = ctx_base.detach().cpu().float().flatten()
        ctx_desc = torch.zeros(d, dtype=torch.float32)
        L = min(ctx_flat.numel(), d)
        ctx_desc[:L] = ctx_flat[:L]

        return {
            "semantic_emb": hidden_flat,
            "content_wte_mean": content_wte_mean,
            "context_descriptor": ctx_desc,
        }

    def _maybe_build_kakeya(self) -> None:
        """Rebuild the kakeya registry once we have enough entries."""
        if len(self.store) < self.cfg.kakeya_min_entries:
            return
        # Only build once (v4.5 does not yet do periodic rebuilds)
        if self.kakeya.sets and any(s.is_active for s in self.kakeya.sets):
            return
        field_corpus: Dict[str, List[Tensor]] = {
            "semantic_emb": [], "content_wte_mean": [], "context_descriptor": [],
        }
        for e in self.store.all_entries():
            fields = e.kakeya_handle.entries
            # If entry has no handle yet (pre-kakeya write), re-extract fields
            # by re-encoding its source_text. Expensive; but only runs once.
            if not fields:
                large = self._reextract_fields_for_entry(e)
                for k in field_corpus:
                    field_corpus[k].append(large[k])
            else:
                # Use the stored raw-tensor snapshot if available on the entry
                # (we cache them under _pending_large_fields during write())
                snap = getattr(e, "_pending_large_fields", None)
                if snap is None:
                    large = self._reextract_fields_for_entry(e)
                else:
                    large = snap
                for k in field_corpus:
                    field_corpus[k].append(large[k])
        stacked = {k: torch.stack(v, dim=0) for k, v in field_corpus.items()}
        bundle_axes = {
            "time":  self.bundle_time.canonical_axis().detach().cpu().float(),
            "topic": self.bundle_topic.canonical_axis().detach().cpu().float(),
            "ctx":   self.bundle_ctx.canonical_axis().detach().cpu().float(),
        }
        self.kakeya.build(stacked, bundle_axes)
        # Now re-encode every entry through the active registry and drop the snapshot
        for e in self.store.all_entries():
            snap = getattr(e, "_pending_large_fields", None)
            if snap is not None:
                e.kakeya_handle = self.kakeya.encode_memory_fields(snap)
                # Remove the snapshot — registry is the source of truth now
                try:
                    delattr(e, "_pending_large_fields")
                except AttributeError:
                    pass

    def _reextract_fields_for_entry(self, e: MemEntry) -> Dict[str, Tensor]:
        """Recompute large fields from e.source_text. Used if a write happened
        before kakeya was built. Best-effort: re-encodes identically because
        the backbone is frozen.
        """
        hidden, ids, mask = self._hidden_pooled(e.source_text)
        return self._extract_large_fields(hidden, e.content_token_ids, e.ctx_base)

    # ─── Public surface ─────────────────────────────────────────────────

    def write(self, text: str, training_mode: bool = False) -> Optional[int]:
        """Encode a text, insert a MemEntry, maybe build kakeya. Returns mid
        (≥ 0 on success, None if rejected — v4.5 never rejects)."""
        assert self.backbone._loaded, "MemLLM4.write requires load() first"
        hidden, ids, mask = self._hidden_pooled(text)

        # ─── Bundle encoding ──────────────────────────────────────────
        time_scalars = self._time_scalars()
        surprise = torch.zeros(1, device=hidden.device)

        time_b, time_f, time_d = self.bundle_time.encode(
            hidden, time_scalars=time_scalars, surprise=surprise,
        )
        content_ids = ids[0].tolist()
        W = self._wte_normed_cache.to(hidden.device)
        topic_b, topic_f, topic_d = self.bundle_topic.encode(
            hidden, content_token_ids=content_ids, wte_normed=W,
        )
        session_summary = self._update_session_summary(hidden)
        ctx_b, ctx_f, ctx_d = self.bundle_ctx.encode(
            hidden, session_summary=session_summary, prev_turns=None,
        )

        # ─── MemEntry ─────────────────────────────────────────────────
        entry = MemEntry(
            mid=-1,
            time_base=time_b[0].detach(), time_fiber=time_f[0].detach(),
            time_dirn=time_d[0].detach(),
            topic_base=topic_b[0].detach(), topic_fiber=topic_f[0].detach(),
            topic_dirn=topic_d[0].detach(),
            ctx_base=ctx_b[0].detach(), ctx_fiber=ctx_f[0].detach(),
            ctx_dirn=ctx_d[0].detach(),
            surprise=0.0, ts=float(self.store._next_mid),
            last=float(self.store._next_mid), cnt=0,
            source_text=text, content_token_ids=content_ids,
        )

        # ─── Large fields ────────────────────────────────────────────
        large = self._extract_large_fields(hidden, content_ids, ctx_b[0])
        if self.kakeya.sets and any(s.is_active for s in self.kakeya.sets):
            entry.kakeya_handle = self.kakeya.encode_memory_fields(large)
        else:
            # Stash for later kakeya build. This is a transient; cleared by
            # _maybe_build_kakeya once it re-encodes through the registry.
            entry._pending_large_fields = large  # type: ignore[attr-defined]

        mid = self.store.add(entry)
        self._maybe_build_kakeya()
        return mid

    def prepare_decode_context(
        self, ids: Tensor, mask: Tensor, update_stats: bool = False,
    ) -> DecodeContext4:
        """Produce a prefix tensor via retrieval + cross-bundle attention.

        v4.5 strategy: attend over ALL entries (flat, no retrieval filter).
        Retrieval via the three DirectionTreeV4s is implemented and available,
        but for small stores (≤ 50) the flat-attend pass is the cleanest
        baseline — it ensures we are measuring the attention + bundle
        mechanism, not retrieval filter noise. Retrieval filtering becomes
        non-optional in v4.6 once `retrieval_topk` becomes a real constraint.
        """
        hs = self.backbone.hidden_states(ids, mask)
        m = mask.unsqueeze(-1).to(hs.dtype)
        q_hidden = ((hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)).float()

        entries = self.store.all_entries()
        if not entries:
            prefix = torch.zeros(
                q_hidden.shape[0], self.cfg.L_mem, self.cfg.d_LLM,
                device=q_hidden.device, dtype=q_hidden.dtype,
            )
        else:
            prefix = self.cross_attn(q_hidden, entries)
        return DecodeContext4(prefix=prefix, n_memories=len(entries))

    def generate(self, prompt: str, mt: int = 40, greedy: bool = True) -> str:
        assert self.backbone._loaded, "MemLLM4.generate requires load() first"
        ids, mask = self._tokenize(prompt)
        ctx = self.prepare_decode_context(ids, mask)

        # Prefix lives in d_LLM; cast to backbone dtype and build mask
        backbone_dtype = next(self.backbone.model.parameters()).dtype
        prefix_embeds = self.bridge.prefix_post_ln(ctx.prefix.to(backbone_dtype))
        prefix_mask = torch.ones(
            ids.shape[0], self.cfg.L_mem, dtype=mask.dtype, device=mask.device,
        )

        out_ids = self.backbone.generate_with_prefix(
            prefix_embeds, prefix_mask, ids, mask,
            max_new_tokens=mt, greedy=greedy,
        )
        new_ids = out_ids[0, ids.shape[1]:].tolist()
        return self.backbone.tok.decode(new_ids, skip_special_tokens=True)
