"""Training-time memory encoding helpers.

`MemLLM4.write()` detaches tensors when storing them in MemEntry so retrieval
is a forward-only operation. For training we need non-detached encoder
outputs. `encode_batch_for_training` runs the three bundles on a list of
texts and returns the raw (base, fiber, dirn) triples with gradients
attached, plus enough bookkeeping to build a temporary in-memory batch
store for use by CrossBundleAttention.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from ams_v4.core.mem_entry import MemEntry
from ams_v4.core.types import Tensor


@dataclass
class BatchEncoded:
    """A batch of encoded memories, retaining gradients.

    All fields are grad-carrying tensors stacked along dim 0 (batch).
    time_scalars used to produce time bundle inputs; kept for diagnostics.
    """
    # Per-memory text + token bookkeeping
    texts: List[str]
    content_token_ids: List[List[int]]

    # Pooled hidden states for each text (no grad — backbone frozen)
    hidden: Tensor                  # (N, d_LLM)

    # Three bundles' outputs, with grad
    time_base: Tensor;  time_fiber: Tensor;  time_dirn: Tensor    # (N, d_time) / d_F_time / d_time
    topic_base: Tensor; topic_fiber: Tensor; topic_dirn: Tensor   # (N, d_topic) / d_F_topic / d_topic
    ctx_base: Tensor;   ctx_fiber: Tensor;   ctx_dirn: Tensor     # (N, d_ctx) / d_F_ctx / d_ctx


def encode_batch_for_training(model, texts: List[str]) -> BatchEncoded:
    """Run the three bundles on every text in `texts`. Gradients retained.

    `model` is a `MemLLM4` instance already loaded. Returns a BatchEncoded.
    """
    assert model.backbone._loaded, "encode_batch_for_training requires MemLLM4.load() first"
    dev = model.backbone.device

    pooled_list: List[Tensor] = []
    content_tokens_list: List[List[int]] = []
    for text in texts:
        # Backbone hidden state (no grad needed on backbone; backbone is frozen)
        ids, mask = model.backbone.tokenize(text)
        with torch.no_grad():
            hs = model.backbone.hidden_states(ids, mask)        # (1, T, d_LLM)
        m = mask.unsqueeze(-1).to(hs.dtype)
        pooled = ((hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)).float()  # (1, d_LLM)
        pooled_list.append(pooled)
        content_tokens_list.append(ids[0].tolist())

    hidden = torch.cat(pooled_list, dim=0)   # (N, d_LLM)
    N = hidden.shape[0]

    # Time scalars: simple (idx, 0, 0) — batches in training don't reflect
    # wall-clock ordering, but idx varies so the encoder sees non-constant input.
    time_scalars = torch.stack([
        torch.tensor([float(i), 0.0, 0.0], device=dev, dtype=torch.float32)
        for i in range(N)
    ], dim=0)
    surprise = torch.zeros(N, device=dev)

    # Three encoders, gradients retained
    time_b, time_f, time_d = model.bundle_time.encode(
        hidden, time_scalars=time_scalars, surprise=surprise,
    )

    W = model._wte_normed_cache
    if W is None:
        W = model._build_wte_normed()
    W = W.to(hidden.device)
    topic_b, topic_f, topic_d = model.bundle_topic.encode(
        hidden, content_token_ids=content_tokens_list, wte_normed=W,
    )

    # Use hidden itself as session_summary in training — no running EMA
    ctx_b, ctx_f, ctx_d = model.bundle_ctx.encode(
        hidden, session_summary=hidden, prev_turns=None,
    )

    return BatchEncoded(
        texts=list(texts),
        content_token_ids=content_tokens_list,
        hidden=hidden,
        time_base=time_b, time_fiber=time_f, time_dirn=time_d,
        topic_base=topic_b, topic_fiber=topic_f, topic_dirn=topic_d,
        ctx_base=ctx_b, ctx_fiber=ctx_f, ctx_dirn=ctx_d,
    )


def batch_to_mementries(be: BatchEncoded) -> List[MemEntry]:
    """Build MemEntry objects from a grad-carrying batch. The MemEntries
    reference the grad tensors DIRECTLY (no detach) so CrossBundleAttention
    can be run with gradients still flowing back through the encoders.

    Caller is responsible for knowing these entries are for one training
    step only — never feed them to MemStore.add() (which mixes with the
    production write path).
    """
    entries: List[MemEntry] = []
    N = be.hidden.shape[0]
    for i in range(N):
        entries.append(MemEntry(
            mid=i,
            time_base=be.time_base[i], time_fiber=be.time_fiber[i], time_dirn=be.time_dirn[i],
            topic_base=be.topic_base[i], topic_fiber=be.topic_fiber[i], topic_dirn=be.topic_dirn[i],
            ctx_base=be.ctx_base[i], ctx_fiber=be.ctx_fiber[i], ctx_dirn=be.ctx_dirn[i],
            surprise=0.0, ts=float(i), last=float(i), cnt=0,
            source_text=be.texts[i],
            content_token_ids=be.content_token_ids[i],
        ))
    return entries
