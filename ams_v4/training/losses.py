"""v4.6 loss terms. See ARCHITECTURE_v4_TRAIN.md §2 for the design.

Each loss returns a scalar 0-D tensor. Callers weight them via
`cfg.loss_weights` and sum.
"""
from __future__ import annotations
from typing import List, Set

import torch
import torch.nn.functional as F

from ams_v4.core.types import Tensor
from ams_v4.training.batch_encode import BatchEncoded, batch_to_mementries


# ─── 2.1 prefix_semantic_anchor ──────────────────────────────────────────

def loss_prefix_semantic_anchor(model, be: BatchEncoded,
                                split_ratio: float = 0.5) -> Tensor:
    """Teacher-forced next-token NLL on the second half of each source_text,
    conditioned on a v4-produced prefix that attended over the batch's memories.

    Pipeline per example:
      1. Build MemEntries from `be` (grad-carrying).
      2. Tokenize source_text; split into (query_part, target_part).
      3. Pool the backbone hidden state of query_part to get q_hidden.
      4. Run CrossBundleAttention(q_hidden, entries) -> prefix (1, L_mem, d_LLM).
      5. Concat (prefix_embeds, query_embeds, target_embeds) and run backbone.
      6. NLL over target positions only.

    The batch's OWN query is included among the memories, so the prefix
    has the answer available — this is the teacher-forcing that lets the
    loss go down during training.
    """
    dev = model.backbone.device
    entries = batch_to_mementries(be)
    tok = model.backbone.tok
    wte = model.backbone.wte

    total = torch.zeros((), device=dev, dtype=torch.float32)
    n_examples = 0

    for i, text in enumerate(be.texts):
        # Tokenize the full source_text (no special tokens — we are running
        # mid-sequence NLL, not completion)
        full_ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
        T = full_ids.shape[1]
        if T < 4:
            continue
        split_at = max(1, int(T * split_ratio))

        query_ids = full_ids[:, :split_at]
        target_ids = full_ids[:, split_at:]
        # Build attention masks
        q_mask = torch.ones_like(query_ids)
        t_mask = torch.ones_like(target_ids)

        # Backbone query hidden (no grad through backbone)
        with torch.no_grad():
            q_hs = model.backbone.hidden_states(query_ids, q_mask)
        m = q_mask.unsqueeze(-1).to(q_hs.dtype)
        q_hidden = ((q_hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)).float()

        # Prefix via cross-bundle attention (grad flows)
        prefix = model.cross_attn(q_hidden, entries)           # (1, L_mem, d_LLM)

        # Build (prefix, query, target) input to backbone
        q_emb = wte(query_ids).to(prefix.dtype)
        t_emb = wte(target_ids).to(prefix.dtype)
        input_embeds = torch.cat([prefix, q_emb, t_emb], dim=1)
        prefix_mask = torch.ones(1, model.cfg.L_mem, dtype=q_mask.dtype, device=dev)
        attn_mask = torch.cat([prefix_mask, q_mask, t_mask], dim=1)

        # Cast to backbone dtype
        backbone_dtype = next(model.backbone.model.parameters()).dtype
        input_embeds = input_embeds.to(backbone_dtype)

        out = model.backbone.model(
            inputs_embeds=input_embeds, attention_mask=attn_mask, use_cache=False,
        )
        logits = out.logits.float()                            # (1, L_mem+T, V)

        # Predicted token at position p is logits[:, p-1]; we want predictions
        # for target tokens, which start at position (L_mem + split_at) in logits.
        # So for target position j (0..len(target)-1), predictor is
        #   logits[:, L_mem + split_at - 1 + j]
        # and the ground-truth token is target_ids[:, j].
        start = model.cfg.L_mem + split_at - 1
        n_t = target_ids.shape[1]
        pred_logits = logits[:, start : start + n_t]            # (1, n_t, V)
        nll = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="mean",
        )
        total = total + nll
        n_examples += 1

    if n_examples == 0:
        return total
    return total / n_examples


# ─── 2.2 bundle_axis_alignment ──────────────────────────────────────────

_CONTENT_TOKEN_ID_MIN = 1000  # skip punctuation and common BPE merges below this


def _jaccard(a: List[int], b: List[int]) -> float:
    """Jaccard restricted to content-ish token ids.

    Dropping token ids < _CONTENT_TOKEN_ID_MIN cuts punctuation and the
    most-common-BPE-merges that every sentence shares (e.g. "I", "the",
    "my"). Heuristic, but effective: for Qwen2.5 and GPT-2 vocabularies,
    the first ~1k ids are dominated by single chars and very-common merges.
    Without this, "positive pair" by Jaccard is driven by shared stopwords
    and the triplet loss collapses every topic_base onto the stopword
    direction.
    """
    def _content(xs: List[int]) -> set:
        return {int(t) for t in xs if int(t) >= _CONTENT_TOKEN_ID_MIN}
    sa, sb = _content(a), _content(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def loss_bundle_axis_alignment(model, be: BatchEncoded) -> Tensor:
    """Sum of three per-bundle contrastive sub-losses."""
    dev = model.backbone.device
    N = be.hidden.shape[0]
    parts: List[Tensor] = []

    # ─── Time sub-term: projection onto canonical axis should track ts ────
    # Use a non-saturating -Pearson(proj, index) loss so gradient always
    # flows through bundle_time._axis_raw; minimizing this pushes
    # proj_time to be monotone in the batch index (= write order).
    ax_time = model.bundle_time.canonical_axis()                # (d_time,)
    proj_time = be.time_base @ ax_time                          # (N,)
    if N >= 2:
        idx = torch.arange(N, device=dev, dtype=proj_time.dtype)
        px = proj_time - proj_time.mean()
        py = idx - idx.mean()
        denom = (px.norm() * py.norm()).clamp(min=1e-8)
        pearson = (px * py).sum() / denom                       # in [-1, 1]
        # Maximize correlation → minimize (1 - pearson).
        parts.append(1.0 - pearson)

    # ─── Topic sub-term: content-word triplet + diversity regularizer ────
    if N >= 3:
        jac = torch.zeros(N, N, device=dev)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                jac[i, j] = _jaccard(be.content_token_ids[i], be.content_token_ids[j])
        cos = be.topic_base @ be.topic_base.T                   # (N, N); base is unit-norm
        mask_self = torch.eye(N, device=dev, dtype=torch.bool)
        jac_masked = jac.clone(); jac_masked[mask_self] = -1.0
        pos_j = jac_masked.argmax(dim=1)
        neg_j = jac_masked.argmin(dim=1)
        i_idx = torch.arange(N, device=dev)
        pos_cos = cos[i_idx, pos_j]; neg_cos = cos[i_idx, neg_j]
        triplet_margin = 0.2
        hinge_topic = F.relu(neg_cos - pos_cos + triplet_margin).mean()
        parts.append(hinge_topic)

        # Diversity regularizer: every off-diagonal pair should have cos ≤
        # diversity_ceiling. Prevents the whole topic_base batch from
        # collapsing onto a single direction even when the triplet loss is
        # nominally satisfied.
        diversity_ceiling = 0.7
        off_diag = cos.masked_fill(mask_self, 0.0)
        over = F.relu(off_diag - diversity_ceiling)
        # Normalize by number of off-diag pairs
        n_pairs = max(1, N * (N - 1))
        parts.append(over.sum() / n_pairs)

    # ─── Context sub-term ─────────────────────────────────────────────────
    # Within a training batch "session" = same batch. Without cross-session
    # contrast, penalize ctx_base for drifting off its canonical axis.
    ax_ctx = model.bundle_ctx.canonical_axis()                  # (d_ctx,)
    ctx_base_n = F.normalize(be.ctx_base, dim=-1, eps=1e-8)
    proj_ctx = ctx_base_n @ ax_ctx                              # (N,)
    # Pull the mean toward 1 (on axis). Bounded loss.
    hinge_ctx = F.relu(0.3 - proj_ctx).mean()
    parts.append(hinge_ctx)

    if not parts:
        return torch.zeros((), device=dev)
    return torch.stack(parts).mean()


# ─── 2.3 cross_bundle_independence ──────────────────────────────────────

def loss_cross_bundle_independence(model, be: BatchEncoded,
                                    target_abs_cos: float = 0.3) -> Tensor:
    """Discourage the three bundles from collapsing to copies of each other.

    Compute pairwise cosine between (time_fiber[i], topic_fiber[i]) etc. —
    since they live in different dim, we project each fiber to a shared
    d_attn space first via a simple mean-pool-ish map. To keep this
    gradient-friendly and parameter-free, we use left-SVD projections:
    each fiber is projected onto its own unit-mean direction. Pragmatic.

    Simpler and enough for v4.6: reduce each fiber to its L2-normalized
    scalar projection onto the batch mean of its OWN bundle, then require
    those three scalars to be weakly correlated (target |corr| ≈ 0.3)
    across the batch.
    """
    dev = model.backbone.device
    # Projections: (N,) per bundle
    t_scalar = F.normalize(be.time_fiber, dim=-1).mean(dim=-1)
    p_scalar = F.normalize(be.topic_fiber, dim=-1).mean(dim=-1)
    c_scalar = F.normalize(be.ctx_fiber, dim=-1).mean(dim=-1)

    if t_scalar.numel() < 2:
        return torch.zeros((), device=dev)

    def _pearson_abs(x, y):
        x = x - x.mean(); y = y - y.mean()
        denom = (x.norm() * y.norm()).clamp(min=1e-8)
        return (x * y).sum().abs() / denom

    pairs = [
        _pearson_abs(t_scalar, p_scalar),
        _pearson_abs(t_scalar, c_scalar),
        _pearson_abs(p_scalar, c_scalar),
    ]
    # L2 distance from target
    return torch.stack([(p - target_abs_cos) ** 2 for p in pairs]).mean()


# ─── 2.4 recon ──────────────────────────────────────────────────────────

def loss_recon(model, be: BatchEncoded) -> Tensor:
    """Relative reconstruction error through kakeya.

    Uses the semantic_emb field only (= pooled hidden) — other fields
    don't carry gradients back to trainable encoders. This is sufficient:
    it drives `base_to_field` maps (once we make them nn.Parameter in a
    later v4.7) and surfaces any PCA/skeleton regression.

    In v4.6 base_to_field is still a plain Tensor (not nn.Parameter), so
    this loss contributes *diagnostic only* — it is computed but backprop
    through it bottoms out at the kakeya operators. That's fine; we keep
    it so the trainer reports it, and it becomes a real gradient signal
    in v4.7 if/when we make base_to_field trainable.
    """
    dev = model.backbone.device
    if not model.kakeya.sets or not any(s.is_active for s in model.kakeya.sets):
        return torch.zeros((), device=dev)

    total = torch.zeros((), device=dev, dtype=torch.float32)
    n = 0
    # Sample a few memories for reconstruction check
    for i in range(min(be.hidden.shape[0], 4)):
        v = be.hidden[i].detach().cpu().float()
        fields = {
            "semantic_emb": v,
            "content_wte_mean": v.clone(),        # placeholder; fine for the diag
            "context_descriptor": torch.zeros_like(v),
        }
        try:
            handle = model.kakeya.encode_memory_fields(fields)
            dec = model.kakeya.decode_field(handle, "semantic_emb")
            if dec is None: continue
            rel = (dec - v).norm() / v.norm().clamp(min=1e-8)
            total = total + rel
            n += 1
        except Exception:
            continue
    if n == 0:
        return total
    # Bring onto device as a non-grad scalar (diagnostic)
    return (total / n).to(dev)


# ─── 2.5 write_policy ───────────────────────────────────────────────────

def loss_write_policy(model, be: BatchEncoded) -> Tensor:
    """Tiny regularizer. Penalize fibers with unusually small norm
    (collapse prevention) and target_text shorter than 3 tokens.
    """
    dev = model.backbone.device
    tn = be.time_fiber.norm(dim=-1)
    pn = be.topic_fiber.norm(dim=-1)
    cn = be.ctx_fiber.norm(dim=-1)
    # Encourage norm >= 1.0 (mild hinge)
    hinge = (F.relu(1.0 - tn).mean()
             + F.relu(1.0 - pn).mean()
             + F.relu(1.0 - cn).mean()) / 3.0

    # Short-text penalty
    short_tokens = sum(1 for ids in be.content_token_ids if len(ids) < 3)
    short_penalty = torch.tensor(
        float(short_tokens) / max(1, len(be.content_token_ids)),
        device=dev, dtype=torch.float32,
    )
    return hinge + 0.01 * short_penalty
