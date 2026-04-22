"""Cfg4 — single config dataclass for AMS v4.

Invariants are checked in __post_init__. Adding a new flag requires adding a
matching invariant or explicitly documenting why none is needed.

Philosophy (from ARCHITECTURE_v4.md §5): not a knob-turning surface. Defaults
are conservative; invariants are strict. Only add what the abstract spec
requires.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional


@dataclass
class Cfg4:
    # ─── Backbone ──────────────────────────────────────────────────────────
    llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_dtype: str = "bf16"
    d_LLM: int = 1536
    vocab_size: int = 151936

    # ─── Bundles (§1.3) ───────────────────────────────────────────────────
    # Three bundles, each with independent base dim and fiber dim.
    d_time: int = 8            # TemporalBundle base space dim
    d_F_time: int = 32         # TemporalBundle fiber dim
    d_topic: int = 16          # TopicBundle base space dim (on S^{d_topic-1})
    d_F_topic: int = 64        # TopicBundle fiber dim
    d_ctx: int = 12            # ContextBundle base space dim
    d_F_ctx: int = 48          # ContextBundle fiber dim

    # Per-bundle attention heads (used by CrossBundleAttention)
    n_heads_time: int = 4
    n_heads_topic: int = 8
    n_heads_ctx: int = 4

    # ─── Kakeya registry (§1.1, §1.2) ─────────────────────────────────────
    # Target number of KakeyaSet instances. The registry may hold fewer if
    # some bundles have < min_entries_to_build memories.
    n_kakeya_sets: int = 4
    kakeya_variance_ratio: float = 0.99     # PCA variance retained per set
    kakeya_K: int = 16                      # segment centers per set
    kakeya_d_res: int = 5                   # sparse residual width per encoded vec
    kakeya_min_entries: int = 8             # don't build skeleton below this count
    kakeya_alignment_tol: float = 1e-3      # §1.3 t_dir alignment tolerance
    kakeya_reconstruction_tol: float = 0.15 # §6 invariant 5

    # Which large memory fields the registry compresses. Anything ≥ this dim
    # MUST be compressed (§6 invariant 2).
    compression_min_dim: int = 256

    # ─── Memory store / DirectionTree ─────────────────────────────────────
    # Three DirectionTrees, one per bundle.
    tree_K: int = 8
    tree_max_leaf: int = 20
    retrieval_topk: int = 8
    retrieval_beam: int = 5

    # ─── Riemannian geometry (ported as-is from v3.46) ────────────────────
    n_geo_pts: int = 8
    geo_max_steps: int = 80
    geo_tol: float = 1e-5
    geo_lr: float = 0.02
    norm_correction_interval: int = 4

    # ─── Attention → prefix (§1.5) ────────────────────────────────────────
    # prefix_slots_time + prefix_slots_topic + prefix_slots_ctx = L_mem
    L_mem: int = 12
    prefix_slots_time: int = 2
    prefix_slots_topic: int = 6
    prefix_slots_ctx: int = 4

    # ─── Training / runtime flags ─────────────────────────────────────────
    strict_shape_checks: bool = True
    write_gate_threshold: float = 0.4
    tau: float = 0.07
    cfg_scale: float = 3.5

    # Loss weights — kept intentionally small and aligned to v4 structure,
    # not v3.46. v3.46 has ~15 loss terms; v4 starts with 5.
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "recon": 1.0,
        "bundle_axis_alignment": 0.5,   # §1.3 t_dir ≈ pushforward of bundle axis
        "cross_bundle_independence": 0.2,
        "prefix_semantic_anchor": 0.5,
        "write_policy": 0.1,
    })

    def __post_init__(self) -> None:
        # Bundle dims must each be ≥ 4 so PCA + K-means are meaningful
        assert self.d_time >= 4, "d_time must be >= 4"
        assert self.d_topic >= 4, "d_topic must be >= 4"
        assert self.d_ctx >= 4, "d_ctx must be >= 4"

        # Fiber dims must be divisible by their head counts
        assert self.d_F_time % self.n_heads_time == 0, \
            "d_F_time must be divisible by n_heads_time"
        assert self.d_F_topic % self.n_heads_topic == 0, \
            "d_F_topic must be divisible by n_heads_topic"
        assert self.d_F_ctx % self.n_heads_ctx == 0, \
            "d_F_ctx must be divisible by n_heads_ctx"

        # Kakeya config
        assert self.n_kakeya_sets >= 2, \
            "abstract architecture requires multiple kakeya sets (§1.1); n_kakeya_sets >= 2"
        assert 0.0 < self.kakeya_variance_ratio <= 1.0
        assert self.kakeya_K >= 2
        assert self.kakeya_d_res >= 0
        assert self.kakeya_min_entries >= 2
        assert self.kakeya_alignment_tol > 0
        assert 0 < self.kakeya_reconstruction_tol < 1

        # Prefix slot budget (§1.5) — must sum to L_mem so no slot is wasted
        slots_sum = (self.prefix_slots_time + self.prefix_slots_topic
                     + self.prefix_slots_ctx)
        assert slots_sum == self.L_mem, (
            f"prefix_slots_{{time,topic,ctx}} must sum to L_mem; "
            f"got {slots_sum} vs L_mem={self.L_mem}"
        )
        assert self.prefix_slots_time >= 1, "each bundle must own at least 1 prefix slot"
        assert self.prefix_slots_topic >= 1
        assert self.prefix_slots_ctx >= 1

        # Geometry sanity
        assert self.n_geo_pts >= 2
        assert 0 < self.tau < 1
        assert self.cfg_scale >= 0

        # Backbone
        assert self.llm_dtype in ("bf16", "fp16", "fp32")
        assert self.compression_min_dim >= 64, \
            "compression_min_dim < 64 would trigger compression on tiny fields — almost certainly a typo"

        # Loss weights
        for k, v in self.loss_weights.items():
            assert v >= 0, f"loss weight {k} must be non-negative"
