"""MemEntry and KakeyaHandle — the memory-atom data structures.

The single biggest shape change vs v3.46: a MemEntry now carries
**three** (base, fiber, dirn) triples — one per bundle — rather than one.
Large fields (semantic_emb, content_wte_mean, context_descriptor) are no
longer stored raw; they live compressed in a KakeyaRegistry and are
referenced through KakeyaHandle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from ams_v4.core.types import Tensor


@dataclass
class CompressedVec:
    """A single vector encoded by one KakeyaSet.

    Matches the abstract Kakeya-set shape:
      v ≈ mean + basis.T @ (alpha · t_dir + t · centers[seg_id] + sparse_residual)
    """
    set_idx: int
    seg_id: int
    alpha: float
    t: float
    residual_vals: Tensor        # (d_res,)  float32 on CPU
    residual_idx: Tensor         # (d_res,)  int64   on CPU


@dataclass
class KakeyaHandle:
    """Maps a MemEntry's large-field names to their CompressedVec locations.

    Example:
      handle.entries = {
          "semantic_emb":      [CompressedVec(set_idx=0, ...),   # time-set encoding
                                CompressedVec(set_idx=1, ...)],  # topic-set encoding
          "content_wte_mean":  [CompressedVec(set_idx=1, ...)],
          "context_descriptor":[CompressedVec(set_idx=2, ...)],
      }

    A given field may be encoded by multiple sets (cross-axis redundancy,
    used by CrossBundleAttention). The list is ordered by set_idx.
    """
    entries: Dict[str, List[CompressedVec]] = field(default_factory=dict)

    def fields(self) -> List[str]:
        return sorted(self.entries.keys())

    def get(self, field_name: str, set_idx: int) -> Optional[CompressedVec]:
        """Return the CompressedVec for (field, set) or None if not present."""
        for cv in self.entries.get(field_name, []):
            if cv.set_idx == set_idx:
                return cv
        return None


@dataclass
class MemEntry:
    """A single memory. Carries three bundle coordinate triples + a kakeya handle.

    Invariant (checked by MemStore.verify_consistency):
      - every large field (shape dim >= Cfg4.compression_min_dim) is
        represented in `kakeya_handle.entries`, not as a raw tensor here.
      - each of the three (base, fiber, dirn) triples is non-None once the
        memory has been written (bundles encode synchronously on write).
    """
    mid: int

    # ─── Temporal bundle coordinates ─────────────────────────────────────
    time_base:  Tensor           # (d_time,)        point on B_time
    time_fiber: Tensor           # (d_F_time,)      fiber at time_base
    time_dirn:  Tensor           # (d_time,)        unit, for temporal DirectionTree

    # ─── Topic bundle coordinates ────────────────────────────────────────
    topic_base:  Tensor          # (d_topic,)       point on S^{d_topic - 1}, ||·||=1
    topic_fiber: Tensor          # (d_F_topic,)
    topic_dirn:  Tensor          # (d_topic,)       unit

    # ─── Context bundle coordinates ──────────────────────────────────────
    ctx_base:  Tensor            # (d_ctx,)
    ctx_fiber: Tensor            # (d_F_ctx,)
    ctx_dirn:  Tensor            # (d_ctx,)

    # ─── Scalars (unchanged from v3.46) ──────────────────────────────────
    surprise: float
    ts: float
    last: float
    cnt: int = 0
    version: int = 0

    # ─── Text + token identity (raw — small enough) ──────────────────────
    source_text: str = ""
    content_token_ids: List[int] = field(default_factory=list)
    rare_keyword_ids: List[int] = field(default_factory=list)

    # ─── Compressed large fields (§1.1, §6 invariant 2) ──────────────────
    kakeya_handle: KakeyaHandle = field(default_factory=KakeyaHandle)

    def device(self) -> torch.device:
        """Return the device the coordinate tensors live on.

        All three (base, fiber, dirn) triples must share the same device.
        """
        return self.time_base.device

    def assert_no_raw_large_fields(self, d_LLM: int, compression_min_dim: int) -> None:
        """§6 invariant 2: no raw d_LLM-sized tensor on self.

        Implementation note: this is defensive — MemEntry has no fields
        typed as Tensor at or above d_LLM dim other than what bundles store
        (which are all << d_LLM by design). If a subclass or serialization
        path ever restores a raw semantic_emb here, this check raises.
        """
        for name, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                if val.numel() >= compression_min_dim and val.shape[-1] >= compression_min_dim:
                    # Allow the three base/fiber/dirn tensors only (they are small by Cfg)
                    if name in {
                        "time_base", "time_fiber", "time_dirn",
                        "topic_base", "topic_fiber", "topic_dirn",
                        "ctx_base", "ctx_fiber", "ctx_dirn",
                    }:
                        continue
                    raise AssertionError(
                        f"MemEntry.{name} has shape {tuple(val.shape)} with last dim >= "
                        f"{compression_min_dim}; must be compressed into kakeya_handle."
                    )
