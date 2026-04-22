"""KakeyaRegistry — owns N KakeyaSet instances and routes fields to them.

This is the layer that makes the abstract "multiple kakeya sets" real.

Routing plan (§1.2 default):

  sets[0] = KakeyaSet(owner="time",  fields=["semantic_emb"])
  sets[1] = KakeyaSet(owner="topic", fields=["semantic_emb", "content_wte_mean"])
  sets[2] = KakeyaSet(owner="ctx",   fields=["context_descriptor"])
  sets[3] = KakeyaSet(owner="topic", fields=["content_wte_mean"])    # secondary topic set

Cross-axis redundancy ("semantic_emb" is in both set 0 and set 1) is
intentional: CrossBundleAttention reads both to reconstruct different
per-axis projections of the same underlying field. This is one of the two
ways the abstract spec's "multiple sets" bites — redundant encoding along
different distinguished directions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import CompressedVec, KakeyaHandle
from ams_v4.core.types import Tensor
from ams_v4.kakeya.set import KakeyaSet


class KakeyaRegistry:
    """Holds N KakeyaSet instances and manages encode/decode by (field, set_idx)."""

    def __init__(self, cfg: Cfg4):
        self.cfg = cfg
        self.sets: List[KakeyaSet] = []
        # Default routing: 4 sets as listed in this module docstring. Can be
        # overridden by calling `define_sets(...)` before first `build`.
        self._default_routing: List[Tuple[str, List[str]]] = [
            ("time",  ["semantic_emb"]),
            ("topic", ["semantic_emb", "content_wte_mean"]),
            ("ctx",   ["context_descriptor"]),
            ("topic", ["content_wte_mean"]),
        ]

    # ─── Configuration ───────────────────────────────────────────────────

    def define_sets(self, routing: List[Tuple[str, List[str]]]) -> None:
        """Install a custom routing before building. Each tuple is (bundle_name, fields).

        Raises if any bundle_name is not in {time, topic, ctx} or if routing
        is shorter than Cfg4.n_kakeya_sets - 1 (we require ≥ 2 sets; §1.1).
        """
        raise NotImplementedError("v4-skel: KakeyaRegistry.define_sets — lands in v4.3")

    # ─── Build / rebuild ──────────────────────────────────────────────────

    def build(self, field_corpus: Dict[str, Tensor],
              bundle_axes: Dict[str, Tensor]) -> None:
        """Build all sets from a corpus of stacked field vectors.

        field_corpus: {field_name -> (N, d_field) stacked vectors}
        bundle_axes:  {bundle_name -> (d_field,) canonical axis pushforward}

        For each configured (owner_bundle, fields) tuple in the routing,
        instantiate a KakeyaSet and call its `build`. The per-set input is
        concat-along-dim-1 of the fields, and the per-set axis is the
        bundle_axes[owner_bundle] projected into that concat layout.
        """
        raise NotImplementedError("v4-skel: KakeyaRegistry.build — lands in v4.3")

    def rebuild_if_needed(self, n_entries: int) -> bool:
        """Trigger rebuild if heuristic thresholds are crossed. Returns True if rebuilt."""
        raise NotImplementedError("v4-skel: KakeyaRegistry.rebuild_if_needed — lands in v4.3")

    # ─── Per-memory encode / decode ──────────────────────────────────────

    def encode_memory_fields(self, fields: Dict[str, Tensor]) -> KakeyaHandle:
        """Encode one memory's large fields into a KakeyaHandle.

        fields: {field_name -> (d_field,) raw vector}
        Returns a KakeyaHandle whose entries map to every (set_idx, field) that
        owns this field in the routing.
        """
        raise NotImplementedError("v4-skel: KakeyaRegistry.encode_memory_fields — lands in v4.3")

    def decode_field(self, handle: KakeyaHandle, field_name: str,
                     preferred_set_idx: Optional[int] = None,
                     device: Optional[torch.device] = None) -> Optional[Tensor]:
        """Reconstruct one field from a memory's handle.

        If preferred_set_idx is given, use that set. Otherwise pick the set
        listed first (smallest set_idx) that encodes this field.
        """
        raise NotImplementedError("v4-skel: KakeyaRegistry.decode_field — lands in v4.3")

    # ─── Invariants ──────────────────────────────────────────────────────

    def verify_invariants(self, n_entries: int) -> List[str]:
        """Checks §6 invariants 3 and 4 for the registry. Returns error list.

        3: at least 2 active sets when n_entries >= kakeya_min_entries.
        4: every active set has alignment error ≤ kakeya_alignment_tol.
        """
        raise NotImplementedError("v4-skel: KakeyaRegistry.verify_invariants — lands in v4.3")
