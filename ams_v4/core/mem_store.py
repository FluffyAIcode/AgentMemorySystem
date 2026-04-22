"""MemStore — owns MemEntries, routes to per-bundle DirectionTrees.

Each bundle has its own DirectionTree (unlike v3.46 which had a single global
tree keyed on `dirn`). This lets bundle-specific retrieval (topic-side / time-
side / context-side) run independently and later merge in CrossBundleAttention.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import MemEntry
from ams_v4.core.types import Tensor


@dataclass
class _Node:
    leaf: bool = True
    ids: List[int] = field(default_factory=list)
    children: List["_Node"] = field(default_factory=list)
    centers: Optional[Tensor] = None
    depth: int = 0


class DirectionTreeV4:
    """Bundle-local direction tree. Indexed on one of (time|topic|ctx)_dirn.

    Structurally identical to v3.46's DirectionTree, but scoped to one bundle
    and retrieval ranking does not mix in cross-bundle rerank terms (those
    live in CrossBundleAttention).
    """

    def __init__(self, cfg: Cfg4, bundle_name: str, store: "MemStore"):
        self.cfg = cfg
        self.bundle_name = bundle_name  # "time" | "topic" | "ctx"
        self._store = store
        self.root = _Node()
        assert bundle_name in ("time", "topic", "ctx"), \
            f"bundle_name must be one of time/topic/ctx, got {bundle_name}"

    def insert(self, mid: int) -> None:
        raise NotImplementedError("v4-skel: DirectionTreeV4.insert — lands in v4.1")

    def remove(self, mid: int) -> None:
        raise NotImplementedError("v4-skel: DirectionTreeV4.remove — lands in v4.1")

    def retrieve(self, qdir: Tensor, beam: int) -> List[Tuple[int, float]]:
        """Beam-retrieve mids by cosine of (mid's dirn for this bundle, qdir).

        Returns: list of (mid, score) sorted by -score then mid asc.
        """
        raise NotImplementedError("v4-skel: DirectionTreeV4.retrieve — lands in v4.1")

    def rebuild(self) -> None:
        raise NotImplementedError("v4-skel: DirectionTreeV4.rebuild — lands in v4.1")

    def _dirn_of(self, entry: MemEntry) -> Tensor:
        """Pick the per-bundle dirn field off a MemEntry. Helper for impl."""
        return {
            "time":  entry.time_dirn,
            "topic": entry.topic_dirn,
            "ctx":   entry.ctx_dirn,
        }[self.bundle_name]


class MemStore:
    """Global memory store. One dict, three DirectionTrees.

    Invariants (§6):
      - verify_consistency() checks all six invariants at once.
      - assert_all_large_fields_compressed() is its own method for targeted
        tests.
    """

    def __init__(self, cfg: Cfg4):
        self.cfg = cfg
        self._entries: Dict[int, MemEntry] = {}
        self._next_mid: int = 0
        self.tree_time = DirectionTreeV4(cfg, "time", self)
        self.tree_topic = DirectionTreeV4(cfg, "topic", self)
        self.tree_ctx = DirectionTreeV4(cfg, "ctx", self)

    # ─── Public surface ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, mid: int) -> bool:
        return mid in self._entries

    def get(self, mid: int) -> Optional[MemEntry]:
        return self._entries.get(mid)

    def all_mids(self) -> List[int]:
        return sorted(self._entries.keys())

    def all_entries(self) -> List[MemEntry]:
        return [self._entries[m] for m in self.all_mids()]

    def add(self, entry: MemEntry) -> int:
        """Insert a new entry. Assigns mid. Routes into all three trees.

        Returns the assigned mid.
        """
        raise NotImplementedError("v4-skel: MemStore.add — lands in v4.1")

    def remove(self, mid: int) -> None:
        raise NotImplementedError("v4-skel: MemStore.remove — lands in v4.1")

    # ─── Invariant checks (§6) ────────────────────────────────────────────

    def verify_consistency(self) -> List[str]:
        """Run all §6 invariants. Returns list of error messages (empty = ok)."""
        raise NotImplementedError("v4-skel: MemStore.verify_consistency — lands in v4.1")

    def assert_all_large_fields_compressed(self) -> None:
        """§6 invariant 2: no raw d_LLM-sized tensor on any MemEntry.

        Delegates to MemEntry.assert_no_raw_large_fields; runs over every entry.
        """
        for e in self._entries.values():
            e.assert_no_raw_large_fields(
                d_LLM=self.cfg.d_LLM,
                compression_min_dim=self.cfg.compression_min_dim,
            )
