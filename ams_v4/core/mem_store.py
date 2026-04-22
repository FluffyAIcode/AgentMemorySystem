"""MemStore — owns MemEntries, routes to per-bundle DirectionTrees.

Ports DirectionTree from scheme_b_v344.py (lines 1189-1378), with three
deliberate simplifications:

  1. No cross-coupling to AMM / content_classifier / wte_normed. The v3.46
     rerank-inside-retrieve path was a workaround for missing axes; in v4
     the bundle geometry IS the axes, so the tree does plain beam-retrieve
     on its bundle's dirn.
  2. No cluster-crowding. Same reason.
  3. One tree per bundle (three total), each indexed on its bundle's dirn.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

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
    """Bundle-local direction tree on unit-vector dirns.

    Insert is recursive down the best-matching child; split by farthest-first
    k-means when a leaf exceeds tree_max_leaf. Retrieve is beam search over
    child centers.
    """

    def __init__(self, cfg: Cfg4, bundle_name: str, store: "MemStore"):
        assert bundle_name in ("time", "topic", "ctx")
        self.cfg = cfg
        self.bundle_name = bundle_name
        self._store = store
        self.root = _Node()

    # ─── Insert ──────────────────────────────────────────────────────────

    def insert(self, mid: int) -> None:
        entry = self._store.get(mid)
        if entry is None:
            return
        self._ins(self.root, mid)

    def _dirn_of(self, mid: int) -> Tensor:
        e = self._store.get(mid)
        if e is None:
            raise KeyError(f"mid {mid} not in store")
        return {
            "time":  e.time_dirn,
            "topic": e.topic_dirn,
            "ctx":   e.ctx_dirn,
        }[self.bundle_name]

    def _ins(self, nd: _Node, mid: int) -> None:
        if nd.leaf:
            nd.ids.append(mid)
            if len(nd.ids) > self.cfg.tree_max_leaf:
                self._split(nd)
        else:
            d = self._dirn_of(mid)
            best = self._best(nd, d)
            self._ins(nd.children[best], mid)
            self._update_centers(nd)

    def _split(self, nd: _Node) -> None:
        ids = nd.ids
        if len(ids) < 2:
            return
        K = min(self.cfg.tree_K, len(ids))
        if K < 2:
            return
        dirs = torch.stack([self._dirn_of(i) for i in ids])
        centered = dirs - dirs.mean(0)
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        except Exception:
            return
        n_comp = min(K, dirs.shape[1])
        proj = centered @ Vh[:n_comp].T
        asgn = self._farthest_kmeans(proj, K)
        children: List[_Node] = []
        for k in range(K):
            ch = _Node(depth=nd.depth + 1)
            ch.ids = [ids[i] for i in range(len(ids)) if asgn[i] == k]
            if ch.ids:
                children.append(ch)
        if len(children) <= 1:
            return
        nd.leaf = False
        nd.children = children
        nd.ids = []
        self._update_centers(nd)
        for ch in nd.children:
            if ch.leaf and len(ch.ids) > self.cfg.tree_max_leaf:
                self._split(ch)

    @staticmethod
    def _farthest_kmeans(data: Tensor, K: int, max_iter: int = 50) -> Tensor:
        N = data.shape[0]
        K = min(K, N)
        if K <= 0:
            return torch.zeros(N, dtype=torch.long, device=data.device)
        ctrs = [data[0].clone()]
        for _ in range(K - 1):
            d2 = torch.cdist(data, torch.stack(ctrs)).min(1)[0].pow(2)
            ctrs.append(data[d2.argmax()].clone())
        ctrs = torch.stack(ctrs)
        asgn = torch.zeros(N, dtype=torch.long, device=data.device)
        for _ in range(max_iter):
            dists = torch.cdist(data, ctrs)
            new = dists.argmin(1)
            if (new == asgn).all():
                break
            asgn = new
            for k in range(K):
                mk = asgn == k
                if mk.any():
                    ctrs[k] = data[mk].mean(0)
                else:
                    far = dists.min(1)[0].argmax()
                    ctrs[k] = data[far].clone()
                    asgn[far] = k
        return asgn

    def _best(self, nd: _Node, d: Tensor) -> int:
        if nd.centers is None or len(nd.children) == 0:
            return 0
        return int((nd.centers @ d).argmax().item())

    # ─── Retrieve ────────────────────────────────────────────────────────

    def retrieve(self, qdir: Tensor, beam: int) -> List[Tuple[int, float]]:
        """Beam-retrieve mids by cosine of (mid's dirn, qdir). Returns sorted
        by -score then mid asc (deterministic tie-break).
        """
        beams: List[Tuple[_Node, float]] = [(self.root, 0.0)]
        results: Dict[int, float] = {}
        while beams:
            nb: List[Tuple[_Node, float]] = []
            for nd, sc in beams:
                if nd.leaf:
                    for mid in nd.ids:
                        if mid in self._store:
                            s = float((qdir @ self._dirn_of(mid)).item()) + sc
                            if mid not in results or s > results[mid]:
                                results[mid] = s
                elif nd.centers is not None:
                    sims = nd.centers @ qdir
                    tk = min(beam, len(nd.children))
                    _, idxs = sims.topk(tk)
                    for i in idxs:
                        nb.append((nd.children[int(i.item())], sc + float(sims[int(i.item())].item())))
                else:
                    for ch in nd.children:
                        nb.append((ch, sc))
            nb.sort(key=lambda x: -x[1])
            beams = nb[:beam]
        return sorted(results.items(), key=lambda x: (-x[1], x[0]))

    # ─── Remove / rebuild ────────────────────────────────────────────────

    def remove(self, mid: int) -> None:
        self._rm(self.root, mid)
        self._rebalance(self.root)

    def _rm(self, nd: _Node, mid: int) -> bool:
        if nd.leaf:
            if mid in nd.ids:
                nd.ids.remove(mid)
                return True
            return False
        return any(self._rm(c, mid) for c in nd.children)

    def _rebalance(self, nd: _Node) -> None:
        if nd.leaf:
            return
        for c in nd.children:
            self._rebalance(c)
        nd.children = [c for c in nd.children if self._count(c) > 0]
        if not nd.children:
            nd.leaf = True
            nd.ids = []
            nd.centers = None
        elif len(nd.children) == 1:
            ch = nd.children[0]
            nd.leaf = ch.leaf
            nd.ids = ch.ids
            nd.children = ch.children
            nd.centers = ch.centers
        else:
            self._update_centers(nd)

    def _count(self, nd: _Node) -> int:
        return len(nd.ids) if nd.leaf else sum(self._count(c) for c in nd.children)

    def _update_centers(self, nd: _Node) -> None:
        cs: List[Tensor] = []
        for c in nd.children:
            ids = self._collect(c)
            dirs = [self._dirn_of(i) for i in ids if i in self._store]
            if not dirs:
                continue
            cs.append(F.normalize(torch.stack(dirs).mean(0), dim=0))
        nd.centers = torch.stack(cs) if cs else None

    def _collect(self, nd: _Node) -> List[int]:
        if nd.leaf:
            return list(nd.ids)
        return [i for c in nd.children for i in self._collect(c)]

    def rebuild(self) -> None:
        mids = [m for m in self._store.all_mids()]
        self.root = _Node()
        for m in mids:
            self.insert(m)

    # ─── Diagnostics ─────────────────────────────────────────────────────

    def size(self) -> int:
        return self._count(self.root)

    def verify(self) -> List[str]:
        errs = []
        tree_mids = set(self._collect(self.root))
        store_mids = set(self._store.all_mids())
        if tree_mids != store_mids:
            errs.append(
                f"tree_{self.bundle_name} ≠ store: "
                f"tree_only={tree_mids - store_mids}, store_only={store_mids - tree_mids}"
            )
        return errs


# ─── MemStore ─────────────────────────────────────────────────────────────

class MemStore:
    """Global memory store. One dict, three DirectionTreeV4s.

    Invariants (§6, asserted by verify_consistency):
      1. Every MemEntry has three (base, fiber, dirn) triples.
      2. No raw d_LLM-sized tensor on any MemEntry.
      3. KakeyaRegistry has ≥ 2 active sets when n ≥ kakeya_min_entries (checked
         only when a registry is passed into verify_consistency).
      4. Kakeya t_dir alignment ≤ alignment_tol (registry check).
      5. Kakeya decode(encode(v)) within reconstruction_tol (registry check).
      6. CrossBundleAttention output shape = (L_mem, d_LLM) — not a store concern.
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
        mid = self._next_mid
        self._next_mid += 1
        entry.mid = mid
        self._entries[mid] = entry
        self.tree_time.insert(mid)
        self.tree_topic.insert(mid)
        self.tree_ctx.insert(mid)
        return mid

    def remove(self, mid: int) -> None:
        if mid not in self._entries:
            return
        self.tree_time.remove(mid)
        self.tree_topic.remove(mid)
        self.tree_ctx.remove(mid)
        del self._entries[mid]

    # ─── Invariant checks (§6) ────────────────────────────────────────────

    def verify_consistency(self, registry: Optional[object] = None) -> List[str]:
        errs: List[str] = []
        # Invariant 1: every entry has all three triples
        for mid, e in self._entries.items():
            for attr in ("time_base", "time_fiber", "time_dirn",
                         "topic_base", "topic_fiber", "topic_dirn",
                         "ctx_base", "ctx_fiber", "ctx_dirn"):
                v = getattr(e, attr, None)
                if not isinstance(v, torch.Tensor):
                    errs.append(f"invariant 1: mem {mid}.{attr} is not a Tensor "
                                f"(got {type(v).__name__})")
        # Invariant 2: no raw large fields
        try:
            self.assert_all_large_fields_compressed()
        except AssertionError as ex:
            errs.append(f"invariant 2: {ex}")

        # Tree consistency
        errs += self.tree_time.verify()
        errs += self.tree_topic.verify()
        errs += self.tree_ctx.verify()

        # Invariants 3 and 4 require a registry
        if registry is not None and hasattr(registry, "verify_invariants"):
            errs += list(registry.verify_invariants(len(self)))
        return errs

    def assert_all_large_fields_compressed(self) -> None:
        for e in self._entries.values():
            e.assert_no_raw_large_fields(
                d_LLM=self.cfg.d_LLM,
                compression_min_dim=self.cfg.compression_min_dim,
            )
