"""KakeyaRegistry — owns N KakeyaSet instances and routes fields to them."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import CompressedVec, KakeyaHandle
from ams_v4.core.types import Tensor
from ams_v4.kakeya.alignment import pushforward
from ams_v4.kakeya.set import KakeyaSet


class KakeyaRegistry:
    """Holds N KakeyaSet instances; manages encode/decode by (field, set_idx)."""

    def __init__(self, cfg: Cfg4):
        self.cfg = cfg
        self.sets: List[KakeyaSet] = []
        # Default routing: 4 sets (see ARCHITECTURE_v4.md §1.1).
        # Users can override with define_sets() before the first build().
        self._routing: List[Tuple[str, List[str]]] = [
            ("time",  ["semantic_emb"]),
            ("topic", ["semantic_emb", "content_wte_mean"]),
            ("ctx",   ["context_descriptor"]),
            ("topic", ["content_wte_mean"]),
        ]
        # Per-routing-key linear map B_bundle → R^{d_field_concat}.
        # Shape: (d_base, d_field_concat); produced at first build.
        self._base_to_field_maps: Dict[str, Tensor] = {}
        # Per-set: for a concatenated (field, ...) vector, which slice is which field?
        self._set_field_offsets: Dict[int, Dict[str, Tuple[int, int]]] = {}

    # ─── Configuration ───────────────────────────────────────────────────

    def define_sets(self, routing: List[Tuple[str, List[str]]]) -> None:
        """Install a custom routing. Must be called before build().

        Each tuple is (bundle_name, fields). Asserts:
          - routing length ≥ 2 (§1.1 / Cfg4.n_kakeya_sets >= 2)
          - bundle_name ∈ {time, topic, ctx}
          - every field in every tuple is a non-empty string
        """
        if self.sets and any(s.is_active for s in self.sets):
            raise RuntimeError("cannot redefine routing after sets have been built")
        assert len(routing) >= 2, (
            f"multiple-kakeya-sets invariant: routing must have ≥ 2 entries, "
            f"got {len(routing)}"
        )
        for i, (owner, fields) in enumerate(routing):
            assert owner in ("time", "topic", "ctx"), \
                f"routing[{i}] owner must be time/topic/ctx, got {owner}"
            assert len(fields) >= 1, f"routing[{i}] fields empty"
            for f in fields:
                assert isinstance(f, str) and f, f"routing[{i}] bad field {f!r}"
        self._routing = [(b, list(fs)) for b, fs in routing]
        self._base_to_field_maps.clear()
        self._set_field_offsets.clear()

    # ─── Build ──────────────────────────────────────────────────────────

    def build(self, field_corpus: Dict[str, Tensor],
              bundle_axes: Dict[str, Tensor]) -> None:
        """Build all sets.

        field_corpus: {field_name: (N, d_field)}  (same N across fields)
        bundle_axes:  {bundle_name: (d_base_bundle,)}  unit vectors
        """
        assert field_corpus, "field_corpus cannot be empty"
        assert set(bundle_axes.keys()) >= {"time", "topic", "ctx"}, \
            "bundle_axes must include time, topic, ctx"

        # Clear any prior sets
        self.sets = []
        self._base_to_field_maps.clear()
        self._set_field_offsets.clear()

        for set_idx, (owner, fields) in enumerate(self._routing):
            # Assemble concatenated input for this set
            vecs_list = []
            offsets: Dict[str, Tuple[int, int]] = {}
            cursor = 0
            missing = False
            for f in fields:
                if f not in field_corpus:
                    missing = True
                    break
                v = field_corpus[f]
                assert v.dim() == 2, f"field_corpus[{f!r}] must be (N, d), got {tuple(v.shape)}"
                offsets[f] = (cursor, cursor + v.shape[-1])
                cursor += v.shape[-1]
                vecs_list.append(v)
            if missing or not vecs_list:
                # Instantiate inactive set so set_idx stays aligned with routing
                kset = KakeyaSet(set_idx, owner, fields, self.cfg)
                self.sets.append(kset)
                continue
            # All fields must have same N
            Ns = {v.shape[0] for v in vecs_list}
            assert len(Ns) == 1, f"mismatched N across fields: {Ns}"
            vecs = torch.cat(vecs_list, dim=-1)   # (N, d_field_concat)
            d_field = vecs.shape[-1]

            # Look up / initialize base→field map for this (owner, fields) combo
            key = f"{owner}::{'+'.join(fields)}"
            d_base = {"time": self.cfg.d_time, "topic": self.cfg.d_topic,
                      "ctx": self.cfg.d_ctx}[owner]
            if key not in self._base_to_field_maps:
                g = torch.Generator(device=vecs.device)
                g.manual_seed(set_idx + 100)
                M = torch.randn(d_base, d_field, device=vecs.device,
                                dtype=vecs.dtype, generator=g) / math.sqrt(d_base)
                self._base_to_field_maps[key] = M
            M = self._base_to_field_maps[key]

            # Pushforward bundle axis into field space
            axis = bundle_axes[owner].to(device=vecs.device, dtype=vecs.dtype)
            axis_in_field = pushforward(axis, M)   # (d_field,)

            # Build the set
            kset = KakeyaSet(set_idx, owner, fields, self.cfg)
            kset.build(vecs, axis_in_field)
            self.sets.append(kset)
            self._set_field_offsets[set_idx] = offsets

    def rebuild_if_needed(self, n_entries: int) -> bool:
        """Placeholder: v4.3 returns False. Rebuild policy lands in v4.6."""
        return False

    # ─── Per-memory encode / decode ─────────────────────────────────────

    def encode_memory_fields(self, fields: Dict[str, Tensor]) -> KakeyaHandle:
        handle = KakeyaHandle()
        for kset in self.sets:
            if not kset.is_active:
                continue
            try:
                vec = torch.cat([fields[f] for f in kset.compressed_fields], dim=-1)
            except KeyError:
                continue
            cv = kset.encode(vec)
            for f in kset.compressed_fields:
                handle.entries.setdefault(f, []).append(cv)
        return handle

    def _field_offset_in_set(self, set_idx: int, field_name: str) -> Tuple[int, int]:
        offsets = self._set_field_offsets.get(set_idx, {})
        return offsets.get(field_name, (0, 0))

    def decode_field(self, handle: KakeyaHandle, field_name: str,
                     preferred_set_idx: Optional[int] = None,
                     device: Optional[torch.device] = None) -> Optional[Tensor]:
        if field_name not in handle.entries:
            return None
        cvs = handle.entries[field_name]
        if preferred_set_idx is not None:
            cvs = [cv for cv in cvs if cv.set_idx == preferred_set_idx]
        if not cvs:
            return None
        cv = cvs[0]
        kset = self.sets[cv.set_idx]
        dev = device or kset.skeleton.basis.device if kset.skeleton else torch.device("cpu")
        full = kset.decode(cv, dev)
        start, end = self._field_offset_in_set(cv.set_idx, field_name)
        if end > start:
            return full[start:end]
        return full

    # ─── Invariants ─────────────────────────────────────────────────────

    def verify_invariants(self, n_entries: int,
                          bundle_axes: Optional[Dict[str, Tensor]] = None,
                          ) -> List[str]:
        errs = []
        active = [s for s in self.sets if s.is_active]
        # §6 invariant 3: ≥ 2 active sets when n ≥ kakeya_min_entries
        if n_entries >= self.cfg.kakeya_min_entries and len(active) < 2:
            errs.append(
                f"invariant 3 (abstract multi-kakeya): active sets = {len(active)} "
                f"< 2 at n_entries = {n_entries}"
            )
        # §6 invariant 4: alignment ≤ tol — needs bundle_axes to recompute pushforward
        if bundle_axes is not None:
            for kset in active:
                key = f"{kset.owner_bundle_name}::{'+'.join(kset.compressed_fields)}"
                M = self._base_to_field_maps.get(key)
                if M is None:
                    errs.append(f"invariant 4: no base_to_field map for {key}")
                    continue
                axis = bundle_axes[kset.owner_bundle_name]
                axis_in_field = pushforward(
                    axis.to(device=M.device, dtype=M.dtype), M,
                )
                e = kset.verify_alignment(axis_in_field)
                if e > self.cfg.kakeya_alignment_tol:
                    errs.append(
                        f"invariant 4: set {kset.set_idx} ({key}) alignment err "
                        f"{e:.4e} > tol {self.cfg.kakeya_alignment_tol}"
                    )
        return errs
