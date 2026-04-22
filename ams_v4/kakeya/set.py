"""KakeyaSet — a single Kakeya-like skeleton bound to one bundle.

Structure (inherits the shape from kakeya_codec.py::KakeyaCodec but generalized):

  skeleton = (basis ∈ R^{d_eff × d_field},     # PCA basis for the compressed field
              mean ∈ R^{d_field},              # PCA mean
              t_dir ∈ R^{d_eff},               # distinguished direction, aligned to
                                                # owner_bundle.canonical_axis (§1.3)
              centers ∈ R^{K × d_eff})         # segment centers on the perp sphere

  encoding(v) = CompressedVec(
      seg_id,      # argmax over centers of v's perp component
      alpha,       # v's projection onto t_dir
      t,           # v's projection onto centers[seg_id]
      residual)    # sparse top-k of the remainder

The bundle alignment constraint in §1.3 says: t_dir must equal the
push-forward of owner_bundle.canonical_axis into the basis subspace, up to
alignment_tol. This is what makes these sets "linked on the fiber bundles" —
the kakeya axis is the bundle axis.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch

from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import CompressedVec
from ams_v4.core.types import Tensor


@dataclass
class KakeyaSkeleton4:
    basis: Tensor         # (d_eff, d_field)
    mean: Tensor          # (d_field,)
    t_dir: Tensor         # (d_eff,) — aligned to owner_bundle.canonical_axis
    centers: Tensor       # (K, d_eff)
    d_eff: int
    K: int
    d_res: int


class KakeyaSet:
    """A single Kakeya set. Compresses one or more memory fields, bound to
    exactly one owner bundle via the alignment constraint.
    """

    def __init__(self, set_idx: int, owner_bundle_name: str,
                 compressed_fields: List[str], cfg: Cfg4):
        self.set_idx = set_idx
        self.owner_bundle_name = owner_bundle_name   # "time" | "topic" | "ctx"
        self.compressed_fields = list(compressed_fields)
        self.cfg = cfg

        # Populated by `build`. Before build: skeleton is None, set is inactive.
        self.skeleton: Optional[KakeyaSkeleton4] = None
        self._n_encoded: int = 0

        assert owner_bundle_name in ("time", "topic", "ctx"), \
            f"owner_bundle_name must be time/topic/ctx, got {owner_bundle_name}"
        assert len(compressed_fields) >= 1, "a KakeyaSet must compress at least one field"

    @property
    def is_active(self) -> bool:
        return self.skeleton is not None

    def build(self, vecs: Tensor, bundle_axis_pushforward: Tensor) -> None:
        """Build the skeleton from a stack of field vectors + the bundle-axis
        pushforward direction.

        vecs: (N, d_field) stacked vectors (for whichever field this set owns —
              when a set owns multiple fields, they are concatenated first).
        bundle_axis_pushforward: (d_eff_target,) — the direction in the PCA
              subspace that the bundle's canonical axis maps to. This is what
              t_dir will be constrained to equal (up to alignment_tol).

        Implementation notes (for v4.3):
          1. Run PCA on vecs → (basis, mean, d_eff).
          2. Solve for t_dir that minimizes
                ||t_dir - project_pca(bundle_axis_pushforward)||^2
             subject to ||t_dir|| = 1. Closed form: just normalize the projection.
          3. Spherical K-means on the perpendicular component of coeffs w.r.t.
             t_dir → (K, d_eff) centers.
          4. Store skeleton.
        """
        raise NotImplementedError("v4-skel: KakeyaSet.build — lands in v4.3")

    def encode(self, v: Tensor) -> CompressedVec:
        """Encode a single field vector.

        v: (d_field,) tensor. Returns CompressedVec with this set's set_idx.
        """
        raise NotImplementedError("v4-skel: KakeyaSet.encode — lands in v4.3")

    def decode(self, cv: CompressedVec, device: torch.device) -> Tensor:
        """Decode a CompressedVec back into (d_field,) on given device."""
        raise NotImplementedError("v4-skel: KakeyaSet.decode — lands in v4.3")

    def verify_alignment(self, bundle_axis_pushforward: Tensor) -> float:
        """Return the alignment error — ||t_dir - proj(bundle_axis)||₂.

        Must be ≤ cfg.kakeya_alignment_tol to satisfy §6 invariant 4.
        """
        raise NotImplementedError("v4-skel: KakeyaSet.verify_alignment — lands in v4.3")
