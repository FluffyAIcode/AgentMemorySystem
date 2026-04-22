"""KakeyaSet — a single Kakeya-like skeleton bound to one bundle."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from ams_v4.core.config import Cfg4
from ams_v4.core.mem_entry import CompressedVec
from ams_v4.core.types import Tensor
from ams_v4.kakeya.alignment import (
    alignment_error, project_into_pca, solve_aligned_t_dir,
)


@dataclass
class KakeyaSkeleton4:
    basis: Tensor         # (d_eff, d_field)
    mean: Tensor          # (d_field,)
    t_dir: Tensor         # (d_eff,)
    centers: Tensor       # (K, d_eff)
    d_eff: int
    K: int
    d_res: int


def _compute_pca(vecs: Tensor, variance_ratio: float) -> tuple:
    """PCA. Ported from kakeya_codec.py::KakeyaCodec._compute_pca.

    vecs: (N, d_field)
    Returns: (basis: (d_eff, d_field), mean: (d_field,), d_eff: int)
    """
    mu = vecs.mean(0)
    centered = vecs - mu.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    cumvar = S.pow(2).cumsum(0) / S.pow(2).sum().clamp(min=1e-12)
    d_eff_arr = (cumvar >= variance_ratio).nonzero(as_tuple=True)[0]
    d_eff = (int(d_eff_arr[0].item()) + 1) if len(d_eff_arr) > 0 else len(S)
    d_eff = max(d_eff, 2)
    d_eff = min(d_eff, Vh.shape[0])
    basis = Vh[:d_eff]
    return basis, mu, d_eff


def _spherical_kmeans(dirs: Tensor, K: int, max_iter: int = 100):
    """Farthest-first init + iterative spherical k-means.

    Ported from kakeya_codec.py::KakeyaCodec._spherical_kmeans.

    dirs: (N, d) unit vectors.
    Returns: (centers: (K_eff, d), assignments: (N,) long)
    """
    N, d = dirs.shape
    K = min(K, N)
    if K <= 1:
        return dirs[:1].clone(), torch.zeros(N, dtype=torch.long, device=dirs.device)
    centers = [dirs[0].clone()]
    for _ in range(K - 1):
        sims = torch.stack([dirs @ c for c in centers], dim=1)
        max_sim = sims.max(dim=1)[0]
        farthest = max_sim.argmin()
        centers.append(dirs[farthest].clone())
    centers = torch.stack(centers)
    assignments = torch.zeros(N, dtype=torch.long, device=dirs.device)
    for _ in range(max_iter):
        sims = dirs @ centers.T
        new_assign = sims.argmax(dim=1)
        if (new_assign == assignments).all():
            break
        assignments = new_assign
        for k in range(K):
            mask = assignments == k
            if mask.any():
                centers[k] = F.normalize(dirs[mask].mean(0), dim=0, eps=1e-8)
            else:
                far = (dirs @ centers.T).max(1)[0].argmin()
                centers[k] = dirs[far].clone()
                assignments[far] = k
    return centers, assignments


class KakeyaSet:
    """A single Kakeya set. Compresses one or more memory fields (concatenated
    along last dim), bound to exactly one owner bundle via alignment.
    """

    def __init__(self, set_idx: int, owner_bundle_name: str,
                 compressed_fields: List[str], cfg: Cfg4):
        self.set_idx = set_idx
        self.owner_bundle_name = owner_bundle_name
        self.compressed_fields = list(compressed_fields)
        self.cfg = cfg

        self.skeleton: Optional[KakeyaSkeleton4] = None
        self._n_encoded: int = 0

        assert owner_bundle_name in ("time", "topic", "ctx")
        assert len(compressed_fields) >= 1

    @property
    def is_active(self) -> bool:
        return self.skeleton is not None

    # ─── Build ──────────────────────────────────────────────────────────

    def build(self, vecs: Tensor, bundle_axis_in_field: Tensor) -> None:
        """Build the skeleton from stacked field vectors + a bundle-axis
        direction in field space (d_field,).
        """
        assert vecs.dim() == 2, f"vecs must be (N, d_field), got {tuple(vecs.shape)}"
        assert bundle_axis_in_field.dim() == 1
        assert bundle_axis_in_field.shape[0] == vecs.shape[1]

        # 1. PCA
        basis, mean, d_eff = _compute_pca(vecs, self.cfg.kakeya_variance_ratio)

        # 2. Pushforward axis into PCA subspace, solve for aligned t_dir
        target_in_pca = project_into_pca(bundle_axis_in_field, basis)
        t_dir, _err = solve_aligned_t_dir(target_in_pca, self.cfg.kakeya_alignment_tol)

        # 3. Spherical K-means on perpendicular components
        coeffs = (vecs - mean.unsqueeze(0)) @ basis.T  # (N, d_eff)
        alphas = coeffs @ t_dir                         # (N,)
        perp = coeffs - alphas.unsqueeze(-1) * t_dir.unsqueeze(0)  # (N, d_eff)
        perp_norms = perp.norm(dim=-1)
        valid_mask = perp_norms > 1e-8
        if int(valid_mask.sum().item()) >= 2:
            perp_dirs = F.normalize(perp[valid_mask], dim=-1)
            K_actual = min(self.cfg.kakeya_K, perp_dirs.shape[0])
            centers, _ = _spherical_kmeans(perp_dirs, K_actual)
        else:
            centers = F.normalize(torch.randn(1, d_eff, device=vecs.device), dim=-1)
            K_actual = 1

        d_res = min(self.cfg.kakeya_d_res, d_eff)

        self.skeleton = KakeyaSkeleton4(
            basis=basis, mean=mean, t_dir=t_dir, centers=centers,
            d_eff=d_eff, K=K_actual, d_res=d_res,
        )

    # ─── Encode / decode ────────────────────────────────────────────────

    def encode(self, v: Tensor) -> CompressedVec:
        """v: (d_field,) → CompressedVec."""
        assert self.skeleton is not None, "KakeyaSet.encode called before build"
        skel = self.skeleton
        assert v.dim() == 1 and v.shape[0] == skel.basis.shape[1], \
            f"expected v shape ({skel.basis.shape[1]},), got {tuple(v.shape)}"

        coeff = (v - skel.mean) @ skel.basis.T   # (d_eff,)
        alpha = float((coeff @ skel.t_dir).item())
        perp = coeff - alpha * skel.t_dir
        perp_norm = perp.norm()
        if perp_norm.item() > 1e-8:
            perp_dir = perp / perp_norm
            sims = skel.centers @ perp_dir
            seg_id = int(sims.argmax().item())
        else:
            seg_id = 0
        t = float((perp @ skel.centers[seg_id]).item())
        residual = perp - t * skel.centers[seg_id]   # (d_eff,)
        if skel.d_res < skel.d_eff:
            _, top_idx = residual.abs().topk(skel.d_res)
            r_vals = residual[top_idx]
        else:
            top_idx = torch.arange(skel.d_eff, device=v.device)
            r_vals = residual
        self._n_encoded += 1
        return CompressedVec(
            set_idx=self.set_idx,
            seg_id=seg_id,
            alpha=alpha,
            t=t,
            residual_vals=r_vals.detach().cpu(),
            residual_idx=top_idx.detach().cpu(),
        )

    def decode(self, cv: CompressedVec, device: torch.device) -> Tensor:
        """cv → (d_field,) reconstructed tensor."""
        assert self.skeleton is not None, "KakeyaSet.decode called before build"
        skel = self.skeleton
        residual = torch.zeros(skel.d_eff, device=device, dtype=skel.basis.dtype)
        idx = cv.residual_idx.to(device)
        vals = cv.residual_vals.to(device=device, dtype=skel.basis.dtype)
        residual[idx] = vals
        perp_approx = cv.t * skel.centers[cv.seg_id].to(device) + residual
        coeff_approx = cv.alpha * skel.t_dir.to(device) + perp_approx
        v_approx = coeff_approx @ skel.basis.to(device) + skel.mean.to(device)
        return v_approx

    # ─── Alignment ──────────────────────────────────────────────────────

    def verify_alignment(self, bundle_axis_in_field: Tensor) -> float:
        """Return alignment error. 0 is perfect; should be ≤ cfg.kakeya_alignment_tol."""
        assert self.skeleton is not None
        target_in_pca = project_into_pca(bundle_axis_in_field, self.skeleton.basis)
        return alignment_error(self.skeleton.t_dir, target_in_pca)
