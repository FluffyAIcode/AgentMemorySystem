"""Kakeya ↔ bundle-axis alignment helpers.

All pure-function math; no state. KakeyaSet and KakeyaRegistry call these.
"""
from __future__ import annotations
from typing import Tuple

import torch

from ams_v4.core.types import Tensor


def pushforward(axis_in_base: Tensor, base_to_field: Tensor) -> Tensor:
    """Pushforward a bundle base-space axis into field space.

    axis_in_base:   (d_base,)
    base_to_field:  (d_base, d_field)
    Returns:        (d_field,) — not normalized.
    """
    assert axis_in_base.dim() == 1
    assert base_to_field.dim() == 2
    assert axis_in_base.shape[0] == base_to_field.shape[0]
    return axis_in_base @ base_to_field


def project_into_pca(direction_in_field: Tensor, basis: Tensor) -> Tensor:
    """Project a field-space direction onto the PCA subspace.

    direction_in_field: (d_field,)
    basis:              (d_eff, d_field)  (rows are PCA basis vectors)
    Returns:            (d_eff,)
    """
    assert direction_in_field.dim() == 1
    assert basis.dim() == 2
    assert direction_in_field.shape[0] == basis.shape[1]
    return basis @ direction_in_field


def alignment_error(t_dir: Tensor, target: Tensor) -> float:
    """Return ||t_dir - normalize(target)||₂.

    Both are (d_eff,). target is normalized before comparison.
    """
    assert t_dir.dim() == 1 and target.dim() == 1
    assert t_dir.shape == target.shape
    target_n = target / target.norm().clamp(min=1e-8)
    return float((t_dir - target_n).norm().item())


def solve_aligned_t_dir(target_direction: Tensor, tol: float = 1e-3) -> Tuple[Tensor, float]:
    """Pick t_dir ∈ unit sphere in R^{d_eff} to minimize distance to target.

    v4.3: closed form — just normalize(target). Returns (t_dir, err).
    err is ||t_dir - normalize(target)|| = 0 by construction, unless target is
    near-zero (in which case we return an arbitrary unit vector with error 1).
    """
    norm = target_direction.norm()
    if norm.item() < 1e-8:
        fallback = torch.zeros_like(target_direction)
        fallback[0] = 1.0
        return fallback, 1.0
    t_dir = target_direction / norm
    return t_dir, 0.0
