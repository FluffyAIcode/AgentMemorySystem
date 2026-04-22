"""Kakeya ↔ bundle-axis alignment helpers.

The §1.3 contract: each KakeyaSet.t_dir must equal the push-forward of its
owner bundle's canonical axis into the PCA subspace, up to alignment_tol.

This file holds the math — separated from KakeyaSet / KakeyaRegistry so the
algebra is reviewable independent of dataclass / indexing plumbing.
"""
from __future__ import annotations
from typing import Tuple

import torch

from ams_v4.core.types import Tensor


def pushforward(axis_in_base: Tensor, base_to_field: Tensor) -> Tensor:
    """Pushforward a bundle base-space axis into the compressed-field space.

    axis_in_base:   (d_base,) unit vector in the bundle's base space.
    base_to_field:  (d_base, d_field) a learned or fixed linear map from the
                    bundle base space to the field space the Kakeya set
                    operates on (e.g. semantic_emb lives in d_LLM = 1536;
                    d_base for TemporalBundle is 8, so base_to_field is
                    (8, 1536)).

    Returns: (d_field,) the image, *not* yet normalized (normalize at the
    caller if you need ||·||=1).

    In v4.3 this is the rectangular matmul `axis_in_base @ base_to_field`.
    """
    raise NotImplementedError("v4-skel: alignment.pushforward — lands in v4.3")


def project_into_pca(direction_in_field: Tensor, basis: Tensor) -> Tensor:
    """Project a direction in field space onto the PCA subspace.

    direction_in_field: (d_field,)
    basis: (d_eff, d_field)  (rows are the PCA basis vectors)

    Returns: (d_eff,) coefficient vector; NOT normalized.
    """
    raise NotImplementedError("v4-skel: alignment.project_into_pca — lands in v4.3")


def alignment_error(t_dir: Tensor, target: Tensor) -> float:
    """Return ||t_dir - target / ||target||||₂.

    Both inputs live in the PCA subspace (dim d_eff). Target is normalized
    before comparison.
    """
    raise NotImplementedError("v4-skel: alignment.alignment_error — lands in v4.3")


def solve_aligned_t_dir(coeffs: Tensor, target_direction: Tensor,
                        tol: float) -> Tuple[Tensor, float]:
    """Pick t_dir ∈ the unit sphere in R^{d_eff} to be as close as possible
    to target_direction while still being a direction that concentrates the
    coeffs (has appreciable projection magnitude).

    In the simplest v4.3 implementation this is just `normalize(target)`
    (constrained to the unit sphere; minimizes the alignment error by
    construction). Future work: balance alignment against "captures most of
    the variance of coeffs".

    Returns (t_dir, alignment_error).
    """
    raise NotImplementedError("v4-skel: alignment.solve_aligned_t_dir — lands in v4.3")
