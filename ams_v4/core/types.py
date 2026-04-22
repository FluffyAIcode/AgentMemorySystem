"""Shared type aliases and shape-tag conventions.

We use plain Python type aliases rather than a runtime-checked tensor library
because v3.46 code has no such dependency and v4 must stay drop-in compatible
with PyTorch ≥ 2.0 without extra imports.

Shape tags are documentation-only (encoded in docstrings), but the helper
`check_shape` enforces them when `Cfg4.strict_shape_checks = True`.
"""
from __future__ import annotations
from typing import Tuple, Optional
import torch

Tensor = torch.Tensor

ShapeTag = Tuple[Optional[int], ...]


def check_shape(t: Tensor, expected: ShapeTag, name: str) -> None:
    """Assert t has a shape compatible with expected (None = any).

    Raises AssertionError with a clear message if shape mismatches. Cost is
    one Python-level tuple compare per call; negligible in the training path,
    free to leave enabled in debug builds.
    """
    if len(t.shape) != len(expected):
        raise AssertionError(
            f"{name}: rank mismatch. got {tuple(t.shape)}, expected {expected}"
        )
    for i, (got, exp) in enumerate(zip(t.shape, expected)):
        if exp is not None and got != exp:
            raise AssertionError(
                f"{name}: dim {i} mismatch. got {tuple(t.shape)}, expected {expected}"
            )
