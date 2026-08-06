"""Smooth basis constructors for Whittaker GAMs."""

from __future__ import annotations

from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.cubic import CRS
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.tensor import TensorInteractionBasis, TensorProductBasis
from whittaker.smooths.tprs import TPRS

__all__ = [
    "CRS",
    "PSpline",
    "SmoothBasis",
    "TensorInteractionBasis",
    "TensorProductBasis",
    "TPRS",
]
