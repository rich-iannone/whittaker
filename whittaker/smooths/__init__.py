"""Smooth basis constructors for Whittaker GAMs."""

from __future__ import annotations

from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.cubic import CRS
from whittaker.smooths.cyclic import CyclicCRS, CyclicPSpline
from whittaker.smooths.factor_smooth import FactorSmoothBasis
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.random import RandomEffectBasis
from whittaker.smooths.shrinkage import ShrinkageCRS, ShrinkageTPRS
from whittaker.smooths.tensor import (
    TensorInteractionBasis,
    TensorProductBasis,
    TensorProductBasisT2,
)
from whittaker.smooths.tprs import TPRS

__all__ = [
    "CRS",
    "CyclicCRS",
    "CyclicPSpline",
    "FactorSmoothBasis",
    "PSpline",
    "RandomEffectBasis",
    "ShrinkageCRS",
    "ShrinkageTPRS",
    "SmoothBasis",
    "TensorInteractionBasis",
    "TensorProductBasis",
    "TensorProductBasisT2",
    "TPRS",
]
