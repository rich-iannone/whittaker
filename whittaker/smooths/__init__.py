"""Smooth basis constructors for Whittaker GAMs."""

from __future__ import annotations

from whittaker.smooths.adaptive import AdaptiveTPRS
from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.cubic import CRS
from whittaker.smooths.cyclic import CyclicCRS, CyclicPSpline
from whittaker.smooths.duchon import DuchonSpline
from whittaker.smooths.factor_smooth import FactorSmoothBasis
from whittaker.smooths.gp import GaussianProcess
from whittaker.smooths.monotone import ConvexPSpline, MonotonePSpline
from whittaker.smooths.mrf import MRFBasis
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.random import RandomEffectBasis
from whittaker.smooths.shrinkage import ShrinkageCRS, ShrinkageTPRS
from whittaker.smooths.soap_film import SoapFilm
from whittaker.smooths.tensor import (
    TensorInteractionBasis,
    TensorProductBasis,
    TensorProductBasisT2,
)
from whittaker.smooths.tprs import TPRS

__all__ = [
    "AdaptiveTPRS",
    "ConvexPSpline",
    "CRS",
    "CyclicCRS",
    "CyclicPSpline",
    "DuchonSpline",
    "FactorSmoothBasis",
    "GaussianProcess",
    "MonotonePSpline",
    "MRFBasis",
    "PSpline",
    "RandomEffectBasis",
    "ShrinkageCRS",
    "ShrinkageTPRS",
    "SmoothBasis",
    "SoapFilm",
    "TensorInteractionBasis",
    "TensorProductBasis",
    "TensorProductBasisT2",
    "TPRS",
]
