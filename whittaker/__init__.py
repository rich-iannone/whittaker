"""Whittaker: a Generalized Additive Model (GAM) library for Python."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from whittaker.families import Binomial, Family, Gamma, Gaussian, Poisson
from whittaker.formula import Formula, InteractionTerm, LinearTerm, OffsetTerm, SmoothTerm
from whittaker.formula import parse as parse_formula
from whittaker.gam import GAM
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix
from whittaker.smooths import CRS, TPRS, PSpline, SmoothBasis

try:
    __version__: str = version("whittaker")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "Binomial",
    "CRS",
    "Family",
    "Formula",
    "GAM",
    "Gamma",
    "Gaussian",
    "Poisson",
    "ModelMatrix",
    "PSpline",
    "InteractionTerm",
    "LinearTerm",
    "OffsetTerm",
    "SmoothBasis",
    "SmoothTerm",
    "TPRS",
    "build_model_matrix",
    "parse_formula",
    "predict_matrix",
]
