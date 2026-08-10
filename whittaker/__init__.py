"""Whittaker: a Generalized Additive Model (GAM) library for Python."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from whittaker.bam import BigGAM
from whittaker.calibration import calibrate_sigma
from whittaker.cross_validation import CVResult, cross_validate
from whittaker.families import Beta, Binomial, Family, Gamma, Gaussian, NegativeBinomial, Poisson
from whittaker.formula import Formula, InteractionTerm, LinearTerm, OffsetTerm, SmoothTerm
from whittaker.formula import parse as parse_formula
from whittaker.gam import GAM
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix
from whittaker.sklearn import GAMClassifier, GAMRegressor
from whittaker.smooths import CRS, TPRS, GaussianProcess, PSpline, SmoothBasis, SoapFilm

try:
    __version__: str = version("whittaker")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "BigGAM",
    "calibrate_sigma",
    "cross_validate",
    "CVResult",
    "Beta",
    "Binomial",
    "CRS",
    "Family",
    "Formula",
    "GAM",
    "GAMClassifier",
    "GAMRegressor",
    "Gamma",
    "Gaussian",
    "GaussianProcess",
    "NegativeBinomial",
    "Poisson",
    "ModelMatrix",
    "PSpline",
    "InteractionTerm",
    "LinearTerm",
    "OffsetTerm",
    "SmoothBasis",
    "SmoothTerm",
    "SoapFilm",
    "TPRS",
    "build_model_matrix",
    "parse_formula",
    "predict_matrix",
]
