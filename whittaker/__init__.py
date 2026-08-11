"""Whittaker: a Generalized Additive Model (GAM) library for Python."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from whittaker.bam import BigGAM
from whittaker.calibration import calibrate_sigma
from whittaker.causal import (
    CATEResult,
    CausalGAM,
    MediationResult,
    TreatmentEffect,
    mediation_analysis,
)
from whittaker.conformal import (
    ConformalPredictor,
    ConformalResult,
    conformal_coverage,
    conformal_fit,
)
from whittaker.cross_validation import CVResult, cross_validate
from whittaker.duckdb import DuckDBGAM
from whittaker.families import (
    Beta,
    Binomial,
    CoxPH,
    Family,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    tw,
)
from whittaker.formula import Formula, InteractionTerm, LinearTerm, OffsetTerm, SmoothTerm
from whittaker.formula import parse as parse_formula
from whittaker.gam import GAM
from whittaker.io import from_mgcv_dict, load_gam, save_gam, to_mgcv_dict
from whittaker.model_matrix import ModelMatrix, build_model_matrix, predict_matrix
from whittaker.polars_streaming import PolarsGAM
from whittaker.quantile_gam import QuantileGAM
from whittaker.sklearn import GAMClassifier, GAMRegressor
from whittaker.smooths import (
    CRS,
    TPRS,
    DuchonSpline,
    GaussianProcess,
    MRFBasis,
    PSpline,
    SmoothBasis,
    SoapFilm,
)

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
    "CausalGAM",
    "CATEResult",
    "ConformalPredictor",
    "ConformalResult",
    "conformal_coverage",
    "conformal_fit",
    "CoxPH",
    "CRS",
    "DuckDBGAM",
    "DuchonSpline",
    "Family",
    "Formula",
    "GAM",
    "GAMClassifier",
    "GAMRegressor",
    "Gamma",
    "Gaussian",
    "GaussianProcess",
    "MediationResult",
    "mediation_analysis",
    "MRFBasis",
    "NegativeBinomial",
    "PolarsGAM",
    "Poisson",
    "QuantileGAM",
    "ModelMatrix",
    "PSpline",
    "InteractionTerm",
    "LinearTerm",
    "OffsetTerm",
    "SmoothBasis",
    "SmoothTerm",
    "SoapFilm",
    "TreatmentEffect",
    "TPRS",
    "tw",
    "build_model_matrix",
    "from_mgcv_dict",
    "load_gam",
    "parse_formula",
    "predict_matrix",
    "save_gam",
    "to_mgcv_dict",
]
