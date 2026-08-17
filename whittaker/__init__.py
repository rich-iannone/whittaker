"""Whittaker: a Generalized Additive Model (GAM) library for Python."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from whittaker.bam import BigGAM
from whittaker.datasets import list_datasets, load_dataset
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
    BetaLS,
    Binomial,
    CoxPH,
    Family,
    GAMLSSFamily,
    Gamma,
    GammaLS,
    Gaussian,
    GaussianLS,
    InverseGaussian,
    Multinomial,
    NegativeBinomial,
    OrderedCategorical,
    Poisson,
    QuantileFamily,
    Tweedie,
    TweedieEstimated,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
    tw,
)
from whittaker.formula import Formula, InteractionTerm, LinearTerm, OffsetTerm, SmoothTerm
from whittaker.formula import parse as parse_formula
from whittaker.functional import CoefficientFunction, FunctionalGAM, FunctionalTerm
from whittaker.gam import GAM, GamCheckResult, PredictionResult, TermsPredictionResult
from whittaker.gamlss import GAMLSS, GAMLSSPrediction
from whittaker.io import from_mgcv_dict, load_gam, save_gam, to_mgcv_dict
from whittaker.model_matrix import ModelMatrix, SmoothInfo, build_model_matrix, predict_matrix
from whittaker.multi_response import MultiResponseGAM, MultiResponseResult, ResidualCorrelation
from whittaker.polars_streaming import PolarsGAM
from whittaker.quantile_gam import QuantileGAM, QuantileGAMResult
from whittaker.sklearn import GAMClassifier, GAMRegressor
from whittaker.smooths import (
    CRS,
    TPRS,
    AdaptiveTPRS,
    ConvexPSpline,
    CyclicCRS,
    CyclicPSpline,
    DuchonSpline,
    FactorSmoothBasis,
    GaussianProcess,
    MonotonePSpline,
    MRFBasis,
    PSpline,
    RandomEffectBasis,
    ShrinkageCRS,
    ShrinkageTPRS,
    SmoothBasis,
    SoapFilm,
    TensorInteractionBasis,
    TensorProductBasis,
    TensorProductBasisT2,
)
from whittaker.streaming import StreamingGAM, StreamingSnapshot

try:
    __version__: str = version("whittaker")
except PackageNotFoundError:  # pragma: no cover - only hit when whittaker isn't installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # Core model
    "GAM",
    "PredictionResult",
    "TermsPredictionResult",
    "GamCheckResult",
    # Formula
    "Formula",
    "SmoothTerm",
    "LinearTerm",
    "InteractionTerm",
    "OffsetTerm",
    "parse_formula",
    # Response families
    "Family",
    "Gaussian",
    "Poisson",
    "Binomial",
    "Gamma",
    "NegativeBinomial",
    "Beta",
    "Tweedie",
    "TweedieEstimated",
    "tw",
    "InverseGaussian",
    "CoxPH",
    "OrderedCategorical",
    "Multinomial",
    # Distributional families (GAMLSS)
    "GAMLSSFamily",
    "GaussianLS",
    "GammaLS",
    "BetaLS",
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
    # Distributional regression
    "GAMLSS",
    "GAMLSSPrediction",
    # Smooth basis types
    "SmoothBasis",
    "TPRS",
    "CRS",
    "PSpline",
    "CyclicCRS",
    "CyclicPSpline",
    "ShrinkageTPRS",
    "ShrinkageCRS",
    "DuchonSpline",
    "GaussianProcess",
    "SoapFilm",
    "MRFBasis",
    "AdaptiveTPRS",
    "RandomEffectBasis",
    "FactorSmoothBasis",
    "TensorProductBasis",
    "TensorInteractionBasis",
    "TensorProductBasisT2",
    # Shape-constrained smooths
    "MonotonePSpline",
    "ConvexPSpline",
    # Quantile regression
    "QuantileGAM",
    "QuantileGAMResult",
    "QuantileFamily",
    "calibrate_sigma",
    # Conformal prediction
    "conformal_fit",
    "conformal_coverage",
    "ConformalPredictor",
    "ConformalResult",
    # Causal inference
    "CausalGAM",
    "TreatmentEffect",
    "CATEResult",
    "mediation_analysis",
    "MediationResult",
    # Streaming and online GAMs
    "StreamingGAM",
    "StreamingSnapshot",
    # Multi-response GAMs
    "MultiResponseGAM",
    "MultiResponseResult",
    "ResidualCorrelation",
    # Functional regression
    "FunctionalGAM",
    "FunctionalTerm",
    "CoefficientFunction",
    # Large datasets
    "BigGAM",
    "PolarsGAM",
    "DuckDBGAM",
    # Cross-validation
    "cross_validate",
    "CVResult",
    # scikit-learn integration
    "GAMRegressor",
    "GAMClassifier",
    # Serialization
    "save_gam",
    "load_gam",
    "from_mgcv_dict",
    "to_mgcv_dict",
    # Model matrix
    "build_model_matrix",
    "predict_matrix",
    "ModelMatrix",
    "SmoothInfo",
    # Datasets
    "load_dataset",
    "list_datasets",
]
