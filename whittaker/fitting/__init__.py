"""GAM fitting algorithms."""

from __future__ import annotations

from whittaker.fitting.inference import (
    AnovaModelRow,
    AnovaResult,
    ConcurvityResult,
    ParametricTestResult,
    SmoothTestResult,
    anova_gam,
    concurvity,
    parametric_tests,
    smooth_tests,
)
from whittaker.fitting.bam import DiscretizedModelMatrix, bam_fit
from whittaker.fitting.gamlss_fit import GAMLSSFitResult, GAMLSSParamResult, gamlss_fit
from whittaker.fitting.pirls import FitResult, pirls_fit

__all__ = [
    "AnovaModelRow",
    "AnovaResult",
    "ConcurvityResult",
    "DiscretizedModelMatrix",
    "FitResult",
    "GAMLSSFitResult",
    "GAMLSSParamResult",
    "ParametricTestResult",
    "SmoothTestResult",
    "bam_fit",
    "anova_gam",
    "concurvity",
    "gamlss_fit",
    "parametric_tests",
    "pirls_fit",
    "smooth_tests",
]
