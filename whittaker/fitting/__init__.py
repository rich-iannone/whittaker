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
from whittaker.fitting.pirls import FitResult, pirls_fit

__all__ = [
    "AnovaModelRow",
    "AnovaResult",
    "ConcurvityResult",
    "FitResult",
    "ParametricTestResult",
    "SmoothTestResult",
    "anova_gam",
    "concurvity",
    "parametric_tests",
    "pirls_fit",
    "smooth_tests",
]
