"""GAM fitting algorithms."""

from __future__ import annotations

from whittaker.fitting.inference import (
    ConcurvityResult,
    ParametricTestResult,
    SmoothTestResult,
    concurvity,
    parametric_tests,
    smooth_tests,
)
from whittaker.fitting.pirls import FitResult, pirls_fit

__all__ = [
    "ConcurvityResult",
    "FitResult",
    "ParametricTestResult",
    "SmoothTestResult",
    "concurvity",
    "parametric_tests",
    "pirls_fit",
    "smooth_tests",
]
