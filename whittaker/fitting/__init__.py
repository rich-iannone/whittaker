"""GAM fitting algorithms."""

from __future__ import annotations

from whittaker.fitting.inference import ParametricTestResult, SmoothTestResult, parametric_tests, smooth_tests
from whittaker.fitting.pirls import FitResult, pirls_fit

__all__ = [
    "FitResult",
    "ParametricTestResult",
    "SmoothTestResult",
    "parametric_tests",
    "pirls_fit",
    "smooth_tests",
]
