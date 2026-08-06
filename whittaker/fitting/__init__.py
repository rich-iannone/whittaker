"""GAM fitting algorithms."""

from __future__ import annotations

from whittaker.fitting.inference import SmoothTestResult, smooth_tests
from whittaker.fitting.pirls import FitResult, pirls_fit

__all__ = [
    "FitResult",
    "SmoothTestResult",
    "pirls_fit",
    "smooth_tests",
]
