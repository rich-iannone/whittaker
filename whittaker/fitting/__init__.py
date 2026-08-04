"""GAM fitting algorithms."""

from __future__ import annotations

from whittaker.fitting.pirls import FitResult, pirls_fit

__all__ = [
    "FitResult",
    "pirls_fit",
]
