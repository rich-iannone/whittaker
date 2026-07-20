"""Whittaker: a next-generation Generalized Additive Model (GAM) library for Python."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from whittaker.formula import Formula, InteractionTerm, LinearTerm, OffsetTerm, SmoothTerm
from whittaker.formula import parse as parse_formula
from whittaker.smooths import CRS, TPRS, SmoothBasis

try:
    __version__: str = version("whittaker")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "CRS",
    "Formula",
    "InteractionTerm",
    "LinearTerm",
    "OffsetTerm",
    "SmoothBasis",
    "SmoothTerm",
    "TPRS",
    "parse_formula",
]
