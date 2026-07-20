"""Smooth basis constructors for Whittaker GAMs."""

from __future__ import annotations

from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.tprs import TPRS

__all__ = [
    "SmoothBasis",
    "TPRS",
]
