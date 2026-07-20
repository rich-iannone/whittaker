"""Whittaker: a next-generation Generalized Additive Model (GAM) library for Python."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("whittaker")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
