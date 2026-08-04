"""Gaussian (Normal) family with identity link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family


class Gaussian(Family):
    """Gaussian family with identity link.

    For the Gaussian family:

    - Link: g(μ) = μ (identity)
    - Variance function: V(μ) = 1
    - Deviance: Σ(y − μ)²
    - Scale parameter φ = σ² (estimated from residuals)
    """

    def link(self, mu: NDArray) -> NDArray:
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        return float(np.sum((y - mu) ** 2))

    def log_likelihood(self, y: NDArray, mu: NDArray, scale: float) -> float:
        n = len(y)
        return float(-0.5 * n * np.log(2 * np.pi * scale) - 0.5 * np.sum((y - mu) ** 2) / scale)

    def initialize(self, y: NDArray) -> NDArray:
        return y.copy()

    def __repr__(self) -> str:
        return "Gaussian(link='identity')"
