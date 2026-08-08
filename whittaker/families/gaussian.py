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

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = (y - mu) ** 2
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        ll_i = -0.5 * np.log(2 * np.pi * scale) - 0.5 * (y - mu) ** 2 / scale
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        return rng.normal(mu, np.sqrt(scale))

    def initialize(self, y: NDArray) -> NDArray:
        return y.copy()

    def __repr__(self) -> str:
        return "Gaussian(link='identity')"
