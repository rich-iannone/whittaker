"""Inverse Gaussian family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class InverseGaussian(Family):
    """Inverse Gaussian family with log link.

    For the Inverse Gaussian family:

    - Link: g(μ) = log(μ)
    - Variance function: V(μ) = μ³
    - Deviance: Σ (y − μ)² / (μ² y)

    This is the Tweedie special case with variance power p = 3.
    """

    def link(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        return np.maximum(mu, _EPS) ** 3

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        return (y_c - mu_c) ** 2 / (mu_c**2 * y_c)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        ll_i = (
            -0.5 * np.log(2.0 * np.pi * scale * y_c**3)
            - (y_c - mu_c) ** 2 / (2.0 * scale * mu_c**2 * y_c)
        )
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return False

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        lam = mu_c / scale
        n = len(mu_c)
        v = rng.standard_normal(n) ** 2
        x = mu_c + (mu_c**2 * v) / (2.0 * lam) - mu_c / (2.0 * lam) * np.sqrt(
            4.0 * mu_c * lam * v + mu_c**2 * v**2
        )
        u = rng.uniform(size=n)
        result = np.where(u <= mu_c / (mu_c + x), x, mu_c**2 / x)
        return np.maximum(result, _EPS)

    def initialize(self, y: NDArray) -> NDArray:
        return np.maximum(y, _EPS) + 0.1

    def __repr__(self) -> str:
        return "InverseGaussian(link='log')"
