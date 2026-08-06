"""Gamma family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Gamma(Family):
    """Gamma family with log (canonical) link.

    For the Gamma family:

    - Link: g(μ) = log(μ)
    - Variance function: V(μ) = μ²
    - Deviance: 2 Σ [-log(y/μ) + (y − μ)/μ]
    """

    def link(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        return np.maximum(mu, _EPS) ** 2

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        return float(2.0 * np.sum(-np.log(y_c / mu_c) + (y - mu_c) / mu_c))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        return 2.0 * (-np.log(y_c / mu_c) + (y - mu_c) / mu_c)

    def log_likelihood(self, y: NDArray, mu: NDArray, scale: float) -> float:
        from scipy.special import gammaln

        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        alpha = 1.0 / scale
        ll = (
            alpha * np.log(alpha)
            + (alpha - 1.0) * np.log(y_c)
            - alpha * y_c / mu_c
            - alpha * np.log(mu_c)
            - gammaln(alpha)
        )
        return float(np.sum(ll))

    @property
    def scale_known(self) -> bool:
        return False

    def initialize(self, y: NDArray) -> NDArray:
        return np.maximum(y, _EPS) + 0.1

    def __repr__(self) -> str:
        return "Gamma(link='log')"
