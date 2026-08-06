"""Negative Binomial (NB2) family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class NegativeBinomial(Family):
    """Negative Binomial family with log link (NB2 parameterization).

    The NB2 variance function is V(μ) = μ + μ²/θ, where θ controls overdispersion. As θ → ∞ the
    distribution converges to Poisson.

    Parameters
    ----------
    theta:
        Initial overdispersion parameter (must be positive). Estimated during fitting via outer
        iteration unless held fixed.

    For the Negative Binomial family:

    - Link: g(μ) = log(μ)
    - Variance function: V(μ) = μ + μ²/θ
    - Deviance: 2 Σ [y log(y/μ) − (y + θ) log((y + θ)/(μ + θ))]
    """

    def __init__(self, theta: float = 1.0) -> None:
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}.")
        self._theta = float(theta)

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"theta must be positive, got {value}.")
        self._theta = float(value)

    def link(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        return mu_c + mu_c**2 / self._theta

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        return float(np.sum(self.unit_deviance(y, mu)))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        theta = self._theta
        dev = np.empty_like(y, dtype=float)
        pos = y > 0
        dev[pos] = y[pos] * np.log(y[pos] / mu_c[pos])
        dev[~pos] = 0.0
        dev -= (y + theta) * np.log((y + theta) / (mu_c + theta))
        return 2.0 * dev

    def log_likelihood(self, y: NDArray, mu: NDArray, scale: float) -> float:
        from scipy.special import gammaln

        mu_c = np.maximum(mu, _EPS)
        theta = self._theta
        ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1.0)
            + theta * np.log(theta / (mu_c + theta))
            + y * np.log(mu_c / (mu_c + theta))
        )
        return float(np.sum(ll))

    @property
    def scale_known(self) -> bool:
        return True

    def initialize(self, y: NDArray) -> NDArray:
        return np.maximum(y, 0.1) + 0.1

    def __repr__(self) -> str:
        return f"NegativeBinomial(theta={self._theta:.4g}, link='log')"
