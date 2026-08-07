"""Poisson family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Poisson(Family):
    """Poisson family with log (canonical) link.

    For the Poisson family:

    - Link: g(μ) = log(μ)
    - Variance function: V(μ) = μ
    - Deviance: 2 Σ [y log(y/μ) − (y − μ)]
    """

    def link(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        return np.maximum(mu, _EPS)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        dev = np.empty_like(y)
        pos = y > 0
        dev[pos] = y[pos] * np.log(y[pos] / mu_c[pos])
        dev[~pos] = 0.0
        dev -= y - mu_c
        return 2.0 * dev

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        from scipy.special import gammaln

        mu_c = np.maximum(mu, _EPS)
        ll_i = y * np.log(mu_c) - mu_c - gammaln(y + 1.0)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return True

    def initialize(self, y: NDArray) -> NDArray:
        return np.maximum(y, 0.1) + 0.1

    def __repr__(self) -> str:
        return "Poisson(link='log')"
