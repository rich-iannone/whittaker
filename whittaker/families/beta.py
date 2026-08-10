"""Beta regression family with logit link and fixed precision."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, expit, logit

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Beta(Family):
    """Beta regression family with logit link.

    Models responses in (0, 1) as Beta-distributed with mean `mu` and fixed precision `phi`. The
    precision is estimated from the data as a scale parameter (`scale = 1/phi`).

    - Link: g(μ) = logit(μ)
    - Variance function: V(μ) = μ(1 − μ) / (1 + φ)
    - Deviance: 2 Σ [ℓ(y; y) − ℓ(y; μ)]

    Parameters
    ----------
    phi:
        Fixed precision parameter. If `None` (default), precision is estimated from the data via the
        scale parameter (`phi = 1/scale`).
    """

    def __init__(self, phi: float | None = None) -> None:
        self._phi = phi

    def link(self, mu: NDArray) -> NDArray:
        return logit(np.clip(mu, _EPS, 1.0 - _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return expit(eta)

    def link_derivative(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return 1.0 / (mu_c * (1.0 - mu_c))

    def variance(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return mu_c * (1.0 - mu_c)

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        y_c = np.clip(y, _EPS, 1.0 - _EPS)
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return 2.0 * (y_c * np.log(y_c / mu_c) + (1.0 - y_c) * np.log((1.0 - y_c) / (1.0 - mu_c)))

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        phi = self._phi if self._phi is not None else 1.0 / max(scale, _EPS)
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        y_c = np.clip(y, _EPS, 1.0 - _EPS)
        a = mu_c * phi
        b = (1.0 - mu_c) * phi
        ll_i = -betaln(a, b) + (a - 1.0) * np.log(y_c) + (b - 1.0) * np.log(1.0 - y_c)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return self._phi is not None

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        phi = self._phi if self._phi is not None else 1.0 / max(scale, _EPS)
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        a = mu_c * phi
        b = (1.0 - mu_c) * phi
        return rng.beta(a, b)

    def initialize(self, y: NDArray) -> NDArray:
        return np.clip(y, 0.01, 0.99)

    def __repr__(self) -> str:
        if self._phi is not None:
            return f"Beta(link='logit', phi={self._phi})"
        return "Beta(link='logit')"
