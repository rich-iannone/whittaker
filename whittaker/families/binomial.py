"""Binomial family with logit link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Binomial(Family):
    """Binomial family with logit (canonical) link.

    For the Binomial family:

    - Link: g(μ) = log(μ / (1 − μ)) (logit)
    - Variance function: V(μ) = μ(1 − μ)
    - Deviance: 2 Σ [y log(y/μ) + (1−y) log((1−y)/(1−μ))]
    """

    def link(self, mu: NDArray) -> NDArray:
        return logit(mu)

    def link_inverse(self, eta: NDArray) -> NDArray:
        return expit(eta)

    def link_derivative(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return 1.0 / (mu_c * (1.0 - mu_c))

    def variance(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return mu_c * (1.0 - mu_c)

    def deviance(self, y: NDArray, mu: NDArray) -> float:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        dev = np.zeros_like(y)
        pos = y > 0
        neg = y < 1
        dev[pos] = y[pos] * np.log(y[pos] / mu_c[pos])
        dev[neg] += (1.0 - y[neg]) * np.log((1.0 - y[neg]) / (1.0 - mu_c[neg]))
        return float(2.0 * np.sum(dev))

    def log_likelihood(self, y: NDArray, mu: NDArray, scale: float) -> float:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        ll = y * np.log(mu_c) + (1.0 - y) * np.log(1.0 - mu_c)
        return float(np.sum(ll))

    @property
    def scale_known(self) -> bool:
        return True

    def initialize(self, y: NDArray) -> NDArray:
        return np.full_like(y, fill_value=(np.mean(y) + 0.5) / 2.0)

    def __repr__(self) -> str:
        return "Binomial(link='logit')"
