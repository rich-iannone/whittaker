"""Quantile regression family via the Extended Log-F (ELF) loss.

Implements the smooth quantile loss from Fasiolo et al. (2021) "Fast calibrated additive quantile
regression." The ELF loss is a smooth approximation to the pinball/check loss that fits within the
standard PIRLS framework via a custom IRLS update.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from whittaker.families.base import Family

_W_FLOOR = 1e-10


def _elf_loss(r: NDArray, tau: float, sigma: float) -> NDArray:
    x = -r / sigma
    softplus = np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -30, 20))))
    return tau * r + sigma * softplus


def _elf_d1(r: NDArray, tau: float, sigma: float) -> NDArray:
    return tau - 1.0 + expit(r / sigma)


def _elf_d2(r: NDArray, sigma: float) -> NDArray:
    s = expit(r / sigma)
    return (1.0 / sigma) * s * (1.0 - s)


class QuantileFamily(Family):
    """Quantile regression via the Extended Log-F (ELF) pseudo-family.

    Parameters
    ----------
    tau:
        Quantile level in `(0, 1)`. Default `0.5` (median regression).
    sigma:
        Bandwidth/learning rate controlling the smoothness of the loss approximation. Smaller values
        give a sharper approximation to the pinball loss. The default is `1.0`.
    """

    def __init__(self, tau: float = 0.5, sigma: float = 1.0) -> None:
        if not 0 < tau < 1:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self._tau = tau
        self._sigma = sigma

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"sigma must be positive, got {value}")
        self._sigma = value

    def link(self, mu: NDArray) -> NDArray:
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray]:
        r = y - mu
        score = _elf_d1(r, self._tau, self._sigma)
        W = np.maximum(_elf_d2(r, self._sigma), _W_FLOOR)
        z = eta + score / W
        return z, W

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = 2.0 * _elf_loss(y - mu, self._tau, self._sigma)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        return 2.0 * _elf_loss(y - mu, self._tau, self._sigma)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        ll_i = -_elf_loss(y - mu, self._tau, self._sigma)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        u = rng.uniform(size=mu.shape)
        return np.where(
            u < self._tau,
            mu + self._sigma * np.log(u / self._tau),
            mu - self._sigma * np.log((1 - u) / (1 - self._tau)),
        )

    def initialize(self, y: NDArray) -> NDArray:
        return y.copy()

    def __repr__(self) -> str:
        return f"Quantile(tau={self._tau}, sigma={self._sigma})"
