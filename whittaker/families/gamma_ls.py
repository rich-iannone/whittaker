"""Gamma location-scale GAMLSS family."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln, polygamma

from whittaker.families.gamlss_base import GAMLSSFamily

_ETA_MAX = 20.0
_MU_FLOOR = 1e-10
_SIGMA_FLOOR = 1e-4


class GammaLS(GAMLSSFamily):
    """Gamma location-scale family for GAMLSS.

    Parameterised by mean `mu > 0` and shape `sigma > 0` where `sigma = 1 / sqrt(shape)`, i.e. the
    coefficient of variation. Uses log link for both parameters.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu", "sigma")

    def link(self, param: str, values: NDArray) -> NDArray:
        return np.log(np.maximum(values, _MU_FLOOR if param == "mu" else _SIGMA_FLOOR))

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        eta_c = np.clip(eta, -_ETA_MAX, _ETA_MAX)
        return np.exp(eta_c)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        floor = _MU_FLOOR if param == "mu" else _SIGMA_FLOOR
        return 1.0 / np.maximum(values, floor)

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        sigma = np.maximum(params["sigma"], _SIGMA_FLOOR)
        alpha = 1.0 / sigma**2
        if param == "mu":
            return alpha * (y / mu - 1.0) / mu
        return (2.0 / sigma**3) * (
            digamma(alpha) - np.log(alpha) - np.log(np.maximum(y, _MU_FLOOR) / mu) - 1.0 + y / mu
        )

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        sigma = np.maximum(params["sigma"], _SIGMA_FLOOR)
        alpha = 1.0 / sigma**2
        if param == "mu":
            return alpha / mu**2
        return np.maximum((4.0 / sigma**4) * (polygamma(1, alpha) * alpha - 1.0), _SIGMA_FLOOR)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        sigma = np.maximum(params["sigma"], _SIGMA_FLOOR)
        alpha = 1.0 / sigma**2
        y_safe = np.maximum(y, _MU_FLOOR)
        ll_i = (
            alpha * np.log(alpha / mu)
            + (alpha - 1.0) * np.log(y_safe)
            - y_safe * alpha / mu
            - gammaln(alpha)
        )
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        y_safe = np.maximum(y, _MU_FLOOR)
        mu_init = y_safe.copy()
        cv = np.std(y_safe) / np.mean(y_safe)
        sigma_init = np.full_like(y_safe, max(cv, 0.1))
        return {"mu": mu_init, "sigma": sigma_init}

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        sigma = np.maximum(params["sigma"], _SIGMA_FLOOR)
        shape = 1.0 / sigma**2
        scale = mu * sigma**2
        return rng.gamma(shape=shape, scale=scale)

    def __repr__(self) -> str:
        return "GammaLS(mu=log, sigma=log)"
