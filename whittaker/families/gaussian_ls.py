"""Gaussian location-scale GAMLSS family."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.gamlss_base import GAMLSSFamily


class GaussianLS(GAMLSSFamily):
    """Gaussian location-scale family for GAMLSS.

    Models both the mean (mu) and the standard deviation (sigma) as functions of covariates. Uses
    identity link for mu and log link for sigma.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu", "sigma")

    def link(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return values
        return np.log(values)

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        if param == "mu":
            return eta
        return np.exp(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return np.ones_like(values)
        return 1.0 / values

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = params["mu"]
        sigma = params["sigma"]
        if param == "mu":
            return (y - mu) / sigma**2
        return -1.0 / sigma + (y - mu) ** 2 / sigma**3

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        sigma = params["sigma"]
        if param == "mu":
            return 1.0 / sigma**2
        return 2.0 / sigma**2

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        mu = params["mu"]
        sigma = params["sigma"]
        ll_i = -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mu) / sigma) ** 2
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        return {
            "mu": y.copy(),
            "sigma": np.full_like(y, np.std(y, ddof=1)),
        }

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        return rng.normal(params["mu"], params["sigma"])

    def __repr__(self) -> str:
        return "GaussianLS(mu=identity, sigma=log)"
