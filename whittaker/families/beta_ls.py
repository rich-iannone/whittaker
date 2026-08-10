"""Beta GAMLSS family (mean-precision parameterisation)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, digamma, expit, logit, polygamma

from whittaker.families.gamlss_base import GAMLSSFamily


class BetaLS(GAMLSSFamily):
    """Beta family for GAMLSS with mean-precision parameterisation.

    Models `y` in (0, 1) with mean `mu` in (0, 1) and precision `phi > 0`. Uses logit link for mu
    and log link for phi.

    If `a = mu * phi` and `b = (1 - mu) * phi` then `y ~ Beta(a, b)`.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu", "phi")

    def link(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return logit(values)
        return np.log(values)

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        if param == "mu":
            return expit(eta)
        return np.exp(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return 1.0 / (values * (1.0 - values))
        return 1.0 / values

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = params["mu"]
        phi = params["phi"]
        a = mu * phi
        b = (1.0 - mu) * phi
        y_star = np.log(y / (1.0 - y))
        mu_star = digamma(a) - digamma(b)
        if param == "mu":
            return phi * (y_star - mu_star)
        return mu * (y_star - mu_star) + digamma(phi) - digamma(b) + np.log(1.0 - y)

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = params["mu"]
        phi = params["phi"]
        a = mu * phi
        b = (1.0 - mu) * phi
        if param == "mu":
            return phi**2 * (polygamma(1, a) + polygamma(1, b))
        return mu**2 * polygamma(1, a) + (1.0 - mu) ** 2 * polygamma(1, b) - polygamma(1, phi)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        mu = params["mu"]
        phi = params["phi"]
        a = mu * phi
        b = (1.0 - mu) * phi
        ll_i = -betaln(a, b) + (a - 1.0) * np.log(y) + (b - 1.0) * np.log(1.0 - y)
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        y_safe = np.clip(y, 0.01, 0.99)
        mu_init = y_safe.copy()
        y_mean = np.mean(y_safe)
        y_var = np.var(y_safe, ddof=1)
        phi_est = y_mean * (1.0 - y_mean) / max(y_var, 1e-6) - 1.0
        phi_init = np.full_like(y_safe, max(phi_est, 1.0))
        return {"mu": mu_init, "phi": phi_init}

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        mu = params["mu"]
        phi = params["phi"]
        a = mu * phi
        b = (1.0 - mu) * phi
        return rng.beta(a, b)

    def __repr__(self) -> str:
        return "BetaLS(mu=logit, phi=log)"
