"""Zero-inflated Poisson (ZIP) and zero-inflated negative binomial (ZINB) GAMLSS families."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, gammaln, logit

from whittaker.families.gamlss_base import GAMLSSFamily

_EPS = np.finfo(float).eps
_MU_FLOOR = 1e-10


class ZeroInflatedPoisson(GAMLSSFamily):
    """Zero-inflated Poisson (ZIP) family for GAMLSS.

    Models count data with excess zeros. Two distributional parameters:

    - `mu`: Poisson mean (log link, mu > 0)
    - `pi`: zero-inflation probability (logit link, 0 < pi < 1)

    The probability mass function is:

        P(Y=0) = pi + (1 - pi) * exp(-mu)
        P(Y=k) = (1 - pi) * Poisson(k; mu)   for k > 0
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu", "pi")

    def link(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return np.log(np.maximum(values, _MU_FLOOR))
        return logit(np.clip(values, _EPS, 1.0 - _EPS))

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        if param == "mu":
            return np.exp(np.clip(eta, -30.0, 30.0))
        return expit(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return 1.0 / np.maximum(values, _MU_FLOOR)
        v = np.clip(values, _EPS, 1.0 - _EPS)
        return 1.0 / (v * (1.0 - v))

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        exp_neg_mu = np.exp(-np.minimum(mu, 700.0))
        p0 = np.maximum(pi + (1.0 - pi) * exp_neg_mu, _EPS)

        if param == "mu":
            return np.where(
                y == 0,
                -(1.0 - pi) * exp_neg_mu / p0,
                y / mu - 1.0,
            )
        return np.where(
            y == 0,
            (1.0 - exp_neg_mu) / p0,
            -1.0 / (1.0 - pi),
        )

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        exp_neg_mu = np.exp(-np.minimum(mu, 700.0))
        p0 = np.maximum(pi + (1.0 - pi) * exp_neg_mu, _EPS)

        if param == "mu":
            w_zero = (1.0 - pi) * exp_neg_mu * (1.0 - (1.0 - pi) * exp_neg_mu / p0) / p0
            w_pos = 1.0 / mu
            return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)
        w_zero = ((1.0 - exp_neg_mu) / p0) ** 2
        w_pos = 1.0 / ((1.0 - pi) ** 2)
        return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        exp_neg_mu = np.exp(-np.minimum(mu, 700.0))
        ll_zero = np.log(np.maximum(pi + (1.0 - pi) * exp_neg_mu, _EPS))
        ll_pos = np.log(1.0 - pi) + y * np.log(mu) - mu - gammaln(y + 1.0)
        ll_i = np.where(y == 0, ll_zero, ll_pos)
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        zero_frac = float(np.mean(y == 0))
        y_pos = y[y > 0]
        mu_est = float(np.mean(y_pos)) if len(y_pos) > 0 else 1.0
        return {
            "mu": np.full_like(y, mu_est, dtype=float),
            "pi": np.full_like(y, np.clip(zero_frac * 0.5, 0.01, 0.5), dtype=float),
        }

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        is_structural_zero = rng.uniform(size=len(mu)) < pi
        counts = rng.poisson(mu)
        return np.where(is_structural_zero, 0, counts).astype(float)

    def __repr__(self) -> str:
        return "ZeroInflatedPoisson(mu=log, pi=logit)"


class ZeroInflatedNegativeBinomial(GAMLSSFamily):
    """Zero-inflated negative binomial (ZINB) family for GAMLSS.

    Models count data with excess zeros and overdispersion. Two distributional parameters modeled as
    smooth functions of covariates:

    - `mu`: NB mean (log link, mu > 0)
    - `pi`: zero-inflation probability (logit link, 0 < pi < 1)

    The overdispersion parameter `theta` is fixed at construction.

    The probability mass function is:

        P(Y=0) = pi + (1 - pi) * NB(0; mu, theta)
        P(Y=k) = (1 - pi) * NB(k; mu, theta)   for k > 0

    where NB uses the mean-size parameterisation: `Var = mu + mu^2/theta`.

    Parameters
    ----------
    theta:
        Negative binomial size (overdispersion) parameter. Must be positive. Larger values mean less
        overdispersion (theta -> inf gives Poisson).
    """

    def __init__(self, theta: float = 1.0) -> None:
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}.")
        self._theta = float(theta)

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu", "pi")

    def link(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return np.log(np.maximum(values, _MU_FLOOR))
        return logit(np.clip(values, _EPS, 1.0 - _EPS))

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        if param == "mu":
            return np.exp(np.clip(eta, -30.0, 30.0))
        return expit(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        if param == "mu":
            return 1.0 / np.maximum(values, _MU_FLOOR)
        v = np.clip(values, _EPS, 1.0 - _EPS)
        return 1.0 / (v * (1.0 - v))

    def _nb_log_pmf(self, k: NDArray, mu: NDArray) -> NDArray:
        theta = self._theta
        return (
            gammaln(k + theta)
            - gammaln(k + 1.0)
            - gammaln(theta)
            + theta * np.log(theta / (mu + theta))
            + k * np.log(mu / (mu + theta))
        )

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        theta = self._theta

        nb_p0 = np.exp(self._nb_log_pmf(np.zeros_like(y), mu))
        p0 = np.maximum(pi + (1.0 - pi) * nb_p0, _EPS)

        if param == "mu":
            dnb_dmu_at_0 = nb_p0 * (-theta / (mu + theta))
            dl_zero = (1.0 - pi) * dnb_dmu_at_0 / p0
            dl_pos = (y - mu) / (mu * (1.0 + mu / theta))
            return np.where(y == 0, dl_zero, dl_pos)

        dl_zero = (1.0 - nb_p0) / p0
        dl_pos = -1.0 / (1.0 - pi)
        return np.where(y == 0, dl_zero, dl_pos)

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        theta = self._theta

        nb_p0 = np.exp(self._nb_log_pmf(np.zeros_like(y), mu))
        p0 = np.maximum(pi + (1.0 - pi) * nb_p0, _EPS)

        if param == "mu":
            w_pos = 1.0 / (mu * (1.0 + mu / theta))
            dnb = nb_p0 * theta / (mu + theta)
            w_zero = (1.0 - pi) * dnb * (1.0 - (1.0 - pi) * dnb / p0) / p0
            return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)

        w_zero = ((1.0 - nb_p0) / p0) ** 2
        w_pos = 1.0 / ((1.0 - pi) ** 2)
        return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)

        nb_log_p0 = self._nb_log_pmf(np.zeros_like(y), mu)
        ll_zero = np.log(np.maximum(pi + (1.0 - pi) * np.exp(nb_log_p0), _EPS))
        ll_pos = np.log(1.0 - pi) + self._nb_log_pmf(y, mu)
        ll_i = np.where(y == 0, ll_zero, ll_pos)
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        zero_frac = float(np.mean(y == 0))
        y_pos = y[y > 0]
        mu_est = float(np.mean(y_pos)) if len(y_pos) > 0 else 1.0
        return {
            "mu": np.full_like(y, mu_est, dtype=float),
            "pi": np.full_like(y, np.clip(zero_frac * 0.5, 0.01, 0.5), dtype=float),
        }

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        theta = self._theta
        p = theta / (mu + theta)
        is_structural_zero = rng.uniform(size=len(mu)) < pi
        counts = rng.negative_binomial(theta, p)
        return np.where(is_structural_zero, 0, counts).astype(float)

    def __repr__(self) -> str:
        return f"ZeroInflatedNegativeBinomial(theta={self._theta:.4g}, mu=log, pi=logit)"
