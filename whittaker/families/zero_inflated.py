r"""Zero-inflated Poisson (ZIP) and zero-inflated negative binomial (ZINB) GAMLSS families."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, gammaln, logit

from whittaker.families.gamlss_base import GAMLSSFamily

_EPS = np.finfo(float).eps
_MU_FLOOR = 1e-10


class ZeroInflatedPoisson(GAMLSSFamily):
    r"""Zero-inflated Poisson (ZIP) family for GAMLSS.

    Count data often has more zeros than a plain `Poisson` model can explain — for example,
    when some observations are structurally incapable of the event occurring at all (a
    "never-taker" always reports zero), in addition to the zeros that arise simply because the
    Poisson mean is low. `ZeroInflatedPoisson` models this as a mixture: with probability `pi`
    an observation is a structural zero, and with probability `1 - pi` it is drawn from an
    ordinary `Poisson(mu)` distribution (which can itself still produce a zero). Both `mu` and
    `pi` are modeled as smooth functions of covariates through `GAMLSS`, so the excess-zero
    probability and the count intensity can each vary independently across the covariate space.
    If overdispersion remains even among the non-structural-zero counts, use
    `ZeroInflatedNegativeBinomial` instead.

    Parameters
    ----------
    None
        `ZeroInflatedPoisson` takes no constructor arguments; both `mu` and `pi` are modeled
        entirely through the formulas supplied to `GAMLSS`.

    Notes
    -----
    Two distributional parameters are modeled, each with its own link:

    $$
    g_{\mu}(\mu) = \log(\mu), \qquad g_{\pi}(\pi) = \log\!\left(\frac{\pi}{1-\pi}\right).
    $$

    The probability mass function is a mixture of a point mass at zero and a Poisson
    distribution:

    $$
    P(Y = 0) = \pi + (1-\pi) e^{-\mu}, \qquad
    P(Y = k) = (1-\pi) \frac{\mu^{k} e^{-\mu}}{k!} \quad \text{for } k > 0.
    $$

    Examples
    --------
    Fit a GAMLSS for count data with excess zeros:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 + 0.4 * np.sin(x))
    pi = expit(-1.0 + 0.8 * np.cos(x))

    is_structural_zero = rng.uniform(size=n) < pi
    counts = rng.poisson(mu)
    y = np.where(is_structural_zero, 0, counts).astype(float)

    data = {"x": x, "y": y}

    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "pi": "y ~ s(x)"},
        family=wk.ZeroInflatedPoisson(),
    )
    model.fit(data)
    print(model.summary())
    ```
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
    r"""Zero-inflated negative binomial (ZINB) family for GAMLSS.

    `ZeroInflatedNegativeBinomial` combines the two departures from plain `Poisson` counts that
    are most common in practice: overdispersion (variance exceeding the mean, as in
    `NegativeBinomial`) and structural excess zeros (as in `ZeroInflatedPoisson`). It is
    appropriate for count data where, even after allowing for a mixture of structural and
    sampling zeros, the remaining positive counts are still more variable than a Poisson model
    would predict — for example, healthcare utilization counts, insurance claim frequencies, or
    ecological abundance data with many true absences plus overdispersed non-zero counts. `mu`
    (the NB mean) and `pi` (the zero-inflation probability) are modeled as smooth functions of
    covariates through `GAMLSS`, while the overdispersion parameter `theta` is fixed at
    construction rather than estimated per observation.

    Parameters
    ----------
    theta : float, default=1.0
        Negative binomial size (overdispersion) parameter, must be positive. Larger values mean
        less overdispersion (`theta -> infinity` recovers `ZeroInflatedPoisson`); smaller values
        mean more overdispersion among the non-structural-zero counts. Unlike `mu` and `pi`,
        `theta` is a single fixed value shared across all observations rather than modeled by a
        smooth predictor.

    Notes
    -----
    Two distributional parameters are modeled, each with its own link:

    $$
    g_{\mu}(\mu) = \log(\mu), \qquad g_{\pi}(\pi) = \log\!\left(\frac{\pi}{1-\pi}\right).
    $$

    The probability mass function is a mixture of a point mass at zero and a Negative Binomial
    distribution using the mean-size parameterization (`Var(NB) = mu + mu^2/theta`):

    $$
    P(Y = 0) = \pi + (1-\pi)\, \mathrm{NB}(0 \mid \mu, \theta), \qquad
    P(Y = k) = (1-\pi)\, \mathrm{NB}(k \mid \mu, \theta) \quad \text{for } k > 0,
    $$

    where

    $$
    \mathrm{NB}(k \mid \mu, \theta) = \binom{k+\theta-1}{k}
    \left(\frac{\theta}{\theta+\mu}\right)^{\theta} \left(\frac{\mu}{\theta+\mu}\right)^{k}.
    $$

    Examples
    --------
    Fit a GAMLSS for overdispersed count data with excess zeros:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 + 0.4 * np.sin(x))
    pi = expit(-1.0 + 0.8 * np.cos(x))
    theta = 2.0

    is_structural_zero = rng.uniform(size=n) < pi
    counts = rng.negative_binomial(theta, theta / (theta + mu))
    y = np.where(is_structural_zero, 0, counts).astype(float)

    data = {"x": x, "y": y}

    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "pi": "y ~ s(x)"},
        family=wk.ZeroInflatedNegativeBinomial(theta=theta),
    )
    model.fit(data)
    print(model.summary())
    ```
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
