r"""Gamma location-scale GAMLSS family."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln, polygamma

from whittaker.families.gamlss_base import GAMLSSFamily

_ETA_MAX = 20.0
_MU_FLOOR = 1e-10
_SIGMA_FLOOR = 1e-4


class GammaLS(GAMLSSFamily):
    r"""Gamma location-scale family for GAMLSS.

    `GammaLS` extends the plain `Gamma` family by letting both the mean `mu` and the coefficient
    of variation `sigma` vary smoothly with covariates, instead of assuming a fixed shape
    parameter. Use it for strictly positive, right-skewed responses where not only the typical
    magnitude but also the relative spread (coefficient of variation) changes systematically
    across the range of the predictors — for example, cost or duration data whose relative
    volatility grows with the covariates rather than staying proportional to `mu` alone. Both
    parameters use the log link, keeping `mu > 0` and `sigma > 0`.

    Parameters
    ----------
    None
        `GammaLS` takes no constructor arguments; both `mu` and `sigma` are modeled entirely
        through the formulas supplied to `GAMLSS`.

    Notes
    -----
    `GammaLS` parameterizes the Gamma distribution by its mean `mu > 0` and its coefficient of
    variation `sigma > 0`, where `sigma = 1 / sqrt(shape)` and `shape = alpha = 1 / sigma^2`.
    Both parameters use the log link:

    $$
    g_{\mu}(\mu) = \log(\mu), \qquad g_{\sigma}(\sigma) = \log(\sigma).
    $$

    The response density is the Gamma density with shape `alpha = 1/sigma^2` and rate
    `alpha/mu`:

    $$
    f(y \mid \mu, \sigma) = \frac{(\alpha/\mu)^{\alpha}}{\Gamma(\alpha)}\, y^{\alpha - 1}
    \exp\!\left(-\frac{\alpha y}{\mu}\right), \qquad \alpha = \frac{1}{\sigma^{2}}.
    $$

    Because `sigma` is the coefficient of variation, $\operatorname{Var}(Y) = \sigma^2 \mu^2$,
    so this is a direct generalization of `Gamma`'s variance function `V(mu) = mu^2` in which the
    proportionality constant `sigma^2` is itself allowed to depend on covariates.

    Examples
    --------
    Fit a GAMLSS where both the mean and relative spread of a positive response vary smoothly:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(1.0 + 0.4 * np.sin(x))
    sigma = 0.2 + 0.15 * np.abs(np.cos(x))
    shape = 1.0 / sigma**2
    y = rng.gamma(shape, mu / shape)

    data = {"x": x, "y": y}

    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
        family=wk.GammaLS(),
    )
    model.fit(data)
    print(model.summary())
    ```
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
