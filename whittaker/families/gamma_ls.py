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
        """Names of the distributional parameters modeled by `GammaLS`.

        Returns
        -------
        tuple[str, ...]
            Always `('mu', 'sigma')`: the mean and the coefficient of variation, in the
            order `GAMLSS` updates them during the RS algorithm.
        """
        return ("mu", "sigma")

    def link(self, param: str, values: NDArray) -> NDArray:
        """Apply the log link for `mu` or `sigma`.

        Both parameters use the log link, so `values` are first floored (`_MU_FLOOR` for
        `mu`, `_SIGMA_FLOOR` for `sigma`) to avoid `log(0)` before taking the logarithm.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"`.
        values
            Parameter values on the natural scale, shape `(n,)`.

        Returns
        -------
        NDArray
            `log(values)`, with `values` clamped away from zero first.
        """
        return np.log(np.maximum(values, _MU_FLOOR if param == "mu" else _SIGMA_FLOOR))

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        """Apply the inverse log link for `mu` or `sigma`.

        `eta` is clipped to `[-_ETA_MAX, _ETA_MAX]` before exponentiating, guarding
        against overflow when the additive predictor drifts to extreme values during
        fitting.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"` (unused directly, since both share the log link).
        eta
            Additive predictor values, shape `(n,)`.

        Returns
        -------
        NDArray
            `exp(eta)` after clipping `eta` for numerical stability.
        """
        eta_c = np.clip(eta, -_ETA_MAX, _ETA_MAX)
        return np.exp(eta_c)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        r"""Derivative of the log link for `mu` or `sigma`.

        For the log link, $d\eta/d\theta = 1/\theta$ for either parameter.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"`, used only to select the appropriate numerical
            floor.
        values
            Parameter values on the natural scale at which to evaluate the derivative,
            shape `(n,)`.

        Returns
        -------
        NDArray
            `1 / values`, with `values` clamped away from zero first.
        """
        floor = _MU_FLOOR if param == "mu" else _SIGMA_FLOOR
        return 1.0 / np.maximum(values, floor)

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""First derivative of the Gamma log-likelihood with respect to `mu` or `sigma`.

        With shape `alpha = 1/sigma^2`, the score functions are

        $$
        \frac{\partial \ell}{\partial \mu} = \frac{\alpha}{\mu}\left(\frac{y}{\mu} - 1\right),
        \qquad
        \frac{\partial \ell}{\partial \sigma} = \frac{2}{\sigma^3}\left(
        \psi(\alpha) - \log\alpha - \log\frac{y}{\mu} - 1 + \frac{y}{\mu}\right),
        $$

        where $\psi$ is the digamma function.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"`.
        y
            Observed response values, shape `(n,)`.
        params
            Current estimates with keys `"mu"` and `"sigma"`, each of shape `(n,)`.

        Returns
        -------
        NDArray
            Elementwise first derivatives, shape `(n,)`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        sigma = np.maximum(params["sigma"], _SIGMA_FLOOR)
        alpha = 1.0 / sigma**2
        if param == "mu":
            return alpha * (y / mu - 1.0) / mu
        return (2.0 / sigma**3) * (
            digamma(alpha) - np.log(alpha) - np.log(np.maximum(y, _MU_FLOOR) / mu) - 1.0 + y / mu
        )

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""Expected Fisher information for `mu` or `sigma`.

        With shape `alpha = 1/sigma^2` and trigamma function $\psi_1$:

        $$
        -E\!\left[\frac{\partial^2 \ell}{\partial \mu^2}\right] = \frac{\alpha}{\mu^2}, \qquad
        -E\!\left[\frac{\partial^2 \ell}{\partial \sigma^2}\right]
        = \frac{4}{\sigma^4}\big(\alpha\,\psi_1(\alpha) - 1\big).
        $$

        The `sigma` term is additionally floored at `_SIGMA_FLOOR` to keep the working
        weight strictly positive.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"`.
        y
            Observed response values, shape `(n,)` (unused, kept for interface
            consistency).
        params
            Current estimates with keys `"mu"` and `"sigma"`, each of shape `(n,)`.

        Returns
        -------
        NDArray
            Elementwise working weights, shape `(n,)`, guaranteed positive.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        sigma = np.maximum(params["sigma"], _SIGMA_FLOOR)
        alpha = 1.0 / sigma**2
        if param == "mu":
            return alpha / mu**2
        return np.maximum((4.0 / sigma**4) * (polygamma(1, alpha) * alpha - 1.0), _SIGMA_FLOOR)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        """Total Gamma log-likelihood at the current `mu` and `sigma`.

        Evaluated using the shape/rate parameterization `alpha = 1/sigma^2`,
        `rate = alpha/mu`, with `y` and `mu` floored at `_MU_FLOOR` and `sigma` floored at
        `_SIGMA_FLOOR` for numerical stability.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        params
            Current estimates with keys `"mu"` and `"sigma"`, each of shape `(n,)`.

        Returns
        -------
        float
            The log-likelihood summed over all observations.
        """
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
        """Starting values for `mu` and `sigma` from the raw response.

        `mu` is initialized at the (floored) observed values themselves, and `sigma` at
        the sample coefficient of variation of `y` (clamped to at least `0.1`), constant
        across observations.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.

        Returns
        -------
        dict[str, NDArray]
            `{"mu": y_safe.copy(), "sigma": <constant array of the sample CV>}`.
        """
        y_safe = np.maximum(y, _MU_FLOOR)
        mu_init = y_safe.copy()
        cv = np.std(y_safe) / np.mean(y_safe)
        sigma_init = np.full_like(y_safe, max(cv, 0.1))
        return {"mu": mu_init, "sigma": sigma_init}

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        """Simulate response values from a Gamma distribution with the given `mu`, `sigma`.

        Converts to the shape/scale parameterization (`shape = 1/sigma^2`,
        `scale = mu * sigma^2`) expected by `rng.gamma`.

        Parameters
        ----------
        params
            Current estimates with keys `"mu"` and `"sigma"`, each of shape `(n,)`.
        rng
            A NumPy random generator used to draw the simulated values.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        sigma = np.maximum(params["sigma"], _SIGMA_FLOOR)
        shape = 1.0 / sigma**2
        scale = mu * sigma**2
        return rng.gamma(shape=shape, scale=scale)

    def __repr__(self) -> str:
        return "GammaLS(mu=log, sigma=log)"
