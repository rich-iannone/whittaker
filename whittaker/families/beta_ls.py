r"""Beta GAMLSS family (mean-precision parameterisation)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, digamma, expit, logit, polygamma

from whittaker.families.gamlss_base import GAMLSSFamily


class BetaLS(GAMLSSFamily):
    r"""Beta family for GAMLSS with mean-precision parameterisation.

    `BetaLS` extends the plain `Beta` family by letting both the mean `mu` and the precision
    `phi` vary smoothly with covariates, rather than treating precision as a single estimated
    constant. Use it for responses strictly between 0 and 1 (rates, proportions, fractions)
    where not only the typical level but also how tightly the response clusters around that
    level changes across the range of the predictors — for example, a proportion that becomes
    more variable in some regions of the covariate space and more tightly concentrated in
    others. The mean uses the logit link (as in `Beta`) and the precision uses the log link,
    keeping `mu` in `(0, 1)` and `phi > 0`.

    Notes
    -----
    The two parameters use distinct link functions:

    $$
    g_{\mu}(\mu) = \log\!\left(\frac{\mu}{1-\mu}\right), \qquad g_{\phi}(\phi) = \log(\phi).
    $$

    If `a = mu * phi` and `b = (1 - mu) * phi`, the response follows `y ~ Beta(a, b)` with
    density

    $$
    f(y \mid \mu, \phi) = \frac{y^{a-1}(1-y)^{b-1}}{B(a, b)}, \qquad
    a = \mu\phi,\ \ b = (1-\mu)\phi.
    $$

    As in `Beta`, larger `phi` concentrates the distribution more tightly around `mu`
    (`Var(Y) = mu(1-mu) / (1+phi)`), but here `phi` is itself modeled as a smooth function of
    covariates rather than a single scalar.

    Examples
    --------
    Fit a GAMLSS where both the mean and precision of a proportion response vary smoothly:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = expit(np.sin(x))
    phi = 10.0 + 15.0 * np.abs(np.cos(x))
    y = rng.beta(mu * phi, (1 - mu) * phi)

    data = {"x": x, "y": y}

    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "phi": "y ~ s(x)"},
        family=wk.BetaLS(),
    )
    model.fit(data)
    print(model.summary())
    ```
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the distributional parameters modeled by `BetaLS`.

        Returns
        -------
        tuple[str, ...]
            Always `('mu', 'phi')`: the mean and the precision, in the order `GAMLSS`
            updates them during the RS algorithm.
        """
        return ("mu", "phi")

    def link(self, param: str, values: NDArray) -> NDArray:
        """Apply the link function for `mu` or `phi`.

        `mu` uses the logit link, keeping its additive predictor unconstrained while
        `link_inverse` maps it back into `(0, 1)`; `phi` uses the log link, keeping it
        positive.

        Parameters
        ----------
        param
            Either `"mu"` or `"phi"`.
        values
            Parameter values on the natural scale, shape `(n,)`.

        Returns
        -------
        NDArray
            `logit(values)` for `"mu"`, `log(values)` for `"phi"`.
        """
        if param == "mu":
            return logit(values)
        return np.log(values)

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        """Apply the inverse link for `mu` or `phi`.

        Maps the additive predictor back to the natural scale: the logistic (`expit`)
        function for `mu`, restoring `(0, 1)`, and the exponential for `phi`, restoring
        positivity.

        Parameters
        ----------
        param
            Either `"mu"` or `"phi"`.
        eta
            Additive predictor values, shape `(n,)`.

        Returns
        -------
        NDArray
            `expit(eta)` for `"mu"`, `exp(eta)` for `"phi"`.
        """
        if param == "mu":
            return expit(eta)
        return np.exp(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        r"""Derivative of the link function for `mu` or `phi`.

        For the logit link, $d\eta/d\mu = 1/(\mu(1-\mu))$; for the log link,
        $d\eta/d\phi = 1/\phi$.

        Parameters
        ----------
        param
            Either `"mu"` or `"phi"`.
        values
            Parameter values on the natural scale at which to evaluate the derivative,
            shape `(n,)`.

        Returns
        -------
        NDArray
            `1 / (values * (1 - values))` for `"mu"`, `1 / values` for `"phi"`.
        """
        if param == "mu":
            return 1.0 / (values * (1.0 - values))
        return 1.0 / values

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""First derivative of the Beta log-likelihood with respect to `mu` or `phi`.

        With `a = mu * phi`, `b = (1 - mu) * phi`, `y* = logit(y)`, and
        `mu* = psi(a) - psi(b)` (where $\psi$ is the digamma function), the score
        functions are

        $$
        \frac{\partial \ell}{\partial \mu} = \phi\,(y^{*} - \mu^{*}), \qquad
        \frac{\partial \ell}{\partial \phi} = \mu\,(y^{*} - \mu^{*})
        + \psi(\phi) - \psi(b) + \log(1 - y).
        $$

        Parameters
        ----------
        param
            Either `"mu"` or `"phi"`.
        y
            Observed response values in `(0, 1)`, shape `(n,)`.
        params
            Current estimates with keys `"mu"` and `"phi"`, each of shape `(n,)`.

        Returns
        -------
        NDArray
            Elementwise first derivatives, shape `(n,)`.
        """
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
        r"""Expected Fisher information for `mu` or `phi`.

        With `a = mu * phi`, `b = (1 - mu) * phi`, and trigamma function $\psi_1$:

        $$
        -E\!\left[\frac{\partial^2 \ell}{\partial \mu^2}\right]
        = \phi^2\big(\psi_1(a) + \psi_1(b)\big),
        \qquad
        -E\!\left[\frac{\partial^2 \ell}{\partial \phi^2}\right]
        = \mu^2 \psi_1(a) + (1-\mu)^2 \psi_1(b) - \psi_1(\phi).
        $$

        Parameters
        ----------
        param
            Either `"mu"` or `"phi"`.
        y
            Observed response values, shape `(n,)` (unused, kept for interface
            consistency).
        params
            Current estimates with keys `"mu"` and `"phi"`, each of shape `(n,)`.

        Returns
        -------
        NDArray
            Elementwise working weights, shape `(n,)`.
        """
        mu = params["mu"]
        phi = params["phi"]
        a = mu * phi
        b = (1.0 - mu) * phi
        if param == "mu":
            return phi**2 * (polygamma(1, a) + polygamma(1, b))
        return mu**2 * polygamma(1, a) + (1.0 - mu) ** 2 * polygamma(1, b) - polygamma(1, phi)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        """Total Beta log-likelihood at the current `mu` and `phi`.

        Evaluated using the shape parameterization `a = mu * phi`, `b = (1 - mu) * phi`.

        Parameters
        ----------
        y
            Observed response values in `(0, 1)`, shape `(n,)`.
        params
            Current estimates with keys `"mu"` and `"phi"`, each of shape `(n,)`.

        Returns
        -------
        float
            The log-likelihood summed over all observations.
        """
        mu = params["mu"]
        phi = params["phi"]
        a = mu * phi
        b = (1.0 - mu) * phi
        ll_i = -betaln(a, b) + (a - 1.0) * np.log(y) + (b - 1.0) * np.log(1.0 - y)
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        """Starting values for `mu` and `phi` from the raw response.

        `mu` is initialized at the observed values themselves (clipped to `[0.01, 0.99]`
        to stay strictly inside the unit interval), and `phi` at a method-of-moments
        estimate of the precision derived from the sample mean and variance of `y`
        (clamped to at least `1.0`), constant across observations.

        Parameters
        ----------
        y
            Observed response values in `(0, 1)`, shape `(n,)`.

        Returns
        -------
        dict[str, NDArray]
            `{"mu": y_safe.copy(), "phi": <constant array of the moment-based precision>}`.
        """
        y_safe = np.clip(y, 0.01, 0.99)
        mu_init = y_safe.copy()
        y_mean = np.mean(y_safe)
        y_var = np.var(y_safe, ddof=1)
        phi_est = y_mean * (1.0 - y_mean) / max(y_var, 1e-6) - 1.0
        phi_init = np.full_like(y_safe, max(phi_est, 1.0))
        return {"mu": mu_init, "phi": phi_init}

    def simulate(self, params: dict[str, NDArray], rng: np.random.Generator) -> NDArray:
        """Simulate response values from a Beta distribution with the given `mu`, `phi`.

        Converts to the shape parameterization (`a = mu * phi`, `b = (1 - mu) * phi`)
        expected by `rng.beta`.

        Parameters
        ----------
        params
            Current estimates with keys `"mu"` and `"phi"`, each of shape `(n,)`.
        rng
            A NumPy random generator used to draw the simulated values.

        Returns
        -------
        NDArray
            Simulated response values in `(0, 1)`, shape `(n,)`.
        """
        mu = params["mu"]
        phi = params["phi"]
        a = mu * phi
        b = (1.0 - mu) * phi
        return rng.beta(a, b)

    def __repr__(self) -> str:
        return "BetaLS(mu=logit, phi=log)"
