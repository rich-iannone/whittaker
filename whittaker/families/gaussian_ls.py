r"""Gaussian location-scale GAMLSS family."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.gamlss_base import GAMLSSFamily


class GaussianLS(GAMLSSFamily):
    r"""Gaussian location-scale family for GAMLSS.

    `GaussianLS` extends the plain `Gaussian` family to allow both the mean `mu` and the
    standard deviation `sigma` to vary smoothly with covariates, rather than assuming constant
    variance. Use it when a continuous, approximately symmetric response shows
    heteroscedasticity — for example, when the spread of measurements grows or shrinks over the
    range of a predictor — and you want the model to capture that varying spread rather than
    average it away. Each parameter has its own additive predictor and its own link function:
    `mu` uses the identity link (as in `Gaussian`), and `sigma` uses the log link, which keeps
    the fitted standard deviation positive.

    Parameters
    ----------
    None
        `GaussianLS` takes no constructor arguments; both `mu` and `sigma` are modeled entirely
        through the formulas supplied to `GAMLSS`.

    Notes
    -----
    The two parameters use distinct link functions:

    $$
    g_{\mu}(\mu) = \mu \qquad \text{(identity)}, \qquad
    g_{\sigma}(\sigma) = \log(\sigma).
    $$

    The response density is the ordinary Gaussian density evaluated at the fitted, observation-
    specific `mu` and `sigma`:

    $$
    f(y \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}
    \exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right),
    $$

    so the log-likelihood contribution for a single observation is
    $\ell_i = -\log\sigma_i - \tfrac{1}{2}\log(2\pi)
    - \tfrac{1}{2}\left(\frac{y_i - \mu_i}{\sigma_i}\right)^2$.
    Unlike `Gaussian`, there is no separate scale parameter to estimate: `sigma` itself is the
    quantity being modeled by its own smooth predictor.

    Examples
    --------
    Fit a GAMLSS with a smoothly varying mean and standard deviation:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.sin(x)
    sigma = 0.2 + 0.3 * np.abs(np.cos(x))
    y = rng.normal(mu, sigma)

    data = {"x": x, "y": y}

    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
        family=wk.GaussianLS(),
    )
    model.fit(data)
    print(model.summary())
    ```
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the distributional parameters modeled by `GaussianLS`.

        Returns
        -------
        tuple[str, ...]
            Always `('mu', 'sigma')`: the mean and the standard deviation, in the order
            `GAMLSS` updates them during the RS algorithm.
        """
        return ("mu", "sigma")

    def link(self, param: str, values: NDArray) -> NDArray:
        """Apply the link function for `mu` or `sigma`.

        `mu` uses the identity link and `sigma` uses the log link, so that `sigma`'s
        additive predictor is unconstrained while the fitted standard deviation stays
        positive after applying `link_inverse`.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"`.
        values
            Parameter values on the natural scale, shape `(n,)`.

        Returns
        -------
        NDArray
            Linked values: `values` unchanged for `"mu"`, `log(values)` for `"sigma"`.
        """
        if param == "mu":
            return values
        return np.log(values)

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        """Apply the inverse link for `mu` or `sigma`.

        Maps the additive predictor `eta` back to the natural scale: unchanged for `mu`
        (identity link), exponentiated for `sigma` (log link) so it stays positive.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"`.
        eta
            Additive predictor values, shape `(n,)`.

        Returns
        -------
        NDArray
            `eta` unchanged for `"mu"`, `exp(eta)` for `"sigma"`.
        """
        if param == "mu":
            return eta
        return np.exp(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        r"""Derivative of the link function for `mu` or `sigma`.

        For the identity link, $d\eta/d\mu = 1$; for the log link,
        $d\eta/d\sigma = 1/\sigma$.

        Parameters
        ----------
        param
            Either `"mu"` or `"sigma"`.
        values
            Parameter values on the natural scale at which to evaluate the derivative,
            shape `(n,)`.

        Returns
        -------
        NDArray
            An array of ones for `"mu"`, or `1 / values` for `"sigma"`.
        """
        if param == "mu":
            return np.ones_like(values)
        return 1.0 / values

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""First derivative of the Gaussian log-likelihood with respect to `mu` or `sigma`.

        The score functions are

        $$
        \frac{\partial \ell}{\partial \mu} = \frac{y - \mu}{\sigma^2}, \qquad
        \frac{\partial \ell}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(y-\mu)^2}{\sigma^3}.
        $$

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
        mu = params["mu"]
        sigma = params["sigma"]
        if param == "mu":
            return (y - mu) / sigma**2
        return -1.0 / sigma + (y - mu) ** 2 / sigma**3

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""Expected Fisher information for `mu` or `sigma`.

        For the Gaussian distribution these expected information terms do not depend on
        `y`:

        $$
        -E\!\left[\frac{\partial^2 \ell}{\partial \mu^2}\right] = \frac{1}{\sigma^2}, \qquad
        -E\!\left[\frac{\partial^2 \ell}{\partial \sigma^2}\right] = \frac{2}{\sigma^2}.
        $$

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
            Elementwise working weights, shape `(n,)`.
        """
        sigma = params["sigma"]
        if param == "mu":
            return 1.0 / sigma**2
        return 2.0 / sigma**2

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        """Total Gaussian log-likelihood at the current `mu` and `sigma`.

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
        mu = params["mu"]
        sigma = params["sigma"]
        ll_i = -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mu) / sigma) ** 2
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        """Starting values for `mu` and `sigma` from the raw response.

        `mu` is initialized at the observed values themselves and `sigma` at the sample
        standard deviation of `y`, constant across observations.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.

        Returns
        -------
        dict[str, NDArray]
            `{"mu": y.copy(), "sigma": <constant array of the sample std>}`.
        """
        return {
            "mu": y.copy(),
            "sigma": np.full_like(y, np.std(y, ddof=1)),
        }

    def simulate(self, params: dict[str, NDArray], rng: np.random.Generator) -> NDArray:
        """Simulate response values from `Normal(mu, sigma)`.

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
        return rng.normal(params["mu"], params["sigma"])

    def __repr__(self) -> str:
        return "GaussianLS(mu=identity, sigma=log)"
