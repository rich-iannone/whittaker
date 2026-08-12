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
    f(y \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right),
    $$

    so the log-likelihood contribution for a single observation is
    $\ell_i = -\log\sigma_i - \tfrac{1}{2}\log(2\pi) - \tfrac{1}{2}\left(\frac{y_i - \mu_i}{\sigma_i}\right)^2$.
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
