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

    Parameters
    ----------
    None
        `BetaLS` takes no constructor arguments; both `mu` and `phi` are modeled entirely
        through the formulas supplied to `GAMLSS`.

    Notes
    -----
    The two parameters use distinct link functions:

    $$
    g_{\mu}(\mu) = \log\!\left(\frac{\mu}{1-\mu}\right), \qquad g_{\phi}(\phi) = \log(\phi).
    $$

    If `a = mu * phi` and `b = (1 - mu) * phi`, the response follows `y ~ Beta(a, b)` with
    density

    $$
    f(y \mid \mu, \phi) = \frac{y^{a-1}(1-y)^{b-1}}{B(a, b)}, \qquad a = \mu\phi,\ \ b = (1-\mu)\phi.
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
