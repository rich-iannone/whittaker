r"""Poisson family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Poisson(Family):
    r"""Poisson family with log (canonical) link.

    The Poisson family models the response as non-negative integer counts, such as the number
    of events observed in a fixed interval of time, space, or exposure. It is the standard
    choice for count data when the variance of the counts is approximately equal to their mean.
    The canonical log link guarantees positive fitted values on the response scale and gives
    the linear predictor a multiplicative interpretation: a one-unit increase in a covariate
    multiplies the expected count by `exp(coefficient)`.

    Parameters
    ----------
    None
        `Poisson` takes no constructor arguments. The scale parameter is fixed at `1` (see
        `scale_known`), since the Poisson distribution has no free dispersion parameter.

    Notes
    -----
    The canonical link is the natural logarithm:

    $$
    g(\mu) = \log(\mu)
    $$

    The variance function is $V(\mu) = \mu$, so the variance equals the mean. If the observed
    variance substantially exceeds the mean (overdispersion), consider `NegativeBinomial` or
    `Tweedie` instead. The deviance is

    $$
    D(y, \hat\mu) = 2 \sum_i \left[ y_i \log\!\left(\frac{y_i}{\hat\mu_i}\right) - (y_i - \hat\mu_i) \right] .
    $$

    Examples
    --------
    Fit a GAM to simulated count data with a smooth, log-linear trend:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 * np.sin(x))
    y = rng.poisson(mu)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.Poisson())
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def link(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        return np.maximum(mu, _EPS)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        dev = np.empty_like(y)
        pos = y > 0
        dev[pos] = y[pos] * np.log(y[pos] / mu_c[pos])
        dev[~pos] = 0.0
        dev -= y - mu_c
        return 2.0 * dev

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        from scipy.special import gammaln

        mu_c = np.maximum(mu, _EPS)
        ll_i = y * np.log(mu_c) - mu_c - gammaln(y + 1.0)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        return rng.poisson(np.maximum(mu, _EPS)).astype(float)

    def initialize(self, y: NDArray) -> NDArray:
        return np.maximum(y, 0.1) + 0.1

    def __repr__(self) -> str:
        return "Poisson(link='log')"
