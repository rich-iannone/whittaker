r"""Binomial family with logit link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Binomial(Family):
    r"""Binomial family with logit (canonical) link.

    The Binomial family models a response that is either binary (`0`/`1`) or a proportion in
    `[0, 1]` (e.g. the fraction of successes out of a known number of trials). The logit link
    maps the unit interval to the whole real line, so the linear predictor is unconstrained
    while the fitted mean is always a valid probability. Use it for classification-style GAMs,
    for aggregated success/trial proportions, or wherever the outcome represents the
    probability of an event. The logit link additionally gives coefficients a log-odds
    interpretation: a one-unit increase in a covariate changes the log-odds of success by
    `coefficient`, and multiplies the odds by `exp(coefficient)`.

    Parameters
    ----------
    None
        `Binomial` takes no constructor arguments. The scale parameter is fixed at `1` (see
        `scale_known`), since the Bernoulli/Binomial distribution has no free dispersion
        parameter.

    Notes
    -----
    The canonical link is the logit function:

    $$
    g(\mu) = \log\!\left(\frac{\mu}{1-\mu}\right)
    $$

    with inverse the logistic sigmoid $\mu = g^{-1}(\eta) = 1 / (1 + e^{-\eta})$. The variance
    function is

    $$
    V(\mu) = \mu(1-\mu),
    $$

    which is largest near $\mu = 0.5$ and shrinks toward zero as $\mu$ approaches either
    boundary. The deviance is

    $$
    D(y, \hat\mu) = 2 \sum_i \left[ y_i \log\!\left(\frac{y_i}{\hat\mu_i}\right)
    + (1 - y_i) \log\!\left(\frac{1 - y_i}{1 - \hat\mu_i}\right) \right] .
    $$

    Examples
    --------
    Fit a GAM to a binary outcome with a smooth, nonlinear log-odds relationship:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(-3, 3, n)
    p = expit(np.sin(x))
    y = rng.binomial(1, p)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.Binomial())
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def link(self, mu: NDArray) -> NDArray:
        return logit(mu)

    def link_inverse(self, eta: NDArray) -> NDArray:
        return expit(eta)

    def link_derivative(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return 1.0 / (mu_c * (1.0 - mu_c))

    def variance(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return mu_c * (1.0 - mu_c)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        dev = np.zeros_like(y)
        pos = y > 0
        neg = y < 1
        dev[pos] = y[pos] * np.log(y[pos] / mu_c[pos])
        dev[neg] += (1.0 - y[neg]) * np.log((1.0 - y[neg]) / (1.0 - mu_c[neg]))
        return 2.0 * dev

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        ll_i = y * np.log(mu_c) + (1.0 - y) * np.log(1.0 - mu_c)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        p = np.clip(mu, _EPS, 1.0 - _EPS)
        return rng.binomial(1, p).astype(float)

    def initialize(self, y: NDArray) -> NDArray:
        return np.full_like(y, fill_value=(np.mean(y) + 0.5) / 2.0)

    def __repr__(self) -> str:
        return "Binomial(link='logit')"
