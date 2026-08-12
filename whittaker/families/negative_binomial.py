r"""Negative Binomial (NB2) family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class NegativeBinomial(Family):
    r"""Negative Binomial family with log link (NB2 parameterization).

    The Negative Binomial family models count data that is overdispersed relative to the
    Poisson distribution, i.e. where the observed variance exceeds the mean. This commonly
    arises when counts are driven by unobserved heterogeneity across observations (e.g. some
    individuals or locations are systematically more prone to events than others). Whittaker
    uses the NB2 parameterization, where the variance function is
    `V(mu) = mu + mu^2/theta` and `theta` controls the degree of overdispersion: as
    `theta -> infinity` the distribution converges to `Poisson`, while smaller `theta` implies
    heavier overdispersion. The canonical log link is used, giving the same multiplicative
    interpretation of coefficients as `Poisson`.

    Parameters
    ----------
    theta : float, default=1.0
        Overdispersion (size) parameter, must be positive. Smaller values of `theta` imply
        greater overdispersion; larger values make the distribution approach `Poisson`. The
        value supplied at construction is used as the starting point and is refined during
        fitting via an outer iteration around P-IRLS unless explicitly held fixed by the
        caller.

    Notes
    -----
    The canonical link is the natural logarithm:

    $$
    g(\mu) = \log(\mu)
    $$

    The variance function is

    $$
    V(\mu) = \mu + \frac{\mu^2}{\theta},
    $$

    so the variance always exceeds the mean by the extra term $\mu^2/\theta$. The deviance is

    $$
    D(y, \hat\mu) = 2 \sum_i \left[ y_i \log\!\left(\frac{y_i}{\hat\mu_i}\right)
    - (y_i + \theta) \log\!\left(\frac{y_i + \theta}{\hat\mu_i + \theta}\right) \right] .
    $$

    Examples
    --------
    Fit a GAM to overdispersed count data:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 * np.sin(x))
    theta = 3.0
    y = rng.negative_binomial(theta, theta / (theta + mu))

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.NegativeBinomial(theta=theta))
    model.fit(data, method="REML")
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

    @theta.setter
    def theta(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"theta must be positive, got {value}.")
        self._theta = float(value)

    def link(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        return mu_c + mu_c**2 / self._theta

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        theta = self._theta
        dev = np.empty_like(y, dtype=float)
        pos = y > 0
        dev[pos] = y[pos] * np.log(y[pos] / mu_c[pos])
        dev[~pos] = 0.0
        dev -= (y + theta) * np.log((y + theta) / (mu_c + theta))
        return 2.0 * dev

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        from scipy.special import gammaln

        mu_c = np.maximum(mu, _EPS)
        theta = self._theta
        ll_i = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1.0)
            + theta * np.log(theta / (mu_c + theta))
            + y * np.log(mu_c / (mu_c + theta))
        )
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        theta = self._theta
        p = theta / (mu_c + theta)
        return rng.negative_binomial(theta, p).astype(float)

    def initialize(self, y: NDArray) -> NDArray:
        return np.maximum(y, 0.1) + 0.1

    def __repr__(self) -> str:
        return f"NegativeBinomial(theta={self._theta:.4g}, link='log')"
