r"""Inverse Gaussian family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class InverseGaussian(Family):
    r"""Inverse Gaussian family with log link.

    The Inverse Gaussian family models strictly positive, continuous responses whose variance
    grows even faster than in the Gamma case, producing heavier right tails. It arises naturally
    as a first-passage-time distribution (e.g. the time for a diffusion process to reach a
    threshold) and is a common choice for lifetime, duration, and other highly skewed positive
    data. `InverseGaussian` is the Tweedie special case with variance power `p = 3` (compare
    `Tweedie` and `tw()`), implemented directly here for exact deviance and log-likelihood
    computations rather than the saddlepoint approximation used by the general `Tweedie` family.
    The log link is used for the same reasons as in `Gamma`: it keeps fitted values positive and
    gives coefficients a multiplicative interpretation.

    Parameters
    ----------
    None
        `InverseGaussian` takes no constructor arguments. The dispersion parameter `phi` is
        estimated from the data during fitting (see `scale_known`).

    Notes
    -----
    The link is the natural logarithm:

    $$
    g(\mu) = \log(\mu)
    $$

    The variance function grows with the cube of the mean:

    $$
    V(\mu) = \mu^{3},
    $$

    making this family appropriate when large means are associated with disproportionately
    large variability. The deviance is

    $$
    D(y, \hat\mu) = \sum_i \frac{(y_i - \hat\mu_i)^2}{\hat\mu_i^2\, y_i} .
    $$

    Examples
    --------
    Fit a GAM to heavy-tailed positive data with a smooth trend:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(1.0 + 0.4 * np.sin(x))
    scale = 0.5
    lam = mu / scale
    y = rng.wald(mu, lam)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.InverseGaussian())
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
        return np.maximum(mu, _EPS) ** 3

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        return (y_c - mu_c) ** 2 / (mu_c**2 * y_c)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        ll_i = (
            -0.5 * np.log(2.0 * np.pi * scale * y_c**3)
            - (y_c - mu_c) ** 2 / (2.0 * scale * mu_c**2 * y_c)
        )
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return False

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        mu_c = np.maximum(mu, _EPS)
        lam = mu_c / scale
        n = len(mu_c)
        v = rng.standard_normal(n) ** 2
        x = mu_c + (mu_c**2 * v) / (2.0 * lam) - mu_c / (2.0 * lam) * np.sqrt(
            4.0 * mu_c * lam * v + mu_c**2 * v**2
        )
        u = rng.uniform(size=n)
        result = np.where(u <= mu_c / (mu_c + x), x, mu_c**2 / x)
        return np.maximum(result, _EPS)

    def initialize(self, y: NDArray) -> NDArray:
        return np.maximum(y, _EPS) + 0.1

    def __repr__(self) -> str:
        return "InverseGaussian(link='log')"
