r"""Beta regression family with logit link and fixed precision."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, expit, logit

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Beta(Family):
    r"""Beta regression family with logit link.

    The Beta family models a continuous response strictly between 0 and 1 — for example rates,
    fractions, or proportions that are not simply the ratio of successes to a known number of
    trials (in which case `Binomial` is usually more appropriate). It parameterizes the Beta
    distribution by its mean `mu` and a precision parameter `phi`, so that larger `phi` yields a
    tighter distribution around `mu` for fixed mean, analogous to the role `theta` plays in
    `NegativeBinomial`. The logit link keeps the fitted mean within `(0, 1)` and gives
    coefficients the same log-odds interpretation as in `Binomial` regression.

    Parameters
    ----------
    phi : float or None, default=None
        Fixed precision parameter, must be positive if provided. If `None` (the default),
        precision is treated as unknown and estimated from the data via the scale parameter
        (`phi = 1 / scale`); in that case `scale_known` is `False`. Passing a fixed `phi` is
        useful when the precision is known a priori or should not be re-estimated.

    Notes
    -----
    The link is the logit function:

    $$
    g(\mu) = \log\!\left(\frac{\mu}{1-\mu}\right)
    $$

    The variance function is

    $$
    V(\mu) = \frac{\mu(1-\mu)}{1+\phi},
    $$

    so larger `phi` (higher precision) shrinks the variance for a given mean. The deviance is
    twice the difference between the saturated and fitted log-likelihoods,

    $$
    D(y, \hat\mu) = 2 \sum_i \left[ \ell(y_i; y_i) - \ell(y_i; \hat\mu_i) \right],
    $$

    where $\ell$ is the Beta log-density parameterized by $(a, b) = (\mu \phi, (1-\mu)\phi)$.

    Examples
    --------
    Fit a GAM to a proportion response with a smooth mean trend:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    mu = expit(np.sin(x))
    phi = 20.0
    y = rng.beta(mu * phi, (1 - mu) * phi)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.Beta())
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def __init__(self, phi: float | None = None) -> None:
        self._phi = phi

    def link(self, mu: NDArray) -> NDArray:
        return logit(np.clip(mu, _EPS, 1.0 - _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return expit(eta)

    def link_derivative(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return 1.0 / (mu_c * (1.0 - mu_c))

    def variance(self, mu: NDArray) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return mu_c * (1.0 - mu_c)

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        y_c = np.clip(y, _EPS, 1.0 - _EPS)
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return 2.0 * (y_c * np.log(y_c / mu_c) + (1.0 - y_c) * np.log((1.0 - y_c) / (1.0 - mu_c)))

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        phi = self._phi if self._phi is not None else 1.0 / max(scale, _EPS)
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        y_c = np.clip(y, _EPS, 1.0 - _EPS)
        a = mu_c * phi
        b = (1.0 - mu_c) * phi
        ll_i = -betaln(a, b) + (a - 1.0) * np.log(y_c) + (b - 1.0) * np.log(1.0 - y_c)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return self._phi is not None

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        phi = self._phi if self._phi is not None else 1.0 / max(scale, _EPS)
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        a = mu_c * phi
        b = (1.0 - mu_c) * phi
        return rng.beta(a, b)

    def initialize(self, y: NDArray) -> NDArray:
        return np.clip(y, 0.01, 0.99)

    def __repr__(self) -> str:
        if self._phi is not None:
            return f"Beta(link='logit', phi={self._phi})"
        return "Beta(link='logit')"
