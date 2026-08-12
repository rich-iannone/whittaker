r"""Gaussian (Normal) family with identity link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family


class Gaussian(Family):
    r"""Gaussian (Normal) family with identity link.

    The Gaussian family models a continuous, unbounded response with constant variance. It is
    the default family in Whittaker and corresponds to classical (penalized) least-squares
    regression: with the identity link, P-IRLS converges in a single step since the working
    response and weights do not depend on the current fit. Use it whenever the response is
    real-valued, approximately symmetric, and its spread does not depend systematically on its
    mean — for example, physical measurements, log-transformed sizes, or residual-like
    quantities. If the variance grows with the mean, or the response is a count, proportion, or
    strictly positive quantity, consider `Poisson`, `Binomial`, `Gamma`, or another family
    instead.

    Parameters
    ----------
    None
        `Gaussian` takes no constructor arguments; the scale parameter `phi` (the residual
        variance `sigma^2`) is estimated from the data during fitting rather than supplied by
        the user.

    Notes
    -----
    The canonical (and only supported) link is the identity function:

    $$
    g(\mu) = \mu
    $$

    so the linear predictor `eta` is directly on the response scale and no back-transformation
    is needed for predictions. The variance function is constant in the mean,

    $$
    V(\mu) = 1, \qquad \operatorname{Var}(Y) = \phi \, V(\mu) = \sigma^2,
    $$

    which is what makes the Gaussian family the special case in which ordinary least squares and
    maximum-likelihood estimation coincide. The deviance is the residual sum of squares:

    $$
    D(y, \hat\mu) = \sum_i (y_i - \hat\mu_i)^2 .
    $$

    Examples
    --------
    Fit a GAM with a smooth term to noisy sine-wave data using the (default) Gaussian family:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.Gaussian())
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def link(self, mu: NDArray) -> NDArray:
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = (y - mu) ** 2
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        ll_i = -0.5 * np.log(2 * np.pi * scale) - 0.5 * (y - mu) ** 2 / scale
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        return rng.normal(mu, np.sqrt(scale))

    def initialize(self, y: NDArray) -> NDArray:
        return y.copy()

    def __repr__(self) -> str:
        return "Gaussian(link='identity')"
