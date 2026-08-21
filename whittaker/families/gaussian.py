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
        r"""Apply the identity link: $\eta = \mu$.

        The Gaussian family uses the identity link, so this is a no-op that returns `mu`
        unchanged.

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Linear predictor values $\eta = \mu$, shape `(n,)`.
        """
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        r"""Apply the inverse identity link: $\mu = \eta$.

        Returns `eta` unchanged, since the identity link is self-inverse.

        Parameters
        ----------
        eta
            Linear predictor values, shape `(n,)`.

        Returns
        -------
        NDArray
            Conditional mean values $\mu = \eta$, shape `(n,)`.
        """
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        r"""Derivative of the identity link: $g'(\mu) = 1$.

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            An array of ones, shape `(n,)`.
        """
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        r"""Constant variance function: $V(\mu) = 1$.

        The Gaussian variance does not depend on the mean, so `Var(Y) = phi * V(mu) = phi`
        (the residual variance `sigma^2`).

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            An array of ones, shape `(n,)`.
        """
        return np.ones_like(mu)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total deviance: the (weighted) residual sum of squares $\sum_i (y_i - \hat\mu_i)^2$.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        float
            The total (weighted) residual sum of squares.
        """
        d = (y - mu) ** 2
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        r"""Gaussian log-likelihood evaluated at variance `scale`.

        Computes $\ell_i = -\tfrac{1}{2}\log(2\pi\phi) - \tfrac{(y_i - \mu_i)^2}{2\phi}$ for
        each observation, where $\phi$ is `scale`, and sums (optionally weighted) over
        observations.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        scale
            Residual variance $\phi = \sigma^2$.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        float
            The total log-likelihood.
        """
        ll_i = -0.5 * np.log(2 * np.pi * scale) - 0.5 * (y - mu) ** 2 / scale
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    def simulate(self, mu: NDArray, scale: float, rng: np.random.Generator) -> NDArray:
        """Simulate Gaussian response values `N(mu, scale)`.

        Parameters
        ----------
        mu
            Mean (fitted values), shape `(n,)`.
        scale
            Residual variance `sigma^2`.
        rng
            A `numpy.random.Generator` instance.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        return rng.normal(mu, np.sqrt(scale))

    def initialize(self, y: NDArray) -> NDArray:
        """Starting values for `mu`: a copy of the observed response `y`.

        Appropriate for the Gaussian family since the identity link places no constraint on the
        valid range of `mu`.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.

        Returns
        -------
        NDArray
            Starting values for `mu`, shape `(n,)`.
        """
        return y.copy()

    def __repr__(self) -> str:
        return "Gaussian(link='identity')"
