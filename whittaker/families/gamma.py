r"""Gamma family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Gamma(Family):
    r"""Gamma family with log link.

    The Gamma family models strictly positive, continuous, right-skewed responses — for
    example, insurance claim sizes, waiting times, rainfall amounts, or other quantities that
    are bounded below by zero and become more variable as their mean grows. Although the
    canonical link for the Gamma distribution is the inverse, Whittaker uses the log link by
    default, since it guarantees positive fitted values and is generally easier to interpret
    (coefficients act multiplicatively on the response, as with `Poisson`). Use `Gamma` when the
    response is positive and continuous and its coefficient of variation is roughly constant
    across the range of fitted values; if instead the variance grows linearly with the mean,
    `Poisson` or `Tweedie` with `1 < p < 2` may fit better.

    Parameters
    ----------
    None
        `Gamma` takes no constructor arguments. The dispersion parameter `phi = 1/alpha` (the
        inverse of the shape parameter) is estimated from the data during fitting (see
        `scale_known`).

    Notes
    -----
    The (non-canonical, but default) link is the natural logarithm:

    $$
    g(\mu) = \log(\mu)
    $$

    The variance function grows with the square of the mean:

    $$
    V(\mu) = \mu^2, \qquad \operatorname{Var}(Y) = \phi \, \mu^2,
    $$

    so the coefficient of variation $\sqrt{\operatorname{Var}(Y)} / \mu = \sqrt{\phi}$ is
    constant. The deviance is

    $$
    D(y, \hat\mu) = 2 \sum_i \left[ -\log\!\left(\frac{y_i}{\hat\mu_i}\right)
    + \frac{y_i - \hat\mu_i}{\hat\mu_i} \right] .
    $$

    Examples
    --------
    Fit a GAM to positive, right-skewed data with a smooth trend:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 + 0.4 * np.sin(x))
    shape = 4.0
    y = rng.gamma(shape, mu / shape)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.Gamma())
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def link(self, mu: NDArray) -> NDArray:
        r"""Apply the (default, non-canonical) log link: $\eta = \log(\mu)$.

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`. Must be positive.

        Returns
        -------
        NDArray
            Linear predictor values $\eta = \log(\mu)$, shape `(n,)`.
        """
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        r"""Apply the inverse log link: $\mu = e^{\eta}$.

        The linear predictor is clipped to `[-30, 30]` before exponentiating to guard against
        overflow while fitting.

        Parameters
        ----------
        eta
            Linear predictor values, shape `(n,)`.

        Returns
        -------
        NDArray
            Conditional mean values $\mu = e^{\eta}$, shape `(n,)`.
        """
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        r"""Derivative of the log link: $g'(\mu) = 1/\mu$.

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Derivative values $1/\mu$, shape `(n,)`.
        """
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        r"""Gamma variance function: $V(\mu) = \mu^2$.

        The variance grows with the square of the mean, so the coefficient of variation
        $\sqrt{\operatorname{Var}(Y)}/\mu = \sqrt\phi$ is constant across the range of `mu`.

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Variance values $\mu^2$, shape `(n,)`.
        """
        return np.maximum(mu, _EPS) ** 2

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total Gamma deviance, the (weighted) sum of `unit_deviance`.

        Parameters
        ----------
        y
            Observed response values (positive), shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        float
            The total (weighted) deviance.
        """
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        r"""Per-observation Gamma deviance contributions.

        Computes $d_i = 2 \left[ -\log(y_i/\hat\mu_i) + (y_i - \hat\mu_i)/\hat\mu_i \right]$.

        Parameters
        ----------
        y
            Observed response values (positive), shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Per-observation deviance contributions, shape `(n,)`.
        """
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        return 2.0 * (-np.log(y_c / mu_c) + (y - mu_c) / mu_c)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        r"""Gamma log-likelihood parameterized by mean `mu` and shape $\alpha = 1/\phi$.

        Evaluates the Gamma log-density at each observation using the shape parameter
        $\alpha = 1/\text{scale}$ and rate $\alpha/\mu_i$, then sums (optionally weighted)
        over observations.

        Parameters
        ----------
        y
            Observed response values (positive), shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        scale
            Dispersion parameter $\phi = 1/\alpha$.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        float
            The total log-likelihood.
        """
        from scipy.special import gammaln

        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        alpha = 1.0 / scale
        ll_i = (
            alpha * np.log(alpha)
            + (alpha - 1.0) * np.log(y_c)
            - alpha * y_c / mu_c
            - alpha * np.log(mu_c)
            - gammaln(alpha)
        )
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion parameter is fixed. Always `False` for Gamma.

        The Gamma shape parameter (and hence the dispersion `phi = 1/shape`) is estimated from
        the data during fitting rather than fixed, unlike `Poisson` or `Binomial` where the
        scale is known in advance.
        """
        return False

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        r"""Simulate Gamma-distributed response values with mean `mu` and dispersion `scale`.

        Parameters
        ----------
        mu
            Mean (fitted values), shape `(n,)`.
        scale
            Estimated dispersion parameter $\phi = 1/\alpha$.
        rng
            A `numpy.random.Generator` instance.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        mu_c = np.maximum(mu, _EPS)
        shape = 1.0 / scale
        sim_scale = mu_c * scale
        return rng.gamma(shape, sim_scale)

    def initialize(self, y: NDArray) -> NDArray:
        """Starting values for `mu`: `y` nudged away from zero.

        Since the log link requires strictly positive `mu`, values are pushed away from zero
        to avoid `log(0)` on the first P-IRLS iteration.

        Parameters
        ----------
        y
            Observed response values (positive), shape `(n,)`.

        Returns
        -------
        NDArray
            Starting values for `mu`, shape `(n,)`.
        """
        return np.maximum(y, _EPS) + 0.1

    def __repr__(self) -> str:
        return "Gamma(link='log')"
