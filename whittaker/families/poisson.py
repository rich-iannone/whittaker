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
        r"""Apply the log link: $\eta = \log(\mu)$.

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`. Should be positive; values are clipped to
            `_EPS` to avoid `log(0)`.

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
        r"""Poisson variance function: $V(\mu) = \mu$.

        The variance equals the mean, so `Var(Y) = phi * V(mu) = mu` since `phi = 1` for
        Poisson (see `scale_known`).

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Variance values $V(\mu) = \mu$, shape `(n,)`.
        """
        return np.maximum(mu, _EPS)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total Poisson deviance, the (weighted) sum of `unit_deviance`.

        Parameters
        ----------
        y
            Observed response values (counts), shape `(n,)`.
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
        r"""Per-observation Poisson deviance contributions.

        Computes
        $d_i = 2 \left[ y_i \log(y_i / \hat\mu_i) - (y_i - \hat\mu_i) \right]$,
        with the convention $y_i \log(y_i / \hat\mu_i) = 0$ when $y_i = 0$.

        Parameters
        ----------
        y
            Observed response values (counts), shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Per-observation deviance contributions, shape `(n,)`.
        """
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
        r"""Poisson log-likelihood $\ell_i = y_i \log(\mu_i) - \mu_i - \log(y_i!)$.

        The `scale` argument is accepted for interface compatibility but ignored, since the
        Poisson distribution has no free dispersion parameter (`scale_known` is `True`).

        Parameters
        ----------
        y
            Observed response values (counts), shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        scale
            Ignored.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        float
            The total log-likelihood.
        """
        from scipy.special import gammaln

        mu_c = np.maximum(mu, _EPS)
        ll_i = y * np.log(mu_c) - mu_c - gammaln(y + 1.0)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion parameter is fixed. Always `True` for Poisson.

        The Poisson distribution has no free dispersion parameter, so the scale is fixed at
        `1` and is never estimated during fitting. This affects how `GAM.summary()` reports
        the scale and how many degrees of freedom are attributed to dispersion estimation.
        """
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        """Simulate Poisson-distributed response values with mean `mu`.

        Parameters
        ----------
        mu
            Mean (fitted values), shape `(n,)`.
        scale
            Ignored (the Poisson distribution has no free dispersion parameter).
        rng
            A `numpy.random.Generator` instance.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        return rng.poisson(np.maximum(mu, _EPS)).astype(float)

    def initialize(self, y: NDArray) -> NDArray:
        """Starting values for `mu`: `y` nudged away from zero.

        Since the log link requires strictly positive `mu`, small or zero counts are pushed
        away from zero to avoid `log(0)` on the first P-IRLS iteration.

        Parameters
        ----------
        y
            Observed response values (counts), shape `(n,)`.

        Returns
        -------
        NDArray
            Starting values for `mu`, shape `(n,)`.
        """
        return np.maximum(y, 0.1) + 0.1

    def __repr__(self) -> str:
        return "Poisson(link='log')"
