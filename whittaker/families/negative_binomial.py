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
        """Overdispersion (size) parameter `theta` of the NB2 distribution.

        Smaller values imply greater overdispersion relative to Poisson; larger values make
        the distribution approach Poisson as `theta -> infinity`. This value is used by
        `variance`, `unit_deviance`, `log_likelihood`, and `simulate`.
        """
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        """Set the overdispersion parameter `theta`.

        Parameters
        ----------
        value
            New value for `theta`. Must be positive.

        Raises
        ------
        ValueError
            If `value` is not positive.
        """
        if value <= 0:
            raise ValueError(f"theta must be positive, got {value}.")
        self._theta = float(value)

    def link(self, mu: NDArray) -> NDArray:
        r"""Apply the log link: $\eta = \log(\mu)$.

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
        r"""NB2 variance function: $V(\mu) = \mu + \mu^2/\theta$.

        The variance always exceeds the mean by the extra term $\mu^2/\theta$, which vanishes
        as `theta -> infinity` (recovering the Poisson variance `V(mu) = mu`).

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Variance values $\mu + \mu^2/\theta$, shape `(n,)`.
        """
        mu_c = np.maximum(mu, _EPS)
        return mu_c + mu_c**2 / self._theta

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total NB2 deviance, the (weighted) sum of `unit_deviance`.

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
        r"""Per-observation NB2 deviance contributions.

        Computes
        $d_i = 2 \left[ y_i \log(y_i/\hat\mu_i)
        - (y_i + \theta) \log\!\left(\frac{y_i+\theta}{\hat\mu_i+\theta}\right) \right]$,
        with the usual convention that the $y_i \log(\cdot)$ term vanishes when $y_i = 0$.

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
        r"""NB2 log-likelihood evaluated using the current `theta`.

        The `scale` argument is accepted for interface compatibility but ignored; the
        overdispersion is instead governed by `theta` (see `scale_known`).

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
        """Whether the dispersion parameter is fixed. Always `True` for NegativeBinomial.

        The NB2 dispersion is governed entirely by `theta` rather than a separate scale
        parameter estimated during P-IRLS, so the scale is treated as fixed at `1`. `theta`
        itself is refined by an outer iteration around P-IRLS rather than the usual scale
        estimation.
        """
        return True

    def simulate(self, mu: NDArray, scale: float, rng: np.random.Generator) -> NDArray:
        """Simulate NB2-distributed response values with mean `mu` and the current `theta`.

        Parameters
        ----------
        mu
            Mean (fitted values), shape `(n,)`.
        scale
            Ignored; overdispersion is governed by `theta`.
        rng
            A `numpy.random.Generator` instance.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        mu_c = np.maximum(mu, _EPS)
        theta = self._theta
        p = theta / (mu_c + theta)
        return rng.negative_binomial(theta, p).astype(float)

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
        return f"NegativeBinomial(theta={self._theta:.4g}, link='log')"
