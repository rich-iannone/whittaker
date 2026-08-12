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
        r"""Inverse Gaussian variance function: $V(\mu) = \mu^3$.

        The variance grows with the cube of the mean, giving this family the heaviest right
        tail among Whittaker's built-in continuous positive families.

        Parameters
        ----------
        mu
            Conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Variance values $\mu^3$, shape `(n,)`.
        """
        return np.maximum(mu, _EPS) ** 3

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total Inverse Gaussian deviance, the (weighted) sum of `unit_deviance`.

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
        r"""Per-observation Inverse Gaussian deviance contributions.

        Computes $d_i = (y_i - \hat\mu_i)^2 / (\hat\mu_i^2\, y_i)$.

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
        return (y_c - mu_c) ** 2 / (mu_c**2 * y_c)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        r"""Inverse Gaussian log-likelihood evaluated at dispersion `scale`.

        Parameters
        ----------
        y
            Observed response values (positive), shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        scale
            Dispersion parameter $\phi$.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        float
            The total log-likelihood.
        """
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, _EPS)
        ll_i = -0.5 * np.log(2.0 * np.pi * scale * y_c**3) - (y_c - mu_c) ** 2 / (
            2.0 * scale * mu_c**2 * y_c
        )
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion parameter is fixed. Always `False` for InverseGaussian.

        The dispersion `phi` is estimated from the data during fitting rather than fixed,
        analogous to how `Gamma` and `Gaussian` estimate their own dispersion parameters.
        This affects how `GAM.fit()` and `GAM.summary()` treat the scale.
        """
        return False

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        r"""Simulate Inverse Gaussian (Wald) response values with mean `mu` and dispersion `scale`.

        Draws are generated via Michael, Schucany & Haas's transformation of a chi-squared
        variate, then adjusted with a uniform random variable to select between the two
        candidate roots.

        Parameters
        ----------
        mu
            Mean (fitted values), shape `(n,)`.
        scale
            Dispersion parameter $\phi$.
        rng
            A `numpy.random.Generator` instance.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        mu_c = np.maximum(mu, _EPS)
        lam = mu_c / scale
        n = len(mu_c)
        v = rng.standard_normal(n) ** 2
        x = (
            mu_c
            + (mu_c**2 * v) / (2.0 * lam)
            - mu_c / (2.0 * lam) * np.sqrt(4.0 * mu_c * lam * v + mu_c**2 * v**2)
        )
        u = rng.uniform(size=n)
        result = np.where(u <= mu_c / (mu_c + x), x, mu_c**2 / x)
        return np.maximum(result, _EPS)

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
        return "InverseGaussian(link='log')"
