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
        r"""Apply the logit link: $\eta = \log(\mu / (1-\mu))$.

        Parameters
        ----------
        mu
            Conditional mean values (probabilities), shape `(n,)`. Must lie in `(0, 1)`.

        Returns
        -------
        NDArray
            Linear predictor (log-odds) values, shape `(n,)`.
        """
        return logit(mu)

    def link_inverse(self, eta: NDArray) -> NDArray:
        r"""Apply the inverse logit link (logistic sigmoid): $\mu = 1 / (1 + e^{-\eta})$.

        Parameters
        ----------
        eta
            Linear predictor (log-odds) values, shape `(n,)`.

        Returns
        -------
        NDArray
            Conditional mean values (probabilities), shape `(n,)`, always in `(0, 1)`.
        """
        return expit(eta)

    def link_derivative(self, mu: NDArray) -> NDArray:
        r"""Derivative of the logit link: $g'(\mu) = 1 / (\mu(1-\mu))$.

        Parameters
        ----------
        mu
            Conditional mean values (probabilities), shape `(n,)`; clipped away from `0` and
            `1` to avoid division by zero.

        Returns
        -------
        NDArray
            Derivative values, shape `(n,)`.
        """
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return 1.0 / (mu_c * (1.0 - mu_c))

    def variance(self, mu: NDArray) -> NDArray:
        r"""Binomial variance function: $V(\mu) = \mu(1-\mu)$.

        Largest at $\mu = 0.5$ and shrinking toward zero as $\mu$ approaches either boundary.

        Parameters
        ----------
        mu
            Conditional mean values (probabilities), shape `(n,)`.

        Returns
        -------
        NDArray
            Variance values $\mu(1-\mu)$, shape `(n,)`.
        """
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        return mu_c * (1.0 - mu_c)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total binomial deviance, the (weighted) sum of `unit_deviance`.

        Parameters
        ----------
        y
            Observed response values (0/1 or proportions in `[0, 1]`), shape `(n,)`.
        mu
            Fitted conditional mean values (probabilities), shape `(n,)`.
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
        r"""Per-observation binomial deviance contributions.

        Computes
        $d_i = 2 \left[ y_i \log(y_i/\hat\mu_i) + (1-y_i)\log((1-y_i)/(1-\hat\mu_i)) \right]$,
        with the usual conventions that the $y_i \log(\cdot)$ term vanishes when $y_i = 0$ and
        the $(1-y_i)\log(\cdot)$ term vanishes when $y_i = 1$.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values (probabilities), shape `(n,)`.

        Returns
        -------
        NDArray
            Per-observation deviance contributions, shape `(n,)`.
        """
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
        r"""Bernoulli/binomial log-likelihood $\ell_i = y_i \log(\mu_i) + (1-y_i)\log(1-\mu_i)$.

        The `scale` argument is accepted for interface compatibility but ignored, since the
        Binomial distribution has no free dispersion parameter (`scale_known` is `True`).

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values (probabilities), shape `(n,)`.
        scale
            Ignored.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        float
            The total log-likelihood.
        """
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        ll_i = y * np.log(mu_c) + (1.0 - y) * np.log(1.0 - mu_c)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    def log_lik_pointwise(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> NDArray:
        mu_c = np.clip(mu, _EPS, 1.0 - _EPS)
        ll_i = y * np.log(mu_c) + (1.0 - y) * np.log(1.0 - mu_c)
        if weights is not None:
            ll_i = weights * ll_i
        return ll_i

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion parameter is fixed. Always `True` for Binomial.

        The Bernoulli/Binomial distribution has no free dispersion parameter, so the scale is
        fixed at `1` and is never estimated during fitting. This affects how `GAM.summary()`
        reports the scale and how many degrees of freedom are attributed to dispersion.
        """
        return True

    def simulate(self, mu: NDArray, scale: float, rng: np.random.Generator) -> NDArray:
        """Simulate Bernoulli response values with success probability `mu`.

        Parameters
        ----------
        mu
            Success probabilities (fitted values), shape `(n,)`.
        scale
            Ignored (the Binomial distribution has no free dispersion parameter).
        rng
            A `numpy.random.Generator` instance.

        Returns
        -------
        NDArray
            Simulated 0/1 response values, shape `(n,)`.
        """
        p = np.clip(mu, _EPS, 1.0 - _EPS)
        return rng.binomial(1, p).astype(float)

    def initialize(self, y: NDArray) -> NDArray:
        """Starting values for `mu`: a constant shrunk toward the overall mean of `y`.

        Every observation is initialized to `(mean(y) + 0.5) / 2`, a value strictly inside
        `(0, 1)` regardless of whether `y` contains only `0`s, only `1`s, or a mix, avoiding
        `logit(0)` or `logit(1)` on the first P-IRLS iteration.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.

        Returns
        -------
        NDArray
            Starting values for `mu`, shape `(n,)`.
        """
        return np.full_like(y, fill_value=(np.mean(y) + 0.5) / 2.0)

    def __repr__(self) -> str:
        return "Binomial(link='logit')"
