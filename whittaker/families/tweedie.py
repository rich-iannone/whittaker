r"""Tweedie family with log link."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class Tweedie(Family):
    r"""Tweedie family with log link.

    The Tweedie family is an exponential dispersion model whose variance function is a power of
    the mean, `V(mu) = mu^p`. It is most widely used for `1 < p < 2`, the compound Poisson-Gamma
    case: a distribution with a point mass at zero and a continuous, right-skewed density on the
    positive reals. This shape is a natural fit for aggregate insurance claims (many
    policyholders have zero claims, and the rest have positive, Gamma-like claim amounts),
    precipitation totals, and biomass or catch data with structural zeros. Whittaker also
    supports `p > 2` for purely positive, heavy-tailed continuous data (including the inverse
    Gaussian case at `p = 3`; see `InverseGaussian`). The Poisson (`p = 1`) and Gamma (`p = 2`)
    boundary cases are excluded here — use `Poisson` or `Gamma` directly for exact likelihood
    computations at those values.

    Parameters
    ----------
    p : float, default=1.5
        Variance power. Must satisfy `1 < p < 2` (compound Poisson-Gamma, the typical choice for
        insurance-type data with zeros) or `p > 2` (positive continuous, heavy-tailed data).
        Values of exactly `1` or `2` are rejected because they correspond to `Poisson` and
        `Gamma`, which have simpler, exact deviance and likelihood formulas. If `p` is unknown,
        use `tw()` to estimate it from the data instead of fixing it here.

    Notes
    -----
    The link is the natural logarithm:

    $$
    g(\mu) = \log(\mu)
    $$

    The variance function is a power of the mean:

    $$
    V(\mu) = \mu^{p},
    $$

    which interpolates between `Poisson`-like behavior (`p` near 1) and `Gamma`-like behavior
    (`p` near 2), or heavier-tailed behavior for `p > 2`. The unit deviance is

    $$
    d(y, \hat\mu) = 2 \left[
    \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y\,\hat\mu^{1-p}}{1-p} + \frac{\hat\mu^{2-p}}{2-p}
    \right],
    $$

    summed over observations to give the total deviance. Because the Tweedie density has no
    closed form for `1 < p < 2`, the log-likelihood is evaluated using the Dunn & Smyth (2005)
    saddlepoint approximation.

    Examples
    --------
    Fit a GAM to compound Poisson-Gamma data with a point mass at zero:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(1.0 + 0.5 * np.sin(x))

    p = 1.5
    scale = 1.0
    lam = mu ** (2 - p) / (scale * (2 - p))
    alpha = (2 - p) / (p - 1)
    gamma_scale = scale * (p - 1) * mu ** (p - 1)
    n_claims = rng.poisson(lam)
    y = np.array(
        [rng.gamma(alpha, gamma_scale[i], size=n_claims[i]).sum() for i in range(n)]
    )

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.Tweedie(p=1.5))
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def __init__(self, p: float = 1.5) -> None:
        if p <= 1 or p == 2:
            raise ValueError(f"Tweedie variance power p must be in (1, 2) or (2, ∞), got {p}.")
        self._p = float(p)

    @property
    def p(self) -> float:
        """Variance power."""
        return self._p

    def link(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, _EPS))

    def link_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(np.clip(eta, -30.0, 30.0))

    def link_derivative(self, mu: NDArray) -> NDArray:
        return 1.0 / np.maximum(mu, _EPS)

    def variance(self, mu: NDArray) -> NDArray:
        return np.maximum(mu, _EPS) ** self._p

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = self.unit_deviance(y, mu)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        p = self._p
        mu_c = np.maximum(mu, _EPS)
        y_c = np.maximum(y, 0.0)

        a = np.where(y_c > 0, y_c ** (2 - p) / ((1 - p) * (2 - p)), 0.0)
        b = y_c * mu_c ** (1 - p) / (1 - p)
        c = mu_c ** (2 - p) / (2 - p)
        return 2.0 * (a - b + c)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        ll_i = self._saddlepoint_log_density(y, mu, scale)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    def _saddlepoint_log_density(self, y: NDArray, mu: NDArray, scale: float) -> NDArray:
        """Saddle-point approximation to the Tweedie log-density (Dunn & Smyth 2005)."""
        p = self._p
        mu_c = np.maximum(mu, _EPS)
        y_c = np.asarray(y, dtype=float)

        dev = self.unit_deviance(y_c, mu_c)
        ll = -dev / (2.0 * scale)

        pos = y_c > _EPS
        ll_pos = np.zeros_like(ll)
        ll_pos[pos] = -0.5 * np.log(2.0 * np.pi * scale * y_c[pos] ** p)
        ll_pos[~pos] = -(mu_c[~pos] ** (2 - p)) / (scale * (2 - p))
        ll += ll_pos

        return ll

    @property
    def scale_known(self) -> bool:
        return False

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        p = self._p
        mu_c = np.maximum(mu, _EPS)
        n = len(mu_c)

        if 1 < p < 2:
            lam = mu_c ** (2 - p) / (scale * (2 - p))
            alpha = (2 - p) / (p - 1)
            gamma_scale = scale * (p - 1) * mu_c ** (p - 1)

            N = rng.poisson(lam)
            result = np.zeros(n)
            for i in range(n):
                if N[i] > 0:
                    result[i] = np.sum(rng.gamma(alpha, gamma_scale[i], size=N[i]))
            return result
        else:
            var = scale * mu_c**p
            gamma_shape = mu_c**2 / var
            gamma_scale = var / mu_c
            return rng.gamma(gamma_shape, gamma_scale)

    def initialize(self, y: NDArray) -> NDArray:
        y_init = np.maximum(y, 0.1)
        return y_init + 0.1

    def __repr__(self) -> str:
        return f"Tweedie(p={self._p}, link='log')"
