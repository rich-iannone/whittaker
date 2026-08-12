r"""Quantile regression family via the Extended Log-F (ELF) loss.

Implements the smooth quantile loss from Fasiolo et al. (2021) "Fast calibrated additive quantile
regression." The ELF loss is a smooth approximation to the pinball/check loss that fits within the
standard PIRLS framework via a custom IRLS update.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from whittaker.families.base import Family

_W_FLOOR = 1e-10


def _elf_loss(r: NDArray, tau: float, sigma: float) -> NDArray:
    x = -r / sigma
    softplus = np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -30, 20))))
    return tau * r + sigma * softplus


def _elf_d1(r: NDArray, tau: float, sigma: float) -> NDArray:
    return tau - 1.0 + expit(r / sigma)


def _elf_d2(r: NDArray, sigma: float) -> NDArray:
    s = expit(r / sigma)
    return (1.0 / sigma) * s * (1.0 - s)


class QuantileFamily(Family):
    r"""Quantile regression via the Extended Log-F (ELF) pseudo-family.

    `QuantileFamily` fits a single conditional quantile `tau` of the response — for example the
    median (`tau=0.5`) or the 90th percentile (`tau=0.9`) — rather than the conditional mean
    targeted by families such as `Gaussian` or `Gamma`. This is useful whenever the object of
    interest is not the average behavior of the response but a specific point in its
    distribution, e.g. modeling the upper tail of a skewed cost distribution, or building
    prediction bands by fitting several quantiles (`tau` values) side by side. Rather than the
    non-smooth pinball ("check") loss used by classical quantile regression, Whittaker uses the
    smooth Extended Log-F (ELF) approximation of Fasiolo et al. (2021), which is differentiable
    and therefore fits within the standard P-IRLS loop via a custom `irls_update`. To fit
    several quantiles jointly with a shared smoothness structure, fit one `QuantileFamily` per
    `tau` and compare/combine the resulting models, or see `ConformalPredictor` for
    distribution-free coverage guarantees around a fitted mean model.

    Parameters
    ----------
    tau : float, default=0.5
        Quantile level to estimate, in `(0, 1)`. `tau=0.5` corresponds to median regression;
        smaller values target lower quantiles and larger values target upper quantiles.
    sigma : float, default=1.0
        Bandwidth (smoothing) parameter controlling how closely the ELF loss approximates the
        non-smooth pinball loss. Smaller values give a sharper, more faithful approximation to
        the pinball loss (and to the check-function optimum) but a less smooth optimization
        surface; larger values give a smoother but more biased approximation. `sigma` may be
        adjusted after construction via the `sigma` property, e.g. to anneal it across fitting
        iterations.

    Notes
    -----
    `QuantileFamily` has no meaningful link or variance function in the usual GLM sense — `link`
    and `link_inverse` are the identity on `eta` — because fitting instead minimizes the ELF loss
    directly. For a residual `r = y - mu`, the ELF loss is

    $$
    \rho_{\tau,\sigma}(r) = \tau r + \sigma \log\!\left(1 + e^{-r/\sigma}\right),
    $$

    which converges to the pinball loss
    $\rho_\tau(r) = \tau r \, \mathbb{1}[r \ge 0] - (1-\tau) r \, \mathbb{1}[r < 0]$ as
    $\sigma \to 0$. Its first and second derivatives with respect to `mu`,

    $$
    \frac{\partial \rho}{\partial \mu} = -\left[\tau - 1 + \operatorname{expit}(r/\sigma)\right],
    \qquad
    \frac{\partial^2 \rho}{\partial \mu^2} = \frac{1}{\sigma}\, s (1 - s), \quad
    s = \operatorname{expit}(r/\sigma),
    $$

    supply the working response `z` and working weight `W` used by the custom `irls_update`. The
    reported "deviance" is `2 * sum(ELF loss)`, so that it reduces to twice the usual pinball
    loss in the limit `sigma -> 0`.

    Examples
    --------
    Fit the 10th, 50th, and 90th percentile curves of a heteroscedastic response:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.sin(x)
    noise_scale = 0.2 + 0.3 * np.abs(np.cos(x))
    y = mu + rng.normal(0, noise_scale, n)

    data = {"x": x, "y": y}

    for tau in (0.1, 0.5, 0.9):
        model = wk.GAM("y ~ s(x)", family=wk.QuantileFamily(tau=tau))
        model.fit(data, method="REML")
        print(f"tau={tau}:")
        print(model.summary())
    ```
    """

    def __init__(self, tau: float = 0.5, sigma: float = 1.0) -> None:
        if not 0 < tau < 1:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self._tau = tau
        self._sigma = sigma

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"sigma must be positive, got {value}")
        self._sigma = value

    def link(self, mu: NDArray) -> NDArray:
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray]:
        r = y - mu
        score = _elf_d1(r, self._tau, self._sigma)
        W = np.maximum(_elf_d2(r, self._sigma), _W_FLOOR)
        z = eta + score / W
        return z, W

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        d = 2.0 * _elf_loss(y - mu, self._tau, self._sigma)
        if weights is not None:
            d = weights * d
        return float(np.sum(d))

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        return 2.0 * _elf_loss(y - mu, self._tau, self._sigma)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        ll_i = -_elf_loss(y - mu, self._tau, self._sigma)
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        u = rng.uniform(size=mu.shape)
        return np.where(
            u < self._tau,
            mu + self._sigma * np.log(u / self._tau),
            mu - self._sigma * np.log((1 - u) / (1 - self._tau)),
        )

    def initialize(self, y: NDArray) -> NDArray:
        return y.copy()

    def __repr__(self) -> str:
        return f"Quantile(tau={self._tau}, sigma={self._sigma})"
