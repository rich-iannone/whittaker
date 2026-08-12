"""Ordered categorical (proportional odds) family."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class OrderedCategorical(Family):
    r"""Ordered categorical (proportional odds / cumulative logit) family.

    `OrderedCategorical` models an ordinal response — one with a small number of categories that
    have a natural order but no meaningful numeric spacing, such as a Likert scale
    ("disagree" / "neutral" / "agree") or a severity grade — using the proportional-odds
    cumulative logit model. A single set of `K - 1` ordered cutpoints (thresholds) `alpha_1 <
    alpha_2 < ... < alpha_{K-1}` is estimated jointly with the smooth/linear predictor `eta`, and
    each cutpoint defines a binary split between "category `k` or below" versus "above category
    `k`". Because the same `eta` (and hence the same covariate effects) is shared across all
    thresholds, covariate effects are assumed to shift the log-odds of being in a higher category
    by the same amount at every threshold — the proportional-odds assumption. Use this family
    for ordinal outcomes with three or more ordered levels; for a two-level (binary) response,
    use `Binomial` instead, and for unordered categorical responses, use `Multinomial`.

    Parameters
    ----------
    n_categories : int
        Number of ordered response categories `K` (must be `>= 2`). Responses passed to
        `GAM.fit()` should be integer-coded `1, 2, ..., K`.

    Notes
    -----
    The response, without an intercept in the design matrix (since the cutpoints absorb it), is
    modeled through cumulative probabilities:

    $$
    P(Y \le k \mid \eta) = \operatorname{expit}(\alpha_k - \eta), \qquad k = 1, \dots, K-1,
    $$

    which is equivalent to a logit link on each cumulative probability,
    $g(P(Y \le k)) = \alpha_k - \eta$. Category probabilities follow by differencing:

    $$
    P(Y = 1) = \operatorname{expit}(\alpha_1 - \eta), \qquad
    P(Y = K) = 1 - \operatorname{expit}(\alpha_{K-1} - \eta),
    $$

    and for interior categories $1 < k < K$,

    $$
    P(Y = k) = \operatorname{expit}(\alpha_k - \eta) - \operatorname{expit}(\alpha_{k-1} - \eta).
    $$

    Because this loss does not fit the standard GLM deviance framework, `link` and
    `link_inverse` are the identity on `eta`, and fitting instead uses a custom `irls_update`
    together with an inner maximum-likelihood step (`_update_cutpoints`) that re-estimates the
    cutpoints `alpha` at each P-IRLS iteration. The deviance reported is $-2$ times the
    multinomial log-likelihood of the observed categories under the fitted probabilities.

    Examples
    --------
    Fit a GAM to a four-level ordinal response with a smooth covariate effect:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(-3, 3, n)
    eta = np.sin(x)
    cutpoints = np.array([-1.5, 0.0, 1.5])

    y = np.empty(n)
    for i in range(n):
        probs = np.diff(
            np.concatenate([[0.0], expit(cutpoints - eta[i]), [1.0]])
        )
        y[i] = rng.choice([1, 2, 3, 4], p=probs)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.OrderedCategorical(n_categories=4))
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def __init__(self, n_categories: int) -> None:
        if n_categories < 2:
            raise ValueError(f"n_categories must be >= 2, got {n_categories}.")
        self._K = n_categories
        self._cutpoints: NDArray | None = None

    @property
    def n_categories(self) -> int:
        return self._K

    @property
    def cutpoints(self) -> NDArray | None:
        return self._cutpoints

    def _category_probs(self, eta: NDArray) -> NDArray:
        K = self._K
        alpha = self._cutpoints
        n = len(eta)
        probs = np.empty((n, K))
        cum_prev = np.zeros(n)
        for k in range(K - 1):
            cum_k = expit(alpha[k] - eta)
            probs[:, k] = np.maximum(cum_k - cum_prev, _EPS)
            cum_prev = cum_k
        probs[:, K - 1] = np.maximum(1.0 - cum_prev, _EPS)
        return probs

    def _init_cutpoints(self, y: NDArray) -> NDArray:
        K = self._K
        n = len(y)
        alphas = np.zeros(K - 1)
        for k in range(K - 1):
            p_k = np.clip(np.mean(y <= k + 1), 0.01, 0.99)
            alphas[k] = np.log(p_k / (1.0 - p_k))
        return alphas

    def _update_cutpoints(self, y: NDArray, eta: NDArray) -> None:
        K = self._K
        y_int = np.round(y).astype(int)

        def neg_ll(alpha_raw: NDArray) -> float:
            alpha = np.cumsum(np.concatenate([[alpha_raw[0]], np.exp(alpha_raw[1:])]))
            n = len(eta)
            probs = np.empty((n, K))
            cum_prev = np.zeros(n)
            for k in range(K - 1):
                cum_k = expit(alpha[k] - eta)
                probs[:, k] = np.maximum(cum_k - cum_prev, _EPS)
                cum_prev = cum_k
            probs[:, K - 1] = np.maximum(1.0 - cum_prev, _EPS)

            ll = 0.0
            for k in range(K):
                mask = y_int == (k + 1)
                if np.any(mask):
                    ll += float(np.sum(np.log(probs[mask, k])))
            return -ll

        alpha0 = self._cutpoints
        raw0 = np.concatenate([[alpha0[0]], np.log(np.maximum(np.diff(alpha0), 1e-6))])
        result = minimize(neg_ll, raw0, method="L-BFGS-B")
        alpha_opt = np.cumsum(np.concatenate([[result.x[0]], np.exp(result.x[1:])]))
        self._cutpoints = alpha_opt

    def link(self, mu: NDArray) -> NDArray:
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray]:
        if self._cutpoints is None:
            self._cutpoints = self._init_cutpoints(y)

        self._update_cutpoints(y, eta)

        K = self._K
        y_int = np.round(y).astype(int)
        probs = self._category_probs(eta)

        dl_deta = np.zeros_like(eta)
        d2l_deta2 = np.zeros_like(eta)

        for i in range(len(eta)):
            k = y_int[i] - 1
            p_k = probs[i, k]
            alpha = self._cutpoints

            if k == 0:
                g = expit(alpha[0] - eta[i])
                dg = g * (1.0 - g)
                dl_deta[i] = -dg / p_k
                d2l_deta2[i] = dg * (1.0 - 2.0 * g) / p_k + (dg / p_k) ** 2
            elif k == K - 1:
                g_prev = expit(alpha[K - 2] - eta[i])
                dg_prev = g_prev * (1.0 - g_prev)
                dl_deta[i] = dg_prev / p_k
                d2l_deta2[i] = -dg_prev * (1.0 - 2.0 * g_prev) / p_k + (dg_prev / p_k) ** 2
            else:
                g_k = expit(alpha[k] - eta[i])
                g_prev = expit(alpha[k - 1] - eta[i])
                dg_k = g_k * (1.0 - g_k)
                dg_prev = g_prev * (1.0 - g_prev)
                dl_deta[i] = (-dg_k + dg_prev) / p_k
                d2l_deta2[i] = (dg_k * (1.0 - 2.0 * g_k) - dg_prev * (1.0 - 2.0 * g_prev)) / p_k + (
                    (-dg_k + dg_prev) / p_k
                ) ** 2

        W = np.maximum(d2l_deta2, _EPS)
        z = eta + dl_deta / W
        return z, W

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        if self._cutpoints is None:
            return float(len(y))
        eta = mu
        probs = self._category_probs(eta)
        y_int = np.round(y).astype(int)
        ll = 0.0
        for k in range(self._K):
            mask = y_int == (k + 1)
            if np.any(mask):
                ll += float(np.sum(np.log(probs[mask, k])))
        return -2.0 * ll

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        if self._cutpoints is None:
            return np.ones_like(y)
        eta = mu
        probs = self._category_probs(eta)
        y_int = np.round(y).astype(int)
        dev = np.empty_like(y)
        for i in range(len(y)):
            dev[i] = -2.0 * np.log(probs[i, y_int[i] - 1])
        return dev

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        return -0.5 * self.deviance(y, mu, weights=weights)

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        if self._cutpoints is None:
            raise RuntimeError("Model must be fitted before simulation.")
        eta = mu
        probs = self._category_probs(eta)
        n = len(eta)
        y = np.empty(n)
        for i in range(n):
            y[i] = rng.choice(np.arange(1, self._K + 1), p=probs[i])
        return y

    def initialize(self, y: NDArray) -> NDArray:
        self._cutpoints = self._init_cutpoints(y)
        return np.zeros_like(y)

    def __repr__(self) -> str:
        return f"OrderedCategorical(K={self._K})"
