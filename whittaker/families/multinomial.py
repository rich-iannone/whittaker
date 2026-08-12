r"""Multinomial logistic family for unordered categorical responses.

Implements a baseline-category logit model with a shared linear predictor and per-category
intercepts. Category probabilities are computed via softmax:

    P(Y = k | eta) = exp(alpha_k + beta_k * eta) / sum_j exp(alpha_j + beta_j * eta)

where the reference category K has alpha_K = 0 and beta_K = 0.

The response should be integer-coded 1, 2, ..., K.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


def _softmax(logits: NDArray) -> NDArray:
    """Row-wise softmax, numerically stable."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


class Multinomial(Family):
    r"""Multinomial logistic family for unordered categorical responses.

    `Multinomial` models a categorical response with `K` unordered levels — for example, a
    choice among several unranked options — using a baseline-category logit model. Unlike
    `OrderedCategorical`, no assumption is made about the ordering of categories or a shared
    direction of covariate effects: each non-reference category `k` gets its own intercept
    `alpha_k` and its own loading coefficient `beta_k` that rescales the shared linear predictor
    `eta`, so different categories can respond differently (even in sign) to the same covariate
    effect. The final category `K` is fixed as the reference, with `alpha_K = 0` and `beta_K =
    0`. Use this family when the response is nominal (has no natural order) with more than two
    levels; for binary outcomes use `Binomial`, and for ordinal outcomes use
    `OrderedCategorical`.

    Parameters
    ----------
    n_categories : int
        Number of response categories `K` (must be `>= 2`). Responses passed to `GAM.fit()`
        should be integer-coded `1, 2, ..., K`, with category `K` as the reference.

    Notes
    -----
    Category probabilities are obtained from a softmax over per-category logits built from the
    shared linear predictor `eta`:

    $$
    P(Y = k \mid \eta) = \frac{\exp(\alpha_k + \beta_k \eta)}{\sum_{j=1}^{K} \exp(\alpha_j + \beta_j \eta)},
    \qquad \alpha_K = \beta_K = 0.
    $$

    As with `OrderedCategorical`, this loss does not fit the standard GLM deviance framework:
    `link` and `link_inverse` are the identity on `eta`, and a custom `irls_update` drives
    P-IRLS while an inner maximum-likelihood step (`_update_params`) re-estimates the
    per-category intercepts `alpha` and loadings `beta` at each iteration. The reported deviance
    is $-2$ times the multinomial log-likelihood of the observed categories under the fitted
    probabilities,

    $$
    D(y, \hat P) = -2 \sum_{i} \log \hat P(Y_i = y_i \mid \eta_i).
    $$

    Examples
    --------
    Fit a GAM to a three-level unordered categorical response:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(-3, 3, n)
    eta = np.sin(x)

    alphas = np.array([0.0, 0.5])
    betas = np.array([1.0, -1.5])

    logits = np.column_stack(
        [alphas[0] + betas[0] * eta, alphas[1] + betas[1] * eta, np.zeros(n)]
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice([1, 2, 3], p=probs[i]) for i in range(n)], dtype=float)

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.Multinomial(n_categories=3))
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def __init__(self, n_categories: int) -> None:
        if n_categories < 2:
            raise ValueError(f"n_categories must be >= 2, got {n_categories}.")
        self._K = n_categories
        self._alphas: NDArray | None = None
        self._betas: NDArray | None = None

    @property
    def n_categories(self) -> int:
        """Number of response categories `K`.

        Returns
        -------
        int
            The value passed to the `Multinomial` constructor, i.e. the number of
            unordered categories in the response, including the reference category `K`.
        """
        return self._K

    @property
    def category_intercepts(self) -> NDArray | None:
        """Fitted per-category intercepts `alpha_1, ..., alpha_{K-1}`.

        Returns
        -------
        NDArray | None
            Array of length `K - 1` holding the intercept for each non-reference
            category, or `None` if the family has not yet been fitted (i.e.
            `irls_update` or `initialize` has not been called). The reference category
            `K` is fixed at `alpha_K = 0` and is not included in this array.
        """
        return self._alphas

    @property
    def category_loadings(self) -> NDArray | None:
        """Fitted per-category loading coefficients `beta_1, ..., beta_{K-1}`.

        Returns
        -------
        NDArray | None
            Array of length `K - 1` holding the coefficient that rescales the shared
            linear predictor `eta` for each non-reference category, or `None` if the
            family has not yet been fitted. The reference category `K` is fixed at
            `beta_K = 0` and is not included in this array.
        """
        return self._betas

    def _category_probs(self, eta: NDArray) -> NDArray:
        """Compute (n, K) matrix of category probabilities given linear predictor eta."""
        K = self._K
        n = len(eta)
        logits = np.zeros((n, K))
        for k in range(K - 1):
            logits[:, k] = self._alphas[k] + self._betas[k] * eta
        return _softmax(logits)

    def _init_params(self, y: NDArray) -> None:
        K = self._K
        y_int = np.round(y).astype(int)
        self._alphas = np.zeros(K - 1)
        for k in range(K - 1):
            p_k = np.clip(np.mean(y_int == k + 1), 0.01, 0.99)
            p_ref = np.clip(np.mean(y_int == K), 0.01, 0.99)
            self._alphas[k] = np.log(p_k / p_ref)
        self._betas = np.ones(K - 1)

    def _update_params(self, y: NDArray, eta: NDArray) -> None:
        K = self._K
        y_int = np.round(y).astype(int)

        def neg_ll(params: NDArray) -> float:
            alphas = params[: K - 1]
            betas = params[K - 1 :]
            n = len(eta)
            logits = np.zeros((n, K))
            for k in range(K - 1):
                logits[:, k] = alphas[k] + betas[k] * eta
            probs = _softmax(logits)
            probs = np.maximum(probs, _EPS)
            ll = 0.0
            for i in range(n):
                ll += np.log(probs[i, y_int[i] - 1])
            return -ll

        params0 = np.concatenate([self._alphas, self._betas])
        result = minimize(neg_ll, params0, method="L-BFGS-B")
        self._alphas = result.x[: K - 1]
        self._betas = result.x[K - 1 :]

    def link(self, mu: NDArray) -> NDArray:
        r"""Identity link, implementing the family-specific `link` for `Multinomial`.

        Because `mu` here already represents the shared linear predictor `eta` fed into
        the per-category softmax (rather than a mean response on the natural scale), the
        link is the identity: $g(\mu) = \mu$.

        Parameters
        ----------
        mu : NDArray
            Linear predictor values.

        Returns
        -------
        NDArray
            The input `mu`, unchanged.
        """
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        r"""Identity inverse link, implementing the family-specific `link_inverse` for `Multinomial`.

        Complements `link`: since `eta` is passed straight through to `_category_probs`
        for the softmax computation, the inverse link is also the identity,
        $g^{-1}(\eta) = \eta$.

        Parameters
        ----------
        eta : NDArray
            Linear predictor values.

        Returns
        -------
        NDArray
            The input `eta`, unchanged.
        """
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        r"""Derivative of the identity link, implementing the family-specific version for `Multinomial`.

        Since `link` is the identity, $g'(\mu) = 1$ everywhere.

        Parameters
        ----------
        mu : NDArray
            Linear predictor values (only used to determine output shape).

        Returns
        -------
        NDArray
            Array of ones with the same shape as `mu`.
        """
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        """Constant variance function, implementing the family-specific `variance` for `Multinomial`.

        The multinomial log-likelihood does not follow the mean-variance relationship
        used by standard GLM families; the working weights used by P-IRLS are instead
        derived directly from the curvature of the multinomial log-likelihood in
        `irls_update`. This method simply returns an array of ones so it is a no-op
        wherever a generic variance function might otherwise be referenced.

        Parameters
        ----------
        mu : NDArray
            Linear predictor values (only used to determine output shape).

        Returns
        -------
        NDArray
            Array of ones with the same shape as `mu`.
        """
        return np.ones_like(mu)

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray]:
        r"""Compute the P-IRLS working response and weights, implementing the family-specific update for `Multinomial`.

        Overrides the default GLM IRLS step: because the multinomial deviance is not a
        standard exponential-family deviance in `eta`, this method first re-estimates the
        per-category intercepts `alpha` and loadings `beta` by maximum likelihood (via
        `_update_params`), initializing them on the first call (via `_init_params`). It
        then computes, for each observation `i` with observed category `k = y_i - 1` and
        fitted probabilities `p = P(Y_i = \cdot \mid \eta_i)`, the gradient and negative
        curvature of the multinomial log-likelihood with respect to `eta_i`:

        $$
        \frac{\partial \ell_i}{\partial \eta_i} = \sum_{j=1}^{K-1} \beta_j \left(
        \mathbb{1}[j = k] - p_j \right), \qquad
        -\frac{\partial^2 \ell_i}{\partial \eta_i^2} = \sum_{j=1}^{K-1} \beta_j^2 p_j
        (1 - p_j) - 2 \sum_{j < l} \beta_j \beta_l p_j p_l.
        $$

        The working weight `W` is the (clipped) negative curvature, and the working
        response is the Newton step `z = eta + grad / W`; both feed into the outer
        P-IRLS smoothing loop.

        Parameters
        ----------
        y : NDArray
            Observed categories, coded `1, 2, ..., K`.
        mu : NDArray
            Unused; present for interface compatibility with other families.
        eta : NDArray
            Current linear predictor values.

        Returns
        -------
        tuple[NDArray, NDArray]
            The working response `z` and working weights `W`, both of shape `(n,)`.
        """
        if self._alphas is None:
            self._init_params(y)

        self._update_params(y, eta)

        K = self._K
        y_int = np.round(y).astype(int)
        probs = self._category_probs(eta)

        dl_deta = np.zeros_like(eta)
        d2l_deta2 = np.zeros_like(eta)

        for i in range(len(eta)):
            k = y_int[i] - 1
            p = probs[i]

            grad = 0.0
            hess = 0.0
            for j in range(K - 1):
                indicator = 1.0 if j == k else 0.0
                b_j = self._betas[j]
                grad += b_j * (indicator - p[j])
                hess -= b_j**2 * p[j] * (1.0 - p[j])
                for l in range(j + 1, K - 1):
                    hess += 2.0 * self._betas[j] * self._betas[l] * p[j] * p[l]

            dl_deta[i] = grad
            d2l_deta2[i] = -hess

        W = np.maximum(d2l_deta2, _EPS)
        z = eta + dl_deta / W
        return z, W

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total multinomial deviance, implementing the family-specific `deviance` for `Multinomial`.

        Computes $-2$ times the multinomial log-likelihood of the observed categories
        `y` under the fitted category probabilities implied by the linear predictor
        `eta = mu`:

        $$
        D(y, \hat P) = -2 \sum_{i=1}^{n} \log \hat P(Y_i = y_i \mid \eta_i).
        $$

        Returns `float(len(y))` as a placeholder if the family has not yet been fitted.

        Parameters
        ----------
        y : NDArray
            Observed categories, coded `1, 2, ..., K`.
        mu : NDArray
            Linear predictor `eta` (this family uses the identity link, so `mu` and
            `eta` coincide).
        weights : NDArray | None, optional
            Accepted for interface compatibility; does not currently affect the
            computed deviance.

        Returns
        -------
        float
            The total deviance.
        """
        if self._alphas is None:
            return float(len(y))
        eta = mu
        probs = self._category_probs(eta)
        y_int = np.round(y).astype(int)
        ll = 0.0
        for k in range(self._K):
            mask = y_int == (k + 1)
            if np.any(mask):
                ll += float(np.sum(np.log(np.maximum(probs[mask, k], _EPS))))
        dev = -2.0 * ll
        if weights is not None:
            return dev
        return dev

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        r"""Per-observation deviance contributions, implementing the family-specific version for `Multinomial`.

        For each observation `i`, returns $-2 \log \hat P(Y_i = y_i \mid \eta_i)$, the
        per-observation contribution to `deviance`. Returns an array of ones if the
        family has not yet been fitted.

        Parameters
        ----------
        y : NDArray
            Observed categories, coded `1, 2, ..., K`.
        mu : NDArray
            Linear predictor `eta` (this family uses the identity link).

        Returns
        -------
        NDArray
            Array of shape `(n,)` with the deviance contribution of each observation;
            these sum to the value returned by `deviance`.
        """
        if self._alphas is None:
            return np.ones_like(y)
        eta = mu
        probs = self._category_probs(eta)
        y_int = np.round(y).astype(int)
        dev = np.empty_like(y)
        for i in range(len(y)):
            dev[i] = -2.0 * np.log(np.maximum(probs[i, y_int[i] - 1], _EPS))
        return dev

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        r"""Multinomial log-likelihood, implementing the family-specific version for `Multinomial`.

        Recovers the log-likelihood from `deviance` via $\ell = -D / 2$, since
        `deviance` is defined as $-2$ times the log-likelihood.

        Parameters
        ----------
        y : NDArray
            Observed categories, coded `1, 2, ..., K`.
        mu : NDArray
            Linear predictor `eta`.
        scale : float
            Unused; present for interface compatibility with other families (the
            multinomial scale parameter is fixed, see `scale_known`).
        weights : NDArray | None, optional
            Forwarded to `deviance`.

        Returns
        -------
        float
            The total log-likelihood of `y` under the fitted model.
        """
        return -0.5 * self.deviance(y, mu, weights=weights)

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion/scale parameter is fixed rather than estimated.

        Returns
        -------
        bool
            Always `True`: the multinomial family has no free scale parameter, since
            its log-likelihood is fully determined by the fitted category
            probabilities.
        """
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        """Draw random responses from the fitted category probabilities, implementing the family-specific `simulate` for `Multinomial`.

        For each observation, computes the category probabilities from the linear
        predictor `eta = mu` via `_category_probs` and draws one category from
        `{1, ..., K}` according to those probabilities using `rng.choice`.

        Parameters
        ----------
        mu : NDArray
            Linear predictor `eta` at which to simulate.
        scale : float
            Unused; present for interface compatibility with other families.
        rng : object
            A random number generator exposing a `choice(a, p=...)` method (e.g. a
            NumPy `Generator`).

        Returns
        -------
        NDArray
            Simulated categories, coded `1, 2, ..., K`, one per row of `mu`.

        Raises
        ------
        RuntimeError
            If called before the family has been fitted.
        """
        if self._alphas is None:
            raise RuntimeError("Model must be fitted before simulation.")
        eta = mu
        probs = self._category_probs(eta)
        n = len(eta)
        y = np.empty(n)
        for i in range(n):
            y[i] = rng.choice(np.arange(1, self._K + 1), p=probs[i])
        return y

    def initialize(self, y: NDArray) -> NDArray:
        """Initialize category parameters and the starting linear predictor.

        Implements the family-specific `initialize` for `Multinomial`: estimates
        starting intercepts `alpha` (from empirical log-odds relative to the reference
        category) and loadings `beta` (set to `1`) via `_init_params`, then returns a
        starting linear predictor of all zeros for the outer P-IRLS loop to refine.

        Parameters
        ----------
        y : NDArray
            Observed categories, coded `1, 2, ..., K`, used to compute starting
            intercepts.

        Returns
        -------
        NDArray
            Initial linear predictor `eta`, an array of zeros with the same shape as
            `y`.
        """
        self._init_params(y)
        return np.zeros_like(y)

    def __repr__(self) -> str:
        return f"Multinomial(K={self._K})"
