"""Multinomial logistic family for unordered categorical responses.

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
    """Multinomial logistic family for unordered categorical responses.

    Uses a baseline-category logit model with K categories (coded 1, ..., K). The last category
    is the reference. A shared smooth linear predictor is scaled by per-category loading
    coefficients, giving each category a different relationship with the covariates.

    Parameters
    ----------
    n_categories:
        Number of response categories K (must be >= 2).
    """

    def __init__(self, n_categories: int) -> None:
        if n_categories < 2:
            raise ValueError(f"n_categories must be >= 2, got {n_categories}.")
        self._K = n_categories
        self._alphas: NDArray | None = None
        self._betas: NDArray | None = None

    @property
    def n_categories(self) -> int:
        return self._K

    @property
    def category_intercepts(self) -> NDArray | None:
        return self._alphas

    @property
    def category_loadings(self) -> NDArray | None:
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
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray]:
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
        return -0.5 * self.deviance(y, mu, weights=weights)

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
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
        self._init_params(y)
        return np.zeros_like(y)

    def __repr__(self) -> str:
        return f"Multinomial(K={self._K})"
