"""Ordered categorical (proportional odds) family."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from whittaker.families.base import Family

_EPS = np.finfo(float).eps


class OrderedCategorical(Family):
    """Ordered categorical (proportional odds / cumulative logit) family.

    Models ordinal responses with `K` categories (coded 1, 2, ..., K) using the cumulative logit
    model:

        P(Y <= k | eta) = expit(alpha_k - eta)

    where `alpha_1 < alpha_2 < ... < alpha_{K-1}` are ordered cutpoints and `eta = X @ beta` is the
    linear predictor (without intercept — the cutpoints absorb it).

    Category probabilities::

        P(Y = 1) = expit(alpha_1 - eta)
        P(Y = k) = expit(alpha_k - eta) - expit(alpha_{k-1} - eta)   for 1 < k < K
        P(Y = K) = 1 - expit(alpha_{K-1} - eta)

    Parameters
    ----------
    n_categories:
        Number of ordered response categories K (must be >= 2).
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
