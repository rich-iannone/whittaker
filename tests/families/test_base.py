"""Tests for whittaker.families.base (the abstract Family base class)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family


class _MinimalFamily(Family):
    """Minimal concrete Family that does not override `initialize`.

    Used to exercise the default `Family.initialize` implementation, since every shipped
    family overrides it with a distribution-specific starting value.
    """

    def link(self, mu: NDArray) -> NDArray:
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        return float(np.sum((y - mu) ** 2))

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        return 0.0

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        return mu


class _LLFamily(_MinimalFamily):
    """Family with a real log-likelihood to exercise the base fallback for log_lik_pointwise."""

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        ll_i = -0.5 * np.log(2 * np.pi * scale) - 0.5 * (y - mu) ** 2 / scale
        if weights is not None:
            ll_i = weights * ll_i
        return float(np.sum(ll_i))


class TestLogLikPointwiseFallback:
    def test_shape(self) -> None:
        fam = _LLFamily()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.1, 1.9, 3.2, 3.8])
        result = fam.log_lik_pointwise(y, mu, scale=1.0)
        assert result.shape == (4,)

    def test_sum_matches_log_likelihood(self) -> None:
        fam = _LLFamily()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.1, 1.9, 3.2, 3.8])
        scale = 0.5
        pw = fam.log_lik_pointwise(y, mu, scale)
        ll = fam.log_likelihood(y, mu, scale)
        np.testing.assert_allclose(np.sum(pw), ll, rtol=1e-10)

    def test_weighted_sum_matches_log_likelihood(self) -> None:
        fam = _LLFamily()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.1, 1.9, 3.2, 3.8])
        w = np.array([1.0, 2.0, 0.5, 1.5])
        scale = 0.5
        pw = fam.log_lik_pointwise(y, mu, scale, weights=w)
        ll = fam.log_likelihood(y, mu, scale, weights=w)
        np.testing.assert_allclose(np.sum(pw), ll, rtol=1e-10)


class TestDefaultInitialize:
    def test_initialize_returns_copy_of_y(self) -> None:
        fam = _MinimalFamily()
        y = np.array([1.0, 2.0, 3.0])
        mu0 = fam.initialize(y)
        np.testing.assert_allclose(mu0, y)
        assert mu0 is not y
