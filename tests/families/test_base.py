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


class TestDefaultInitialize:
    def test_initialize_returns_copy_of_y(self) -> None:
        fam = _MinimalFamily()
        y = np.array([1.0, 2.0, 3.0])
        mu0 = fam.initialize(y)
        np.testing.assert_allclose(mu0, y)
        assert mu0 is not y
