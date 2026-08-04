"""Tests for whittaker.families.gaussian (Gaussian family with identity link)."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from whittaker.families.gaussian import Gaussian

RNG = np.random.default_rng(23)


class TestLink:
    def test_link_is_identity(self) -> None:
        g = Gaussian()
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(g.link(mu), mu)

    def test_link_inverse_is_identity(self) -> None:
        g = Gaussian()
        eta = np.array([-1.0, 0.0, 5.0])
        assert_allclose(g.link_inverse(eta), eta)

    def test_link_derivative_is_one(self) -> None:
        g = Gaussian()
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(g.link_derivative(mu), 1.0)

    def test_link_roundtrip(self) -> None:
        g = Gaussian()
        mu = RNG.standard_normal(50)
        assert_allclose(g.link_inverse(g.link(mu)), mu)


class TestVariance:
    def test_variance_is_one(self) -> None:
        g = Gaussian()
        mu = np.array([1.0, -5.0, 100.0])
        assert_allclose(g.variance(mu), 1.0)


class TestDeviance:
    def test_deviance_is_sum_squared_residuals(self) -> None:
        g = Gaussian()
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.5, 2.0])
        expected = np.sum((y - mu) ** 2)
        assert_allclose(g.deviance(y, mu), expected)

    def test_deviance_zero_at_perfect_fit(self) -> None:
        g = Gaussian()
        y = np.array([1.0, 2.0, 3.0])
        assert_allclose(g.deviance(y, y), 0.0)


class TestLogLikelihood:
    def test_log_likelihood_value(self) -> None:
        g = Gaussian()
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        scale = 1.0
        n = len(y)
        expected = -0.5 * n * np.log(2 * np.pi * scale)
        assert_allclose(g.log_likelihood(y, mu, scale), expected)

    def test_log_likelihood_decreases_with_worse_fit(self) -> None:
        g = Gaussian()
        y = np.array([1.0, 2.0, 3.0])
        ll_good = g.log_likelihood(y, y, 1.0)
        ll_bad = g.log_likelihood(y, np.zeros(3), 1.0)
        assert ll_good > ll_bad


class TestInitialize:
    def test_initialize_returns_copy_of_y(self) -> None:
        g = Gaussian()
        y = np.array([1.0, 2.0, 3.0])
        mu0 = g.initialize(y)
        assert_allclose(mu0, y)
        mu0[0] = 999.0
        assert y[0] == 1.0


class TestRepr:
    def test_repr(self) -> None:
        g = Gaussian()
        assert "Gaussian" in repr(g)
        assert "identity" in repr(g)
