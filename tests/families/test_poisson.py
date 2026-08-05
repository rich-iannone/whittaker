"""Tests for whittaker.families.poisson (Poisson family with log link)."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from whittaker.families.poisson import Poisson

RNG = np.random.default_rng(23)


class TestLink:
    def test_log_link(self) -> None:
        g = Poisson()
        mu = np.array([1.0, 2.0, np.e])
        expected = np.log(mu)
        assert_allclose(g.link(mu), expected, atol=1e-12)

    def test_link_inverse_is_exp(self) -> None:
        g = Poisson()
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(g.link_inverse(eta), np.exp(eta), atol=1e-12)

    def test_link_roundtrip(self) -> None:
        g = Poisson()
        mu = np.array([0.5, 1.0, 5.0, 10.0])
        assert_allclose(g.link_inverse(g.link(mu)), mu, atol=1e-12)

    def test_link_inverse_clips_large_eta(self) -> None:
        g = Poisson()
        eta = np.array([100.0])
        result = g.link_inverse(eta)
        assert np.isfinite(result[0])

    def test_link_derivative_is_reciprocal(self) -> None:
        g = Poisson()
        mu = np.array([0.5, 1.0, 5.0])
        assert_allclose(g.link_derivative(mu), 1.0 / mu, atol=1e-12)


class TestVariance:
    def test_variance_equals_mu(self) -> None:
        g = Poisson()
        mu = np.array([0.5, 1.0, 10.0])
        assert_allclose(g.variance(mu), mu, atol=1e-12)

    def test_variance_positive_near_zero(self) -> None:
        g = Poisson()
        mu = np.array([1e-20])
        assert g.variance(mu)[0] > 0


class TestDeviance:
    def test_deviance_zero_at_perfect_fit(self) -> None:
        g = Poisson()
        y = np.array([1.0, 3.0, 5.0])
        assert_allclose(g.deviance(y, y), 0.0, atol=1e-12)

    def test_deviance_positive_for_imperfect_fit(self) -> None:
        g = Poisson()
        y = np.array([1.0, 5.0, 10.0])
        mu = np.array([2.0, 3.0, 8.0])
        assert g.deviance(y, mu) > 0

    def test_deviance_handles_zero_counts(self) -> None:
        g = Poisson()
        y = np.array([0.0, 0.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        dev = g.deviance(y, mu)
        assert np.isfinite(dev)
        assert dev > 0

    def test_deviance_increases_with_worse_fit(self) -> None:
        g = Poisson()
        y = np.array([1.0, 5.0, 10.0])
        mu_good = np.array([1.2, 4.8, 9.5])
        mu_bad = np.array([5.0, 5.0, 5.0])
        assert g.deviance(y, mu_bad) > g.deviance(y, mu_good)


class TestLogLikelihood:
    def test_log_likelihood_finite(self) -> None:
        g = Poisson()
        y = np.array([0.0, 1.0, 3.0, 5.0])
        mu = np.array([1.0, 2.0, 3.0, 4.0])
        ll = g.log_likelihood(y, mu, 1.0)
        assert np.isfinite(ll)

    def test_log_likelihood_improves_with_better_fit(self) -> None:
        g = Poisson()
        y = np.array([1.0, 5.0, 10.0])
        ll_good = g.log_likelihood(y, np.array([1.2, 4.8, 9.5]), 1.0)
        ll_bad = g.log_likelihood(y, np.array([5.0, 5.0, 5.0]), 1.0)
        assert ll_good > ll_bad


class TestInitialize:
    def test_initialize_positive(self) -> None:
        g = Poisson()
        y = np.array([0.0, 0.0, 1.0, 5.0])
        mu0 = g.initialize(y)
        assert np.all(mu0 > 0)

    def test_initialize_handles_zeros(self) -> None:
        g = Poisson()
        y = np.zeros(10)
        mu0 = g.initialize(y)
        assert np.all(mu0 > 0)


class TestScaleKnown:
    def test_scale_is_known(self) -> None:
        g = Poisson()
        assert g.scale_known is True


class TestRepr:
    def test_repr(self) -> None:
        g = Poisson()
        assert "Poisson" in repr(g)
        assert "log" in repr(g)
