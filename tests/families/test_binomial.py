"""Tests for whittaker.families.binomial (Binomial family with logit link)."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial

RNG = np.random.default_rng(23)


class TestLink:
    def test_logit_at_half(self) -> None:
        g = Binomial()
        assert_allclose(g.link(np.array([0.5])), [0.0], atol=1e-12)

    def test_logit_symmetry(self) -> None:
        g = Binomial()
        mu = np.array([0.2, 0.8])
        eta = g.link(mu)
        assert_allclose(eta[0], -eta[1], atol=1e-12)

    def test_link_inverse_is_expit(self) -> None:
        g = Binomial()
        eta = np.array([-2.0, 0.0, 2.0])
        expected = 1.0 / (1.0 + np.exp(-eta))
        assert_allclose(g.link_inverse(eta), expected, atol=1e-12)

    def test_link_roundtrip(self) -> None:
        g = Binomial()
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert_allclose(g.link_inverse(g.link(mu)), mu, atol=1e-12)

    def test_link_derivative_positive(self) -> None:
        g = Binomial()
        mu = np.array([0.1, 0.5, 0.9])
        assert np.all(g.link_derivative(mu) > 0)

    def test_link_derivative_at_half(self) -> None:
        g = Binomial()
        assert_allclose(g.link_derivative(np.array([0.5])), [4.0], atol=1e-12)


class TestVariance:
    def test_variance_at_half(self) -> None:
        g = Binomial()
        assert_allclose(g.variance(np.array([0.5])), [0.25], atol=1e-12)

    def test_variance_symmetric(self) -> None:
        g = Binomial()
        assert_allclose(
            g.variance(np.array([0.3])),
            g.variance(np.array([0.7])),
            atol=1e-12,
        )

    def test_variance_positive(self) -> None:
        g = Binomial()
        mu = np.array([0.01, 0.5, 0.99])
        assert np.all(g.variance(mu) > 0)


class TestDeviance:
    def test_deviance_zero_at_perfect_fit(self) -> None:
        g = Binomial()
        y = np.array([0.0, 1.0, 1.0, 0.0])
        mu = np.array([1e-15, 1.0 - 1e-15, 1.0 - 1e-15, 1e-15])
        assert g.deviance(y, mu) < 1e-10

    def test_deviance_positive_for_imperfect_fit(self) -> None:
        g = Binomial()
        y = np.array([0.0, 1.0, 1.0, 0.0])
        mu = np.array([0.5, 0.5, 0.5, 0.5])
        assert g.deviance(y, mu) > 0

    def test_deviance_increases_with_worse_fit(self) -> None:
        g = Binomial()
        y = np.array([1.0, 1.0, 0.0, 0.0])
        mu_good = np.array([0.9, 0.8, 0.2, 0.1])
        mu_bad = np.array([0.5, 0.5, 0.5, 0.5])
        assert g.deviance(y, mu_bad) > g.deviance(y, mu_good)


class TestLogLikelihood:
    def test_log_likelihood_negative(self) -> None:
        g = Binomial()
        y = np.array([1.0, 0.0, 1.0])
        mu = np.array([0.8, 0.3, 0.6])
        assert g.log_likelihood(y, mu, 1.0) < 0

    def test_log_likelihood_improves_with_better_fit(self) -> None:
        g = Binomial()
        y = np.array([1.0, 0.0])
        ll_good = g.log_likelihood(y, np.array([0.9, 0.1]), 1.0)
        ll_bad = g.log_likelihood(y, np.array([0.5, 0.5]), 1.0)
        assert ll_good > ll_bad


class TestInitialize:
    def test_initialize_between_zero_and_one(self) -> None:
        g = Binomial()
        y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        mu0 = g.initialize(y)
        assert np.all(mu0 > 0)
        assert np.all(mu0 < 1)


class TestScaleKnown:
    def test_scale_is_known(self) -> None:
        g = Binomial()
        assert g.scale_known is True


class TestRepr:
    def test_repr(self) -> None:
        g = Binomial()
        assert "Binomial" in repr(g)
        assert "logit" in repr(g)
