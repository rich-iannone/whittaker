"""Tests for the Negative Binomial family."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.negative_binomial import NegativeBinomial


class TestNegativeBinomialFamily:
    def setup_method(self) -> None:
        self.fam = NegativeBinomial(theta=2.0)

    def test_link(self) -> None:
        mu = np.array([1.0, 2.0, np.e])
        assert_allclose(self.fam.link(mu), np.log(mu))

    def test_link_inverse(self) -> None:
        eta = np.array([0.0, 1.0, -1.0])
        assert_allclose(self.fam.link_inverse(eta), np.exp(eta))

    def test_link_derivative(self) -> None:
        mu = np.array([1.0, 2.0, 5.0])
        assert_allclose(self.fam.link_derivative(mu), 1.0 / mu)

    def test_link_roundtrip(self) -> None:
        mu = np.array([0.5, 1.0, 3.0, 10.0])
        assert_allclose(self.fam.link_inverse(self.fam.link(mu)), mu)

    def test_variance(self) -> None:
        mu = np.array([1.0, 2.0, 3.0])
        theta = 2.0
        expected = mu + mu**2 / theta
        assert_allclose(self.fam.variance(mu), expected)

    def test_variance_approaches_poisson(self) -> None:
        mu = np.array([1.0, 2.0, 5.0])
        fam_large_theta = NegativeBinomial(theta=1e8)
        assert_allclose(fam_large_theta.variance(mu), mu, rtol=1e-6)

    def test_deviance_at_perfect_fit(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert_allclose(self.fam.deviance(y, y), 0.0, atol=1e-12)

    def test_deviance_positive(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.5, 1.5, 3.5, 3.5])
        assert self.fam.deviance(y, mu) > 0

    def test_unit_deviance_sums_to_deviance(self) -> None:
        y = np.array([0.0, 1.0, 3.0, 5.0, 10.0])
        mu = np.array([0.5, 1.5, 2.0, 4.0, 8.0])
        assert_allclose(np.sum(self.fam.unit_deviance(y, mu)), self.fam.deviance(y, mu))

    def test_deviance_weighted(self) -> None:
        y = np.array([0.0, 1.0, 3.0, 5.0, 10.0])
        mu = np.array([0.5, 1.5, 2.0, 4.0, 8.0])
        w = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        assert_allclose(self.fam.deviance(y, mu, weights=w), 2.0 * self.fam.deviance(y, mu))

    def test_scale_known_true(self) -> None:
        assert self.fam.scale_known is True

    def test_initialize(self) -> None:
        y = np.array([0.0, 1.0, 5.0])
        mu0 = self.fam.initialize(y)
        assert np.all(mu0 > 0)

    def test_log_likelihood_finite(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 5.0])
        mu = np.array([1.0, 1.5, 2.5, 4.0])
        ll = self.fam.log_likelihood(y, mu, scale=1.0)
        assert np.isfinite(ll)

    def test_log_likelihood_better_at_truth(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ll_good = self.fam.log_likelihood(y, y, scale=1.0)
        ll_bad = self.fam.log_likelihood(y, np.full_like(y, 3.0), scale=1.0)
        assert ll_good > ll_bad

    def test_log_likelihood_weighted(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 5.0])
        mu = np.array([1.0, 1.5, 2.5, 4.0])
        w = np.array([2.0, 2.0, 2.0, 2.0])
        ll_unweighted = self.fam.log_likelihood(y, mu, scale=1.0)
        ll_weighted = self.fam.log_likelihood(y, mu, scale=1.0, weights=w)
        assert_allclose(ll_weighted, 2.0 * ll_unweighted)

    def test_theta_property(self) -> None:
        fam = NegativeBinomial(theta=5.0)
        assert fam.theta == 5.0
        fam.theta = 10.0
        assert fam.theta == 10.0

    def test_theta_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            NegativeBinomial(theta=0.0)
        with pytest.raises(ValueError, match="positive"):
            NegativeBinomial(theta=-1.0)
        fam = NegativeBinomial(theta=1.0)
        with pytest.raises(ValueError, match="positive"):
            fam.theta = 0.0

    def test_simulate_shape_and_finite(self) -> None:
        rng = np.random.default_rng(42)
        mu = np.array([1.0, 5.0, 10.0, 20.0])
        y = self.fam.simulate(mu, scale=1.0, rng=rng)
        assert y.shape == mu.shape
        assert np.isfinite(y).all()
        assert np.all(y >= 0)

    def test_repr(self) -> None:
        fam = NegativeBinomial(theta=3.5)
        r = repr(fam)
        assert "NegativeBinomial" in r
        assert "3.5" in r
        assert "log" in r
