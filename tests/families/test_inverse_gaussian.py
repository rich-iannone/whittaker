"""Tests for Inverse Gaussian family."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from whittaker.families.inverse_gaussian import InverseGaussian

RNG = np.random.default_rng(23)
_EPS = np.finfo(float).eps


class TestInverseGaussianLink:
    def test_link(self):
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(InverseGaussian().link(mu), np.log(mu))

    def test_link_inverse(self):
        eta = np.array([0.0, 1.0, -1.0])
        assert_allclose(InverseGaussian().link_inverse(eta), np.exp(eta))

    def test_link_derivative(self):
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(InverseGaussian().link_derivative(mu), 1.0 / mu)

    def test_link_inverse_of_link(self):
        mu = RNG.uniform(0.5, 5.0, 50)
        ig = InverseGaussian()
        assert_allclose(ig.link_inverse(ig.link(mu)), mu, rtol=1e-10)


class TestInverseGaussianVariance:
    def test_variance(self):
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(InverseGaussian().variance(mu), mu**3)


class TestInverseGaussianDeviance:
    def test_unit_deviance_at_mu(self):
        mu = np.array([1.0, 2.0, 5.0])
        assert_allclose(InverseGaussian().unit_deviance(mu, mu), 0.0, atol=1e-14)

    def test_unit_deviance_positive(self):
        mu = RNG.uniform(0.5, 5.0, 50)
        y = RNG.uniform(0.5, 5.0, 50)
        d = InverseGaussian().unit_deviance(y, mu)
        assert np.all(d >= 0)

    def test_deviance_with_weights(self):
        mu = np.array([1.0, 2.0, 3.0])
        y = np.array([1.1, 1.9, 3.2])
        w = np.array([1.0, 2.0, 1.0])
        ig = InverseGaussian()
        d_w = ig.deviance(y, mu, weights=w)
        d_manual = float(np.sum(w * ig.unit_deviance(y, mu)))
        assert_allclose(d_w, d_manual)


class TestInverseGaussianLogLikelihood:
    def test_log_likelihood_finite(self):
        mu = RNG.uniform(0.5, 5.0, 50)
        y = RNG.uniform(0.5, 5.0, 50)
        ll = InverseGaussian().log_likelihood(y, mu, scale=1.0)
        assert np.isfinite(ll)

    def test_log_likelihood_better_at_truth(self):
        mu_true = np.array([2.0, 3.0, 4.0])
        mu_bad = np.array([5.0, 1.0, 1.0])
        y = mu_true.copy()
        ig = InverseGaussian()
        ll_true = ig.log_likelihood(y, mu_true, scale=0.5)
        ll_bad = ig.log_likelihood(y, mu_bad, scale=0.5)
        assert ll_true > ll_bad


class TestInverseGaussianSimulate:
    def test_simulate_shape(self):
        mu = np.ones(100) * 2.0
        y = InverseGaussian().simulate(mu, 0.5, RNG)
        assert y.shape == (100,)

    def test_simulate_positive(self):
        mu = RNG.uniform(0.5, 5.0, 500)
        y = InverseGaussian().simulate(mu, 0.5, RNG)
        assert np.all(y > 0)

    def test_simulate_mean_near_mu(self):
        mu = np.ones(10000) * 3.0
        y = InverseGaussian().simulate(mu, 0.1, RNG)
        assert_allclose(np.mean(y), 3.0, atol=0.1)


class TestInverseGaussianMisc:
    def test_scale_known(self):
        assert not InverseGaussian().scale_known

    def test_initialize(self):
        y = np.array([-1.0, 0.0, 1.0, 5.0])
        mu0 = InverseGaussian().initialize(y)
        assert np.all(mu0 > 0)

    def test_repr(self):
        assert "InverseGaussian" in repr(InverseGaussian())
