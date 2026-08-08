"""Tests for the Tweedie family."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.tweedie import Tweedie

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestTweedieConstruction:
    def test_default_p(self):
        tw = Tweedie()
        assert tw.p == 1.5

    def test_custom_p(self):
        tw = Tweedie(p=1.8)
        assert tw.p == 1.8

    def test_p_above_2(self):
        tw = Tweedie(p=3.0)
        assert tw.p == 3.0

    def test_p_equals_1_raises(self):
        with pytest.raises(ValueError, match="Tweedie"):
            Tweedie(p=1.0)

    def test_p_equals_2_raises(self):
        with pytest.raises(ValueError, match="Tweedie"):
            Tweedie(p=2.0)

    def test_p_below_1_raises(self):
        with pytest.raises(ValueError, match="Tweedie"):
            Tweedie(p=0.5)

    def test_repr(self):
        tw = Tweedie(p=1.5)
        assert "Tweedie" in repr(tw)
        assert "1.5" in repr(tw)


# ---------------------------------------------------------------------------
# Link function
# ---------------------------------------------------------------------------


class TestTweedieLink:
    def test_link(self):
        tw = Tweedie()
        mu = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(tw.link(mu), np.log(mu))

    def test_link_inverse(self):
        tw = Tweedie()
        eta = np.array([0.0, 1.0, -1.0])
        np.testing.assert_allclose(tw.link_inverse(eta), np.exp(eta))

    def test_link_roundtrip(self):
        tw = Tweedie()
        mu = np.array([0.5, 1.0, 5.0])
        np.testing.assert_allclose(tw.link_inverse(tw.link(mu)), mu, atol=1e-12)

    def test_link_derivative(self):
        tw = Tweedie()
        mu = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(tw.link_derivative(mu), 1.0 / mu)


# ---------------------------------------------------------------------------
# Variance function
# ---------------------------------------------------------------------------


class TestTweedieVariance:
    def test_variance_p15(self):
        tw = Tweedie(p=1.5)
        mu = np.array([1.0, 2.0, 4.0])
        expected = mu**1.5
        np.testing.assert_allclose(tw.variance(mu), expected)

    def test_variance_p3(self):
        tw = Tweedie(p=3.0)
        mu = np.array([1.0, 2.0, 4.0])
        np.testing.assert_allclose(tw.variance(mu), mu**3)


# ---------------------------------------------------------------------------
# Deviance
# ---------------------------------------------------------------------------


class TestTweedieDeviance:
    def test_unit_deviance_nonneg(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu = rng.uniform(0.5, 5.0, 100)
        y = rng.uniform(0.0, 5.0, 100)
        d = tw.unit_deviance(y, mu)
        assert np.all(d >= -1e-10)

    def test_unit_deviance_zero_at_y_eq_mu(self):
        tw = Tweedie(p=1.5)
        mu = np.array([1.0, 2.0, 3.0])
        d = tw.unit_deviance(mu, mu)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_deviance_positive(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu = rng.uniform(0.5, 5.0, 50)
        y = rng.uniform(0.0, 5.0, 50)
        assert tw.deviance(y, mu) > 0

    def test_deviance_with_weights(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu = rng.uniform(0.5, 5.0, 50)
        y = rng.uniform(0.0, 5.0, 50)
        w = np.ones(50) * 2.0
        assert tw.deviance(y, mu, weights=w) == pytest.approx(2.0 * tw.deviance(y, mu))

    def test_unit_deviance_with_zeros(self):
        tw = Tweedie(p=1.5)
        y = np.array([0.0, 0.0, 1.0, 2.0])
        mu = np.array([1.0, 2.0, 1.0, 2.0])
        d = tw.unit_deviance(y, mu)
        assert np.all(np.isfinite(d))
        assert np.all(d >= -1e-10)


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------


class TestTweedieLogLikelihood:
    def test_log_likelihood_finite(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu = rng.uniform(0.5, 5.0, 50)
        y = rng.uniform(0.0, 5.0, 50)
        ll = tw.log_likelihood(y, mu, scale=1.0)
        assert np.isfinite(ll)

    def test_log_likelihood_with_zeros(self):
        tw = Tweedie(p=1.5)
        y = np.array([0.0, 0.0, 1.0, 2.0])
        mu = np.array([1.0, 2.0, 1.0, 2.0])
        ll = tw.log_likelihood(y, mu, scale=1.0)
        assert np.isfinite(ll)

    def test_log_likelihood_better_fit(self):
        tw = Tweedie(p=1.5)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu_good = y.copy()
        mu_bad = np.full_like(y, 2.5)
        ll_good = tw.log_likelihood(y, mu_good, scale=1.0)
        ll_bad = tw.log_likelihood(y, mu_bad, scale=1.0)
        assert ll_good > ll_bad

    def test_log_likelihood_with_weights(self):
        tw = Tweedie(p=1.5)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.5, 3.5])
        w = np.ones(3) * 2.0
        ll_w = tw.log_likelihood(y, mu, scale=1.0, weights=w)
        ll_no = tw.log_likelihood(y, mu, scale=1.0)
        assert ll_w == pytest.approx(2.0 * ll_no)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


class TestTweedieScale:
    def test_scale_not_known(self):
        tw = Tweedie()
        assert tw.scale_known is False


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


class TestTweedieInitialize:
    def test_initialize_positive(self):
        tw = Tweedie()
        y = np.array([0.0, 0.0, 1.0, 5.0])
        mu0 = tw.initialize(y)
        assert np.all(mu0 > 0)


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------


class TestTweedieSimulate:
    def test_simulate_shape(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu = np.array([1.0, 2.0, 3.0])
        sim = tw.simulate(mu, scale=1.0, rng=rng)
        assert sim.shape == (3,)

    def test_simulate_nonneg(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu = np.ones(100) * 2.0
        sim = tw.simulate(mu, scale=1.0, rng=rng)
        assert np.all(sim >= 0)

    def test_simulate_has_zeros(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu = np.ones(500) * 0.3
        sim = tw.simulate(mu, scale=1.0, rng=rng)
        assert np.any(sim == 0)
        assert np.any(sim > 0)

    def test_simulate_mean_approx(self):
        tw = Tweedie(p=1.5)
        rng = np.random.default_rng(23)
        mu_val = 3.0
        mu = np.full(10000, mu_val)
        sim = tw.simulate(mu, scale=0.5, rng=rng)
        np.testing.assert_allclose(sim.mean(), mu_val, rtol=0.1)

    def test_simulate_p_above_2(self):
        tw = Tweedie(p=3.0)
        rng = np.random.default_rng(23)
        mu = np.array([1.0, 2.0, 3.0])
        sim = tw.simulate(mu, scale=0.5, rng=rng)
        assert sim.shape == (3,)
        assert np.all(sim > 0)
        assert np.all(np.isfinite(sim))
