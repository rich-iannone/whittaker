"""Tests for Beta regression family."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from whittaker.families.beta import Beta
from whittaker.gam import GAM


class TestBetaFamily:
    def test_link_logit(self):
        fam = Beta()
        mu = np.array([0.2, 0.5, 0.8])
        eta = fam.link(mu)
        np.testing.assert_allclose(eta, np.log(mu / (1 - mu)))

    def test_link_inverse(self):
        fam = Beta()
        eta = np.array([-1.0, 0.0, 1.0])
        mu = fam.link_inverse(eta)
        assert np.all(mu > 0) and np.all(mu < 1)
        np.testing.assert_allclose(mu, 1.0 / (1.0 + np.exp(-eta)))

    def test_link_roundtrip(self):
        fam = Beta()
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(fam.link_inverse(fam.link(mu)), mu)

    def test_variance(self):
        fam = Beta()
        mu = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(fam.variance(mu), mu * (1 - mu))

    def test_deviance_positive(self):
        fam = Beta()
        rng = np.random.default_rng(23)
        y = rng.beta(2, 5, size=100)
        mu = np.clip(np.full(100, np.mean(y)), 0.01, 0.99)
        assert fam.deviance(y, mu) > 0

    def test_deviance_zero_at_perfect_fit(self):
        fam = Beta()
        y = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(fam.deviance(y, y), 0.0, atol=1e-12)

    def test_deviance_with_weights(self):
        fam = Beta()
        y = np.array([0.2, 0.5, 0.8])
        mu = np.array([0.3, 0.4, 0.6])
        weights = np.array([1.0, 2.0, 3.0])
        unweighted = fam.unit_deviance(y, mu)
        expected = float(np.sum(weights * unweighted))
        actual = fam.deviance(y, mu, weights=weights)
        np.testing.assert_allclose(actual, expected)
        assert actual != fam.deviance(y, mu)

    def test_log_likelihood_vs_scipy(self):
        fam = Beta(phi=10.0)
        mu = np.array([0.3, 0.5, 0.7])
        y = np.array([0.25, 0.55, 0.65])
        a = mu * 10.0
        b = (1 - mu) * 10.0
        expected = float(np.sum(stats.beta.logpdf(y, a, b)))
        actual = fam.log_likelihood(y, mu, scale=0.1)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_log_likelihood_with_weights(self):
        fam = Beta(phi=10.0)
        mu = np.array([0.3, 0.5, 0.7])
        y = np.array([0.25, 0.55, 0.65])
        weights = np.array([1.0, 2.0, 0.5])
        unweighted_ll = fam.log_likelihood(y, mu, scale=0.1)
        weighted_ll = fam.log_likelihood(y, mu, scale=0.1, weights=weights)
        assert weighted_ll != unweighted_ll

    def test_scale_known_with_fixed_phi(self):
        fam = Beta(phi=5.0)
        assert fam.scale_known is True

    def test_scale_unknown_default(self):
        fam = Beta()
        assert fam.scale_known is False

    def test_simulate(self):
        fam = Beta(phi=10.0)
        rng = np.random.default_rng(23)
        mu = np.full(1000, 0.4)
        sim = fam.simulate(mu, scale=0.1, rng=rng)
        assert sim.shape == (1000,)
        assert np.all(sim > 0) and np.all(sim < 1)
        np.testing.assert_allclose(np.mean(sim), 0.4, atol=0.05)

    def test_initialize(self):
        fam = Beta()
        y = np.array([0.0, 0.5, 1.0])
        mu = fam.initialize(y)
        assert np.all(mu > 0) and np.all(mu < 1)

    def test_repr(self):
        assert "logit" in repr(Beta())
        assert "phi=5" in repr(Beta(phi=5.0))


class TestBetaGAM:
    @pytest.fixture()
    def beta_data(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = 1.0 / (1.0 + np.exp(-(np.sin(x))))
        phi = 20.0
        a = mu_true * phi
        b = (1.0 - mu_true) * phi
        y = rng.beta(a, b)
        return {"x": x, "y": y}, mu_true

    def test_converges(self, beta_data):
        data, _ = beta_data
        model = GAM("y ~ s(x)", family=Beta())
        model.fit(data)
        assert model.is_fitted

    def test_mu_recovery(self, beta_data):
        data, mu_true = beta_data
        model = GAM("y ~ s(x)", family=Beta())
        model.fit(data)
        pred = model.predict(data).values
        assert np.corrcoef(mu_true, pred)[0, 1] > 0.95

    def test_predictions_in_unit_interval(self, beta_data):
        data, _ = beta_data
        model = GAM("y ~ s(x)", family=Beta())
        model.fit(data)
        pred = model.predict(data).values
        assert np.all(pred > 0) and np.all(pred < 1)

    def test_with_reml(self, beta_data):
        data, _ = beta_data
        model = GAM("y ~ s(x)", family=Beta())
        model.fit(data, method="REML")
        assert model.is_fitted
        pred = model.predict(data).values
        assert np.all(pred > 0) and np.all(pred < 1)

    def test_summary(self, beta_data):
        data, _ = beta_data
        model = GAM("y ~ s(x)", family=Beta())
        model.fit(data)
        s = model.summary()
        assert "Beta" in s

    def test_simulate(self, beta_data):
        data, _ = beta_data
        model = GAM("y ~ s(x)", family=Beta())
        model.fit(data)
        sims = model.simulate(n_sim=5, seed=23)
        assert sims.shape == (len(data["y"]), 5)
        assert np.all(sims > 0) and np.all(sims < 1)

    def test_fixed_phi(self, beta_data):
        data, _ = beta_data
        model = GAM("y ~ s(x)", family=Beta(phi=20.0))
        model.fit(data)
        pred = model.predict(data).values
        assert np.all(pred > 0) and np.all(pred < 1)
