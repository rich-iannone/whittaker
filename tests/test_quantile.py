"""Tests for Quantile GAM integration."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, QuantileFamily
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def symmetric_data():
    rng = np.random.default_rng(23)
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture()
def heteroscedastic_data():
    rng = np.random.default_rng(23)
    n = 400
    x = rng.uniform(0, 2 * np.pi, n)
    noise_scale = 0.1 + 0.5 * (x / (2 * np.pi))
    y = np.sin(x) + rng.normal(0, noise_scale, n)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestQuantileFitting:
    def test_fit_median(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        assert gam.is_fitted

    def test_fit_gcv(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data)
        assert gam.is_fitted

    def test_fit_ml(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="ML")
        assert gam.is_fitted

    def test_coefficients_finite(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        assert np.isfinite(gam.coefficients).all()


# ---------------------------------------------------------------------------
# Quantile ordering
# ---------------------------------------------------------------------------


class TestQuantileOrdering:
    def test_quantiles_ordered(self, symmetric_data):
        results = {}
        for tau in [0.1, 0.5, 0.9]:
            gam = GAM("y ~ s(x)", family=QuantileFamily(tau=tau, sigma=0.1))
            gam.fit(symmetric_data, method="REML")
            results[tau] = gam.predict(symmetric_data).values
        assert np.all(results[0.1] < results[0.5])
        assert np.all(results[0.5] < results[0.9])

    def test_coverage(self, symmetric_data):
        gam10 = GAM("y ~ s(x)", family=QuantileFamily(tau=0.1, sigma=0.1))
        gam10.fit(symmetric_data, method="REML")
        pred10 = gam10.predict(symmetric_data).values

        gam90 = GAM("y ~ s(x)", family=QuantileFamily(tau=0.9, sigma=0.1))
        gam90.fit(symmetric_data, method="REML")
        pred90 = gam90.predict(symmetric_data).values

        y = symmetric_data["y"]
        coverage = np.mean((y >= pred10) & (y <= pred90))
        assert 0.65 < coverage < 0.95


# ---------------------------------------------------------------------------
# Median vs mean
# ---------------------------------------------------------------------------


class TestMedianVsMean:
    def test_median_close_to_mean_symmetric(self, symmetric_data):
        gam_q = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam_q.fit(symmetric_data, method="REML")
        pred_q = gam_q.predict(symmetric_data).values

        gam_g = GAM("y ~ s(x)", family=Gaussian())
        gam_g.fit(symmetric_data, method="REML")
        pred_g = gam_g.predict(symmetric_data).values

        assert np.max(np.abs(pred_q - pred_g)) < 0.5


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestQuantilePrediction:
    def test_predict_se(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        result = gam.predict(symmetric_data, se=True)
        assert result.se is not None
        assert np.isfinite(result.se).all()

    def test_predict_new_data(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        new = {"x": np.linspace(0, 2 * np.pi, 50)}
        result = gam.predict(new)
        assert result.values.shape == (50,)
        assert np.isfinite(result.values).all()

    def test_predict_terms(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        result = gam.predict(symmetric_data, type="terms")
        assert result.terms is not None


# ---------------------------------------------------------------------------
# Sigma effect
# ---------------------------------------------------------------------------


class TestSigmaEffect:
    def test_larger_sigma_smoother(self, symmetric_data):
        preds = {}
        for sigma in [0.05, 1.0]:
            gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=sigma))
            gam.fit(symmetric_data, method="REML")
            preds[sigma] = gam.predict(symmetric_data).values
        diff_small = np.diff(preds[0.05][np.argsort(symmetric_data["x"])])
        diff_large = np.diff(preds[1.0][np.argsort(symmetric_data["x"])])
        assert np.std(diff_large) <= np.std(diff_small) * 2


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestQuantileDiagnostics:
    def test_summary(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        s = gam.summary()
        assert "GAM fit summary" in s
        assert "Quantile" in s

    def test_residuals(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        r = gam.get_residuals("response")
        assert np.isfinite(r).all()

    def test_aic_bic(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        assert np.isfinite(gam.aic)
        assert np.isfinite(gam.bic)


# ---------------------------------------------------------------------------
# Multiple smooths
# ---------------------------------------------------------------------------


class TestQuantileMultipleSmooths:
    def test_two_smooths(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM("y ~ s(x1) + s(x2)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(data, method="REML")
        assert gam.is_fitted
        assert np.isfinite(gam.coefficients).all()


# ---------------------------------------------------------------------------
# Heteroscedastic data
# ---------------------------------------------------------------------------


class TestQuantileHeteroscedastic:
    def test_wider_intervals_where_noise_is_larger(self, heteroscedastic_data):
        data = heteroscedastic_data

        gam10 = GAM("y ~ s(x)", family=QuantileFamily(tau=0.1, sigma=0.1))
        gam10.fit(data, method="REML")

        gam90 = GAM("y ~ s(x)", family=QuantileFamily(tau=0.9, sigma=0.1))
        gam90.fit(data, method="REML")

        x = data["x"]
        pred10 = gam10.predict(data).values
        pred90 = gam90.predict(data).values
        width = pred90 - pred10

        low_x = width[x < np.pi]
        high_x = width[x > np.pi]
        assert np.mean(high_x) > np.mean(low_x)


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------


class TestQuantileSimulate:
    def test_simulate(self, symmetric_data):
        gam = GAM("y ~ s(x)", family=QuantileFamily(tau=0.5, sigma=0.1))
        gam.fit(symmetric_data, method="REML")
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape == (len(symmetric_data["y"]), 10)
        assert np.isfinite(sims).all()


# ---------------------------------------------------------------------------
# Direct QuantileFamily method tests
# ---------------------------------------------------------------------------


class TestQuantileFamilyDirect:
    """Tests for QuantileFamily constructor validations and individual methods."""

    def test_invalid_tau_zero_raises(self):
        with pytest.raises(ValueError, match="tau must be in"):
            QuantileFamily(tau=0.0)

    def test_invalid_tau_one_raises(self):
        with pytest.raises(ValueError, match="tau must be in"):
            QuantileFamily(tau=1.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            QuantileFamily(sigma=-0.5)

    def test_tau_property(self):
        fam = QuantileFamily(tau=0.3, sigma=0.5)
        assert fam.tau == pytest.approx(0.3)

    def test_sigma_property(self):
        fam = QuantileFamily(tau=0.3, sigma=0.5)
        assert fam.sigma == pytest.approx(0.5)

    def test_sigma_setter_zero_raises(self):
        fam = QuantileFamily(tau=0.5)
        with pytest.raises(ValueError, match="sigma must be positive"):
            fam.sigma = 0.0

    def test_sigma_setter_negative_raises(self):
        fam = QuantileFamily(tau=0.5)
        with pytest.raises(ValueError, match="sigma must be positive"):
            fam.sigma = -1.0

    def test_sigma_setter_valid(self):
        fam = QuantileFamily(tau=0.5, sigma=0.1)
        fam.sigma = 0.5
        assert fam.sigma == pytest.approx(0.5)

    def test_link_derivative(self):
        fam = QuantileFamily(tau=0.5)
        mu = np.array([1.0, 2.0, 3.0])
        result = fam.link_derivative(mu)
        np.testing.assert_allclose(result, np.ones(3))

    def test_variance(self):
        fam = QuantileFamily(tau=0.5)
        mu = np.array([0.0, 1.0, -1.0])
        result = fam.variance(mu)
        np.testing.assert_allclose(result, np.ones(3))

    def test_deviance_with_weights(self):
        fam = QuantileFamily(tau=0.5, sigma=0.1)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        w = np.array([1.0, 2.0, 0.5])
        dev = fam.deviance(y, mu, weights=w)
        assert np.isfinite(dev)
        dev_unweighted = fam.deviance(y, mu)
        assert dev != dev_unweighted

    def test_unit_deviance(self):
        fam = QuantileFamily(tau=0.5, sigma=0.1)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        ud = fam.unit_deviance(y, mu)
        assert ud.shape == y.shape
        assert np.all(ud >= 0)

    def test_log_likelihood_with_weights(self):
        fam = QuantileFamily(tau=0.5, sigma=0.1)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        w = np.array([1.0, 2.0, 0.5])
        ll = fam.log_likelihood(y, mu, scale=1.0, weights=w)
        assert np.isfinite(ll)
        ll_unweighted = fam.log_likelihood(y, mu, scale=1.0)
        assert ll != ll_unweighted

    def test_simulate(self):
        fam = QuantileFamily(tau=0.5, sigma=0.1)
        rng = np.random.default_rng(0)
        mu = np.array([1.0, 2.0, 3.0])
        samples = fam.simulate(mu, scale=1.0, rng=rng)
        assert samples.shape == mu.shape
        assert np.isfinite(samples).all()

    def test_repr(self):
        fam = QuantileFamily(tau=0.75, sigma=0.2)
        r = repr(fam)
        assert "0.75" in r
        assert "0.2" in r
