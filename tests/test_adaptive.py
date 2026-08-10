"""Tests for adaptive smooth GAM integration (bs='ad')."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def smooth_data():
    rng = np.random.default_rng(23)
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture()
def spatially_varying_data():
    rng = np.random.default_rng(23)
    n = 400
    x = rng.uniform(0, 4 * np.pi, n)
    noise = np.where(x < 2 * np.pi, 0.1, 0.6)
    y = np.sin(x) + rng.normal(0, noise, n)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestAdaptiveFitting:
    def test_fit_gcv(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data)
        assert gam.is_fitted

    def test_fit_reml(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        assert gam.is_fitted

    def test_fit_ml(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="ML")
        assert gam.is_fitted

    def test_deviance_explained(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        assert gam.deviance_explained > 0.7

    def test_multiple_smoothing_params(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        assert len(gam.smoothing_params) > 1

    def test_coefficients_finite(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        assert np.isfinite(gam.coefficients).all()


# ---------------------------------------------------------------------------
# n_penalties
# ---------------------------------------------------------------------------


class TestAdaptiveNPenalties:
    def test_custom_n_penalties(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad", n_penalties=3)', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        assert gam.is_fitted
        assert len(gam.smoothing_params) == 3

    def test_n_penalties_1(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad", n_penalties=1)', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        assert len(gam.smoothing_params) == 1


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestAdaptivePrediction:
    def test_predict(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        result = gam.predict(smooth_data)
        assert result.values.shape == (len(smooth_data["y"]),)
        assert np.isfinite(result.values).all()

    def test_predict_se(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        result = gam.predict(smooth_data, se=True)
        assert result.se is not None
        assert np.all(result.se > 0)

    def test_predict_confidence_interval(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        result = gam.predict(smooth_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower <= result.upper)

    def test_predict_new_data(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        new = {"x": np.linspace(0, 2 * np.pi, 50)}
        result = gam.predict(new)
        assert result.values.shape == (50,)
        assert np.isfinite(result.values).all()

    def test_predict_terms(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        result = gam.predict(smooth_data, type="terms")
        assert result.terms is not None


# ---------------------------------------------------------------------------
# Mixed with other bases
# ---------------------------------------------------------------------------


class TestAdaptiveMixed:
    def test_adaptive_plus_tprs(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM('y ~ s(x1, bs="ad") + s(x2)', family=Gaussian())
        gam.fit(data, method="REML")
        assert gam.is_fitted
        assert gam.deviance_explained > 0.5

    def test_adaptive_poisson(self):
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        mu = np.exp(0.5 * np.sin(x))
        y = rng.poisson(mu, n)
        data = {"x": x, "y": y}
        gam = GAM('y ~ s(x, bs="ad")', family=Poisson())
        gam.fit(data, method="REML")
        assert gam.is_fitted
        assert np.isfinite(gam.coefficients).all()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestAdaptiveDiagnostics:
    def test_summary(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        s = gam.summary()
        assert "GAM fit summary" in s

    def test_gam_check(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        result = gam.gam_check(n_sim=20)
        assert result.deviance_explained > 0.5

    def test_aic_bic(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        assert np.isfinite(gam.aic)
        assert np.isfinite(gam.bic)

    def test_smooth_tests(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        tests = gam.smooth_tests()
        assert len(tests) == 1

    def test_residuals(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        for rtype in ("response", "pearson", "deviance", "working"):
            r = gam.get_residuals(rtype)
            assert np.isfinite(r).all()


# ---------------------------------------------------------------------------
# Select / weights
# ---------------------------------------------------------------------------


class TestAdaptiveCombinations:
    def test_with_select(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, select=True)
        assert gam.is_fitted

    def test_with_weights(self, smooth_data):
        n = len(smooth_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, weights=w)
        assert gam.is_fitted

    def test_simulate(self, smooth_data):
        gam = GAM('y ~ s(x, bs="ad")', family=Gaussian())
        gam.fit(smooth_data, method="REML")
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape == (len(smooth_data["y"]), 10)
        assert np.isfinite(sims).all()
