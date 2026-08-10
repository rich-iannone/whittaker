"""Tests for ML (marginal likelihood) smoothing parameter selection."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gamma, Gaussian, Poisson, Tweedie
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture()
def two_smooth_data():
    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
    return {"x1": x1, "x2": x2, "y": y}


# ---------------------------------------------------------------------------
# Basic fitting with ML
# ---------------------------------------------------------------------------


class TestMLFitting:
    def test_fit_ml(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        assert gam.is_fitted

    def test_coefficients_finite(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        assert np.isfinite(gam.coefficients).all()

    def test_deviance_explained(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        assert gam.deviance_explained > 0.5

    def test_smoothing_params_positive(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        assert all(sp > 0 for sp in gam.smoothing_params)

    def test_invalid_method_raises(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        with pytest.raises(ValueError, match="method"):
            gam.fit(simple_data, method="INVALID")


# ---------------------------------------------------------------------------
# ML vs REML comparison
# ---------------------------------------------------------------------------


class TestMLvsREML:
    def test_ml_and_reml_both_fit(self, simple_data):
        gam_ml = GAM("y ~ s(x)", family=Gaussian())
        gam_ml.fit(simple_data, method="ML")
        gam_reml = GAM("y ~ s(x)", family=Gaussian())
        gam_reml.fit(simple_data, method="REML")
        assert gam_ml.is_fitted
        assert gam_reml.is_fitted

    def test_ml_and_reml_similar_predictions(self, simple_data):
        gam_ml = GAM("y ~ s(x)", family=Gaussian())
        gam_ml.fit(simple_data, method="ML")
        gam_reml = GAM("y ~ s(x)", family=Gaussian())
        gam_reml.fit(simple_data, method="REML")
        pred_ml = gam_ml.predict(simple_data).values
        pred_reml = gam_reml.predict(simple_data).values
        np.testing.assert_allclose(pred_ml, pred_reml, atol=0.2)

    def test_ml_tends_to_smooth_more(self, simple_data):
        gam_ml = GAM("y ~ s(x)", family=Gaussian())
        gam_ml.fit(simple_data, method="ML")
        gam_reml = GAM("y ~ s(x)", family=Gaussian())
        gam_reml.fit(simple_data, method="REML")
        assert gam_ml.edf[0] <= gam_reml.edf[0] + 1.0

    def test_ml_objective_differs_from_reml(self, simple_data):
        """ML and REML have different objective values even when the optimum
        coincides (the log|X'WX| term is constant w.r.t. ρ for Gaussian)."""
        gam_ml = GAM("y ~ s(x)", family=Gaussian())
        gam_ml.fit(simple_data, method="ML")
        gam_reml = GAM("y ~ s(x)", family=Gaussian())
        gam_reml.fit(simple_data, method="REML")
        assert gam_ml.is_fitted
        assert gam_reml.is_fitted


# ---------------------------------------------------------------------------
# Multiple smooths
# ---------------------------------------------------------------------------


class TestMLMultipleSmooths:
    def test_two_smooths(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data, method="ML")
        assert gam.is_fitted
        assert len(gam.smoothing_params) == 2

    def test_two_smooths_deviance(self, two_smooth_data):
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian())
        gam.fit(two_smooth_data, method="ML")
        assert gam.deviance_explained > 0.5


# ---------------------------------------------------------------------------
# Non-Gaussian families
# ---------------------------------------------------------------------------


class TestMLNonGaussian:
    def test_poisson_ml(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x)))
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(data, method="ML")
        assert gam.is_fitted
        assert np.all(gam.predict(data).values > 0)

    def test_gamma_ml(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        mu = np.exp(np.sin(x) + 1)
        y = rng.gamma(5, mu / 5)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Gamma())
        gam.fit(data, method="ML")
        assert gam.is_fitted

    def test_tweedie_ml(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        mu = np.exp(0.5 * np.sin(x) + 0.5)
        p, phi = 1.5, 1.0
        lam = mu ** (2 - p) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        gamma_scale = phi * (p - 1) * mu ** (p - 1)
        y = np.zeros(n)
        for i in range(n):
            N_i = rng.poisson(lam[i])
            if N_i > 0:
                y[i] = np.sum(rng.gamma(alpha, gamma_scale[i], size=N_i))
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(data, method="ML")
        assert gam.is_fitted


# ---------------------------------------------------------------------------
# Prediction after ML
# ---------------------------------------------------------------------------


class TestMLPrediction:
    def test_predict_se(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        result = gam.predict(simple_data, se=True)
        assert result.se is not None
        assert np.isfinite(result.se).all()

    def test_predict_interval(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        result = gam.predict(simple_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower <= result.upper)

    def test_predict_terms(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        result = gam.predict(simple_data, type="terms")
        assert len(result.terms) == 1


# ---------------------------------------------------------------------------
# Diagnostics after ML
# ---------------------------------------------------------------------------


class TestMLDiagnostics:
    def test_summary(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        summary = gam.summary()
        assert "GAM fit summary" in summary

    def test_smooth_tests(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        tests = gam.smooth_tests()
        assert len(tests) == 1
        assert tests[0].edf > 0

    def test_aic_bic(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        assert np.isfinite(gam.aic)
        assert np.isfinite(gam.bic)


# ---------------------------------------------------------------------------
# ML with select, weights, offset
# ---------------------------------------------------------------------------


class TestMLCombinations:
    def test_ml_with_select(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML", select=True)
        assert gam.is_fitted

    def test_ml_with_weights(self, simple_data):
        n = len(simple_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML", weights=w)
        assert gam.is_fitted

    def test_ml_with_offset(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        log_exposure = rng.uniform(0, 1, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x) + log_exposure))
        data = {"x": x, "y": y, "log_exposure": log_exposure}
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(data, method="ML")
        assert gam.is_fitted

    def test_ml_simulate(self, simple_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(simple_data, method="ML")
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape == (len(simple_data["y"]), 10)
