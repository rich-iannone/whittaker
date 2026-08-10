"""Tests for Inverse Gaussian GAM fitting."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import InverseGaussian
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ig_data():
    rng = np.random.default_rng(23)
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    mu = np.exp(0.5 * np.sin(x) + 1.0)
    ig = InverseGaussian()
    y = ig.simulate(mu, 0.5, rng)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestInverseGaussianFitting:
    def test_fit(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        assert gam.is_fitted

    def test_fit_reml(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data, method="REML")
        assert gam.is_fitted

    def test_coefficients_finite(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        assert np.isfinite(gam.coefficients).all()

    def test_scale_estimated(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        assert gam.scale > 0
        assert np.isfinite(gam.scale)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestInverseGaussianPrediction:
    def test_predict_positive(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        result = gam.predict(ig_data)
        assert np.all(result.values > 0)
        assert np.isfinite(result.values).all()

    def test_predict_se(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        result = gam.predict(ig_data, se=True)
        assert result.se is not None
        assert np.all(result.se > 0)

    def test_predict_new_data(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        new = {"x": np.linspace(0, 2 * np.pi, 50)}
        result = gam.predict(new)
        assert result.values.shape == (50,)
        assert np.all(result.values > 0)

    def test_predict_link(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        link_result = gam.predict(ig_data, type="link")
        resp_result = gam.predict(ig_data)
        np.testing.assert_allclose(np.exp(link_result.values), resp_result.values, atol=1e-10)

    def test_predict_confidence_interval(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        result = gam.predict(ig_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower > 0)
        assert np.all(result.lower <= result.upper)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestInverseGaussianDiagnostics:
    def test_summary(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        summary = gam.summary()
        assert "GAM fit summary" in summary
        assert "InverseGaussian" in summary

    def test_residuals(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        for rtype in ("response", "pearson", "deviance", "working"):
            r = gam.get_residuals(rtype)
            assert np.isfinite(r).all()

    def test_aic_bic(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        assert np.isfinite(gam.aic)
        assert np.isfinite(gam.bic)

    def test_smooth_tests(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        tests = gam.smooth_tests()
        assert len(tests) == 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class TestInverseGaussianSimulation:
    def test_simulate(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape == (len(ig_data["y"]), 10)
        assert np.all(sims > 0)

    def test_simulate_unconditional(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data)
        sims = gam.simulate(n_sim=10, seed=23, unconditional=True)
        assert np.isfinite(sims).all()
        assert np.all(sims > 0)


# ---------------------------------------------------------------------------
# With select / weights
# ---------------------------------------------------------------------------


class TestInverseGaussianCombinations:
    def test_with_select(self, ig_data):
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data, select=True)
        assert gam.is_fitted

    def test_with_weights(self, ig_data):
        n = len(ig_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=InverseGaussian())
        gam.fit(ig_data, weights=w)
        assert gam.is_fitted
