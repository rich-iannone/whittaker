"""Tests for Tweedie GAM fitting."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Tweedie
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tweedie_data():
    """Compound Poisson-Gamma data (p=1.5) with a smooth effect."""
    rng = np.random.default_rng(23)
    n = 300
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
    return {"x": x, "y": y}


@pytest.fixture()
def tweedie_two_smooth():
    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 2 * np.pi, n)
    mu = np.exp(0.3 * np.sin(x1) + 0.2 * np.cos(x2) + 0.5)
    p, phi = 1.5, 1.0

    lam = mu ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    gamma_scale = phi * (p - 1) * mu ** (p - 1)

    y = np.zeros(n)
    for i in range(n):
        N_i = rng.poisson(lam[i])
        if N_i > 0:
            y[i] = np.sum(rng.gamma(alpha, gamma_scale[i], size=N_i))
    return {"x1": x1, "x2": x2, "y": y}


# ---------------------------------------------------------------------------
# Basic fitting
# ---------------------------------------------------------------------------


class TestTweedieFitting:
    def test_fit(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        assert gam.is_fitted

    def test_fit_reml(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data, method="REML")
        assert gam.is_fitted

    def test_coefficients_finite(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        assert np.isfinite(gam.coefficients).all()

    def test_deviance_explained(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        assert gam.deviance_explained > 0

    def test_scale_estimated(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        assert gam.scale > 0
        assert np.isfinite(gam.scale)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestTweediePrediction:
    def test_predict_response(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        result = gam.predict(tweedie_data)
        assert np.all(result.values > 0)
        assert np.isfinite(result.values).all()

    def test_predict_link(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        link_result = gam.predict(tweedie_data, type="link")
        resp_result = gam.predict(tweedie_data)
        np.testing.assert_allclose(np.exp(link_result.values), resp_result.values, atol=1e-10)

    def test_predict_se(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        result = gam.predict(tweedie_data, se=True)
        assert result.se is not None
        assert np.all(result.se > 0)
        assert np.isfinite(result.se).all()

    def test_predict_new_data(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        new_data = {"x": np.linspace(0, 2 * np.pi, 50)}
        result = gam.predict(new_data)
        assert result.values.shape == (50,)
        assert np.all(result.values > 0)

    def test_predict_confidence_interval(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        result = gam.predict(tweedie_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower > 0)
        assert np.all(result.lower <= result.upper)

    def test_predict_terms(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        result = gam.predict(tweedie_data, type="terms")
        assert len(result.terms) == 1


# ---------------------------------------------------------------------------
# Summary and diagnostics
# ---------------------------------------------------------------------------


class TestTweedieDiagnostics:
    def test_summary(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        summary = gam.summary()
        assert "GAM fit summary" in summary
        assert "Tweedie" in summary

    def test_smooth_tests(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        tests = gam.smooth_tests()
        assert len(tests) == 1
        assert tests[0].edf > 0

    def test_residuals(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        for rtype in ("response", "pearson", "deviance", "working"):
            r = gam.get_residuals(rtype)
            assert np.isfinite(r).all()

    def test_aic_bic(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        assert np.isfinite(gam.aic)
        assert np.isfinite(gam.bic)


# ---------------------------------------------------------------------------
# With select
# ---------------------------------------------------------------------------


class TestTweedieWithSelect:
    def test_select(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data, select=True)
        assert gam.is_fitted

    def test_select_reml(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data, select=True, method="REML")
        assert gam.is_fitted


# ---------------------------------------------------------------------------
# With weights
# ---------------------------------------------------------------------------


class TestTweedieWithWeights:
    def test_with_weights(self, tweedie_data):
        n = len(tweedie_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data, weights=w)
        assert gam.is_fitted


# ---------------------------------------------------------------------------
# With offset
# ---------------------------------------------------------------------------


class TestTweedieWithOffset:
    def test_with_offset(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        log_exposure = rng.uniform(0, 1, n)
        mu = np.exp(0.3 * np.sin(x) + log_exposure)
        p, phi = 1.5, 0.5
        lam = mu ** (2 - p) / (phi * (2 - p))
        alpha = (2 - p) / (p - 1)
        gamma_scale = phi * (p - 1) * mu ** (p - 1)

        y = np.zeros(n)
        for i in range(n):
            N_i = rng.poisson(lam[i])
            if N_i > 0:
                y[i] = np.sum(rng.gamma(alpha, gamma_scale[i], size=N_i))
        data = {"x": x, "y": y, "log_exposure": log_exposure}
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Tweedie(p=1.5))
        gam.fit(data)
        assert gam.is_fitted
        result = gam.predict(data)
        assert np.all(result.values > 0)


# ---------------------------------------------------------------------------
# Posterior simulation
# ---------------------------------------------------------------------------


class TestTweedieSimulation:
    def test_simulate_conditional(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape == (len(tweedie_data["y"]), 10)
        assert np.all(sims > 0)

    def test_simulate_unconditional(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        sims = gam.simulate(n_sim=50, seed=23, unconditional=True)
        assert np.isfinite(sims).all()
        assert np.all(sims >= 0)


# ---------------------------------------------------------------------------
# Two smooths
# ---------------------------------------------------------------------------


class TestTweedieTwoSmooths:
    def test_two_smooths(self, tweedie_two_smooth):
        gam = GAM("y ~ s(x1) + s(x2)", family=Tweedie(p=1.5))
        gam.fit(tweedie_two_smooth)
        assert gam.is_fitted
        assert len(gam.edf) == 2
        assert gam.deviance_explained > 0


# ---------------------------------------------------------------------------
# Different variance powers
# ---------------------------------------------------------------------------


class TestTweedieVaryingP:
    def test_p_near_1(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.1))
        gam.fit(tweedie_data)
        assert gam.is_fitted

    def test_p_near_2(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.9))
        gam.fit(tweedie_data)
        assert gam.is_fitted

    def test_p_above_2(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        mu = np.exp(0.5 * np.sin(x) + 1.0)
        y = rng.gamma(5, mu / 5)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Tweedie(p=2.5))
        gam.fit(data)
        assert gam.is_fitted
        assert gam.deviance_explained > 0


# ---------------------------------------------------------------------------
# k_check
# ---------------------------------------------------------------------------


class TestTweedieKCheck:
    def test_k_check(self, tweedie_data):
        gam = GAM("y ~ s(x)", family=Tweedie(p=1.5))
        gam.fit(tweedie_data)
        results = gam.k_check(n_sim=50)
        assert len(results) == 1
        assert results[0].k_index > 0
