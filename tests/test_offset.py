"""Tests for offset support in GAM fitting and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gaussian, Poisson
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def poisson_rate_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    log_exposure = rng.uniform(np.log(1), np.log(10), n)
    rate = np.exp(0.5 * np.sin(x))
    y = rng.poisson(rate * np.exp(log_exposure))
    return {"x": x, "y": y, "log_exposure": log_exposure}


@pytest.fixture()
def gaussian_offset_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    offset_vals = rng.uniform(-1, 1, n)
    y = np.sin(x) + offset_vals + rng.normal(0, 0.3, n)
    return {"x": x, "y": y, "off": offset_vals}


# ---------------------------------------------------------------------------
# Poisson rate model (classic offset use case)
# ---------------------------------------------------------------------------


class TestPoissonOffset:
    def test_fit_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        assert gam.is_fitted

    def test_deviance_explained(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        assert gam.deviance_explained > 0

    def test_predict_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        result = gam.predict(poisson_rate_data)
        assert result.values.shape == (len(poisson_rate_data["y"]),)
        assert np.all(result.values > 0)
        assert np.isfinite(result.values).all()

    def test_predict_on_new_data(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        rng = np.random.default_rng(99)
        new_data = {
            "x": rng.uniform(0, 2 * np.pi, 50),
            "log_exposure": np.zeros(50),
        }
        result = gam.predict(new_data)
        assert result.values.shape == (50,)
        assert np.all(result.values > 0)

    def test_offset_zero_recovers_rate(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        x_grid = np.linspace(0, 2 * np.pi, 100)
        zero_offset = np.zeros(100)
        result_zero = gam.predict({"x": x_grid, "log_exposure": zero_offset})
        result_one = gam.predict({"x": x_grid, "log_exposure": np.ones(100)})
        np.testing.assert_allclose(result_one.values, result_zero.values * np.e, rtol=0.01)

    def test_predict_se_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        result = gam.predict(poisson_rate_data, se=True)
        assert result.se is not None
        assert np.isfinite(result.se).all()

    def test_predict_interval_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        result = gam.predict(poisson_rate_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower >= 0)
        assert np.all(result.lower <= result.upper)

    def test_predict_link_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        link_result = gam.predict(poisson_rate_data, type="link")
        resp_result = gam.predict(poisson_rate_data)
        np.testing.assert_allclose(np.exp(link_result.values), resp_result.values, atol=1e-10)

    def test_summary_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        summary = gam.summary()
        assert "GAM fit summary" in summary

    def test_smooth_tests_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        tests = gam.smooth_tests()
        assert len(tests) == 1


# ---------------------------------------------------------------------------
# Gaussian with offset
# ---------------------------------------------------------------------------


class TestGaussianOffset:
    def test_fit_with_offset(self, gaussian_offset_data):
        gam = GAM("y ~ s(x) + offset(off)", family=Gaussian())
        gam.fit(gaussian_offset_data)
        assert gam.is_fitted

    def test_predict_with_offset(self, gaussian_offset_data):
        gam = GAM("y ~ s(x) + offset(off)", family=Gaussian())
        gam.fit(gaussian_offset_data)
        result = gam.predict(gaussian_offset_data)
        assert np.isfinite(result.values).all()

    def test_offset_included_in_prediction(self, gaussian_offset_data):
        gam = GAM("y ~ s(x) + offset(off)", family=Gaussian())
        gam.fit(gaussian_offset_data)
        x_grid = np.linspace(0, 2 * np.pi, 50)
        result_zero = gam.predict({"x": x_grid, "off": np.zeros(50)})
        result_shift = gam.predict({"x": x_grid, "off": np.full(50, 5.0)})
        np.testing.assert_allclose(result_shift.values, result_zero.values + 5.0, atol=1e-10)

    def test_offset_improves_fit(self, gaussian_offset_data):
        gam_no_off = GAM("y ~ s(x)", family=Gaussian())
        gam_no_off.fit(gaussian_offset_data)
        gam_off = GAM("y ~ s(x) + offset(off)", family=Gaussian())
        gam_off.fit(gaussian_offset_data)
        assert gam_off.deviance < gam_no_off.deviance


# ---------------------------------------------------------------------------
# Predict terms with offset
# ---------------------------------------------------------------------------


class TestPredictTermsWithOffset:
    def test_terms_dont_include_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        result = gam.predict(poisson_rate_data, type="terms")
        assert len(result.terms) == 1

    def test_terms_plus_intercept_plus_offset_equals_eta(self, gaussian_offset_data):
        gam = GAM("y ~ s(x) + offset(off)", family=Gaussian())
        gam.fit(gaussian_offset_data)
        pred = gam.predict(gaussian_offset_data)
        terms = gam.predict(gaussian_offset_data, type="terms")
        intercept = gam.coefficients[0]
        eta_from_terms = sum(terms.terms.values()) + intercept + gaussian_offset_data["off"]
        np.testing.assert_allclose(eta_from_terms, pred.linear_predictor, atol=1e-10)


# ---------------------------------------------------------------------------
# Missing offset in new data
# ---------------------------------------------------------------------------


class TestOffsetMissingData:
    def test_missing_offset_column_raises(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data)
        with pytest.raises(KeyError):
            gam.predict({"x": np.ones(10)})


# ---------------------------------------------------------------------------
# No offset model unchanged
# ---------------------------------------------------------------------------


class TestNoOffset:
    def test_no_offset_predict_unchanged(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(data)
        result = gam.predict(data)
        assert result.values.shape == (n,)
        assert np.isfinite(result.values).all()


# ---------------------------------------------------------------------------
# Offset with weights
# ---------------------------------------------------------------------------


class TestOffsetWithWeights:
    def test_offset_and_weights_together(self, poisson_rate_data):
        n = len(poisson_rate_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data, weights=w)
        assert gam.is_fitted
        result = gam.predict(poisson_rate_data)
        assert np.all(result.values > 0)


# ---------------------------------------------------------------------------
# REML with offset
# ---------------------------------------------------------------------------


class TestOffsetREML:
    def test_reml_with_offset(self, poisson_rate_data):
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(poisson_rate_data, method="REML")
        assert gam.is_fitted
        assert gam.deviance_explained > 0
