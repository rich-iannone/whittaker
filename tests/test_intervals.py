"""Tests for confidence and prediction intervals."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as t_dist

from whittaker.families import Gamma, Gaussian, Poisson
from whittaker.gam import GAM, PredictionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gaussian_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture()
def poisson_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    mu = np.exp(0.5 * np.sin(x))
    y = rng.poisson(mu)
    return {"x": x, "y": y}


@pytest.fixture()
def fitted_gaussian(gaussian_data):
    return GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data)


@pytest.fixture()
def fitted_poisson(poisson_data):
    return GAM("y ~ s(x)", family=Poisson()).fit(poisson_data)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    def test_returns_lower_upper(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence")
        assert isinstance(result, PredictionResult)
        assert result.lower is not None
        assert result.upper is not None

    def test_lower_below_upper(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence")
        assert np.all(result.lower <= result.upper)

    def test_values_inside_interval(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence")
        assert np.all(result.values >= result.lower - 1e-10)
        assert np.all(result.values <= result.upper + 1e-10)

    def test_shapes(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence")
        n = len(gaussian_data["y"])
        assert result.lower.shape == (n,)
        assert result.upper.shape == (n,)

    def test_finite(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence")
        assert np.isfinite(result.lower).all()
        assert np.isfinite(result.upper).all()

    def test_no_se_by_default(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence")
        assert result.se is None

    def test_se_when_requested(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence", se=True)
        assert result.se is not None

    def test_level_default_95(self, fitted_gaussian, gaussian_data):
        result_95 = fitted_gaussian.predict(gaussian_data, interval="confidence", level=0.95)
        result_99 = fitted_gaussian.predict(gaussian_data, interval="confidence", level=0.99)
        width_95 = result_95.upper - result_95.lower
        width_99 = result_99.upper - result_99.lower
        assert np.all(width_99 >= width_95 - 1e-10)

    def test_narrower_at_lower_level(self, fitted_gaussian, gaussian_data):
        result_90 = fitted_gaussian.predict(gaussian_data, interval="confidence", level=0.90)
        result_95 = fitted_gaussian.predict(gaussian_data, interval="confidence", level=0.95)
        width_90 = result_90.upper - result_90.lower
        width_95 = result_95.upper - result_95.lower
        assert np.all(width_95 >= width_90 - 1e-10)

    def test_poisson_positive_bounds(self, fitted_poisson, poisson_data):
        result = fitted_poisson.predict(poisson_data, interval="confidence")
        assert np.all(result.lower >= 0)
        assert np.all(result.upper >= 0)

    def test_poisson_uses_z_quantile(self, fitted_poisson, poisson_data):
        result = fitted_poisson.predict(poisson_data, interval="confidence", se=True)
        z = norm.ppf(0.975)
        eta = result.linear_predictor
        se = result.se
        expected_lower = np.exp(eta - z * se)
        expected_upper = np.exp(eta + z * se)
        np.testing.assert_allclose(result.lower, expected_lower, rtol=1e-10)
        np.testing.assert_allclose(result.upper, expected_upper, rtol=1e-10)

    def test_gaussian_uses_t_quantile(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="confidence", se=True)
        n = len(gaussian_data["y"])
        residual_df = n - fitted_gaussian.edf_total
        t_val = t_dist.ppf(0.975, df=residual_df)
        eta = result.linear_predictor
        se = result.se
        expected_lower = eta - t_val * se
        expected_upper = eta + t_val * se
        np.testing.assert_allclose(result.lower, expected_lower, rtol=1e-10)
        np.testing.assert_allclose(result.upper, expected_upper, rtol=1e-10)

    def test_on_new_data(self, fitted_gaussian):
        rng = np.random.default_rng(99)
        new_data = {"x": rng.uniform(0, 2 * np.pi, 50)}
        result = fitted_gaussian.predict(new_data, interval="confidence")
        assert result.lower.shape == (50,)
        assert result.upper.shape == (50,)
        assert np.all(result.lower <= result.upper)


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------


class TestPredictionInterval:
    def test_returns_lower_upper(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="prediction")
        assert result.lower is not None
        assert result.upper is not None

    def test_lower_below_upper(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="prediction")
        assert np.all(result.lower <= result.upper)

    def test_wider_than_confidence(self, fitted_gaussian, gaussian_data):
        ci = fitted_gaussian.predict(gaussian_data, interval="confidence")
        pi = fitted_gaussian.predict(gaussian_data, interval="prediction")
        ci_width = ci.upper - ci.lower
        pi_width = pi.upper - pi.lower
        assert np.all(pi_width >= ci_width - 1e-10)

    def test_shapes(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="prediction")
        n = len(gaussian_data["y"])
        assert result.lower.shape == (n,)
        assert result.upper.shape == (n,)

    def test_finite(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="prediction")
        assert np.isfinite(result.lower).all()
        assert np.isfinite(result.upper).all()

    def test_gaussian_width_includes_scale(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, interval="prediction", se=True)
        n = len(gaussian_data["y"])
        residual_df = n - fitted_gaussian.edf_total
        t_val = t_dist.ppf(0.975, df=residual_df)
        se_eta = result.se
        scale = fitted_gaussian.scale
        total_se = np.sqrt(se_eta**2 + scale)
        expected_lower = result.linear_predictor - t_val * total_se
        expected_upper = result.linear_predictor + t_val * total_se
        np.testing.assert_allclose(result.lower, expected_lower, rtol=1e-10)
        np.testing.assert_allclose(result.upper, expected_upper, rtol=1e-10)

    def test_poisson_wider_than_confidence(self, fitted_poisson, poisson_data):
        ci = fitted_poisson.predict(poisson_data, interval="confidence")
        pi = fitted_poisson.predict(poisson_data, interval="prediction")
        ci_width = ci.upper - ci.lower
        pi_width = pi.upper - pi.lower
        assert np.all(pi_width >= ci_width - 1e-10)

    def test_poisson_positive_bounds(self, fitted_poisson, poisson_data):
        result = fitted_poisson.predict(poisson_data, interval="prediction")
        assert np.all(result.lower >= 0)
        assert np.all(result.upper >= 0)

    def test_level_affects_width(self, fitted_gaussian, gaussian_data):
        pi_90 = fitted_gaussian.predict(gaussian_data, interval="prediction", level=0.90)
        pi_99 = fitted_gaussian.predict(gaussian_data, interval="prediction", level=0.99)
        width_90 = pi_90.upper - pi_90.lower
        width_99 = pi_99.upper - pi_99.lower
        assert np.all(width_99 >= width_90 - 1e-10)

    def test_on_new_data(self, fitted_gaussian):
        rng = np.random.default_rng(99)
        new_data = {"x": rng.uniform(0, 2 * np.pi, 50)}
        result = fitted_gaussian.predict(new_data, interval="prediction")
        assert result.lower.shape == (50,)
        assert np.all(result.lower <= result.upper)


# ---------------------------------------------------------------------------
# Link-scale intervals
# ---------------------------------------------------------------------------


class TestLinkScaleInterval:
    def test_confidence_on_link(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, type="link", interval="confidence")
        assert result.lower is not None
        assert result.upper is not None
        assert np.all(result.lower <= result.upper)

    def test_prediction_on_link(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, type="link", interval="prediction")
        assert result.lower is not None
        assert np.all(result.lower <= result.upper)

    def test_link_ci_symmetric_around_eta(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data, type="link", interval="confidence")
        eta = result.linear_predictor
        np.testing.assert_allclose(eta - result.lower, result.upper - eta, rtol=1e-10)

    def test_poisson_link_ci_stays_on_link(self, fitted_poisson, poisson_data):
        link_result = fitted_poisson.predict(poisson_data, type="link", interval="confidence")
        resp_result = fitted_poisson.predict(poisson_data, interval="confidence")
        np.testing.assert_allclose(np.exp(link_result.lower), resp_result.lower, rtol=1e-10)
        np.testing.assert_allclose(np.exp(link_result.upper), resp_result.upper, rtol=1e-10)


# ---------------------------------------------------------------------------
# Gamma family
# ---------------------------------------------------------------------------


class TestGammaInterval:
    def test_gamma_confidence(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0.5, 3.0, n)
        mu = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Gamma()).fit(data)
        result = gam.predict(data, interval="confidence")
        assert np.all(result.lower > 0)
        assert np.all(result.lower <= result.upper)

    def test_gamma_prediction_wider(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0.5, 3.0, n)
        mu = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Gamma()).fit(data)
        ci = gam.predict(data, interval="confidence")
        pi = gam.predict(data, interval="prediction")
        ci_width = ci.upper - ci.lower
        pi_width = pi.upper - pi.lower
        assert np.all(pi_width >= ci_width - 1e-10)


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------


class TestIntervalValidation:
    def test_unknown_interval_raises(self, fitted_gaussian, gaussian_data):
        with pytest.raises(ValueError, match="Unknown interval type"):
            fitted_gaussian.predict(gaussian_data, interval="unknown")

    def test_interval_not_supported_for_terms(self, fitted_gaussian, gaussian_data):
        with pytest.raises(ValueError, match="not supported for type='terms'"):
            fitted_gaussian.predict(gaussian_data, type="terms", interval="confidence")

    def test_no_interval_by_default(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data)
        assert result.lower is None
        assert result.upper is None

    def test_no_interval_no_se_overhead(self, fitted_gaussian, gaussian_data):
        result = fitted_gaussian.predict(gaussian_data)
        assert result.se is None
        assert result.lower is None


# ---------------------------------------------------------------------------
# Two-smooth model
# ---------------------------------------------------------------------------


class TestMultiSmoothInterval:
    def test_two_smooth_confidence(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(data)
        result = gam.predict(data, interval="confidence")
        assert np.all(result.lower <= result.upper)
        assert np.isfinite(result.lower).all()

    def test_two_smooth_prediction_wider(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(data)
        ci = gam.predict(data, interval="confidence")
        pi = gam.predict(data, interval="prediction")
        assert np.all((pi.upper - pi.lower) >= (ci.upper - ci.lower) - 1e-10)
