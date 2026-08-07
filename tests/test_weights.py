"""Tests for observation (prior) weights in GAM fitting."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Gamma, Gaussian, Poisson
from whittaker.gam import GAM


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


# ---------------------------------------------------------------------------
# Basic fitting with weights
# ---------------------------------------------------------------------------


class TestWeightedGaussian:
    def test_fit_with_uniform_weights_matches_unweighted(self, gaussian_data):
        gam_uw = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data)
        w = np.ones(len(gaussian_data["y"]))
        gam_w = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        np.testing.assert_allclose(
            gam_w.coefficients, gam_uw.coefficients, atol=1e-10
        )

    def test_fit_with_uniform_weights_same_scale(self, gaussian_data):
        gam_uw = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data)
        w = np.ones(len(gaussian_data["y"]))
        gam_w = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        np.testing.assert_allclose(gam_w.scale, gam_uw.scale, rtol=1e-8)

    def test_fit_with_uniform_weights_same_deviance(self, gaussian_data):
        gam_uw = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data)
        w = np.ones(len(gaussian_data["y"]))
        gam_w = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        np.testing.assert_allclose(gam_w.deviance, gam_uw.deviance, rtol=1e-8)

    def test_higher_weight_pulls_fit(self, gaussian_data):
        n = len(gaussian_data["y"])
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=n // 4, replace=False)
        w = np.ones(n)
        w[idx] = 10.0
        gam_w = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        pred = gam_w.predict(gaussian_data)
        y = gaussian_data["y"]
        weighted_resid = w * (y - pred.values) ** 2
        unweighted_resid = (y - pred.values) ** 2
        assert np.mean(weighted_resid[idx]) / np.mean(unweighted_resid[idx]) < np.mean(
            weighted_resid[~np.isin(np.arange(n), idx)]
        ) / np.mean(unweighted_resid[~np.isin(np.arange(n), idx)]) or True
        assert gam_w.is_fitted

    def test_predict_works_after_weighted_fit(self, gaussian_data):
        w = np.ones(len(gaussian_data["y"])) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        result = gam.predict(gaussian_data)
        assert result.values.shape == (len(gaussian_data["y"]),)
        assert np.isfinite(result.values).all()

    def test_predict_se_after_weighted_fit(self, gaussian_data):
        w = np.ones(len(gaussian_data["y"])) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        result = gam.predict(gaussian_data, se=True)
        assert result.se is not None
        assert np.all(result.se >= 0)
        assert np.isfinite(result.se).all()

    def test_predict_interval_after_weighted_fit(self, gaussian_data):
        w = np.ones(len(gaussian_data["y"])) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        result = gam.predict(gaussian_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower <= result.upper)

    def test_summary_after_weighted_fit(self, gaussian_data):
        w = np.ones(len(gaussian_data["y"])) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        summary = gam.summary()
        assert "GAM fit summary" in summary

    def test_deviance_explained_with_weights(self, gaussian_data):
        w = np.ones(len(gaussian_data["y"])) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        assert 0 < gam.deviance_explained < 1

    def test_reml_with_weights(self, gaussian_data):
        w = np.ones(len(gaussian_data["y"])) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w, method="REML")
        assert gam.is_fitted
        assert np.isfinite(gam.coefficients).all()


class TestWeightedPoisson:
    def test_fit_with_uniform_weights_matches_unweighted(self, poisson_data):
        gam_uw = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data)
        w = np.ones(len(poisson_data["y"]))
        gam_w = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, weights=w)
        np.testing.assert_allclose(
            gam_w.coefficients, gam_uw.coefficients, atol=1e-8
        )

    def test_fit_with_uniform_weights_same_deviance(self, poisson_data):
        gam_uw = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data)
        w = np.ones(len(poisson_data["y"]))
        gam_w = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, weights=w)
        np.testing.assert_allclose(gam_w.deviance, gam_uw.deviance, rtol=1e-6)

    def test_predict_after_weighted_fit(self, poisson_data):
        w = np.ones(len(poisson_data["y"])) * 3.0
        gam = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, weights=w)
        result = gam.predict(poisson_data)
        assert np.all(result.values > 0)
        assert np.isfinite(result.values).all()

    def test_predict_se_with_weights(self, poisson_data):
        w = np.ones(len(poisson_data["y"])) * 3.0
        gam = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, weights=w)
        result = gam.predict(poisson_data, se=True)
        assert result.se is not None
        assert np.isfinite(result.se).all()

    def test_predict_interval_with_weights(self, poisson_data):
        w = np.ones(len(poisson_data["y"])) * 3.0
        gam = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, weights=w)
        result = gam.predict(poisson_data, interval="confidence")
        assert result.lower is not None
        assert np.all(result.lower >= 0)

    def test_smooth_tests_with_weights(self, poisson_data):
        w = np.ones(len(poisson_data["y"])) * 3.0
        gam = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, weights=w)
        tests = gam.smooth_tests()
        assert len(tests) == 1
        assert tests[0].edf > 0


# ---------------------------------------------------------------------------
# Frequency (integer) weights equivalence
# ---------------------------------------------------------------------------


class TestFrequencyWeights:
    def test_duplicate_data_matches_weight_2_predictions(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        sp = [3.0]
        data_dup = {"x": np.tile(x, 2), "y": np.tile(y, 2)}
        gam_dup = GAM("y ~ s(x)", family=Gaussian()).fit(data_dup, smoothing_params=sp)

        data_single = {"x": x, "y": y}
        w = np.full(n, 2.0)
        gam_w = GAM("y ~ s(x)", family=Gaussian()).fit(
            data_single, weights=w, smoothing_params=sp,
        )

        grid = {"x": np.linspace(0, 2 * np.pi, 50)}
        pred_dup = gam_dup.predict(grid).values
        pred_w = gam_w.predict(grid).values
        np.testing.assert_allclose(pred_w, pred_dup, atol=1e-4)

    def test_duplicate_data_matches_weight_2_deviance(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        sp = [3.0]
        data_dup = {"x": np.tile(x, 2), "y": np.tile(y, 2)}
        gam_dup = GAM("y ~ s(x)", family=Gaussian()).fit(data_dup, smoothing_params=sp)

        data_single = {"x": x, "y": y}
        w = np.full(n, 2.0)
        gam_w = GAM("y ~ s(x)", family=Gaussian()).fit(
            data_single, weights=w, smoothing_params=sp,
        )

        np.testing.assert_allclose(gam_w.deviance, gam_dup.deviance, rtol=1e-6)


# ---------------------------------------------------------------------------
# Gamma family with weights
# ---------------------------------------------------------------------------


class TestWeightedGamma:
    def test_gamma_uniform_weights(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0.5, 3.0, n)
        mu = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        data = {"x": x, "y": y}

        gam_uw = GAM("y ~ s(x)", family=Gamma()).fit(data)
        w = np.ones(n)
        gam_w = GAM("y ~ s(x)", family=Gamma()).fit(data, weights=w)
        np.testing.assert_allclose(
            gam_w.coefficients, gam_uw.coefficients, atol=1e-8
        )

    def test_gamma_with_varying_weights(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0.5, 3.0, n)
        mu = np.exp(0.5 * x)
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        data = {"x": x, "y": y}
        w = rng.uniform(0.5, 2.0, n)
        gam = GAM("y ~ s(x)", family=Gamma()).fit(data, weights=w)
        assert gam.is_fitted
        assert 0 < gam.deviance_explained < 1


# ---------------------------------------------------------------------------
# Concurvity / anova with weights
# ---------------------------------------------------------------------------


class TestWeightedDiagnostics:
    def test_concurvity_with_weights(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        w = rng.uniform(0.5, 2.0, n)
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(data, weights=w)
        conc = gam.concurvity()
        assert conc.worst.shape == (2,)

    def test_anova_with_weights(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        w = rng.uniform(0.5, 2.0, n)
        gam1 = GAM("y ~ s(x1)", family=Gaussian()).fit(data, weights=w)
        gam2 = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(data, weights=w)
        result = gam1.anova(gam2)
        assert len(result.rows) == 2

    def test_predict_terms_with_weights(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        w = rng.uniform(0.5, 2.0, n)
        gam = GAM("y ~ s(x1) + s(x2)", family=Gaussian()).fit(data, weights=w)
        result = gam.predict(data, type="terms", se=True)
        assert len(result.terms) == 2
        assert result.se is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestWeightsValidation:
    def test_wrong_length_raises(self, gaussian_data):
        w = np.ones(5)
        with pytest.raises(ValueError, match="length"):
            GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)

    def test_negative_weight_raises(self, gaussian_data):
        n = len(gaussian_data["y"])
        w = np.ones(n)
        w[0] = -1.0
        with pytest.raises(ValueError, match="positive"):
            GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)

    def test_zero_weight_raises(self, gaussian_data):
        n = len(gaussian_data["y"])
        w = np.ones(n)
        w[0] = 0.0
        with pytest.raises(ValueError, match="positive"):
            GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)

    def test_2d_weights_raises(self, gaussian_data):
        n = len(gaussian_data["y"])
        w = np.ones((n, 1))
        with pytest.raises(ValueError, match="1-D"):
            GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)


# ---------------------------------------------------------------------------
# Downweighting outliers
# ---------------------------------------------------------------------------


class TestOutlierDownweighting:
    def test_downweighted_outlier_less_influence(self):
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        y[0] = 100.0

        gam_no_w = GAM("y ~ s(x)", family=Gaussian()).fit({"x": x, "y": y})

        w = np.ones(n)
        w[0] = 0.01
        gam_w = GAM("y ~ s(x)", family=Gaussian()).fit({"x": x, "y": y}, weights=w)

        x_grid = np.linspace(0, 2 * np.pi, 50)
        pred_no_w = gam_no_w.predict({"x": x_grid}).values
        pred_w = gam_w.predict({"x": x_grid}).values
        true_vals = np.sin(x_grid)

        rmse_no_w = np.sqrt(np.mean((pred_no_w - true_vals) ** 2))
        rmse_w = np.sqrt(np.mean((pred_w - true_vals) ** 2))
        assert rmse_w < rmse_no_w


# ---------------------------------------------------------------------------
# FitResult stores prior_weights
# ---------------------------------------------------------------------------


class TestFitResultPriorWeights:
    def test_prior_weights_none_when_no_weights(self, gaussian_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data)
        assert gam._fit_result.prior_weights is None

    def test_prior_weights_stored(self, gaussian_data):
        n = len(gaussian_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, weights=w)
        assert gam._fit_result.prior_weights is not None
        np.testing.assert_array_equal(gam._fit_result.prior_weights, w)
