"""Tests for GAM.smoothing_sensitivity() and SensitivityResult."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker import GAM, SensitivityResult
from whittaker.families.poisson import Poisson

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(0)
    n = 100
    x = np.linspace(0, 5, n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def reml_model(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data, method="REML")


@pytest.fixture(scope="module")
def sens_result(reml_model):
    return reml_model.smoothing_sensitivity()


# ---------------------------------------------------------------------------
# SensitivityResult structure
# ---------------------------------------------------------------------------


class TestSensitivityResultStructure:
    """Verify the dataclass fields and shapes."""

    def test_returns_correct_type(self, sens_result):
        assert isinstance(sens_result, SensitivityResult)

    def test_default_n_steps(self, sens_result):
        assert sens_result.n_steps == 11

    def test_predictions_shape(self, sens_result, gaussian_data):
        n = len(gaussian_data["y"])
        assert sens_result.predictions.shape == (11, n)

    def test_edf_total_shape(self, sens_result):
        assert sens_result.edf_total.shape == (11,)

    def test_deviance_explained_shape(self, sens_result):
        assert sens_result.deviance_explained.shape == (11,)

    def test_gcv_scores_shape(self, sens_result):
        assert sens_result.gcv_scores.shape == (11,)

    def test_aic_values_shape(self, sens_result):
        assert sens_result.aic_values.shape == (11,)

    def test_smoothing_params_shape(self, sens_result):
        assert sens_result.smoothing_params.shape[0] == 11
        assert sens_result.smoothing_params.shape[1] >= 1

    def test_n_obs(self, sens_result, gaussian_data):
        assert sens_result.n_obs == len(gaussian_data["y"])

    def test_baseline_idx_in_range(self, sens_result):
        assert 0 <= sens_result.baseline_idx < sens_result.n_steps

    def test_repr(self, sens_result):
        r = repr(sens_result)
        assert "SensitivityResult" in r
        assert "Multiplier range" in r
        assert "EDF range" in r
        assert "Max |prediction change|" in r


# ---------------------------------------------------------------------------
# Multiplier behavior
# ---------------------------------------------------------------------------


class TestMultiplierBehavior:
    """Verify the multiplier grid and baseline detection."""

    def test_default_multiplier_range(self, sens_result):
        assert sens_result.multipliers[0] == pytest.approx(0.01, rel=0.01)
        assert sens_result.multipliers[-1] == pytest.approx(100.0, rel=0.01)

    def test_baseline_multiplier_is_one(self, sens_result):
        assert sens_result.multipliers[sens_result.baseline_idx] == pytest.approx(1.0)

    def test_custom_multipliers(self, reml_model):
        mults = [0.1, 0.5, 1.0, 2.0, 10.0]
        result = reml_model.smoothing_sensitivity(multipliers=mults)

        assert result.n_steps == 5
        np.testing.assert_allclose(result.multipliers, mults)

    def test_custom_n_steps(self, reml_model):
        result = reml_model.smoothing_sensitivity(n_steps=5)

        assert result.n_steps == 5

    def test_custom_log_range(self, reml_model):
        result = reml_model.smoothing_sensitivity(log_range=(-1.0, 1.0), n_steps=5)

        assert result.multipliers[0] == pytest.approx(0.1, rel=0.01)
        assert result.multipliers[-1] == pytest.approx(10.0, rel=0.01)


# ---------------------------------------------------------------------------
# Sensitivity behavior
# ---------------------------------------------------------------------------


class TestSensitivityBehavior:
    """Verify that varying smoothing parameters produces expected patterns."""

    def test_baseline_predictions_match_original(self, reml_model, sens_result):
        baseline_preds = sens_result.baseline_predictions
        original_preds = reml_model.predict(reml_model._data).values

        np.testing.assert_allclose(baseline_preds, original_preds, atol=1e-4)

    def test_max_abs_change_zero_at_baseline(self, sens_result):
        mac = sens_result.max_abs_change()

        assert mac[sens_result.baseline_idx] == pytest.approx(0.0, abs=1e-4)

    def test_more_smoothing_reduces_edf(self, sens_result):
        baseline_edf = sens_result.edf_total[sens_result.baseline_idx]
        high_lambda_edf = sens_result.edf_total[-1]

        assert high_lambda_edf < baseline_edf

    def test_less_smoothing_increases_edf(self, sens_result):
        baseline_edf = sens_result.edf_total[sens_result.baseline_idx]
        low_lambda_edf = sens_result.edf_total[0]

        assert low_lambda_edf > baseline_edf

    def test_edf_monotonically_decreasing(self, sens_result):
        for i in range(1, sens_result.n_steps):
            assert sens_result.edf_total[i] <= sens_result.edf_total[i - 1] + 1e-10

    def test_predictions_vary_across_steps(self, sens_result):
        pred_range = np.ptp(sens_result.predictions, axis=0)

        assert np.max(pred_range) > 0.01

    def test_all_values_finite(self, sens_result):
        assert np.all(np.isfinite(sens_result.predictions))
        assert np.all(np.isfinite(sens_result.edf_total))
        assert np.all(np.isfinite(sens_result.deviance_explained))
        assert np.all(np.isfinite(sens_result.gcv_scores))
        assert np.all(np.isfinite(sens_result.aic_values))


# ---------------------------------------------------------------------------
# New data predictions
# ---------------------------------------------------------------------------


class TestNewData:
    """Verify predictions on new data."""

    def test_new_data_shape(self, reml_model):
        x_new = np.linspace(0, 5, 50)
        result = reml_model.smoothing_sensitivity(new_data={"x": x_new})

        assert result.predictions.shape == (11, 50)

    def test_new_data_n_obs(self, reml_model):
        x_new = np.linspace(0, 5, 50)
        result = reml_model.smoothing_sensitivity(new_data={"x": x_new})

        assert result.n_obs == 50


# ---------------------------------------------------------------------------
# Fitting methods
# ---------------------------------------------------------------------------


class TestFittingMethods:
    """Verify compatibility with different frequentist methods."""

    def test_gcv_model(self, gaussian_data):
        model = GAM("y ~ s(x)").fit(gaussian_data, method="GCV")
        result = model.smoothing_sensitivity(n_steps=5)

        assert isinstance(result, SensitivityResult)
        assert result.n_steps == 5

    def test_ml_model(self, gaussian_data):
        model = GAM("y ~ s(x)").fit(gaussian_data, method="ML")
        result = model.smoothing_sensitivity(n_steps=5)

        assert isinstance(result, SensitivityResult)


# ---------------------------------------------------------------------------
# Non-Gaussian family
# ---------------------------------------------------------------------------


class TestPoissonSensitivity:
    def test_poisson(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x))).astype(float)
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Poisson()).fit(data)
        result = model.smoothing_sensitivity(n_steps=5)

        assert isinstance(result, SensitivityResult)
        assert np.all(result.predictions >= 0)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unfitted_raises(self):
        gam = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            gam.smoothing_sensitivity()

    def test_vi_raises(self, gaussian_data):
        model = GAM("y ~ s(x)").fit(gaussian_data, method="VI")
        with pytest.raises(NotImplementedError, match="frequentist"):
            model.smoothing_sensitivity()

    def test_mcmc_raises(self, gaussian_data):
        model = GAM("y ~ s(x)").fit(
            gaussian_data,
            method="MCMC",
            mcmc_options={"n_chains": 2, "n_samples": 200, "n_warmup": 100, "seed": 23},
        )
        with pytest.raises(NotImplementedError, match="frequentist"):
            model.smoothing_sensitivity()
