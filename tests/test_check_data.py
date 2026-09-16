"""Tests for the GAM.check_data() method and CheckDataResult dataclass."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from whittaker.families.poisson import Poisson
from whittaker.gam import GAM, CheckDataResult


def _gaussian_data(n=200):
    rng = np.random.default_rng(23)
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


def _poisson_data(n=200):
    rng = np.random.default_rng(23)
    x = np.linspace(0, 2, n)
    mu = np.exp(0.5 + 0.8 * x)
    y = rng.poisson(mu).astype(float)
    return {"x": x, "y": y}


class TestCheckDataResult:
    @pytest.fixture()
    def fitted_model(self):
        data = _gaussian_data()
        model = GAM("y ~ s(x)")
        model.fit(data)
        return model

    @pytest.fixture()
    def result(self, fitted_model):
        return fitted_model.check_data()

    def test_return_type(self, result):
        assert isinstance(result, CheckDataResult)

    def test_array_shapes(self, result):
        n = 200

        assert result.deviance_residuals.shape == (n,)
        assert result.pearson_residuals.shape == (n,)
        assert result.fitted_values.shape == (n,)
        assert result.response.shape == (n,)
        assert result.qq_theoretical.shape == (n,)
        assert result.qq_observed.shape == (n,)

    def test_n_obs_property(self, result):
        assert result.n_obs == 200

    def test_repr(self, result):
        r = repr(result)

        assert r == "CheckDataResult(n_obs=200)"

    def test_qq_observed_is_sorted(self, result):
        npt.assert_array_equal(result.qq_observed, np.sort(result.qq_observed))

    def test_qq_theoretical_is_sorted(self, result):
        npt.assert_array_equal(result.qq_theoretical, np.sort(result.qq_theoretical))

    def test_qq_lengths_match_n_obs(self, result):
        assert len(result.qq_observed) == result.n_obs
        assert len(result.qq_theoretical) == result.n_obs

    def test_pearson_and_deviance_residuals_differ(self):
        # For Gaussian, pearson and deviance residuals are identical,
        # so use Poisson where they differ.
        data = _poisson_data()
        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)
        result = model.check_data()

        assert not np.array_equal(result.pearson_residuals, result.deviance_residuals)

    def test_fitted_values_match_model(self, fitted_model):
        result = fitted_model.check_data()

        npt.assert_array_almost_equal(result.fitted_values, fitted_model.fitted_values)

    def test_response_matches_training_data(self, fitted_model):
        data = _gaussian_data()
        result = fitted_model.check_data()

        npt.assert_array_almost_equal(result.response, data["y"])

    def test_unfitted_model_raises(self):
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.check_data()

    def test_poisson_family(self):
        data = _poisson_data()
        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)
        result = model.check_data()

        assert isinstance(result, CheckDataResult)
        assert result.n_obs == 200
        assert result.deviance_residuals.shape == (200,)
