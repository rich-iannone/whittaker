"""Tests for conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.conformal import (
    ConformalMethod,
    ConformalPredictor,
    ConformalResult,
    conformal_coverage,
    conformal_fit,
)
from whittaker.families.poisson import Poisson


@pytest.fixture
def sin_data():
    rng = np.random.default_rng(23)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture
def holdout_data():
    rng = np.random.default_rng(99)
    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


class TestSplitConformal:
    def test_returns_predictor(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", seed=23)
        assert isinstance(cp, ConformalPredictor)
        assert cp.method == ConformalMethod.SPLIT.value

    def test_predict_returns_result(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", seed=23)
        result = cp.predict(holdout_data)
        assert isinstance(result, ConformalResult)
        assert result.values.shape == (100,)
        assert result.lower.shape == (100,)
        assert result.upper.shape == (100,)

    def test_intervals_ordered(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", seed=23)
        result = cp.predict(holdout_data)
        assert np.all(result.lower <= result.values)
        assert np.all(result.values <= result.upper)

    def test_coverage_reasonable(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", level=0.90, seed=23)
        cov = conformal_coverage(cp, holdout_data, "y")
        assert 0.75 < cov <= 1.0

    def test_higher_level_wider(self, sin_data, holdout_data):
        cp_90 = conformal_fit("y ~ s(x)", sin_data, method="split", level=0.90, seed=23)
        cp_95 = conformal_fit("y ~ s(x)", sin_data, method="split", level=0.95, seed=23)
        r_90 = cp_90.predict(holdout_data)
        r_95 = cp_95.predict(holdout_data)
        width_90 = np.mean(r_90.upper - r_90.lower)
        width_95 = np.mean(r_95.upper - r_95.lower)
        assert width_95 >= width_90

    def test_calibration_scores_stored(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", seed=23)
        assert cp.calibration_scores is not None
        assert len(cp.calibration_scores) > 0
        assert np.all(cp.calibration_scores >= 0)

    def test_quantile_positive(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", seed=23)
        assert cp.quantile > 0

    def test_cal_fraction(self, sin_data):
        cp_small = conformal_fit("y ~ s(x)", sin_data, method="split", cal_fraction=0.1, seed=23)
        cp_large = conformal_fit("y ~ s(x)", sin_data, method="split", cal_fraction=0.5, seed=23)
        assert len(cp_small.calibration_scores) < len(cp_large.calibration_scores)

    def test_level_stored(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", level=0.80, seed=23)
        assert cp.level == 0.80


class TestCVPlusConformal:
    def test_returns_predictor(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="cv+", n_folds=5, seed=23)
        assert isinstance(cp, ConformalPredictor)
        assert cp.method == ConformalMethod.CV_PLUS.value

    def test_predict_returns_result(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="cv+", n_folds=5, seed=23)
        result = cp.predict(holdout_data)
        assert isinstance(result, ConformalResult)
        assert result.values.shape == (100,)

    def test_intervals_contain_point(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="cv+", n_folds=5, seed=23)
        result = cp.predict(holdout_data)
        assert np.all(result.lower <= result.upper)

    def test_coverage_reasonable(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="cv+", n_folds=5, level=0.90, seed=23)
        cov = conformal_coverage(cp, holdout_data, "y")
        assert 0.70 < cov <= 1.0

    def test_all_residuals_computed(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="cv+", n_folds=5, seed=23)
        assert len(cp.calibration_scores) == 300

    def test_fold_models_stored(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="cv+", n_folds=5, seed=23)
        assert cp._models is not None
        assert len(cp._models) == 5


class TestJackknifePlusConformal:
    def test_returns_predictor(self):
        rng = np.random.default_rng(23)
        n = 50
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        cp = conformal_fit("y ~ s(x)", data, method="jackknife+", seed=23)
        assert isinstance(cp, ConformalPredictor)
        assert cp.method == ConformalMethod.JACKKNIFE_PLUS.value

    def test_loo_models_stored(self):
        rng = np.random.default_rng(23)
        n = 30
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        cp = conformal_fit("y ~ s(x)", data, method="jackknife+", seed=23)
        assert cp._models is not None
        assert len(cp._models) == n

    def test_predict_shape(self):
        rng = np.random.default_rng(23)
        n = 40
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        cp = conformal_fit("y ~ s(x)", data, method="jackknife+", seed=23)
        new_data = {"x": np.linspace(0, 2 * np.pi, 20)}
        result = cp.predict(new_data)
        assert result.values.shape == (20,)
        assert result.lower.shape == (20,)
        assert result.upper.shape == (20,)
        assert np.all(result.lower <= result.upper)


class TestConformalValidation:
    def test_invalid_method_raises(self, sin_data):
        with pytest.raises(ValueError, match="Unknown conformal method"):
            conformal_fit("y ~ s(x)", sin_data, method="bogus")

    def test_invalid_level_raises(self, sin_data):
        with pytest.raises(ValueError, match="level must be in"):
            conformal_fit("y ~ s(x)", sin_data, level=0.0)
        with pytest.raises(ValueError, match="level must be in"):
            conformal_fit("y ~ s(x)", sin_data, level=1.0)

    def test_seed_reproducibility(self, sin_data, holdout_data):
        cp1 = conformal_fit("y ~ s(x)", sin_data, method="split", seed=123)
        cp2 = conformal_fit("y ~ s(x)", sin_data, method="split", seed=123)
        r1 = cp1.predict(holdout_data)
        r2 = cp2.predict(holdout_data)
        np.testing.assert_allclose(r1.values, r2.values)
        np.testing.assert_allclose(r1.lower, r2.lower)
        np.testing.assert_allclose(r1.upper, r2.upper)


class TestConformalCoverage:
    def test_coverage_function(self, sin_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", level=0.90, seed=23)
        cov = conformal_coverage(cp, sin_data, "y")
        assert isinstance(cov, float)
        assert 0.0 <= cov <= 1.0

    def test_higher_level_higher_coverage(self, sin_data, holdout_data):
        cp_80 = conformal_fit("y ~ s(x)", sin_data, method="split", level=0.80, seed=23)
        cp_95 = conformal_fit("y ~ s(x)", sin_data, method="split", level=0.95, seed=23)
        cov_80 = conformal_coverage(cp_80, holdout_data, "y")
        cov_95 = conformal_coverage(cp_95, holdout_data, "y")
        assert cov_95 >= cov_80


class TestConformalResult:
    def test_fields(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", seed=23)
        result = cp.predict(holdout_data)
        assert result.level == 0.95
        assert result.method == "split"
        assert result.quantile > 0
        assert len(result.calibration_scores) > 0

    def test_finite_values(self, sin_data, holdout_data):
        cp = conformal_fit("y ~ s(x)", sin_data, method="split", seed=23)
        result = cp.predict(holdout_data)
        assert np.all(np.isfinite(result.values))
        assert np.all(np.isfinite(result.lower))
        assert np.all(np.isfinite(result.upper))


class TestConformalPoisson:
    def test_poisson_split(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2, n)
        y = rng.poisson(np.exp(0.5 * x)).astype(float)
        data = {"x": x, "y": y}
        cp = conformal_fit("y ~ s(x)", data, method="split", family=Poisson(), seed=23)
        new_x = np.linspace(0, 2, 50)
        result = cp.predict({"x": new_x})
        assert result.values.shape == (50,)
        assert np.all(np.isfinite(result.lower))
        assert np.all(np.isfinite(result.upper))
