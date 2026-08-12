"""Tests for K-fold cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.cross_validation import CVResult, cross_validate


@pytest.fixture()
def gaussian_data():
    rng = np.random.default_rng(23)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


class TestCrossValidate:
    def test_returns_cv_result(self, gaussian_data):
        result = cross_validate("y ~ s(x)", gaussian_data, n_folds=3, seed=23)
        assert isinstance(result, CVResult)

    def test_cv_score_positive(self, gaussian_data):
        result = cross_validate("y ~ s(x)", gaussian_data, n_folds=3, seed=23)
        assert result.cv_score > 0

    def test_cv_se_positive(self, gaussian_data):
        result = cross_validate("y ~ s(x)", gaussian_data, n_folds=3, seed=23)
        assert result.cv_se > 0

    def test_n_folds_matches(self, gaussian_data):
        result = cross_validate("y ~ s(x)", gaussian_data, n_folds=3, seed=23)
        assert result.n_folds == 3
        assert len(result.cv_scores) == 3

    def test_mse_metric(self, gaussian_data):
        result = cross_validate(
            "y ~ s(x)",
            gaussian_data,
            n_folds=3,
            metric="mse",
            seed=23,
        )
        assert result.cv_score > 0

    def test_better_model_lower_cv(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        cv_good = cross_validate(
            "y ~ s(x)",
            data,
            n_folds=3,
            metric="mse",
            seed=23,
        )
        cv_bad = cross_validate(
            "y ~ 1",
            data,
            n_folds=3,
            metric="mse",
            seed=23,
        )
        assert cv_good.cv_score < cv_bad.cv_score

    def test_deterministic_with_seed(self, gaussian_data):
        r1 = cross_validate("y ~ s(x)", gaussian_data, n_folds=3, seed=23)
        r2 = cross_validate("y ~ s(x)", gaussian_data, n_folds=3, seed=23)
        np.testing.assert_array_equal(r1.cv_scores, r2.cv_scores)

    def test_fold_scores_all_finite(self, gaussian_data):
        result = cross_validate("y ~ s(x)", gaussian_data, n_folds=3, seed=23)
        assert np.all(np.isfinite(result.cv_scores))

    def test_n_folds_exceeding_n_obs_skips_empty_folds(self):
        """When n_folds exceeds the number of observations, some fold ids never get assigned
        any test points; those folds must be skipped rather than crashing on an empty test
        set."""
        rng = np.random.default_rng(23)
        n = 30
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)

        result = cross_validate("y ~ s(x, k=5)", {"x": x, "y": y}, n_folds=40, seed=23)

        assert result.n_folds == 40
        assert np.isfinite(result.cv_score)
        assert np.any(result.cv_scores == 0.0)
