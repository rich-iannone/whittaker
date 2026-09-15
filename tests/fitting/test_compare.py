"""Tests for compare() and ComparisonResult."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker import GAM, ComparisonResult, ComparisonRow, compare
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
def three_models(gaussian_data):
    m1 = GAM("y ~ x").fit(gaussian_data)
    m2 = GAM("y ~ s(x, k=5)").fit(gaussian_data)
    m3 = GAM("y ~ s(x, k=15)").fit(gaussian_data)
    return m1, m2, m3


@pytest.fixture(scope="module")
def comparison(three_models):
    return compare(*three_models)


# ---------------------------------------------------------------------------
# ComparisonResult structure
# ---------------------------------------------------------------------------


class TestComparisonResultStructure:
    """Verify the result type and fields."""

    def test_returns_correct_type(self, comparison):
        assert isinstance(comparison, ComparisonResult)

    def test_n_models(self, comparison):
        assert comparison.n_models == 3

    def test_rows_are_comparison_rows(self, comparison):
        for row in comparison.rows:
            assert isinstance(row, ComparisonRow)

    def test_all_fields_present(self, comparison):
        row = comparison.rows[0]
        assert hasattr(row, "label")
        assert hasattr(row, "aic")
        assert hasattr(row, "delta_aic")
        assert hasattr(row, "bic")
        assert hasattr(row, "deviance_explained")
        assert hasattr(row, "r_squared_adj")
        assert hasattr(row, "edf_total")
        assert hasattr(row, "gcv_score")
        assert hasattr(row, "scale")
        assert hasattr(row, "n_obs")

    def test_n_obs_consistent(self, comparison, gaussian_data):
        for row in comparison.rows:
            assert row.n_obs == len(gaussian_data["y"])

    def test_repr(self, comparison):
        r = repr(comparison)
        assert "Model Comparison" in r
        assert "AIC" in r
        assert "ΔAIC" in r
        assert "BIC" in r
        assert "Dev.Expl." in r


# ---------------------------------------------------------------------------
# Sorting and delta AIC
# ---------------------------------------------------------------------------


class TestSorting:
    """Verify models are sorted by AIC with correct delta values."""

    def test_sorted_by_aic(self, comparison):
        aics = [r.aic for r in comparison.rows]
        assert aics == sorted(aics)

    def test_best_has_zero_delta(self, comparison):
        assert comparison.rows[0].delta_aic == 0.0

    def test_delta_aic_positive(self, comparison):
        for row in comparison.rows:
            assert row.delta_aic >= 0.0

    def test_delta_aic_correct(self, comparison):
        best_aic = comparison.rows[0].aic
        for row in comparison.rows:
            assert row.delta_aic == pytest.approx(row.aic - best_aic)

    def test_best_property(self, comparison):
        assert comparison.best is comparison.rows[0]


# ---------------------------------------------------------------------------
# Consistency with goodness_of_fit
# ---------------------------------------------------------------------------


class TestConsistency:
    """Verify compare() values match individual goodness_of_fit() calls."""

    def test_aic_matches(self, three_models):
        result = compare(*three_models)
        for model in three_models:
            gof = model.goodness_of_fit()
            matching = [r for r in result.rows if r.label == repr(model.formula)]
            assert len(matching) == 1
            assert matching[0].aic == pytest.approx(gof.aic)

    def test_bic_matches(self, three_models):
        result = compare(*three_models)
        for model in three_models:
            gof = model.goodness_of_fit()
            matching = [r for r in result.rows if r.label == repr(model.formula)]
            assert matching[0].bic == pytest.approx(gof.bic)

    def test_deviance_explained_matches(self, three_models):
        result = compare(*three_models)
        for model in three_models:
            gof = model.goodness_of_fit()
            matching = [r for r in result.rows if r.label == repr(model.formula)]
            assert matching[0].deviance_explained == pytest.approx(gof.deviance_explained)

    def test_edf_matches(self, three_models):
        result = compare(*three_models)
        for model in three_models:
            gof = model.goodness_of_fit()
            matching = [r for r in result.rows if r.label == repr(model.formula)]
            assert matching[0].edf_total == pytest.approx(gof.edf_total)


# ---------------------------------------------------------------------------
# Two-model comparison
# ---------------------------------------------------------------------------


class TestTwoModels:
    """Verify compare() works with exactly two models."""

    def test_two_models(self, gaussian_data):
        m1 = GAM("y ~ x").fit(gaussian_data)
        m2 = GAM("y ~ s(x)").fit(gaussian_data)
        result = compare(m1, m2)

        assert result.n_models == 2
        assert result.rows[0].delta_aic == 0.0


# ---------------------------------------------------------------------------
# Bayesian fits (GCV is None)
# ---------------------------------------------------------------------------


class TestBayesianFits:
    """Verify compare() handles Bayesian fits where GCV is None."""

    def test_vi_models(self, gaussian_data):
        m1 = GAM("y ~ x").fit(gaussian_data, method="VI")
        m2 = GAM("y ~ s(x)").fit(gaussian_data, method="VI")
        result = compare(m1, m2)

        assert result.n_models == 2
        for row in result.rows:
            assert row.gcv_score is None
        assert "---" in repr(result)


# ---------------------------------------------------------------------------
# Non-Gaussian family
# ---------------------------------------------------------------------------


class TestPoissonCompare:
    def test_poisson(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x))).astype(float)
        data = {"x": x, "y": y}

        m1 = GAM("y ~ x", family=Poisson()).fit(data)
        m2 = GAM("y ~ s(x)", family=Poisson()).fit(data)
        result = compare(m1, m2)

        assert isinstance(result, ComparisonResult)
        assert result.n_models == 2


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_fewer_than_two_raises(self, gaussian_data):
        m = GAM("y ~ s(x)").fit(gaussian_data)
        with pytest.raises(ValueError, match="at least 2"):
            compare(m)

    def test_unfitted_raises(self, gaussian_data):
        m1 = GAM("y ~ s(x)").fit(gaussian_data)
        m2 = GAM("y ~ x")
        with pytest.raises(RuntimeError, match="not been fitted"):
            compare(m1, m2)
