"""Tests for anova_gam() model comparison."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Binomial, Gaussian, Poisson
from whittaker.fitting.inference import AnovaModelRow, AnovaResult, anova_gam
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
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    mu = np.exp(0.5 * np.sin(x))
    y = rng.poisson(mu)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Low-level anova_gam tests
# ---------------------------------------------------------------------------


class TestAnovaGam:
    def test_requires_at_least_two_models(self, gaussian_data):
        gam = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data)
        with pytest.raises(ValueError, match="at least 2 models"):
            anova_gam(
                (gam._fit_result, gam._model_matrix),
                scale_known=False,
            )

    def test_different_n_obs_raises(self):
        rng = np.random.default_rng(0)
        data1 = {"x": rng.uniform(size=100), "y": rng.normal(size=100)}
        data2 = {"x": rng.uniform(size=150), "y": rng.normal(size=150)}
        g1 = GAM("y ~ s(x)", family=Gaussian()).fit(data1)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(data2)
        with pytest.raises(ValueError, match="same data"):
            anova_gam(
                (g1._fit_result, g1._model_matrix),
                (g2._fit_result, g2._model_matrix),
                scale_known=False,
            )

    def test_two_gaussian_models(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = anova_gam(
            (g1._fit_result, g1._model_matrix),
            (g2._fit_result, g2._model_matrix),
            scale_known=False,
        )
        assert isinstance(result, AnovaResult)
        assert result.test == "F"
        assert len(result.rows) == 2
        assert result.rows[0].stat is None
        assert result.rows[1].stat is not None
        assert result.rows[1].p_value is not None
        assert 0.0 <= result.rows[1].p_value <= 1.0

    def test_two_poisson_models(self, poisson_data):
        g1 = GAM("y ~ s(x, k=4)", family=Poisson()).fit(poisson_data)
        g2 = GAM("y ~ s(x, k=15)", family=Poisson()).fit(poisson_data)
        result = anova_gam(
            (g1._fit_result, g1._model_matrix),
            (g2._fit_result, g2._model_matrix),
            scale_known=True,
        )
        assert result.test == "Chisq"
        assert len(result.rows) == 2
        assert result.rows[1].stat is not None
        assert result.rows[1].p_value is not None

    def test_sorts_by_edf(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = anova_gam(
            (g2._fit_result, g2._model_matrix),
            (g1._fit_result, g1._model_matrix),
            scale_known=False,
        )
        assert result.rows[0].resid_df > result.rows[1].resid_df

    def test_three_models(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=8)", family=Gaussian()).fit(gaussian_data)
        g3 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = anova_gam(
            (g1._fit_result, g1._model_matrix),
            (g2._fit_result, g2._model_matrix),
            (g3._fit_result, g3._model_matrix),
            scale_known=False,
        )
        assert len(result.rows) == 3
        assert result.rows[0].stat is None
        assert result.rows[1].stat is not None
        assert result.rows[2].stat is not None
        for row in result.rows[1:]:
            assert 0.0 <= row.p_value <= 1.0

    def test_scale_override(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = anova_gam(
            (g1._fit_result, g1._model_matrix),
            (g2._fit_result, g2._model_matrix),
            scale_known=False,
            scale_override=1.0,
        )
        assert result.scale == 1.0

    def test_str_representation(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = anova_gam(
            (g1._fit_result, g1._model_matrix),
            (g2._fit_result, g2._model_matrix),
            scale_known=False,
        )
        table_str = str(result)
        assert "Analysis of Deviance Table" in table_str
        assert "Resid.Df" in table_str
        assert "F" in table_str

    def test_chisq_str(self, poisson_data):
        g1 = GAM("y ~ s(x, k=4)", family=Poisson()).fit(poisson_data)
        g2 = GAM("y ~ s(x, k=15)", family=Poisson()).fit(poisson_data)
        result = anova_gam(
            (g1._fit_result, g1._model_matrix),
            (g2._fit_result, g2._model_matrix),
            scale_known=True,
        )
        table_str = str(result)
        assert "Chisq" in table_str


# ---------------------------------------------------------------------------
# GAM.anova() method tests
# ---------------------------------------------------------------------------


class TestGAMAnova:
    def test_basic_comparison(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = g1.anova(g2)
        assert isinstance(result, AnovaResult)
        assert len(result.rows) == 2

    def test_reverse_order_same_result(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        r1 = g1.anova(g2)
        r2 = g2.anova(g1)
        assert r1.rows[0].resid_dev == r2.rows[0].resid_dev
        assert r1.rows[1].p_value == r2.rows[1].p_value

    def test_unfitted_model_raises(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian())
        with pytest.raises(RuntimeError, match="fitted"):
            g1.anova(g2)

    def test_different_family_raises(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        rng = np.random.default_rng(0)
        pois_data = {"x": gaussian_data["x"], "y": rng.poisson(2, len(gaussian_data["x"]))}
        g2 = GAM("y ~ s(x, k=4)", family=Poisson()).fit(pois_data)
        with pytest.raises(ValueError, match="same family"):
            g1.anova(g2)

    def test_three_models_via_method(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=8)", family=Gaussian()).fit(gaussian_data)
        g3 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = g1.anova(g2, g3)
        assert len(result.rows) == 3

    def test_poisson_anova(self, poisson_data):
        g1 = GAM("y ~ s(x, k=4)", family=Poisson()).fit(poisson_data)
        g2 = GAM("y ~ s(x, k=15)", family=Poisson()).fit(poisson_data)
        result = g1.anova(g2)
        assert result.test == "Chisq"
        assert result.scale == 1.0

    def test_f_test_uses_largest_model_scale(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = g1.anova(g2)
        assert result.test == "F"
        assert result.scale == pytest.approx(g2._fit_result.scale, rel=0.01)

    def test_significant_smooth_detected(self, gaussian_data):
        rng = np.random.default_rng(99)
        noise_data = {
            "x": gaussian_data["x"],
            "y": rng.normal(0, 1, len(gaussian_data["x"])),
        }
        g_null = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(noise_data)
        g_signal = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        assert g_signal._fit_result.deviance < g_null._fit_result.deviance

    def test_deviance_decreases_with_complexity(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = g1.anova(g2)
        assert result.rows[1].deviance > 0

    def test_positive_df_difference(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = g1.anova(g2)
        assert result.rows[1].df > 0

    def test_result_rows_are_anova_model_row(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=4)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=15)", family=Gaussian()).fit(gaussian_data)
        result = g1.anova(g2)
        for row in result.rows:
            assert isinstance(row, AnovaModelRow)

    def test_caller_not_fitted_raises(self):
        g1 = GAM("y ~ s(x)", family=Gaussian())
        g2 = GAM("y ~ s(x)", family=Gaussian())
        with pytest.raises(RuntimeError, match="fitted"):
            g1.anova(g2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAnovaEdgeCases:
    def test_identical_models(self, gaussian_data):
        g1 = GAM("y ~ s(x, k=10)", family=Gaussian()).fit(gaussian_data)
        g2 = GAM("y ~ s(x, k=10)", family=Gaussian()).fit(gaussian_data)
        result = g1.anova(g2)
        assert len(result.rows) == 2
        assert abs(result.rows[1].deviance) < 1e-6

    def test_binomial_anova(self):
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        p = 1.0 / (1.0 + np.exp(-np.sin(x)))
        y = rng.binomial(1, p).astype(float)
        data = {"x": x, "y": y}
        g1 = GAM("y ~ s(x, k=4)", family=Binomial()).fit(data)
        g2 = GAM("y ~ s(x, k=10)", family=Binomial()).fit(data)
        result = g1.anova(g2)
        assert result.test == "Chisq"
        assert result.scale == 1.0
        assert 0.0 <= result.rows[1].p_value <= 1.0
