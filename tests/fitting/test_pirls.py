"""Tests for whittaker.fitting.pirls (P-IRLS fitting engine)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.families.poisson import Poisson
from whittaker.fitting import pirls as pirls_mod
from whittaker.fitting.pirls import (
    FitResult,
    _gcv_score,
    _penalized_solve,
    _reml_objective,
    pirls_fit,
)
from whittaker.formula.parser import parse
from whittaker.model_matrix import build_model_matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(23)


def _sin_data(n: int = 200) -> dict[str, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, n)
    return {
        "y": np.sin(x) + RNG.normal(0, 0.2, n),
        "x": x,
    }


def _linear_data(n: int = 200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(99)
    x = np.linspace(0, 1, n)
    return {
        "y": 3.0 * x + 1.0 + rng.normal(0, 0.1, n),
        "x": x,
    }


def _multi_data(n: int = 200) -> dict[str, np.ndarray]:
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = np.linspace(0, np.pi, n)
    return {
        "y": np.sin(x1) + 0.5 * np.cos(x2) + RNG.normal(0, 0.2, n),
        "x1": x1,
        "x2": x2,
    }


# ---------------------------------------------------------------------------
# _gcv_score
# ---------------------------------------------------------------------------


class TestGCVScore:
    def test_gcv_positive(self) -> None:
        score = _gcv_score(deviance=10.0, n=100, hat_trace=5.0)
        assert score > 0

    def test_gcv_increases_with_deviance(self) -> None:
        s1 = _gcv_score(deviance=10.0, n=100, hat_trace=5.0)
        s2 = _gcv_score(deviance=20.0, n=100, hat_trace=5.0)
        assert s2 > s1

    def test_gcv_inf_when_overfit(self) -> None:
        score = _gcv_score(deviance=1.0, n=10, hat_trace=10.0)
        assert score == np.inf

    def test_gcv_formula(self) -> None:
        dev, n, tr = 50.0, 200, 10.0
        expected = n * dev / (n - tr) ** 2
        assert_allclose(_gcv_score(dev, n, tr), expected)


# ---------------------------------------------------------------------------
# _penalized_solve
# ---------------------------------------------------------------------------


class TestPenalizedSolve:
    def test_unpenalized_equals_ols(self) -> None:
        n, p = 50, 3
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)

        beta, _ = _penalized_solve(X, y, penalties=[], sp=[])
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        assert_allclose(beta, beta_ols, atol=1e-10)

    def test_penalty_shrinks_coefficients(self) -> None:
        n, p = 50, 5
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        S = np.eye(p)

        beta_unpen, _ = _penalized_solve(X, y, [], [])
        beta_pen, _ = _penalized_solve(X, y, [S], [10.0])
        assert np.linalg.norm(beta_pen) < np.linalg.norm(beta_unpen)

    def test_hat_trace_bounded(self) -> None:
        n, p = 50, 5
        rng = np.random.default_rng(2)
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        S = np.eye(p)

        _, hat_arr = _penalized_solve(X, y, [S], [1.0])
        hat_trace = float(hat_arr[0])
        assert 0 < hat_trace < p


# ---------------------------------------------------------------------------
# pirls_fit — basic Gaussian
# ---------------------------------------------------------------------------


class TestPirlsFitGaussian:
    def test_returns_fit_result(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert isinstance(result, FitResult)

    def test_converged(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.converged

    def test_one_iteration_for_gaussian(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.n_iter == 1

    def test_coefficients_shape(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.coefficients.shape == (mm.n_coefs,)

    def test_fitted_values_shape(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.fitted_values.shape == (mm.n_obs,)

    def test_residuals_sum_approximately_zero(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert abs(np.mean(result.residuals)) < 0.05

    def test_fitted_plus_residuals_equals_y(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert_allclose(result.fitted_values + result.residuals, data["y"], atol=1e-10)


# ---------------------------------------------------------------------------
# pirls_fit — smoothing parameter selection
# ---------------------------------------------------------------------------


class TestSmoothingSelection:
    def test_auto_selects_positive_sp(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert all(sp > 0 for sp in result.smoothing_params)

    def test_fixed_sp_used(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, smoothing_params=[5.0])
        assert result.smoothing_params == [5.0]

    def test_wrong_number_of_sp_raises(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        with pytest.raises(ValueError, match="smoothing parameters"):
            pirls_fit(mm, smoothing_params=[1.0, 2.0])

    def test_gcv_score_positive(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.gcv_score > 0

    def test_per_term_sp_differ_for_different_complexity(self) -> None:
        rng = np.random.default_rng(58)
        n = 600
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(3 * x1) + 0.5 * x2 + rng.normal(0, 0.15, n)

        formula = parse("y ~ s(x1, k=15) + s(x2, k=15)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})
        result = pirls_fit(mm)

        assert len(result.smoothing_params) == 2
        assert result.smoothing_params[0] != result.smoothing_params[1]
        edf_wiggly, edf_linear = result.edf
        assert edf_wiggly > edf_linear

    def test_per_term_edf_wiggly_vs_linear(self) -> None:
        rng = np.random.default_rng(99)
        n = 400
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(3 * x1) + 0.5 * x2 + rng.normal(0, 0.15, n)

        formula = parse("y ~ s(x1, k=15) + s(x2, k=15)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})
        result = pirls_fit(mm)

        edf_wiggly, edf_linear = result.edf
        assert edf_wiggly > edf_linear * 3

    def test_gcv_auto_better_than_arbitrary_sp(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=15)")
        mm = build_model_matrix(formula, data)

        result_auto = pirls_fit(mm)
        result_oversmooth = pirls_fit(mm, smoothing_params=[1000.0])
        result_undersmooth = pirls_fit(mm, smoothing_params=[1e-8])

        assert result_auto.gcv_score <= result_oversmooth.gcv_score
        assert result_auto.gcv_score <= result_undersmooth.gcv_score

    def test_no_penalties_gives_empty_sp(self) -> None:
        formula = parse("y ~ x")
        data = _sin_data()
        data["x"] = data["x"]
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.smoothing_params == []


# ---------------------------------------------------------------------------
# pirls_fit — EDF
# ---------------------------------------------------------------------------


class TestEDF:
    def test_edf_per_smooth(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert len(result.edf) == 1
        assert 1.0 < result.edf[0] < 10.0

    def test_edf_multiple_smooths(self) -> None:
        data = _multi_data()
        formula = parse("y ~ s(x1, k=10) + s(x2, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert len(result.edf) == 2
        assert all(0.0 < e < 10.0 for e in result.edf)

    def test_edf_total_reasonable(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert 2.0 < result.edf_total < 10.0

    def test_linear_data_low_edf(self) -> None:
        data = _linear_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.edf[0] < 4.0


# ---------------------------------------------------------------------------
# pirls_fit — scale estimation
# ---------------------------------------------------------------------------


class TestScaleEstimation:
    def test_scale_positive(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert result.scale > 0

    def test_scale_close_to_true_variance(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=20)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        assert abs(result.scale - 0.04) < 0.03


# ---------------------------------------------------------------------------
# pirls_fit — numerical accuracy
# ---------------------------------------------------------------------------


class TestNumericalAccuracy:
    def test_sin_recovery(self) -> None:
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y_true = np.sin(x)
        y = y_true + RNG.normal(0, 0.15, n)

        formula = parse("y ~ s(x, k=20)")
        data = {"y": y, "x": x}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)

        rmse = np.sqrt(np.mean((result.fitted_values - y_true) ** 2))
        assert rmse < 0.1

    def test_two_smooth_recovery(self) -> None:
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(0, np.pi, n)
        y_true = np.sin(x1) + 0.5 * np.cos(x2)
        y = y_true + RNG.normal(0, 0.2, n)

        formula = parse("y ~ s(x1, k=15) + s(x2, k=15)")
        data = {"y": y, "x1": x1, "x2": x2}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)

        rmse = np.sqrt(np.mean((result.fitted_values - y_true) ** 2))
        assert rmse < 0.25

    def test_higher_k_at_least_as_good(self) -> None:
        data = _sin_data(300)
        rmses = []
        for k in [5, 15]:
            formula = parse(f"y ~ s(x, k={k})")
            mm = build_model_matrix(formula, data)
            result = pirls_fit(mm)

            x = data["x"]
            y_true = np.sin(x)
            rmse = np.sqrt(np.mean((result.fitted_values - y_true) ** 2))
            rmses.append(rmse)

        assert rmses[1] <= rmses[0] + 0.01

    def test_different_basis_types(self) -> None:
        data = _sin_data(200)
        x = data["x"]
        y_true = np.sin(x)

        for bs in ["tp", "cr", "ps"]:
            formula = parse(f"y ~ s(x, bs='{bs}', k=15)")
            mm = build_model_matrix(formula, data)
            result = pirls_fit(mm)

            rmse = np.sqrt(np.mean((result.fitted_values - y_true) ** 2))
            assert rmse < 0.15, f"bs={bs!r} RMSE={rmse:.4f} too high"


# ---------------------------------------------------------------------------
# pirls_fit — Binomial (logistic GAM)
# ---------------------------------------------------------------------------


class TestBinomialFit:
    def test_binomial_converges(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(-3, 3, n)
        eta_true = np.sin(x)
        p_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Binomial())
        assert result.converged
        assert result.n_iter > 1

    def test_binomial_scale_is_one(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        y = rng.binomial(1, 0.5 * np.ones(n), n).astype(float)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Binomial())
        assert result.scale == 1.0

    def test_binomial_fitted_values_in_01(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(-3, 3, n)
        eta_true = 1.5 * np.sin(x)
        p_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Binomial())
        assert np.all(result.fitted_values > 0)
        assert np.all(result.fitted_values < 1)

    def test_binomial_recovers_smooth_effect(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(-3, 3, n)
        eta_true = np.sin(x)
        p_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=15)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Binomial())

        rmse = np.sqrt(np.mean((result.fitted_values - p_true) ** 2))
        assert rmse < 0.1

    def test_binomial_edf_reasonable(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(-3, 3, n)
        eta_true = np.sin(x)
        p_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Binomial())
        assert 1.0 < result.edf[0] < 10.0

    def test_binomial_fixed_sp(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        y = rng.binomial(1, 0.5 * np.ones(n), n).astype(float)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Binomial(), smoothing_params=[10.0])
        assert result.smoothing_params == [10.0]
        assert result.converged


# ---------------------------------------------------------------------------
# pirls_fit — Poisson (count GAM)
# ---------------------------------------------------------------------------


class TestPoissonFit:
    def test_poisson_converges(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.8 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson())
        assert result.converged
        assert result.n_iter > 1

    def test_poisson_scale_is_one(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        y = rng.poisson(3.0, n).astype(float)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson())
        assert result.scale == 1.0

    def test_poisson_fitted_values_positive(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.8 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson())
        assert np.all(result.fitted_values > 0)

    def test_poisson_recovers_smooth_effect(self) -> None:
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.8 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        formula = parse("y ~ s(x, k=15)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson())

        rmse = np.sqrt(np.mean((result.fitted_values - mu_true) ** 2))
        assert rmse < 0.5

    def test_poisson_edf_reasonable(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.8 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson())
        assert 1.0 < result.edf[0] < 10.0

    def test_poisson_fixed_sp(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        y = rng.poisson(3.0, n).astype(float)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson(), smoothing_params=[10.0])
        assert result.smoothing_params == [10.0]
        assert result.converged

    def test_poisson_handles_zero_counts(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(-0.5 + 0.5 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)
        assert np.sum(y == 0) > 0

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson())
        assert result.converged
        assert np.all(np.isfinite(result.fitted_values))


# ---------------------------------------------------------------------------
# REML smoothing parameter selection
# ---------------------------------------------------------------------------


class TestREML:
    def test_reml_selects_positive_sp(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="REML")
        assert all(sp > 0 for sp in result.smoothing_params)

    def test_reml_converges_gaussian(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="REML")
        assert result.converged

    def test_reml_sin_recovery(self) -> None:
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y_true = np.sin(x)
        y = y_true + RNG.normal(0, 0.15, n)

        formula = parse("y ~ s(x, k=20)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, method="REML")

        rmse = np.sqrt(np.mean((result.fitted_values - y_true) ** 2))
        assert rmse < 0.1

    def test_reml_edf_reasonable(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="REML")
        assert 1.0 < result.edf[0] < 10.0

    def test_reml_multiple_smooths(self) -> None:
        data = _multi_data()
        formula = parse("y ~ s(x1, k=10) + s(x2, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="REML")
        assert len(result.smoothing_params) == 2
        assert all(sp > 0 for sp in result.smoothing_params)

    def test_reml_linear_data_low_edf(self) -> None:
        data = _linear_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="REML")
        assert result.edf[0] < 4.0

    def test_reml_better_than_arbitrary_sp(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=15)")
        mm = build_model_matrix(formula, data)

        result_reml = pirls_fit(mm, method="REML")
        result_oversmooth = pirls_fit(mm, smoothing_params=[1000.0])
        result_undersmooth = pirls_fit(mm, smoothing_params=[1e-8])

        y_true = np.sin(data["x"])
        rmse_reml = np.sqrt(np.mean((result_reml.fitted_values - y_true) ** 2))
        rmse_over = np.sqrt(np.mean((result_oversmooth.fitted_values - y_true) ** 2))
        rmse_under = np.sqrt(np.mean((result_undersmooth.fitted_values - y_true) ** 2))

        assert rmse_reml < rmse_over
        assert rmse_reml < rmse_under

    def test_reml_binomial(self) -> None:
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(-3, 3, n)
        p_true = 1.0 / (1.0 + np.exp(-np.sin(x)))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Binomial(), method="REML")
        assert result.converged
        assert result.scale == 1.0
        assert np.all(result.fitted_values > 0)
        assert np.all(result.fitted_values < 1)

    def test_reml_poisson(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.5 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, Poisson(), method="REML")
        assert result.converged
        assert result.scale == 1.0
        assert np.all(result.fitted_values > 0)

    def test_reml_invalid_method_raises(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        with pytest.raises(ValueError, match="method"):
            pirls_fit(mm, method="invalid")

    def test_reml_no_penalties_gives_empty_sp(self) -> None:
        formula = parse("y ~ x")
        data = _sin_data()
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="REML")
        assert result.smoothing_params == []

    def test_reml_gradient_finite_difference(self) -> None:
        """Verify analytic gradient against finite differences."""
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})

        from whittaker.fitting.pirls import _penalty_ranks

        pen_ranks = _penalty_ranks(mm.penalties)
        n_unpen = 1 + sum(s.null_space_dim for s in mm.smooths)

        log_sp = np.array([1.0])

        val, grad = _reml_objective(
            log_sp,
            mm.X,
            y,
            mm.penalties,
            pen_ranks,
            n_unpen,
            scale_known=False,
        )

        eps = 1e-5
        log_sp_plus = log_sp + eps
        val_plus, _ = _reml_objective(
            log_sp_plus,
            mm.X,
            y,
            mm.penalties,
            pen_ranks,
            n_unpen,
            scale_known=False,
        )
        fd_grad = (val_plus - val) / eps

        assert_allclose(grad[0], fd_grad, rtol=1e-3)


# ---------------------------------------------------------------------------
# pirls_fit — ML smoothing parameter selection
# ---------------------------------------------------------------------------


class TestML:
    def test_ml_selects_positive_sp(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="ML")
        assert all(sp > 0 for sp in result.smoothing_params)

    def test_ml_converges_gaussian(self) -> None:
        data = _sin_data()
        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="ML")
        assert result.converged

    def test_ml_no_penalties_gives_empty_sp(self) -> None:
        """Exercises `_select_smoothing_params_ml`'s early return when n_sp == 0."""
        formula = parse("y ~ x")
        data = _sin_data()
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, method="ML")
        assert result.smoothing_params == []


# ---------------------------------------------------------------------------
# _reml_objective — numerical edge cases
# ---------------------------------------------------------------------------


class TestReMLObjectiveEdgeCases:
    def test_singular_a_matrix_returns_large_penalty(self) -> None:
        """A negative-definite penalty can make `A` non-positive-definite; the Cholesky
        factorization then raises `LinAlgError`, which should be caught and turned into a
        large finite objective value rather than propagating."""
        X = np.zeros((3, 3))
        y = np.array([1.0, 2.0, 3.0])
        penalties = [-np.eye(3)]
        log_sp = np.array([0.0])

        val, grad = _reml_objective(log_sp, X, y, penalties, [3], 0, scale_known=True, scale=1.0)
        assert val == 1e20
        assert_allclose(grad, np.zeros_like(log_sp))

    def test_ml_xtx_cholesky_failure_falls_back_to_zero(self, monkeypatch) -> None:
        """When the extra Cholesky factorization used for the ML log-determinant term fails,
        the code should catch `LinAlgError` and treat `log_det_XtX` as 0 instead of raising."""
        orig_cho_factor = pirls_mod.cho_factor
        calls = {"n": 0}

        def fake_cho_factor(A, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise np.linalg.LinAlgError("forced failure")
            return orig_cho_factor(A, *args, **kwargs)

        monkeypatch.setattr(pirls_mod, "cho_factor", fake_cho_factor)

        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 3))
        y = rng.standard_normal(10)
        penalties = [np.eye(3)]
        log_sp = np.array([0.0])

        val, grad = _reml_objective(
            log_sp, X, y, penalties, [3], 0, scale_known=True, scale=1.0, ml=True
        )
        assert calls["n"] == 2
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# pirls_fit — Negative Binomial outer-loop non-convergence
# ---------------------------------------------------------------------------


class TestNegativeBinomialOuterLoop:
    def test_outer_loop_exhausts_without_theta_convergence(self, monkeypatch) -> None:
        """If theta never stabilizes, the outer `for _outer in range(max_outer)` loop should
        run to completion and fall through to its `else` clause instead of breaking early."""

        def never_converging_theta(y, mu, theta_old):
            return theta_old * 2.0

        monkeypatch.setattr(pirls_mod, "_estimate_nb_theta", never_converging_theta)

        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        mu_true = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu_true).astype(float)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_mod.pirls_fit(mm, NegativeBinomial(theta=1.0))

        assert np.all(np.isfinite(result.fitted_values))
