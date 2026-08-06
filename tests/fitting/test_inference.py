"""Tests for whittaker.fitting.inference (approximate p-values)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.binomial import Binomial
from whittaker.families.poisson import Poisson
from whittaker.fitting.inference import (
    SmoothTestResult,
    _bayesian_covariance,
    _smooth_test,
    smooth_tests,
)
from whittaker.fitting.pirls import pirls_fit
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix


class TestBayesianCovariance:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X, mm.penalties, result.smoothing_params, result.scale,
        )
        p = mm.X.shape[1]
        assert V.shape == (p, p)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X, mm.penalties, result.smoothing_params, result.scale,
        )
        assert_allclose(V, V.T, atol=1e-12)

    def test_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X, mm.penalties, result.smoothing_params, result.scale,
        )
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals >= -1e-10)

    def test_with_weights(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(-2, 2, n)
        p_true = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, family=Binomial())

        assert result.weights is not None
        V = _bayesian_covariance(
            mm.X, mm.penalties, result.smoothing_params, result.scale,
            W=result.weights,
        )
        assert V.shape == (mm.n_coefs, mm.n_coefs)


class TestSmoothTest:
    def test_significant_smooth_small_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X, mm.penalties, result.smoothing_params, result.scale,
        )
        info = mm.smooths[0]
        cs, ce = info.col_start, info.col_end

        stat, ref_df, pval = _smooth_test(
            result.coefficients[cs:ce],
            V[cs:ce, cs:ce],
            mm.X[:, cs:ce],
            result.edf[0],
        )
        assert stat > 0
        assert ref_df >= 1
        assert pval < 1e-10

    def test_noise_smooth_large_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = rng.normal(0, 1, n)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X, mm.penalties, result.smoothing_params, result.scale,
        )
        info = mm.smooths[0]
        cs, ce = info.col_start, info.col_end

        stat, ref_df, pval = _smooth_test(
            result.coefficients[cs:ce],
            V[cs:ce, cs:ce],
            mm.X[:, cs:ce],
            result.edf[0],
        )
        assert pval > 0.05

    def test_ref_df_positive(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X, mm.penalties, result.smoothing_params, result.scale,
        )
        info = mm.smooths[0]
        cs, ce = info.col_start, info.col_end

        stat, ref_df, pval = _smooth_test(
            result.coefficients[cs:ce],
            V[cs:ce, cs:ce],
            mm.X[:, cs:ce],
            result.edf[0],
        )
        assert ref_df >= 1


class TestSmoothTests:
    def test_returns_list_of_results(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        tests = smooth_tests(result, mm)
        assert len(tests) == 1
        assert isinstance(tests[0], SmoothTestResult)

    def test_multi_smooth(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + rng.normal(0, 0.3, n)

        formula = parse("y ~ s(x1, k=10) + s(x2, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})
        result = pirls_fit(mm)

        tests = smooth_tests(result, mm)
        assert len(tests) == 2
        assert tests[0].p_value < 0.01
        assert tests[1].p_value > 0.05

    def test_binomial(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        x = np.linspace(-3, 3, n)
        p_true = 1.0 / (1.0 + np.exp(-np.sin(x)))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, family=Binomial())

        tests = smooth_tests(result, mm)
        assert len(tests) == 1
        assert tests[0].p_value < 0.001

    def test_poisson(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        mu = np.exp(0.5 + 0.5 * np.sin(x))
        y = rng.poisson(mu).astype(float)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, family=Poisson())

        tests = smooth_tests(result, mm)
        assert len(tests) == 1
        assert tests[0].p_value < 0.001

    def test_tensor_product(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + np.cos(x2) + rng.normal(0, 0.3, n)

        formula = parse("y ~ te(x1, x2, k=5)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})
        result = pirls_fit(mm)

        tests = smooth_tests(result, mm)
        assert len(tests) == 1
        assert tests[0].p_value < 1e-10


class TestGAMSmoothTests:
    def test_gam_smooth_tests_method(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        tests = model.smooth_tests()
        assert len(tests) == 1
        assert tests[0].p_value < 1e-10

    def test_summary_contains_pvalues(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        text = model.summary()
        assert "p-value" in text
        assert "Chi.sq" in text

    def test_unfitted_raises(self) -> None:
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.smooth_tests()

    def test_pvalue_in_zero_one(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = np.linspace(0, 1, n)
        y = rng.normal(0, 1, n)

        model = GAM("y ~ s(x, k=8)").fit({"y": y, "x": x})
        tests = model.smooth_tests()
        for t in tests:
            assert 0 <= t.p_value <= 1

    def test_reml_pvalues(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10)").fit(
            {"y": y, "x": x}, method="REML"
        )
        tests = model.smooth_tests()
        assert tests[0].p_value < 1e-10
