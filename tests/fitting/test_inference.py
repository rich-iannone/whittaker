"""Tests for whittaker.fitting.inference (approximate p-values)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.beta import Beta
from whittaker.families.binomial import Binomial
from whittaker.families.gamma import Gamma
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.families.poisson import Poisson
from whittaker.fitting import inference as inference_mod
from whittaker.fitting.inference import (
    SmoothTestResult,
    _bayesian_covariance,
    _estimate_concurvity,
    _k_index_1d,
    _k_index_nd,
    _observed_concurvity,
    _smooth_test,
    _unconditional_covariance,
    _worst_concurvity,
    concurvity,
    k_check,
    marginal_effects,
    pairwise_comparisons,
    quantile_residuals,
    smooth_derivatives,
    smooth_tests,
)
from whittaker.fitting.pirls import pirls_fit
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix


class TestBayesianCovariance:
    def test_shape(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
        )
        p = mm.X.shape[1]
        assert V.shape == (p, p)

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
        )
        assert_allclose(V, V.T, atol=1e-12)

    def test_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
        )
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals >= -1e-10)

    def test_with_weights(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-2, 2, n)
        p_true = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, p_true, n).astype(float)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, family=Binomial())

        assert result.weights is not None
        V = _bayesian_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
            W=result.weights,
        )
        assert V.shape == (mm.n_coefs, mm.n_coefs)


class TestSmoothTest:
    def test_significant_smooth_small_pvalue(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
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
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = rng.normal(0, 1, n)

        formula = parse("y ~ s(x, k=10)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
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
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)

        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm)

        V = _bayesian_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
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
        rng = np.random.default_rng(23)
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
        rng = np.random.default_rng(23)
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
        rng = np.random.default_rng(23)
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
        rng = np.random.default_rng(23)
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
        rng = np.random.default_rng(23)
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
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})
        tests = model.smooth_tests()
        assert len(tests) == 1
        assert tests[0].p_value < 1e-10

    def test_summary_contains_pvalues(self) -> None:
        rng = np.random.default_rng(23)
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
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 1, n)
        y = rng.normal(0, 1, n)

        model = GAM("y ~ s(x, k=8)").fit({"y": y, "x": x})
        tests = model.smooth_tests()
        for t in tests:
            assert 0 <= t.p_value <= 1

    def test_reml_pvalues(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)

        model = GAM("y ~ s(x, k=10)").fit({"y": y, "x": x}, method="REML")
        tests = model.smooth_tests()
        assert tests[0].p_value < 1e-10


# ---------------------------------------------------------------------------
# _unconditional_covariance
# ---------------------------------------------------------------------------


class TestUnconditionalCovariance:
    @pytest.fixture()
    def reml_fit(self):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        formula = parse("y ~ s(x, k=8)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, method="REML")
        return mm, result

    def test_no_penalties_returns_bayesian_covariance(self, reml_fit) -> None:
        """When there are no smoothing parameters, V_c should just be V_p (line 161)."""
        mm, result = reml_fit
        V_p = _bayesian_covariance(mm.X, [], [], result.scale)
        V_c = _unconditional_covariance(mm.X, [], [], result.scale, result.coefficients, "REML")
        assert_allclose(V_c, V_p)

    def test_default_y_uses_fitted_values(self, reml_fit) -> None:
        """Omitting *y* should fall back to X @ beta (line 199)."""
        mm, result = reml_fit
        V_c = _unconditional_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
            result.coefficients,
            "REML",
        )
        assert V_c.shape == (mm.n_coefs, mm.n_coefs)
        assert_allclose(V_c, V_c.T)

    def test_ml_method(self, reml_fit) -> None:
        mm, result = reml_fit
        V_c = _unconditional_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
            result.coefficients,
            "ML",
            y=result.fitted_values,
        )
        assert V_c.shape == (mm.n_coefs, mm.n_coefs)

    def test_linalg_error_falls_back_to_bayesian_covariance(self, reml_fit, monkeypatch) -> None:
        """If the eigendecomposition of the Hessian raises, `V_p` should be returned
        instead of propagating the error (lines 246-247)."""
        mm, result = reml_fit
        V_p = _bayesian_covariance(mm.X, mm.penalties, result.smoothing_params, result.scale)

        orig_eigh = np.linalg.eigh
        n_sp = len(result.smoothing_params)

        def fake_eigh(A, *args, **kwargs):
            if A.shape == (n_sp, n_sp):
                raise np.linalg.LinAlgError("forced failure")
            return orig_eigh(A, *args, **kwargs)

        monkeypatch.setattr(inference_mod.np.linalg, "eigh", fake_eigh)

        V_c = _unconditional_covariance(
            mm.X,
            mm.penalties,
            result.smoothing_params,
            result.scale,
            result.coefficients,
            "REML",
        )
        assert_allclose(V_c, V_p)


# ---------------------------------------------------------------------------
# _smooth_test edge cases
# ---------------------------------------------------------------------------


class TestSmoothTestEdgeCases:
    def test_zero_covariance_returns_trivial_result(self) -> None:
        """When V_j has no positive eigenvalues, `_smooth_test` should return the
        trivial (stat=0, ref_df=1, p_value=1) result rather than dividing by zero (line 302)."""
        rng = np.random.default_rng(0)
        beta_j = np.zeros(3)
        V_j = np.zeros((3, 3))
        X_j = rng.standard_normal((20, 3))
        stat, ref_df, pval = _smooth_test(beta_j, V_j, X_j, edf_j=1.0)
        assert stat == 0.0
        assert ref_df == 1.0
        assert pval == 1.0


# ---------------------------------------------------------------------------
# Concurvity diagnostics
# ---------------------------------------------------------------------------


class TestConcurvityHelpers:
    def test_worst_concurvity_empty_comparator(self) -> None:
        """No comparator columns means zero concurvity (line 497)."""
        rng = np.random.default_rng(0)
        X_j = rng.standard_normal((20, 3))
        Q_rest = np.zeros((20, 0))
        assert _worst_concurvity(X_j, Q_rest) == 0.0

    def test_worst_concurvity_degenerate_x(self) -> None:
        """An all-zero `X_j` has no non-degenerate singular values (line 502)."""
        X_zero = np.zeros((20, 3))
        Q_rest = np.random.default_rng(1).standard_normal((20, 2))
        assert _worst_concurvity(X_zero, Q_rest) == 0.0

    def test_observed_concurvity_constant_smooth(self) -> None:
        """A constant fitted smooth has zero total sum of squares (line 513)."""
        f_j = np.full(20, 5.0)
        Q_rest = np.random.default_rng(1).standard_normal((20, 2))
        assert _observed_concurvity(f_j, Q_rest) == 0.0

    def test_estimate_concurvity_constant_smooth(self) -> None:
        """Same degenerate case for the estimate-based measure (line 524)."""
        f_j = np.full(20, 5.0)
        Q_rest = np.random.default_rng(1).standard_normal((20, 2))
        assert _estimate_concurvity(f_j, Q_rest) == 0.0

    def test_concurvity_by_level_labels(self) -> None:
        """Factor `by=` smooths should get a `:level` suffix on their labels (line 565)."""
        rng = np.random.default_rng(23)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        grp = rng.choice(["a", "b"], size=n)
        y = np.sin(x1) + rng.normal(0, 0.2, n)
        formula = parse("y ~ s(x1, by=grp, k=8)")
        data = {"y": y, "x1": x1, "grp": grp}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        cr = concurvity(result, mm)
        assert cr.labels == [
            "s(x1, k=8, by='grp'):a",
            "s(x1, k=8, by='grp'):b",
        ]

    def test_concurvity_pairwise_by_level_labels(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        grp = rng.choice(["a", "b"], size=n)
        y = np.sin(x1) + rng.normal(0, 0.2, n)
        formula = parse("y ~ s(x1, by=grp, k=8)")
        data = {"y": y, "x1": x1, "grp": grp}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        cr = concurvity(result, mm, full=False)
        assert cr.worst.shape == (2, 2)
        assert cr.labels == [
            "s(x1, k=8, by='grp'):a",
            "s(x1, k=8, by='grp'):b",
        ]


# ---------------------------------------------------------------------------
# k_check
# ---------------------------------------------------------------------------


class TestKCheckHelpers:
    def test_k_index_1d_zero_residual_variance(self) -> None:
        """Degenerate (zero-variance) residuals should return 1.0 rather than dividing
        by zero (line 835)."""
        resid = np.zeros(20)
        cov = np.linspace(0, 1, 20)
        assert _k_index_1d(resid, cov) == 1.0

    def test_k_index_nd_zero_residual_variance(self) -> None:
        """Same degenerate case for the multi-dimensional k-index (line 854)."""
        resid = np.zeros(20)
        covs = [np.linspace(0, 1, 20), np.linspace(1, 2, 20)]
        assert _k_index_nd(resid, covs) == 1.0

    def test_k_check_by_level_labels(self) -> None:
        """Factor `by=` smooths should get a `:level` suffix on their labels (line 914)."""
        rng = np.random.default_rng(23)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        grp = rng.choice(["a", "b"], size=n)
        y = np.sin(x1) + rng.normal(0, 0.2, n)
        formula = parse("y ~ s(x1, by=grp, k=8)")
        data = {"y": y, "x1": x1, "grp": grp}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        kc = k_check(result, mm, data, result.residuals, n_sim=20)
        assert [r.term_label for r in kc] == [
            "s(x1, k=8, by='grp'):a",
            "s(x1, k=8, by='grp'):b",
        ]


# ---------------------------------------------------------------------------
# quantile_residuals
# ---------------------------------------------------------------------------


class TestQuantileResidualsFamilies:
    @pytest.fixture()
    def small_x(self):
        return np.linspace(0, 1, 100)

    def _fit(self, x, y, family):
        formula = parse("y ~ s(x, k=6)")
        mm = build_model_matrix(formula, {"y": y, "x": x})
        result = pirls_fit(mm, family=family)
        return mm, result

    def test_poisson_branch(self, small_x) -> None:
        rng = np.random.default_rng(23)
        y = rng.poisson(np.exp(0.5 + 0.3 * small_x)).astype(float)
        mm, result = self._fit(small_x, y, Poisson())
        qr = quantile_residuals(result, mm, Poisson(), seed=0)
        assert np.all(np.isfinite(qr))

    def test_binomial_branch(self, small_x) -> None:
        rng = np.random.default_rng(23)
        y = rng.binomial(1, 1 / (1 + np.exp(-small_x))).astype(float)
        mm, result = self._fit(small_x, y, Binomial())
        qr = quantile_residuals(result, mm, Binomial(), seed=0)
        assert np.all(np.isfinite(qr))

    def test_negative_binomial_branch(self, small_x) -> None:
        rng = np.random.default_rng(23)
        y = rng.poisson(np.exp(0.5 + 0.3 * small_x)).astype(float)
        family = NegativeBinomial(theta=3.0)
        mm, result = self._fit(small_x, y, family)
        qr = quantile_residuals(result, mm, family, seed=0)
        assert np.all(np.isfinite(qr))

    def test_gamma_branch(self, small_x) -> None:
        rng = np.random.default_rng(23)
        y = rng.gamma(2.0, np.exp(0.3 * small_x))
        mm, result = self._fit(small_x, y, Gamma())
        qr = quantile_residuals(result, mm, Gamma(), seed=0)
        assert np.all(np.isfinite(qr))

    def test_beta_branch(self, small_x) -> None:
        rng = np.random.default_rng(23)
        y = np.clip(1 / (1 + np.exp(-small_x)) + rng.normal(0, 0.05, 100), 0.01, 0.99)
        mm, result = self._fit(small_x, y, Beta())
        qr = quantile_residuals(result, mm, Beta(), seed=0)
        assert np.all(np.isfinite(qr))

    def test_unrecognized_family_falls_back_to_generic_normal(self, small_x) -> None:
        """Families without a dedicated branch should use the generic residual formula
        based on `family.variance` (line 1047)."""

        class OtherFamily(Poisson):
            pass

        OtherFamily.__name__ = "SomeOtherFamily"

        rng = np.random.default_rng(23)
        y = rng.poisson(np.exp(0.5 + 0.3 * small_x)).astype(float)
        mm, result = self._fit(small_x, y, Poisson())
        qr = quantile_residuals(result, mm, OtherFamily(), seed=0)
        assert np.all(np.isfinite(qr))


# ---------------------------------------------------------------------------
# smooth_derivatives / marginal_effects / pairwise_comparisons
# ---------------------------------------------------------------------------


class TestDerivativeInferenceEdgeCases:
    @pytest.fixture()
    def by_factor_fit(self):
        """A model with a categorical `by=` smooth, encoded as an object array of
        numeric labels so that `numpy.mean` still works on the raw column (needed
        because these functions build prediction grids by averaging every
        non-focal column)."""
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 2 * np.pi, n)
        grp = np.array(rng.integers(0, 2, n), dtype=object)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        formula = parse("y ~ s(x, by=grp, k=8)")
        data = {"y": y, "x": x, "grp": grp}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        return mm, result, data

    @pytest.fixture()
    def prior_weighted_fit(self):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        formula = parse("y ~ s(x, k=8)")
        data = {"y": y, "x": x}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm, prior_weights=np.ones(n))
        return mm, result, data

    @pytest.fixture()
    def linear_and_smooth_fit(self):
        rng = np.random.default_rng(23)
        n = 150
        x = np.linspace(0, 2 * np.pi, n)
        x2 = rng.normal(size=n)
        y = np.sin(x) + rng.normal(0, 0.3, n)
        formula = parse("y ~ s(x, k=8) + x2")
        data = {"y": y, "x": x, "x2": x2}
        mm = build_model_matrix(formula, data)
        result = pirls_fit(mm)
        return mm, result, data

    def test_smooth_derivatives_by_level_label(self, by_factor_fit) -> None:
        mm, result, data = by_factor_fit
        results = smooth_derivatives(result, mm, "x", data)
        labels = {r.term for r in results}
        assert labels == {"s(x, k=8, by='grp'):0", "s(x, k=8, by='grp'):1"}

    def test_smooth_derivatives_computes_v_beta_from_prior_weights(
        self, prior_weighted_fit
    ) -> None:
        mm, result, data = prior_weighted_fit
        assert result.weights is None
        assert result.prior_weights is not None
        results = smooth_derivatives(result, mm, "x", data)
        assert np.all(np.isfinite(results[0].derivative))

    def test_smooth_derivatives_missing_variable_raises(self, prior_weighted_fit) -> None:
        mm, result, data = prior_weighted_fit
        with pytest.raises(ValueError, match="not found"):
            smooth_derivatives(result, mm, "nonexistent", data)

    def test_smooth_derivatives_no_smooth_for_variable_raises(self, linear_and_smooth_fit) -> None:
        mm, result, data = linear_and_smooth_fit
        with pytest.raises(ValueError, match="No smooth terms"):
            smooth_derivatives(result, mm, "x2", data)

    def test_marginal_effects_by_level_label(self, by_factor_fit) -> None:
        mm, result, data = by_factor_fit
        results = marginal_effects(result, mm, "x", data)
        labels = {r.term for r in results}
        assert labels == {"s(x, k=8, by='grp'):0", "s(x, k=8, by='grp'):1"}

    def test_marginal_effects_computes_v_beta_from_prior_weights(self, prior_weighted_fit) -> None:
        mm, result, data = prior_weighted_fit
        results = marginal_effects(result, mm, "x", data)
        assert np.all(np.isfinite(results[0].effect))

    def test_pairwise_comparisons_by_level_label(self, by_factor_fit) -> None:
        mm, result, data = by_factor_fit
        results = pairwise_comparisons(result, mm, "x", data, pairs=[({}, {})])
        labels = {r.term for r in results}
        assert labels == {"s(x, k=8, by='grp'):0", "s(x, k=8, by='grp'):1"}

    def test_pairwise_comparisons_missing_variable_raises(self, prior_weighted_fit) -> None:
        mm, result, data = prior_weighted_fit
        with pytest.raises(ValueError, match="not found"):
            pairwise_comparisons(result, mm, "nonexistent", data, pairs=[({}, {})])

    def test_pairwise_comparisons_computes_v_beta_from_prior_weights(
        self, prior_weighted_fit
    ) -> None:
        mm, result, data = prior_weighted_fit
        results = pairwise_comparisons(result, mm, "x", data, pairs=[({}, {})])
        assert np.all(np.isfinite(results[0].difference))
