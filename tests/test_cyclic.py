"""Tests for cyclic (periodic) smooth bases: CyclicCRS (cc) and CyclicPSpline (cp)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.poisson import Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.smooths.cyclic import CyclicCRS, CyclicPSpline

# ---------------------------------------------------------------------------
# CyclicCRS unit tests
# ---------------------------------------------------------------------------


class TestCyclicCRS:
    def test_n_basis_is_k_minus_1(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=8).fit(x)
        assert basis.n_basis == 7

    def test_basis_matrix_shape(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=8).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (50, 7)

    def test_penalty_matrix_shape(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=8).fit(x)
        S = basis.penalty_matrix()
        assert S.shape == (7, 7)

    def test_penalty_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=10).fit(x)
        S = basis.penalty_matrix()
        assert_allclose(S, S.T, atol=1e-14)

    def test_penalty_psd(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=10).fit(x)
        eigvals = np.linalg.eigvalsh(basis.penalty_matrix())
        assert np.all(eigvals >= -1e-10)

    def test_null_space_dimension(self) -> None:
        assert CyclicCRS(k=8).null_space_dimension() == 1

    def test_penalty_null_space_is_constant(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=8).fit(x)
        S = basis.penalty_matrix()
        ones = np.ones(basis.n_basis)
        assert_allclose(S @ ones, 0.0, atol=1e-10)

    def test_periodicity_at_endpoints(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 2 * np.pi, 100)
        basis = CyclicCRS(k=10).fit(x)
        x_min, x_max = x.min(), x.max()
        B_lo = basis.basis_matrix(np.array([x_min]))
        B_hi = basis.basis_matrix(np.array([x_max]))
        assert_allclose(B_lo, B_hi, atol=1e-12)

    def test_periodicity_outside_range(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        basis = CyclicCRS(k=8).fit(x)
        period = x.max() - x.min()
        x_test = np.array([0.3])
        B1 = basis.basis_matrix(x_test)
        B2 = basis.basis_matrix(x_test + period)
        B3 = basis.basis_matrix(x_test - period)
        assert_allclose(B1, B2, atol=1e-12)
        assert_allclose(B1, B3, atol=1e-12)

    def test_k_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            CyclicCRS(k=3)

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="smaller than k"):
            CyclicCRS(k=10).fit(np.array([1.0, 2.0, 3.0]))

    def test_identifiability_constraints(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=8).fit(x)
        C = basis.identifiability_constraints()
        assert C is not None
        assert C.shape == (1, 7)

    def test_knots_property(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicCRS(k=8).fit(x)
        knots = basis.knots
        assert len(knots) == 8


# ---------------------------------------------------------------------------
# CyclicPSpline unit tests
# ---------------------------------------------------------------------------


class TestCyclicPSpline:
    def test_n_basis_is_k(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10).fit(x)
        assert basis.n_basis == 10

    def test_basis_matrix_shape(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (50, 10)

    def test_penalty_matrix_shape(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10).fit(x)
        S = basis.penalty_matrix()
        assert S.shape == (10, 10)

    def test_penalty_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10).fit(x)
        S = basis.penalty_matrix()
        assert_allclose(S, S.T, atol=1e-14)

    def test_penalty_psd(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10).fit(x)
        eigvals = np.linalg.eigvalsh(basis.penalty_matrix())
        assert np.all(eigvals >= -1e-10)

    def test_null_space_dimension(self) -> None:
        assert CyclicPSpline(k=10).null_space_dimension() == 1

    def test_penalty_null_space_is_constant(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10).fit(x)
        S = basis.penalty_matrix()
        ones = np.ones(basis.n_basis)
        assert_allclose(S @ ones, 0.0, atol=1e-12)

    def test_periodicity_at_endpoints(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 2 * np.pi, 100)
        basis = CyclicPSpline(k=10).fit(x)
        x_min, x_max = x.min(), x.max()
        B_lo = basis.basis_matrix(np.array([x_min]))
        B_hi = basis.basis_matrix(np.array([x_max]))
        assert_allclose(B_lo, B_hi, atol=1e-10)

    def test_periodicity_outside_range(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        basis = CyclicPSpline(k=10).fit(x)
        period = x.max() - x.min()
        x_test = np.array([0.3])
        B1 = basis.basis_matrix(x_test)
        B2 = basis.basis_matrix(x_test + period)
        B3 = basis.basis_matrix(x_test - period)
        assert_allclose(B1, B2, atol=1e-10)
        assert_allclose(B1, B3, atol=1e-10)

    def test_partition_of_unity(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        basis = CyclicPSpline(k=10).fit(x)
        B = basis.basis_matrix(x)
        assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)

    def test_custom_degree(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10, degree=2).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (50, 10)

    def test_custom_penalty_order(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10, m=1).fit(x)
        S = basis.penalty_matrix()
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-10)
        assert np.sum(eigvals < 1e-10) == 1

    def test_k_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            CyclicPSpline(k=1)

    def test_k_too_small_for_degree_raises(self) -> None:
        with pytest.raises(ValueError, match="k=3 is too small"):
            CyclicPSpline(k=3, degree=3)

    def test_m_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="must be less than k"):
            CyclicPSpline(k=5, m=5)

    def test_identical_values_raises(self) -> None:
        with pytest.raises(ValueError, match="identical"):
            CyclicPSpline(k=5).fit(np.ones(20))

    def test_identifiability_constraints(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=50)
        basis = CyclicPSpline(k=10).fit(x)
        C = basis.identifiability_constraints()
        assert C is not None
        assert C.shape == (1, 10)


# ---------------------------------------------------------------------------
# Model matrix integration
# ---------------------------------------------------------------------------


class TestCyclicModelMatrix:
    def test_cc_builds_model_matrix(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        formula = parse('y ~ s(x, bs="cc", k=8)')
        mm = build_model_matrix(formula, {"y": y, "x": x})
        assert mm.X.shape[0] == n
        assert len(mm.smooths) == 1

    def test_cp_builds_model_matrix(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        formula = parse('y ~ s(x, bs="cp", k=10)')
        mm = build_model_matrix(formula, {"y": y, "x": x})
        assert mm.X.shape[0] == n
        assert len(mm.smooths) == 1

    def test_cc_fewer_cols_than_cr(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(size=n)
        y = 0.1 * rng.standard_normal(n)

        mm_cr = build_model_matrix(parse('y ~ s(x, bs="cr", k=8)'), {"y": y, "x": x})
        mm_cc = build_model_matrix(parse('y ~ s(x, bs="cc", k=8)'), {"y": y, "x": x})

        cr_cols = mm_cr.smooths[0].col_end - mm_cr.smooths[0].col_start
        cc_cols = mm_cc.smooths[0].col_end - mm_cc.smooths[0].col_start
        assert cc_cols < cr_cols


# ---------------------------------------------------------------------------
# GAM integration tests
# ---------------------------------------------------------------------------


class TestCyclicGAM:
    def test_cc_fit_converges(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        assert model.is_fitted

    def test_cp_fit_converges(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cp", k=10)').fit({"y": y, "x": x})
        assert model.is_fitted

    def test_cc_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        assert model.deviance_explained > 0.8

    def test_cp_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cp", k=10)').fit({"y": y, "x": x})
        assert model.deviance_explained > 0.8

    def test_cc_predict_periodic(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        x_min, x_max = x.min(), x.max()
        pred = model.predict({"x": np.array([x_min, x_max])})
        assert_allclose(pred.values[0], pred.values[1], atol=1e-10)

    def test_cp_predict_periodic(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cp", k=10)').fit({"y": y, "x": x})
        x_min, x_max = x.min(), x.max()
        pred = model.predict({"x": np.array([x_min, x_max])})
        assert_allclose(pred.values[0], pred.values[1], atol=1e-10)

    def test_cc_predict_with_se(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        pred = model.predict({"x": np.array([1.0, 2.0, 3.0])}, se=True)
        assert pred.se is not None
        assert np.all(pred.se > 0)

    def test_cc_aic_bic_finite(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_cc_smooth_tests(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        y = 3 * np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        tests = model.smooth_tests()
        assert len(tests) == 1
        assert tests[0].p_value < 0.05

    def test_cc_residuals(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        for rtype in ("response", "pearson", "deviance", "working"):
            r = model.get_residuals(rtype)
            assert np.all(np.isfinite(r))

    def test_cc_with_reml(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x}, method="REML")
        assert model.is_fitted

    def test_cc_summary_output(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        text = model.summary()
        assert "s(x" in text

    def test_cc_poisson(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 2 * np.pi, n)
        mu = np.exp(0.5 + 0.5 * np.sin(x))
        y = rng.poisson(mu).astype(float)

        model = GAM('y ~ s(x, bs="cc", k=8)', family=Poisson()).fit({"y": y, "x": x})
        assert model.is_fitted

    def test_cc_with_other_smooths(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + x2**2 + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x1, bs="cc", k=8) + s(x2, k=6)').fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted
        assert len(model.edf) == 2

    def test_cc_in_tensor_product(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) * x2 + 0.1 * rng.standard_normal(n)

        model = GAM('y ~ te(x1, x2, bs="cc", k=5)').fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted

    def test_cp_with_reml(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cp", k=10)').fit({"y": y, "x": x}, method="REML")
        assert model.is_fitted

    def test_cc_concurvity(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + x2 + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x1, bs="cc", k=8) + s(x2, k=6)').fit({"y": y, "x1": x1, "x2": x2})
        c = model.concurvity(full=True)
        assert c.worst.shape == (2,)
        assert np.all(np.isfinite(c.worst))

    def test_cc_plot(self) -> None:
        pytest.importorskip("altair")

        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cc", k=10)').fit({"y": y, "x": x})
        chart = model.plot()
        assert chart.to_dict() is not None

    def test_cp_plot(self) -> None:
        pytest.importorskip("altair")

        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cp", k=10)').fit({"y": y, "x": x})
        chart = model.plot()
        assert chart.to_dict() is not None
