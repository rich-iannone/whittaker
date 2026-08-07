"""Tests for ti() tensor product interaction smooths."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.poisson import Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.smooths.tensor import TensorInteractionBasis, TensorProductBasis
from whittaker.smooths.tprs import TPRS

# ---------------------------------------------------------------------------
# TensorInteractionBasis unit tests
# ---------------------------------------------------------------------------


class TestTensorInteractionBasis:
    def test_fewer_columns_than_te(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(size=(n, 2))
        k = 6

        te = TensorProductBasis([TPRS(k=k), TPRS(k=k)])
        te.fit(x)

        ti = TensorInteractionBasis([TPRS(k=k), TPRS(k=k)])
        ti.fit(x)

        assert ti.n_basis < te.n_basis

    def test_basis_matrix_shape(self) -> None:
        rng = np.random.default_rng(42)
        n = 80
        x = rng.uniform(size=(n, 2))
        k = 6

        ti = TensorInteractionBasis([TPRS(k=k), TPRS(k=k)])
        ti.fit(x)
        B = ti.basis_matrix(x)

        assert B.shape == (n, ti.n_basis)

    def test_null_space_dimension_is_zero(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        ti = TensorInteractionBasis([TPRS(k=5), TPRS(k=5)])
        ti.fit(x)

        assert ti.null_space_dimension() == 0

    def test_penalty_matrices_count(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        ti = TensorInteractionBasis([TPRS(k=5), TPRS(k=5)])
        ti.fit(x)

        pens = ti.penalty_matrices()
        assert len(pens) == 2

    def test_penalty_matrices_psd(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        ti = TensorInteractionBasis([TPRS(k=6), TPRS(k=6)])
        ti.fit(x)

        for P in ti.penalty_matrices():
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-10)

    def test_penalty_matrix_shape(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        ti = TensorInteractionBasis([TPRS(k=5), TPRS(k=5)])
        ti.fit(x)

        S = ti.penalty_matrix()
        assert S.shape == (ti.n_basis, ti.n_basis)

    def test_requires_two_marginals(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            TensorInteractionBasis([TPRS(k=5)])

    def test_fit_validates_columns(self) -> None:
        ti = TensorInteractionBasis([TPRS(k=5), TPRS(k=5)])
        x = np.random.default_rng(42).uniform(size=(50, 3))
        with pytest.raises(ValueError, match="Expected 2 columns"):
            ti.fit(x)

    def test_three_way_interaction(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(size=(n, 3))
        k = 5

        ti = TensorInteractionBasis([TPRS(k=k), TPRS(k=k), TPRS(k=k)])
        ti.fit(x)
        B = ti.basis_matrix(x)

        assert B.shape[0] == n
        assert B.shape[1] == ti.n_basis
        assert len(ti.penalty_matrices()) == 3

    def test_identifiability_constraints(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(80, 2))

        ti = TensorInteractionBasis([TPRS(k=6), TPRS(k=6)])
        ti.fit(x)

        C = ti.identifiability_constraints()
        assert C is not None
        assert C.shape[0] == 1
        assert C.shape[1] == ti.n_basis

    def test_range_dims_correct(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        m1, m2 = TPRS(k=6), TPRS(k=5)
        ti = TensorInteractionBasis([m1, m2])
        ti.fit(x)

        nsd1 = m1.null_space_dimension()
        nsd2 = m2.null_space_dimension()
        expected_basis = (6 - nsd1) * (5 - nsd2)
        assert ti.n_basis == expected_basis


# ---------------------------------------------------------------------------
# Model matrix integration
# ---------------------------------------------------------------------------


class TestTiModelMatrix:
    def test_ti_builds_model_matrix(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) + np.cos(x2) + 0.1 * rng.standard_normal(n)

        formula = parse("y ~ ti(x1, x2, k=5)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})

        assert mm.X.shape[0] == n
        assert len(mm.smooths) == 1
        assert len(mm.penalties) == 2

    def test_anova_decomposition(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) + np.cos(x2) + x1 * x2 + 0.1 * rng.standard_normal(n)

        formula = parse("y ~ s(x1, k=5) + s(x2, k=5) + ti(x1, x2, k=5)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})

        assert len(mm.smooths) == 3
        ti_info = mm.smooths[2]
        assert ti_info.null_space_dim == 0

    def test_ti_fewer_cols_than_te(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 0.1 * rng.standard_normal(n)

        mm_te = build_model_matrix(parse("y ~ te(x1, x2, k=5)"), {"y": y, "x1": x1, "x2": x2})
        mm_ti = build_model_matrix(parse("y ~ ti(x1, x2, k=5)"), {"y": y, "x1": x1, "x2": x2})

        te_cols = mm_te.smooths[0].col_end - mm_te.smooths[0].col_start
        ti_cols = mm_ti.smooths[0].col_end - mm_ti.smooths[0].col_start
        assert ti_cols < te_cols


# ---------------------------------------------------------------------------
# GAM integration tests
# ---------------------------------------------------------------------------


class TestTiGAM:
    def test_ti_fit_converges(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(2 * x1) * np.cos(2 * x2) + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ ti(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted

    def test_anova_decomposition_fit(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) + np.cos(3 * x2) + 2 * x1 * x2 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=6) + s(x2, k=6) + ti(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert model.is_fitted
        assert len(model.edf) == 3

    def test_ti_predict(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5) + ti(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        pred = model.predict({"x1": np.array([0.5]), "x2": np.array([0.5])}, se=True)
        assert np.isfinite(pred.values[0])
        assert pred.se is not None
        assert pred.se[0] > 0

    def test_ti_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 3 * x1 * x2 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=6) + s(x2, k=6) + ti(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert 0.0 < model.deviance_explained < 1.0

    def test_ti_aic_bic_finite(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1 * x2) + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ ti(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_ti_smooth_tests(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 3 * x1 * x2 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=6) + s(x2, k=6) + ti(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        tests = model.smooth_tests()
        assert len(tests) == 3
        ti_test = tests[2]
        assert ti_test.p_value < 0.05

    def test_ti_residuals(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ ti(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        for rtype in ("response", "pearson", "deviance", "working"):
            r = model.get_residuals(rtype)
            assert np.all(np.isfinite(r))

    def test_ti_with_reml(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5) + ti(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}, method="REML"
        )
        assert model.is_fitted

    def test_ti_summary(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5) + ti(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        text = model.summary()
        assert "ti(x1, x2" in text

    def test_ti_poisson(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        mu = np.exp(0.5 + x1 * x2)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ s(x1, k=5) + s(x2, k=5) + ti(x1, x2, k=5)", family=Poisson()).fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert model.is_fitted
        assert 0.0 < model.deviance_explained < 1.0

    def test_anova_vs_te_comparable_fit(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) + np.cos(3 * x2) + 2 * x1 * x2 + 0.3 * rng.standard_normal(n)

        model_te = GAM("y ~ te(x1, x2, k=6)").fit({"y": y, "x1": x1, "x2": x2})
        model_anova = GAM("y ~ s(x1, k=6) + s(x2, k=6) + ti(x1, x2, k=6)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )

        rmse_te = np.sqrt(np.mean(model_te.get_residuals("response") ** 2))
        rmse_anova = np.sqrt(np.mean(model_anova.get_residuals("response") ** 2))
        assert abs(rmse_te - rmse_anova) / rmse_te < 0.5
