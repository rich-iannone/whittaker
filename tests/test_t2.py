"""Tests for t2() tensor product smooths with full penalty decomposition."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.poisson import Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.smooths.tensor import TensorProductBasis, TensorProductBasisT2
from whittaker.smooths.tprs import TPRS


# ---------------------------------------------------------------------------
# TensorProductBasisT2 unit tests
# ---------------------------------------------------------------------------


class TestTensorProductBasisT2:
    def test_is_subclass_of_te(self) -> None:
        assert issubclass(TensorProductBasisT2, TensorProductBasis)

    def test_same_basis_as_te(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(80, 2))

        te = TensorProductBasis([TPRS(k=5), TPRS(k=5)])
        te.fit(x)

        t2 = TensorProductBasisT2([TPRS(k=5), TPRS(k=5)])
        t2.fit(x)

        assert te.n_basis == t2.n_basis
        assert_allclose(te.basis_matrix(x), t2.basis_matrix(x))

    def test_more_penalties_than_te(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        te = TensorProductBasis([TPRS(k=5), TPRS(k=5)])
        te.fit(x)

        t2 = TensorProductBasisT2([TPRS(k=5), TPRS(k=5)])
        t2.fit(x)

        assert len(t2.penalty_matrices()) == 3
        assert len(te.penalty_matrices()) == 2

    def test_two_marginals_3_penalties(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))
        t2 = TensorProductBasisT2([TPRS(k=5), TPRS(k=5)])
        t2.fit(x)
        assert len(t2.penalty_matrices()) == 3  # 2^2 - 1

    def test_three_marginals_7_penalties(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(80, 3))
        t2 = TensorProductBasisT2([TPRS(k=4), TPRS(k=4), TPRS(k=4)])
        t2.fit(x)
        assert len(t2.penalty_matrices()) == 7  # 2^3 - 1

    def test_penalty_matrices_psd(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))
        t2 = TensorProductBasisT2([TPRS(k=6), TPRS(k=6)])
        t2.fit(x)

        for P in t2.penalty_matrices():
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-10)

    def test_penalty_matrices_correct_shape(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))
        t2 = TensorProductBasisT2([TPRS(k=5), TPRS(k=6)])
        t2.fit(x)

        expected_size = 5 * 6
        for P in t2.penalty_matrices():
            assert P.shape == (expected_size, expected_size)

    def test_penalty_matrices_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))
        t2 = TensorProductBasisT2([TPRS(k=5), TPRS(k=5)])
        t2.fit(x)

        for P in t2.penalty_matrices():
            assert_allclose(P, P.T, atol=1e-14)

    def test_first_two_penalties_match_te(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        te = TensorProductBasis([TPRS(k=5), TPRS(k=5)])
        te.fit(x)

        t2 = TensorProductBasisT2([TPRS(k=5), TPRS(k=5)])
        t2.fit(x)

        te_pens = te.penalty_matrices()
        t2_pens = t2.penalty_matrices()

        assert_allclose(t2_pens[0], te_pens[0], atol=1e-14)
        assert_allclose(t2_pens[1], te_pens[1], atol=1e-14)

    def test_third_penalty_is_kron_product(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        m1, m2 = TPRS(k=5), TPRS(k=6)
        t2 = TensorProductBasisT2([m1, m2])
        t2.fit(x)

        S1 = m1.penalty_matrix()
        S2 = m2.penalty_matrix()
        expected = np.kron(S1, S2)

        t2_pens = t2.penalty_matrices()
        assert_allclose(t2_pens[2], expected, atol=1e-14)

    def test_null_space_dimension_same_as_te(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(size=(50, 2))

        te = TensorProductBasis([TPRS(k=5), TPRS(k=5)])
        te.fit(x)

        t2 = TensorProductBasisT2([TPRS(k=5), TPRS(k=5)])
        t2.fit(x)

        assert t2.null_space_dimension() == te.null_space_dimension()

    def test_requires_two_marginals(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            TensorProductBasisT2([TPRS(k=5)])


# ---------------------------------------------------------------------------
# Model matrix integration
# ---------------------------------------------------------------------------


class TestT2ModelMatrix:
    def test_t2_builds_model_matrix(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) * np.cos(x2) + 0.1 * rng.standard_normal(n)

        formula = parse("y ~ t2(x1, x2, k=5)")
        mm = build_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})

        assert mm.X.shape[0] == n
        assert len(mm.smooths) == 1
        assert len(mm.penalties) == 3

    def test_t2_same_basis_cols_as_te(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 0.1 * rng.standard_normal(n)

        mm_te = build_model_matrix(
            parse("y ~ te(x1, x2, k=5)"), {"y": y, "x1": x1, "x2": x2}
        )
        mm_t2 = build_model_matrix(
            parse("y ~ t2(x1, x2, k=5)"), {"y": y, "x1": x1, "x2": x2}
        )

        te_cols = mm_te.smooths[0].col_end - mm_te.smooths[0].col_start
        t2_cols = mm_t2.smooths[0].col_end - mm_t2.smooths[0].col_start
        assert t2_cols == te_cols

    def test_t2_more_penalties_than_te(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 0.1 * rng.standard_normal(n)

        mm_te = build_model_matrix(
            parse("y ~ te(x1, x2, k=5)"), {"y": y, "x1": x1, "x2": x2}
        )
        mm_t2 = build_model_matrix(
            parse("y ~ t2(x1, x2, k=5)"), {"y": y, "x1": x1, "x2": x2}
        )

        assert len(mm_t2.penalties) > len(mm_te.penalties)

    def test_t2_penalty_indices(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 0.1 * rng.standard_normal(n)

        mm = build_model_matrix(
            parse("y ~ t2(x1, x2, k=5)"), {"y": y, "x1": x1, "x2": x2}
        )
        assert len(mm.smooths[0].penalty_indices) == 3


# ---------------------------------------------------------------------------
# GAM integration tests
# ---------------------------------------------------------------------------


class TestT2GAM:
    def test_t2_fit_converges(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) * np.cos(3 * x2) + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted

    def test_t2_has_3_smoothing_params(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) * np.cos(x2) + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        assert len(model.smoothing_params) == 3

    def test_t2_predict(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        pred = model.predict({"x1": np.array([0.5]), "x2": np.array([0.5])}, se=True)
        assert np.isfinite(pred.values[0])
        assert pred.se is not None
        assert pred.se[0] > 0

    def test_t2_deviance_explained(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 3 * np.sin(x1) * np.cos(x2) + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=6)").fit({"y": y, "x1": x1, "x2": x2})
        assert 0.0 < model.deviance_explained < 1.0

    def test_t2_aic_bic_finite(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1 * x2) + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_t2_smooth_tests(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = 3 * np.sin(x1) * np.cos(x2) + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        tests = model.smooth_tests()
        assert len(tests) == 1
        assert tests[0].p_value < 0.05

    def test_t2_residuals(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        for rtype in ("response", "pearson", "deviance", "working"):
            r = model.get_residuals(rtype)
            assert np.all(np.isfinite(r))

    def test_t2_with_reml(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}, method="REML"
        )
        assert model.is_fitted

    def test_t2_summary(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        text = model.summary()
        assert "t2(x1, x2" in text

    def test_t2_poisson(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        mu = np.exp(0.5 + x1 * x2)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ t2(x1, x2, k=5)", family=Poisson()).fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert model.is_fitted
        assert 0.0 < model.deviance_explained < 1.0

    def test_t2_with_univariate_smooth(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) + x1 * x2 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=6) + t2(x1, x2, k=5)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        assert model.is_fitted
        assert len(model.edf) == 2

    def test_t2_comparable_fit_to_te(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) * np.cos(3 * x2) + 0.3 * rng.standard_normal(n)

        model_te = GAM("y ~ te(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        model_t2 = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})

        rmse_te = np.sqrt(np.mean(model_te.get_residuals("response") ** 2))
        rmse_t2 = np.sqrt(np.mean(model_t2.get_residuals("response") ** 2))
        assert abs(rmse_te - rmse_t2) / max(rmse_te, 1e-10) < 0.5

    def test_t2_concurvity(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) + x1 * x2 + 0.2 * rng.standard_normal(n)

        model = GAM("y ~ s(x1, k=5) + t2(x1, x2, k=4)").fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        c = model.concurvity(full=True)
        assert c.worst.shape == (2,)
        assert np.all(np.isfinite(c.worst))

    def test_t2_plot(self) -> None:
        pytest.importorskip("altair")
        import altair as alt

        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ t2(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        chart = model.plot()
        assert isinstance(chart, alt.HConcatChart)
        assert chart.to_dict() is not None
