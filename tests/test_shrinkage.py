"""Tests for shrinkage smooth bases: ShrinkageTPRS (ts) and ShrinkageCRS (cs)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.poisson import Poisson
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.smooths.cubic import CRS
from whittaker.smooths.shrinkage import ShrinkageCRS, ShrinkageTPRS, _null_space_penalty
from whittaker.smooths.tprs import TPRS

# ---------------------------------------------------------------------------
# ShrinkageTPRS unit tests
# ---------------------------------------------------------------------------


class TestShrinkageTPRS:
    def test_is_subclass_of_tprs(self) -> None:
        assert issubclass(ShrinkageTPRS, TPRS)

    def test_same_basis_as_tprs(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)

        tp = TPRS(k=8).fit(x)
        ts = ShrinkageTPRS(k=8).fit(x)

        assert_allclose(tp.basis_matrix(x), ts.basis_matrix(x))

    def test_two_penalties(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        ts = ShrinkageTPRS(k=8).fit(x)
        pens = ts.penalty_matrices()
        assert len(pens) == 2

    def test_first_penalty_matches_tprs(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)

        tp = TPRS(k=8).fit(x)
        ts = ShrinkageTPRS(k=8).fit(x)

        assert_allclose(ts.penalty_matrices()[0], tp.penalty_matrix())

    def test_null_space_penalty_structure(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        ts = ShrinkageTPRS(k=8).fit(x)

        S_null = ts.penalty_matrices()[1]
        M = 2  # d=1, M=d+1=2 for TPRS
        assert_allclose(S_null[:M, :M], np.eye(M), atol=1e-14)
        assert_allclose(S_null[M:, :], 0.0, atol=1e-14)
        assert_allclose(S_null[:, M:], 0.0, atol=1e-14)

    def test_null_space_dimension_is_zero(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        ts = ShrinkageTPRS(k=8).fit(x)
        assert ts.null_space_dimension() == 0

    def test_penalties_psd(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        ts = ShrinkageTPRS(k=8).fit(x)
        for P in ts.penalty_matrices():
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-10)

    def test_penalties_symmetric(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        ts = ShrinkageTPRS(k=8).fit(x)
        for P in ts.penalty_matrices():
            assert_allclose(P, P.T, atol=1e-14)

    def test_combined_penalties_full_rank(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        ts = ShrinkageTPRS(k=8).fit(x)
        pens = ts.penalty_matrices()
        S_combined = pens[0] + pens[1]
        eigvals = np.linalg.eigvalsh(S_combined)
        assert np.all(eigvals > 1e-12)

    def test_penalty_matrix_shape(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        ts = ShrinkageTPRS(k=8).fit(x)
        for P in ts.penalty_matrices():
            assert P.shape == (8, 8)


# ---------------------------------------------------------------------------
# ShrinkageCRS unit tests
# ---------------------------------------------------------------------------


class TestShrinkageCRS:
    def test_is_subclass_of_crs(self) -> None:
        assert issubclass(ShrinkageCRS, CRS)

    def test_same_basis_as_crs(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)

        cr = CRS(k=8).fit(x)
        cs = ShrinkageCRS(k=8).fit(x)

        assert_allclose(cr.basis_matrix(x), cs.basis_matrix(x))

    def test_two_penalties(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cs = ShrinkageCRS(k=8).fit(x)
        pens = cs.penalty_matrices()
        assert len(pens) == 2

    def test_first_penalty_matches_crs(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)

        cr = CRS(k=8).fit(x)
        cs = ShrinkageCRS(k=8).fit(x)

        assert_allclose(cs.penalty_matrices()[0], cr.penalty_matrix())

    def test_null_space_penalty_rank(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cs = ShrinkageCRS(k=8).fit(x)

        S_null = cs.penalty_matrices()[1]
        eigvals = np.linalg.eigvalsh(S_null)
        n_positive = np.sum(eigvals > 1e-10)
        assert n_positive == 2  # constant + linear null space

    def test_null_space_dimension_is_zero(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cs = ShrinkageCRS(k=8).fit(x)
        assert cs.null_space_dimension() == 0

    def test_penalties_psd(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cs = ShrinkageCRS(k=8).fit(x)
        for P in cs.penalty_matrices():
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-10)

    def test_penalties_symmetric(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cs = ShrinkageCRS(k=8).fit(x)
        for P in cs.penalty_matrices():
            assert_allclose(P, P.T, atol=1e-14)

    def test_combined_penalties_full_rank(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cs = ShrinkageCRS(k=8).fit(x)
        pens = cs.penalty_matrices()
        S_combined = pens[0] + pens[1]
        eigvals = np.linalg.eigvalsh(S_combined)
        assert np.all(eigvals > 1e-12)

    def test_null_space_penalty_orthogonal_to_range(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cs = ShrinkageCRS(k=8).fit(x)
        S_wiggle, S_null = cs.penalty_matrices()
        assert_allclose(S_wiggle @ S_null, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _null_space_penalty helper tests
# ---------------------------------------------------------------------------


class TestNullSpacePenalty:
    def test_projects_onto_null_space(self) -> None:
        S = np.diag([0.0, 0.0, 1.0, 2.0, 3.0])
        S_null = _null_space_penalty(S)
        expected = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
        assert_allclose(S_null, expected, atol=1e-14)

    def test_idempotent(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.uniform(size=50)
        cr = CRS(k=8).fit(x)
        S_null = _null_space_penalty(cr.penalty_matrix())
        assert_allclose(S_null @ S_null, S_null, atol=1e-10)


# ---------------------------------------------------------------------------
# Model matrix integration
# ---------------------------------------------------------------------------


class TestShrinkageModelMatrix:
    def test_ts_builds_model_matrix(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(size=n)
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        formula = parse('y ~ s(x, bs="ts")')
        mm = build_model_matrix(formula, {"y": y, "x": x})
        assert mm.X.shape[0] == n
        assert len(mm.smooths) == 1
        assert len(mm.penalties) == 2

    def test_cs_builds_model_matrix(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(size=n)
        y = np.sin(x) + 0.1 * rng.standard_normal(n)

        formula = parse('y ~ s(x, bs="cs")')
        mm = build_model_matrix(formula, {"y": y, "x": x})
        assert mm.X.shape[0] == n
        assert len(mm.penalties) == 2

    def test_ts_same_basis_cols_as_tp(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(size=n)
        y = 0.1 * rng.standard_normal(n)

        mm_tp = build_model_matrix(parse("y ~ s(x, k=8)"), {"y": y, "x": x})
        mm_ts = build_model_matrix(parse('y ~ s(x, bs="ts", k=8)'), {"y": y, "x": x})

        tp_cols = mm_tp.smooths[0].col_end - mm_tp.smooths[0].col_start
        ts_cols = mm_ts.smooths[0].col_end - mm_ts.smooths[0].col_start
        assert ts_cols == tp_cols

    def test_ts_penalty_indices(self) -> None:
        rng = np.random.default_rng(23)
        n = 100
        x = rng.uniform(size=n)
        y = 0.1 * rng.standard_normal(n)

        mm = build_model_matrix(parse('y ~ s(x, bs="ts", k=8)'), {"y": y, "x": x})
        assert len(mm.smooths[0].penalty_indices) == 2


# ---------------------------------------------------------------------------
# GAM integration tests
# ---------------------------------------------------------------------------


class TestShrinkageGAM:
    def test_ts_fit_converges(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        assert model.is_fitted

    def test_cs_fit_converges(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cs")').fit({"y": y, "x": x})
        assert model.is_fitted

    def test_ts_two_smoothing_params_per_smooth(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        assert len(model.smoothing_params) == 2

    def test_ts_deviance_explained(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        assert model.deviance_explained > 0.5

    def test_cs_deviance_explained(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cs")').fit({"y": y, "x": x})
        assert model.deviance_explained > 0.5

    def test_ts_shrinks_noise_variable(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x1, bs="ts") + s(x2, bs="ts")').fit({"y": y, "x1": x1, "x2": x2})
        assert model.edf[1] < model.edf[0]

    def test_cs_shrinks_noise_variable(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x1, bs="cs") + s(x2, bs="cs")').fit({"y": y, "x1": x1, "x2": x2})
        assert model.edf[1] < model.edf[0]

    def test_ts_predict(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        pred = model.predict({"x": np.array([0.2, 0.5, 0.8])}, se=True)
        assert np.all(np.isfinite(pred.values))
        assert pred.se is not None
        assert np.all(pred.se > 0)

    def test_ts_aic_bic_finite(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_ts_smooth_tests(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(size=n)
        y = 3 * np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        tests = model.smooth_tests()
        assert len(tests) == 1
        assert tests[0].p_value < 0.05

    def test_ts_residuals(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        for rtype in ("response", "pearson", "deviance", "working"):
            r = model.get_residuals(rtype)
            assert np.all(np.isfinite(r))

    def test_ts_with_reml(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x}, method="REML")
        assert model.is_fitted

    def test_ts_summary(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        text = model.summary()
        assert "s(x" in text

    def test_ts_poisson(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(size=n)
        mu = np.exp(0.5 + np.sin(3 * x))
        y = rng.poisson(mu).astype(float)

        model = GAM('y ~ s(x, bs="ts")', family=Poisson()).fit({"y": y, "x": x})
        assert model.is_fitted

    def test_ts_with_other_smooths(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(3 * x1) + x2**2 + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x1, bs="ts") + s(x2, bs="cr")').fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted
        assert len(model.edf) == 2

    def test_ts_comparable_fit_to_tp(self) -> None:
        rng = np.random.default_rng(23)
        n = 300
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.3 * rng.standard_normal(n)

        model_tp = GAM("y ~ s(x, k=8)").fit({"y": y, "x": x})
        model_ts = GAM('y ~ s(x, bs="ts", k=8)').fit({"y": y, "x": x})

        rmse_tp = np.sqrt(np.mean(model_tp.get_residuals("response") ** 2))
        rmse_ts = np.sqrt(np.mean(model_ts.get_residuals("response") ** 2))
        assert abs(rmse_tp - rmse_ts) / max(rmse_tp, 1e-10) < 0.3

    def test_ts_in_tensor_product(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM('y ~ te(x1, x2, bs="ts", k=5)').fit({"y": y, "x1": x1, "x2": x2})
        assert model.is_fitted

    def test_ts_concurvity(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = np.sin(x1) + x2 + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x1, bs="ts", k=8) + s(x2, bs="ts", k=6)').fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        c = model.concurvity(full=True)
        assert c.worst.shape == (2,)
        assert np.all(np.isfinite(c.worst))

    def test_ts_plot(self) -> None:
        pytest.importorskip("altair")

        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="ts")').fit({"y": y, "x": x})
        chart = model.plot()
        assert chart.to_dict() is not None

    def test_cs_plot(self) -> None:
        pytest.importorskip("altair")

        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(size=n)
        y = np.sin(3 * x) + 0.2 * rng.standard_normal(n)

        model = GAM('y ~ s(x, bs="cs")').fit({"y": y, "x": x})
        chart = model.plot()
        assert chart.to_dict() is not None
