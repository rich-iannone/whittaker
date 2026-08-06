"""Tests for whittaker.model_matrix (design matrix + penalty construction)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.formula.parser import parse
from whittaker.formula.terms import SmoothTerm
from whittaker.model_matrix import (
    ModelMatrix,
    _apply_constraint,
    _apply_constraint_to_penalty,
    _extract_column,
    _resolve_basis,
    build_model_matrix,
    predict_matrix,
)
from whittaker.smooths import CRS, TPRS, PSpline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(23)


def _simple_data(n: int = 100) -> dict[str, np.ndarray]:
    x = np.linspace(0, 1, n)
    return {
        "y": np.sin(2 * np.pi * x) + RNG.normal(0, 0.1, n),
        "x": x,
    }


def _multi_data(n: int = 100) -> dict[str, np.ndarray]:
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    return {
        "y": np.sin(2 * np.pi * x1) + 0.5 * x2 + RNG.normal(0, 0.1, n),
        "x1": x1,
        "x2": x2,
        "group": RNG.choice([0.0, 1.0], n),
    }


# ---------------------------------------------------------------------------
# _resolve_basis
# ---------------------------------------------------------------------------


class TestResolveBasis:
    def test_tp_returns_tprs(self) -> None:
        term = SmoothTerm(variables=("x",), bs="tp")
        basis = _resolve_basis(term)
        assert isinstance(basis, TPRS)

    def test_cr_returns_crs(self) -> None:
        term = SmoothTerm(variables=("x",), bs="cr")
        basis = _resolve_basis(term)
        assert isinstance(basis, CRS)

    def test_ps_returns_pspline(self) -> None:
        term = SmoothTerm(variables=("x",), bs="ps")
        basis = _resolve_basis(term)
        assert isinstance(basis, PSpline)

    def test_custom_k(self) -> None:
        term = SmoothTerm(variables=("x",), bs="cr", k=15)
        basis = _resolve_basis(term)
        assert isinstance(basis, CRS)
        assert basis.k == 15

    def test_pspline_extra_kwargs(self) -> None:
        term = SmoothTerm(variables=("x",), bs="ps", k=12, extra={"degree": 4, "m": 3})
        basis = _resolve_basis(term)
        assert isinstance(basis, PSpline)
        assert basis.k == 12
        assert basis.degree == 4
        assert basis.m == 3

    def test_unknown_bs_raises(self) -> None:
        term = SmoothTerm(variables=("x",), bs="zz")
        with pytest.raises(ValueError, match="Unknown basis type"):
            _resolve_basis(term)


# ---------------------------------------------------------------------------
# _extract_column
# ---------------------------------------------------------------------------


class TestExtractColumn:
    def test_returns_1d_float(self) -> None:
        data = {"x": np.array([1, 2, 3])}
        result = _extract_column(data, "x")
        assert result.dtype == float
        assert result.ndim == 1

    def test_missing_column_raises(self) -> None:
        with pytest.raises(KeyError, match="Column 'z' required"):
            _extract_column({"x": np.array([1])}, "z")


# ---------------------------------------------------------------------------
# _apply_constraint / _apply_constraint_to_penalty
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_constraint_reduces_dimension(self) -> None:
        rng = np.random.default_rng(0)
        B = rng.standard_normal((50, 10))
        C = B.mean(axis=0, keepdims=True)
        B_c = _apply_constraint(B, C)
        assert B_c.shape == (50, 9)

    def test_constrained_basis_satisfies_constraint(self) -> None:
        rng = np.random.default_rng(0)
        B = rng.standard_normal((50, 10))
        C = B.mean(axis=0, keepdims=True)
        B_c = _apply_constraint(B, C)
        Q, _ = np.linalg.qr(C.T, mode="complete")
        Z = Q[:, 1:]
        assert_allclose(C @ Z, 0.0, atol=1e-12)

    def test_penalty_projection_preserves_symmetry(self) -> None:
        S = np.eye(10)
        C = np.ones((1, 10)) / 10
        S_c = _apply_constraint_to_penalty(S, C)
        assert S_c.shape == (9, 9)
        assert_allclose(S_c, S_c.T, atol=1e-14)

    def test_penalty_projection_preserves_psd(self) -> None:
        rng = np.random.default_rng(1)
        A = rng.standard_normal((10, 10))
        S = A.T @ A
        C = np.ones((1, 10))
        S_c = _apply_constraint_to_penalty(S, C)
        eigenvalues = np.linalg.eigvalsh(S_c)
        assert np.all(eigenvalues >= -1e-12)


# ---------------------------------------------------------------------------
# build_model_matrix — basic structure
# ---------------------------------------------------------------------------


class TestBuildModelMatrixBasic:
    def test_single_smooth(self) -> None:
        formula = parse("y ~ s(x)")
        data = _simple_data()
        result = build_model_matrix(formula, data)

        assert isinstance(result, ModelMatrix)
        assert result.n_obs == 100
        assert result.has_intercept
        assert result.n_parametric == 0
        assert len(result.smooths) == 1
        assert len(result.penalties) == 1

    def test_intercept_is_column_of_ones(self) -> None:
        formula = parse("y ~ s(x)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert_allclose(result.X[:, 0], 1.0)

    def test_no_intercept(self) -> None:
        formula = parse("y ~ 0 + s(x)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert not result.has_intercept
        assert result.column_names[0] != "(Intercept)"

    def test_column_names_start_with_intercept(self) -> None:
        formula = parse("y ~ s(x)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert result.column_names[0] == "(Intercept)"

    def test_response_extracted(self) -> None:
        formula = parse("y ~ s(x)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert_allclose(result.response, data["y"])

    def test_design_matrix_shape_with_constraint(self) -> None:
        formula = parse("y ~ s(x, k=10)")
        data = _simple_data()
        result = build_model_matrix(formula, data, apply_constraints=True)
        assert result.X.shape == (100, 1 + 9)

    def test_design_matrix_shape_without_constraint(self) -> None:
        formula = parse("y ~ s(x, k=10)")
        data = _simple_data()
        result = build_model_matrix(formula, data, apply_constraints=False)
        assert result.X.shape == (100, 1 + 10)


# ---------------------------------------------------------------------------
# build_model_matrix — parametric terms
# ---------------------------------------------------------------------------


class TestBuildModelMatrixParametric:
    def test_linear_term(self) -> None:
        formula = parse("y ~ x1 + s(x2)")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        assert result.n_parametric == 1
        assert "x1" in result.column_names

    def test_interaction_term_full(self) -> None:
        formula = parse("y ~ x1 * x2 + s(x1)")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        assert "x1" in result.column_names
        assert "x2" in result.column_names
        assert "x1:x2" in result.column_names
        assert result.n_parametric == 3

    def test_linear_column_values(self) -> None:
        formula = parse("y ~ x1")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        col_idx = result.column_names.index("x1")
        assert_allclose(result.X[:, col_idx], data["x1"])


# ---------------------------------------------------------------------------
# build_model_matrix — multiple smooths
# ---------------------------------------------------------------------------


class TestBuildModelMatrixMultipleSmooths:
    def test_two_smooths(self) -> None:
        formula = parse("y ~ s(x1) + s(x2)")
        data = _multi_data()
        result = build_model_matrix(formula, data)

        assert len(result.smooths) == 2
        assert len(result.penalties) == 2

    def test_smooth_blocks_non_overlapping(self) -> None:
        formula = parse("y ~ s(x1) + s(x2)")
        data = _multi_data()
        result = build_model_matrix(formula, data)

        s0 = result.smooths[0]
        s1 = result.smooths[1]
        assert s0.col_end <= s1.col_start

    def test_penalty_matrices_are_block_diagonal(self) -> None:
        formula = parse("y ~ s(x1, k=8) + s(x2, k=8)")
        data = _multi_data()
        result = build_model_matrix(formula, data)

        for i, (pen, info) in enumerate(zip(result.penalties, result.smooths)):
            assert pen.shape == (result.n_coefs, result.n_coefs)
            block = pen[info.col_start : info.col_end, info.col_start : info.col_end]
            assert np.any(block != 0.0)

            mask = np.ones_like(pen, dtype=bool)
            mask[info.col_start : info.col_end, info.col_start : info.col_end] = False
            assert_allclose(pen[mask], 0.0)


# ---------------------------------------------------------------------------
# build_model_matrix — different basis types
# ---------------------------------------------------------------------------


class TestBuildModelMatrixBasisTypes:
    def test_cr_basis(self) -> None:
        formula = parse("y ~ s(x, bs='cr', k=8)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert isinstance(result.smooths[0].basis, CRS)

    def test_ps_basis(self) -> None:
        formula = parse("y ~ s(x, bs='ps', k=8)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert isinstance(result.smooths[0].basis, PSpline)

    def test_tp_basis(self) -> None:
        formula = parse("y ~ s(x, bs='tp', k=8)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert isinstance(result.smooths[0].basis, TPRS)

    def test_mixed_bases(self) -> None:
        formula = parse("y ~ s(x1, bs='cr', k=8) + s(x2, bs='ps', k=8)")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        assert isinstance(result.smooths[0].basis, CRS)
        assert isinstance(result.smooths[1].basis, PSpline)


# ---------------------------------------------------------------------------
# build_model_matrix — penalty properties
# ---------------------------------------------------------------------------


class TestPenaltyProperties:
    def test_each_penalty_is_symmetric(self) -> None:
        formula = parse("y ~ s(x1, k=8) + s(x2, k=8)")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        for pen in result.penalties:
            assert_allclose(pen, pen.T, atol=1e-14)

    def test_each_penalty_is_psd(self) -> None:
        formula = parse("y ~ s(x1, k=8) + s(x2, k=8)")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        for pen in result.penalties:
            eigenvalues = np.linalg.eigvalsh(pen)
            assert np.all(eigenvalues >= -1e-10)

    def test_combined_penalty_shape(self) -> None:
        formula = parse("y ~ s(x1, k=8) + s(x2, k=8)")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        combined = result.penalty_matrix
        assert combined.shape == (result.n_coefs, result.n_coefs)

    def test_parametric_columns_unpenalized(self) -> None:
        formula = parse("y ~ x1 + s(x2, k=8)")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        combined = result.penalty_matrix
        n_unpen = 1 + result.n_parametric  # intercept + linear
        assert_allclose(combined[:n_unpen, :], 0.0)
        assert_allclose(combined[:, :n_unpen], 0.0)

    def test_no_smooths_gives_empty_penalties(self) -> None:
        formula = parse("y ~ x1")
        data = _multi_data()
        result = build_model_matrix(formula, data)
        assert result.penalties == []
        assert_allclose(result.penalty_matrix, 0.0)


# ---------------------------------------------------------------------------
# build_model_matrix — offset
# ---------------------------------------------------------------------------


class TestOffset:
    def test_offset_extracted(self) -> None:
        formula = parse("y ~ s(x) + offset(x)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert result.offset is not None
        assert_allclose(result.offset, data["x"])

    def test_no_offset_gives_none(self) -> None:
        formula = parse("y ~ s(x)")
        data = _simple_data()
        result = build_model_matrix(formula, data)
        assert result.offset is None


# ---------------------------------------------------------------------------
# build_model_matrix — error handling
# ---------------------------------------------------------------------------


class TestBuildModelMatrixErrors:
    def test_missing_column_raises(self) -> None:
        formula = parse("y ~ s(z)")
        data = _simple_data()
        with pytest.raises(KeyError, match="Column 'z'"):
            build_model_matrix(formula, data)

    def test_missing_response_raises(self) -> None:
        formula = parse("z ~ s(x)")
        data = _simple_data()
        with pytest.raises(KeyError, match="Column 'z'"):
            build_model_matrix(formula, data)

    def test_unequal_lengths_raises(self) -> None:
        formula = parse("y ~ s(x)")
        data = {"y": np.zeros(10), "x": np.zeros(20)}
        with pytest.raises(ValueError, match="same length"):
            build_model_matrix(formula, data)

    def test_unsupported_smooth_type_raises(self) -> None:
        formula = parse("y ~ s(x1, k=5)")
        formula.terms[0] = SmoothTerm(
            variables=("x1",), smooth_type="t2", bs="cr", k=5, extra={}
        )
        data = _multi_data()
        with pytest.raises(NotImplementedError, match="not yet supported"):
            build_model_matrix(formula, data)

    def test_unknown_basis_raises(self) -> None:
        formula = parse("y ~ s(x, bs='zz')")
        data = _simple_data()
        with pytest.raises(ValueError, match="Unknown basis type"):
            build_model_matrix(formula, data)


# ---------------------------------------------------------------------------
# predict_matrix
# ---------------------------------------------------------------------------


class TestPredictMatrix:
    def test_same_shape_as_training(self) -> None:
        formula = parse("y ~ s(x, k=8)")
        data = _simple_data(100)
        model = build_model_matrix(formula, data)

        new_data = {"x": np.linspace(0, 1, 50)}
        X_new = predict_matrix(model, new_data)
        assert X_new.shape == (50, model.n_coefs)

    def test_reproduces_training_matrix(self) -> None:
        formula = parse("y ~ s(x, k=8)")
        data = _simple_data(100)
        model = build_model_matrix(formula, data)

        X_pred = predict_matrix(model, {"x": data["x"]})
        assert_allclose(X_pred, model.X, atol=1e-10)

    def test_with_linear_and_smooth(self) -> None:
        formula = parse("y ~ x1 + s(x2, k=8)")
        data = _multi_data(80)
        model = build_model_matrix(formula, data)

        new_data = {"x1": data["x1"], "x2": data["x2"]}
        X_new = predict_matrix(model, new_data)
        assert_allclose(X_new, model.X, atol=1e-10)

    def test_prediction_intercept_column(self) -> None:
        formula = parse("y ~ s(x, k=8)")
        data = _simple_data()
        model = build_model_matrix(formula, data)

        new_data = {"x": np.linspace(0, 1, 30)}
        X_new = predict_matrix(model, new_data)
        assert_allclose(X_new[:, 0], 1.0)

    def test_multiple_smooths(self) -> None:
        formula = parse("y ~ s(x1, k=8) + s(x2, k=8)")
        data = _multi_data(80)
        model = build_model_matrix(formula, data)

        new_data = {"x1": data["x1"], "x2": data["x2"]}
        X_new = predict_matrix(model, new_data)
        assert_allclose(X_new, model.X, atol=1e-10)


# ---------------------------------------------------------------------------
# Numerical: penalized least squares produces smooth fit
# ---------------------------------------------------------------------------


class TestNumericalSmooth:
    def test_penalized_least_squares_fit(self) -> None:
        n = 200
        x = np.linspace(0, 1, n)
        y_true = np.sin(2 * np.pi * x)
        y = y_true + RNG.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=20)")
        data = {"y": y, "x": x}
        model = build_model_matrix(formula, data)

        lam = 0.001
        S_total = lam * model.penalty_matrix
        XtX = model.X.T @ model.X + S_total
        Xty = model.X.T @ y
        beta = np.linalg.solve(XtX, Xty)
        y_hat = model.X @ beta

        residual = np.sqrt(np.mean((y_hat - y_true) ** 2))
        assert residual < 0.15

    def test_higher_lambda_gives_smoother_fit(self) -> None:
        n = 200
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + RNG.normal(0, 0.2, n)

        formula = parse("y ~ s(x, k=20)")
        data = {"y": y, "x": x}
        model = build_model_matrix(formula, data)

        roughnesses = []
        for lam in [0.01, 1.0, 100.0]:
            S_total = lam * model.penalty_matrix
            beta = np.linalg.solve(model.X.T @ model.X + S_total, model.X.T @ y)
            roughness = float(beta @ model.penalty_matrix @ beta)
            roughnesses.append(roughness)

        assert roughnesses[0] > roughnesses[1] > roughnesses[2]
