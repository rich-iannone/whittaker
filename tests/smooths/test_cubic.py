"""Tests for whittaker.smooths.cubic (Cubic Regression Splines)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.smooths.cubic import CRS, _build_Q, _build_R

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_x(n: int = 60) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# _build_Q
# ---------------------------------------------------------------------------


class TestBuildQ:
    def test_shape(self) -> None:
        h = np.ones(5)  # k=6 → k-2=4
        Q = _build_Q(h)
        assert Q.shape == (6, 4)

    def test_column_sums_zero(self) -> None:
        h = RNG.uniform(0.5, 2.0, 7)
        Q = _build_Q(h)
        assert_allclose(Q.sum(axis=0), 0.0, atol=1e-12)

    def test_Q_annihilates_linear(self) -> None:
        # Q' applied to [t_0, ..., t_{k-1}] should be zero.
        h = np.array([1.0, 2.0, 0.5, 1.5])  # k=5
        Q = _build_Q(h)
        knots = np.concatenate([[0.0], np.cumsum(h)])
        assert_allclose(Q.T @ knots, 0.0, atol=1e-12)

    def test_Q_annihilates_constant(self) -> None:
        h = np.array([1.0, 2.0, 0.5, 1.5])
        Q = _build_Q(h)
        assert_allclose(Q.T @ np.ones(len(h) + 1), 0.0, atol=1e-12)

    def test_sparsity_structure(self) -> None:
        # Each column has exactly 3 non-zero entries.
        h = np.ones(6)
        Q = _build_Q(h)
        for col in range(Q.shape[1]):
            assert np.count_nonzero(Q[:, col]) == 3


# ---------------------------------------------------------------------------
# _build_R
# ---------------------------------------------------------------------------


class TestBuildR:
    def test_shape(self) -> None:
        h = np.ones(5)  # k=6, k-2=4
        R = _build_R(h)
        assert R.shape == (4, 4)

    def test_symmetry(self) -> None:
        h = RNG.uniform(0.5, 2.0, 5)
        R = _build_R(h)
        assert_allclose(R, R.T)

    def test_positive_definite(self) -> None:
        h = RNG.uniform(0.5, 2.0, 6)
        R = _build_R(h)
        eigs = np.linalg.eigvalsh(R)
        assert np.all(eigs > 0)

    def test_diagonal_entries(self) -> None:
        h = np.array([1.0, 2.0, 3.0, 4.0])
        R = _build_R(h)
        # R[l,l] = (h[l] + h[l+1]) / 3
        expected_diag = [(h[l] + h[l + 1]) / 3.0 for l in range(len(h) - 1)]
        assert_allclose(np.diag(R), expected_diag)

    def test_offdiagonal_entries(self) -> None:
        h = np.array([1.0, 2.0, 3.0, 4.0])
        R = _build_R(h)
        # R[l, l+1] = h[l+1] / 6
        expected_off = [h[l + 1] / 6.0 for l in range(len(h) - 2)]
        assert_allclose(np.diag(R, 1), expected_off)


# ---------------------------------------------------------------------------
# CRS constructor
# ---------------------------------------------------------------------------


class TestCRSConstructor:
    def test_default_k(self) -> None:
        assert CRS().k == 10

    def test_custom_k(self) -> None:
        assert CRS(k=15).k == 15

    def test_k_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            CRS(k=2)

    def test_not_fitted_initially(self) -> None:
        assert not CRS().is_fitted

    def test_n_basis_before_fit(self) -> None:
        assert CRS(k=8).n_basis == 8


# ---------------------------------------------------------------------------
# CRS.fit
# ---------------------------------------------------------------------------


class TestCRSFit:
    def test_fit_returns_self(self) -> None:
        basis = CRS(k=8)
        assert basis.fit(_make_x()) is basis

    def test_is_fitted_after_fit(self) -> None:
        assert CRS(k=8).fit(_make_x()).is_fitted

    def test_knots_length(self) -> None:
        basis = CRS(k=8).fit(_make_x())
        assert len(basis.knots) == 8

    def test_knots_are_sorted(self) -> None:
        x = RNG.uniform(0, 5, 80)
        basis = CRS(k=10).fit(x)
        assert np.all(np.diff(basis.knots) > 0)

    def test_knots_span_data_range(self) -> None:
        x = _make_x(60)
        basis = CRS(k=10).fit(x)
        assert_allclose(basis.knots[0], x.min())
        assert_allclose(basis.knots[-1], x.max())

    def test_k_larger_than_n_raises(self) -> None:
        with pytest.raises(ValueError, match="smaller than k"):
            CRS(k=20).fit(np.linspace(0, 1, 10))

    def test_column_vector_input(self) -> None:
        x = _make_x(30)[:, np.newaxis]
        basis = CRS(k=8).fit(x)
        assert basis.is_fitted

    def test_multidim_input_raises(self) -> None:
        x = np.ones((20, 2))
        with pytest.raises(ValueError, match="univariate"):
            CRS(k=8).fit(x)

    def test_tied_quantile_knots_fall_back_to_linspace(self) -> None:
        # Heavily repeated values push several quantiles to the same value,
        # so quantile-based knots contain ties and the fit must fall back
        # to an evenly-spaced linspace over the data range.
        x = np.concatenate([np.zeros(50), np.linspace(0.0, 1.0, 10)])
        basis = CRS(k=8).fit(x)
        assert len(np.unique(basis.knots)) == 8
        assert_allclose(basis.knots, np.linspace(x.min(), x.max(), 8))


# ---------------------------------------------------------------------------
# CRS.basis_matrix
# ---------------------------------------------------------------------------


class TestCRSBasisMatrix:
    def test_shape_at_training_points(self) -> None:
        x = _make_x(50)
        B = CRS(k=10).fit(x).basis_matrix(x)
        assert B.shape == (50, 10)

    def test_shape_at_new_points(self) -> None:
        x_train = _make_x(50)
        x_new = np.linspace(0.1, 0.9, 20)
        B = CRS(k=10).fit(x_train).basis_matrix(x_new)
        assert B.shape == (20, 10)

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            CRS(k=8).basis_matrix(np.array([0.5]))

    def test_partition_of_unity_at_knots(self) -> None:
        # At every knot, exactly one basis function equals 1 and the rest 0.
        x = _make_x(40)
        basis = CRS(k=8).fit(x)
        B = basis.basis_matrix(basis.knots)
        assert_allclose(B, np.eye(8), atol=1e-10)

    def test_rows_sum_to_one_interior(self) -> None:
        # B @ [1,...,1] = 1 everywhere (constant function is exactly reproduced).
        x = _make_x(50)
        basis = CRS(k=10).fit(x)
        B = basis.basis_matrix(x)
        assert_allclose(B @ np.ones(10), np.ones(50), atol=1e-10)

    def test_rows_sum_to_one_extrapolation(self) -> None:
        x = _make_x(50)
        basis = CRS(k=10).fit(x)
        x_ext = np.array([-0.5, 1.5, 2.0])
        B = basis.basis_matrix(x_ext)
        assert_allclose(B @ np.ones(10), np.ones(3), atol=1e-10)

    def test_linear_reproduction_interior(self) -> None:
        # B @ knots = x for all interior evaluation points.
        x = _make_x(50)
        basis = CRS(k=10).fit(x)
        B = basis.basis_matrix(x)
        assert_allclose(B @ basis.knots, x, atol=1e-10)

    def test_linear_reproduction_extrapolation(self) -> None:
        x = _make_x(50)
        basis = CRS(k=10).fit(x)
        x_ext = np.array([-0.3, 1.4])
        B = basis.basis_matrix(x_ext)
        assert_allclose(B @ basis.knots, x_ext, atol=1e-10)

    def test_single_point(self) -> None:
        basis = CRS(k=8).fit(_make_x())
        B = basis.basis_matrix(np.array([0.5]))
        assert B.shape == (1, 8)

    def test_column_vector_input_accepted(self) -> None:
        x = _make_x(40)
        basis = CRS(k=8).fit(x)
        B1 = basis.basis_matrix(x)
        B2 = basis.basis_matrix(x[:, np.newaxis])
        assert_allclose(B1, B2)

    def test_continuity_at_knots(self) -> None:
        # Basis functions should be continuous: values just left and right of
        # each interior knot should match.
        x = _make_x(60)
        basis = CRS(k=8).fit(x)
        eps = 1e-7
        for t in basis.knots[1:-1]:
            B_left = basis.basis_matrix(np.array([t - eps]))
            B_right = basis.basis_matrix(np.array([t + eps]))
            assert_allclose(B_left, B_right, atol=1e-5)


# ---------------------------------------------------------------------------
# CRS.penalty_matrix
# ---------------------------------------------------------------------------


class TestCRSPenaltyMatrix:
    def test_shape(self) -> None:
        S = CRS(k=10).fit(_make_x()).penalty_matrix()
        assert S.shape == (10, 10)

    def test_symmetry(self) -> None:
        S = CRS(k=10).fit(_make_x()).penalty_matrix()
        assert_allclose(S, S.T)

    def test_positive_semidefinite(self) -> None:
        S = CRS(k=10).fit(_make_x()).penalty_matrix()
        eigs = np.linalg.eigvalsh(S)
        assert np.all(eigs >= -1e-10)

    def test_rank_is_k_minus_2(self) -> None:
        k = 10
        S = CRS(k=k).fit(_make_x()).penalty_matrix()
        assert np.linalg.matrix_rank(S, tol=1e-8) == k - 2

    def test_null_space_contains_constant(self) -> None:
        basis = CRS(k=10).fit(_make_x())
        S = basis.penalty_matrix()
        ones = np.ones(10)
        assert_allclose(S @ ones, 0.0, atol=1e-10)

    def test_null_space_contains_linear(self) -> None:
        basis = CRS(k=10).fit(_make_x())
        S = basis.penalty_matrix()
        assert_allclose(S @ basis.knots, 0.0, atol=1e-10)

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            CRS(k=8).penalty_matrix()


# ---------------------------------------------------------------------------
# CRS.null_space_dimension
# ---------------------------------------------------------------------------


class TestCRSNullSpaceDimension:
    def test_always_two(self) -> None:
        assert CRS(k=10).null_space_dimension() == 2

    def test_independent_of_k(self) -> None:
        for k in [3, 5, 10, 20]:
            assert CRS(k=k).null_space_dimension() == 2


# ---------------------------------------------------------------------------
# CRS.identifiability_constraints
# ---------------------------------------------------------------------------


class TestCRSIdentifiabilityConstraints:
    def test_shape(self) -> None:
        basis = CRS(k=10).fit(_make_x())
        C = basis.identifiability_constraints()
        assert C is not None
        assert C.shape == (1, 10)

    def test_equals_mean_of_knot_basis_rows(self) -> None:
        x = _make_x(50)
        basis = CRS(k=10).fit(x)
        C = basis.identifiability_constraints()
        B_knots = basis.basis_matrix(basis.knots)
        assert_allclose(C, B_knots.mean(axis=0, keepdims=True))

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            CRS(k=8).identifiability_constraints()


# ---------------------------------------------------------------------------
# Numerical / interpolation properties
# ---------------------------------------------------------------------------


class TestCRSNumerical:
    def test_basis_rank_equals_k(self) -> None:
        x = _make_x(80)
        k = 12
        B = CRS(k=k).fit(x).basis_matrix(x)
        assert np.linalg.matrix_rank(B) == k

    def test_smooth_fit_1d(self) -> None:
        """Least-squares + mild penalty should reconstruct a smooth function."""
        rng = np.random.default_rng(3)
        x = np.linspace(0.0, 2 * np.pi, 100)
        y = np.sin(x) + rng.normal(0, 0.05, 100)

        basis = CRS(k=15).fit(x)
        B = basis.basis_matrix(x)
        S = basis.penalty_matrix()

        lam = 1e-4
        coef = np.linalg.solve(B.T @ B + lam * S, B.T @ y)
        y_hat = B @ coef

        assert_allclose(y_hat, np.sin(x), atol=0.15)

    def test_prediction_consistency_at_training_points(self) -> None:
        """basis_matrix evaluated at training x must equal evaluation at same x."""
        x = _make_x(40)
        basis = CRS(k=10).fit(x)
        B1 = basis.basis_matrix(x)
        B2 = basis.basis_matrix(x.copy())
        assert_allclose(B1, B2)

    def test_penalty_quadratic_form_nonneg(self) -> None:
        """β' S β ≥ 0 for all β."""
        rng = np.random.default_rng(7)
        basis = CRS(k=10).fit(_make_x())
        S = basis.penalty_matrix()
        for _ in range(20):
            beta = rng.standard_normal(10)
            assert beta @ S @ beta >= -1e-12

    def test_linear_function_has_zero_penalty(self) -> None:
        """Any affine function β = a + b * knots has zero penalty."""
        basis = CRS(k=10).fit(_make_x())
        S = basis.penalty_matrix()
        a, b = 3.1, -2.7
        beta = a * np.ones(10) + b * basis.knots
        assert_allclose(beta @ S @ beta, 0.0, atol=1e-8)

    def test_nonlinear_function_has_positive_penalty(self) -> None:
        """A genuinely curved β should yield a positive penalty."""
        basis = CRS(k=10).fit(_make_x())
        S = basis.penalty_matrix()
        beta = np.sin(basis.knots * 4)
        assert beta @ S @ beta > 1e-6
