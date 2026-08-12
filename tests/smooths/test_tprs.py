"""Tests for whittaker.smooths.tprs (Thin Plate Regression Splines)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.smooths.tprs import TPRS, _kernel_matrix, _polynomial_null_space, _radial_basis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(23)


def _make_1d(n: int = 50) -> np.ndarray:
    """Uniformly spaced 1-D training data on [0, 1]."""
    return np.linspace(0, 1, n)


def _make_2d(n: int = 50) -> np.ndarray:
    """Random 2-D training data on [0, 1]^2."""
    return RNG.uniform(0, 1, size=(n, 2))


# ---------------------------------------------------------------------------
# _radial_basis
# ---------------------------------------------------------------------------


class TestRadialBasis:
    def test_1d_m2_uses_cubic(self) -> None:
        # d=1, m=2 → power = 2*2 - 1 = 3 (odd) → r^3
        r = np.array([0.0, 1.0, 2.0, 3.0])
        result = _radial_basis(r, d=1, m=2)
        assert_allclose(result, r**3)

    def test_2d_m2_uses_r2_log_r(self) -> None:
        # d=2, m=2 → power = 2 (even) → r^2 * log(r), 0 at r=0
        r = np.array([0.0, 1.0, 2.0])
        result = _radial_basis(r, d=2, m=2)
        expected = np.array([0.0, 0.0, 4.0 * np.log(2.0)])
        assert_allclose(result, expected)

    def test_zero_radius_gives_zero_for_even_power(self) -> None:
        result = _radial_basis(np.array([0.0]), d=2, m=2)
        assert result[0] == 0.0

    def test_zero_radius_gives_zero_for_odd_power(self) -> None:
        result = _radial_basis(np.array([0.0]), d=1, m=2)
        assert result[0] == 0.0

    def test_invalid_power_raises(self) -> None:
        # m=1, d=3 → power = -1 ≤ 0 → error
        with pytest.raises(ValueError, match="too low for dimension"):
            _radial_basis(np.array([1.0]), d=3, m=1)


# ---------------------------------------------------------------------------
# _polynomial_null_space
# ---------------------------------------------------------------------------


class TestPolynomialNullSpace:
    def test_1d_shape(self) -> None:
        x = np.linspace(0, 1, 20)[:, np.newaxis]
        T = _polynomial_null_space(x, m=2)
        assert T.shape == (20, 2)  # [1, x]

    def test_2d_shape(self) -> None:
        x = np.ones((30, 2))
        T = _polynomial_null_space(x, m=2)
        assert T.shape == (30, 3)  # [1, x1, x2]

    def test_first_column_is_ones(self) -> None:
        x = RNG.standard_normal((25, 2))
        T = _polynomial_null_space(x, m=2)
        assert_allclose(T[:, 0], np.ones(25))

    def test_remaining_columns_match_x(self) -> None:
        x = RNG.standard_normal((15, 2))
        T = _polynomial_null_space(x, m=2)
        assert_allclose(T[:, 1:], x)

    def test_m3_quadratic_terms(self) -> None:
        x = RNG.standard_normal((20, 2))
        T = _polynomial_null_space(x, m=3)
        # M = C(2+2, 2) = 6: [1, x1, x2, x1², x1·x2, x2²]
        assert T.shape == (20, 6)
        assert_allclose(T[:, 0], np.ones(20))
        assert_allclose(T[:, 1], x[:, 0])
        assert_allclose(T[:, 2], x[:, 1])
        assert_allclose(T[:, 3], x[:, 0] ** 2)
        assert_allclose(T[:, 4], x[:, 0] * x[:, 1])
        assert_allclose(T[:, 5], x[:, 1] ** 2)


# ---------------------------------------------------------------------------
# _kernel_matrix
# ---------------------------------------------------------------------------


class TestKernelMatrix:
    def test_square_1d(self) -> None:
        x = np.linspace(0, 1, 10)[:, np.newaxis]
        E = _kernel_matrix(x, x, d=1, m=2)
        assert E.shape == (10, 10)

    def test_rectangular(self) -> None:
        x1 = np.linspace(0, 1, 8)[:, np.newaxis]
        x2 = np.linspace(0, 1, 5)[:, np.newaxis]
        E = _kernel_matrix(x1, x2, d=1, m=2)
        assert E.shape == (8, 5)

    def test_diagonal_is_zero_for_1d(self) -> None:
        # η(0) = 0^3 = 0
        x = np.linspace(0, 1, 10)[:, np.newaxis]
        E = _kernel_matrix(x, x, d=1, m=2)
        assert_allclose(np.diag(E), 0.0)

    def test_diagonal_is_zero_for_2d(self) -> None:
        # η(0) = 0 for even power too
        x = RNG.uniform(0, 1, (10, 2))
        E = _kernel_matrix(x, x, d=2, m=2)
        assert_allclose(np.diag(E), 0.0)

    def test_symmetry(self) -> None:
        x = np.linspace(0, 1, 12)[:, np.newaxis]
        E = _kernel_matrix(x, x, d=1, m=2)
        assert_allclose(E, E.T)


# ---------------------------------------------------------------------------
# TPRS constructor validation
# ---------------------------------------------------------------------------


class TestTPRSConstructor:
    def test_default_k_and_m(self) -> None:
        basis = TPRS()
        assert basis.k == 10
        assert basis.m == 2

    def test_custom_k(self) -> None:
        basis = TPRS(k=15)
        assert basis.k == 15

    def test_m3_supported(self) -> None:
        basis = TPRS(k=20, m=3)
        assert basis.m == 3

    def test_m_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            TPRS(m=1)

    def test_k_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            TPRS(k=1)

    def test_not_fitted_initially(self) -> None:
        basis = TPRS()
        assert not basis.is_fitted

    def test_n_basis_before_fit(self) -> None:
        assert TPRS(k=7).n_basis == 7


# ---------------------------------------------------------------------------
# TPRS.fit — 1D
# ---------------------------------------------------------------------------


class TestTPRSFit1D:
    def test_fit_returns_self(self) -> None:
        basis = TPRS(k=8)
        result = basis.fit(_make_1d())
        assert result is basis

    def test_is_fitted_after_fit(self) -> None:
        basis = TPRS(k=8).fit(_make_1d())
        assert basis.is_fitted

    def test_dimension_stored(self) -> None:
        basis = TPRS(k=8).fit(_make_1d())
        assert basis.d == 1

    def test_null_space_dim_1d(self) -> None:
        basis = TPRS(k=8).fit(_make_1d())
        assert basis.null_space_dimension() == 2  # M = d + 1 = 2

    def test_null_space_dim_property_matches_method(self) -> None:
        basis = TPRS(k=8).fit(_make_1d())
        assert basis.null_space_dim == basis.null_space_dimension()

    def test_eigenvalues_length(self) -> None:
        k = 8
        basis = TPRS(k=k).fit(_make_1d())
        assert len(basis.eigenvalues) == k - 2  # k - M = 6

    def test_eigenvalues_non_negative(self) -> None:
        basis = TPRS(k=8).fit(_make_1d())
        assert np.all(basis.eigenvalues >= -1e-10)

    def test_eigenvalues_descending(self) -> None:
        basis = TPRS(k=10).fit(_make_1d(n=60))
        ev = basis.eigenvalues
        assert np.all(np.diff(ev) <= 1e-10)

    def test_k_larger_than_n_raises(self) -> None:
        with pytest.raises(ValueError, match="smaller than k"):
            TPRS(k=20).fit(np.linspace(0, 1, 10))

    def test_k_too_small_for_dim_raises(self) -> None:
        # k=2 → r = k - M = 2 - 2 = 0 → no spline columns
        with pytest.raises(ValueError, match="too small for d"):
            TPRS(k=2).fit(_make_1d())


# ---------------------------------------------------------------------------
# TPRS.fit — 2D
# ---------------------------------------------------------------------------


class TestTPRSFit2D:
    def test_dimension_stored(self) -> None:
        basis = TPRS(k=10).fit(_make_2d())
        assert basis.d == 2

    def test_null_space_dim_2d(self) -> None:
        basis = TPRS(k=10).fit(_make_2d())
        assert basis.null_space_dimension() == 3  # M = d + 1 = 3

    def test_eigenvalues_length_2d(self) -> None:
        k = 10
        basis = TPRS(k=k).fit(_make_2d())
        assert len(basis.eigenvalues) == k - 3  # k - M = 7


# ---------------------------------------------------------------------------
# TPRS.basis_matrix
# ---------------------------------------------------------------------------


class TestTPRSBasisMatrix:
    def test_shape_at_training_points_1d(self) -> None:
        x = _make_1d(n=40)
        basis = TPRS(k=10).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (40, 10)

    def test_shape_at_new_points_1d(self) -> None:
        x_train = _make_1d(n=40)
        x_new = np.linspace(0.1, 0.9, 15)
        basis = TPRS(k=10).fit(x_train)
        B = basis.basis_matrix(x_new)
        assert B.shape == (15, 10)

    def test_shape_at_training_points_2d(self) -> None:
        x = _make_2d(n=40)
        basis = TPRS(k=10).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (40, 10)

    def test_first_column_is_ones_1d(self) -> None:
        x = _make_1d(n=30)
        basis = TPRS(k=8).fit(x)
        B = basis.basis_matrix(x)
        assert_allclose(B[:, 0], np.ones(30))

    def test_second_column_is_x_1d(self) -> None:
        x = _make_1d(n=30)
        basis = TPRS(k=8).fit(x)
        B = basis.basis_matrix(x)
        assert_allclose(B[:, 1], x)

    def test_first_column_is_ones_2d(self) -> None:
        x = _make_2d(n=30)
        basis = TPRS(k=10).fit(x)
        B = basis.basis_matrix(x)
        assert_allclose(B[:, 0], np.ones(30))

    def test_wrong_dimension_raises(self) -> None:
        x1d = _make_1d(n=30)
        basis = TPRS(k=8).fit(x1d)
        x2d = _make_2d(n=10)
        with pytest.raises(ValueError, match="covariate column"):
            basis.basis_matrix(x2d)

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            TPRS(k=8).basis_matrix(np.array([0.5]))

    def test_accepts_column_vector_1d(self) -> None:
        x = _make_1d(n=20)
        basis = TPRS(k=8).fit(x)
        B_flat = basis.basis_matrix(x)
        B_col = basis.basis_matrix(x[:, np.newaxis])
        assert_allclose(B_flat, B_col)

    def test_single_new_point(self) -> None:
        x_train = _make_1d(n=30)
        basis = TPRS(k=8).fit(x_train)
        B = basis.basis_matrix(np.array([0.5]))
        assert B.shape == (1, 8)


# ---------------------------------------------------------------------------
# TPRS.penalty_matrix
# ---------------------------------------------------------------------------


class TestTPRSPenaltyMatrix:
    def test_shape_1d(self) -> None:
        basis = TPRS(k=10).fit(_make_1d())
        S = basis.penalty_matrix()
        assert S.shape == (10, 10)

    def test_null_space_block_is_zero_1d(self) -> None:
        # The first M=2 rows and columns should be zero.
        basis = TPRS(k=10).fit(_make_1d())
        S = basis.penalty_matrix()
        M = basis.null_space_dimension()
        assert_allclose(S[:M, :], 0.0)
        assert_allclose(S[:, :M], 0.0)

    def test_penalised_block_is_diagonal_1d(self) -> None:
        basis = TPRS(k=10).fit(_make_1d())
        S = basis.penalty_matrix()
        M = basis.null_space_dimension()
        S_pen = S[M:, M:]
        # Off-diagonals should be zero.
        assert_allclose(S_pen - np.diag(np.diag(S_pen)), 0.0)

    def test_penalty_diagonal_matches_eigenvalues(self) -> None:
        basis = TPRS(k=10).fit(_make_1d())
        S = basis.penalty_matrix()
        M = basis.null_space_dimension()
        assert_allclose(np.diag(S[M:, M:]), basis.eigenvalues)

    def test_penalty_is_positive_semidefinite(self) -> None:
        basis = TPRS(k=10).fit(_make_1d())
        S = basis.penalty_matrix()
        eigenvalues = np.linalg.eigvalsh(S)
        assert np.all(eigenvalues >= -1e-10)

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            TPRS(k=8).penalty_matrix()

    def test_shape_2d(self) -> None:
        basis = TPRS(k=10).fit(_make_2d())
        S = basis.penalty_matrix()
        assert S.shape == (10, 10)

    def test_null_space_block_is_zero_2d(self) -> None:
        basis = TPRS(k=10).fit(_make_2d())
        S = basis.penalty_matrix()
        M = basis.null_space_dimension()  # = 3
        assert_allclose(S[:M, :], 0.0)
        assert_allclose(S[:, :M], 0.0)


# ---------------------------------------------------------------------------
# TPRS.identifiability_constraints
# ---------------------------------------------------------------------------


class TestTPRSIdentifiabilityConstraints:
    def test_shape(self) -> None:
        basis = TPRS(k=10).fit(_make_1d())
        C = basis.identifiability_constraints()
        assert C is not None
        assert C.shape == (1, 10)

    def test_mean_of_training_basis(self) -> None:
        x = _make_1d(n=40)
        basis = TPRS(k=10).fit(x)
        C = basis.identifiability_constraints()
        B = basis.basis_matrix(x)
        assert_allclose(C, B.mean(axis=0, keepdims=True))

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            TPRS(k=8).identifiability_constraints()


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------


class TestTPRSNumericalProperties:
    def test_basis_rank_equals_k(self) -> None:
        """The k columns of B should span a k-dimensional space."""
        x = _make_1d(n=80)
        k = 12
        basis = TPRS(k=k).fit(x)
        B = basis.basis_matrix(x)
        assert np.linalg.matrix_rank(B) == k

    def test_null_space_columns_interpolate_linearly_1d(self) -> None:
        """The null-space columns [1, x] should reproduce affine functions exactly."""
        x_train = _make_1d(n=50)
        basis = TPRS(k=10).fit(x_train)
        B = basis.basis_matrix(x_train)
        # Column 0 = 1, column 1 = x.  An affine function a + b*x should be
        # reconstructed perfectly using only the null-space columns.
        a, b = 2.3, -1.7
        f_exact = a + b * x_train
        f_approx = a * B[:, 0] + b * B[:, 1]
        assert_allclose(f_approx, f_exact)

    def test_prediction_consistency(self) -> None:
        """basis_matrix(x_train) should equal the prediction matrix at training points."""
        x_train = _make_1d(n=40)
        basis = TPRS(k=10).fit(x_train)
        B_train = basis.basis_matrix(x_train)
        # Evaluating at training points via the prediction path should give the same matrix.
        B_pred = basis.basis_matrix(x_train.copy())
        assert_allclose(B_train, B_pred)

    def test_smooth_interpolation_1d(self) -> None:
        """A TPRS fit should closely reconstruct a smooth function on training data."""
        rng = np.random.default_rng(7)
        n = 100
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.05, n)

        basis = TPRS(k=15).fit(x)
        B = basis.basis_matrix(x)
        S = basis.penalty_matrix()

        # Fit with mild smoothing (small lambda).
        lam = 1e-4
        BtB = B.T @ B
        pen = lam * S
        coef = np.linalg.solve(BtB + pen, B.T @ y)
        y_hat = B @ coef

        # Fitted values should be close to true signal.
        assert_allclose(y_hat, np.sin(x), atol=0.15)

    def test_smooth_interpolation_2d(self) -> None:
        """2D TPRS should reconstruct a smooth surface on training data."""
        rng = np.random.default_rng(13)
        n = 80
        x = rng.uniform(0, 1, (n, 2))
        y = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]) + rng.normal(0, 0.05, n)

        basis = TPRS(k=15).fit(x)
        B = basis.basis_matrix(x)
        S = basis.penalty_matrix()

        lam = 1e-3
        BtB = B.T @ B
        pen = lam * S
        coef = np.linalg.solve(BtB + pen, B.T @ y)
        y_hat = B @ coef

        y_true = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        assert_allclose(y_hat, y_true, atol=0.25)

    def test_penalty_null_space_matches_polynomial_columns(self) -> None:
        """Vectors in the null space of S should correspond to the polynomial columns."""
        x = _make_1d(n=50)
        basis = TPRS(k=10).fit(x)
        S = basis.penalty_matrix()
        M = basis.null_space_dimension()

        # Each of the M polynomial basis vectors should be in the null space of S.
        basis.basis_matrix(x)
        for j in range(M):
            Sv = S @ np.eye(10)[:, j]
            assert_allclose(Sv, 0.0, atol=1e-12)
