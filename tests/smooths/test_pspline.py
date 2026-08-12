"""Tests for whittaker.smooths.pspline (P-Splines)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.smooths.pspline import PSpline, _bspline_knots, _diff_matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(1)


def _make_x(n: int = 60) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# _bspline_knots
# ---------------------------------------------------------------------------


class TestBsplineKnots:
    def test_length(self) -> None:
        t = _bspline_knots(0.0, 1.0, k=10, degree=3)
        assert len(t) == 10 + 3 + 1  # k + degree + 1

    def test_sorted(self) -> None:
        t = _bspline_knots(0.0, 1.0, k=10, degree=3)
        assert np.all(np.diff(t) >= 0)

    def test_left_boundary_repeated(self) -> None:
        degree = 3
        t = _bspline_knots(0.0, 1.0, k=10, degree=degree)
        assert_allclose(t[: degree + 1], 0.0)

    def test_right_boundary_approximately_x_max(self) -> None:
        degree = 3
        t = _bspline_knots(0.0, 1.0, k=10, degree=degree)
        # Last degree+1 values should be close to x_max (one is nudged slightly)
        assert_allclose(t[-(degree + 1) : -1], 1.0)
        assert t[-1] > 1.0  # nudged

    def test_no_interior_knots_when_k_equals_degree_plus_one(self) -> None:
        # k = degree + 1 → no interior knots
        t = _bspline_knots(0.0, 1.0, k=4, degree=3)
        assert len(t) == 4 + 3 + 1

    def test_k_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="too small for degree"):
            _bspline_knots(0.0, 1.0, k=2, degree=3)


# ---------------------------------------------------------------------------
# _diff_matrix
# ---------------------------------------------------------------------------


class TestDiffMatrix:
    def test_shape_order1(self) -> None:
        D = _diff_matrix(k=6, m=1)
        assert D.shape == (5, 6)

    def test_shape_order2(self) -> None:
        D = _diff_matrix(k=6, m=2)
        assert D.shape == (4, 6)

    def test_first_order_entries(self) -> None:
        D = _diff_matrix(k=4, m=1)
        expected = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]], dtype=float)
        assert_allclose(D, expected)

    def test_second_order_entries(self) -> None:
        D = _diff_matrix(k=5, m=2)
        expected = np.array([[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]], dtype=float)
        assert_allclose(D, expected)

    def test_annihilates_constant_m1(self) -> None:
        D = _diff_matrix(k=8, m=1)
        assert_allclose(D @ np.ones(8), 0.0, atol=1e-14)

    def test_annihilates_constant_m2(self) -> None:
        D = _diff_matrix(k=8, m=2)
        assert_allclose(D @ np.ones(8), 0.0, atol=1e-14)

    def test_annihilates_linear_m2(self) -> None:
        D = _diff_matrix(k=8, m=2)
        assert_allclose(D @ np.arange(8, dtype=float), 0.0, atol=1e-12)

    def test_m_equals_k_raises(self) -> None:
        with pytest.raises(ValueError, match="must be less than k"):
            _diff_matrix(k=5, m=5)


# ---------------------------------------------------------------------------
# PSpline constructor
# ---------------------------------------------------------------------------


class TestPSplineConstructor:
    def test_defaults(self) -> None:
        ps = PSpline()
        assert ps.k == 10
        assert ps.degree == 3
        assert ps.m == 2

    def test_custom_params(self) -> None:
        ps = PSpline(k=15, degree=2, m=3)
        assert ps.k == 15
        assert ps.degree == 2
        assert ps.m == 3

    def test_k_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            PSpline(k=1)

    def test_degree_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="degree must be at least 1"):
            PSpline(degree=0)

    def test_m_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PSpline(m=0)

    def test_m_ge_k_raises(self) -> None:
        with pytest.raises(ValueError, match="must be less than k"):
            PSpline(k=5, m=5)

    def test_not_fitted_initially(self) -> None:
        assert not PSpline().is_fitted

    def test_n_basis_before_fit(self) -> None:
        assert PSpline(k=12).n_basis == 12


# ---------------------------------------------------------------------------
# PSpline.fit
# ---------------------------------------------------------------------------


class TestPSplineFit:
    def test_fit_returns_self(self) -> None:
        ps = PSpline(k=8)
        assert ps.fit(_make_x()) is ps

    def test_is_fitted_after_fit(self) -> None:
        assert PSpline(k=8).fit(_make_x()).is_fitted

    def test_knot_vector_length(self) -> None:
        ps = PSpline(k=10, degree=3).fit(_make_x())
        assert len(ps.knots) == 10 + 3 + 1

    def test_interior_knots_count(self) -> None:
        ps = PSpline(k=10, degree=3).fit(_make_x())
        assert len(ps.interior_knots) == 10 - 3 - 1  # = 6

    def test_interior_knots_empty_when_k_equals_degree_plus_one(self) -> None:
        ps = PSpline(k=4, degree=3, m=1).fit(_make_x())
        assert len(ps.interior_knots) == 0

    def test_n_smaller_than_k_raises(self) -> None:
        with pytest.raises(ValueError, match="smaller than k"):
            PSpline(k=20).fit(np.linspace(0, 1, 10))

    def test_constant_x_raises(self) -> None:
        with pytest.raises(ValueError, match="identical"):
            PSpline(k=5, m=1).fit(np.ones(10))

    def test_m_mutated_after_construction_raises_on_fit(self) -> None:
        # The constructor validates m < k, but fit() re-checks it in case the
        # attribute is mutated afterwards (e.g. by user code or deserialization).
        ps = PSpline(k=5, m=2)
        ps.m = 5
        with pytest.raises(ValueError, match="must be less than k"):
            ps.fit(_make_x())

    def test_column_vector_accepted(self) -> None:
        x = _make_x(30)[:, np.newaxis]
        assert PSpline(k=8).fit(x).is_fitted

    def test_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="univariate"):
            PSpline(k=8).fit(np.ones((20, 2)))

    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_fit_various_degrees(self, degree: int) -> None:
        ps = PSpline(k=max(degree + 2, 5), degree=degree, m=1).fit(_make_x())
        assert ps.is_fitted

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_fit_various_penalty_orders(self, m: int) -> None:
        ps = PSpline(k=10, m=m).fit(_make_x())
        assert ps.is_fitted


# ---------------------------------------------------------------------------
# PSpline.basis_matrix
# ---------------------------------------------------------------------------


class TestPSplineBasisMatrix:
    def test_shape_at_training_points(self) -> None:
        x = _make_x(50)
        B = PSpline(k=10).fit(x).basis_matrix(x)
        assert B.shape == (50, 10)

    def test_shape_at_new_points(self) -> None:
        x_train = _make_x(50)
        x_new = np.linspace(0.2, 0.8, 25)
        B = PSpline(k=10).fit(x_train).basis_matrix(x_new)
        assert B.shape == (25, 10)

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            PSpline(k=8).basis_matrix(np.array([0.5]))

    def test_rows_nonnegative(self) -> None:
        # B-spline basis functions are non-negative everywhere.
        x = _make_x(50)
        B = PSpline(k=10).fit(x).basis_matrix(x)
        assert np.all(B >= -1e-12)

    def test_rows_sum_to_one(self) -> None:
        # Partition of unity: B-spline rows sum to 1 everywhere.
        x = _make_x(50)
        B = PSpline(k=10).fit(x).basis_matrix(x)
        assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)

    def test_partition_of_unity_extrapolation(self) -> None:
        x_train = _make_x(50)
        ps = PSpline(k=10).fit(x_train)
        x_ext = np.array([-0.5, 1.5])
        B = ps.basis_matrix(x_ext)
        assert_allclose(B.sum(axis=1), 1.0, atol=1e-10)

    def test_single_point(self) -> None:
        ps = PSpline(k=8).fit(_make_x())
        assert ps.basis_matrix(np.array([0.5])).shape == (1, 8)

    def test_column_vector_accepted(self) -> None:
        x = _make_x(40)
        ps = PSpline(k=8).fit(x)
        B1 = ps.basis_matrix(x)
        B2 = ps.basis_matrix(x[:, np.newaxis])
        assert_allclose(B1, B2)

    def test_multidim_raises(self) -> None:
        ps = PSpline(k=8).fit(_make_x())
        with pytest.raises(ValueError, match="univariate"):
            ps.basis_matrix(np.ones((10, 2)))

    def test_linear_degree1_reproduces_identity(self) -> None:
        # Degree-1 B-splines: the function f(x) = x on [0,1] should be
        # exactly reproducible within the training range.
        x = _make_x(80)
        ps = PSpline(k=10, degree=1, m=1).fit(x)
        B = ps.basis_matrix(x)
        # Fit a linear function via regression (no penalty).
        np.linspace(0.0, 1.0, 10)
        coef, *_ = np.linalg.lstsq(B, x, rcond=None)
        assert_allclose(B @ coef, x, atol=1e-6)


# ---------------------------------------------------------------------------
# PSpline.penalty_matrix
# ---------------------------------------------------------------------------


class TestPSplinePenaltyMatrix:
    def test_shape(self) -> None:
        S = PSpline(k=10).fit(_make_x()).penalty_matrix()
        assert S.shape == (10, 10)

    def test_symmetry(self) -> None:
        S = PSpline(k=10).fit(_make_x()).penalty_matrix()
        assert_allclose(S, S.T)

    def test_positive_semidefinite(self) -> None:
        S = PSpline(k=10).fit(_make_x()).penalty_matrix()
        eigs = np.linalg.eigvalsh(S)
        assert np.all(eigs >= -1e-10)

    def test_rank_is_k_minus_m(self) -> None:
        for m in [1, 2, 3]:
            k = 10
            S = PSpline(k=k, m=m).fit(_make_x()).penalty_matrix()
            assert np.linalg.matrix_rank(S, tol=1e-8) == k - m

    def test_null_space_constant(self) -> None:
        S = PSpline(k=10, m=2).fit(_make_x()).penalty_matrix()
        assert_allclose(S @ np.ones(10), 0.0, atol=1e-10)

    def test_null_space_linear_index(self) -> None:
        S = PSpline(k=10, m=2).fit(_make_x()).penalty_matrix()
        v = np.arange(10, dtype=float)
        assert_allclose(S @ v, 0.0, atol=1e-10)

    def test_null_space_m1_only_constant(self) -> None:
        # For m=1, only the constant is in the null space.
        S = PSpline(k=8, m=1).fit(_make_x()).penalty_matrix()
        assert_allclose(S @ np.ones(8), 0.0, atol=1e-10)
        # A linear sequence should NOT be in the null space.
        v = np.arange(8, dtype=float)
        assert np.linalg.norm(S @ v) > 1e-8

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            PSpline(k=8).penalty_matrix()

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_penalty_order_m(self, m: int) -> None:
        k = 12
        S = PSpline(k=k, m=m).fit(_make_x()).penalty_matrix()
        # Null space should have dimension m.
        assert np.linalg.matrix_rank(S, tol=1e-8) == k - m


# ---------------------------------------------------------------------------
# PSpline.null_space_dimension
# ---------------------------------------------------------------------------


class TestPSplineNullSpaceDimension:
    @pytest.mark.parametrize("m", [1, 2, 3, 4])
    def test_equals_m(self, m: int) -> None:
        assert PSpline(k=10, m=m).null_space_dimension() == m


# ---------------------------------------------------------------------------
# PSpline.identifiability_constraints
# ---------------------------------------------------------------------------


class TestPSplineIdentifiabilityConstraints:
    def test_shape(self) -> None:
        C = PSpline(k=10).fit(_make_x()).identifiability_constraints()
        assert C is not None
        assert C.shape == (1, 10)

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fitted"):
            PSpline(k=8).identifiability_constraints()

    def test_values_are_basis_means(self) -> None:
        x = _make_x(60)
        ps = PSpline(k=10).fit(x)
        C = ps.identifiability_constraints()
        assert C is not None
        # The constraint rows should sum to 1 (partition of unity).
        assert_allclose(C.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Numerical / smoothing properties
# ---------------------------------------------------------------------------


class TestPSplineNumerical:
    def test_basis_rank_equals_k(self) -> None:
        x = _make_x(80)
        k = 12
        B = PSpline(k=k).fit(x).basis_matrix(x)
        assert np.linalg.matrix_rank(B) == k

    def test_smooth_fit_1d(self) -> None:
        rng = np.random.default_rng(5)
        x = np.linspace(0.0, 2 * np.pi, 100)
        y = np.sin(x) + rng.normal(0, 0.05, 100)

        ps = PSpline(k=15).fit(x)
        B = ps.basis_matrix(x)
        S = ps.penalty_matrix()

        lam = 1e-3
        coef = np.linalg.solve(B.T @ B + lam * S, B.T @ y)
        y_hat = B @ coef

        assert_allclose(y_hat, np.sin(x), atol=0.15)

    def test_penalty_quadratic_form_nonneg(self) -> None:
        ps = PSpline(k=10).fit(_make_x())
        S = ps.penalty_matrix()
        for _ in range(20):
            beta = RNG.standard_normal(10)
            assert beta @ S @ beta >= -1e-12

    def test_constant_coefficient_zero_penalty(self) -> None:
        ps = PSpline(k=10, m=2).fit(_make_x())
        S = ps.penalty_matrix()
        beta = np.ones(10)
        assert_allclose(beta @ S @ beta, 0.0, atol=1e-10)

    def test_linear_coefficient_zero_penalty(self) -> None:
        ps = PSpline(k=10, m=2).fit(_make_x())
        S = ps.penalty_matrix()
        beta = np.arange(10, dtype=float)
        assert_allclose(beta @ S @ beta, 0.0, atol=1e-8)

    def test_curved_coefficient_positive_penalty(self) -> None:
        ps = PSpline(k=10, m=2).fit(_make_x())
        S = ps.penalty_matrix()
        beta = np.sin(np.linspace(0, np.pi, 10))
        assert beta @ S @ beta > 1e-6

    def test_prediction_consistency(self) -> None:
        x = _make_x(50)
        ps = PSpline(k=10).fit(x)
        B1 = ps.basis_matrix(x)
        B2 = ps.basis_matrix(x.copy())
        assert_allclose(B1, B2)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_various_degrees_fit_and_evaluate(self, degree: int) -> None:
        x = _make_x(60)
        ps = PSpline(k=10, degree=degree, m=min(2, degree)).fit(x)
        B = ps.basis_matrix(np.linspace(0.1, 0.9, 20))
        assert B.shape == (20, 10)
        assert np.all(B >= -1e-12)

    def test_more_knots_gives_better_fit(self) -> None:
        rng = np.random.default_rng(9)
        x = np.linspace(0, 1, 200)
        y = np.sin(4 * np.pi * x) + rng.normal(0, 0.02, 200)

        errors = []
        for k in [5, 15, 30]:
            ps = PSpline(k=k).fit(x)
            B = ps.basis_matrix(x)
            S = ps.penalty_matrix()
            coef = np.linalg.solve(B.T @ B + 1e-4 * S, B.T @ y)
            residuals = y - B @ coef
            errors.append(np.mean(residuals**2))

        # More basis functions → lower training error with mild penalty.
        assert errors[0] > errors[1] > errors[2]
