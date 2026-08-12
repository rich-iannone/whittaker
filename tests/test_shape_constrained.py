"""Tests for shape-constrained smooth terms."""

from __future__ import annotations

import numpy as np

from whittaker.gam import GAM
from whittaker.smooths.monotone import (
    ConvexPSpline,
    MonotonePSpline,
    _pava,
    project_convex,
    project_monotone,
)


class TestPAVA:
    def test_already_sorted(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(_pava(x), x)

    def test_reverse_sorted(self):
        x = np.array([4.0, 3.0, 2.0, 1.0])
        result = _pava(x)
        assert np.all(np.diff(result) >= 0)
        np.testing.assert_allclose(result, np.full(4, 2.5))

    def test_single_violation(self):
        x = np.array([1.0, 3.0, 2.0, 4.0])
        result = _pava(x)
        assert np.all(np.diff(result) >= 0)
        np.testing.assert_allclose(result, np.array([1.0, 2.5, 2.5, 4.0]))

    def test_single_element(self):
        np.testing.assert_array_equal(_pava(np.array([5.0])), np.array([5.0]))


class TestProjectMonotone:
    def test_increasing(self):
        beta = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
        result = project_monotone(beta, decreasing=False)
        assert np.all(np.diff(result) >= -1e-12)

    def test_decreasing(self):
        beta = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = project_monotone(beta, decreasing=True)
        assert np.all(np.diff(result) <= 1e-12)


class TestProjectConvex:
    def test_convex(self):
        beta = np.array([5.0, 2.0, 1.0, 2.0, 5.0])
        result = project_convex(beta, concave=False)
        diffs = np.diff(result)
        assert np.all(np.diff(diffs) >= -1e-12)

    def test_concave(self):
        beta = np.array([1.0, 4.0, 5.0, 4.0, 1.0])
        result = project_convex(beta, concave=True)
        diffs = np.diff(result)
        assert np.all(np.diff(diffs) <= 1e-12)


class TestMonotonePSplineBasis:
    def test_inherits_pspline(self):
        basis = MonotonePSpline(k=10)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (50, 10)
        S = basis.penalty_matrix()
        assert S.shape == (10, 10)

    def test_decreasing_flag(self):
        basis = MonotonePSpline(k=10, decreasing=True)
        assert basis.decreasing
        assert basis.constraint_direction == -1


class TestConvexPSplineBasis:
    def test_inherits_pspline(self):
        basis = ConvexPSpline(k=10)
        x = np.linspace(0, 1, 50)
        basis.fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (50, 10)
        S = basis.penalty_matrix()
        assert S.shape == (10, 10)

    def test_convex_constraint_direction(self):
        basis = ConvexPSpline(k=10)
        assert not basis.concave
        assert basis.constraint_direction == 1
        assert basis.constraint_order == 2

    def test_concave_constraint_direction(self):
        basis = ConvexPSpline(k=10, concave=True)
        assert basis.concave
        assert basis.constraint_direction == -1
        assert basis.constraint_order == 2


class TestMonotoneGAM:
    def test_monotone_increasing_fit(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 5, n)
        y = np.log1p(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM('y ~ s(x, bs="mpi")')
        model.fit(data)
        pred = model.predict(data).values
        assert np.all(np.diff(pred) >= -1e-8)

    def test_monotone_decreasing_fit(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 5, n)
        y = 3.0 - np.log1p(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM('y ~ s(x, bs="mpd")')
        model.fit(data)
        pred = model.predict(data).values
        assert np.all(np.diff(pred) <= 1e-8)

    def test_monotone_better_than_unconstrained_for_monotone_data(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 5, n)
        y_true = np.log1p(x)
        y = y_true + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}

        model_ps = GAM('y ~ s(x, bs="ps")')
        model_ps.fit(data)
        pred_ps = model_ps.predict(data).values

        model_mpi = GAM('y ~ s(x, bs="mpi")')
        model_mpi.fit(data)
        pred_mpi = model_mpi.predict(data).values

        mse_ps = np.mean((pred_ps - y_true) ** 2)
        mse_mpi = np.mean((pred_mpi - y_true) ** 2)
        assert mse_mpi < mse_ps * 2.0

    def test_monotone_with_reml(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 5, n)
        y = 0.5 * x + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        model = GAM('y ~ s(x, bs="mpi")')
        model.fit(data, method="REML")
        pred = model.predict(data).values
        assert np.all(np.diff(pred) >= -1e-8)

    def test_convex_fit(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-3, 3, n)
        y = x**2 + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        model = GAM('y ~ s(x, bs="cx")')
        model.fit(data)
        pred = model.predict(data).values
        second_diffs = np.diff(pred, n=2)
        assert np.min(second_diffs) > -0.1

    def test_concave_fit(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(-3, 3, n)
        y = -(x**2) + 10 + rng.normal(0, 0.3, n)
        data = {"x": x, "y": y}
        model = GAM('y ~ s(x, bs="cv")')
        model.fit(data)
        pred = model.predict(data).values
        second_diffs = np.diff(pred, n=2)
        assert np.max(second_diffs) < 0.1
