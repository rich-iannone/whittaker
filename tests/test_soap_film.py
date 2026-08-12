"""Tests for the soap film smooth (bs='so')."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.formula.terms import Formula, SmoothTerm
from whittaker.gam import GAM
from whittaker.smooths.soap_film import (
    SoapFilm,
    _boundary_distance,
    _point_in_polygon,
    _points_in_domain,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square_boundary():
    return [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)]


def _square_with_hole():
    outer = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    hole = np.array([[0.3, 0.3], [0.3, 0.7], [0.7, 0.7], [0.7, 0.3]], dtype=float)
    return [outer, hole]


def _simple_2d_data(n=300, seed=23):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.05, 0.95, n)
    x2 = rng.uniform(0.05, 0.95, n)
    y = np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2) + rng.normal(0, 0.1, n)
    return x1, x2, y


# ---------------------------------------------------------------------------
# Point-in-polygon tests
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    def test_inside(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert _point_in_polygon(0.5, 0.5, poly) is True

    def test_outside(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert _point_in_polygon(2.0, 2.0, poly) is False

    def test_domain_with_hole(self):
        bnd = _square_with_hole()
        assert _points_in_domain(np.array([[0.1, 0.1]]), bnd)[0] is np.True_
        assert _points_in_domain(np.array([[0.5, 0.5]]), bnd)[0] is np.False_

    def test_domain_point_outside_outer_boundary(self):
        # Point clearly outside the outer boundary polygon; exercises the
        # `continue` branch in `_points_in_domain` before hole-checking happens.
        bnd = _square_with_hole()
        mask = _points_in_domain(np.array([[2.0, 2.0]]), bnd)
        assert mask[0] is np.False_


class TestBoundaryDistance:
    def test_distance_outside_square(self):
        square = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)]
        d = _boundary_distance(np.array([2.0, 0.5]), square)
        assert np.isclose(d, 1.0)

    def test_distance_inside_square(self):
        square = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)]
        d = _boundary_distance(np.array([0.5, 0.5]), square)
        assert np.isclose(d, 0.5)

    def test_distance_on_boundary_is_zero(self):
        square = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)]
        d = _boundary_distance(np.array([0.0, 0.5]), square)
        assert np.isclose(d, 0.0)


# ---------------------------------------------------------------------------
# SoapFilm basis tests
# ---------------------------------------------------------------------------


class TestSoapFilmBasis:
    def test_fit_and_basis_shapes(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0.05, 0.95, (200, 2))
        knots = np.array(
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
                [0.5, 0.5],
            ]
        )
        basis = SoapFilm(boundary=_square_boundary(), knots=knots, k=5)
        basis.fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (200, 5)
        assert np.all(np.isfinite(B))

    def test_penalty_shape(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0.05, 0.95, (200, 2))
        knots = np.array(
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
                [0.5, 0.5],
            ]
        )
        basis = SoapFilm(boundary=_square_boundary(), knots=knots, k=5)
        basis.fit(x)
        S = basis.penalty_matrix()
        assert S.shape == (5, 5)
        assert np.allclose(S, S.T)

    def test_null_space_dimension(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0.05, 0.95, (200, 2))
        basis = SoapFilm(boundary=_square_boundary(), k=10)
        basis.fit(x)
        assert basis.null_space_dimension() == 0

    def test_auto_boundary(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, (200, 2))
        basis = SoapFilm(k=10)
        basis.fit(x)
        assert basis.n_basis > 0
        B = basis.basis_matrix(x)
        assert B.shape[0] == 200
        assert B.shape[1] == basis.n_basis

    def test_auto_knots(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0.05, 0.95, (200, 2))
        basis = SoapFilm(boundary=_square_boundary(), k=15)
        basis.fit(x)
        assert basis.n_basis <= 15

    def test_requires_2d(self):
        x = np.linspace(0, 1, 100)
        basis = SoapFilm(k=5)
        with pytest.raises(ValueError, match="2 covariates"):
            basis.fit(x)

    def test_is_fitted(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, (100, 2))
        basis = SoapFilm(k=5)
        assert not basis.is_fitted
        basis.fit(x)
        assert basis.is_fitted

    def test_repr(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0.05, 0.95, (100, 2))
        basis = SoapFilm(boundary=_square_boundary(), k=6)
        basis.fit(x)
        r = repr(basis)
        assert "SoapFilm" in r
        assert str(basis.n_basis) in r

    def test_basis_matrix_point_outside_triangulation(self):
        # A point far outside the fitted domain should fall back to a nearest-knot
        # one-hot indicator (find_simplex returns -1 for it).
        rng = np.random.default_rng(23)
        x = rng.uniform(0.05, 0.95, (100, 2))
        basis = SoapFilm(boundary=_square_boundary(), k=8)
        basis.fit(x)
        far = np.array([[10.0, 10.0]])
        B = basis.basis_matrix(far)
        assert B.shape == (1, basis.n_basis)
        assert np.count_nonzero(B[0]) == 1
        assert np.isclose(B[0].sum(), 1.0)
        # Nearest knot to (10, 10) should be the knot with the largest coordinates.
        nearest = np.argmin(np.linalg.norm(basis._knots - far[0], axis=1))
        assert B[0, nearest] == 1.0

    def test_duplicate_boundary_points_deduplicated(self):
        # A boundary loop with a repeated vertex introduces a zero-length segment
        # whose start point and midpoint coincide with an existing sample point,
        # which should be filtered out as a duplicate after rounding.
        outer = np.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=float,
        )
        knots = np.array(
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
                [0.5, 0.5],
            ]
        )
        rng = np.random.default_rng(23)
        x = rng.uniform(0.05, 0.95, (100, 2))
        basis = SoapFilm(boundary=[outer], knots=knots, k=5)
        basis.fit(x)
        n_boundary_pts = basis._all_pts.shape[0] - basis._n_interior
        # 5 edges * 2 sample points each = 10 raw samples, minus at least one
        # duplicate introduced by the repeated vertex.
        assert n_boundary_pts < 10
        B = basis.basis_matrix(x)
        assert np.all(np.isfinite(B))


# ---------------------------------------------------------------------------
# GAM integration tests
# ---------------------------------------------------------------------------


def _soap_formula(bnd, knots):
    term = SmoothTerm(
        variables=("x1", "x2"),
        bs="so",
        extra={"xt": {"boundary": bnd, "knots": knots}},
    )
    return Formula(response="y", terms=[term])


class TestSoapFilmGAM:
    def test_converges(self):
        x1, x2, y = _simple_2d_data(n=400)
        bnd = _square_boundary()
        knots = np.column_stack(
            [
                np.repeat([0.25, 0.5, 0.75], 3),
                np.tile([0.25, 0.5, 0.75], 3),
            ]
        )
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM(_soap_formula(bnd, knots))
        model.fit(data)
        assert model.is_fitted

    def test_predict(self):
        x1, x2, y = _simple_2d_data(n=400)
        bnd = _square_boundary()
        knots = np.column_stack(
            [
                np.repeat([0.25, 0.5, 0.75], 3),
                np.tile([0.25, 0.5, 0.75], 3),
            ]
        )
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM(_soap_formula(bnd, knots))
        model.fit(data)
        pred = model.predict(data)
        assert pred.values.shape == (400,)
        assert np.all(np.isfinite(pred.values))

    def test_summary(self):
        x1, x2, y = _simple_2d_data(n=300)
        bnd = _square_boundary()
        knots = np.column_stack(
            [
                np.repeat([0.25, 0.5, 0.75], 3),
                np.tile([0.25, 0.5, 0.75], 3),
            ]
        )
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM(_soap_formula(bnd, knots))
        model.fit(data)
        s = model.summary()
        assert "s(x1, x2" in s

    def test_prediction_reasonable(self):
        x1, x2, y = _simple_2d_data(n=500)
        bnd = _square_boundary()
        knots = np.column_stack(
            [
                np.repeat([0.2, 0.4, 0.6, 0.8], 4),
                np.tile([0.2, 0.4, 0.6, 0.8], 4),
            ]
        )
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM(_soap_formula(bnd, knots))
        model.fit(data)
        pred = model.predict(data).values
        assert np.corrcoef(y, pred)[0, 1] > 0.3
