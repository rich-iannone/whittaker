"""Tests for Markov random field smooth basis."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.formula.terms import Formula, SmoothTerm
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix
from whittaker.smooths.mrf import MRFBasis


def _triangle_neighborhood():
    return {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}


def _line_neighborhood():
    return {
        "R1": ["R2"],
        "R2": ["R1", "R3"],
        "R3": ["R2", "R4"],
        "R4": ["R3", "R5"],
        "R5": ["R4"],
    }


def _spatial_data(n=300, seed=23):
    rng = np.random.default_rng(seed)
    regions = ["R1", "R2", "R3", "R4", "R5"]
    region_effects = {"R1": -2.0, "R2": -1.0, "R3": 0.0, "R4": 1.0, "R5": 2.0}
    region_labels = rng.choice(regions, n)
    y = np.array([region_effects[r] for r in region_labels]) + rng.normal(0, 0.5, n)
    return {"region": region_labels, "y": y}


class TestMRFBasis:
    def test_fit_discovers_levels(self):
        nb = _triangle_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        x = np.array(["A", "B", "C", "A", "B"])
        basis.fit(x)
        assert basis.n_basis == 3
        assert set(basis.levels) == {"A", "B", "C"}

    def test_basis_matrix_is_one_hot(self):
        nb = _triangle_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        x = np.array(["A", "B", "C", "A", "C"])
        basis.fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (5, 3)
        np.testing.assert_array_equal(B.sum(axis=1), 1.0)
        for row in B:
            assert np.count_nonzero(row) == 1

    def test_penalty_is_laplacian(self):
        nb = _triangle_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        x = np.array(["A", "B", "C"])
        basis.fit(x)
        L = basis.penalty_matrix()
        assert L.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(L), [2, 2, 2])
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert L[i, j] == -1.0

    def test_penalty_symmetric_psd(self):
        nb = _line_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        regions = list(nb.keys())
        x = np.array(regions)
        basis.fit(x)
        L = basis.penalty_matrix()
        np.testing.assert_allclose(L, L.T)
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)

    def test_null_space_dimension_connected(self):
        nb = _triangle_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        x = np.array(["A", "B", "C"])
        basis.fit(x)
        assert basis.null_space_dimension() == 1

    def test_null_space_dimension_disconnected(self):
        nb = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
        basis = MRFBasis(neighborhood=nb)
        x = np.array(["A", "B", "C", "D"])
        basis.fit(x)
        assert basis.null_space_dimension() == 2

    def test_identifiability_constraint(self):
        nb = _line_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        x = np.array(list(nb.keys()))
        basis.fit(x)
        C = basis.identifiability_constraints()
        assert C.shape == (1, 5)
        np.testing.assert_allclose(C, np.ones((1, 5)) / 5)

    def test_unseen_level_gets_zero_row(self):
        nb = _triangle_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        x = np.array(["A", "B", "C"])
        basis.fit(x)
        B = basis.basis_matrix(np.array(["A", "D", "B"]))
        assert B.shape == (3, 3)
        np.testing.assert_array_equal(B[1], [0, 0, 0])

    def test_k_limits_levels(self):
        nb = _line_neighborhood()
        basis = MRFBasis(k=3, neighborhood=nb)
        x = np.array(list(nb.keys()) * 10)
        basis.fit(x)
        assert basis.n_basis == 3

    def test_missing_neighborhood_raises(self):
        basis = MRFBasis()
        x = np.array(["A", "B", "C"])
        with pytest.raises(ValueError, match="neighborhood"):
            basis.fit(x)

    def test_adjacency_matrix_input(self):
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        basis = MRFBasis(neighborhood=A)
        x = np.array(["A", "B", "C"])
        basis.fit(x)
        L = basis.penalty_matrix()
        assert L.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(L), [2, 2, 2])

    def test_line_laplacian(self):
        nb = _line_neighborhood()
        basis = MRFBasis(neighborhood=nb)
        x = np.array(list(nb.keys()))
        basis.fit(x)
        L = basis.penalty_matrix()
        expected_diag = [1, 2, 2, 2, 1]
        np.testing.assert_array_equal(np.diag(L), expected_diag)


class TestMRFModelMatrix:
    def test_builds_with_mrf(self):
        nb = _line_neighborhood()
        data = _spatial_data()
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        mm = build_model_matrix(formula, data)
        assert mm.X.shape[0] == 300
        assert mm.X.shape[1] > 1
        assert len(mm.penalties) >= 1

    def test_penalty_block_shape(self):
        nb = _line_neighborhood()
        data = _spatial_data()
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        mm = build_model_matrix(formula, data)
        assert mm.penalties[0].shape[0] == mm.X.shape[1]


class TestMRFGAM:
    def test_fit_gaussian(self):
        nb = _line_neighborhood()
        data = _spatial_data(n=500)
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        model = GAM(formula)
        model.fit(data)
        assert model.is_fitted

    def test_spatial_pattern_recovered(self):
        nb = _line_neighborhood()
        data = _spatial_data(n=1000, seed=23)
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        model = GAM(formula)
        model.fit(data)
        assert model.deviance < model.null_deviance

    def test_predict(self):
        nb = _line_neighborhood()
        data = _spatial_data(n=300)
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        model = GAM(formula)
        model.fit(data)
        pred = model.predict(data)
        assert pred.values.shape == (300,)
        assert np.all(np.isfinite(pred.values))

    def test_predict_with_se(self):
        nb = _line_neighborhood()
        data = _spatial_data(n=300)
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        model = GAM(formula)
        model.fit(data)
        pred = model.predict(data, se=True)
        assert pred.se is not None
        assert np.all(pred.se >= 0)

    def test_with_other_smooths(self):
        nb = _line_neighborhood()
        rng = np.random.default_rng(23)
        n = 500
        regions = rng.choice(list(nb.keys()), n)
        x = np.linspace(0, 2 * np.pi, n)
        region_effects = {"R1": -2.0, "R2": -1.0, "R3": 0.0, "R4": 1.0, "R5": 2.0}
        y = np.sin(x) + np.array([region_effects[r] for r in regions]) + rng.normal(0, 0.3, n)
        data = {"x": x, "region": regions, "y": y}

        mrf_term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        smooth_term = SmoothTerm(
            variables=("x",),
            smooth_type="s",
            bs="tp",
            k=10,
            extra={},
        )
        formula = Formula(response="y", terms=[smooth_term, mrf_term], intercept=True)
        model = GAM(formula)
        model.fit(data)
        assert model.is_fitted
        assert model.deviance < model.null_deviance

    def test_reml(self):
        nb = _line_neighborhood()
        data = _spatial_data(n=300)
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        model = GAM(formula)
        model.fit(data, method="REML")
        assert model.is_fitted

    def test_summary_contains_mrf(self):
        nb = _line_neighborhood()
        data = _spatial_data(n=300)
        term = SmoothTerm(
            variables=("region",),
            smooth_type="s",
            bs="mrf",
            k=-1,
            extra={"xt": {"neighborhood": nb}},
        )
        formula = Formula(response="y", terms=[term], intercept=True)
        model = GAM(formula)
        model.fit(data)
        s = model.summary()
        assert "region" in s
