"""Tests for multi-dimensional TPRS (d >= 2)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families import Gaussian, Poisson
from whittaker.gam import GAM
from whittaker.smooths.tprs import TPRS, _null_space_dimension, _polynomial_null_space

RNG = np.random.default_rng(23)

# ---------------------------------------------------------------------------
# Null-space dimension helper
# ---------------------------------------------------------------------------


class TestNullSpaceDimension:
    def test_m2_d1(self):
        assert _null_space_dimension(d=1, m=2) == 2

    def test_m2_d2(self):
        assert _null_space_dimension(d=2, m=2) == 3

    def test_m2_d3(self):
        assert _null_space_dimension(d=3, m=2) == 4

    def test_m3_d1(self):
        # C(2+1, 1) = 3: [1, x, x²]
        assert _null_space_dimension(d=1, m=3) == 3

    def test_m3_d2(self):
        # C(2+2, 2) = 6: [1, x1, x2, x1², x1x2, x2²]
        assert _null_space_dimension(d=2, m=3) == 6

    def test_m3_d4(self):
        # C(2+4, 4) = 15
        assert _null_space_dimension(d=4, m=3) == 15

    def test_m4_d2(self):
        # C(3+2, 2) = 10
        assert _null_space_dimension(d=2, m=4) == 10


# ---------------------------------------------------------------------------
# Polynomial null space
# ---------------------------------------------------------------------------


class TestPolynomialNullSpaceHigherOrder:
    def test_m3_d1(self):
        x = RNG.standard_normal((30, 1))
        T = _polynomial_null_space(x, m=3)
        assert T.shape == (30, 3)
        assert_allclose(T[:, 0], 1.0)
        assert_allclose(T[:, 1], x[:, 0])
        assert_allclose(T[:, 2], x[:, 0] ** 2)

    def test_m3_d3(self):
        x = RNG.standard_normal((30, 3))
        T = _polynomial_null_space(x, m=3)
        # M = C(2+3, 3) = 10
        assert T.shape == (30, 10)

    def test_m4_d1(self):
        x = RNG.standard_normal((30, 1))
        T = _polynomial_null_space(x, m=4)
        # [1, x, x², x³]
        assert T.shape == (30, 4)
        assert_allclose(T[:, 3], x[:, 0] ** 3)


# ---------------------------------------------------------------------------
# TPRS basis for d=2 (m=2)
# ---------------------------------------------------------------------------


class TestTPRSD2:
    def test_fit_and_basis(self):
        x = RNG.uniform(0, 1, (100, 2))
        basis = TPRS(k=15).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (100, 15)

    def test_null_space_dim(self):
        x = RNG.uniform(0, 1, (100, 2))
        basis = TPRS(k=15).fit(x)
        assert basis.null_space_dimension() == 3

    def test_penalty_shape(self):
        x = RNG.uniform(0, 1, (100, 2))
        basis = TPRS(k=15).fit(x)
        S = basis.penalty_matrix()
        assert S.shape == (15, 15)
        assert_allclose(S[:3, :3], 0.0)

    def test_predict_new_data(self):
        x_train = RNG.uniform(0, 1, (100, 2))
        x_new = RNG.uniform(0, 1, (50, 2))
        basis = TPRS(k=15).fit(x_train)
        B = basis.basis_matrix(x_new)
        assert B.shape == (50, 15)

    def test_identifiability_constraints(self):
        x = RNG.uniform(0, 1, (100, 2))
        basis = TPRS(k=15).fit(x)
        C = basis.identifiability_constraints()
        assert C.shape == (1, 15)

    def test_eigenvalues_positive(self):
        x = RNG.uniform(0, 1, (100, 2))
        basis = TPRS(k=15).fit(x)
        ev = basis.eigenvalues
        assert len(ev) == 12
        assert np.all(ev > 0)


# ---------------------------------------------------------------------------
# TPRS basis for d=3 (m=2)
# ---------------------------------------------------------------------------


class TestTPRSD3:
    def test_fit_and_basis(self):
        x = RNG.uniform(0, 1, (150, 3))
        basis = TPRS(k=20).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (150, 20)

    def test_null_space_dim(self):
        x = RNG.uniform(0, 1, (150, 3))
        basis = TPRS(k=20).fit(x)
        assert basis.null_space_dimension() == 4

    def test_penalty_structure(self):
        x = RNG.uniform(0, 1, (150, 3))
        basis = TPRS(k=20).fit(x)
        S = basis.penalty_matrix()
        assert_allclose(S[:4, :4], 0.0)
        assert S.shape == (20, 20)


# ---------------------------------------------------------------------------
# TPRS basis for d=4 (m=3)
# ---------------------------------------------------------------------------


class TestTPRSD4:
    def test_m2_too_low_for_d4(self):
        x = RNG.uniform(0, 1, (200, 4))
        basis = TPRS(k=30)
        with pytest.raises(ValueError, match="too low"):
            basis.fit(x)

    def test_fit_m3(self):
        x = RNG.uniform(0, 1, (200, 4))
        basis = TPRS(k=30, m=3).fit(x)
        B = basis.basis_matrix(x)
        # M = C(2+4, 4) = 15, so 30-15=15 spline columns
        assert B.shape == (200, 30)
        assert basis.null_space_dimension() == 15

    def test_penalty_m3(self):
        x = RNG.uniform(0, 1, (200, 4))
        basis = TPRS(k=30, m=3).fit(x)
        S = basis.penalty_matrix()
        assert_allclose(S[:15, :15], 0.0)

    def test_predict_new_data_m3(self):
        x_train = RNG.uniform(0, 1, (200, 4))
        x_new = RNG.uniform(0, 1, (50, 4))
        basis = TPRS(k=30, m=3).fit(x_train)
        B = basis.basis_matrix(x_new)
        assert B.shape == (50, 30)


# ---------------------------------------------------------------------------
# TPRS basis for d=5 (m=3)
# ---------------------------------------------------------------------------


class TestTPRSD5:
    def test_fit_m3(self):
        x = RNG.uniform(0, 1, (300, 5))
        # M = C(2+5, 5) = 21
        basis = TPRS(k=35, m=3).fit(x)
        assert basis.null_space_dimension() == 21
        B = basis.basis_matrix(x)
        assert B.shape == (300, 35)


# ---------------------------------------------------------------------------
# GAM integration: s(x1, x2) with d=2
# ---------------------------------------------------------------------------


class TestGAMD2:
    @pytest.fixture()
    def data_2d(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + np.cos(x2) + rng.normal(0, 0.3, n)
        return {"x1": x1, "x2": x2, "y": y}

    def test_fit(self, data_2d):
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data_2d, method="REML")
        assert gam.is_fitted

    def test_deviance_explained(self, data_2d):
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data_2d, method="REML")
        assert gam.deviance_explained > 0.7

    def test_predict(self, data_2d):
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data_2d, method="REML")
        pred = gam.predict(data_2d)
        assert np.isfinite(pred.values).all()

    def test_predict_se(self, data_2d):
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data_2d, method="REML")
        pred = gam.predict(data_2d, se=True)
        assert pred.se is not None
        assert np.all(pred.se > 0)

    def test_predict_new_data(self, data_2d):
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data_2d, method="REML")
        new = {"x1": np.linspace(0, 2 * np.pi, 20), "x2": np.linspace(0, 2 * np.pi, 20)}
        pred = gam.predict(new)
        assert pred.values.shape == (20,)

    def test_summary(self, data_2d):
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data_2d, method="REML")
        s = gam.summary()
        assert "GAM fit summary" in s

    def test_with_select(self, data_2d):
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data_2d, method="REML", select=True)
        assert gam.is_fitted

    def test_higher_k(self, data_2d):
        gam = GAM("y ~ s(x1, x2, k=20)", family=Gaussian())
        gam.fit(data_2d, method="REML")
        assert gam.is_fitted
        assert gam.deviance_explained > 0.7


# ---------------------------------------------------------------------------
# GAM integration: s(x1, x2) with Poisson
# ---------------------------------------------------------------------------


class TestGAMD2Poisson:
    def test_fit(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = rng.poisson(np.exp(0.3 * np.sin(x1) + 0.2 * np.cos(x2)))
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM("y ~ s(x1, x2)", family=Poisson())
        gam.fit(data, method="REML")
        assert gam.is_fitted
        assert np.all(gam.predict(data).values > 0)


# ---------------------------------------------------------------------------
# GAM integration: s(x1, x2, x3) with d=3
# ---------------------------------------------------------------------------


class TestGAMD3:
    def test_fit(self):
        rng = np.random.default_rng(23)
        n = 400
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        x3 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + 0.5 * np.cos(x2) + 0.3 * np.sin(x3) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "x3": x3, "y": y}
        gam = GAM("y ~ s(x1, x2, x3, k=20)", family=Gaussian())
        gam.fit(data, method="REML")
        assert gam.is_fitted
        assert gam.deviance_explained > 0.3

    def test_predict_new_data(self):
        rng = np.random.default_rng(23)
        n = 400
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        x3 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "x3": x3, "y": y}
        gam = GAM("y ~ s(x1, x2, x3, k=20)", family=Gaussian())
        gam.fit(data, method="REML")
        new = {
            "x1": rng.uniform(0, 2 * np.pi, 30),
            "x2": rng.uniform(0, 2 * np.pi, 30),
            "x3": rng.uniform(0, 2 * np.pi, 30),
        }
        pred = gam.predict(new)
        assert pred.values.shape == (30,)
        assert np.isfinite(pred.values).all()


# ---------------------------------------------------------------------------
# GAM integration: mixed s(x1, x2) + s(x3)
# ---------------------------------------------------------------------------


class TestGAMMixed:
    def test_2d_plus_1d(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        x3 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) * np.cos(x2) + 0.5 * np.sin(x3) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "x3": x3, "y": y}
        gam = GAM("y ~ s(x1, x2) + s(x3)", family=Gaussian())
        gam.fit(data, method="REML")
        assert gam.is_fitted
        assert len(gam.edf) == 2

    def test_mixed_deviance(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        x3 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) * np.cos(x2) + 0.5 * np.sin(x3) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "x3": x3, "y": y}
        gam = GAM("y ~ s(x1, x2) + s(x3)", family=Gaussian())
        gam.fit(data, method="REML")
        assert gam.deviance_explained > 0.5


# ---------------------------------------------------------------------------
# k_check and simulate with multi-dimensional TPRS
# ---------------------------------------------------------------------------


class TestMultiDimDiagnostics:
    @pytest.fixture()
    def fitted_2d(self):
        rng = np.random.default_rng(23)
        n = 200
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + np.cos(x2) + rng.normal(0, 0.3, n)
        data = {"x1": x1, "x2": x2, "y": y}
        gam = GAM("y ~ s(x1, x2)", family=Gaussian())
        gam.fit(data, method="REML")
        return gam, data

    def test_k_check(self, fitted_2d):
        gam, _ = fitted_2d
        results = gam.k_check(n_sim=50)
        assert len(results) == 1
        assert results[0].k_index > 0

    def test_simulate(self, fitted_2d):
        gam, data = fitted_2d
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape == (len(data["y"]), 10)
        assert np.isfinite(sims).all()

    def test_smooth_tests(self, fitted_2d):
        gam, _ = fitted_2d
        tests = gam.smooth_tests()
        assert len(tests) == 1
        assert tests[0].edf > 0

    def test_aic_bic(self, fitted_2d):
        gam, _ = fitted_2d
        assert np.isfinite(gam.aic)
        assert np.isfinite(gam.bic)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_k_too_small_for_d2(self):
        x = RNG.uniform(0, 1, (100, 2))
        basis = TPRS(k=3)
        with pytest.raises(ValueError, match="too small"):
            basis.fit(x)

    def test_k_equals_m_plus_1_for_d2(self):
        x = RNG.uniform(0, 1, (100, 2))
        basis = TPRS(k=4).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (100, 4)

    def test_dimension_mismatch_predict(self):
        x_train = RNG.uniform(0, 1, (100, 2))
        x_bad = RNG.uniform(0, 1, (50, 3))
        basis = TPRS(k=15).fit(x_train)
        with pytest.raises(ValueError, match="covariate"):
            basis.basis_matrix(x_bad)

    def test_n_less_than_k(self):
        x = RNG.uniform(0, 1, (10, 2))
        basis = TPRS(k=20)
        with pytest.raises(ValueError, match="smaller than"):
            basis.fit(x)
