"""Tests for the Gaussian process smooth (bs='gp')."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.gam import GAM
from whittaker.smooths.gp import GaussianProcess, _cov_matrix

# ---------------------------------------------------------------------------
# Covariance function tests
# ---------------------------------------------------------------------------


class TestCovarianceFunctions:
    def test_exp_symmetry(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(size=(50, 1))
        C = _cov_matrix(x, x, "exp", rho=0.5)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_sqexp_positive_definite(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(size=(30, 1))
        C = _cov_matrix(x, x, "sqexp", rho=0.3)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > -1e-10)

    def test_matern32_diagonal_one(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(size=(20, 2))
        C = _cov_matrix(x, x, "matern32", rho=1.0)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-6)

    def test_matern52_diagonal_one(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(size=(20, 1))
        C = _cov_matrix(x, x, "matern52", rho=1.0)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-6)

    def test_unknown_cov_raises(self):
        x = np.array([[0.0], [1.0]])
        with pytest.raises(ValueError, match="Unknown covariance"):
            _cov_matrix(x, x, "rbf", rho=1.0)


# ---------------------------------------------------------------------------
# GP basis tests
# ---------------------------------------------------------------------------


class TestGaussianProcessBasis:
    def test_fit_and_shapes(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        basis = GaussianProcess(k=10)
        basis.fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (100, 10)
        assert np.all(np.isfinite(B))

    def test_penalty_shape(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        basis = GaussianProcess(k=8).fit(x)
        S = basis.penalty_matrix()
        assert S.shape == (8, 8)
        assert np.all(np.diag(S) > 0)

    def test_2d_fit(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(size=(200, 2))
        basis = GaussianProcess(k=15).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (200, 15)

    def test_all_cov_types(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        for cov in ["exp", "matern32", "matern52", "sqexp"]:
            basis = GaussianProcess(k=8, cov=cov).fit(x)
            B = basis.basis_matrix(x)
            assert B.shape == (100, 8), f"Failed for cov={cov}"

    def test_null_space_dimension(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 50)
        basis = GaussianProcess(k=8).fit(x)
        assert basis.null_space_dimension() == 0

    def test_is_fitted(self):
        basis = GaussianProcess(k=5)
        assert not basis.is_fitted
        basis.fit(np.linspace(0, 1, 50))
        assert basis.is_fitted

    def test_n_too_small_raises(self):
        basis = GaussianProcess(k=20)
        with pytest.raises(ValueError, match="Reduce k"):
            basis.fit(np.linspace(0, 1, 10))

    def test_invalid_cov_raises(self):
        with pytest.raises(ValueError, match="Unknown covariance"):
            GaussianProcess(k=5, cov="invalid")


# ---------------------------------------------------------------------------
# GAM integration tests
# ---------------------------------------------------------------------------


class TestGPGAM:
    def test_1d_converges(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='gp')")
        model.fit(data)
        assert model.is_fitted

    def test_1d_predict(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='gp')")
        model.fit(data)
        pred = model.predict(data)
        assert pred.values.shape == (n,)
        assert np.corrcoef(np.sin(x), pred.values)[0, 1] > 0.9

    def test_1d_summary(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='gp')")
        model.fit(data)
        s = model.summary()
        assert "s(x" in s

    def test_sqexp_via_xt(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='gp', xt='sqexp')")
        model.fit(data)
        assert model.is_fitted
        pred = model.predict(data).values
        assert np.corrcoef(np.sin(x), pred)[0, 1] > 0.8

    def test_2d_converges(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM("y ~ s(x1, x2, bs='gp', k=20)")
        model.fit(data)
        assert model.is_fitted

    def test_with_other_smooth(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.normal(0, 1, n)
        y = np.sin(x1) + 0.5 * x2 + rng.normal(0, 0.2, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM("y ~ s(x1, bs='gp') + s(x2)")
        model.fit(data)
        assert model.is_fitted
        pred = model.predict(data).values
        assert np.corrcoef(y, pred)[0, 1] > 0.8
