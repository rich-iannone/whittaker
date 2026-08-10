"""Tests for Duchon splines (bs='ds')."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.gam import GAM
from whittaker.smooths.duchon import DuchonSpline, _duchon_radial_basis

# ---------------------------------------------------------------------------
# Radial basis tests
# ---------------------------------------------------------------------------


class TestDuchonRadialBasis:
    def test_even_exponent_uses_log(self):
        r = np.array([0.0, 0.5, 1.0, 2.0])
        result = _duchon_radial_basis(r, s=1.0)
        assert result[0] == 0.0
        np.testing.assert_allclose(result[2], 0.0)
        assert result[3] == pytest.approx(4.0 * np.log(2.0))

    def test_odd_exponent_no_log(self):
        r = np.array([0.0, 1.0, 2.0])
        result = _duchon_radial_basis(r, s=0.5)
        np.testing.assert_allclose(result, r)

    def test_fractional_s(self):
        r = np.array([0.0, 1.0, 2.0])
        result = _duchon_radial_basis(r, s=0.75)
        np.testing.assert_allclose(result, r**1.5)


# ---------------------------------------------------------------------------
# DuchonSpline basis tests
# ---------------------------------------------------------------------------


class TestDuchonSplineBasis:
    def test_fit_and_shapes(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        basis = DuchonSpline(k=10, m=2).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (100, 10)
        assert np.all(np.isfinite(B))

    def test_penalty_shape(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        basis = DuchonSpline(k=8).fit(x)
        S = basis.penalty_matrix()
        assert S.shape == (8, 8)
        np.testing.assert_allclose(S, S.T)

    def test_null_space_dimension(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        basis = DuchonSpline(k=10, m=2).fit(x)
        assert basis.null_space_dimension() == 2

    def test_m_as_list(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        basis = DuchonSpline(k=10, m=[0.5, 1]).fit(x)
        assert basis.null_space_dimension() == 1
        B = basis.basis_matrix(x)
        assert B.shape == (100, 10)

    def test_m_as_tuple(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        basis = DuchonSpline(k=10, m=(1.5, 2)).fit(x)
        assert basis.null_space_dimension() == 2

    def test_2d_fit(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(size=(200, 2))
        basis = DuchonSpline(k=15, m=2).fit(x)
        B = basis.basis_matrix(x)
        assert B.shape == (200, 15)
        assert basis.null_space_dimension() == 3

    def test_is_fitted(self):
        basis = DuchonSpline(k=5)
        assert not basis.is_fitted
        basis.fit(np.linspace(0, 1, 50))
        assert basis.is_fitted

    def test_invalid_s_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            DuchonSpline(k=5, m=[-0.5, 1])

    def test_invalid_m_order_raises(self):
        with pytest.raises(ValueError, match="≥ 1"):
            DuchonSpline(k=5, m=[1.0, 0])

    def test_k_too_small_raises(self):
        basis = DuchonSpline(k=2, m=2)
        with pytest.raises(ValueError, match="too small"):
            basis.fit(np.linspace(0, 1, 100))

    def test_recovers_tprs_for_matching_params(self):
        from whittaker.smooths.tprs import TPRS

        rng = np.random.default_rng(23)
        x = rng.uniform(0, 1, 100)
        tprs = TPRS(k=10, m=2).fit(x)
        duchon = DuchonSpline(k=10, m=[0.5, 2]).fit(x)
        B_tprs = tprs.basis_matrix(x)
        B_ds = duchon.basis_matrix(x)
        assert B_tprs.shape == B_ds.shape


# ---------------------------------------------------------------------------
# GAM integration tests
# ---------------------------------------------------------------------------


class TestDuchonGAM:
    def test_1d_converges(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='ds')")
        model.fit(data)
        assert model.is_fitted

    def test_1d_predict(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='ds')")
        model.fit(data)
        pred = model.predict(data)
        assert pred.values.shape == (n,)
        assert np.corrcoef(np.sin(x), pred.values)[0, 1] > 0.95

    def test_with_m_param(self):
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='ds', m=[1.5, 2])")
        model.fit(data)
        assert model.is_fitted
        pred = model.predict(data).values
        assert np.corrcoef(np.sin(x), pred)[0, 1] > 0.9

    def test_summary(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x, bs='ds')")
        model.fit(data)
        s = model.summary()
        assert "s(x" in s

    def test_2d_converges(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM("y ~ s(x1, x2, bs='ds', k=20)")
        model.fit(data)
        assert model.is_fitted

    def test_with_other_smooth(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.normal(0, 1, n)
        y = np.sin(x1) + 0.5 * x2 + rng.normal(0, 0.2, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = GAM("y ~ s(x1, bs='ds') + s(x2)")
        model.fit(data)
        assert model.is_fitted
        pred = model.predict(data).values
        assert np.corrcoef(y, pred)[0, 1] > 0.8
