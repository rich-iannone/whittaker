"""Tests for BigGAM (large-scale discretized fitting)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.bam import BigGAM, build_discretized_model_matrix
from whittaker.families.binomial import Binomial
from whittaker.families.poisson import Poisson
from whittaker.fitting.bam import _discretize_1d, _discretize_nd
from whittaker.formula.parser import parse
from whittaker.gam import GAM


class TestDiscretize:
    def test_1d_basic(self):
        x = np.linspace(0, 1, 1000)
        unique_vals, indices = _discretize_1d(x, 50)
        assert len(unique_vals) <= 50
        assert len(indices) == 1000
        assert indices.min() >= 0
        assert indices.max() < len(unique_vals)

    def test_1d_constant(self):
        x = np.full(100, 3.0)
        unique_vals, indices = _discretize_1d(x, 50)
        assert len(unique_vals) == 1
        assert np.all(indices == 0)

    def test_1d_covers_range(self):
        x = np.linspace(-5, 5, 10000)
        unique_vals, indices = _discretize_1d(x, 100)
        assert unique_vals[0] == pytest.approx(-5.0, abs=0.11)
        assert unique_vals[-1] == pytest.approx(5.0, abs=0.11)

    def test_nd_2d(self):
        rng = np.random.default_rng(23)
        x = rng.uniform(size=(500, 2))
        unique_x, indices = _discretize_nd(x, 20)
        assert unique_x.shape[1] == 2
        assert len(indices) == 500
        assert unique_x.shape[0] <= 20 * 20


class TestBuildDiscretized:
    @pytest.fixture()
    def simple_data(self):
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        return {"x": x, "y": y}

    def test_builds(self, simple_data):
        formula = parse("y ~ s(x)")
        dm = build_discretized_model_matrix(formula, simple_data, n_discrete=100)
        assert dm.n_obs == 500
        assert dm.n_cols > 1
        assert len(dm.blocks) >= 1
        assert dm.has_intercept

    def test_penalties_match_pirls(self, simple_data):
        from whittaker.model_matrix import build_model_matrix

        formula = parse("y ~ s(x)")
        dm = build_discretized_model_matrix(formula, simple_data, n_discrete=200)
        mm = build_model_matrix(formula, simple_data)
        assert len(dm.penalties) == len(mm.penalties)
        assert dm.n_cols == mm.X.shape[1]

    def test_n_discrete_controls_grid(self, simple_data):
        formula = parse("y ~ s(x)")
        dm50 = build_discretized_model_matrix(formula, simple_data, n_discrete=50)
        dm200 = build_discretized_model_matrix(formula, simple_data, n_discrete=200)
        d50 = dm50.blocks[0].unique_basis.shape[0]
        d200 = dm200.blocks[0].unique_basis.shape[0]
        assert d50 <= 50
        assert d200 <= 200
        assert d50 < d200


class TestBigGAMGaussian:
    @pytest.fixture()
    def sin_data(self):
        rng = np.random.default_rng(23)
        n = 2000
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        return {"x": x, "y": y}

    def test_converges(self, sin_data):
        model = BigGAM("y ~ s(x)")
        model.fit(sin_data)
        assert model.is_fitted

    def test_agrees_with_gam(self, sin_data):
        big = BigGAM("y ~ s(x)", n_discrete=200)
        big.fit(sin_data, method="GCV")
        gam = GAM("y ~ s(x)")
        gam.fit(sin_data, method="GCV")
        pred_big = big.predict(sin_data).values
        pred_gam = gam.predict(sin_data).values
        np.testing.assert_allclose(pred_big, pred_gam, atol=0.05)

    def test_predict(self, sin_data):
        model = BigGAM("y ~ s(x)")
        model.fit(sin_data)
        pred = model.predict(sin_data)
        assert pred.values.shape == (2000,)
        assert np.corrcoef(np.sin(sin_data["x"]), pred.values)[0, 1] > 0.95

    def test_predict_se(self, sin_data):
        model = BigGAM("y ~ s(x)")
        model.fit(sin_data)
        pred = model.predict(sin_data, se=True)
        assert pred.se is not None
        assert np.all(pred.se > 0)

    def test_summary(self, sin_data):
        model = BigGAM("y ~ s(x)")
        model.fit(sin_data)
        s = model.summary()
        assert "s(x)" in s

    def test_edf(self, sin_data):
        model = BigGAM("y ~ s(x)")
        model.fit(sin_data)
        edf = model.edf
        assert len(edf) == 1
        assert 1.0 < edf[0] < 10.0


class TestBigGAMMultiSmooth:
    @pytest.fixture()
    def two_smooth_data(self):
        rng = np.random.default_rng(23)
        n = 2000
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = np.linspace(-3, 3, n)
        y = np.sin(x1) + 0.5 * x2**2 + rng.normal(0, 0.5, n)
        return {"x1": x1, "x2": x2, "y": y}

    def test_converges(self, two_smooth_data):
        model = BigGAM("y ~ s(x1) + s(x2)")
        model.fit(two_smooth_data)
        assert model.is_fitted

    def test_agrees_with_gam(self, two_smooth_data):
        big = BigGAM("y ~ s(x1) + s(x2)", n_discrete=200)
        big.fit(two_smooth_data, method="GCV")
        gam = GAM("y ~ s(x1) + s(x2)")
        gam.fit(two_smooth_data, method="GCV")
        pred_big = big.predict(two_smooth_data).values
        pred_gam = gam.predict(two_smooth_data).values
        np.testing.assert_allclose(pred_big, pred_gam, atol=0.15)


class TestBigGAMPoisson:
    @pytest.fixture()
    def count_data(self):
        rng = np.random.default_rng(23)
        n = 2000
        x = np.linspace(0, 4, n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
        return {"x": x, "y": y}

    def test_converges(self, count_data):
        model = BigGAM("y ~ s(x)", family=Poisson())
        model.fit(count_data)
        assert model.is_fitted

    def test_prediction_correlated(self, count_data):
        model = BigGAM("y ~ s(x)", family=Poisson())
        model.fit(count_data)
        pred = model.predict(count_data).values
        mu_true = np.exp(0.5 + 0.3 * count_data["x"])
        assert np.corrcoef(mu_true, pred)[0, 1] > 0.9


class TestBigGAMBinomial:
    @pytest.fixture()
    def binary_data(self):
        rng = np.random.default_rng(23)
        n = 2000
        x = np.linspace(-3, 3, n)
        p = 1 / (1 + np.exp(-x))
        y = rng.binomial(1, p).astype(float)
        return {"x": x, "y": y}

    def test_converges(self, binary_data):
        model = BigGAM("y ~ s(x)", family=Binomial())
        model.fit(binary_data)
        assert model.is_fitted


class TestBigGAMREML:
    def test_reml_converges(self):
        rng = np.random.default_rng(23)
        n = 1000
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = BigGAM("y ~ s(x)")
        model.fit(data, method="REML")
        assert model.is_fitted

    def test_freml_alias(self):
        rng = np.random.default_rng(23)
        n = 1000
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        data = {"x": x, "y": y}
        model = BigGAM("y ~ s(x)")
        model.fit(data, method="fREML")
        assert model.is_fitted


class TestBigGAMParametric:
    def test_with_linear_term(self):
        rng = np.random.default_rng(23)
        n = 1000
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.normal(0, 1, n)
        y = np.sin(x1) + 2.0 * x2 + rng.normal(0, 0.2, n)
        data = {"x1": x1, "x2": x2, "y": y}
        model = BigGAM("y ~ s(x1) + x2")
        model.fit(data)
        assert model.is_fitted
        pred = model.predict(data)
        assert np.corrcoef(y, pred.values)[0, 1] > 0.9
