"""Tests for BigGAM (large-scale discretized fitting)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.bam import BigGAM, build_discretized_model_matrix
from whittaker.families.binomial import Binomial
from whittaker.families.poisson import Poisson
from whittaker.fitting.bam import _discretize_1d, _discretize_nd
from whittaker.formula.parser import parse
from whittaker.formula.terms import SmoothTerm
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


class TestBuildDiscretizedErrors:
    def test_unequal_lengths_raises(self):
        formula = parse("y ~ s(x)")
        with pytest.raises(ValueError, match="same length"):
            build_discretized_model_matrix(formula, {"y": np.zeros(10), "x": np.zeros(20)})

    def test_unsupported_smooth_type_raises(self):
        formula = parse("y ~ s(x1, k=5)")
        formula.terms[0] = SmoothTerm(variables=("x1",), smooth_type="xx", bs="cr", k=5, extra={})
        rng = np.random.default_rng(0)
        data = {"y": rng.normal(size=100), "x1": rng.normal(size=100)}
        with pytest.raises(NotImplementedError, match="not supported in BigGAM"):
            build_discretized_model_matrix(formula, data)

    def test_factor_smooth_interaction_raises(self):
        rng = np.random.default_rng(0)
        n = 100
        x1 = rng.normal(size=n)
        group = rng.choice(["a", "b", "c"], size=n)
        formula = parse("y ~ s(x1, group, bs='fs')")
        data = {"y": rng.normal(size=n), "x1": x1, "group": group}
        with pytest.raises(NotImplementedError, match="Factor smooth interactions"):
            build_discretized_model_matrix(formula, data)


class TestBuildDiscretizedParametricTerms:
    def test_full_interaction_term(self):
        rng = np.random.default_rng(0)
        n = 200
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = x1 * x2 + rng.normal(0, 0.1, n)
        formula = parse("y ~ x1 * x2")
        dm = build_discretized_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})
        assert dm.column_names == ["(Intercept)", "x1", "x2", "x1:x2"]
        assert dm.n_parametric == 3

    def test_offset_term(self):
        rng = np.random.default_rng(0)
        n = 200
        x1 = np.linspace(0, 2 * np.pi, n)
        off = rng.normal(size=n)
        y = np.sin(x1) + off + rng.normal(0, 0.1, n)
        formula = parse("y ~ s(x1) + offset(off)")
        dm = build_discretized_model_matrix(formula, {"y": y, "x1": x1, "off": off})
        assert dm.offset is not None
        assert dm.offset_expressions == ["off"]
        np.testing.assert_allclose(dm.offset, off)


class TestBuildDiscretizedTensorSmooths:
    @pytest.fixture()
    def tensor_data(self):
        rng = np.random.default_rng(0)
        n = 300
        x1 = rng.uniform(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 2 * np.pi, n)
        y = np.sin(x1) + np.cos(x2) + rng.normal(0, 0.2, n)
        return {"y": y, "x1": x1, "x2": x2}

    @pytest.mark.parametrize("smooth_type", ["te", "ti", "t2"])
    def test_tensor_smooth_types_build(self, tensor_data, smooth_type):
        formula = parse(f"y ~ {smooth_type}(x1, x2, k=5)")
        dm = build_discretized_model_matrix(formula, tensor_data)
        assert dm.n_cols > 1
        assert len(dm.blocks) == 1

    def test_tensor_smooth_fits(self, tensor_data):
        model = BigGAM("y ~ te(x1, x2, k=5)")
        model.fit(tensor_data)
        assert model.is_fitted


class TestBuildDiscretizedMultivariateSmooth:
    def test_multivariate_isotropic_smooth(self):
        rng = np.random.default_rng(0)
        n = 300
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x1) * x2 + rng.normal(0, 0.1, n)
        formula = parse("y ~ s(x1, x2, k=10)")
        dm = build_discretized_model_matrix(formula, {"y": y, "x1": x1, "x2": x2})
        assert dm.n_cols > 1
        assert len(dm.blocks) == 1


class TestBuildDiscretizedRandomEffect:
    def test_random_effect_basis_builds(self):
        rng = np.random.default_rng(0)
        n = 200
        group = rng.integers(0, 5, n).astype(str)
        y = rng.normal(size=n)
        formula = parse("y ~ s(group, bs='re')")
        dm = build_discretized_model_matrix(formula, {"y": y, "group": group})
        assert len(dm.blocks) == 1
        assert dm.blocks[0].unique_basis.shape[0] > 0
        np.testing.assert_array_equal(dm.blocks[0].indices, np.arange(n))

    def test_random_effect_basis_fits(self):
        rng = np.random.default_rng(0)
        n = 300
        group = rng.integers(0, 6, n).astype(str)
        group_effects = {str(g): rng.normal(0, 1.0) for g in range(6)}
        y = np.array([group_effects[g] for g in group]) + rng.normal(0, 0.2, n)
        model = BigGAM("y ~ s(group, bs='re')")
        model.fit({"y": y, "group": group})
        assert model.is_fitted


class TestBuildDiscretizedSelect:
    def test_select_adds_null_space_penalty(self):
        rng = np.random.default_rng(0)
        n = 200
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)
        formula = parse("y ~ s(x, k=8)")
        data = {"y": y, "x": x}
        dm_plain = build_discretized_model_matrix(formula, data, select=False)
        dm_select = build_discretized_model_matrix(formula, data, select=True)
        assert len(dm_select.penalties) > len(dm_plain.penalties)

    def test_select_fits(self):
        rng = np.random.default_rng(0)
        n = 200
        x = np.linspace(0, 1, n)
        y = x + rng.normal(0, 0.1, n)
        model = BigGAM("y ~ s(x, k=8)")
        model.fit({"y": y, "x": x}, select=True)
        assert model.is_fitted


class TestBuildDiscretizedByVariable:
    def test_factor_by_creates_one_block_per_level(self):
        rng = np.random.default_rng(0)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        grp = rng.choice(["a", "b"], size=n)
        y = np.sin(x1) + rng.normal(0, 0.2, n)
        formula = parse("y ~ s(x1, by=grp, k=8)")
        dm = build_discretized_model_matrix(formula, {"y": y, "x1": x1, "grp": grp})
        assert [info.by_level for info in dm.smooth_infos] == ["a", "b"]
        assert len(dm.blocks) == 2

    def test_numeric_by_creates_one_block(self):
        rng = np.random.default_rng(0)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        z = rng.normal(size=n)
        y = np.sin(x1) * z + rng.normal(0, 0.2, n)
        formula = parse("y ~ s(x1, by=z, k=8)")
        dm = build_discretized_model_matrix(formula, {"y": y, "x1": x1, "z": z})
        assert len(dm.blocks) == 1
        assert dm.smooth_infos[0].by_var == "z"
        assert dm.blocks[0].by_weights is not None

    def test_by_variable_model_fits(self):
        rng = np.random.default_rng(0)
        n = 400
        x1 = np.linspace(0, 2 * np.pi, n)
        grp = rng.choice(["a", "b"], size=n)
        y = np.sin(x1) + rng.normal(0, 0.2, n)
        model = BigGAM("y ~ s(x1, by=grp, k=8)")
        model.fit({"y": y, "x1": x1, "grp": grp})
        assert model.is_fitted


class TestBigGAMFitWeights:
    @pytest.fixture()
    def sin_data(self):
        rng = np.random.default_rng(23)
        n = 500
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        return {"x": x, "y": y}

    def test_wrong_shape_weights_raises(self, sin_data):
        model = BigGAM("y ~ s(x)")
        with pytest.raises(ValueError, match="weights must be a 1-D array"):
            model.fit(sin_data, weights=np.ones(10))

    def test_non_positive_weights_raises(self, sin_data):
        model = BigGAM("y ~ s(x)")
        with pytest.raises(ValueError, match="All weights must be positive"):
            model.fit(sin_data, weights=-np.ones(len(sin_data["x"])))

    def test_positive_weights_fit_succeeds(self, sin_data):
        model = BigGAM("y ~ s(x)")
        model.fit(sin_data, weights=np.ones(len(sin_data["x"])))
        assert model.is_fitted


class TestBigGAMSmoothTests:
    def test_smooth_tests_by_level_labels(self):
        rng = np.random.default_rng(23)
        n = 400
        x1 = np.linspace(0, 2 * np.pi, n)
        grp = rng.choice(["a", "b"], size=n)
        y = np.sin(x1) + rng.normal(0, 0.2, n)
        model = BigGAM("y ~ s(x1, by=grp, k=8)")
        model.fit({"y": y, "x1": x1, "grp": grp})
        tests = model.smooth_tests()
        labels = {t.term_label for t in tests}
        assert any(":a" in label for label in labels)
        assert any(":b" in label for label in labels)

    def test_smooth_tests_uses_prior_weights_when_irls_weights_absent(self):
        """For a Gaussian fit with explicit prior weights, `fit.weights` (the IRLS working
        weights) stays `None`, so `smooth_tests()` should fall back to `fit.prior_weights`."""
        rng = np.random.default_rng(23)
        n = 400
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + rng.normal(0, 0.2, n)
        model = BigGAM("y ~ s(x)")
        model.fit({"y": y, "x": x}, weights=np.ones(n))
        assert model._fit_result.weights is None
        assert model._fit_result.prior_weights is not None
        tests = model.smooth_tests()
        assert len(tests) == 1
        assert np.isfinite(tests[0].p_value)

    def test_smooth_tests_significant_vs_noise(self):
        rng = np.random.default_rng(23)
        n = 500
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + rng.normal(0, 0.3, n)
        model = BigGAM("y ~ s(x1, k=10) + s(x2, k=10)")
        model.fit({"y": y, "x1": x1, "x2": x2})
        tests = model.smooth_tests()
        assert tests[0].p_value < 0.01
        assert tests[1].p_value > 0.05
