"""Tests for causal GAMs: DML, CATE, and mediation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.causal import (
    CATEResult,
    CausalGAM,
    MediationResult,
    TreatmentEffect,
    mediation_analysis,
)


@pytest.fixture
def linear_treatment_data():
    """Data with a known constant treatment effect of 2.0."""
    rng = np.random.default_rng(23)
    n = 400
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    d = rng.binomial(1, 0.5, n).astype(float)
    y = 2.0 * d + np.sin(x1) + 2 * x2 + rng.normal(0, 0.5, n)
    return {"x1": x1, "x2": x2, "d": d, "y": y}


@pytest.fixture
def heterogeneous_data():
    """Data with treatment effect that varies with x1."""
    rng = np.random.default_rng(23)
    n = 400
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    d = rng.binomial(1, 0.5, n).astype(float)
    cate_true = np.sin(x1)
    y = cate_true * d + 2 * x2 + rng.normal(0, 0.3, n)
    return {"x1": x1, "x2": x2, "d": d, "y": y}


@pytest.fixture
def mediation_data():
    """Data with treatment -> mediator -> outcome pathway."""
    rng = np.random.default_rng(23)
    n = 300
    x1 = rng.uniform(0, 1, n)
    d = rng.binomial(1, 0.5, n).astype(float)
    m = 1.5 * d + np.sin(2 * np.pi * x1) + rng.normal(0, 0.3, n)
    y = 0.5 * d + 1.0 * m + x1 + rng.normal(0, 0.3, n)
    return {"x1": x1, "d": d, "m": m, "y": y}


class TestCausalGAMInit:
    def test_default_init(self):
        model = CausalGAM("y", "d", ["x1", "x2"])
        assert model.outcome == "y"
        assert model.treatment == "d"
        assert model.confounders == ["x1", "x2"]
        assert model.method == "partially_linear"
        assert not model.is_fitted

    def test_interactive_method(self):
        model = CausalGAM("y", "d", ["x1"], method="interactive")
        assert model.method == "interactive"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="must be"):
            CausalGAM("y", "d", ["x1"], method="bogus")

    def test_repr_unfitted(self):
        model = CausalGAM("y", "d", ["x1"])
        r = repr(model)
        assert "CausalGAM" in r
        assert "unfitted" in r

    def test_repr_fitted(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        r = repr(model)
        assert "fitted" in r


class TestCausalGAMFit:
    def test_fit_partially_linear(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        assert model.is_fitted

    def test_fit_returns_self(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        result = model.fit(linear_treatment_data, seed=23)
        assert result is model

    def test_unfitted_raises(self):
        model = CausalGAM("y", "d", ["x1"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.treatment_effect()

    def test_fit_interactive(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        assert model.is_fitted


class TestTreatmentEffect:
    def test_returns_treatment_effect(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect()
        assert isinstance(te, TreatmentEffect)

    def test_ate_near_true_value(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect()
        assert abs(te.ate - 2.0) < 0.5

    def test_ci_contains_true_value(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect(level=0.95)
        assert te.ci_lower < 2.0 < te.ci_upper

    def test_se_positive(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect()
        assert te.se > 0

    def test_p_value_significant(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect()
        assert te.p_value < 0.05

    def test_ci_ordering(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect()
        assert te.ci_lower < te.ate < te.ci_upper

    def test_higher_level_wider_ci(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te_90 = model.treatment_effect(level=0.90)
        te_99 = model.treatment_effect(level=0.99)
        width_90 = te_90.ci_upper - te_90.ci_lower
        width_99 = te_99.ci_upper - te_99.ci_lower
        assert width_99 > width_90

    def test_n_obs(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect()
        assert te.n_obs == 400

    def test_treatment_effect_repr(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        te = model.treatment_effect()
        r = repr(te)
        assert "ate=" in r
        assert "se=" in r
        assert "CI=" in r


class TestCATE:
    def test_cate_shape(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        result = model.cate(variable="x1", n_points=50)
        assert isinstance(result, CATEResult)
        assert result.x.shape == (50,)
        assert result.cate.shape == (50,)
        assert result.se.shape == (50,)

    def test_cate_ci_ordering(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        result = model.cate(variable="x1", n_points=50)
        assert np.all(result.lower <= result.cate)
        assert np.all(result.cate <= result.upper)

    def test_cate_default_variable(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        result = model.cate(n_points=30)
        assert result.variable == "x1"

    def test_cate_requires_interactive(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        with pytest.raises(ValueError, match="interactive"):
            model.cate()

    def test_cate_invalid_variable(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        with pytest.raises(ValueError, match="not among"):
            model.cate(variable="z")

    def test_cate_with_new_data(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        new_data = {
            "x1": np.linspace(0, 2 * np.pi, 20),
            "x2": np.full(20, 0.5),
        }
        result = model.cate(new_data, variable="x1")
        assert result.cate.shape == (20,)

    def test_cate_finite_values(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        result = model.cate(variable="x1", n_points=30)
        assert np.all(np.isfinite(result.cate))
        assert np.all(np.isfinite(result.se))

    def test_cate_too_few_observations_raises(self):
        """With too few total observations, fewer than 20 pseudo-outcomes are ever available,
        so `_fit_cate` bails out and leaves `_cate_model` as `None`; calling `.cate()` should
        then raise instead of using a nonexistent model."""
        rng = np.random.default_rng(0)
        n = 15
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        d = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * d + np.sin(x1) + rng.normal(0, 0.2, n)
        data = {"x1": x1, "x2": x2, "d": d, "y": y}

        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive", n_folds=3)
        model.fit(data, seed=23)
        with pytest.raises(RuntimeError, match="too few valid"):
            model.cate()


class TestResiduals:
    def test_residuals_shape(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        res_y, res_d = model.residuals()
        assert res_y.shape == (400,)
        assert res_d.shape == (400,)

    def test_residuals_centered(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        res_y, res_d = model.residuals()
        assert abs(np.mean(res_y)) < 1.0
        assert abs(np.mean(res_d)) < 0.5

    def test_residuals_are_copies(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        r1, _ = model.residuals()
        r2, _ = model.residuals()
        assert r1 is not r2


class TestSummary:
    def test_summary_content(self, linear_treatment_data):
        model = CausalGAM("y", "d", ["x1", "x2"])
        model.fit(linear_treatment_data, seed=23)
        s = model.summary()
        assert "CausalGAM" in s
        assert "ATE" in s
        assert "p-value" in s

    def test_summary_interactive(self, heterogeneous_data):
        model = CausalGAM("y", "d", ["x1", "x2"], method="interactive")
        model.fit(heterogeneous_data, seed=23)
        s = model.summary()
        assert "CATE" in s


class TestSeedReproducibility:
    def test_same_seed_same_result(self, linear_treatment_data):
        m1 = CausalGAM("y", "d", ["x1", "x2"])
        m1.fit(linear_treatment_data, seed=123)
        m2 = CausalGAM("y", "d", ["x1", "x2"])
        m2.fit(linear_treatment_data, seed=123)
        assert m1.treatment_effect().ate == m2.treatment_effect().ate


class TestMediationAnalysis:
    def test_returns_result(self, mediation_data):
        result = mediation_analysis(
            "y",
            "d",
            "m",
            ["x1"],
            mediation_data,
            n_simulations=20,
            seed=23,
        )
        assert isinstance(result, MediationResult)

    def test_effects_decomposition(self, mediation_data):
        result = mediation_analysis(
            "y",
            "d",
            "m",
            ["x1"],
            mediation_data,
            n_simulations=20,
            seed=23,
        )
        np.testing.assert_allclose(
            result.total_effect,
            result.direct_effect + result.indirect_effect,
            atol=1e-10,
        )

    def test_indirect_positive(self, mediation_data):
        result = mediation_analysis(
            "y",
            "d",
            "m",
            ["x1"],
            mediation_data,
            n_simulations=20,
            seed=23,
        )
        assert result.indirect_effect > 0

    def test_proportion_mediated(self, mediation_data):
        result = mediation_analysis(
            "y",
            "d",
            "m",
            ["x1"],
            mediation_data,
            n_simulations=20,
            seed=23,
        )
        assert 0.0 < result.proportion_mediated < 1.0

    def test_se_positive(self, mediation_data):
        result = mediation_analysis(
            "y",
            "d",
            "m",
            ["x1"],
            mediation_data,
            n_simulations=20,
            seed=23,
        )
        assert result.total_se > 0
        assert result.direct_se > 0
        assert result.indirect_se > 0

    def test_n_obs(self, mediation_data):
        result = mediation_analysis(
            "y",
            "d",
            "m",
            ["x1"],
            mediation_data,
            n_simulations=20,
            seed=23,
        )
        assert result.n_obs == 300

    def test_repr(self, mediation_data):
        result = mediation_analysis(
            "y",
            "d",
            "m",
            ["x1"],
            mediation_data,
            n_simulations=20,
            seed=23,
        )
        r = repr(result)
        assert "total=" in r
        assert "direct=" in r
        assert "indirect=" in r
        assert "proportion_mediated=" in r
