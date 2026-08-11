"""Tests for marginal effects and derivative-based inference."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.poisson import Poisson
from whittaker.gam import GAM


@pytest.fixture
def sin_model():
    rng = np.random.default_rng(42)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)
    data = {"x": x, "y": y}
    model = GAM("y ~ s(x)")
    model.fit(data)
    return model, data


@pytest.fixture
def two_smooth_model():
    rng = np.random.default_rng(42)
    n = 300
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(x1) + 2.0 * x2 + rng.normal(0, 0.2, n)
    data = {"x1": x1, "x2": x2, "y": y}
    model = GAM("y ~ s(x1) + s(x2)")
    model.fit(data)
    return model, data


@pytest.fixture
def by_model():
    rng = np.random.default_rng(42)
    n = 400
    x = np.linspace(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    y = np.sin(x) * z + rng.normal(0, 0.2, n)
    data = {"x": x, "z": z, "y": y}
    model = GAM("y ~ s(x, by=z)")
    model.fit(data)
    return model, data


class TestDerivatives:
    def test_first_derivative_shape(self, sin_model):
        model, _ = sin_model
        results = model.derivatives("x", n_points=50)
        assert len(results) == 1
        r = results[0]
        assert r.x.shape == (50,)
        assert r.derivative.shape == (50,)
        assert r.se.shape == (50,)
        assert r.lower.shape == (50,)
        assert r.upper.shape == (50,)
        assert r.order == 1

    def test_first_derivative_approx_cos(self, sin_model):
        model, _ = sin_model
        results = model.derivatives("x", n_points=100)
        r = results[0]
        expected = np.cos(r.x)
        interior = (r.x > 0.5) & (r.x < 2 * np.pi - 0.5)
        np.testing.assert_allclose(
            r.derivative[interior], expected[interior], atol=0.3
        )

    def test_second_derivative_shape(self, sin_model):
        model, _ = sin_model
        results = model.derivatives("x", order=2, n_points=50)
        r = results[0]
        assert r.order == 2
        assert r.derivative.shape == (50,)

    def test_second_derivative_approx_neg_sin(self, sin_model):
        model, _ = sin_model
        results = model.derivatives("x", order=2, n_points=100)
        r = results[0]
        expected = -np.sin(r.x)
        interior = (r.x > 1.0) & (r.x < 2 * np.pi - 1.0)
        np.testing.assert_allclose(
            r.derivative[interior], expected[interior], atol=0.5
        )

    def test_ci_contains_derivative(self, sin_model):
        model, _ = sin_model
        results = model.derivatives("x", n_points=50, level=0.95)
        r = results[0]
        assert np.all(r.lower <= r.derivative)
        assert np.all(r.derivative <= r.upper)

    def test_se_positive(self, sin_model):
        model, _ = sin_model
        results = model.derivatives("x", n_points=50)
        r = results[0]
        assert np.all(r.se >= 0)

    def test_invalid_order_raises(self, sin_model):
        model, _ = sin_model
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            model.derivatives("x", order=3)

    def test_invalid_variable_raises(self, sin_model):
        model, _ = sin_model
        with pytest.raises(ValueError, match="not found"):
            model.derivatives("z")

    def test_two_smooth_returns_one(self, two_smooth_model):
        model, _ = two_smooth_model
        results = model.derivatives("x1")
        assert len(results) == 1
        assert "x1" in results[0].term

    def test_by_smooth_derivative(self, by_model):
        model, _ = by_model
        results = model.derivatives("x")
        assert len(results) >= 1

    def test_finite_values(self, sin_model):
        model, _ = sin_model
        results = model.derivatives("x")
        r = results[0]
        assert np.all(np.isfinite(r.derivative))
        assert np.all(np.isfinite(r.se))
        assert np.all(np.isfinite(r.lower))
        assert np.all(np.isfinite(r.upper))

    def test_custom_eps(self, sin_model):
        model, _ = sin_model
        r1 = model.derivatives("x", eps=0.01)[0]
        r2 = model.derivatives("x", eps=0.001)[0]
        np.testing.assert_allclose(r1.derivative, r2.derivative, atol=0.1)


class TestMarginalEffects:
    def test_basic_shape(self, sin_model):
        model, _ = sin_model
        results = model.marginal_effects("x", n_points=50)
        assert len(results) == 1
        r = results[0]
        assert r.x.shape == (50,)
        assert r.effect.shape == (50,)
        assert r.se.shape == (50,)
        assert r.variable == "x"

    def test_captures_sin_pattern(self, sin_model):
        model, _ = sin_model
        results = model.marginal_effects("x", n_points=100)
        r = results[0]
        peak_idx = np.argmin(np.abs(r.x - np.pi / 2))
        trough_idx = np.argmin(np.abs(r.x - 3 * np.pi / 2))
        assert r.effect[peak_idx] > r.effect[trough_idx]

    def test_ci_ordering(self, sin_model):
        model, _ = sin_model
        results = model.marginal_effects("x", n_points=50)
        r = results[0]
        assert np.all(r.lower <= r.effect)
        assert np.all(r.effect <= r.upper)

    def test_at_single_value(self, two_smooth_model):
        model, _ = two_smooth_model
        results = model.marginal_effects("x1", at={"x2": 0.5})
        assert len(results) == 1
        assert results[0].by_values == {"x2": 0.5}

    def test_at_multiple_values(self, two_smooth_model):
        model, _ = two_smooth_model
        results = model.marginal_effects("x1", at={"x2": [0.2, 0.8]})
        assert len(results) == 2
        assert results[0].by_values["x2"] == 0.2
        assert results[1].by_values["x2"] == 0.8

    def test_different_at_values_differ(self, two_smooth_model):
        model, _ = two_smooth_model
        r1 = model.marginal_effects("x2", at={"x1": 0.0})[0]
        r2 = model.marginal_effects("x2", at={"x1": 3.0})[0]
        assert r1.effect.shape == r2.effect.shape

    def test_invalid_variable_raises(self, sin_model):
        model, _ = sin_model
        with pytest.raises(ValueError, match="not found"):
            model.marginal_effects("z")

    def test_no_smooth_for_var_raises(self, two_smooth_model):
        model, _ = two_smooth_model
        with pytest.raises(ValueError, match="No smooth terms"):
            model.marginal_effects("y")

    def test_finite_values(self, sin_model):
        model, _ = sin_model
        results = model.marginal_effects("x")
        r = results[0]
        assert np.all(np.isfinite(r.effect))
        assert np.all(np.isfinite(r.se))

    def test_two_smooth_focal_x2(self, two_smooth_model):
        model, _ = two_smooth_model
        results = model.marginal_effects("x2")
        assert len(results) == 1
        r = results[0]
        assert r.variable == "x2"
        assert r.effect[-1] > r.effect[0]


class TestPairwiseComparisons:
    def test_basic_shape(self, two_smooth_model):
        model, _ = two_smooth_model
        pairs = [({"x2": 0.8}, {"x2": 0.2})]
        results = model.pairwise_comparisons("x1", pairs, n_points=50)
        assert len(results) == 1
        r = results[0]
        assert r.x.shape == (50,)
        assert r.difference.shape == (50,)
        assert r.se.shape == (50,)

    def test_ci_ordering(self, two_smooth_model):
        model, _ = two_smooth_model
        pairs = [({"x2": 0.8}, {"x2": 0.2})]
        results = model.pairwise_comparisons("x1", pairs, n_points=50)
        r = results[0]
        assert np.all(r.lower <= r.difference)
        assert np.all(r.difference <= r.upper)

    def test_label_format(self, two_smooth_model):
        model, _ = two_smooth_model
        pairs = [({"x2": 0.8}, {"x2": 0.2})]
        results = model.pairwise_comparisons("x1", pairs)
        r = results[0]
        assert "x2=0.8" in r.label
        assert "x2=0.2" in r.label

    def test_symmetric_difference(self, two_smooth_model):
        model, _ = two_smooth_model
        pairs_ab = [({"x2": 0.8}, {"x2": 0.2})]
        pairs_ba = [({"x2": 0.2}, {"x2": 0.8})]
        r_ab = model.pairwise_comparisons("x1", pairs_ab, n_points=30)[0]
        r_ba = model.pairwise_comparisons("x1", pairs_ba, n_points=30)[0]
        np.testing.assert_allclose(r_ab.difference, -r_ba.difference, atol=1e-10)

    def test_multiple_pairs(self, two_smooth_model):
        model, _ = two_smooth_model
        pairs = [
            ({"x2": 0.8}, {"x2": 0.2}),
            ({"x2": 0.5}, {"x2": 0.2}),
        ]
        results = model.pairwise_comparisons("x1", pairs)
        assert len(results) == 2

    def test_finite_values(self, two_smooth_model):
        model, _ = two_smooth_model
        pairs = [({"x2": 0.8}, {"x2": 0.2})]
        results = model.pairwise_comparisons("x1", pairs)
        r = results[0]
        assert np.all(np.isfinite(r.difference))
        assert np.all(np.isfinite(r.se))

    def test_zero_diff_same_condition(self, two_smooth_model):
        model, _ = two_smooth_model
        pairs = [({"x2": 0.5}, {"x2": 0.5})]
        results = model.pairwise_comparisons("x1", pairs, n_points=30)
        r = results[0]
        np.testing.assert_allclose(r.difference, 0.0, atol=1e-10)


class TestDerivativesPoisson:
    def test_poisson_derivative(self):
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * x)).astype(float)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)
        results = model.derivatives("x", n_points=50)
        assert len(results) == 1
        r = results[0]
        assert np.all(np.isfinite(r.derivative))
        assert np.all(r.derivative > 0)


class TestMarginalEffectsPoisson:
    def test_poisson_marginal(self):
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * x)).astype(float)
        data = {"x": x, "y": y}
        model = GAM("y ~ s(x)", family=Poisson())
        model.fit(data)
        results = model.marginal_effects("x", n_points=50)
        assert len(results) == 1
        r = results[0]
        assert np.all(np.isfinite(r.effect))
