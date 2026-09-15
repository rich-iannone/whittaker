"""Tests for stacking weights (whittaker/fitting/stacking.py)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker import GAM, StackingResult, stacking
from whittaker.families.poisson import Poisson
from whittaker.fitting.loo import LOOResult
from whittaker.fitting.stacking import _stacking_weights, _validate_pointwise
from whittaker.fitting.waic import WAICResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(0)
    n = 80
    x = np.linspace(0, 5, n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def vi_smooth(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data, method="VI")


@pytest.fixture(scope="module")
def vi_linear(gaussian_data):
    return GAM("y ~ x").fit(gaussian_data, method="VI")


@pytest.fixture(scope="module")
def vi_smooth_k5(gaussian_data):
    return GAM("y ~ s(x, k=5)").fit(gaussian_data, method="VI")


# ---------------------------------------------------------------------------
# StackingResult structure
# ---------------------------------------------------------------------------


class TestStackingResult:
    """Verify structure and properties of StackingResult."""

    def test_returns_correct_type(self, vi_smooth, vi_linear):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2)

        assert isinstance(result, StackingResult)

    def test_weights_sum_to_one(self, vi_smooth, vi_linear):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2)

        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)

    def test_weights_nonneg(self, vi_smooth, vi_linear):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2)

        assert np.all(result.weights >= 0.0)

    def test_n_models(self, vi_smooth, vi_linear, vi_smooth_k5):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
            loo3 = vi_smooth_k5.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2, loo3)

        assert result.n_models == 3
        assert result.weights.shape == (3,)

    def test_n_obs(self, vi_smooth, vi_linear, gaussian_data):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2)

        assert result.n_obs == len(gaussian_data["y"])

    def test_method_loo(self, vi_smooth, vi_linear):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2)

        assert result.method == "loo"

    def test_method_waic(self, vi_smooth, vi_linear):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        result = stacking(w1, w2)

        assert result.method == "waic"

    def test_repr_contains_weights(self, vi_smooth, vi_linear):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2)

        r = repr(result)
        assert "StackingResult" in r
        assert "Weights" in r
        assert "Model 1" in r
        assert "Model 2" in r


# ---------------------------------------------------------------------------
# Weight behavior
# ---------------------------------------------------------------------------


class TestWeightBehavior:
    """Verify stacking weights behave correctly in known scenarios."""

    def test_better_model_gets_higher_weight(self, vi_smooth, vi_linear):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        result = stacking(loo1, loo2)

        if loo1.elpd_loo > loo2.elpd_loo:
            assert result.weights[0] >= result.weights[1]
        else:
            assert result.weights[1] >= result.weights[0]

    def test_identical_models_equal_weights(self):
        n = 50
        pw = np.random.default_rng(0).standard_normal(n) - 5.0
        r1 = LOOResult(
            elpd_loo=float(pw.sum()),
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=pw,
            pareto_k=np.zeros(n),
            n_bad_k=0,
        )
        r2 = LOOResult(
            elpd_loo=float(pw.sum()),
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=pw.copy(),
            pareto_k=np.zeros(n),
            n_bad_k=0,
        )
        result = stacking(r1, r2)

        np.testing.assert_allclose(result.weights[0], result.weights[1], atol=0.05)

    def test_dominant_model_gets_weight_near_one(self):
        n = 100
        rng = np.random.default_rng(23)
        pw_good = rng.standard_normal(n) * 0.1 - 2.0
        pw_bad = rng.standard_normal(n) * 0.1 - 10.0
        r1 = LOOResult(
            elpd_loo=float(pw_good.sum()),
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=pw_good,
            pareto_k=np.zeros(n),
            n_bad_k=0,
        )
        r2 = LOOResult(
            elpd_loo=float(pw_bad.sum()),
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=pw_bad,
            pareto_k=np.zeros(n),
            n_bad_k=0,
        )
        result = stacking(r1, r2)

        assert result.weights[0] > 0.95


# ---------------------------------------------------------------------------
# WAIC stacking
# ---------------------------------------------------------------------------


class TestWAICStacking:
    """Verify stacking works with WAICResult inputs."""

    def test_waic_stacking(self, vi_smooth, vi_linear):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        result = stacking(w1, w2)

        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)
        assert np.all(result.weights >= 0.0)


# ---------------------------------------------------------------------------
# Three or more models
# ---------------------------------------------------------------------------


class TestMultiModel:
    """Verify stacking with three models."""

    def test_three_model_weights_sum_to_one(self, vi_smooth, vi_linear, vi_smooth_k5):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        w3 = vi_smooth_k5.waic(n_draws=300, seed=0)
        result = stacking(w1, w2, w3)

        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)
        assert result.n_models == 3

    def test_elpd_stacking_finite(self, vi_smooth, vi_linear, vi_smooth_k5):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        w3 = vi_smooth_k5.waic(n_draws=300, seed=0)
        result = stacking(w1, w2, w3)

        assert np.isfinite(result.elpd_stacking)
        assert result.se_elpd_stacking > 0


# ---------------------------------------------------------------------------
# Core optimization (unit test of _stacking_weights)
# ---------------------------------------------------------------------------


class TestStackingWeightsCore:
    """Test _stacking_weights with synthetic log predictive densities."""

    def test_uniform_lpd_gives_equal_weights(self):
        n, K = 50, 3
        rng = np.random.default_rng(0)
        lpd = np.tile(rng.standard_normal(n)[:, np.newaxis], (1, K))
        weights = _stacking_weights(lpd)

        np.testing.assert_allclose(weights, np.ones(K) / K, atol=0.05)

    def test_one_model_dominates(self):
        n = 100
        rng = np.random.default_rng(7)
        lpd = np.column_stack(
            [
                rng.standard_normal(n) * 0.1 - 1.0,
                rng.standard_normal(n) * 0.1 - 5.0,
                rng.standard_normal(n) * 0.1 - 10.0,
            ]
        )
        weights = _stacking_weights(lpd)

        assert weights[0] > 0.9
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)

    def test_single_model(self):
        lpd = np.random.default_rng(0).standard_normal((30, 1))
        weights = _stacking_weights(lpd)

        np.testing.assert_array_equal(weights, [1.0])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_fewer_than_two_models(self):
        r = LOOResult(
            elpd_loo=0.0,
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=np.zeros(10),
            pareto_k=np.zeros(10),
            n_bad_k=0,
        )
        with pytest.raises(ValueError, match="at least 2"):
            stacking(r)

    def test_mismatched_n_obs(self):
        r1 = LOOResult(
            elpd_loo=0.0,
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=np.zeros(50),
            pareto_k=np.zeros(50),
            n_bad_k=0,
        )
        r2 = LOOResult(
            elpd_loo=0.0,
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=np.zeros(60),
            pareto_k=np.zeros(60),
            n_bad_k=0,
        )
        with pytest.raises(ValueError, match="same number"):
            stacking(r1, r2)

    def test_mixed_types_raises(self):
        r1 = LOOResult(
            elpd_loo=0.0,
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=np.zeros(50),
            pareto_k=np.zeros(50),
            n_bad_k=0,
        )
        r2 = WAICResult(
            elpd_waic=0.0,
            se_elpd_waic=1.0,
            p_waic=1.0,
            waic=0.0,
            pointwise=np.zeros(50),
        )
        with pytest.raises(TypeError, match="same type"):
            stacking(r1, r2)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Expected"):
            _validate_pointwise(["not_a_result", "also_not"])


# ---------------------------------------------------------------------------
# Poisson family
# ---------------------------------------------------------------------------


class TestPoissonStacking:
    """Verify stacking with a non-Gaussian family."""

    def test_poisson_stacking(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x))).astype(float)
        data = {"x": x, "y": y}

        m1 = GAM("y ~ s(x)", family=Poisson()).fit(data, method="VI")
        m2 = GAM("y ~ x", family=Poisson()).fit(data, method="VI")

        w1 = m1.waic(n_draws=300, seed=0)
        w2 = m2.waic(n_draws=300, seed=0)
        result = stacking(w1, w2)

        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)
        assert np.all(result.weights >= 0.0)
        assert np.isfinite(result.elpd_stacking)
