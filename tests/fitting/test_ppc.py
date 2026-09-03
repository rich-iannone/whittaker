"""Tests for posterior predictive checks (whittaker/fitting/ppc.py)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker import GAM, PPCResult
from whittaker.families.gaussian import Gaussian
from whittaker.families.poisson import Poisson
from whittaker.fitting.ppc import compute_ppc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(0)
    n = 150
    x = np.linspace(0, 5, n)
    y = np.sin(x) + rng.normal(scale=0.3, size=n)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def poisson_data():
    rng = np.random.default_rng(1)
    n = 150
    x = np.linspace(0, 3, n)
    y = rng.poisson(np.exp(0.5 * x)).astype(float)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def reml_model(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data)


@pytest.fixture(scope="module")
def vi_model(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data, method="VI")


@pytest.fixture(scope="module")
def mcmc_model(gaussian_data):
    return GAM("y ~ s(x)").fit(
        gaussian_data,
        method="MCMC",
        mcmc_options={"n_chains": 2, "n_samples": 200, "n_warmup": 100, "seed": 0},
    )


@pytest.fixture(scope="module")
def poisson_model(poisson_data):
    return GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")


# ---------------------------------------------------------------------------
# PPCResult structure
# ---------------------------------------------------------------------------


class TestPPCResultStructure:
    def test_y_rep_shape(self, reml_model, gaussian_data):
        result = reml_model.ppc(n_sim=200, seed=0)
        n = len(gaussian_data["y"])
        assert result.y_rep.shape == (n, 200)

    def test_observed_shape(self, reml_model, gaussian_data):
        result = reml_model.ppc(n_sim=200, seed=0)
        n = len(gaussian_data["y"])
        assert result.observed.shape == (n,)

    def test_observed_matches_training_y(self, reml_model, gaussian_data):
        result = reml_model.ppc(n_sim=200, seed=0)
        assert np.allclose(result.observed, gaussian_data["y"])

    def test_returns_ppc_result_type(self, reml_model):
        result = reml_model.ppc(n_sim=100, seed=0)
        assert isinstance(result, PPCResult)

    def test_default_stat_names(self, reml_model):
        result = reml_model.ppc(n_sim=100, seed=0)
        assert set(result.stat_names) == {"mean", "sd", "min", "max", "prop_zero"}

    def test_stat_returns_tuple(self, reml_model):
        result = reml_model.ppc(n_sim=100, seed=0)
        obs_val, rep_vals = result.stat("mean")
        assert isinstance(obs_val, float)
        assert rep_vals.shape == (100,)

    def test_repr_contains_key_strings(self, reml_model):
        result = reml_model.ppc(n_sim=100, seed=0)
        r = repr(result)
        assert "PPCResult" in r
        assert "mean" in r
        assert "sd" in r
        assert "p-value" in r


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------


class TestPValues:
    def test_p_values_in_unit_interval(self, reml_model):
        result = reml_model.ppc(n_sim=500, seed=0)
        for name in result.stat_names:
            p = result.p_value(name)
            assert 0.0 <= p <= 1.0, f"p-value for {name!r} out of [0,1]: {p}"

    def test_well_specified_model_p_values_not_extreme(self, reml_model):
        # A well-specified model should have p-values not all near 0 or 1.
        result = reml_model.ppc(n_sim=500, seed=0)
        for name in ("mean", "sd"):
            p = result.p_value(name)
            assert 0.05 < p < 0.95, f"p-value for {name!r} unexpectedly extreme: {p}"

    def test_misspecified_model_extreme_p_value(self, gaussian_data):
        # Fit an intercept-only model to data with a strong trend; max(y_rep) should
        # be far from max(y_obs) because the null model can't reach the tails.
        null_model = GAM("y ~ 1", family=Gaussian()).fit(gaussian_data)
        result = null_model.ppc(n_sim=500, seed=0)
        p_max = result.p_value("max")
        # The null model almost certainly under-predicts the max.
        assert p_max < 0.1 or p_max > 0.9


# ---------------------------------------------------------------------------
# Works for all fitting methods
# ---------------------------------------------------------------------------


class TestAllMethods:
    def test_reml_ppc_runs(self, reml_model, gaussian_data):
        result = reml_model.ppc(n_sim=100, seed=0)
        assert result.y_rep.shape[0] == len(gaussian_data["y"])

    def test_vi_ppc_runs(self, vi_model, gaussian_data):
        result = vi_model.ppc(n_sim=100, seed=0)
        assert result.y_rep.shape[0] == len(gaussian_data["y"])

    def test_mcmc_ppc_runs(self, mcmc_model, gaussian_data):
        result = mcmc_model.ppc(n_sim=100, seed=0)
        assert result.y_rep.shape[0] == len(gaussian_data["y"])


# ---------------------------------------------------------------------------
# Poisson (count) family — prop_zero is meaningful
# ---------------------------------------------------------------------------


class TestPoissonPPC:
    def test_poisson_ppc_runs(self, poisson_model, poisson_data):
        result = poisson_model.ppc(n_sim=200, seed=0)
        assert result.y_rep.shape[0] == len(poisson_data["y"])

    def test_poisson_y_rep_non_negative(self, poisson_model):
        result = poisson_model.ppc(n_sim=200, seed=0)
        assert np.all(result.y_rep >= 0)

    def test_prop_zero_p_value_exists(self, poisson_model):
        result = poisson_model.ppc(n_sim=200, seed=0)
        p = result.p_value("prop_zero")
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self, reml_model):
        r1 = reml_model.ppc(n_sim=100, seed=23)
        r2 = reml_model.ppc(n_sim=100, seed=23)
        assert np.allclose(r1.y_rep, r2.y_rep)

    def test_different_seeds_different_results(self, reml_model):
        r1 = reml_model.ppc(n_sim=100, seed=0)
        r2 = reml_model.ppc(n_sim=100, seed=1)
        assert not np.allclose(r1.y_rep, r2.y_rep)


# ---------------------------------------------------------------------------
# compute_ppc unit test
# ---------------------------------------------------------------------------


class TestComputePPC:
    def test_trivial_constant_data(self):
        # If y_obs and all y_rep are constant, p_value(mean) should be 0.5 or 1.0.
        y_obs = np.ones(20)
        y_rep = np.ones((20, 100))
        result = compute_ppc(y_rep, y_obs)
        p = result.p_value("mean")
        # All reps equal obs, so p = P(T_rep >= T_obs) = 1.0.
        assert p == 1.0

    def test_observed_stat_correct(self):
        rng = np.random.default_rng(5)
        y_obs = rng.normal(size=50)
        y_rep = rng.normal(size=(50, 200))
        result = compute_ppc(y_rep, y_obs)
        obs_mean, _ = result.stat("mean")
        assert abs(obs_mean - float(np.mean(y_obs))) < 1e-10
