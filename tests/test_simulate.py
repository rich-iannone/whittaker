"""Tests for posterior simulation (GAM.simulate)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson
from whittaker.gam import GAM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gaussian_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture()
def poisson_data():
    rng = np.random.default_rng(23)
    n = 200
    x = rng.uniform(0, 2 * np.pi, n)
    y = rng.poisson(np.exp(0.5 * np.sin(x)))
    return {"x": x, "y": y}


@pytest.fixture()
def binomial_data():
    rng = np.random.default_rng(23)
    n = 300
    x = rng.uniform(-3, 3, n)
    p = 1.0 / (1.0 + np.exp(-x))
    y = rng.binomial(1, p).astype(float)
    return {"x": x, "y": y}


@pytest.fixture()
def fitted_gaussian(gaussian_data):
    gam = GAM("y ~ s(x)", family=Gaussian())
    gam.fit(gaussian_data)
    return gam, gaussian_data


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestSimulateBasic:
    def test_returns_array(self, fitted_gaussian):
        gam, data = fitted_gaussian
        sims = gam.simulate(n_sim=10, seed=23)
        assert isinstance(sims, np.ndarray)

    def test_shape_default_data(self, fitted_gaussian):
        gam, data = fitted_gaussian
        n = len(data["y"])
        sims = gam.simulate(n_sim=50, seed=23)
        assert sims.shape == (n, 50)

    def test_shape_new_data(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        new_data = {"x": np.linspace(0, 2 * np.pi, 30)}
        sims = gam.simulate(new_data, n_sim=20, seed=23)
        assert sims.shape == (30, 20)

    def test_finite_values(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        sims = gam.simulate(n_sim=50, seed=23)
        assert np.isfinite(sims).all()

    def test_unfitted_raises(self):
        gam = GAM("y ~ s(x)", family=Gaussian())
        with pytest.raises(RuntimeError):
            gam.simulate()

    def test_seed_reproducibility(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        s1 = gam.simulate(n_sim=10, seed=23)
        s2 = gam.simulate(n_sim=10, seed=23)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        s1 = gam.simulate(n_sim=10, seed=23)
        s2 = gam.simulate(n_sim=10, seed=99)
        assert not np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# Posterior mean and variance
# ---------------------------------------------------------------------------


class TestPosteriorStatistics:
    def test_mean_near_fitted(self, fitted_gaussian):
        gam, data = fitted_gaussian
        sims = gam.simulate(n_sim=2000, seed=23)
        sim_mean = sims.mean(axis=1)
        fitted_vals = gam.predict(data).values
        np.testing.assert_allclose(sim_mean, fitted_vals, atol=0.1)

    def test_variance_positive(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        sims = gam.simulate(n_sim=100, seed=23)
        sim_var = sims.var(axis=1)
        assert np.all(sim_var > 0)

    def test_sim_se_near_predict_se(self, fitted_gaussian):
        gam, data = fitted_gaussian
        sims = gam.simulate(n_sim=5000, seed=23)
        sim_se = sims.std(axis=1)
        pred = gam.predict(data, se=True)
        np.testing.assert_allclose(sim_se, pred.se, atol=0.05, rtol=0.3)


# ---------------------------------------------------------------------------
# Unconditional (response noise)
# ---------------------------------------------------------------------------


class TestUnconditional:
    def test_unconditional_wider(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        cond = gam.simulate(n_sim=500, seed=23, unconditional=False)
        uncond = gam.simulate(n_sim=500, seed=23, unconditional=True)
        assert uncond.var() > cond.var()

    def test_unconditional_gaussian(self, fitted_gaussian):
        gam, _ = fitted_gaussian
        sims = gam.simulate(n_sim=100, seed=23, unconditional=True)
        assert np.isfinite(sims).all()

    def test_unconditional_poisson(self, poisson_data):
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(poisson_data)
        sims = gam.simulate(n_sim=100, seed=23, unconditional=True)
        assert np.isfinite(sims).all()
        assert np.all(sims >= 0)
        assert np.all(sims == np.floor(sims))

    def test_unconditional_binomial(self, binomial_data):
        gam = GAM("y ~ s(x)", family=Binomial())
        gam.fit(binomial_data)
        sims = gam.simulate(n_sim=100, seed=23, unconditional=True)
        assert set(np.unique(sims)).issubset({0.0, 1.0})

    def test_unconditional_gamma(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.exp(np.sin(x)) + rng.gamma(5, 0.1, n)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Gamma())
        gam.fit(data)
        sims = gam.simulate(n_sim=100, seed=23, unconditional=True)
        assert np.isfinite(sims).all()
        assert np.all(sims > 0)

    def test_unconditional_negative_binomial(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.negative_binomial(5, 0.5, n)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=NegativeBinomial(theta=5.0))
        gam.fit(data)
        sims = gam.simulate(n_sim=100, seed=23, unconditional=True)
        assert np.isfinite(sims).all()
        assert np.all(sims >= 0)
        assert np.all(sims == np.floor(sims))


# ---------------------------------------------------------------------------
# All families (conditional)
# ---------------------------------------------------------------------------


class TestSimulateAllFamilies:
    def test_gaussian(self, gaussian_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(gaussian_data)
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape[1] == 10

    def test_poisson(self, poisson_data):
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(poisson_data)
        sims = gam.simulate(n_sim=10, seed=23)
        assert np.all(sims > 0)

    def test_binomial(self, binomial_data):
        gam = GAM("y ~ s(x)", family=Binomial())
        gam.fit(binomial_data)
        sims = gam.simulate(n_sim=10, seed=23)
        assert np.all(sims >= 0) and np.all(sims <= 1)

    def test_gamma(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = np.exp(np.sin(x)) + rng.gamma(5, 0.1, n)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=Gamma())
        gam.fit(data)
        sims = gam.simulate(n_sim=10, seed=23)
        assert np.all(sims > 0)

    def test_negative_binomial(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        y = rng.negative_binomial(5, 0.5, n)
        data = {"x": x, "y": y}
        gam = GAM("y ~ s(x)", family=NegativeBinomial(theta=5.0))
        gam.fit(data)
        sims = gam.simulate(n_sim=10, seed=23)
        assert np.isfinite(sims).all()


# ---------------------------------------------------------------------------
# With offset
# ---------------------------------------------------------------------------


class TestSimulateWithOffset:
    def test_offset_affects_simulation(self):
        rng = np.random.default_rng(23)
        n = 200
        x = rng.uniform(0, 2 * np.pi, n)
        log_exposure = rng.uniform(0, 2, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x) + log_exposure))
        data = {"x": x, "y": y, "log_exposure": log_exposure}
        gam = GAM("y ~ s(x) + offset(log_exposure)", family=Poisson())
        gam.fit(data)

        x_grid = np.linspace(0, 2 * np.pi, 50)
        sims_zero = gam.simulate({"x": x_grid, "log_exposure": np.zeros(50)}, n_sim=100, seed=23)
        sims_one = gam.simulate({"x": x_grid, "log_exposure": np.ones(50)}, n_sim=100, seed=23)
        assert sims_one.mean() > sims_zero.mean()


# ---------------------------------------------------------------------------
# With weights
# ---------------------------------------------------------------------------


class TestSimulateWithWeights:
    def test_with_weights(self, gaussian_data):
        n = len(gaussian_data["y"])
        w = np.ones(n) * 2.0
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(gaussian_data, weights=w)
        sims = gam.simulate(n_sim=10, seed=23)
        assert sims.shape == (n, 10)


# ---------------------------------------------------------------------------
# With select
# ---------------------------------------------------------------------------


class TestSimulateWithSelect:
    def test_with_select(self, gaussian_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(gaussian_data, select=True)
        sims = gam.simulate(n_sim=10, seed=23)
        assert np.isfinite(sims).all()


# ---------------------------------------------------------------------------
# Simultaneous confidence bands from simulation
# ---------------------------------------------------------------------------


class TestSimulationBasedBands:
    def test_pointwise_coverage(self, gaussian_data):
        gam = GAM("y ~ s(x)", family=Gaussian())
        gam.fit(gaussian_data)
        x_grid = np.linspace(0, 2 * np.pi, 100)
        sims = gam.simulate({"x": x_grid}, n_sim=1000, seed=23)
        lower = np.percentile(sims, 2.5, axis=1)
        upper = np.percentile(sims, 97.5, axis=1)
        truth = np.sin(x_grid)
        covered = (truth >= lower) & (truth <= upper)
        coverage = covered.mean()
        assert coverage > 0.8
