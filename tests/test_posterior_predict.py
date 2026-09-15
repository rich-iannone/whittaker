"""Tests for GAM.posterior_predict() and PosteriorPredictResult."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker import GAM, PosteriorPredictResult
from whittaker.families.poisson import Poisson

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
def vi_model(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data, method="VI")


@pytest.fixture(scope="module")
def mcmc_model(gaussian_data):
    return GAM("y ~ s(x)").fit(
        gaussian_data,
        method="MCMC",
        mcmc_options={"n_chains": 2, "n_samples": 200, "n_warmup": 100, "seed": 23},
    )


@pytest.fixture(scope="module")
def reml_model(gaussian_data):
    return GAM("y ~ s(x)").fit(gaussian_data, method="REML")


# ---------------------------------------------------------------------------
# PosteriorPredictResult structure
# ---------------------------------------------------------------------------


class TestPosteriorPredictResult:
    """Verify the result dataclass and its convenience methods."""

    def test_returns_correct_type(self, vi_model):
        pp = vi_model.posterior_predict(n_draws=100, seed=0)

        assert isinstance(pp, PosteriorPredictResult)

    def test_shape(self, vi_model, gaussian_data):
        n = len(gaussian_data["y"])
        pp = vi_model.posterior_predict(n_draws=200, seed=0)

        assert pp.samples.shape == (n, 200)
        assert pp.n_obs == n
        assert pp.n_draws == 200

    def test_mean_shape(self, vi_model, gaussian_data):
        n = len(gaussian_data["y"])
        pp = vi_model.posterior_predict(n_draws=100, seed=0)

        assert pp.mean().shape == (n,)

    def test_std_shape(self, vi_model, gaussian_data):
        n = len(gaussian_data["y"])
        pp = vi_model.posterior_predict(n_draws=100, seed=0)

        assert pp.std().shape == (n,)
        assert np.all(pp.std() > 0)

    def test_quantile_scalar(self, vi_model, gaussian_data):
        n = len(gaussian_data["y"])
        pp = vi_model.posterior_predict(n_draws=100, seed=0)
        q50 = pp.quantile(0.5)

        assert q50.shape == (n,)

    def test_quantile_list(self, vi_model, gaussian_data):
        n = len(gaussian_data["y"])
        pp = vi_model.posterior_predict(n_draws=100, seed=0)
        qs = pp.quantile([0.1, 0.5, 0.9])

        assert qs.shape == (3, n)

    def test_interval_default(self, vi_model, gaussian_data):
        n = len(gaussian_data["y"])
        pp = vi_model.posterior_predict(n_draws=500, seed=0)
        lower, upper = pp.interval()

        assert lower.shape == (n,)
        assert upper.shape == (n,)
        assert np.all(upper >= lower)

    def test_interval_narrower_at_lower_level(self, vi_model):
        pp = vi_model.posterior_predict(n_draws=500, seed=0)
        lo_90, hi_90 = pp.interval(0.90)
        lo_99, hi_99 = pp.interval(0.99)

        width_90 = np.mean(hi_90 - lo_90)
        width_99 = np.mean(hi_99 - lo_99)
        assert width_99 > width_90

    def test_repr(self, vi_model):
        pp = vi_model.posterior_predict(n_draws=100, seed=0)

        assert "PosteriorPredictResult" in repr(pp)
        assert "n_obs=" in repr(pp)
        assert "n_draws=100" in repr(pp)


# ---------------------------------------------------------------------------
# New data predictions
# ---------------------------------------------------------------------------


class TestNewData:
    """Verify posterior_predict works with new data points."""

    def test_new_data_shape(self, vi_model):
        x_new = np.linspace(0, 5, 20)
        pp = vi_model.posterior_predict({"x": x_new}, n_draws=100, seed=0)

        assert pp.samples.shape == (20, 100)

    def test_training_data_default(self, vi_model, gaussian_data):
        n = len(gaussian_data["y"])
        pp = vi_model.posterior_predict(n_draws=50, seed=0)

        assert pp.n_obs == n


# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------


class TestInferenceMethods:
    """Verify posterior_predict works with all inference methods."""

    def test_vi(self, vi_model, gaussian_data):
        pp = vi_model.posterior_predict(n_draws=100, seed=0)

        assert pp.samples.shape[0] == len(gaussian_data["y"])
        assert np.all(np.isfinite(pp.samples))

    def test_mcmc(self, mcmc_model, gaussian_data):
        pp = mcmc_model.posterior_predict(n_draws=100, seed=0)

        assert pp.samples.shape[0] == len(gaussian_data["y"])
        assert np.all(np.isfinite(pp.samples))

    def test_reml(self, reml_model, gaussian_data):
        pp = reml_model.posterior_predict(n_draws=100, seed=0)

        assert pp.samples.shape[0] == len(gaussian_data["y"])
        assert np.all(np.isfinite(pp.samples))


# ---------------------------------------------------------------------------
# Observation noise
# ---------------------------------------------------------------------------


class TestObservationNoise:
    """Verify that posterior_predict includes observation noise (not just mean uncertainty)."""

    def test_wider_than_mean(self, vi_model):
        pp = vi_model.posterior_predict(n_draws=1000, seed=23)
        mu_draws = vi_model.simulate(n_sim=1000, seed=23, unconditional=False)

        pp_spread = np.mean(np.std(pp.samples, axis=1))
        mu_spread = np.mean(np.std(mu_draws, axis=1))
        assert pp_spread > mu_spread


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self, vi_model):
        pp1 = vi_model.posterior_predict(n_draws=100, seed=23)
        pp2 = vi_model.posterior_predict(n_draws=100, seed=23)

        np.testing.assert_array_equal(pp1.samples, pp2.samples)

    def test_different_seed_different_result(self, vi_model):
        pp1 = vi_model.posterior_predict(n_draws=100, seed=0)
        pp2 = vi_model.posterior_predict(n_draws=100, seed=1)

        assert not np.array_equal(pp1.samples, pp2.samples)


# ---------------------------------------------------------------------------
# Poisson family
# ---------------------------------------------------------------------------


class TestPoissonPosteriorPredict:
    """Verify posterior_predict with a non-Gaussian family."""

    def test_poisson_produces_nonneg(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x))).astype(float)
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Poisson()).fit(data, method="VI")
        pp = model.posterior_predict(n_draws=200, seed=0)

        assert pp.samples.shape == (n, 200)
        assert np.all(pp.samples >= 0)

    def test_poisson_integer_valued(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x))).astype(float)
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Poisson()).fit(data, method="VI")
        pp = model.posterior_predict(n_draws=100, seed=0)

        np.testing.assert_array_equal(pp.samples, np.round(pp.samples))


# ---------------------------------------------------------------------------
# Unfitted model
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unfitted_raises(self):
        gam = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            gam.posterior_predict()
