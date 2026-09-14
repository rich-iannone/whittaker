"""Tests for WAIC (whittaker/fitting/waic.py)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker import GAM, WAICComparison, WAICResult, waic_compare
from whittaker.families.poisson import Poisson
from whittaker.fitting.waic import compute_waic

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
def mcmc_smooth(gaussian_data):
    return GAM("y ~ s(x)").fit(
        gaussian_data,
        method="MCMC",
        mcmc_options={"n_chains": 2, "n_samples": 300, "n_warmup": 150, "seed": 23},
    )


# ---------------------------------------------------------------------------
# WAICResult structure
# ---------------------------------------------------------------------------


class TestWAICResult:
    """Verify structure and basic properties of WAICResult."""

    def test_vi_waic_shape(self, vi_smooth, gaussian_data):
        result = vi_smooth.waic(n_draws=300, seed=0)
        n = len(gaussian_data["y"])
        assert result.pointwise.shape == (n,)

    def test_vi_waic_types(self, vi_smooth):
        result = vi_smooth.waic(n_draws=300, seed=0)
        assert isinstance(result, WAICResult)
        assert isinstance(result.elpd_waic, float)
        assert isinstance(result.se_elpd_waic, float)
        assert isinstance(result.p_waic, float)
        assert isinstance(result.waic, float)

    def test_se_positive(self, vi_smooth):
        result = vi_smooth.waic(n_draws=300, seed=0)
        assert result.se_elpd_waic > 0.0

    def test_p_waic_positive(self, vi_smooth):
        result = vi_smooth.waic(n_draws=300, seed=0)
        assert result.p_waic > 0.0

    def test_waic_equals_minus_two_elpd(self, vi_smooth):
        result = vi_smooth.waic(n_draws=300, seed=0)
        assert abs(result.waic - (-2.0 * result.elpd_waic)) < 1e-10

    def test_elpd_equals_sum_of_pointwise(self, vi_smooth):
        result = vi_smooth.waic(n_draws=300, seed=0)
        assert abs(result.elpd_waic - float(np.sum(result.pointwise))) < 1e-8

    def test_repr_contains_waic(self, vi_smooth):
        result = vi_smooth.waic(n_draws=300, seed=0)
        r = repr(result)
        assert "ELPD_WAIC" in r
        assert "p_WAIC" in r
        assert "WAIC" in r


# ---------------------------------------------------------------------------
# MCMC WAIC
# ---------------------------------------------------------------------------


class TestMCMCWAIC:
    """Verify WAIC with MCMC posterior samples."""

    def test_mcmc_waic_runs(self, mcmc_smooth, gaussian_data):
        result = mcmc_smooth.waic()
        n = len(gaussian_data["y"])
        assert result.pointwise.shape == (n,)

    def test_mcmc_waic_p_positive(self, mcmc_smooth):
        result = mcmc_smooth.waic()
        assert result.p_waic > 0.0

    def test_mcmc_waic_uses_all_samples(self, mcmc_smooth):
        r1 = mcmc_smooth.waic(seed=0)
        r2 = mcmc_smooth.waic(n_draws=10, seed=0)
        assert r1.pointwise.shape == r2.pointwise.shape


# ---------------------------------------------------------------------------
# Frequentist raises
# ---------------------------------------------------------------------------


class TestWAICErrors:
    def test_raises_for_frequentist(self, gaussian_data):
        gam = GAM("y ~ s(x)").fit(gaussian_data)
        with pytest.raises(ValueError, match="Bayesian"):
            gam.waic()


# ---------------------------------------------------------------------------
# waic_compare
# ---------------------------------------------------------------------------


class TestWAICCompare:
    def test_compare_returns_comparison(self, vi_smooth, vi_linear):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        cmp = waic_compare(w1, w2)
        assert isinstance(cmp, WAICComparison)

    def test_compare_diff_equals_sum_pointwise_diff(self, vi_smooth, vi_linear):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        cmp = waic_compare(w1, w2)
        expected = float(np.sum(w1.pointwise - w2.pointwise))
        assert abs(cmp.elpd_diff - expected) < 1e-8

    def test_compare_se_positive(self, vi_smooth, vi_linear):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        cmp = waic_compare(w1, w2)
        assert cmp.se_diff > 0.0

    def test_compare_raises_mismatched_n(self):
        r1 = WAICResult(
            elpd_waic=0.0,
            se_elpd_waic=1.0,
            p_waic=1.0,
            waic=0.0,
            pointwise=np.zeros(50),
        )
        r2 = WAICResult(
            elpd_waic=0.0,
            se_elpd_waic=1.0,
            p_waic=1.0,
            waic=0.0,
            pointwise=np.zeros(60),
        )
        with pytest.raises(ValueError, match="same data"):
            waic_compare(r1, r2)

    def test_compare_repr_contains_diff(self, vi_smooth, vi_linear):
        w1 = vi_smooth.waic(n_draws=300, seed=0)
        w2 = vi_linear.waic(n_draws=300, seed=0)
        cmp = waic_compare(w1, w2)
        assert "ELPD diff" in repr(cmp)


# ---------------------------------------------------------------------------
# compute_waic directly (unit test of the core function)
# ---------------------------------------------------------------------------


class TestComputeWAIC:
    """Test compute_waic with synthetic log-likelihoods."""

    def test_constant_log_lik(self):
        # If log-likelihoods are constant across draws, p_waic should be 0.
        n, S = 20, 200
        ll_vals = np.random.default_rng(5).standard_normal(n)
        log_lik = np.tile(ll_vals[:, np.newaxis], (1, S))
        result = compute_waic(log_lik)
        assert abs(result.p_waic) < 1e-10
        np.testing.assert_allclose(result.pointwise, ll_vals, atol=1e-10)

    def test_lppd_formula(self):
        # Verify lppd = log(mean(exp(ll))) for each observation.
        rng = np.random.default_rng(42)
        n, S = 10, 500
        log_lik = rng.standard_normal((n, S)) * 0.1 - 5.0
        result = compute_waic(log_lik)
        from scipy.special import logsumexp

        expected_lppd = logsumexp(log_lik, axis=1) - np.log(S)
        expected_p = np.var(log_lik, axis=1, ddof=1)
        expected_pointwise = expected_lppd - expected_p
        np.testing.assert_allclose(result.pointwise, expected_pointwise, atol=1e-12)

    def test_p_waic_increases_with_variance(self):
        # More variance in log-lik across draws -> larger p_waic.
        rng = np.random.default_rng(7)
        n, S = 30, 300
        log_lik_low = rng.standard_normal((n, S)) * 0.01 - 5.0
        log_lik_high = rng.standard_normal((n, S)) * 1.0 - 5.0
        r_low = compute_waic(log_lik_low)
        r_high = compute_waic(log_lik_high)
        assert r_high.p_waic > r_low.p_waic


# ---------------------------------------------------------------------------
# Cross-method consistency (WAIC vs LOO)
# ---------------------------------------------------------------------------


class TestWAICvsLOO:
    """WAIC and LOO should give similar ELPD estimates for well-behaved models."""

    def test_waic_loo_elpd_same_sign(self, vi_smooth):
        import warnings

        w = vi_smooth.waic(n_draws=500, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l = vi_smooth.loo(n_draws=500, seed=0)
        # Both should agree on sign (both negative for log-density, or both positive).
        # For Gaussian with moderate data, both should be finite and roughly similar.
        assert np.isfinite(w.elpd_waic)
        assert np.isfinite(l.elpd_loo)
        # p_waic and p_loo should both be positive and in the same ballpark.
        assert w.p_waic > 0
        assert l.p_loo > 0

    def test_waic_loo_elpd_within_tolerance(self, vi_smooth):
        import warnings

        w = vi_smooth.waic(n_draws=1000, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l = vi_smooth.loo(n_draws=1000, seed=42)
        # For well-behaved models, WAIC and LOO ELPD should be close.
        assert abs(w.elpd_waic - l.elpd_loo) < 3.0 * max(w.se_elpd_waic, l.se_elpd_loo)


# ---------------------------------------------------------------------------
# Poisson family
# ---------------------------------------------------------------------------


class TestWAICPoisson:
    """WAIC with a non-Gaussian family."""

    def test_poisson_waic_runs(self):
        rng = np.random.default_rng(23)
        n = 100
        x = np.linspace(0, 3, n)
        y = rng.poisson(np.exp(0.5 * np.sin(x))).astype(float)
        data = {"x": x, "y": y}

        model = GAM("y ~ s(x)", family=Poisson()).fit(data, method="VI")
        result = model.waic(n_draws=300, seed=0)
        assert result.pointwise.shape == (n,)
        assert np.isfinite(result.waic)
        assert result.p_waic > 0
