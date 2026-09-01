"""Tests for PSIS-LOO cross-validation (whittaker/fitting/loo.py)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from whittaker import GAM, LOOComparison, LOOResult, loo_compare
from whittaker.fitting.loo import _psis_smooth_one, compute_loo

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
# PSIS internals
# ---------------------------------------------------------------------------


class TestPSISSmoothOne:
    """Unit tests for the per-observation PSIS smoothing step."""

    def test_returns_same_length(self):
        rng = np.random.default_rng(0)
        lw = rng.standard_normal(500)
        lw_smooth, k = _psis_smooth_one(lw)
        assert lw_smooth.shape == lw.shape

    def test_k_hat_is_finite(self):
        rng = np.random.default_rng(1)
        lw = rng.standard_normal(500)
        _, k = _psis_smooth_one(lw)
        assert np.isfinite(k)

    def test_constant_weights_k_zero(self):
        # All equal weights — no variation in tail, k should be 0.
        lw = np.zeros(200)
        _, k = _psis_smooth_one(lw)
        assert k == 0.0

    def test_smoothing_reduces_extreme_outlier(self):
        # Insert one large outlier; smoothed value should be smaller.
        rng = np.random.default_rng(2)
        lw = rng.standard_normal(500)
        lw[0] = 100.0  # huge outlier
        lw_smooth, _ = _psis_smooth_one(lw)
        # The smoothed max should be less than the raw max.
        assert lw_smooth.max() < lw.max()


# ---------------------------------------------------------------------------
# LOOResult structure
# ---------------------------------------------------------------------------


class TestLOOResult:
    """Verify structure and basic properties of LOOResult."""

    def test_vi_loo_shape(self, vi_smooth, gaussian_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vi_smooth.loo(n_draws=300, seed=0)
        n = len(gaussian_data["y"])
        assert result.pointwise.shape == (n,)
        assert result.pareto_k.shape == (n,)

    def test_vi_loo_types(self, vi_smooth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vi_smooth.loo(n_draws=300, seed=0)
        assert isinstance(result, LOOResult)
        assert isinstance(result.elpd_loo, float)
        assert isinstance(result.se_elpd_loo, float)
        assert isinstance(result.p_loo, float)
        assert isinstance(result.n_bad_k, int)

    def test_se_positive(self, vi_smooth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vi_smooth.loo(n_draws=300, seed=0)
        assert result.se_elpd_loo > 0.0

    def test_n_bad_k_consistent(self, vi_smooth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vi_smooth.loo(n_draws=300, seed=0)
        assert result.n_bad_k == int(np.sum(result.pareto_k > 0.7))

    def test_elpd_equals_sum_of_pointwise(self, vi_smooth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vi_smooth.loo(n_draws=300, seed=0)
        assert abs(result.elpd_loo - float(np.sum(result.pointwise))) < 1e-8

    def test_repr_contains_elpd(self, vi_smooth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vi_smooth.loo(n_draws=300, seed=0)
        assert "ELPD_LOO" in repr(result)


# ---------------------------------------------------------------------------
# MCMC LOO
# ---------------------------------------------------------------------------


class TestMCMCLOO:
    """Verify LOO with MCMC posterior samples."""

    def test_mcmc_loo_runs(self, mcmc_smooth, gaussian_data):
        result = mcmc_smooth.loo()
        n = len(gaussian_data["y"])
        assert result.pointwise.shape == (n,)

    def test_mcmc_loo_uses_all_samples(self, mcmc_smooth):
        # MCMC uses all stored draws regardless of n_draws argument.
        r1 = mcmc_smooth.loo(seed=0)
        r2 = mcmc_smooth.loo(n_draws=10, seed=0)  # n_draws ignored for MCMC
        assert r1.pointwise.shape == r2.pointwise.shape

    def test_mcmc_p_loo_positive(self, mcmc_smooth):
        result = mcmc_smooth.loo()
        assert result.p_loo > 0.0


# ---------------------------------------------------------------------------
# Frequentist raises
# ---------------------------------------------------------------------------


class TestLOOErrors:
    def test_raises_for_frequentist(self, gaussian_data):
        gam = GAM("y ~ s(x)").fit(gaussian_data)
        with pytest.raises(ValueError, match="Bayesian"):
            gam.loo()


# ---------------------------------------------------------------------------
# loo_compare
# ---------------------------------------------------------------------------


class TestLOOCompare:
    def test_compare_returns_comparison(self, vi_smooth, vi_linear):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        cmp = loo_compare(loo1, loo2)
        assert isinstance(cmp, LOOComparison)

    def test_compare_diff_equals_sum_pointwise_diff(self, vi_smooth, vi_linear):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        cmp = loo_compare(loo1, loo2)
        expected = float(np.sum(loo1.pointwise - loo2.pointwise))
        assert abs(cmp.elpd_diff - expected) < 1e-8

    def test_compare_se_positive(self, vi_smooth, vi_linear):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        cmp = loo_compare(loo1, loo2)
        assert cmp.se_diff > 0.0

    def test_compare_raises_mismatched_n(self):
        n1, n2 = 50, 60
        r1 = LOOResult(
            elpd_loo=0.0,
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=np.zeros(n1),
            pareto_k=np.zeros(n1),
            n_bad_k=0,
        )
        r2 = LOOResult(
            elpd_loo=0.0,
            se_elpd_loo=1.0,
            p_loo=1.0,
            pointwise=np.zeros(n2),
            pareto_k=np.zeros(n2),
            n_bad_k=0,
        )
        with pytest.raises(ValueError, match="same data"):
            loo_compare(r1, r2)

    def test_compare_repr_contains_diff(self, vi_smooth, vi_linear):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loo1 = vi_smooth.loo(n_draws=300, seed=0)
            loo2 = vi_linear.loo(n_draws=300, seed=0)
        cmp = loo_compare(loo1, loo2)
        assert "ELPD diff" in repr(cmp)


# ---------------------------------------------------------------------------
# compute_loo directly (unit test of the core function)
# ---------------------------------------------------------------------------


class TestComputeLOO:
    """Test compute_loo with synthetic log-likelihoods."""

    def test_trivial_case(self):
        # If all log-likelihoods are equal across draws, IS weights are uniform
        # and elpd_i = ll_i (constant across draws).
        rng = np.random.default_rng(5)
        n, S = 20, 200
        ll = rng.standard_normal(n)
        log_lik = np.tile(ll[:, np.newaxis], (1, S))
        lpd_full = ll.copy()
        result = compute_loo(log_lik, lpd_full)
        assert np.allclose(result.pointwise, ll, atol=1e-6)
        assert abs(result.p_loo) < 1e-5
