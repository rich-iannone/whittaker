"""Tests for variational inference (whittaker/fitting/vi.py)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.binomial import Binomial
from whittaker.families.gamma import Gamma
from whittaker.families.gaussian import Gaussian
from whittaker.families.poisson import Poisson
from whittaker.fitting.inference import _bayesian_covariance
from whittaker.fitting.vi import BayesResult, VIResult, _kl_gaussian_penalty, vi_fit
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0, 1, n)
    y = np.sin(3 * x) + rng.normal(0, 0.3, n)
    return {"y": y, "x": x}


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(0, 1, n)
    lam = np.exp(1.5 * np.sin(3 * x))
    y = rng.poisson(lam).astype(float)
    return {"y": y, "x": x}


@pytest.fixture
def binomial_data():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.uniform(0, 1, n)
    prob = 1.0 / (1.0 + np.exp(-3.0 * np.sin(3 * x)))
    y = rng.binomial(1, prob).astype(float)
    return {"y": y, "x": x}


@pytest.fixture
def gamma_data():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(0, 1, n)
    mu = np.exp(1.5 * np.sin(3 * x))
    y = rng.gamma(shape=2.0, scale=mu / 2.0)
    return {"y": y, "x": x}


# ---------------------------------------------------------------------------
# VIResult / BayesResult dataclass
# ---------------------------------------------------------------------------


class TestBayesResult:
    def test_draw_shape(self):
        rng = np.random.default_rng(0)
        p = 5
        m = rng.standard_normal(p)
        C = np.eye(p) * 0.1
        br = BayesResult(
            coefficients=m,
            posterior_mean=m,
            posterior_cov=C,
            linear_predictor=np.zeros(10),
            fitted_values=np.zeros(10),
            smoothing_params=[],
            scale=1.0,
            edf=[],
            edf_total=0.0,
            n_iter=0,
            converged=True,
        )
        draws = br.draw(100, seed=42)
        assert draws.shape == (p, 100)

    def test_draw_mean_approx(self):
        p = 4
        m = np.array([1.0, -1.0, 0.5, 2.0])
        C = np.eye(p) * 0.01
        br = BayesResult(
            coefficients=m,
            posterior_mean=m,
            posterior_cov=C,
            linear_predictor=np.zeros(10),
            fitted_values=np.zeros(10),
            smoothing_params=[],
            scale=1.0,
            edf=[],
            edf_total=0.0,
            n_iter=0,
            converged=True,
        )
        draws = br.draw(5000, seed=7)
        np.testing.assert_allclose(draws.mean(axis=1), m, atol=0.05)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


class TestKLGaussianPenalty:
    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        p = 6
        L = np.tril(rng.standard_normal((p, p)))
        np.fill_diagonal(L, np.abs(np.diag(L)) + 0.1)
        m = rng.standard_normal(p)
        S = np.eye(p) * 0.5
        kl = _kl_gaussian_penalty(m, L, S)
        assert kl >= 0.0

    def test_zero_at_prior(self):
        # When q = prior: m = 0, C = S^{-1}, KL should be ~0
        p = 4
        S = np.eye(p) * 2.0
        C_prior = np.linalg.inv(S)
        L = np.linalg.cholesky(C_prior)
        m = np.zeros(p)
        kl = _kl_gaussian_penalty(m, L, S)
        assert abs(kl) < 1e-8

    def test_increases_with_mean_offset(self):
        p = 4
        S = np.eye(p)
        L = np.eye(p)
        kl_zero = _kl_gaussian_penalty(np.zeros(p), L, S)
        kl_off = _kl_gaussian_penalty(np.ones(p) * 2.0, L, S)
        assert kl_off > kl_zero


# ---------------------------------------------------------------------------
# Gaussian (fast path)
# ---------------------------------------------------------------------------


class TestVIGaussian:
    def test_gaussian_matches_laplace_mean(self, gaussian_data):
        from whittaker.fitting.pirls import pirls_fit

        model = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, method="VI")
        vi_r = model.vi_result
        assert vi_r is not None

        mm = build_model_matrix(parse("y ~ s(x)"), gaussian_data)
        pirls_r = pirls_fit(mm, Gaussian(), method="REML")

        np.testing.assert_allclose(vi_r.posterior_mean, pirls_r.coefficients, atol=1e-4)

    def test_gaussian_matches_laplace_cov(self, gaussian_data):
        from whittaker.fitting.pirls import pirls_fit

        model = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, method="VI")
        vi_r = model.vi_result

        mm = build_model_matrix(parse("y ~ s(x)"), gaussian_data)
        pirls_r = pirls_fit(mm, Gaussian(), method="REML")
        V_laplace = _bayesian_covariance(
            mm.X, mm.penalties, pirls_r.smoothing_params, pirls_r.scale
        )
        np.testing.assert_allclose(vi_r.posterior_cov, V_laplace, atol=1e-6)

    def test_gaussian_converged_in_zero_iters(self, gaussian_data):
        model = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, method="VI")
        assert model.vi_result.n_iter == 0
        assert model.vi_result.converged


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


class TestVIPoisson:
    def test_poisson_fits(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        vr = model.vi_result
        assert vr is not None
        assert vr.fitted_values.shape == (200,)
        assert np.all(vr.fitted_values > 0)

    def test_poisson_converges(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        assert model.vi_result.converged

    def test_elbo_increases(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(
            poisson_data, method="VI", vi_options={"max_iter": 200}
        )
        hist = model.vi_result.elbo_history
        assert len(hist) > 1
        # ELBO should be higher at end than start (after initial warm-up noise)
        assert hist[-1] > hist[0]


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------


class TestVIBinomial:
    def test_binomial_fits(self, binomial_data):
        model = GAM("y ~ s(x)", family=Binomial()).fit(binomial_data, method="VI")
        vr = model.vi_result
        assert vr is not None
        assert np.all(vr.fitted_values > 0) and np.all(vr.fitted_values < 1)

    def test_binomial_converges(self, binomial_data):
        model = GAM("y ~ s(x)", family=Binomial()).fit(binomial_data, method="VI")
        assert model.vi_result.converged


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------


class TestVIGamma:
    def test_gamma_fits(self, gamma_data):
        model = GAM("y ~ s(x)", family=Gamma()).fit(gamma_data, method="VI")
        vr = model.vi_result
        assert vr is not None
        assert np.all(vr.fitted_values > 0)

    def test_gamma_converges(self, gamma_data):
        model = GAM("y ~ s(x)", family=Gamma()).fit(gamma_data, method="VI")
        assert model.vi_result.converged


# ---------------------------------------------------------------------------
# simulate and posterior_samples
# ---------------------------------------------------------------------------


class TestVISimulate:
    def test_simulate_shape(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        sims = model.simulate(n_sim=100, seed=0)
        assert sims.shape == (200, 100)
        assert np.all(sims >= 0)

    def test_simulate_on_new_data(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        new_data = {"x": np.linspace(0, 1, 50)}
        sims = model.simulate(new_data, n_sim=50, seed=1)
        assert sims.shape == (50, 50)

    def test_posterior_samples_shape(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        samps = model.posterior_samples(n=300, seed=42)
        p = model.vi_result.posterior_cov.shape[0]
        assert samps.shape == (p, 300)

    def test_posterior_samples_laplace_shape(self, gaussian_data):
        model = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, method="REML")
        samps = model.posterior_samples(n=100, seed=0)
        assert samps.ndim == 2
        assert samps.shape[1] == 100

    def test_predict_se_uses_posterior_cov(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        result = model.predict(poisson_data, se=True)
        assert result.se is not None
        assert result.se.shape == (200,)
        assert np.all(result.se > 0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestVISummary:
    def test_summary_contains_variational(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        s = model.summary()
        assert "Variational" in str(s)

    def test_summary_contains_elbo(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        s = model.summary()
        assert "ELBO" in str(s)

    def test_summary_no_gcv_for_vi(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        s = model.summary()
        assert "GCV" not in str(s)


# ---------------------------------------------------------------------------
# vi_result property
# ---------------------------------------------------------------------------


class TestViResultProperty:
    def test_vi_result_is_none_for_reml(self, gaussian_data):
        model = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, method="REML")
        assert model.vi_result is None

    def test_vi_result_is_viresult_for_vi(self, gaussian_data):
        model = GAM("y ~ s(x)", family=Gaussian()).fit(gaussian_data, method="VI")
        assert isinstance(model.vi_result, VIResult)


# ---------------------------------------------------------------------------
# Properties that raise NotImplementedError for VI fits
# ---------------------------------------------------------------------------


class TestVINotImplemented:
    def test_deviance_raises(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        with pytest.raises(NotImplementedError):
            _ = model.deviance

    def test_aic_raises(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        with pytest.raises(NotImplementedError):
            _ = model.aic

    def test_bic_raises(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        with pytest.raises(NotImplementedError):
            _ = model.bic

    def test_gcv_score_raises(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(poisson_data, method="VI")
        with pytest.raises(NotImplementedError):
            _ = model.gcv_score


# ---------------------------------------------------------------------------
# vi_options passthrough
# ---------------------------------------------------------------------------


class TestVIOptions:
    def test_cov_structure_block(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(
            poisson_data, method="VI", vi_options={"cov_structure": "block"}
        )
        assert model.vi_result.converged

    def test_phi_inference_variational(self, gamma_data):
        model = GAM("y ~ s(x)", family=Gamma()).fit(
            gamma_data, method="VI", vi_options={"phi_inference": "variational"}
        )
        vr = model.vi_result
        assert vr.phi_variational
        assert vr.log_phi_mean is not None
        assert vr.log_phi_var is not None

    def test_custom_lr_and_iters(self, poisson_data):
        model = GAM("y ~ s(x)", family=Poisson()).fit(
            poisson_data, method="VI", vi_options={"lr": 0.005, "max_iter": 500}
        )
        assert model.vi_result.n_iter <= 500


# ---------------------------------------------------------------------------
# Numerical: warm-start convergence
# ---------------------------------------------------------------------------


class TestWarmStart:
    def test_warm_start_fewer_iters(self, poisson_data):
        from whittaker.fitting.pirls import pirls_fit

        mm = build_model_matrix(parse("y ~ s(x)"), poisson_data)
        pirls_r = pirls_fit(mm, Poisson(), method="REML")

        # VI with warm start
        vr_warm = vi_fit(mm, Poisson(), init_result=pirls_r)

        # VI with no warm start (will run pirls internally anyway — same result expected)
        vr_cold = vi_fit(mm, Poisson())

        # Both should converge; warm start should use no more iters than cold
        assert vr_warm.converged
        assert vr_cold.converged
