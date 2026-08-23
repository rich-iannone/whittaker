"""Tests for whittaker.fitting.mcmc (HMC posterior sampler)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.gaussian import Gaussian
from whittaker.families.poisson import Poisson
from whittaker.fitting.mcmc import (
    _grad_U,
    _leapfrog,
    _potential_U,
    mcmc_fit,
)
from whittaker.fitting.pirls import pirls_fit
from whittaker.formula.parser import parse
from whittaker.gam import GAM
from whittaker.model_matrix import build_model_matrix

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2024)


def _gaussian_data(n: int = 200, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    return {"y": np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=n), "x": x}


def _poisson_data(n: int = 200, seed: int = 2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    lam = np.exp(np.sin(2 * np.pi * x) + 1.5)
    return {"y": rng.poisson(lam).astype(float), "x": x}


def _build_components(data: dict, formula: str = "y ~ s(x)", family=None):
    """Return (mm, fr, S_lambda, family) for a fitted GAM."""
    if family is None:
        family = Gaussian()
    formula_obj = parse(formula)
    mm = build_model_matrix(formula_obj, data)
    fr = pirls_fit(mm, family, method="REML")
    sp = list(fr.smoothing_params)
    p = mm.X.shape[1]
    S_lambda = np.zeros((p, p))
    for lam, pen in zip(sp, mm.penalties, strict=False):
        S_lambda += lam * pen
    return mm, fr, S_lambda, family


# ---------------------------------------------------------------------------
# Leapfrog unit tests
# ---------------------------------------------------------------------------


class TestLeapfrog:
    """Geometric properties of the leapfrog integrator."""

    def _simple_system(self):
        """Small Gaussian system for leapfrog tests."""
        rng = np.random.default_rng(7)
        n, p = 50, 3
        X = rng.standard_normal((n, p))
        beta_true = np.array([1.0, -0.5, 2.0])
        scale = 0.5
        y = X @ beta_true + rng.normal(scale=np.sqrt(scale), size=n)
        S_lambda = 0.1 * np.eye(p)
        family = Gaussian()
        M_diag_inv = np.ones(p)  # identity mass
        return X, y, S_lambda, family, scale, M_diag_inv

    def test_leapfrog_reversibility(self):
        """Reversing momentum after one leapfrog trip returns to the start."""
        X, y, S, fam, scale, M_inv = self._simple_system()
        rng = np.random.default_rng(11)
        beta0 = rng.standard_normal(X.shape[1])
        p0 = rng.standard_normal(X.shape[1])

        beta1, p1 = _leapfrog(beta0, p0, M_inv, 0.05, 10, X, y, S, fam, scale, None, None)
        beta2, p2 = _leapfrog(beta1, -p1, M_inv, 0.05, 10, X, y, S, fam, scale, None, None)

        assert_allclose(beta2, beta0, atol=1e-10)
        assert_allclose(-p2, p0, atol=1e-10)

    def test_leapfrog_volume_preservation(self):
        """Jacobian of leapfrog map has determinant ≈ 1 (symplecticity)."""
        X, y, S, fam, scale, M_inv = self._simple_system()
        rng = np.random.default_rng(13)
        p = X.shape[1]
        beta0 = rng.standard_normal(p)
        p0 = rng.standard_normal(p)
        eps = 0.03
        h = 1e-5

        # Numerical Jacobian of the leapfrog map (beta0, p0) -> (beta1, p1)
        def F(z):
            b, pm = _leapfrog(z[:p], z[p:], M_inv, eps, 5, X, y, S, fam, scale, None, None)
            return np.concatenate([b, pm])

        z0 = np.concatenate([beta0, p0])
        J = np.zeros((2 * p, 2 * p))
        for i in range(2 * p):
            zp = z0.copy()
            zp[i] += h
            zm = z0.copy()
            zm[i] -= h
            J[:, i] = (F(zp) - F(zm)) / (2 * h)

        assert_allclose(abs(np.linalg.det(J)), 1.0, atol=1e-6)

    def test_energy_conservation(self):
        """Hamiltonian is approximately conserved for small step sizes."""
        X, y, S, fam, scale, M_inv = self._simple_system()
        rng = np.random.default_rng(17)
        p = X.shape[1]
        M_diag = 1.0 / M_inv  # identity → M_diag = ones
        beta0 = rng.standard_normal(p)
        p0 = rng.standard_normal(p) * np.sqrt(M_diag)

        eps = 0.01
        H0 = _potential_U(beta0, X, y, S, fam, scale, None, None) + 0.5 * float(
            np.sum(p0**2 * M_inv)
        )
        beta1, p1 = _leapfrog(beta0, p0, M_inv, eps, 20, X, y, S, fam, scale, None, None)
        H1 = _potential_U(beta1, X, y, S, fam, scale, None, None) + 0.5 * float(
            np.sum(p1**2 * M_inv)
        )

        # Leapfrog conserves H up to O(ε²) per step; tolerance ~ L * ε² * ‖H‖
        assert abs(H1 - H0) < 1.0


# ---------------------------------------------------------------------------
# Gradient / potential consistency
# ---------------------------------------------------------------------------


class TestGradientConsistency:
    """_grad_U must be the numerical gradient of _potential_U."""

    def _check(self, data, family, formula="y ~ s(x)"):
        mm, fr, S_lambda, fam = _build_components(data, formula, family)
        X, y, scale, offset = mm.X, mm.response, fr.scale, mm.offset
        rng = np.random.default_rng(99)
        beta = fr.coefficients + rng.standard_normal(len(fr.coefficients)) * 0.1

        g_analytic = _grad_U(beta, X, y, S_lambda, fam, scale, None, offset)

        h = 1e-5
        g_numeric = np.zeros_like(beta)
        for i in range(len(beta)):
            bp = beta.copy()
            bp[i] += h
            bm = beta.copy()
            bm[i] -= h
            g_numeric[i] = (
                _potential_U(bp, X, y, S_lambda, fam, scale, None, offset)
                - _potential_U(bm, X, y, S_lambda, fam, scale, None, offset)
            ) / (2 * h)

        rel_err = np.linalg.norm(g_analytic - g_numeric) / (np.linalg.norm(g_numeric) + 1e-15)
        assert rel_err < 1e-7, f"Relative gradient error {rel_err:.2e}"

    def test_gaussian_gradient_matches_numerical(self):
        self._check(_gaussian_data(), Gaussian())

    def test_poisson_gradient_matches_numerical(self):
        self._check(_poisson_data(), Poisson())

    def test_gradient_zero_at_map_gaussian(self):
        """∇U(β̂) ≈ 0 at the P-IRLS MAP for Gaussian."""
        data = _gaussian_data()
        mm, fr, S_lambda, fam = _build_components(data)
        g = _grad_U(fr.coefficients, mm.X, mm.response, S_lambda, fam, fr.scale, None, mm.offset)
        assert np.linalg.norm(g) < 1e-8

    def test_gradient_zero_at_map_poisson(self):
        """∇U(β̂) ≈ 0 at the P-IRLS MAP for Poisson."""
        data = _poisson_data()
        mm, fr, S_lambda, fam = _build_components(data, family=Poisson())
        g = _grad_U(fr.coefficients, mm.X, mm.response, S_lambda, fam, fr.scale, None, mm.offset)
        assert np.linalg.norm(g) < 1e-8


# ---------------------------------------------------------------------------
# Posterior correctness (Gaussian — exact Laplace)
# ---------------------------------------------------------------------------


class TestGaussianPosterior:
    """For Gaussian response the HMC posterior must match the Laplace approximation."""

    @pytest.fixture(scope="class")
    def fitted(self):
        data = _gaussian_data(seed=5)
        mm, fr, S_lambda, fam = _build_components(data)
        X, scale = mm.X, fr.scale

        # Analytical posterior
        A = X.T @ X + S_lambda
        cov_true = scale * np.linalg.inv(A)
        std_true = np.sqrt(np.diag(cov_true))

        # MCMC via full mcmc_fit
        mr = mcmc_fit(mm, fam, n_chains=4, n_samples=1000, n_warmup=500, leapfrog_steps=30, seed=42)
        return mr, fr.coefficients, std_true

    def test_gaussian_matches_laplace_mean(self, fitted):
        """MCMC posterior mean ≈ P-IRLS MAP within 3 × SE."""
        mr, beta_map, std_true = fitted
        beta_mean = mr.samples.mean(axis=1)
        # ESS-based standard error
        se = std_true / np.sqrt(np.maximum(mr.ess, 1.0))
        diff = np.abs(beta_mean - beta_map)
        assert np.all(diff < 3 * se + 1e-3), (
            f"Mean deviates by more than 3 SE: max ratio = {(diff / se).max():.2f}"
        )

    def test_gaussian_matches_laplace_cov(self, fitted):
        """MCMC std ≈ analytical posterior std within 20%."""
        mr, _, std_true = fitted
        ratio = mr.samples.std(axis=1) / std_true
        assert np.all(ratio > 0.75), f"Min ratio: {ratio.min():.3f}"
        assert np.all(ratio < 1.25), f"Max ratio: {ratio.max():.3f}"


# ---------------------------------------------------------------------------
# Posterior correctness (Poisson)
# ---------------------------------------------------------------------------


class TestPoissonPosterior:
    @pytest.fixture(scope="class")
    def fitted(self):
        data = _poisson_data(seed=6)
        gam = GAM("y ~ s(x)", family=Poisson())
        gam.fit(
            data,
            method="MCMC",
            mcmc_options={
                "n_chains": 4,
                "n_samples": 1000,
                "n_warmup": 500,
                "leapfrog_steps": 30,
                "seed": 55,
            },
        )
        return gam

    def test_poisson_r_hat_converges(self, fitted):
        """R-hat < 1.1 for all coefficients."""
        mr = fitted.mcmc_result
        assert mr.r_hat.max() < 1.1, f"R-hat max = {mr.r_hat.max():.4f}"

    def test_poisson_ess_adequate(self, fitted):
        """ESS / n_samples > 0.05 for all coefficients."""
        mr = fitted.mcmc_result
        ess_ratio = mr.ess / (mr.n_chains * mr.n_samples)
        assert ess_ratio.min() > 0.05, f"Min ESS ratio = {ess_ratio.min():.4f}"

    def test_poisson_posterior_mean_close_to_pirls(self, fitted):
        """MCMC posterior mean matches the MCMC's internal P-IRLS MAP within 10%."""
        gam = fitted
        mr = gam.mcmc_result
        mm = gam._model_matrix
        fr_internal = pirls_fit(mm, Poisson(), method="REML")
        beta_map = fr_internal.coefficients
        beta_mean = mr.samples.mean(axis=1)
        rel_diff = np.linalg.norm(beta_mean - beta_map) / (np.linalg.norm(beta_map) + 1e-15)
        assert rel_diff < 0.10, f"Relative diff MCMC mean vs MAP: {rel_diff:.4f}"


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


class TestConvergenceDiagnostics:
    @pytest.fixture(scope="class")
    def mr(self):
        data = _gaussian_data(seed=3)
        gam = GAM("y ~ s(x)")
        gam.fit(
            data,
            method="MCMC",
            mcmc_options={
                "n_chains": 4,
                "n_samples": 500,
                "n_warmup": 300,
                "leapfrog_steps": 20,
                "seed": 17,
            },
        )
        return gam.mcmc_result

    def test_r_hat_converges(self, mr):
        """4 dispersed chains converge: R-hat < 1.1."""
        assert mr.r_hat.max() < 1.1

    def test_acceptance_rate_in_range(self, mr):
        """Post-warmup acceptance rate is in [0.5, 0.95]."""
        assert 0.5 <= mr.acceptance_rate <= 0.95

    def test_ess_min_positive(self, mr):
        """Minimum ESS is strictly positive."""
        assert mr.ess.min() > 0


# ---------------------------------------------------------------------------
# MCMCResult API
# ---------------------------------------------------------------------------


class TestMCMCResultAPI:
    @pytest.fixture(scope="class")
    def gam(self):
        data = _gaussian_data(seed=4)
        gam = GAM("y ~ s(x)")
        gam.fit(
            data,
            method="MCMC",
            mcmc_options={
                "n_chains": 2,
                "n_samples": 200,
                "n_warmup": 100,
                "leapfrog_steps": 10,
                "seed": 7,
            },
        )
        return gam

    def test_mcmc_result_draw_shape(self, gam):
        """MCMCResult.draw(n) returns shape (p, n)."""
        mr = gam.mcmc_result
        p = mr.samples.shape[0]
        draws = mr.draw(500, seed=0)
        assert draws.shape == (p, 500)

    def test_mcmc_options_n_samples_respected(self, gam):
        """n_samples=200 stored in MCMCResult."""
        assert gam.mcmc_result.n_samples == 200

    def test_mcmc_result_is_none_for_reml(self):
        """mcmc_result is None after a non-MCMC fit."""
        data = _gaussian_data(seed=9)
        gam = GAM("y ~ s(x)").fit(data)
        assert gam.mcmc_result is None

    def test_summary_contains_rhat(self, gam):
        """summary() includes R-hat diagnostics."""
        assert "R-hat" in gam.summary()

    def test_deviance_raises(self, gam):
        """deviance property raises NotImplementedError for MCMC fits."""
        with pytest.raises(NotImplementedError):
            _ = gam.deviance

    def test_predict_se_works(self, gam):
        """predict(se=True) runs without error after MCMC fit."""
        data = _gaussian_data(seed=4)
        result = gam.predict(data, se=True)
        assert result is not None

    def test_simulate_shape(self, gam):
        """simulate(n_sim=50) returns array of shape (n_obs, 50)."""
        data = _gaussian_data(seed=4)
        sim = gam.simulate(data, n_sim=50)
        assert sim.shape == (len(data["y"]), 50)
