"""Tests for whittaker.fitting.mcmc (HMC and NUTS posterior samplers)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whittaker.families.gaussian import Gaussian
from whittaker.families.poisson import Poisson
from whittaker.fitting.mcmc import (
    _build_tree,
    _ess_autocorr,
    _grad_U,
    _hmc_chain,
    _leapfrog,
    _nuts_chain,
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
        mr = mcmc_fit(mm, fam, n_chains=4, n_samples=1000, n_warmup=500, leapfrog_steps=30, seed=23)
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
        """summary() includes R-hat, ESS bulk, ESS tail, and sampler label."""
        s = gam.summary()
        assert "R-hat" in s
        assert "NUTS" in s
        assert "Tree depth" in s
        assert "bulk" in s
        assert "tail" in s

    def test_n_divergent_accessible(self, gam):
        """n_divergent is present and non-negative on MCMCResult."""
        mr = gam.mcmc_result
        assert isinstance(mr.n_divergent, int)
        assert mr.n_divergent >= 0

    def test_ess_tail_shape_matches_ess(self, gam):
        """ess_tail has the same shape as ess."""
        mr = gam.mcmc_result
        assert mr.ess_tail.shape == mr.ess.shape

    def test_ess_tail_positive(self, gam):
        """ess_tail values are positive for a converged chain."""
        mr = gam.mcmc_result
        assert np.all(mr.ess_tail > 0)

    def test_deviance_raises(self, gam):
        """deviance property raises NotImplementedError for MCMC fits."""
        with pytest.raises(NotImplementedError):
            _ = gam.deviance

    def test_summary_includes_divergence_warning(self, gam):
        """summary() appends divergence line when n_divergent > 0."""
        mr = gam.mcmc_result
        original = mr.n_divergent
        try:
            mr.n_divergent = 3
            s = gam.summary()
            assert "Divergent" in s
            assert "3 transitions" in s
        finally:
            mr.n_divergent = original

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

    def test_credible_interval_shape_and_bounds(self, gam):
        """predict(interval='credible') returns lower <= values <= upper."""
        data = _gaussian_data(seed=4)
        r = gam.predict(data, interval="credible", level=0.95, n_sim=200, seed=99)
        assert r.lower.shape == r.values.shape
        assert r.upper.shape == r.values.shape
        assert np.all(r.lower <= r.values)
        assert np.all(r.values <= r.upper)


# ---------------------------------------------------------------------------
# NUTS-specific tests
# ---------------------------------------------------------------------------


class TestNUTS:
    """Verify NUTS-specific behaviour: default sampler, tree depth, HMC fallback."""

    @pytest.fixture(scope="class")
    def nuts_result(self):
        """NUTS fit on a small Gaussian problem (fast)."""
        data = _gaussian_data(seed=10)
        gam = GAM("y ~ s(x)")
        gam.fit(
            data,
            method="MCMC",
            mcmc_options={
                "sampler": "NUTS",
                "n_chains": 2,
                "n_samples": 300,
                "n_warmup": 200,
                "seed": 99,
            },
        )
        return gam.mcmc_result

    @pytest.fixture(scope="class")
    def hmc_result(self):
        """Explicit HMC fit for comparison.

        Uses 500 samples so the rank-normalized split R-hat (which halves each chain to
        4 × 250-sample half-chains) is stable enough for a 1.2 threshold.
        """
        data = _gaussian_data(seed=10)
        gam = GAM("y ~ s(x)")
        gam.fit(
            data,
            method="MCMC",
            mcmc_options={
                "sampler": "HMC",
                "leapfrog_steps": 20,
                "n_chains": 2,
                "n_samples": 500,
                "n_warmup": 300,
                "seed": 99,
            },
        )
        return gam.mcmc_result

    def test_nuts_is_default_sampler(self):
        """mcmc_fit() with no sampler argument uses NUTS (mean_tree_depth > 0)."""
        data = _gaussian_data(seed=11)
        mm, fr, S_lambda, fam = _build_components(data)
        mr = mcmc_fit(mm, fam, n_chains=2, n_samples=200, n_warmup=100, seed=7)
        assert mr.mean_tree_depth > 0.0, "Default sampler should be NUTS (mean_tree_depth > 0)"

    def test_nuts_tree_depth_at_least_one(self, nuts_result):
        """NUTS expands the tree beyond the trivial single-step base case."""
        assert nuts_result.mean_tree_depth >= 1.0, (
            f"Expected mean_tree_depth >= 1, got {nuts_result.mean_tree_depth:.2f}"
        )

    def test_nuts_mean_tree_depth_reasonable(self, nuts_result):
        """Mean tree depth is in the typical range (1–10) for a well-conditioned problem."""
        assert 1.0 <= nuts_result.mean_tree_depth <= 10.0, (
            f"Unexpected mean_tree_depth = {nuts_result.mean_tree_depth:.2f}"
        )

    def test_nuts_acceptance_rate_in_range(self, nuts_result):
        """NUTS mean per-leaf α is in [0.5, 1.0]."""
        assert 0.5 <= nuts_result.acceptance_rate <= 1.0, (
            f"Unexpected acceptance_rate = {nuts_result.acceptance_rate:.3f}"
        )

    def test_nuts_r_hat_converges(self, nuts_result):
        """NUTS chains converge: R-hat < 1.1 for all parameters."""
        assert nuts_result.r_hat.max() < 1.1, f"R-hat max = {nuts_result.r_hat.max():.4f}"

    def test_hmc_mean_tree_depth_is_zero(self, hmc_result):
        """mean_tree_depth is 0.0 for static-L HMC."""
        assert hmc_result.mean_tree_depth == 0.0, (
            f"Expected 0.0 for HMC, got {hmc_result.mean_tree_depth}"
        )

    def test_hmc_sampler_still_works(self, hmc_result):
        """sampler='HMC' produces valid diagnostics.

        Rank-normalized split R-hat is more conservative than classic R-hat (it splits 2 chains into
        4 half-chains), so the threshold here is 1.2 rather than 1.1.
        """
        assert hmc_result.r_hat.max() < 1.2
        assert 0.5 <= hmc_result.acceptance_rate <= 1.0

    def test_max_tree_depth_respected(self):
        """With max_tree_depth=2 the mean depth stays at or below 2."""
        data = _gaussian_data(seed=12)
        mm, fr, S_lambda, fam = _build_components(data)
        mr = mcmc_fit(
            mm,
            fam,
            n_chains=2,
            n_samples=200,
            n_warmup=100,
            max_tree_depth=2,
            seed=13,
        )
        assert mr.mean_tree_depth <= 2.0, (
            f"mean_tree_depth {mr.mean_tree_depth:.2f} exceeded max_tree_depth=2"
        )

    def test_nuts_no_divergences_on_gaussian(self, nuts_result):
        """Well-conditioned Gaussian posterior should produce zero divergences."""
        assert nuts_result.n_divergent == 0, (
            f"Expected 0 divergences, got {nuts_result.n_divergent}"
        )

    def test_nuts_n_divergent_is_int(self, nuts_result):
        """n_divergent must be a non-negative integer."""
        assert isinstance(nuts_result.n_divergent, int)
        assert nuts_result.n_divergent >= 0

    def test_hmc_no_divergences_on_gaussian(self, hmc_result):
        """HMC on a well-conditioned Gaussian should produce zero divergences."""
        assert hmc_result.n_divergent == 0, f"Expected 0 divergences, got {hmc_result.n_divergent}"

    def test_hmc_n_divergent_is_int(self, hmc_result):
        """HMC n_divergent must be a non-negative integer."""
        assert isinstance(hmc_result.n_divergent, int)
        assert hmc_result.n_divergent >= 0


# ---------------------------------------------------------------------------
# Mass matrix adaptation tests
# ---------------------------------------------------------------------------


class TestMassMatrixAdaptation:
    """Verify two-phase warmup mass matrix adaptation improves mixing."""

    def test_nuts_converges_on_mixed_scale_problem(self):
        """NUTS with mass matrix adaptation converges on a problem with very different coefficient
        scales (intercept ~10, smooth coefs ~0.01).

        A fixed identity mass matrix would struggle here; adaptation rescales momentum to match the
        posterior geometry.
        """
        rng = np.random.default_rng(23)
        n = 300
        x = np.linspace(0, 1, n)
        # Large intercept, tiny smooth variation to create a scale mismatch
        lam = np.exp(5.0 + 0.05 * np.sin(2 * np.pi * x))
        data = {"y": rng.poisson(lam).astype(float), "x": x}

        mm, _, _, fam = _build_components(data, family=Poisson())
        mr = mcmc_fit(
            mm,
            fam,
            n_chains=2,
            n_samples=300,
            n_warmup=400,
            seed=7,
        )
        assert mr.r_hat.max() < 1.2, (
            f"R-hat too high on mixed-scale problem: {mr.r_hat.max():.4f}. "
            "Mass matrix adaptation may not be working correctly."
        )

    def test_hmc_converges_on_mixed_scale_problem(self):
        """HMC with mass matrix adaptation converges on a mixed-scale Poisson problem."""
        rng = np.random.default_rng(43)
        n = 300
        x = np.linspace(0, 1, n)
        lam = np.exp(5.0 + 0.05 * np.sin(2 * np.pi * x))
        data = {"y": rng.poisson(lam).astype(float), "x": x}

        mm, _, _, fam = _build_components(data, family=Poisson())
        mr = mcmc_fit(
            mm,
            fam,
            n_chains=2,
            n_samples=300,
            n_warmup=400,
            sampler="HMC",
            leapfrog_steps=20,
            seed=8,
        )
        assert mr.r_hat.max() < 1.2, (
            f"R-hat too high for HMC on mixed-scale problem: {mr.r_hat.max():.4f}."
        )

    def test_short_warmup_does_not_trigger_adaptation(self):
        """With fewer than 20 warmup steps, adaptation is skipped (n_warm1 < 10) and the sampler
        still produces a valid result rather than crashing.
        """
        data = _gaussian_data(seed=5)
        mm, _, _, fam = _build_components(data)
        mr = mcmc_fit(
            mm,
            fam,
            n_chains=1,
            n_samples=50,
            n_warmup=10,
            seed=9,
        )
        assert mr.samples.shape[1] == 50
        assert mr.r_hat.shape[0] > 0


# ---------------------------------------------------------------------------
# Offset and weights branches in _grad_U / _potential_U
# ---------------------------------------------------------------------------


class TestGradUBranches:
    """Cover the offset and weights branches in _grad_U and _potential_U."""

    def _system(self):
        rng = np.random.default_rng(7)
        n, p = 50, 3
        X = rng.standard_normal((n, p))
        beta = np.array([0.5, -0.3, 1.0])
        S_lambda = 0.1 * np.eye(p)
        family = Gaussian()
        scale = 1.0
        y = X @ beta + rng.normal(scale=0.3, size=n)
        return X, y, S_lambda, family, scale, beta, n

    def test_grad_U_with_offset(self):
        """_grad_U with a non-None offset produces a different gradient than without."""
        X, y, S, fam, scale, beta, n = self._system()
        offset = np.ones(n) * 0.5
        g_no_off = _grad_U(beta, X, y, S, fam, scale, None, None)
        g_off = _grad_U(beta, X, y, S, fam, scale, None, offset)

        assert not np.allclose(g_no_off, g_off)

    def test_grad_U_with_weights(self):
        """_grad_U with observation weights produces a different gradient than without."""
        X, y, S, fam, scale, beta, n = self._system()
        weights = np.full(n, 2.0)
        g_no_w = _grad_U(beta, X, y, S, fam, scale, None, None)
        g_w = _grad_U(beta, X, y, S, fam, scale, weights, None)

        assert not np.allclose(g_no_w, g_w)

    def test_potential_U_with_offset(self):
        """_potential_U with a non-None offset produces a different value than without."""
        X, y, S, fam, scale, beta, n = self._system()
        offset = np.ones(n) * 0.5
        u_no_off = _potential_U(beta, X, y, S, fam, scale, None, None)
        u_off = _potential_U(beta, X, y, S, fam, scale, None, offset)

        assert u_no_off != u_off


# ---------------------------------------------------------------------------
# _ess_autocorr degenerate branch (near-zero variance)
# ---------------------------------------------------------------------------


class TestEssAutocorrDegenerate:
    """Cover the acv0 < 1e-15 branch: constant chains → ESS equals n_total."""

    def test_constant_chain_returns_n_total(self):
        """A chain with zero variance (all identical draws) has ESS = n_total."""
        # All draws identical → acv0 ≈ 0, so ESS should be set to n_total
        chains = np.ones((2, 50, 3))
        result = _ess_autocorr(chains)
        assert_allclose(result, 100.0)  # n_total = 2 * 50


# ---------------------------------------------------------------------------
# Direct chain function tests (bypass ProcessPoolExecutor for coverage)
# ---------------------------------------------------------------------------


class TestHMCChainDirect:
    """Call _hmc_chain directly to cover the HMC sampling code path."""

    def _inputs(self):
        data = _gaussian_data(seed=20)
        mm, fr, S_lambda, fam = _build_components(data)
        X, y = mm.X, mm.response
        p = X.shape[1]
        M_diag = np.ones(p)
        return fr.coefficients, M_diag, X, y, S_lambda, fam, fr.scale

    def test_hmc_chain_returns_correct_shapes(self):
        """_hmc_chain returns (samples, step_size, acceptance_rate, 0.0, n_divergent)."""
        beta, M, X, y, S, fam, scale = self._inputs()
        samples, eps, rate, tree_depth, n_div = _hmc_chain(
            beta,
            M,
            X,
            y,
            S,
            fam,
            scale,
            None,
            None,
            n_samples=30,
            n_warmup=20,
            leapfrog_steps=5,
            step_size_init=0.1,
            target_accept=0.65,
            seed=1,
        )

        assert samples.shape == (len(beta), 30)
        assert eps > 0
        assert 0.0 <= rate <= 1.0
        assert tree_depth == 0.0  # HMC always returns 0.0 for tree depth
        assert isinstance(n_div, int)

    def test_hmc_chain_with_offset(self):
        """_hmc_chain runs correctly when an offset is provided."""
        beta, M, X, y, S, fam, scale = self._inputs()
        offset = np.zeros(len(y))
        samples, *_ = _hmc_chain(
            beta,
            M,
            X,
            y,
            S,
            fam,
            scale,
            None,
            offset,
            n_samples=20,
            n_warmup=10,
            leapfrog_steps=5,
            step_size_init=0.1,
            target_accept=0.65,
            seed=2,
        )

        assert samples.shape[1] == 20


class TestNUTSChainDirect:
    """Call _nuts_chain directly to cover the NUTS sampling code path."""

    def _inputs(self):
        data = _gaussian_data(seed=21)
        mm, fr, S_lambda, fam = _build_components(data)
        X, y = mm.X, mm.response
        p = X.shape[1]
        M_diag = np.ones(p)
        return fr.coefficients, M_diag, X, y, S_lambda, fam, fr.scale

    def test_nuts_chain_returns_correct_shapes(self):
        """_nuts_chain returns (samples, step_size, acceptance_rate, mean_depth, n_divergent)."""
        beta, M, X, y, S, fam, scale = self._inputs()
        samples, eps, rate, depth, n_div = _nuts_chain(
            beta,
            M,
            X,
            y,
            S,
            fam,
            scale,
            None,
            None,
            n_samples=30,
            n_warmup=20,
            max_tree_depth=5,
            step_size_init=0.1,
            target_accept=0.65,
            seed=3,
        )

        assert samples.shape == (len(beta), 30)
        assert eps > 0
        assert 0.0 <= rate <= 1.0
        assert 0.0 < depth <= 5.0
        assert isinstance(n_div, int)

    def test_nuts_chain_with_offset(self):
        """_nuts_chain runs correctly when an offset is provided."""
        beta, M, X, y, S, fam, scale = self._inputs()
        offset = np.zeros(len(y))
        samples, *_ = _nuts_chain(
            beta,
            M,
            X,
            y,
            S,
            fam,
            scale,
            None,
            offset,
            n_samples=20,
            n_warmup=10,
            max_tree_depth=5,
            step_size_init=0.1,
            target_accept=0.65,
            seed=4,
        )

        assert samples.shape[1] == 20


class TestBuildTreeDirect:
    """Call _build_tree directly to cover the NUTS tree-building code path."""

    def _system(self):
        data = _gaussian_data(seed=22)
        mm, fr, S_lambda, fam = _build_components(data)
        X, y = mm.X, mm.response
        p = X.shape[1]
        M_diag_inv = np.ones(p)
        beta = fr.coefficients.copy()
        U = _potential_U(beta, X, y, S_lambda, fam, fr.scale, None, None)
        return beta, X, y, S_lambda, fam, fr.scale, M_diag_inv, U

    def test_build_tree_base_case(self):
        """_build_tree with j=0 takes exactly one leapfrog step."""
        beta, X, y, S, fam, scale, M_inv, U = self._system()
        rng = np.random.default_rng(50)
        p_mom = rng.standard_normal(len(beta))
        eps = 0.05
        H0 = U + 0.5 * float(np.sum(p_mom**2))
        log_u = -H0 - 1.0

        result = _build_tree(
            beta,
            p_mom,
            log_u,
            1,
            0,
            eps,
            H0,
            M_inv,
            X,
            y,
            S,
            fam,
            scale,
            None,
            None,
            rng,
        )

        assert len(result) == 10  # 10-tuple

        beta_minus, p_minus, beta_plus, p_plus, beta_prime, n, s, a_s, na, nd = result

        assert beta_minus.shape == beta.shape
        assert isinstance(n, int)
        assert isinstance(nd, int)

    def test_build_tree_recursive(self):
        """_build_tree with j=1 recurses and returns a valid subtree."""
        beta, X, y, S, fam, scale, M_inv, U = self._system()
        rng = np.random.default_rng(51)
        p_mom = rng.standard_normal(len(beta))
        eps = 0.05
        H0 = U + 0.5 * float(np.sum(p_mom**2))
        log_u = -H0 - 1.0

        result = _build_tree(
            beta,
            p_mom,
            log_u,
            1,
            1,
            eps,
            H0,
            M_inv,
            X,
            y,
            S,
            fam,
            scale,
            None,
            None,
            rng,
        )

        assert len(result) == 10


# ---------------------------------------------------------------------------
# mcmc_fit with offset
# ---------------------------------------------------------------------------


class TestMCMCFitOffset:
    """Cover the offset branch inside mcmc_fit."""

    def test_mcmc_fit_with_offset(self):
        """mcmc_fit runs correctly when the model matrix has a non-None offset."""
        rng = np.random.default_rng(77)
        n = 100
        x = np.linspace(0, 1, n)
        offset = np.full(n, 0.5)
        lam = np.exp(np.sin(2 * np.pi * x) + 1.0 + offset)
        data = {"y": rng.poisson(lam).astype(float), "x": x, "off": offset}

        formula_obj = parse("y ~ s(x) + offset(off)")
        mm = build_model_matrix(formula_obj, data)
        fam = Poisson()
        mr = mcmc_fit(mm, fam, n_chains=1, n_samples=50, n_warmup=30, seed=5)

        assert mr.samples.shape[1] == 50
        assert mr.r_hat.shape[0] > 0
