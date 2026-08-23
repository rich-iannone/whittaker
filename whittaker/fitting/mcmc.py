"""Hamiltonian Monte Carlo (HMC) posterior sampling for GAMs.

Static-L HMC with diagonal mass matrix, Nesterov dual-averaging step-size adaptation, and
n_chains independent chains run in parallel via concurrent.futures.ProcessPoolExecutor.
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from whittaker.families.base import Family
    from whittaker.fitting.pirls import FitResult
    from whittaker.model_matrix import ModelMatrix

from whittaker.fitting.vi import BayesResult

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MCMCResult(BayesResult):
    """Result of HMC posterior sampling for a GAM.

    Attributes
    ----------
    samples:
        All retained posterior draws, shape `(p, n_chains * n_samples)`.
        Warmup draws are excluded.
    r_hat:
        Per-parameter Gelman-Rubin statistic, shape `(p,)`.  Values close
        to 1.0 indicate convergence across chains; values above 1.1 warrant
        concern.
    ess:
        Per-parameter effective sample size, shape `(p,)`.
    acceptance_rate:
        Mean Metropolis acceptance rate across all chains and post-warmup steps.
    step_size:
        Final adapted leapfrog step size ε (averaged across chains).
    n_chains:
        Number of independent Markov chains.
    n_samples:
        Post-warmup draws per chain.
    n_warmup:
        Warmup (discarded) draws per chain.
    """

    samples: NDArray = field(default_factory=lambda: np.empty(0))
    r_hat: NDArray = field(default_factory=lambda: np.empty(0))
    ess: NDArray = field(default_factory=lambda: np.empty(0))
    acceptance_rate: float = 0.0
    step_size: float = 0.0
    n_chains: int = 1
    n_samples: int = 0
    n_warmup: int = 0
    method: str = "MCMC"

    def draw(self, n: int, *, seed: int | None = None) -> NDArray:
        """Draw `n` coefficient vectors with replacement from posterior draws.

        Returns
        -------
        NDArray
            Shape `(p, n)`.
        """
        rng = np.random.default_rng(seed)
        n_total = self.samples.shape[1]
        idx = rng.integers(0, n_total, size=n)
        return self.samples[:, idx]


# ---------------------------------------------------------------------------
# Gradient of the negative log-posterior
# ---------------------------------------------------------------------------


def _grad_U(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    S_lambda: NDArray,
    family: Family,
    scale: float,
    weights: NDArray | None,
    offset: NDArray | None,
) -> NDArray:
    """Gradient of U(β) = −log p(y|β) − log p(β|λ) w.r.t. β.

    For exponential-family likelihoods:
        ∂/∂β log p(y|β) = X' r,  r_i = (y_i − μ_i) / (V(μ_i) g'(μ_i) φ)
    so:
        ∇U(β) = −X'r + S_λ β
    """
    eta = X @ beta
    if offset is not None:
        eta = eta + offset
    mu = family.link_inverse(eta)
    dmu_deta = 1.0 / family.link_derivative(mu)
    var_mu = family.variance(mu)
    r = (y - mu) * dmu_deta / (var_mu * scale + 1e-15)
    if weights is not None:
        r = r * weights
    # Penalty uses S_λ/scale so that the gradient is zero at the P-IRLS MAP.
    # Whittaker's P-IRLS uses unit IRLS weights, so the implied posterior prior
    # has precision S_λ/scale (not S_λ).  For families with scale=1 (Poisson,
    # Binomial), this is a no-op.
    return -(X.T @ r) + (S_lambda @ beta) / scale


def _potential_U(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    S_lambda: NDArray,
    family: Family,
    scale: float,
    weights: NDArray | None,
    offset: NDArray | None,
) -> float:
    """U(β) = −log p(y|β) + ½ β' (S_λ/scale) β."""
    eta = X @ beta
    if offset is not None:
        eta = eta + offset
    mu = family.link_inverse(eta)
    ll = family.log_likelihood(y, mu, scale, weights=weights)
    penalty = 0.5 * float(beta @ (S_lambda @ beta)) / scale
    return -ll + penalty


# ---------------------------------------------------------------------------
# Leapfrog integrator
# ---------------------------------------------------------------------------


def _leapfrog(
    beta: NDArray,
    p_mom: NDArray,
    M_diag_inv: NDArray,
    step_size: float,
    n_steps: int,
    X: NDArray,
    y: NDArray,
    S_lambda: NDArray,
    family: Family,
    scale: float,
    weights: NDArray | None,
    offset: NDArray | None,
) -> tuple[NDArray, NDArray]:
    """Run `n_steps` leapfrog steps.

    Symplectic integration of Hamilton's equations:
        dβ/dt =  M⁻¹ p
        dp/dt = −∇U(β)

    Returns
    -------
    (beta_new, p_new) : tuple of NDArray
    """
    grad = _grad_U(beta, X, y, S_lambda, family, scale, weights, offset)
    p_mom = p_mom - 0.5 * step_size * grad

    for _ in range(n_steps - 1):
        beta = beta + step_size * M_diag_inv * p_mom
        grad = _grad_U(beta, X, y, S_lambda, family, scale, weights, offset)
        p_mom = p_mom - step_size * grad

    beta = beta + step_size * M_diag_inv * p_mom
    grad = _grad_U(beta, X, y, S_lambda, family, scale, weights, offset)
    p_mom = p_mom - 0.5 * step_size * grad

    return beta, p_mom


# ---------------------------------------------------------------------------
# Single HMC chain
# ---------------------------------------------------------------------------


def _hmc_chain(
    beta_init: NDArray,
    M_diag: NDArray,
    X: NDArray,
    y: NDArray,
    S_lambda: NDArray,
    family: Family,
    scale: float,
    weights: NDArray | None,
    offset: NDArray | None,
    n_samples: int,
    n_warmup: int,
    leapfrog_steps: int,
    step_size_init: float,
    target_accept: float,
    seed: int,
) -> tuple[NDArray, float, float]:
    """Run one HMC chain with dual-averaging step-size adaptation.

    Returns
    -------
    samples : NDArray, shape (p, n_samples)
    adapted_step_size : float
    acceptance_rate : float
        Post-warmup acceptance rate.
    """
    rng = np.random.default_rng(seed)
    M_diag_inv = 1.0 / M_diag
    p = len(beta_init)

    beta = beta_init.copy()
    U_curr = _potential_U(beta, X, y, S_lambda, family, scale, weights, offset)

    # Dual-averaging parameters (Nesterov 2009, as used in Stan)
    eps = step_size_init
    mu = float(np.log(10.0 * eps))
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    # --- Warmup ---
    for m in range(1, n_warmup + 1):
        p_mom = rng.standard_normal(p) * np.sqrt(M_diag)
        K_curr = 0.5 * float(np.sum(p_mom**2 * M_diag_inv))

        beta_prop, p_prop = _leapfrog(
            beta,
            p_mom,
            M_diag_inv,
            eps,
            leapfrog_steps,
            X,
            y,
            S_lambda,
            family,
            scale,
            weights,
            offset,
        )
        U_prop = _potential_U(beta_prop, X, y, S_lambda, family, scale, weights, offset)
        K_prop = 0.5 * float(np.sum(p_prop**2 * M_diag_inv))

        log_accept = -(U_prop + K_prop) + (U_curr + K_curr)
        alpha = min(1.0, float(np.exp(np.clip(log_accept, -700.0, 700.0))))

        # Dual-averaging update
        H_bar = (1.0 - 1.0 / (m + t0)) * H_bar + (1.0 / (m + t0)) * (target_accept - alpha)
        log_eps = mu - (float(np.sqrt(m)) / gamma) * H_bar
        eps = float(np.exp(log_eps))
        log_eps_bar = m ** (-kappa) * log_eps + (1.0 - m ** (-kappa)) * log_eps_bar

        if rng.uniform() < alpha:
            beta = beta_prop
            U_curr = U_prop

    # Switch to averaged step size
    eps = float(np.exp(log_eps_bar))

    # --- Sampling ---
    samples = np.empty((p, n_samples))
    n_accepted = 0

    for t in range(n_samples):
        p_mom = rng.standard_normal(p) * np.sqrt(M_diag)
        K_curr = 0.5 * float(np.sum(p_mom**2 * M_diag_inv))

        beta_prop, p_prop = _leapfrog(
            beta,
            p_mom,
            M_diag_inv,
            eps,
            leapfrog_steps,
            X,
            y,
            S_lambda,
            family,
            scale,
            weights,
            offset,
        )
        U_prop = _potential_U(beta_prop, X, y, S_lambda, family, scale, weights, offset)
        K_prop = 0.5 * float(np.sum(p_prop**2 * M_diag_inv))

        log_accept = -(U_prop + K_prop) + (U_curr + K_curr)
        alpha = min(1.0, float(np.exp(np.clip(log_accept, -700.0, 700.0))))

        if rng.uniform() < alpha:
            beta = beta_prop
            U_curr = U_prop
            n_accepted += 1

        samples[:, t] = beta

    return samples, eps, n_accepted / n_samples


# Top-level wrapper (must be at module level for ProcessPoolExecutor pickling)
def _chain_worker(
    beta_init: NDArray,
    M_diag: NDArray,
    X: NDArray,
    y: NDArray,
    S_lambda: NDArray,
    family: Family,
    scale: float,
    weights: NDArray | None,
    offset: NDArray | None,
    n_samples: int,
    n_warmup: int,
    leapfrog_steps: int,
    step_size_init: float,
    target_accept: float,
    seed: int,
) -> tuple[NDArray, float, float]:
    return _hmc_chain(
        beta_init,
        M_diag,
        X,
        y,
        S_lambda,
        family,
        scale,
        weights,
        offset,
        n_samples,
        n_warmup,
        leapfrog_steps,
        step_size_init,
        target_accept,
        seed,
    )


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


def _r_hat(chains: NDArray) -> NDArray:
    """Gelman-Rubin R-hat statistic.

    Parameters
    ----------
    chains : NDArray, shape `(n_chains, n_samples, p)`

    Returns
    -------
    NDArray, shape `(p,)`
    """
    n_chains, n_samples, _ = chains.shape
    chain_means = chains.mean(axis=1)  # (n_chains, p)

    B = n_samples * np.var(chain_means, axis=0, ddof=1)  # between-chain variance × n
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)  # within-chain variance

    var_plus = ((n_samples - 1) / n_samples) * W + B / n_samples
    return np.sqrt(var_plus / (W + 1e-15))


def _ess(chains: NDArray) -> NDArray:
    """Effective sample size via Geyer's initial positive sequence.

    Parameters
    ----------
    chains : NDArray, shape `(n_chains, n_samples, p)`

    Returns
    -------
    NDArray, shape `(p,)`
    """
    n_chains, n_samples, p = chains.shape
    n_total = n_chains * n_samples

    ess = np.empty(p)
    for j in range(p):
        pooled = chains[:, :, j].ravel()
        pooled -= pooled.mean()

        # FFT-based normalized autocorrelation
        n_pad = 2 * n_total
        fft = np.fft.rfft(pooled, n=n_pad)
        acf_raw = np.fft.irfft(fft * np.conj(fft))[:n_total].real
        acv0 = acf_raw[0]
        if acv0 < 1e-15:
            ess[j] = float(n_total)
            continue
        acf = acf_raw / acv0

        # Geyer's initial positive sequence: sum pairs until non-positive
        rho_sum = 0.0
        for t in range(1, n_total // 2):
            pair = acf[2 * t] + acf[2 * t + 1]
            if pair <= 0.0:
                break
            rho_sum += pair

        ess[j] = n_total / (1.0 + 2.0 * rho_sum)

    return ess


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def mcmc_fit(
    model: ModelMatrix,
    family: Family,
    *,
    smoothing_params: list[float] | None = None,
    prior_weights: NDArray | None = None,
    n_samples: int = 1000,
    n_warmup: int = 500,
    n_chains: int = 4,
    leapfrog_steps: int = 10,
    target_accept: float = 0.65,
    seed: int | None = None,
    init_result: FitResult | None = None,
) -> MCMCResult:
    """Fit a GAM using Hamiltonian Monte Carlo.

    Parameters
    ----------
    model:
        Model matrix from `build_model_matrix()`.
    family:
        Response distribution family.
    smoothing_params:
        Fixed smoothing parameters.  If `None`, P-IRLS with REML is run
        first to estimate them; MCMC then samples the posterior with λ fixed.
    prior_weights:
        Observation weights, shape `(n,)`.
    n_samples:
        Post-warmup draws per chain (default 1000).
    n_warmup:
        Warmup draws per chain (default 500).  These are discarded.
    n_chains:
        Number of independent Markov chains (default 4).
    leapfrog_steps:
        Number of leapfrog steps per proposal `L` (default 10).
    target_accept:
        Target Metropolis acceptance rate for dual-averaging (default 0.65).
    seed:
        Master random seed.  Each chain is seeded deterministically from this.
    init_result:
        Optional pre-computed `FitResult` to use as warm start.  If
        `None`, `pirls_fit()` is called internally with `method="REML"`.

    Returns
    -------
    MCMCResult
    """
    from whittaker.fitting.inference import _bayesian_covariance
    from whittaker.fitting.pirls import _edf_per_smooth, pirls_fit

    X = model.X
    y = model.response
    _n, p = X.shape
    offset = model.offset
    penalties = model.penalties
    pw = prior_weights

    # 1. Warm-start from P-IRLS
    if init_result is None:
        pirls_method = "REML" if penalties else "GCV"
        init_result = pirls_fit(
            model,
            family,
            smoothing_params=smoothing_params,
            method=pirls_method,
            prior_weights=pw,
        )

    sp = list(init_result.smoothing_params)
    scale = init_result.scale

    # 2. Total penalty matrix S_λ = Σ λ_j S_j
    S_lambda = np.zeros((p, p))
    for lam, pen in zip(sp, penalties, strict=False):
        S_lambda += lam * pen

    # 3. Diagonal mass matrix M = diag(1/V_β) = diagonal precision.
    #
    # With this convention the momentum p_i ~ N(0, M_ii) has std 1/sqrt(V_β_ii) and the
    # leapfrog position step ε M^{-1} p = ε V_β_ii p_i ~ N(0, ε² V_β_ii), which is proportional
    # to the posterior standard deviation sqrt(V_β_ii).  This gives step size ε = O(1) regardless
    # of the absolute scale of the posterior.
    W_init = init_result.weights
    V_beta = _bayesian_covariance(X, penalties, sp, scale, W=W_init)
    V_diag = np.maximum(np.diag(V_beta), 1e-15)
    M_diag = 1.0 / V_diag  # precision; M_diag_inv inside _hmc_chain = V_diag

    # Stan-style initial step size: O(1) in the whitened (unit-mass) coordinate system.
    step_size_init = 0.3 / float(p**0.25)

    # 4. Per-chain seeds and dispersed initializations from MAP
    rng_master = np.random.default_rng(seed)
    chain_seeds = rng_master.integers(0, 2**31, size=n_chains).tolist()

    beta_map = init_result.coefficients.copy()
    chain_inits: list[NDArray] = []
    for k in range(n_chains):
        perturb_rng = np.random.default_rng(int(chain_seeds[k]) + 1_000_000)
        # Perturb by 0.1 * posterior_std in each direction (V_diag = posterior variance)
        noise = perturb_rng.standard_normal(p) * np.sqrt(0.1 * V_diag)
        chain_inits.append(beta_map + noise)

    # 5. Run chains in parallel (fall back to sequential if pickling fails)
    chain_args = [
        (
            chain_inits[k],
            M_diag,
            X,
            y,
            S_lambda,
            family,
            scale,
            pw,
            offset,
            n_samples,
            n_warmup,
            leapfrog_steps,
            step_size_init,
            target_accept,
            int(chain_seeds[k]),
        )
        for k in range(n_chains)
    ]
    n_workers = min(n_chains, os.cpu_count() or 1)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_chain_worker, *args) for args in chain_args]
            chain_results = [f.result() for f in futures]
    except Exception:
        # Sequential fallback when families or the runtime environment prevent forking
        chain_results = [_chain_worker(*args) for args in chain_args]

    # 6. Collect samples and scalar summaries
    # each chain_result: (samples (p, n_samples), step_size, acceptance_rate)
    all_samples = np.concatenate([cr[0] for cr in chain_results], axis=1)  # (p, n_chains*n_samples)
    final_step_size = float(np.mean([cr[1] for cr in chain_results]))
    acceptance_rate = float(np.mean([cr[2] for cr in chain_results]))

    # 7. Convergence diagnostics over (n_chains, n_samples, p) array
    chains_3d = np.stack([cr[0].T for cr in chain_results], axis=0)  # (n_chains, n_samples, p)
    r_hat = _r_hat(chains_3d)
    ess_vals = _ess(chains_3d)

    # 8. Posterior mean and covariance from samples
    posterior_mean = all_samples.mean(axis=1)
    posterior_cov = np.cov(all_samples) if all_samples.shape[1] > 1 else np.diag(M_diag)

    # 9. EDF (evaluated at posterior mean, same formula as P-IRLS)
    smooths_info = [(s.col_start, s.col_end) for s in model.smooths]
    edf_vals = _edf_per_smooth(X, penalties, sp, smooths_info, W=W_init)
    edf_total = sum(edf_vals) + (1 if model.has_intercept else 0) + model.n_parametric

    # 10. Fitted values and residuals at posterior mean
    eta_final = X @ posterior_mean
    if offset is not None:
        eta_final = eta_final + offset
    mu_final = family.link_inverse(eta_final)

    return MCMCResult(
        coefficients=posterior_mean,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        linear_predictor=eta_final,
        fitted_values=mu_final,
        smoothing_params=sp,
        scale=scale,
        edf=edf_vals,
        edf_total=edf_total,
        n_iter=n_warmup + n_samples,
        converged=True,
        weights=W_init,
        prior_weights=pw,
        samples=all_samples,
        r_hat=r_hat,
        ess=ess_vals,
        acceptance_rate=acceptance_rate,
        step_size=final_step_size,
        n_chains=n_chains,
        n_samples=n_samples,
        n_warmup=n_warmup,
        method="MCMC",
    )
