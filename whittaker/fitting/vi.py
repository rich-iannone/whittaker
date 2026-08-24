"""Full-rank Gaussian variational inference for GAMs.

Implements ELBO optimization. The variational family is q(β) = N(m, C) with C = LL' (Cholesky
parameterization). The ELBO is maximized with Adam. The expected log-likelihood is approximated via
Gauss-Hermite quadrature so the code is family-agnostic and requires no autodiff framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite

if TYPE_CHECKING:
    from whittaker.families.base import Family
    from whittaker.fitting.pirls import FitResult
    from whittaker.model_matrix import ModelMatrix


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BayesResult:
    """Common base for Bayesian inference results (VI, MCMC).

    Provides a uniform interface so that downstream code (`simulate`,
    `predict`, `summary`) can work with any posterior approximation.
    """

    coefficients: NDArray
    posterior_mean: NDArray
    posterior_cov: NDArray
    linear_predictor: NDArray
    fitted_values: NDArray
    smoothing_params: list[float]
    scale: float
    edf: list[float]
    edf_total: float
    n_iter: int
    converged: bool
    weights: NDArray | None = None
    prior_weights: NDArray | None = None
    method: str = "Bayes"

    def draw(self, n: int, *, seed: int | None = None) -> NDArray:
        """Draw `n` coefficient vectors from the posterior.

        Returns
        -------
        NDArray
            Shape `(p, n)`.
        """
        rng = np.random.default_rng(seed)
        C = self.posterior_cov
        C = (C + C.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 0.0)
        L = eigvecs * np.sqrt(eigvals)[np.newaxis, :]
        z = rng.standard_normal((len(self.posterior_mean), n))
        return self.posterior_mean[:, np.newaxis] + L @ z


@dataclass
class VIResult(BayesResult):
    """Result of full-rank Gaussian variational inference for a GAM.

    Attributes
    ----------
    posterior_chol:
        Lower-triangular Cholesky factor `L` of the posterior covariance,
        so `posterior_cov = L @ L.T`.
    elbo_history:
        ELBO value recorded at each optimisation step.  Should be
        non-decreasing after the first few warm-up iterations; sustained
        decrease indicates the learning rate is too high.
    elbo:
        Final ELBO value at convergence.
    phi_variational:
        `True` when the scale parameter φ was included in the variational
        family (`phi_inference="variational"`).
    log_phi_mean:
        Posterior mean of log φ (only set when `phi_variational=True`).
    log_phi_var:
        Posterior variance of log φ (only set when `phi_variational=True`).
    """

    posterior_chol: NDArray = field(default_factory=lambda: np.empty(0))
    elbo_history: NDArray = field(default_factory=lambda: np.empty(0))
    elbo: float = float("-inf")
    phi_variational: bool = False
    log_phi_mean: float | None = None
    log_phi_var: float | None = None
    method: str = "VI"


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature helpers
# ---------------------------------------------------------------------------


def _gh_nodes_weights(n: int) -> tuple[NDArray, NDArray]:
    """Return nodes and weights for Gauss-Hermite quadrature.

    The nodes `z` and weights `w` satisfy:

        ∫ f(x) exp(-x²) dx ≈ Σ_k w_k f(z_k)

    For a standard Gaussian expectation E_{x~N(0,1)}[f(x)] we use:

        E[f(x)] ≈ (1/√π) Σ_k w_k f(√2 z_k)
    """
    z, w = roots_hermite(n)

    # Transform: standard normal convention
    # z_std = √2 * z_gh,  w_std = w_gh / √π
    z_std = np.sqrt(2.0) * z
    w_std = w / np.sqrt(np.pi)
    return z_std, w_std


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def _kl_gaussian_penalty(
    m: NDArray,
    L: NDArray,
    S_lambda: NDArray,
) -> float:
    """KL( N(m, LL') || p(β|λ) ) using the pseudo-inverse of S_λ.

    The smoothing prior is improper (rank-deficient) in the null space.  We
    handle this by computing the KL only over the penalised subspace:

        KL = 0.5 · [tr(S_λ C) + m' S_λ m − rank(S_λ) − log_pdet(S_λ) − log det(C)]

    where `log_pdet` is the log pseudo-determinant (sum of logs of non-zero
    eigenvalues) and `det(C) = prod(diag(L))²`.
    """
    # C = L @ L.T; tr(S_λ C) = tr(S_λ L L') = ||L' @ S_λ^{1/2}||_F²
    # but computing via tr(S_λ L L') is cheaper directly:
    SL = S_lambda @ L  # (p, p)
    trace_term = float(np.sum(L * SL))  # tr(S_λ LL') = tr(L' S_λ L)

    quad_term = float(m @ S_lambda @ m)

    # log pseudo-determinant of S_λ
    eigvals_S = np.linalg.eigvalsh(S_lambda)
    tol = np.max(np.abs(eigvals_S)) * S_lambda.shape[0] * np.finfo(float).eps * 10
    rank_S = int(np.sum(eigvals_S > tol))
    log_pdet_S = float(np.sum(np.log(eigvals_S[eigvals_S > tol])))

    # log det(C) = 2 * sum(log diag(L))
    log_det_C = 2.0 * float(np.sum(np.log(np.abs(np.diag(L)))))

    # KL(N(m,C) || N(0, S_λ^{-1})) = 0.5[tr(S_λC) + m'S_λm - rank - log_pdet(S_λ) - log det(C)]
    kl = 0.5 * (trace_term + quad_term - rank_S - log_pdet_S - log_det_C)
    return kl


# ---------------------------------------------------------------------------
# Expected log-likelihood via quadrature
# ---------------------------------------------------------------------------


def _expected_ll_quadrature(
    eta_mean: NDArray,
    eta_std: NDArray,
    y: NDArray,
    family: Family,
    scale: float,
    z_gh: NDArray,
    w_gh: NDArray,
    weights: NDArray | None = None,
) -> float:
    """Approximate E_q[log p(y | β)] via Gauss-Hermite quadrature.

    For each observation i:
        E[log p(y_i | η_i)] ≈ Σ_k w_k log p(y_i | g^{-1}(η_mean_i + σ_i z_k))

    Parameters
    ----------
    eta_mean:
        Mean linear predictor under q, shape `(n,)`.
    eta_std:
        Standard deviation of linear predictor under q, shape `(n,)`.
    y:
        Observed responses, shape `(n,)`.
    family:
        Response family (used only for `link_inverse` and `log_likelihood`).
    scale:
        Current scale estimate (φ).
    z_gh, w_gh:
        Gauss-Hermite nodes and weights (standard-normal convention).
    weights:
        Optional prior weights, shape `(n,)`.
    """
    ell = 0.0
    for _k, (zk, wk) in enumerate(zip(z_gh, w_gh, strict=True)):
        # Quadrature point on linear predictor scale
        eta_k = eta_mean + eta_std * zk  # (n,)
        mu_k = family.link_inverse(eta_k)

        # log p(y | mu_k): use log_likelihood / n to get per-observation average,
        # then weight by quadrature weight
        ll_k = family.log_likelihood(y, mu_k, scale, weights=weights)
        ell += wk * ll_k

    return float(ell)


# ---------------------------------------------------------------------------
# ELBO and gradients (finite-difference for L, analytic for m via KL)
# ---------------------------------------------------------------------------


def _elbo_and_grad(
    m: NDArray,
    L: NDArray,
    X: NDArray,
    y: NDArray,
    S_lambda: NDArray,
    family: Family,
    scale: float,
    z_gh: NDArray,
    w_gh: NDArray,
    weights: NDArray | None,
    offset: NDArray | None,
) -> tuple[float, NDArray, NDArray]:
    """Compute ELBO and gradients w.r.t. `m` and `L`.

    Returns
    -------
    elbo : float
    grad_m : NDArray, shape (p,)
    grad_L : NDArray, shape (p, p), lower-triangular only meaningful
    """
    # Linear predictor moments under q
    # η_i ~ N(x_i'm, x_i' C x_i);  C = LL'
    # η_mean = X @ m + offset
    # η_var_i = ||x_i' L||²  (O(np), not O(np²))
    XL = X @ L  # (n, p)
    eta_mean = X @ m
    if offset is not None:
        eta_mean = eta_mean + offset
    eta_var = np.sum(XL**2, axis=1)  # (n,)
    eta_std = np.sqrt(np.maximum(eta_var, 1e-12))

    # Expected log-likelihood
    ell = _expected_ll_quadrature(eta_mean, eta_std, y, family, scale, z_gh, w_gh, weights)

    # KL divergence
    kl = _kl_gaussian_penalty(m, L, S_lambda)

    elbo = ell - kl

    # --- Gradient w.r.t. m ---
    # ∂ELBO/∂m = ∂E_q[log p]/∂m − ∂KL/∂m
    # ∂KL/∂m = S_λ m  (analytic)
    # ∂E_q[log p]/∂m: reparametrisation where η_mean = X m, so
    #   ∂E_q/∂m = X' E_z[∂ log p / ∂η_mean]

    # Approximate the expectation via the same GH nodes:
    dEll_deta_mean = np.zeros(len(y))
    for zk, wk in zip(z_gh, w_gh, strict=True):
        eta_k = eta_mean + eta_std * zk
        mu_k = family.link_inverse(eta_k)
        dmu_deta = 1.0 / family.link_derivative(mu_k)
        var_k = family.variance(mu_k)
        # ∂ log p / ∂η = (y − μ) / (V(μ) · g'(μ))  for canonical exponential family
        d_logp_deta = (y - mu_k) * dmu_deta / (var_k * scale + 1e-15)
        if weights is not None:
            d_logp_deta = d_logp_deta * weights
        dEll_deta_mean += wk * d_logp_deta

    grad_m = X.T @ dEll_deta_mean - S_lambda @ m

    # --- Gradient w.r.t. L ---
    # ∂ELBO/∂L = ∂E_q[log p]/∂L − ∂KL/∂L
    # ∂KL/∂L  = S_λ L − L^{-T}  (analytic, but L^{-T} is expensive)
    #          = S_λ L − (L^T)^{-1}
    # ∂E_q[log p]/∂L: via reparametrisation, η_std_i = ||XL_i||/η_std_i · XL_i
    #   ∂E_q/∂L ≈ X' (dEll_deta_std ⊙ (1/η_std)) · L (chain rule, simplified)

    # Gradient of E_q[log p] w.r.t. η_std:
    dEll_deta_std = np.zeros(len(y))
    for zk, wk in zip(z_gh, w_gh, strict=True):
        eta_k = eta_mean + eta_std * zk
        mu_k = family.link_inverse(eta_k)
        dmu_deta = 1.0 / family.link_derivative(mu_k)
        var_k = family.variance(mu_k)
        d_logp_deta = (y - mu_k) * dmu_deta / (var_k * scale + 1e-15)
        if weights is not None:
            d_logp_deta = d_logp_deta * weights
        # ∂η_k/∂η_std = zk, so ∂E/∂η_std_i += wk * zk * d_logp_deta_i
        dEll_deta_std += wk * zk * d_logp_deta

    # ∂η_std_i/∂L = XL_i / η_std_i  (row vector)
    # ∂E_q/∂L = X' diag(dEll_deta_std / η_std) X L
    scale_vec = dEll_deta_std / eta_std  # (n,)
    grad_L_ell = X.T @ (scale_vec[:, np.newaxis] * XL)  # (p, p)

    # ∂KL/∂L:
    # KL = 0.5 [tr(S_λ LL') + m' S_λ m − rank − log_pdet(S) − log det(C)]
    # ∂KL/∂L = S_λ L  (from trace term)  − L^{-T} (from log det(C) = 2 sum log |diag(L)|)
    # For numerical stability, compute L^{-T} via triangular solve.
    try:
        L_inv_T = np.linalg.solve(L.T, np.eye(L.shape[0]))  # (p, p)
    except np.linalg.LinAlgError:  # pragma: no cover
        L_inv_T = np.zeros_like(L)

    grad_L_kl = S_lambda @ L - L_inv_T

    # Apply lower-triangular mask (only lower triangle is free)
    mask = np.tril(np.ones_like(L))
    grad_L = (grad_L_ell - grad_L_kl) * mask

    return elbo, grad_m, grad_L


# ---------------------------------------------------------------------------
# Adam optimiser state
# ---------------------------------------------------------------------------


def _adam_init(m: NDArray, L: NDArray) -> dict:
    return {
        "t": 0,
        "m_m": np.zeros_like(m),
        "v_m": np.zeros_like(m),
        "m_L": np.zeros_like(L),
        "v_L": np.zeros_like(L),
    }


def _adam_step(
    m: NDArray,
    L: NDArray,
    grad_m: NDArray,
    grad_L: NDArray,
    state: dict,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[NDArray, NDArray, dict]:
    """One Adam update step for (m, L)."""
    state = dict(state)
    state["t"] += 1
    t = state["t"]

    # m update
    state["m_m"] = beta1 * state["m_m"] + (1 - beta1) * grad_m
    state["v_m"] = beta2 * state["v_m"] + (1 - beta2) * grad_m**2
    m_hat = state["m_m"] / (1 - beta1**t)
    v_hat = state["v_m"] / (1 - beta2**t)
    m_new = m + lr * m_hat / (np.sqrt(v_hat) + eps)

    # L update
    state["m_L"] = beta1 * state["m_L"] + (1 - beta1) * grad_L
    state["v_L"] = beta2 * state["v_L"] + (1 - beta2) * grad_L**2
    m_hat_L = state["m_L"] / (1 - beta1**t)
    v_hat_L = state["v_L"] / (1 - beta2**t)
    L_new = L + lr * m_hat_L / (np.sqrt(v_hat_L) + eps)

    # Keep L lower-triangular with positive diagonal
    L_new = np.tril(L_new)
    diag_idx = np.arange(L_new.shape[0])
    L_new[diag_idx, diag_idx] = np.abs(L_new[diag_idx, diag_idx])

    # Ensure diagonal is strictly positive (softplus-like floor)
    L_new[diag_idx, diag_idx] = np.maximum(L_new[diag_idx, diag_idx], 1e-8)

    return m_new, L_new, state


# ---------------------------------------------------------------------------
# EDF helper (reuse pirls logic)
# ---------------------------------------------------------------------------


def _vi_edf(
    X: NDArray,
    penalties: list[NDArray],
    sp: list[float],
    smooths_info: list[tuple[int, int]],
    W: NDArray | None = None,
) -> list[float]:
    """Compute EDF per smooth from the posterior mean (same formula as P-IRLS)."""
    from whittaker.fitting.pirls import _edf_per_smooth

    return _edf_per_smooth(X, penalties, sp, smooths_info, W=W)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def vi_fit(
    model: ModelMatrix,
    family: Family,
    *,
    smoothing_params: list[float] | None = None,
    prior_weights: NDArray | None = None,
    n_quad: int = 20,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    patience: int = 5,
    seed: int | None = None,
    cov_structure: str = "full",
    phi_inference: str = "fixed",
    init_result: FitResult | None = None,
) -> VIResult:
    """Fit a GAM using full-rank Gaussian variational inference.

    Parameters
    ----------
    model:
        Model matrix from `build_model_matrix()`.
    family:
        Response distribution family.
    smoothing_params:
        Fixed smoothing parameters.  If `None`, P-IRLS with REML is run
        first to estimate them, then VI refines the posterior.
    prior_weights:
        Observation weights, shape `(n,)`.
    n_quad:
        Number of Gauss-Hermite quadrature points (the default is 20).
    lr:
        Adam learning rate (the default is 0.01).
    max_iter:
        Maximum optimization iterations (the default is 1000).
    tol:
        Convergence threshold on relative ELBO change (the default is 1e-4).
    patience:
        Number of consecutive steps below `tol` before stopping (the default is 5).
    seed:
        Random seed (not used in the deterministic full-data path, reserved for future stochastic
        VI).
    cov_structure:
        `"full"` (the default) (full-rank Cholesky; `"block"`) is block-diagonal Cholesky with one
        block per smooth term (cheaper for large p).
    phi_inference:
        `"fixed"` (default): φ fixed at P-IRLS estimate;
        `"variational"`: φ included in the variational family as
        `q(φ) = LogNormal(μ_φ, σ_φ²)` (experimental).
    init_result:
        Optional pre-computed `FitResult` from `pirls_fit()` to use as the warm start. If `None`,
        `pirls_fit()` is called internally with `method="REML"`.

    Returns
    -------
    VIResult
    """
    from whittaker.families.gaussian import Gaussian
    from whittaker.fitting.inference import _bayesian_covariance
    from whittaker.fitting.pirls import pirls_fit

    X = model.X
    y = model.response
    n, p = X.shape
    offset = model.offset
    penalties = model.penalties
    pw = prior_weights

    # ------------------------------------------------------------------
    # Step 1: warm-start from P-IRLS
    # ------------------------------------------------------------------
    if init_result is None:
        pirls_method = "REML" if penalties else "GCV"
        init_result = pirls_fit(
            model,
            family,
            smoothing_params=smoothing_params,
            method=pirls_method,
            prior_weights=pw,
        )

    sp: list[float] = list(init_result.smoothing_params)
    scale: float = init_result.scale

    # ------------------------------------------------------------------
    # Fast path: for Gaussian response with identity link, the Laplace
    # approximation is the exact posterior (skip the optimisation loop).
    # ------------------------------------------------------------------
    if isinstance(family, Gaussian):
        V_fast = _bayesian_covariance(X, penalties, sp, init_result.scale, W=init_result.weights)
        V_fast = (V_fast + V_fast.T) * 0.5
        jitter = 1e-8 * np.eye(p)
        try:
            L_fast = np.linalg.cholesky(V_fast + jitter)
        except np.linalg.LinAlgError:  # pragma: no cover
            eigvals_f, eigvecs_f = np.linalg.eigh(V_fast)
            eigvals_f = np.maximum(eigvals_f, 1e-8)
            L_fast = eigvecs_f @ np.diag(np.sqrt(eigvals_f))

        m_fast = init_result.coefficients.copy()
        eta_fast = X @ m_fast
        if offset is not None:
            eta_fast = eta_fast + offset
        mu_fast = family.link_inverse(eta_fast)
        smooths_info_fast = [(s.col_start, s.col_end) for s in model.smooths]
        edf_fast = _vi_edf(X, penalties, sp, smooths_info_fast, W=init_result.weights)
        edf_total_fast = sum(edf_fast) + (1 if model.has_intercept else 0) + model.n_parametric
        return VIResult(
            coefficients=m_fast,
            posterior_mean=m_fast,
            posterior_cov=L_fast @ L_fast.T,
            posterior_chol=L_fast,
            linear_predictor=eta_fast,
            fitted_values=mu_fast,
            smoothing_params=sp,
            scale=init_result.scale,
            edf=edf_fast,
            edf_total=edf_total_fast,
            n_iter=0,
            converged=True,
            weights=init_result.weights,
            prior_weights=pw,
            elbo_history=np.empty(0),
            elbo=float("nan"),
            method="VI",
        )

    # ------------------------------------------------------------------
    # Step 2: build S_λ and initialise (m, L)
    # ------------------------------------------------------------------
    S_lambda = np.zeros((p, p))
    for lam, pen in zip(sp, penalties, strict=False):
        S_lambda += lam * pen

    # Posterior covariance from Laplace approx as warm start
    W_init = init_result.weights  # IRLS weights at convergence
    V_beta = _bayesian_covariance(X, penalties, sp, scale, W=W_init)
    V_beta = (V_beta + V_beta.T) * 0.5

    # Cholesky of V_beta; add small jitter for numerical stability
    jitter = 1e-8 * np.eye(p)
    try:
        L = np.linalg.cholesky(V_beta + jitter)
    except np.linalg.LinAlgError:  # pragma: no cover
        eigvals, eigvecs = np.linalg.eigh(V_beta)
        eigvals = np.maximum(eigvals, 1e-8)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    if cov_structure == "block":
        # Zero out cross-block entries: one block per smooth term
        block_mask = np.zeros((p, p), dtype=bool)
        if model.smooths:
            for s_info in model.smooths:
                s, e = s_info.col_start, s_info.col_end
                block_mask[s:e, s:e] = True

        # Parametric / intercept columns form their own block
        param_cols = [
            j for j in range(p) if not any(s.col_start <= j < s.col_end for s in model.smooths)
        ]
        if param_cols:
            idx = np.array(param_cols)
            block_mask[np.ix_(idx, idx)] = True
        L = np.tril(L) * np.tril(block_mask)
        diag_idx = np.arange(p)
        L[diag_idx, diag_idx] = np.maximum(np.abs(np.diag(L)), 1e-8)

    m = init_result.coefficients.copy()

    # ------------------------------------------------------------------
    # Step 3: Gauss-Hermite nodes and weights
    # ------------------------------------------------------------------
    z_gh, w_gh = _gh_nodes_weights(n_quad)

    # ------------------------------------------------------------------
    # Step 4: Adam optimisation loop
    # ------------------------------------------------------------------
    adam_state = _adam_init(m, L)
    elbo_history: list[float] = []
    prev_elbo = float("-inf")
    patience_count = 0
    converged = False

    for iteration in range(max_iter):
        elbo, grad_m, grad_L = _elbo_and_grad(
            m, L, X, y, S_lambda, family, scale, z_gh, w_gh, pw, offset
        )
        elbo_history.append(elbo)

        m, L, adam_state = _adam_step(m, L, grad_m, grad_L, adam_state, lr=lr)

        # Convergence check on relative ELBO change
        if iteration > 0:
            rel_change = abs(elbo - prev_elbo) / (abs(prev_elbo) + 1.0)
            if rel_change < tol:
                patience_count += 1
                if patience_count >= patience:
                    converged = True
                    break
            else:
                patience_count = 0

        prev_elbo = elbo

    # ------------------------------------------------------------------
    # Step 5: optional variational φ (LogNormal marginal)
    # ------------------------------------------------------------------
    phi_variational = phi_inference == "variational"
    log_phi_mean: float | None = None
    log_phi_var: float | None = None

    if phi_variational and not family.scale_known:
        # Simple moment-matching: use residuals under posterior mean
        eta_mean = X @ m
        if offset is not None:
            eta_mean = eta_mean + offset
        mu_hat = family.link_inverse(eta_mean)
        resid_sq = (y - mu_hat) ** 2
        if pw is not None:
            n_eff = float(np.sum(pw))
            dev_approx = float(np.sum(pw * resid_sq))
        else:
            n_eff = float(n)
            dev_approx = float(np.sum(resid_sq))

        # Simple log-normal approximation: mean and variance of log φ
        smooths_info = [(s.col_start, s.col_end) for s in model.smooths]
        edf_vals = _vi_edf(X, penalties, sp, smooths_info, W=W_init)
        edf_tot = sum(edf_vals) + (1 if model.has_intercept else 0) + model.n_parametric
        dof = max(n_eff - edf_tot, 1.0)
        phi_est = dev_approx / dof
        log_phi_mean = float(np.log(phi_est))
        log_phi_var = float(2.0 / dof)  # delta-method approximation
        scale = phi_est

    # ------------------------------------------------------------------
    # Step 6: assemble VIResult
    # ------------------------------------------------------------------
    eta_final = X @ m
    if offset is not None:
        eta_final = eta_final + offset
    mu_final = family.link_inverse(eta_final)

    smooths_info = [(s.col_start, s.col_end) for s in model.smooths]
    edf_vals = _vi_edf(X, penalties, sp, smooths_info, W=W_init)
    edf_total = sum(edf_vals) + (1 if model.has_intercept else 0) + model.n_parametric

    C = L @ L.T

    return VIResult(
        coefficients=m,
        posterior_mean=m,
        posterior_cov=C,
        posterior_chol=L,
        linear_predictor=eta_final,
        fitted_values=mu_final,
        smoothing_params=sp,
        scale=scale,
        edf=edf_vals,
        edf_total=edf_total,
        n_iter=len(elbo_history),
        converged=converged,
        weights=W_init,
        prior_weights=pw,
        elbo_history=np.array(elbo_history),
        elbo=elbo_history[-1] if elbo_history else float("-inf"),
        phi_variational=phi_variational,
        log_phi_mean=log_phi_mean,
        log_phi_var=log_phi_var,
        method="VI",
    )
