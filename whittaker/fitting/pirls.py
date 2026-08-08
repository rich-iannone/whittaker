"""Penalized Iteratively Reweighted Least Squares (P-IRLS) fitting.

This module implements the core GAM fitting algorithm following Wood (2017,
*Generalized Additive Models: An Introduction with R*, 2nd ed.).

For Gaussian response with identity link the P-IRLS loop collapses to a single penalized least
squares solve:

    (X'X + Σ λ_j S_j) β = X'y

For non-Gaussian families (future), the full iterative algorithm applies:

    1. Form pseudo-data z = η + (y − μ) / (∂μ/∂η)
    2. Form working weights W = (∂μ/∂η)² / V(μ)
    3. Solve (X'WX + Σ λ_j S_j) β = X'Wz
    4. Update η = Xβ, μ = g⁻¹(η)
    5. Repeat until convergence

Smoothing parameters λ are selected by minimizing GCV score over a grid, with optional refinement
via Brent's method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar

from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.model_matrix import ModelMatrix


@dataclass
class FitResult:
    """Result of a GAM fit.

    Attributes
    ----------
    coefficients:
        Estimated coefficient vector β, shape `(p,)`.
    linear_predictor:
        η = Xβ + offset, shape `(n,)`.
    fitted_values:
        μ = g⁻¹(η), shape `(n,)`.
    smoothing_params:
        Smoothing parameter λ_j for each penalty (one per penalty matrix).
    scale:
        Estimated scale parameter φ (σ² for Gaussian).
    gcv_score:
        GCV score at the fitted smoothing parameters.
    edf:
        Effective degrees of freedom per smooth term.
    edf_total:
        Total model effective degrees of freedom (trace of hat matrix).
    deviance:
        Model deviance at convergence.
    n_iter:
        Number of P-IRLS iterations (1 for Gaussian/identity).
    converged:
        Whether the P-IRLS loop converged.
    hat_matrix_trace:
        Trace of the hat (influence) matrix.
    residuals:
        Working residuals y − μ, shape `(n,)`.
    """

    coefficients: NDArray
    linear_predictor: NDArray
    fitted_values: NDArray
    smoothing_params: list[float]
    scale: float
    gcv_score: float
    edf: list[float]
    edf_total: float
    deviance: float
    n_iter: int
    converged: bool
    hat_matrix_trace: float
    residuals: NDArray
    weights: NDArray | None = None
    prior_weights: NDArray | None = None
    null_deviance: float | None = None
    aic: float | None = None
    bic: float | None = None


def _penalized_solve(
    X: NDArray,
    y: NDArray,
    penalties: list[NDArray],
    sp: list[float],
    W: NDArray | None = None,
    offset: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Solve the penalized (weighted) least squares system.

    Solves `(X'WX + Σ λ_j S_j) β = X'Wz` via Cholesky factorization.

    Returns `(β, hat_matrix_diagonal)` where the hat matrix diagonal is computed from the Cholesky
    factor for use in GCV/EDF calculations.
    """
    n, p = X.shape

    z = y.copy()
    if offset is not None:
        z = z - offset

    if W is not None:
        sqrtW = np.sqrt(W)
        Xw = X * sqrtW[:, np.newaxis]
        zw = z * sqrtW
    else:
        Xw = X
        zw = z

    XtX = Xw.T @ Xw
    Xtz = Xw.T @ zw

    S_total = np.zeros((p, p))
    for lam, pen in zip(sp, penalties):
        S_total += lam * pen

    A = XtX + S_total
    A = (A + A.T) * 0.5
    # Small ridge for numerical stability when penalty leaves unpenalized columns
    ridge = np.finfo(float).eps * max(np.trace(A) / p, 1.0)
    A[np.diag_indices_from(A)] += ridge

    cho, lower = cho_factor(A)
    beta = cho_solve((cho, lower), Xtz)

    # Hat matrix: H = X (X'WX + S)⁻¹ X'W
    # trace(H) = trace((X'WX + S)⁻¹ X'WX)
    # Efficient: solve A⁻¹ (X'WX) and take trace
    A_inv_XtWX = cho_solve((cho, lower), XtX)
    hat_trace = float(np.trace(A_inv_XtWX))

    return beta, np.full(1, hat_trace)


def _gcv_score(
    deviance: float,
    n: int,
    hat_trace: float,
) -> float:
    """Compute the GCV score.

    GCV = n * deviance / (n − tr(H))²

    where tr(H) is the trace of the hat (influence) matrix.
    """
    denom = n - hat_trace
    if denom <= 0:
        return np.inf
    return n * deviance / (denom * denom)


def _edf_per_smooth(
    X: NDArray,
    penalties: list[NDArray],
    sp: list[float],
    smooths_info: list[tuple[int, int]],
    W: NDArray | None = None,
) -> list[float]:
    """Compute effective degrees of freedom for each smooth term.

    EDF_j = trace(F_j) where F_j is the j-th diagonal block of (X'WX + Σ λ_j S_j)⁻¹ X'WX
    corresponding to smooth j's columns.
    """
    n, p = X.shape

    if W is not None:
        sqrtW = np.sqrt(W)
        Xw = X * sqrtW[:, np.newaxis]
    else:
        Xw = X

    XtWX = Xw.T @ Xw

    S_total = np.zeros((p, p))
    for lam, pen in zip(sp, penalties):
        S_total += lam * pen

    A = XtWX + S_total
    A = (A + A.T) * 0.5
    ridge = np.finfo(float).eps * max(np.trace(A) / p, 1.0)
    A[np.diag_indices_from(A)] += ridge

    cho, lower = cho_factor(A)
    F = cho_solve((cho, lower), XtWX)  # (p, p)

    edfs = []
    for col_start, col_end in smooths_info:
        edf_j = float(np.trace(F[col_start:col_end, col_start:col_end]))
        edfs.append(edf_j)

    return edfs


def _eval_gcv(
    X: NDArray,
    z: NDArray,
    penalties: list[NDArray],
    sp: list[float],
    W: NDArray | None = None,
    offset: NDArray | None = None,
) -> float:
    """Evaluate GCV score on a (possibly weighted) linear working model."""
    n = X.shape[0]
    try:
        beta, hat_arr = _penalized_solve(X, z, penalties, sp, W=W, offset=offset)
    except np.linalg.LinAlgError:
        return np.inf
    hat_trace = float(hat_arr[0])
    eta = X @ beta
    if offset is not None:
        eta = eta + offset
    resid = z - eta
    if W is not None:
        dev = float(np.sum(W * resid**2))
    else:
        dev = float(np.sum(resid**2))
    return _gcv_score(dev, n, hat_trace)


def _select_smoothing_params_gcv(
    X: NDArray,
    z: NDArray,
    penalties: list[NDArray],
    W: NDArray | None = None,
    offset: NDArray | None = None,
    n_sp: int = 1,
) -> list[float]:
    """Select smoothing parameters by minimizing GCV score.

    For a single smoothing parameter, uses Brent's method on log(λ).
    For multiple parameters, uses coordinate-wise Brent optimization:
    each λ_j is optimized in turn while the others are held fixed,
    cycling until the GCV score converges.

    When *W* is provided, optimizes the weighted working-model GCV used in
    performance iteration for non-Gaussian families.
    """
    if n_sp == 0:
        return []

    if len(penalties) == 1:

        def scalar_obj(log_sp: float) -> float:
            return _eval_gcv(X, z, penalties, [np.exp(log_sp)], W=W, offset=offset)

        result = minimize_scalar(scalar_obj, bounds=(-15, 15), method="bounded")
        return [float(np.exp(result.x))]

    # Multiple penalties: initialize with shared-λ, then coordinate descent.
    def shared_obj(log_sp: float) -> float:
        sp_val = np.exp(log_sp)
        return _eval_gcv(X, z, penalties, [sp_val] * len(penalties), W=W, offset=offset)

    init_result = minimize_scalar(shared_obj, bounds=(-15, 15), method="bounded")
    sp = [float(np.exp(init_result.x))] * len(penalties)

    max_cycles = 20
    tol = 1e-6

    for _ in range(max_cycles):
        gcv_before = _eval_gcv(X, z, penalties, sp, W=W, offset=offset)

        for j in range(len(penalties)):

            def coord_obj(log_sp_j: float) -> float:
                sp_trial = list(sp)
                sp_trial[j] = np.exp(log_sp_j)
                return _eval_gcv(X, z, penalties, sp_trial, W=W, offset=offset)

            result = minimize_scalar(
                coord_obj,
                bounds=(-15, 15),
                method="bounded",
            )
            sp[j] = float(np.exp(result.x))

        gcv_after = _eval_gcv(X, z, penalties, sp, W=W, offset=offset)
        if abs(gcv_after - gcv_before) / (abs(gcv_before) + 1e-12) < tol:
            break

    return sp


def _penalty_ranks(penalties: list[NDArray]) -> list[int]:
    """Compute the rank of each penalty matrix from its eigenvalues."""
    ranks = []
    for pen in penalties:
        eigvals = np.linalg.eigvalsh(pen)
        ranks.append(int(np.sum(eigvals > np.max(eigvals) * pen.shape[0] * np.finfo(float).eps)))
    return ranks


def _reml_objective(
    log_sp: NDArray,
    X: NDArray,
    z: NDArray,
    penalties: list[NDArray],
    penalty_ranks: list[int],
    n_unpenalized: int,
    scale_known: bool,
    scale: float = 1.0,
    W: NDArray | None = None,
    offset: NDArray | None = None,
    ml: bool = False,
) -> tuple[float, NDArray]:
    """Evaluate the negative REML (or ML) log-likelihood and its gradient.

    Parameters
    ----------
    log_sp:
        Log smoothing parameters ρ = log(λ), shape `(n_sp,)`.
    n_unpenalized:
        Number of totally unpenalized parameters (intercept + parametric terms + penalty null-space
        dimensions).
    scale_known:
        If `True`, use fixed *scale*; otherwise profile out φ (Gaussian).
    ml:
        If `True`, compute ML instead of REML by adding `0.5 * log|X'WX|`.

    Returns
    -------
    (value, gradient)
        Negative REML (or ML) and its gradient w.r.t. ρ.
    """
    n, p = X.shape
    sp = np.exp(log_sp)

    y = z.copy()
    if offset is not None:
        y = y - offset

    if W is not None:
        sqrtW = np.sqrt(W)
        Xw = X * sqrtW[:, np.newaxis]
        yw = y * sqrtW
    else:
        Xw = X
        yw = y

    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw

    S_total = np.zeros((p, p))
    for lam, pen in zip(sp, penalties):
        S_total += lam * pen

    A = XtX + S_total
    A = (A + A.T) * 0.5
    ridge = np.finfo(float).eps * max(np.trace(A) / p, 1.0)
    A[np.diag_indices_from(A)] += ridge

    try:
        cho, lower = cho_factor(A)
    except np.linalg.LinAlgError:
        return 1e20, np.zeros_like(log_sp)

    beta = cho_solve((cho, lower), Xty)

    # Penalized deviance: D_pen = ||√W(z - Xβ̂)||² + β̂'S_λ β̂ = y'y - β̂'X'y
    d_pen = float(np.dot(yw, yw) - np.dot(beta, Xty))

    # log|A| from Cholesky diagonal
    log_det_A = 2.0 * float(np.sum(np.log(np.diag(cho))))

    # log|S_λ|⁺ = Σ m_j · log(λ_j)  (block-diagonal penalties)
    log_det_S = sum(m * rho for m, rho in zip(penalty_ranks, log_sp))

    M = n_unpenalized
    if scale_known:
        val = d_pen / (2.0 * scale) + 0.5 * log_det_A - 0.5 * log_det_S
    else:
        d_pen = max(d_pen, 1e-30)
        val = 0.5 * (n - M) * np.log(d_pen) + 0.5 * log_det_A - 0.5 * log_det_S

    if ml:
        XtX_ml = (XtX + XtX.T) * 0.5
        ridge_ml = np.finfo(float).eps * max(np.trace(XtX_ml) / p, 1.0)
        XtX_ml[np.diag_indices_from(XtX_ml)] += ridge_ml
        try:
            cho_xtx = cho_factor(XtX_ml)[0]
            log_det_XtX = 2.0 * float(np.sum(np.log(np.diag(cho_xtx))))
        except np.linalg.LinAlgError:
            log_det_XtX = 0.0
        val -= 0.5 * log_det_XtX

    # Gradient w.r.t. ρ_j
    grad = np.zeros_like(log_sp)
    for j, (lam_j, pen_j, m_j) in enumerate(zip(sp, penalties, penalty_ranks)):
        beta_Sj_beta = float(beta @ pen_j @ beta)
        tr_Ainv_Sj = float(np.trace(cho_solve((cho, lower), pen_j)))

        if scale_known:
            grad[j] = lam_j * beta_Sj_beta / (2.0 * scale) + lam_j * 0.5 * tr_Ainv_Sj - 0.5 * m_j
        else:
            grad[j] = (
                (n - M) * lam_j * beta_Sj_beta / (2.0 * d_pen)
                + lam_j * 0.5 * tr_Ainv_Sj
                - 0.5 * m_j
            )

    return float(val), grad


def _select_smoothing_params_reml(
    X: NDArray,
    z: NDArray,
    penalties: list[NDArray],
    penalty_ranks: list[int],
    n_unpenalized: int,
    scale_known: bool,
    scale: float = 1.0,
    W: NDArray | None = None,
    offset: NDArray | None = None,
    n_sp: int = 1,
) -> list[float]:
    """Select smoothing parameters by maximizing REML.

    Uses L-BFGS-B on ρ = log(λ) with analytic gradients.
    """
    if n_sp == 0:
        return []

    rho_init = np.zeros(n_sp)

    def objective(rho: NDArray) -> tuple[float, NDArray]:
        return _reml_objective(
            rho,
            X,
            z,
            penalties,
            penalty_ranks,
            n_unpenalized,
            scale_known,
            scale,
            W=W,
            offset=offset,
        )

    result = minimize(
        objective,
        rho_init,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-20, 20)] * n_sp,
    )

    return [float(np.exp(r)) for r in result.x]


def _select_smoothing_params_ml(
    X: NDArray,
    z: NDArray,
    penalties: list[NDArray],
    penalty_ranks: list[int],
    n_unpenalized: int,
    scale_known: bool,
    scale: float = 1.0,
    W: NDArray | None = None,
    offset: NDArray | None = None,
    n_sp: int = 1,
) -> list[float]:
    """Select smoothing parameters by maximizing ML (marginal likelihood).

    Uses L-BFGS-B on ρ = log(λ) with analytic gradients.
    """
    if n_sp == 0:
        return []

    rho_init = np.zeros(n_sp)

    def objective(rho: NDArray) -> tuple[float, NDArray]:
        return _reml_objective(
            rho,
            X,
            z,
            penalties,
            penalty_ranks,
            n_unpenalized,
            scale_known,
            scale,
            W=W,
            offset=offset,
            ml=True,
        )

    result = minimize(
        objective,
        rho_init,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-20, 20)] * n_sp,
    )

    return [float(np.exp(r)) for r in result.x]


def _estimate_nb_theta(
    y: NDArray,
    mu: NDArray,
    theta_old: float,
) -> float:
    """Estimate NB θ by maximizing the profile log-likelihood.

    Given fitted μ from the current P-IRLS, find θ that maximizes ℓ(θ | y, μ) using Brent's method
    on log(θ).
    """
    from scipy.special import gammaln

    mu_c = np.maximum(mu, np.finfo(float).eps)

    def neg_ll(log_theta: float) -> float:
        theta = np.exp(log_theta)
        ll = np.sum(
            gammaln(y + theta)
            - gammaln(theta)
            + theta * np.log(theta / (mu_c + theta))
            + y * np.log(mu_c / (mu_c + theta))
        )
        return -float(ll)

    result = minimize_scalar(
        neg_ll,
        bounds=(np.log(0.01), np.log(1e6)),
        method="bounded",
    )
    return float(np.exp(result.x))


def pirls_fit(
    model: ModelMatrix,
    family: Family | None = None,
    *,
    smoothing_params: list[float] | None = None,
    method: str = "GCV",
    max_iter: int = 50,
    tol: float = 1e-7,
    prior_weights: NDArray | None = None,
) -> FitResult:
    """Fit a GAM using Penalized IRLS.

    For Gaussian with identity link, this reduces to a single penalized least squares solve with
    automatically selected smoothing parameters.

    Parameters
    ----------
    model:
        A `ModelMatrix` from `build_model_matrix()`.
    family:
        Response distribution family. Defaults to `Gaussian()`.
    smoothing_params:
        Fixed smoothing parameters λ_j, one per smooth term. If `None`, smoothing parameters are
        selected automatically via *method*.
    method:
        Smoothing parameter selection method: `"GCV"` for Generalized Cross-Validation, or `"REML"`
        for Restricted Maximum Likelihood. Ignored when *smoothing_params* is provided.
    max_iter:
        Maximum number of P-IRLS iterations.
    tol:
        Convergence tolerance on relative change in deviance.
    prior_weights:
        Observation weights, shape `(n,)`. Must be positive. When provided, the weighted deviance
        `sum(w_i * d_i)` is minimized and the weighted IRLS system is solved.

    Returns
    -------
    FitResult
        Fitted model results including coefficients, fitted values, smoothing parameters, EDF, and
        diagnostics.
    """
    if family is None:
        family = Gaussian()

    method_upper = method.upper()
    if method_upper not in ("GCV", "REML", "ML"):
        raise ValueError(f"method must be 'GCV', 'REML', or 'ML', got {method!r}.")

    use_reml = method_upper == "REML"
    use_ml = method_upper == "ML"

    X = model.X
    y = model.response
    n, p = X.shape
    offset = model.offset

    auto_select = smoothing_params is None
    if not auto_select:
        if len(smoothing_params) != len(model.penalties):
            raise ValueError(
                f"Expected {len(model.penalties)} smoothing parameters, "
                f"got {len(smoothing_params)}."
            )
        sp = list(smoothing_params)
    else:
        sp = []

    # Precompute REML/ML-specific quantities
    pen_ranks: list[int] = []
    n_unpenalized = 0
    if (use_reml or use_ml) and auto_select and model.penalties:
        pen_ranks = _penalty_ranks(model.penalties)
        n_unpenalized = (1 if model.has_intercept else 0) + model.n_parametric
        for s_info in model.smooths:
            n_unpenalized += s_info.null_space_dim

    def _select_sp(
        z: NDArray,
        W: NDArray | None = None,
    ) -> list[float]:
        """Dispatch to GCV, REML, or ML selection."""
        if use_reml:
            return _select_smoothing_params_reml(
                X,
                z,
                model.penalties,
                pen_ranks,
                n_unpenalized,
                scale_known=family.scale_known,
                W=W,
                offset=offset,
                n_sp=len(model.penalties),
            )
        if use_ml:
            return _select_smoothing_params_ml(
                X,
                z,
                model.penalties,
                pen_ranks,
                n_unpenalized,
                scale_known=family.scale_known,
                W=W,
                offset=offset,
                n_sp=len(model.penalties),
            )
        return _select_smoothing_params_gcv(
            X,
            z,
            model.penalties,
            W=W,
            offset=offset,
            n_sp=len(model.penalties),
        )

    pw = prior_weights

    is_gaussian_identity = isinstance(family, Gaussian)
    W_final: NDArray | None = None

    if is_gaussian_identity:
        if not sp:
            sp = _select_sp(y, W=pw)
        beta, hat_arr = _penalized_solve(X, y, model.penalties, sp, W=pw, offset=offset)
        hat_trace = float(hat_arr[0])

        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        mu = family.link_inverse(eta)
        dev = family.deviance(y, mu, weights=pw)
        n_iter = 1
        converged = True
    else:
        is_nb = isinstance(family, NegativeBinomial)
        max_outer = 20 if is_nb else 1
        theta_tol = 1e-4

        mu = family.initialize(y)
        eta = family.link(mu)
        converged = False
        n_iter = 0
        beta = np.zeros(p)

        for _outer in range(max_outer):
            dev_old = np.inf
            inner_converged = False

            for iteration in range(max_iter):
                n_iter += 1

                dmu_deta = 1.0 / family.link_derivative(mu)
                W_irls = dmu_deta**2 / family.variance(mu)
                W_total = pw * W_irls if pw is not None else W_irls
                z = eta + (y - mu) / dmu_deta

                if auto_select:
                    sp = _select_sp(z, W=W_total)

                beta, hat_arr = _penalized_solve(
                    X,
                    z,
                    model.penalties,
                    sp,
                    W=W_total,
                    offset=offset,
                )

                eta = X @ beta
                if offset is not None:
                    eta = eta + offset
                mu = family.link_inverse(eta)
                dev = family.deviance(y, mu, weights=pw)

                if dev_old != np.inf and abs(dev - dev_old) / (abs(dev_old) + 0.1) < tol:
                    inner_converged = True
                    break

                dev_old = dev

            if not is_nb:
                converged = inner_converged
                break

            theta_old = family.theta
            theta_new = _estimate_nb_theta(y, mu, theta_old)
            family.theta = theta_new

            if abs(theta_new - theta_old) / (abs(theta_old) + 1e-8) < theta_tol:
                converged = inner_converged
                break
        else:
            converged = inner_converged

        if is_nb:
            dev = family.deviance(y, mu, weights=pw)

        hat_trace = float(hat_arr[0])
        W_final = W_total

    smooths_info = [(s.col_start, s.col_end) for s in model.smooths]
    W_for_edf = pw if is_gaussian_identity else W_final
    edf = _edf_per_smooth(X, model.penalties, sp, smooths_info, W=W_for_edf)
    edf_total = sum(edf) + (1 if model.has_intercept else 0) + model.n_parametric

    n_eff = float(np.sum(pw)) if pw is not None else float(n)

    if family.scale_known:
        scale = 1.0
    else:
        residual_dof = n_eff - edf_total
        scale = dev / residual_dof if residual_dof > 0 else dev / n_eff

    gcv = _gcv_score(dev, n, hat_trace)

    if pw is not None:
        y_mean = float(np.average(y, weights=pw))
    else:
        y_mean = float(np.mean(y))
    mu_null = family.link_inverse(np.full_like(y, family.link(np.atleast_1d(y_mean))[0]))
    null_dev = family.deviance(y, mu_null, weights=pw)

    ll = family.log_likelihood(y, mu, scale, weights=pw)
    aic = -2.0 * ll + 2.0 * edf_total
    bic = -2.0 * ll + np.log(n) * edf_total

    return FitResult(
        coefficients=beta,
        linear_predictor=eta,
        fitted_values=mu,
        smoothing_params=sp,
        scale=scale,
        gcv_score=gcv,
        edf=edf,
        edf_total=edf_total,
        deviance=dev,
        n_iter=n_iter,
        converged=converged,
        hat_matrix_trace=hat_trace,
        residuals=y - mu,
        weights=W_final,
        prior_weights=pw,
        null_deviance=float(null_dev),
        aic=float(aic),
        bic=float(bic),
    )
