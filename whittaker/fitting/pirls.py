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
from scipy.optimize import minimize_scalar

from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
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
        Smoothing parameter λ_j for each smooth term.
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


def _select_smoothing_params_gcv(
    X: NDArray,
    y: NDArray,
    penalties: list[NDArray],
    family: Family,
    offset: NDArray | None = None,
    n_sp: int = 1,
) -> list[float]:
    """Select smoothing parameters by minimizing GCV score.

    For a single smoothing parameter (shared across all terms), uses Brent's method on a log scale.
    For multiple parameters, uses a coordinate-wise search.
    """
    n = X.shape[0]

    if n_sp == 0:
        return []

    def gcv_objective(log_sp: float) -> float:
        sp_val = np.exp(log_sp)
        sp_list = [sp_val] * len(penalties)

        beta, hat_arr = _penalized_solve(X, y, penalties, sp_list, offset=offset)
        hat_trace = float(hat_arr[0])

        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        mu = family.link_inverse(eta)
        dev = family.deviance(y, mu)

        return _gcv_score(dev, n, hat_trace)

    result = minimize_scalar(gcv_objective, bounds=(-15, 15), method="bounded")
    optimal_sp = float(np.exp(result.x))

    return [optimal_sp] * len(penalties)


def pirls_fit(
    model: ModelMatrix,
    family: Family | None = None,
    *,
    smoothing_params: list[float] | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> FitResult:
    """Fit a GAM using Penalized IRLS.

    For Gaussian with identity link, this reduces to a single penalized least squares solve with
    GCV-selected smoothing parameters.

    Parameters
    ----------
    model:
        A `ModelMatrix` from `build_model_matrix()`.
    family:
        Response distribution family. Defaults to `Gaussian()`.
    smoothing_params:
        Fixed smoothing parameters λ_j, one per smooth term. If `None`, smoothing parameters are
        selected automatically via GCV.
    max_iter:
        Maximum number of P-IRLS iterations.
    tol:
        Convergence tolerance on relative change in deviance.

    Returns
    -------
    FitResult
        Fitted model results including coefficients, fitted values, smoothing parameters, EDF, and
        diagnostics.
    """
    if family is None:
        family = Gaussian()

    X = model.X
    y = model.response
    n, p = X.shape
    offset = model.offset

    if smoothing_params is not None:
        if len(smoothing_params) != len(model.penalties):
            raise ValueError(
                f"Expected {len(model.penalties)} smoothing parameters, "
                f"got {len(smoothing_params)}."
            )
        sp = list(smoothing_params)
    else:
        sp = _select_smoothing_params_gcv(
            X,
            y,
            model.penalties,
            family,
            offset=offset,
            n_sp=len(model.penalties),
        )

    is_gaussian_identity = isinstance(family, Gaussian)

    if is_gaussian_identity:
        beta, hat_arr = _penalized_solve(X, y, model.penalties, sp, offset=offset)
        hat_trace = float(hat_arr[0])

        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        mu = family.link_inverse(eta)
        dev = family.deviance(y, mu)
        n_iter = 1
        converged = True
    else:
        mu = family.initialize(y)
        eta = family.link(mu)
        dev_old = np.inf
        converged = False
        n_iter = 0
        beta = np.zeros(p)

        for iteration in range(max_iter):
            n_iter = iteration + 1

            dmu_deta = 1.0 / family.link_derivative(mu)
            W = dmu_deta**2 / family.variance(mu)

            z = eta + (y - mu) / dmu_deta

            beta, hat_arr = _penalized_solve(X, z, model.penalties, sp, W=W, offset=offset)

            eta = X @ beta
            if offset is not None:
                eta = eta + offset
            mu = family.link_inverse(eta)
            dev = family.deviance(y, mu)

            if dev_old != np.inf and abs(dev - dev_old) / (abs(dev_old) + 0.1) < tol:
                converged = True
                break

            dev_old = dev

        hat_trace = float(hat_arr[0])

    smooths_info = [(s.col_start, s.col_end) for s in model.smooths]
    edf = _edf_per_smooth(X, model.penalties, sp, smooths_info)
    edf_total = sum(edf) + (1 if model.has_intercept else 0) + model.n_parametric

    residual_dof = n - edf_total
    scale = dev / residual_dof if residual_dof > 0 else dev / n

    gcv = _gcv_score(dev, n, hat_trace)

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
    )
