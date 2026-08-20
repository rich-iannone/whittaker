"""Large-scale GAM fitting via discretized basis evaluation.

Implements the bam approach (Wood, Li, & Shaddick, 2017) for fitting GAMs to datasets too large for
the standard dense X'WX computation. Covariates are discretized to a grid of representative values,
and basis functions are evaluated only at the unique grid points. The full n x p design matrix is
never materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar

from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.families.negative_binomial import NegativeBinomial
from whittaker.fitting.pirls import FitResult, _estimate_nb_theta, _gcv_score, _penalty_ranks
from whittaker.model_matrix import SmoothInfo


@dataclass
class DiscretizedBlock:
    """Compressed basis for one column block in the model matrix."""

    unique_basis: NDArray
    indices: NDArray
    col_start: int
    col_end: int
    by_weights: NDArray | None = None


@dataclass
class DiscretizedModelMatrix:
    """Compressed design matrix for large-scale fitting."""

    blocks: list[DiscretizedBlock]
    parametric_cols: NDArray | None
    n_param_cols: int
    penalties: list[NDArray]
    smooth_infos: list[SmoothInfo]
    n_obs: int
    n_cols: int
    response: NDArray
    offset: NDArray | None
    has_intercept: bool
    n_parametric: int
    column_names: list[str] = field(default_factory=list)
    offset_expressions: list[str] = field(default_factory=list)


def _discretize_1d(x: NDArray, n_discrete: int) -> tuple[NDArray, NDArray]:
    """Round a 1-D covariate to *n_discrete* equally-spaced grid points."""
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.array([xmin]), np.zeros(len(x), dtype=np.intp)
    grid = np.linspace(xmin, xmax, n_discrete)
    raw = np.round((x - xmin) / (xmax - xmin) * (n_discrete - 1)).astype(np.intp)
    raw = np.clip(raw, 0, n_discrete - 1)
    used, inverse = np.unique(raw, return_inverse=True)
    return grid[used], inverse


def _discretize_nd(x: NDArray, n_discrete: int) -> tuple[NDArray, NDArray]:
    """Discretize multi-dimensional covariates to a joint grid."""
    if x.ndim == 1:
        return _discretize_1d(x, n_discrete)
    d = x.shape[1]
    grids: list[NDArray] = []
    col_indices: list[NDArray] = []
    for j in range(d):
        uv, idx = _discretize_1d(x[:, j], n_discrete)
        grids.append(uv)
        col_indices.append(idx)
    combined = np.column_stack(col_indices)
    unique_combos, inverse = np.unique(combined, axis=0, return_inverse=True)
    unique_x = np.column_stack([grids[j][unique_combos[:, j]] for j in range(d)])
    return unique_x, inverse


# ---------------------------------------------------------------------------
# Discretized linear algebra
# ---------------------------------------------------------------------------


def _compute_XtWX(dm: DiscretizedModelMatrix, w: NDArray) -> NDArray:
    """Compute X'WX without materializing the full X."""
    p = dm.n_cols
    XtWX = np.zeros((p, p))
    nc = dm.n_param_cols

    if dm.parametric_cols is not None:
        Pw = dm.parametric_cols * w[:, None]
        XtWX[:nc, :nc] = Pw.T @ dm.parametric_cols

    for blk in dm.blocks:
        cs, ce = blk.col_start, blk.col_end
        w_eff = w * blk.by_weights**2 if blk.by_weights is not None else w
        W_agg = np.bincount(blk.indices, weights=w_eff, minlength=len(blk.unique_basis))
        XtWX[cs:ce, cs:ce] = (blk.unique_basis * W_agg[:, None]).T @ blk.unique_basis

    if dm.parametric_cols is not None:
        for blk in dm.blocks:
            cs, ce = blk.col_start, blk.col_end
            for pc in range(nc):
                x_p = dm.parametric_cols[:, pc]
                weff = w * x_p * blk.by_weights if blk.by_weights is not None else w * x_p
                agg = np.bincount(blk.indices, weights=weff, minlength=len(blk.unique_basis))
                cross = blk.unique_basis.T @ agg
                XtWX[cs:ce, pc] = cross
                XtWX[pc, cs:ce] = cross

    blocks = dm.blocks
    for i in range(len(blocks)):
        bi = blocks[i]
        for j in range(i + 1, len(blocks)):
            bj = blocks[j]
            d_i, d_j = len(bi.unique_basis), len(bj.unique_basis)
            w_cross = w.copy()
            if bi.by_weights is not None:
                w_cross = w_cross * bi.by_weights
            if bj.by_weights is not None:
                w_cross = w_cross * bj.by_weights
            joint = bi.indices * d_j + bj.indices
            M = np.bincount(joint, weights=w_cross, minlength=d_i * d_j).reshape(d_i, d_j)
            cross = bi.unique_basis.T @ M @ bj.unique_basis
            XtWX[bi.col_start : bi.col_end, bj.col_start : bj.col_end] = cross
            XtWX[bj.col_start : bj.col_end, bi.col_start : bi.col_end] = cross.T

    return XtWX


def _compute_Xtz(dm: DiscretizedModelMatrix, w: NDArray, z: NDArray) -> NDArray:
    """Compute X'Wz without materializing the full X."""
    p = dm.n_cols
    Xtz = np.zeros(p)
    wz = w * z
    nc = dm.n_param_cols

    if dm.parametric_cols is not None:
        Xtz[:nc] = dm.parametric_cols.T @ wz

    for blk in dm.blocks:
        cs, ce = blk.col_start, blk.col_end
        wz_eff = wz * blk.by_weights if blk.by_weights is not None else wz
        agg = np.bincount(blk.indices, weights=wz_eff, minlength=len(blk.unique_basis))
        Xtz[cs:ce] = blk.unique_basis.T @ agg

    return Xtz


def _compute_eta(dm: DiscretizedModelMatrix, beta: NDArray) -> NDArray:
    """Compute eta = X @ beta + offset without materializing X."""
    eta = np.zeros(dm.n_obs)
    nc = dm.n_param_cols

    if dm.parametric_cols is not None:
        eta += dm.parametric_cols @ beta[:nc]

    for blk in dm.blocks:
        cs, ce = blk.col_start, blk.col_end
        vals = (blk.unique_basis @ beta[cs:ce])[blk.indices]
        if blk.by_weights is not None:
            vals = vals * blk.by_weights
        eta += vals

    if dm.offset is not None:
        eta = eta + dm.offset

    return eta


# ---------------------------------------------------------------------------
# Penalised solve and EDF (discretized variants)
# ---------------------------------------------------------------------------


def _penalized_solve_disc(
    dm: DiscretizedModelMatrix,
    z: NDArray,
    sp: list[float],
    W: NDArray | None = None,
) -> tuple[NDArray, float]:
    """Solve (X'WX + S_lambda) beta = X'Wz using discretized accumulation."""
    p = dm.n_cols
    w = W if W is not None else np.ones(dm.n_obs)

    z_eff = z - dm.offset if dm.offset is not None else z

    XtWX = _compute_XtWX(dm, w)
    Xtz = _compute_Xtz(dm, w, z_eff)

    S_total = np.zeros((p, p))
    for lam, pen in zip(sp, dm.penalties, strict=False):
        S_total += lam * pen

    A = XtWX + S_total
    A = (A + A.T) * 0.5
    ridge = np.finfo(float).eps * max(np.trace(A) / p, 1.0)
    A[np.diag_indices_from(A)] += ridge

    cho, lower = cho_factor(A)
    beta = cho_solve((cho, lower), Xtz)

    A_inv_XtWX = cho_solve((cho, lower), XtWX)
    hat_trace = float(np.trace(A_inv_XtWX))

    return beta, hat_trace


def _edf_per_smooth_disc(
    dm: DiscretizedModelMatrix,
    sp: list[float],
    W: NDArray | None = None,
) -> list[float]:
    """Compute per-smooth effective degrees of freedom."""
    p = dm.n_cols
    w = W if W is not None else np.ones(dm.n_obs)

    XtWX = _compute_XtWX(dm, w)

    S_total = np.zeros((p, p))
    for lam, pen in zip(sp, dm.penalties, strict=False):
        S_total += lam * pen

    A = XtWX + S_total
    A = (A + A.T) * 0.5
    ridge = np.finfo(float).eps * max(np.trace(A) / p, 1.0)
    A[np.diag_indices_from(A)] += ridge

    cho, lower = cho_factor(A)
    F = cho_solve((cho, lower), XtWX)

    edfs = []
    for s_info in dm.smooth_infos:
        cs, ce = s_info.col_start, s_info.col_end
        edfs.append(float(np.trace(F[cs:ce, cs:ce])))
    return edfs


# ---------------------------------------------------------------------------
# Smoothing parameter selection (discretized)
# ---------------------------------------------------------------------------


def _eval_gcv_disc(dm: DiscretizedModelMatrix, z: NDArray, sp: list[float], W: NDArray | None):
    n = dm.n_obs
    try:
        beta, hat_trace = _penalized_solve_disc(dm, z, sp, W=W)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.inf
    eta = _compute_eta(dm, beta)
    resid = z - eta
    dev = float(np.sum(W * resid**2)) if W is not None else float(np.sum(resid**2))
    return _gcv_score(dev, n, hat_trace)


def _select_sp_gcv_disc(
    dm: DiscretizedModelMatrix, z: NDArray, W: NDArray | None = None
) -> list[float]:
    n_sp = len(dm.penalties)
    if n_sp == 0:
        return []

    if n_sp == 1:

        def scalar_obj(log_sp: float) -> float:
            return _eval_gcv_disc(dm, z, [np.exp(log_sp)], W)

        result = minimize_scalar(scalar_obj, bounds=(-15, 15), method="bounded")
        return [float(np.exp(result.x))]

    def shared_obj(log_sp: float) -> float:
        return _eval_gcv_disc(dm, z, [np.exp(log_sp)] * n_sp, W)

    init = minimize_scalar(shared_obj, bounds=(-15, 15), method="bounded")
    sp = [float(np.exp(init.x))] * n_sp

    for _ in range(20):
        gcv_before = _eval_gcv_disc(dm, z, sp, W)
        for j in range(n_sp):

            def coord_obj(log_sp_j: float, _j=j) -> float:
                sp_t = list(sp)
                sp_t[_j] = np.exp(log_sp_j)
                return _eval_gcv_disc(dm, z, sp_t, W)

            res = minimize_scalar(coord_obj, bounds=(-15, 15), method="bounded")
            sp[j] = float(np.exp(res.x))
        gcv_after = _eval_gcv_disc(dm, z, sp, W)
        if abs(gcv_after - gcv_before) / (abs(gcv_before) + 1e-12) < 1e-6:
            break
    return sp


def _reml_objective_disc(
    log_sp: NDArray,
    dm: DiscretizedModelMatrix,
    z: NDArray,
    penalty_ranks: list[int],
    n_unpenalized: int,
    scale_known: bool,
    scale: float = 1.0,
    W: NDArray | None = None,
    ml: bool = False,
) -> tuple[float, NDArray]:
    """Negative REML (or ML) and gradient using discretized X'WX."""
    n = dm.n_obs
    p = dm.n_cols
    sp = np.exp(log_sp)
    w = W if W is not None else np.ones(n)

    y = z - dm.offset if dm.offset is not None else z.copy()

    XtWX = _compute_XtWX(dm, w)
    Xty = _compute_Xtz(dm, w, y)

    S_total = np.zeros((p, p))
    for lam, pen in zip(sp, dm.penalties, strict=False):
        S_total += lam * pen

    A = XtWX + S_total
    A = (A + A.T) * 0.5
    ridge = np.finfo(float).eps * max(np.trace(A) / p, 1.0)
    A[np.diag_indices_from(A)] += ridge

    try:
        cho, lower = cho_factor(A)
    except np.linalg.LinAlgError:  # pragma: no cover
        return 1e20, np.zeros_like(log_sp)

    beta = cho_solve((cho, lower), Xty)
    yWy = float(np.dot(y * w, y))
    d_pen = yWy - float(np.dot(beta, Xty))

    log_det_A = 2.0 * float(np.sum(np.log(np.diag(cho))))
    log_det_S = sum(m * rho for m, rho in zip(penalty_ranks, log_sp, strict=False))

    M = n_unpenalized
    if scale_known:
        val = d_pen / (2.0 * scale) + 0.5 * log_det_A - 0.5 * log_det_S
    else:
        d_pen = max(d_pen, 1e-30)
        val = 0.5 * (n - M) * np.log(d_pen) + 0.5 * log_det_A - 0.5 * log_det_S

    if ml:
        XtX_ml = (XtWX + XtWX.T) * 0.5
        ridge_ml = np.finfo(float).eps * max(np.trace(XtX_ml) / p, 1.0)
        XtX_ml[np.diag_indices_from(XtX_ml)] += ridge_ml
        try:
            cho_xtx = cho_factor(XtX_ml)[0]
            log_det_XtX = 2.0 * float(np.sum(np.log(np.diag(cho_xtx))))
        except np.linalg.LinAlgError:  # pragma: no cover
            log_det_XtX = 0.0
        val -= 0.5 * log_det_XtX

    grad = np.zeros_like(log_sp)
    for j, (lam_j, pen_j, m_j) in enumerate(zip(sp, dm.penalties, penalty_ranks, strict=False)):
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


def _select_sp_reml_disc(
    dm: DiscretizedModelMatrix,
    z: NDArray,
    penalty_ranks: list[int],
    n_unpenalized: int,
    scale_known: bool,
    W: NDArray | None = None,
    ml: bool = False,
) -> list[float]:
    n_sp = len(dm.penalties)
    if n_sp == 0:
        return []
    rho_init = np.zeros(n_sp)

    def objective(rho: NDArray) -> tuple[float, NDArray]:
        return _reml_objective_disc(
            rho, dm, z, penalty_ranks, n_unpenalized, scale_known, W=W, ml=ml
        )

    result = minimize(objective, rho_init, method="L-BFGS-B", jac=True, bounds=[(-20, 20)] * n_sp)
    return [float(np.exp(r)) for r in result.x]


# ---------------------------------------------------------------------------
# Main fitting entry point
# ---------------------------------------------------------------------------


def bam_fit(
    dm: DiscretizedModelMatrix,
    family: Family | None = None,
    *,
    smoothing_params: list[float] | None = None,
    method: str = "GCV",
    max_iter: int = 50,
    tol: float = 1e-7,
    prior_weights: NDArray | None = None,
) -> FitResult:
    """Fit a GAM using discretized P-IRLS for large datasets.

    Mirrors :func:`~whittaker.fitting.pirls.pirls_fit` but accumulates X'WX and X'Wz from
    discretized basis blocks instead of the full design matrix.
    """
    if family is None:  # pragma: no cover
        family = Gaussian()

    method_upper = method.upper()
    if method_upper == "FREML":
        method_upper = "REML"
    if method_upper not in ("GCV", "REML", "ML"):
        raise ValueError(f"method must be 'GCV', 'REML', 'ML', or 'fREML', got {method!r}.")

    use_reml = method_upper == "REML"
    use_ml = method_upper == "ML"

    y = dm.response
    n = dm.n_obs
    p = dm.n_cols

    auto_select = smoothing_params is None
    if not auto_select:
        if len(smoothing_params) != len(dm.penalties):
            raise ValueError(
                f"Expected {len(dm.penalties)} smoothing parameters, got {len(smoothing_params)}."
            )
        sp = list(smoothing_params)
    else:
        sp: list[float] = []

    pen_ranks: list[int] = []
    n_unpenalized = 0
    if (use_reml or use_ml) and auto_select and dm.penalties:
        pen_ranks = _penalty_ranks(dm.penalties)
        n_unpenalized = (1 if dm.has_intercept else 0) + dm.n_parametric
        for s_info in dm.smooth_infos:
            n_unpenalized += s_info.null_space_dim

    def _select_sp(z: NDArray, W: NDArray | None = None) -> list[float]:
        if use_reml:
            return _select_sp_reml_disc(dm, z, pen_ranks, n_unpenalized, family.scale_known, W=W)
        if use_ml:
            return _select_sp_reml_disc(
                dm, z, pen_ranks, n_unpenalized, family.scale_known, W=W, ml=True
            )
        return _select_sp_gcv_disc(dm, z, W=W)

    pw = prior_weights
    W_final: NDArray | None = None
    z_final: NDArray = y
    is_gaussian_identity = isinstance(family, Gaussian)

    if is_gaussian_identity:
        if not sp:
            sp = _select_sp(y, W=pw)
        beta, hat_trace = _penalized_solve_disc(dm, y, sp, W=pw)
        eta = _compute_eta(dm, beta)
        mu = family.link_inverse(eta)
        dev = family.deviance(y, mu, weights=pw)
        n_iter = 1
        converged = True
    else:
        is_nb = isinstance(family, NegativeBinomial)
        max_outer = 20 if is_nb else 1

        mu = family.initialize(y)
        eta = family.link(mu)
        converged = False
        n_iter = 0
        beta = np.zeros(p)
        hat_trace = 0.0

        dev = np.inf
        z = eta.copy()
        W_total = pw if pw is not None else np.ones(n, dtype=float)
        inner_converged = False

        for _outer in range(max_outer):
            dev_old = np.inf
            inner_converged = False

            for _ in range(max_iter):
                n_iter += 1

                custom = family.irls_update(y, mu, eta)
                if custom is not None:
                    z, W_irls = custom
                else:
                    dmu_deta = 1.0 / family.link_derivative(mu)
                    W_irls = dmu_deta**2 / family.variance(mu)
                    z = eta + (y - mu) / dmu_deta
                W_total = pw * W_irls if pw is not None else W_irls

                if auto_select:
                    sp = _select_sp(z, W=W_total)

                beta, hat_trace = _penalized_solve_disc(dm, z, sp, W=W_total)
                eta = _compute_eta(dm, beta)
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
            if abs(theta_new - theta_old) / (abs(theta_old) + 1e-8) < 1e-4:
                converged = inner_converged
                break
        else:  # pragma: no cover
            converged = inner_converged

        if is_nb:
            dev = family.deviance(y, mu, weights=pw)

        W_final = W_total
        z_final = z

    W_for_edf = pw if is_gaussian_identity else W_final
    edf = _edf_per_smooth_disc(dm, sp, W=W_for_edf)
    edf_total = sum(edf) + (1 if dm.has_intercept else 0) + dm.n_parametric

    n_eff = float(np.sum(pw)) if pw is not None else float(n)
    if family.scale_known:
        scale = 1.0
    else:
        residual_dof = n_eff - edf_total
        scale = dev / residual_dof if residual_dof > 0 else dev / n_eff

    gcv = _gcv_score(dev, n, hat_trace)

    y_mean = float(np.average(y, weights=pw)) if pw is not None else float(np.mean(y))
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
        method=method_upper,
        pseudo_data=z_final,
    )
