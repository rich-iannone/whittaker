"""GAMLSS fitting via the RS algorithm (Rigby & Stasinopoulos 2005)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from whittaker.families.gamlss_base import GAMLSSFamily
from whittaker.fitting.pirls import (
    _edf_per_smooth,
    _penalized_solve,
    _penalty_ranks,
    _select_smoothing_params_gcv,
    _select_smoothing_params_ml,
    _select_smoothing_params_reml,
)
from whittaker.model_matrix import ModelMatrix

_W_FLOOR = 1e-10
_Z_CLIP = 1e6


@dataclass
class GAMLSSParamResult:
    """Fit result for a single distributional parameter."""

    coefficients: NDArray
    linear_predictor: NDArray
    fitted_values: NDArray
    smoothing_params: list[float]
    edf: list[float]
    edf_total: float


@dataclass
class GAMLSSFitResult:
    """Result of a GAMLSS fit."""

    params: dict[str, GAMLSSParamResult]
    log_likelihood: float
    global_deviance: float
    n_iter: int
    converged: bool
    aic: float
    bic: float
    n_obs: int
    response: NDArray = field(repr=False)


def _sanitize(z: NDArray, W: NDArray) -> tuple[NDArray, NDArray]:
    z = np.where(np.isfinite(z), z, 0.0)
    z = np.clip(z, -_Z_CLIP, _Z_CLIP)
    W = np.where(np.isfinite(W), W, _W_FLOOR)
    W = np.maximum(W, _W_FLOOR)
    return z, W


def _select_sp_for_param(
    X: NDArray,
    z: NDArray,
    penalties: list[NDArray],
    method: str,
    pen_ranks: list[int],
    n_unpenalized: int,
    W: NDArray | None = None,
    offset: NDArray | None = None,
) -> list[float]:
    n_sp = len(penalties)
    if n_sp == 0:
        return []
    if method == "REML":
        return _select_smoothing_params_reml(
            X,
            z,
            penalties,
            pen_ranks,
            n_unpenalized,
            scale_known=True,
            W=W,
            offset=offset,
            n_sp=n_sp,
        )
    if method == "ML":
        return _select_smoothing_params_ml(
            X,
            z,
            penalties,
            pen_ranks,
            n_unpenalized,
            scale_known=True,
            W=W,
            offset=offset,
            n_sp=n_sp,
        )
    return _select_smoothing_params_gcv(
        X,
        z,
        penalties,
        W=W,
        offset=offset,
        n_sp=n_sp,
    )


def _compute_zw(
    family: GAMLSSFamily,
    name: str,
    y: NDArray,
    params: dict[str, NDArray],
    eta: NDArray,
) -> tuple[NDArray, NDArray]:
    dl = family.dl_dtheta(name, y, params)
    d2l = np.maximum(family.d2l_dtheta2(name, y, params), _W_FLOOR)
    g_prime = family.link_derivative(name, params[name])
    dtheta_deta = 1.0 / g_prime
    W_irls = d2l * dtheta_deta**2
    z = eta + dl * g_prime / d2l
    return _sanitize(z, W_irls)


def gamlss_fit(
    models: dict[str, ModelMatrix],
    family: GAMLSSFamily,
    y: NDArray,
    *,
    method: str = "GCV",
    max_outer: int = 50,
    max_inner: int = 20,
    tol: float = 1e-6,
) -> GAMLSSFitResult:
    """Fit a GAMLSS model using the RS algorithm."""
    method_upper = method.upper()
    if method_upper not in ("GCV", "REML", "ML"):
        raise ValueError(f"method must be 'GCV', 'REML', or 'ML', got {method!r}.")

    n = len(y)
    param_names = family.parameter_names

    params = family.initialize(y)
    etas: dict[str, NDArray] = {}
    betas: dict[str, NDArray] = {}
    sp_dict: dict[str, list[float]] = {}

    for name in param_names:
        model = models[name]
        etas[name] = family.link(name, params[name])
        betas[name] = np.zeros(model.X.shape[1])
        sp_dict[name] = []

    pen_ranks_dict: dict[str, list[int]] = {}
    n_unpen_dict: dict[str, int] = {}
    for name in param_names:
        model = models[name]
        if model.penalties and method_upper in ("REML", "ML"):
            pen_ranks_dict[name] = _penalty_ranks(model.penalties)
            n_unpen = (1 if model.has_intercept else 0) + model.n_parametric
            for s_info in model.smooths:
                n_unpen += s_info.null_space_dim
            n_unpen_dict[name] = n_unpen
        else:
            pen_ranks_dict[name] = []
            n_unpen_dict[name] = 0

    ll_old = -np.inf
    converged = False
    n_iter = 0

    for outer in range(max_outer):
        n_iter += 1

        for name in param_names:
            model = models[name]
            X = model.X
            offset = model.offset

            z, W_irls = _compute_zw(family, name, y, params, etas[name])

            for _inner in range(max_inner):
                if model.penalties:
                    sp_dict[name] = _select_sp_for_param(
                        X,
                        z,
                        model.penalties,
                        method_upper,
                        pen_ranks_dict[name],
                        n_unpen_dict[name],
                        W=W_irls,
                        offset=offset,
                    )

                sp = sp_dict[name] if sp_dict[name] else [1.0] * len(model.penalties)
                beta, _ = _penalized_solve(X, z, model.penalties, sp, W=W_irls, offset=offset)

                eta_new = X @ beta
                if offset is not None:
                    eta_new = eta_new + offset
                theta_new = family.link_inverse(name, eta_new)

                params_trial = dict(params)
                params_trial[name] = theta_new

                z_new, W_new = _compute_zw(family, name, y, params_trial, eta_new)

                rel_change = np.max(np.abs(beta - betas[name])) / (np.max(np.abs(beta)) + 1e-8)
                betas[name] = beta
                etas[name] = eta_new
                params[name] = theta_new
                W_irls = W_new
                z = z_new

                if rel_change < tol:
                    break

        ll_new = family.log_likelihood(y, params)
        if np.isfinite(ll_new) and abs(ll_new - ll_old) / (abs(ll_old) + 0.1) < tol:
            converged = True
            break
        ll_old = ll_new

    ll_final = family.log_likelihood(y, params)
    global_dev = -2.0 * ll_final

    param_results: dict[str, GAMLSSParamResult] = {}
    edf_total_all = 0.0
    for name in param_names:
        model = models[name]
        X = model.X
        sp = sp_dict[name] if sp_dict[name] else [1.0] * len(model.penalties)
        smooths_info = [(s.col_start, s.col_end) for s in model.smooths]

        _, W_irls = _compute_zw(family, name, y, params, etas[name])

        edf = _edf_per_smooth(X, model.penalties, sp, smooths_info, W=W_irls)
        edf_total = sum(edf) + (1 if model.has_intercept else 0) + model.n_parametric

        param_results[name] = GAMLSSParamResult(
            coefficients=betas[name],
            linear_predictor=etas[name],
            fitted_values=params[name],
            smoothing_params=sp,
            edf=edf,
            edf_total=edf_total,
        )
        edf_total_all += edf_total

    aic = -2.0 * ll_final + 2.0 * edf_total_all
    bic = -2.0 * ll_final + np.log(n) * edf_total_all

    return GAMLSSFitResult(
        params=param_results,
        log_likelihood=ll_final,
        global_deviance=global_dev,
        n_iter=n_iter,
        converged=converged,
        aic=aic,
        bic=bic,
        n_obs=n,
        response=y,
    )
