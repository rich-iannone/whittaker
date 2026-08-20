r"""GAMLSS fitting via the RS algorithm (Rigby & Stasinopoulos 2005)."""

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
    """Fit result for a single distributional parameter.

    One `GAMLSSParamResult` is produced per parameter of the `GAMLSSFamily` (e.g. one for `mu` and
    one for `sigma` in a location-scale model), holding that parameter's final coefficients,
    predictor, fitted values, and smoothing diagnostics from the RS algorithm.

    Attributes
    ----------
    coefficients:
        Fitted coefficient vector `beta_k` for this parameter's model matrix.
    linear_predictor:
        Final linear predictor `eta_k = X_k @ beta_k` (plus offset, if any).
    fitted_values:
        Fitted values on the parameter's own scale, `theta_k = g_k^{-1}(eta_k)`.
    smoothing_params:
        Selected smoothing parameters, one per penalized smooth term in this parameter's formula.
    edf:
        Effective degrees of freedom for each smooth term.
    edf_total:
        Total effective degrees of freedom for this parameter (sum of `edf` plus unpenalized terms).
    """

    coefficients: NDArray
    linear_predictor: NDArray
    fitted_values: NDArray
    smoothing_params: list[float]
    edf: list[float]
    edf_total: float


@dataclass
class GAMLSSFitResult:
    """Result of a GAMLSS fit.

    Aggregates per-parameter results (`params`) together with fit-level statistics computed from
    the joint log-likelihood across all parameters.

    Attributes
    ----------
    params:
        Dict mapping parameter name to its `GAMLSSParamResult`.
    log_likelihood:
        Final joint log-likelihood of the response under the fitted distribution.
    global_deviance:
        `-2 * log_likelihood`, the GAMLSS analogue of deviance, summed across all observations and
        aggregated over all distributional parameters.
    n_iter:
        Number of outer RS iterations performed.
    converged:
        Whether the RS algorithm converged (relative change in log-likelihood below `tol`) before
        exhausting `max_outer` iterations.
    aic:
        Akaike information criterion, `global_deviance + 2 * edf_total_all`, where `edf_total_all`
        sums the effective degrees of freedom across all parameters.
    bic:
        Bayesian information criterion, `global_deviance + log(n) * edf_total_all`.
    n_obs:
        Number of observations.
    response:
        The response vector used for fitting.
    """

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
    if n_sp == 0:  # pragma: no cover
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
    r"""Fit a GAMLSS model using the RS algorithm.

    Implements the outer loop of the Rigby & Stasinopoulos (2005) fitting algorithm for
    Generalized Additive Models for Location, Scale, and Shape. Each distributional parameter
    `theta_k` has its own model matrix, penalties, and link function `g_k`. On each outer
    iteration, the algorithm cycles through the parameters in `family.parameter_names` order and,
    for each one, forms working responses and IRLS weights from the family's score and Fisher
    information with respect to that parameter (via `_compute_zw`), holding all other parameters
    fixed at their current fitted values:

    $$z_k = \eta_k + \frac{\ell'_k}{\ell''_k} \, g_k'(\theta_k), \qquad
    w_k = -\ell''_k \, / \, g_k'(\theta_k)^2$$

    where `\ell'_k` and `\ell''_k` are the first and second derivatives of the per-observation
    log-likelihood with respect to `theta_k`. A penalized weighted least squares problem is then
    solved for `beta_k` (re-selecting smoothing parameters at each inner iteration if the parameter
    has penalized smooth terms), and the process repeats for up to `max_inner` iterations or until
    the relative change in `beta_k` is below `tol`. After all parameters have been updated once, the
    joint log-likelihood is recomputed; the outer loop stops when its relative change is below `tol`
    or after `max_outer` iterations.

    Parameters
    ----------
    models:
        Dict mapping each parameter name to its `ModelMatrix` (design matrix, penalties, and smooth
        term metadata), one per parameter in `family.parameter_names`.
    family:
        A `GAMLSSFamily` supplying the link functions, their derivatives, the log-likelihood, and
        the score/information functions (`dl_dtheta`, `d2l_dtheta2`) needed to form IRLS working
        weights for each parameter.
    y:
        Response vector, shared across all parameters.
    method:
        Smoothing parameter selection method used for every parameter's smooth terms: `"GCV"`
        (default), `"REML"`, or `"ML"`.
    max_outer:
        Maximum number of outer RS iterations (passes over all parameters).
    max_inner:
        Maximum inner IRLS iterations per parameter within a single outer step.
    tol:
        Relative convergence tolerance, used both for the inner coefficient updates and for the
        outer log-likelihood change.

    Returns
    -------
    GAMLSSFitResult
        The fitted per-parameter results together with joint log-likelihood, global deviance, AIC,
        BIC, and convergence diagnostics.
    """
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

    for _outer in range(max_outer):
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
