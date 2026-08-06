"""Inference for GAM terms: parametric Wald tests, smooth p-values, and concurvity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist

from whittaker.fitting.pirls import FitResult
from whittaker.model_matrix import ModelMatrix


@dataclass
class ParametricTestResult:
    """Result of a Wald test for a parametric coefficient.

    Attributes
    ----------
    term_label:
        Human-readable label for the term.
    estimate:
        Coefficient estimate β̂.
    se:
        Standard error of β̂.
    stat:
        Test statistic (t for Gaussian, z for known-scale families).
    p_value:
        Two-sided p-value.
    """

    term_label: str
    estimate: float
    se: float
    stat: float
    p_value: float


@dataclass
class SmoothTestResult:
    """Result of an approximate test for H_0: f_j = 0.

    Attributes
    ----------
    term_label:
        Human-readable label for the smooth term.
    stat:
        Chi-squared test statistic.
    edf:
        Effective degrees of freedom for the smooth.
    ref_df:
        Reference degrees of freedom for the chi-squared test.
    p_value:
        Approximate p-value.
    """

    term_label: str
    stat: float
    edf: float
    ref_df: float
    p_value: float


def _bayesian_covariance(
    X: NDArray,
    penalties: list[NDArray],
    sp: list[float],
    scale: float,
    W: NDArray | None = None,
) -> NDArray:
    """Compute V_β = scale * (X'WX + S_λ)^{-1}."""
    if W is not None:
        sqrtW = np.sqrt(W)
        Xw = X * sqrtW[:, np.newaxis]
    else:
        Xw = X

    XtWX = Xw.T @ Xw

    S_total = np.zeros_like(XtWX)
    for lam, pen in zip(sp, penalties):
        S_total += lam * pen

    A = XtWX + S_total
    A = (A + A.T) * 0.5

    eigvals, eigvecs = np.linalg.eigh(A)
    p = A.shape[0]
    tol = np.max(eigvals) * p * np.finfo(float).eps
    eigvals_inv = np.zeros_like(eigvals)
    keep = eigvals > tol
    eigvals_inv[keep] = 1.0 / eigvals[keep]

    V_beta = (eigvecs * eigvals_inv[np.newaxis, :]) @ eigvecs.T
    return scale * V_beta


def _smooth_test(
    beta_j: NDArray,
    V_j: NDArray,
    X_j: NDArray,
    edf_j: float,
) -> tuple[float, float, float]:
    """Compute approximate p-value for H_0: f_j = 0.

    Follows Wood (2013) "On p-values for smooth components of an extended generalized additive
    model". Uses the QR-transformed Bayesian covariance with eigendecomposition to determine the
    effective test rank and reference distribution.

    Parameters
    ----------
    beta_j:
        Coefficient estimates for smooth j.
    V_j:
        Bayesian covariance block for smooth j.
    X_j:
        Model matrix columns for smooth j.
    edf_j:
        Effective degrees of freedom for smooth j.

    Returns
    -------
    stat:
        Chi-squared test statistic.
    ref_df:
        Reference degrees of freedom (may be non-integer).
    p_value:
        Approximate p-value.
    """
    k = len(beta_j)

    Q, R = np.linalg.qr(X_j, mode="reduced")
    V_trans = R @ V_j @ R.T
    V_trans = (V_trans + V_trans.T) * 0.5

    eigvals, eigvecs = np.linalg.eigh(V_trans)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    tol = 5.0 * np.finfo(float).eps * max(eigvals[0], 0.0)
    n_pos = int(np.sum(eigvals > tol))

    if n_pos == 0:
        return 0.0, 1.0, 1.0

    r = max(1, int(np.ceil(edf_j)))
    r = min(r, n_pos)

    gamma = R @ beta_j
    proj = eigvecs[:, :r].T @ gamma
    stat = float(np.sum(proj**2 / eigvals[:r]))

    ref_df = max(float(r), 1.0)
    p_value = float(chi2.sf(stat, df=ref_df))

    return stat, ref_df, p_value


def smooth_tests(
    fit: FitResult,
    model: ModelMatrix,
) -> list[SmoothTestResult]:
    """Compute approximate p-values for all smooth terms.

    Parameters
    ----------
    fit:
        Result from `pirls_fit()`.
    model:
        The model matrix used for fitting.

    Returns
    -------
    list[SmoothTestResult]
        One result per smooth term, in formula order.
    """
    X = model.X
    V_beta = _bayesian_covariance(
        X,
        model.penalties,
        fit.smoothing_params,
        fit.scale,
        W=fit.weights,
    )

    results: list[SmoothTestResult] = []
    for idx, info in enumerate(model.smooths):
        cs, ce = info.col_start, info.col_end
        beta_j = fit.coefficients[cs:ce]
        V_j = V_beta[cs:ce, cs:ce]
        X_j = X[:, cs:ce]
        edf_j = fit.edf[idx]

        stat, ref_df, pval = _smooth_test(beta_j, V_j, X_j, edf_j)

        label = repr(info.term)
        if info.by_level is not None:
            label = f"{label}:{info.by_level}"

        results.append(
            SmoothTestResult(
                term_label=label,
                stat=stat,
                edf=edf_j,
                ref_df=ref_df,
                p_value=pval,
            )
        )

    return results


def parametric_tests(
    fit: FitResult,
    model: ModelMatrix,
    scale_known: bool,
) -> list[ParametricTestResult]:
    """Compute Wald tests for parametric (non-smooth) coefficients.

    For families with unknown scale (Gaussian, Gamma), uses the t-distribution with residual degrees
    of freedom. For known-scale families (Binomial, Poisson), uses the standard normal (z-test).

    Parameters
    ----------
    fit:
        Result from `pirls_fit()`.
    model:
        The model matrix used for fitting.
    scale_known:
        Whether the family has a known scale parameter.

    Returns
    -------
    list[ParametricTestResult]
        One result per parametric coefficient (intercept + linear/interaction terms).
    """
    X = model.X
    V_beta = _bayesian_covariance(
        X,
        model.penalties,
        fit.smoothing_params,
        fit.scale,
        W=fit.weights,
    )

    n = X.shape[0]
    residual_df = n - fit.edf_total

    results: list[ParametricTestResult] = []

    n_param_cols = (1 if model.has_intercept else 0) + model.n_parametric
    for j in range(n_param_cols):
        beta_j = float(fit.coefficients[j])
        se_j = float(np.sqrt(max(V_beta[j, j], 0.0)))

        if se_j < 1e-30:
            stat = 0.0
            pval = 1.0
        elif scale_known:
            stat = beta_j / se_j
            pval = float(2.0 * norm.sf(abs(stat)))
        else:
            stat = beta_j / se_j
            pval = float(2.0 * t_dist.sf(abs(stat), df=max(residual_df, 1.0)))

        results.append(
            ParametricTestResult(
                term_label=model.column_names[j],
                estimate=beta_j,
                se=se_j,
                stat=stat,
                p_value=pval,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Concurvity diagnostics (Wood 2017, §1.8.1)
# ---------------------------------------------------------------------------


@dataclass
class ConcurvityResult:
    """Concurvity diagnostics for smooth terms.

    Values range from 0 (no concurvity) to 1 (complete confounding). When `full=True`, each array
    has shape `(n_smooths,)` measuring each smooth against all other model terms combined. When
    `full=False`, each array has shape `(n_smooths, n_smooths)` with pairwise measures.

    Attributes
    ----------
    worst:
        Upper-bound concurvity: the maximum proportion of each smooth's basis space that lies in the
        space of the comparator.
    observed:
        Concurvity of the actual fitted smooth function.
    estimate:
        Concurvity based on the estimated smooth's squared norm relative to the null model.
    labels:
        Smooth term labels in the same order as the array axes.
    full:
        Whether this is a full (overall) or pairwise result.
    """

    worst: NDArray
    observed: NDArray
    estimate: NDArray
    labels: list[str] = field(default_factory=list)
    full: bool = True


def _qr_q(M: NDArray) -> NDArray:
    """Thin QR: return only Q with orthonormal columns spanning col(M)."""
    Q, R = np.linalg.qr(M, mode="reduced")
    diag_abs = np.abs(np.diag(R))
    tol = max(diag_abs.max(), 1.0) * M.shape[0] * np.finfo(float).eps
    keep = diag_abs > tol
    return Q[:, keep]


def _worst_concurvity(X_j: NDArray, Q_rest: NDArray) -> float:
    """Largest proportion of col(X_j) that lies in col(Q_rest)."""
    if Q_rest.shape[1] == 0:
        return 0.0
    proj_coefs = Q_rest.T @ X_j
    U, s, Vt = np.linalg.svd(X_j, full_matrices=False)
    good = s > 1e-10 * max(s[0], 1e-30)
    if not np.any(good):
        return 0.0
    M = proj_coefs @ Vt[good].T @ np.diag(1.0 / s[good])
    sv = np.linalg.svd(M, compute_uv=False)
    return float(min(sv[0] ** 2, 1.0)) if len(sv) > 0 else 0.0


def _observed_concurvity(f_j: NDArray, Q_rest: NDArray) -> float:
    """Proportion of the fitted smooth that lies in the comparator's space."""
    f_c = f_j - f_j.mean()
    ss_total = float(f_c @ f_c)
    if ss_total < 1e-30:
        return 0.0
    proj = Q_rest.T @ f_c
    ss_proj = float(proj @ proj)
    return min(ss_proj / ss_total, 1.0)


def _estimate_concurvity(f_j: NDArray, Q_rest: NDArray) -> float:
    """Concurvity based on R-squared of f_j regressed on the comparator space."""
    f_c = f_j - f_j.mean()
    ss_total = float(f_c @ f_c)
    if ss_total < 1e-30:
        return 0.0
    proj = Q_rest @ (Q_rest.T @ f_c)
    ss_resid = float((f_c - proj) @ (f_c - proj))
    return max(min(1.0 - ss_resid / ss_total, 1.0), 0.0)


def concurvity(
    fit: FitResult,
    model: ModelMatrix,
    *,
    full: bool = True,
) -> ConcurvityResult:
    """Compute concurvity diagnostics for all smooth terms.

    Concurvity is the GAM analogue of collinearity: it measures how well each smooth can be
    approximated by the other terms in the model. High concurvity (> 0.8) indicates that smooth
    estimates may be unstable.

    Parameters
    ----------
    fit:
        Result from `pirls_fit()`.
    model:
        The model matrix used for fitting.
    full:
        If `True` (default), compute overall concurvity for each smooth against all remaining model
        terms. If `False`, compute pairwise concurvity between each pair of smooths.

    Returns
    -------
    ConcurvityResult
        Concurvity diagnostics with `worst`, `observed`, and `estimate` measures.
    """
    X = model.X
    smooths = model.smooths
    n_smooths = len(smooths)

    labels = []
    for info in smooths:
        label = repr(info.term)
        if info.by_level is not None:
            label = f"{label}:{info.by_level}"
        labels.append(label)

    smooth_ranges = [(info.col_start, info.col_end) for info in smooths]

    fitted_smooths = []
    for cs, ce in smooth_ranges:
        fitted_smooths.append(X[:, cs:ce] @ fit.coefficients[cs:ce])

    if full:
        worst = np.zeros(n_smooths)
        observed = np.zeros(n_smooths)
        estimate = np.zeros(n_smooths)

        for j in range(n_smooths):
            cs_j, ce_j = smooth_ranges[j]
            X_j = X[:, cs_j:ce_j]
            f_j = fitted_smooths[j]

            rest_cols = list(range(cs_j)) + list(range(ce_j, X.shape[1]))
            X_rest = X[:, rest_cols]
            Q_rest = _qr_q(X_rest)

            worst[j] = _worst_concurvity(X_j, Q_rest)
            observed[j] = _observed_concurvity(f_j, Q_rest)
            estimate[j] = _estimate_concurvity(f_j, Q_rest)

        return ConcurvityResult(
            worst=worst, observed=observed, estimate=estimate, labels=labels, full=True
        )

    worst = np.zeros((n_smooths, n_smooths))
    observed = np.zeros((n_smooths, n_smooths))
    estimate = np.zeros((n_smooths, n_smooths))

    for i in range(n_smooths):
        cs_i, ce_i = smooth_ranges[i]
        X_i = X[:, cs_i:ce_i]
        f_i = fitted_smooths[i]

        for j in range(n_smooths):
            if i == j:
                worst[i, j] = 1.0
                observed[i, j] = 1.0
                estimate[i, j] = 1.0
                continue

            cs_j, ce_j = smooth_ranges[j]
            X_j = X[:, cs_j:ce_j]
            Q_j = _qr_q(X_j)

            worst[i, j] = _worst_concurvity(X_i, Q_j)
            observed[i, j] = _observed_concurvity(f_i, Q_j)
            estimate[i, j] = _estimate_concurvity(f_i, Q_j)

    return ConcurvityResult(
        worst=worst, observed=observed, estimate=estimate, labels=labels, full=False
    )
