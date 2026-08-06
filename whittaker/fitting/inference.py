"""Approximate p-values for smooth terms (Wood 2013)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from whittaker.fitting.pirls import FitResult
from whittaker.model_matrix import ModelMatrix


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

        results.append(
            SmoothTestResult(
                term_label=repr(info.term),
                stat=stat,
                edf=edf_j,
                ref_df=ref_df,
                p_value=pval,
            )
        )

    return results
