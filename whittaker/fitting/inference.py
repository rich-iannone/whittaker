"""Inference for GAM terms: parametric Wald tests, smooth p-values, concurvity, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm
from scipy.stats import f as f_dist
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


def _unconditional_covariance(
    X: NDArray,
    penalties: list[NDArray],
    sp: list[float],
    scale: float,
    beta: NDArray,
    method: str,
    W: NDArray | None = None,
    penalty_ranks: list[int] | None = None,
    n_unpenalized: int = 0,
    y: NDArray | None = None,
    offset: NDArray | None = None,
) -> NDArray:
    """Compute the unconditional covariance V_c (Marra & Wood 2012).

    V_c = V_p + M V_ρ M'

    where V_p is the Bayesian posterior covariance conditional on ρ,
    V_ρ is the covariance of ρ = log(λ) from the inverse Hessian of the
    REML/ML objective, and M = dβ̂/dρ_j = -A⁻¹ λ_j S_j β̂.

    This accounts for the additional uncertainty from estimating the
    smoothing parameters, producing wider (more honest) intervals.

    Parameters
    ----------
    X:
        Model matrix, shape `(n, p)`.
    penalties:
        Penalty matrices, each shape `(p, p)`.
    sp:
        Smoothing parameters λ_j.
    scale:
        Estimated scale parameter φ.
    beta:
        Estimated coefficients β̂, shape `(p,)`.
    method:
        Smoothing parameter selection method (`"REML"` or `"ML"`).
    W:
        Combined weights (IRLS × prior), shape `(n,)`, or `None`.
    penalty_ranks:
        Ranks of each penalty matrix.
    n_unpenalized:
        Number of unpenalized parameters.
    y:
        Response vector (or pseudo-data for non-Gaussian), shape `(n,)`.
    offset:
        Offset vector, shape `(n,)`, or `None`.

    Returns
    -------
    NDArray
        Unconditional covariance matrix V_c, shape `(p, p)`.
    """
    from whittaker.fitting.pirls import _reml_objective

    V_p = _bayesian_covariance(X, penalties, sp, scale, W=W)

    n_sp = len(sp)
    if n_sp == 0:
        return V_p

    log_sp = np.log(np.array(sp))

    if penalty_ranks is None:
        from whittaker.fitting.pirls import _penalty_ranks

        penalty_ranks = _penalty_ranks(penalties)

    scale_known = scale == 1.0

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
    A_inv = (eigvecs * eigvals_inv[np.newaxis, :]) @ eigvecs.T

    M = np.zeros((p, n_sp))
    for j, (lam_j, pen_j) in enumerate(zip(sp, penalties)):
        M[:, j] = -A_inv @ (lam_j * pen_j @ beta)

    ml = method.upper() == "ML"

    if y is None:
        y = X @ beta

    eps = 1e-4
    H = np.zeros((n_sp, n_sp))
    for j in range(n_sp):
        rho_plus = log_sp.copy()
        rho_plus[j] += eps
        rho_minus = log_sp.copy()
        rho_minus[j] -= eps

        _, grad_plus = _reml_objective(
            rho_plus,
            X,
            y,
            penalties,
            penalty_ranks,
            n_unpenalized,
            scale_known,
            scale,
            W=W,
            offset=offset,
            ml=ml,
        )
        _, grad_minus = _reml_objective(
            rho_minus,
            X,
            y,
            penalties,
            penalty_ranks,
            n_unpenalized,
            scale_known,
            scale,
            W=W,
            offset=offset,
            ml=ml,
        )
        H[j, :] = (grad_plus - grad_minus) / (2.0 * eps)

    H = (H + H.T) * 0.5

    try:
        eigvals_h, eigvecs_h = np.linalg.eigh(H)
        tol_h = np.max(np.abs(eigvals_h)) * n_sp * np.finfo(float).eps
        eigvals_h_inv = np.zeros_like(eigvals_h)
        keep_h = eigvals_h > tol_h
        eigvals_h_inv[keep_h] = 1.0 / eigvals_h[keep_h]
        V_rho = (eigvecs_h * eigvals_h_inv[np.newaxis, :]) @ eigvecs_h.T
    except np.linalg.LinAlgError:
        return V_p

    V_c = V_p + M @ V_rho @ M.T
    V_c = (V_c + V_c.T) * 0.5

    return V_c


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


def _combined_weights(fit: FitResult) -> NDArray | None:
    """Combine IRLS working weights and prior weights."""
    W_irls = fit.weights
    pw = fit.prior_weights
    if W_irls is not None and pw is not None:
        return pw * W_irls
    if W_irls is not None:
        return W_irls
    if pw is not None:
        return pw
    return None


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
        W=_combined_weights(fit),
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
        W=_combined_weights(fit),
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


# ---------------------------------------------------------------------------
# ANOVA-style model comparison (deviance-difference tests)
# ---------------------------------------------------------------------------


@dataclass
class AnovaModelRow:
    """One row of an ANOVA model-comparison table.

    Attributes
    ----------
    resid_df:
        Residual degrees of freedom (n − edf_total).
    resid_dev:
        Residual deviance.
    df:
        Difference in residual df from the previous (simpler) model. `None` for the first model.
    deviance:
        Deviance difference from the previous model. `None` for the first model.
    stat:
        Test statistic (chi-squared or F). `None` for the first model.
    p_value:
        p-value from the test. `None` for the first model.
    """

    resid_df: float
    resid_dev: float
    df: float | None = None
    deviance: float | None = None
    stat: float | None = None
    p_value: float | None = None


@dataclass
class AnovaResult:
    """Result of an ANOVA-style comparison of nested GAM models.

    Attributes
    ----------
    rows:
        One `AnovaModelRow` per model, ordered from simplest to most complex.
    scale:
        Dispersion parameter used for the test. For known-scale families this is `1`. For
        unknown-scale families it is the estimated scale from the most complex model.
    test:
        `"Chisq"` or `"F"`, depending on whether the family has known scale.
    """

    rows: list[AnovaModelRow]
    scale: float
    test: str

    def __str__(self) -> str:
        stat_label = self.test
        lines = [
            "Analysis of Deviance Table",
            "",
            f"{'Model':>7} {'Resid.Df':>10} {'Resid.Dev':>12} {'Df':>8} {'Deviance':>12}"
            f" {stat_label:>10} {'Pr(>' + stat_label + ')':>14}",
            f"{'-----':>7} {'-' * 10} {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 14}",
        ]
        for i, row in enumerate(self.rows):
            label = str(i + 1)
            df_str = f"{row.df:8.2f}" if row.df is not None else " " * 8
            dev_str = f"{row.deviance:12.4f}" if row.deviance is not None else " " * 12
            stat_str = f"{row.stat:10.4f}" if row.stat is not None else " " * 10
            p_str = f"{row.p_value:14.6g}" if row.p_value is not None else " " * 14
            lines.append(
                f"{label:>7} {row.resid_df:10.2f} {row.resid_dev:12.4f} "
                f"{df_str} {dev_str} {stat_str} {p_str}"
            )
        return "\n".join(lines)


def anova_gam(
    *models: tuple[FitResult, ModelMatrix],
    scale_known: bool,
    scale_override: float | None = None,
) -> AnovaResult:
    """ANOVA-style deviance-difference tests for nested GAM models.

    Compares a sequence of nested models (simplest to most complex) using sequential
    deviance-difference tests. For known-scale families (Poisson, Binomial) a chi-squared test is
    used. For unknown-scale families (Gaussian, Gamma) an F-test is used.

    Parameters
    ----------
    *models:
        Two or more `(FitResult, ModelMatrix)` tuples. Models are automatically sorted from simplest
        (fewest edf) to most complex.
    scale_known:
        Whether the family has a known scale parameter.
    scale_override:
        If provided, use this scale instead of estimating from the largest model. Only relevant for
        unknown-scale families.

    Returns
    -------
    AnovaResult
        Table of sequential deviance comparisons.

    Raises
    ------
    ValueError
        If fewer than 2 models are provided, or models have different numbers of observations.
    """
    if len(models) < 2:
        raise ValueError("anova_gam requires at least 2 models.")

    n_obs_set = {mm.n_obs for _, mm in models}
    if len(n_obs_set) > 1:
        raise ValueError(f"All models must be fitted to the same data (got n_obs = {n_obs_set}).")

    sorted_models = sorted(models, key=lambda pair: pair[0].edf_total)

    n = sorted_models[0][1].n_obs

    if scale_known:
        phi = 1.0
        test_name = "Chisq"
    else:
        largest_fit = sorted_models[-1][0]
        phi = scale_override if scale_override is not None else largest_fit.scale
        test_name = "F"

    rows: list[AnovaModelRow] = []

    fit_0, mm_0 = sorted_models[0]
    rows.append(AnovaModelRow(resid_df=n - fit_0.edf_total, resid_dev=fit_0.deviance))

    for i in range(1, len(sorted_models)):
        fit_prev, _ = sorted_models[i - 1]
        fit_curr, _ = sorted_models[i]

        delta_df = fit_curr.edf_total - fit_prev.edf_total
        delta_dev = fit_prev.deviance - fit_curr.deviance

        resid_df = n - fit_curr.edf_total

        if delta_df <= 0:
            rows.append(
                AnovaModelRow(
                    resid_df=resid_df,
                    resid_dev=fit_curr.deviance,
                    df=delta_df,
                    deviance=delta_dev,
                )
            )
            continue

        if scale_known:
            stat = delta_dev / phi
            p_value = float(chi2.sf(stat, df=delta_df))
        else:
            stat = (delta_dev / delta_df) / phi
            p_value = float(f_dist.sf(stat, dfn=delta_df, dfd=max(resid_df, 1.0)))

        rows.append(
            AnovaModelRow(
                resid_df=resid_df,
                resid_dev=fit_curr.deviance,
                df=delta_df,
                deviance=delta_dev,
                stat=stat,
                p_value=p_value,
            )
        )

    return AnovaResult(rows=rows, scale=phi, test=test_name)


# ---------------------------------------------------------------------------
# Basis dimension adequacy check (k-index test)
# ---------------------------------------------------------------------------


@dataclass
class KCheckResult:
    """Result of basis dimension adequacy check for a single smooth.

    Attributes
    ----------
    term_label:
        Human-readable label for the smooth term.
    k_prime:
        Basis dimension after identifiability constraints (upper bound on EDF).
    edf:
        Effective degrees of freedom for the smooth.
    k_index:
        Ratio of neighbor-differencing variance estimate to overall residual variance. Values well
        below 1 suggest `k` may be too small.
    p_value:
        Simulation-based p-value. Low values indicate the basis dimension may be inadequate.
    """

    term_label: str
    k_prime: int
    edf: float
    k_index: float
    p_value: float


def _k_index_1d(residuals: NDArray, covariate: NDArray) -> float:
    """Compute the k-index for a single smooth by neighbor-differencing."""
    order = np.argsort(covariate)
    r_sorted = residuals[order]
    diffs = np.diff(r_sorted)
    neighbor_var = np.mean(diffs**2) / 2.0
    resid_var = np.var(residuals, ddof=1)
    if resid_var < 1e-15:
        return 1.0
    return neighbor_var / resid_var


def _k_index_nd(residuals: NDArray, covariates: list[NDArray]) -> float:
    """Compute the k-index for a multi-dimensional smooth.

    Orders observations by their distance to a random reference point (the observation nearest the
    median of each covariate), then applies the neighbor-differencing estimator.
    """
    coords = np.column_stack(covariates)
    center = np.median(coords, axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    order = np.argsort(dists)
    r_sorted = residuals[order]
    diffs = np.diff(r_sorted)
    neighbor_var = np.mean(diffs**2) / 2.0
    resid_var = np.var(residuals, ddof=1)
    if resid_var < 1e-15:
        return 1.0
    return neighbor_var / resid_var


def k_check(
    fit: FitResult,
    mm: ModelMatrix,
    data: dict[str, NDArray],
    residuals: NDArray,
    *,
    n_sim: int = 400,
) -> list[KCheckResult]:
    """Check basis dimension adequacy for all smooths.

    Parameters
    ----------
    fit:
        Fitted model result.
    mm:
        Model matrix with smooth metadata.
    data:
        Original data dict used during fitting.
    residuals:
        Deviance residuals from the fitted model.
    n_sim:
        Number of permutations for the p-value simulation.

    Returns
    -------
    list[KCheckResult]
        One result per smooth term.
    """
    rng = np.random.default_rng()
    results: list[KCheckResult] = []

    for i, info in enumerate(mm.smooths):
        variables = info.term.variables
        k_prime = info.col_end - info.col_start

        if len(variables) == 1:
            cov = np.asarray(data[variables[0]], dtype=float)
            k_obs = _k_index_1d(residuals, cov)
        else:
            covs = [np.asarray(data[v], dtype=float) for v in variables]
            k_obs = _k_index_nd(residuals, covs)

        count_le = 0
        for _ in range(n_sim):
            perm_resid = rng.permutation(residuals)
            if len(variables) == 1:
                k_sim = _k_index_1d(perm_resid, cov)
            else:
                k_sim = _k_index_nd(perm_resid, covs)
            if k_sim <= k_obs:
                count_le += 1

        p_value = count_le / n_sim

        label = repr(info.term)
        if info.by_level is not None:
            label += f":{info.by_level}"

        results.append(
            KCheckResult(
                term_label=label,
                k_prime=k_prime,
                edf=fit.edf[i],
                k_index=k_obs,
                p_value=p_value,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Influence / leverage diagnostics
# ---------------------------------------------------------------------------


@dataclass
class InfluenceResult:
    """Observation-level influence diagnostics.

    Attributes
    ----------
    hat_values:
        Leverage (diagonal of the hat matrix), shape `(n,)`.
    cooks_distance:
        Cook's distance for each observation, shape `(n,)`.
    """

    hat_values: NDArray
    cooks_distance: NDArray


def influence(fit: FitResult, mm: ModelMatrix) -> InfluenceResult:
    """Compute hat values and Cook's distance for each observation."""
    X = mm.X
    sp = fit.smoothing_params
    W = _combined_weights(fit)

    if W is not None:
        sqrtW = np.sqrt(W)
        Xw = X * sqrtW[:, np.newaxis]
    else:
        Xw = X

    S_total = np.zeros((X.shape[1], X.shape[1]))
    for lam, pen in zip(sp, mm.penalties):
        S_total += lam * pen

    A = Xw.T @ Xw + S_total
    A = (A + A.T) * 0.5

    eigvals, eigvecs = np.linalg.eigh(A)
    tol = np.max(eigvals) * A.shape[0] * np.finfo(float).eps
    eigvals_inv = np.zeros_like(eigvals)
    keep = eigvals > tol
    eigvals_inv[keep] = 1.0 / eigvals[keep]

    A_inv = (eigvecs * eigvals_inv[np.newaxis, :]) @ eigvecs.T

    H_diag = np.einsum("ij,jk,ik->i", Xw, A_inv, Xw)
    H_diag = np.clip(H_diag, 0.0, 1.0)

    resid = fit.residuals
    p_eff = fit.edf_total
    scale = fit.scale
    denom = scale * (1.0 - H_diag) ** 2
    denom = np.maximum(denom, np.finfo(float).eps)
    cooks_d = resid**2 * H_diag / (p_eff * denom)

    return InfluenceResult(hat_values=H_diag, cooks_distance=cooks_d)


# ---------------------------------------------------------------------------
# Randomized quantile residuals (Dunn & Smyth 1996)
# ---------------------------------------------------------------------------


def quantile_residuals(
    fit: FitResult,
    mm: ModelMatrix,
    family: object,
    *,
    seed: int | None = None,
) -> NDArray:
    """Compute randomized quantile residuals.

    For continuous families, the CDF at each observation should be uniform.
    For discrete families, a random uniform on [F(y-1), F(y)] is used.
    The result is transformed via the standard normal quantile function,
    so properly specified models produce approximately N(0,1) residuals.
    """
    from scipy.stats import beta as beta_dist
    from scipy.stats import gamma, nbinom, poisson

    rng = np.random.default_rng(seed)
    y = mm.response
    mu = fit.fitted_values
    scale = fit.scale

    family_name = type(family).__name__

    if family_name == "Gaussian":
        u = norm.cdf(y, loc=mu, scale=np.sqrt(np.maximum(scale, 1e-20)))
    elif family_name == "Poisson":
        u_upper = poisson.cdf(y, mu)
        u_lower = poisson.cdf(y - 1, mu)
        u = rng.uniform(u_lower, u_upper)
    elif family_name == "Binomial":
        from scipy.stats import binom

        u_upper = binom.cdf(y, 1, mu)
        u_lower = binom.cdf(y - 1, 1, mu)
        u = rng.uniform(u_lower, u_upper)
    elif family_name == "NegativeBinomial":
        theta = getattr(family, "theta", 1.0)
        p_nb = theta / (theta + mu)
        u_upper = nbinom.cdf(y, theta, p_nb)
        u_lower = nbinom.cdf(y - 1, theta, p_nb)
        u = rng.uniform(u_lower, u_upper)
    elif family_name == "Gamma":
        shape_param = 1.0 / max(scale, 1e-20)
        gamma_scale = mu / shape_param
        u = gamma.cdf(y, a=shape_param, scale=gamma_scale)
    elif family_name == "Beta":
        prec = 1.0 / max(scale, 1e-20)
        a = mu * prec
        b = (1.0 - mu) * prec
        u = beta_dist.cdf(y, a, b)
    else:
        u = norm.cdf((y - mu) / np.sqrt(scale * np.maximum(family.variance(mu), 1e-10)))

    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    return norm.ppf(u)


# ---------------------------------------------------------------------------
# Dispersion test
# ---------------------------------------------------------------------------


@dataclass
class DispersionTestResult:
    """Result of a dispersion test.

    Attributes
    ----------
    dispersion:
        Estimated dispersion ratio (should be ~1 for correctly specified Poisson/Binomial).
    chi2_stat:
        Chi-squared test statistic.
    p_value:
        Two-sided p-value.
    """

    dispersion: float
    chi2_stat: float
    p_value: float


def dispersion_test(fit: FitResult, mm: ModelMatrix, family: object) -> DispersionTestResult:
    """Test for overdispersion in Poisson or Binomial models."""
    y = mm.response
    mu = fit.fitted_values
    n = len(y)
    p_eff = fit.edf_total

    v = family.variance(mu)
    pearson_resid_sq = (y - mu) ** 2 / np.maximum(v, np.finfo(float).eps)

    X2 = float(np.sum(pearson_resid_sq))
    residual_df = n - p_eff
    dispersion = X2 / residual_df if residual_df > 0 else X2 / n

    p_value = float(chi2.sf(X2, df=max(residual_df, 1)))

    return DispersionTestResult(
        dispersion=dispersion,
        chi2_stat=X2,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# VIF for parametric terms
# ---------------------------------------------------------------------------


@dataclass
class VIFResult:
    """Variance inflation factor for a parametric term."""

    term: str
    vif: float


def vif(mm: ModelMatrix) -> list[VIFResult]:
    """Compute variance inflation factors for parametric (linear) terms.

    VIF measures collinearity among the parametric predictors. Only applies to
    models with 2+ parametric terms (excluding the intercept).
    """
    start = 1 if mm.has_intercept else 0
    end = start + mm.n_parametric

    if mm.n_parametric < 2:
        return []

    X_param = mm.X[:, start:end]
    param_names = mm.column_names[start:end]

    results = []
    for j in range(X_param.shape[1]):
        y_j = X_param[:, j]
        X_others = np.delete(X_param, j, axis=1)
        X_design = np.column_stack([np.ones(len(y_j)), X_others])

        beta, _, _, _ = np.linalg.lstsq(X_design, y_j, rcond=None)
        y_hat = X_design @ beta
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r_sq = 1.0 - ss_res / max(ss_tot, np.finfo(float).eps)
        vif_val = 1.0 / max(1.0 - r_sq, np.finfo(float).eps)

        results.append(VIFResult(term=param_names[j], vif=vif_val))

    return results


# ---------------------------------------------------------------------------
# Derivative-based inference
# ---------------------------------------------------------------------------


@dataclass
class DerivativeResult:
    """Result of smooth derivative estimation.

    Attributes
    ----------
    term:
        Label for the smooth term.
    x:
        Covariate values at which derivatives are evaluated.
    derivative:
        Estimated derivative values, shape `(n,)`.
    se:
        Standard errors of the derivative estimates.
    lower:
        Lower confidence band.
    upper:
        Upper confidence band.
    level:
        Confidence level used.
    order:
        Derivative order (1 or 2).
    """

    term: str
    x: NDArray
    derivative: NDArray
    se: NDArray
    lower: NDArray
    upper: NDArray
    level: float
    order: int


def smooth_derivatives(
    fit: FitResult,
    mm: ModelMatrix,
    variable: str,
    data: dict[str, NDArray],
    *,
    order: int = 1,
    n_points: int = 200,
    level: float = 0.95,
    eps: float | None = None,
    unconditional: bool = False,
    V_beta: NDArray | None = None,
) -> list[DerivativeResult]:
    """Estimate derivatives of smooth terms with respect to a variable.

    Uses central finite differences on the basis matrix: for each evaluation
    point x, compute [B(x+eps) - B(x-eps)] / (2*eps) to get the derivative of
    each basis function, then multiply by the coefficient vector.

    Standard errors are computed via the delta method.
    """
    from whittaker.model_matrix import predict_matrix

    beta = fit.coefficients
    n_data = len(next(iter(data.values())))

    x_var = data.get(variable)
    if x_var is None:
        raise ValueError(f"Variable {variable!r} not found in data.")

    x_grid = np.linspace(float(x_var.min()), float(x_var.max()), n_points)

    if eps is None:
        x_range = float(x_var.max()) - float(x_var.min())
        eps = x_range / (n_points * 10) if order == 1 else x_range / (n_points * 5)
        eps = max(eps, 1e-7)

    base_data = {
        k: np.full(n_points, np.mean(v)) if k != variable else x_grid for k, v in data.items()
    }

    if V_beta is None:
        W = fit.weights
        if fit.prior_weights is not None and W is None:
            W = fit.prior_weights
        V_beta = _bayesian_covariance(mm.X, mm.penalties, fit.smoothing_params, fit.scale, W=W)

    z_crit = norm.ppf(1.0 - (1.0 - level) / 2.0)

    results = []
    for info in mm.smooths:
        term_vars = info.term.variables
        if variable not in term_vars:
            continue

        cs, ce = info.col_start, info.col_end
        beta_j = beta[cs:ce]
        V_j = V_beta[cs:ce, cs:ce]

        label = repr(info.term)
        if info.by_level is not None:
            label = f"{label}:{info.by_level}"

        if order == 1:
            data_plus = {k: (v + eps if k == variable else v.copy()) for k, v in base_data.items()}
            data_minus = {k: (v - eps if k == variable else v.copy()) for k, v in base_data.items()}
            X_plus = predict_matrix(mm, data_plus)
            X_minus = predict_matrix(mm, data_minus)
            dX_j = (X_plus[:, cs:ce] - X_minus[:, cs:ce]) / (2.0 * eps)
        elif order == 2:
            data_plus = {k: (v + eps if k == variable else v.copy()) for k, v in base_data.items()}
            data_minus = {k: (v - eps if k == variable else v.copy()) for k, v in base_data.items()}
            data_center = base_data
            X_plus = predict_matrix(mm, data_plus)
            X_minus = predict_matrix(mm, data_minus)
            X_center = predict_matrix(mm, data_center)
            dX_j = (X_plus[:, cs:ce] - 2.0 * X_center[:, cs:ce] + X_minus[:, cs:ce]) / (eps**2)
        else:
            raise ValueError(f"order must be 1 or 2, got {order}")

        deriv_vals = dX_j @ beta_j
        var_deriv = np.sum(dX_j * (dX_j @ V_j), axis=1)
        se_deriv = np.sqrt(np.maximum(var_deriv, 0.0))

        lower = deriv_vals - z_crit * se_deriv
        upper = deriv_vals + z_crit * se_deriv

        results.append(
            DerivativeResult(
                term=label,
                x=x_grid,
                derivative=deriv_vals,
                se=se_deriv,
                lower=lower,
                upper=upper,
                level=level,
                order=order,
            )
        )

    if not results:
        raise ValueError(f"No smooth terms found involving variable {variable!r}.")

    return results


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------


@dataclass
class MarginalEffectResult:
    """Result of marginal effect estimation for one smooth term.

    Attributes
    ----------
    term:
        Label for the smooth term.
    variable:
        The focal variable.
    x:
        Covariate values for the focal variable.
    effect:
        Estimated marginal effect (partial effect on the linear predictor).
    se:
        Standard errors.
    lower:
        Lower confidence band.
    upper:
        Upper confidence band.
    level:
        Confidence level used.
    by_values:
        Dict of conditioning variable values, if any.
    """

    term: str
    variable: str
    x: NDArray
    effect: NDArray
    se: NDArray
    lower: NDArray
    upper: NDArray
    level: float
    by_values: dict[str, float] | None = None


def marginal_effects(
    fit: FitResult,
    mm: ModelMatrix,
    variable: str,
    data: dict[str, NDArray],
    *,
    at: dict[str, float | list[float]] | None = None,
    n_points: int = 200,
    level: float = 0.95,
    unconditional: bool = False,
    V_beta: NDArray | None = None,
) -> list[MarginalEffectResult]:
    """Compute marginal (partial) effects of a variable.

    Evaluates the smooth term(s) involving *variable* over a grid while holding other variables
    fixed at their means or at values specified via *at*. This is the `gratia::smooth_estimates()` /
    `marginaleffects` equivalent.
    """
    from whittaker.model_matrix import predict_matrix

    beta = fit.coefficients

    x_var = data.get(variable)
    if x_var is None:
        raise ValueError(f"Variable {variable!r} not found in data.")

    x_grid = np.linspace(float(x_var.min()), float(x_var.max()), n_points)

    if V_beta is None:
        W = fit.weights
        if fit.prior_weights is not None and W is None:
            W = fit.prior_weights
        V_beta = _bayesian_covariance(mm.X, mm.penalties, fit.smoothing_params, fit.scale, W=W)

    z_crit = norm.ppf(1.0 - (1.0 - level) / 2.0)

    at_expanded: list[dict[str, float]] = [{}]
    if at is not None:
        combos = [{}]
        for var_name, vals in at.items():
            if isinstance(vals, (int, float)):
                vals = [vals]
            new_combos = []
            for combo in combos:
                for v in vals:
                    c = dict(combo)
                    c[var_name] = float(v)
                    new_combos.append(c)
            combos = new_combos
        at_expanded = combos

    results = []
    for at_vals in at_expanded:
        pred_data: dict[str, NDArray] = {}
        for k, v in data.items():
            if k == variable:
                pred_data[k] = x_grid
            elif k in at_vals:
                pred_data[k] = np.full(n_points, at_vals[k])
            else:
                pred_data[k] = np.full(n_points, np.mean(v))

        X_new = predict_matrix(mm, pred_data)

        for info in mm.smooths:
            if variable not in info.term.variables:
                continue

            cs, ce = info.col_start, info.col_end
            beta_j = beta[cs:ce]
            V_j = V_beta[cs:ce, cs:ce]

            label = repr(info.term)
            if info.by_level is not None:
                label = f"{label}:{info.by_level}"

            X_j = X_new[:, cs:ce]
            effect = X_j @ beta_j
            var_j = np.sum(X_j * (X_j @ V_j), axis=1)
            se_j = np.sqrt(np.maximum(var_j, 0.0))

            lower = effect - z_crit * se_j
            upper = effect + z_crit * se_j

            results.append(
                MarginalEffectResult(
                    term=label,
                    variable=variable,
                    x=x_grid.copy(),
                    effect=effect,
                    se=se_j,
                    lower=lower,
                    upper=upper,
                    level=level,
                    by_values=at_vals if at_vals else None,
                )
            )

    if not results:
        raise ValueError(f"No smooth terms found involving variable {variable!r}.")

    return results


# ---------------------------------------------------------------------------
# Pairwise comparisons / contrasts
# ---------------------------------------------------------------------------


@dataclass
class ContrastResult:
    """Result of a pairwise comparison between two conditions.

    Attributes
    ----------
    term:
        Smooth term label.
    x:
        Covariate grid.
    difference:
        Estimated difference (condition1 - condition2).
    se:
        Standard error of the difference.
    lower:
        Lower confidence bound.
    upper:
        Upper confidence bound.
    level:
        Confidence level.
    label:
        Description of the comparison.
    """

    term: str
    x: NDArray
    difference: NDArray
    se: NDArray
    lower: NDArray
    upper: NDArray
    level: float
    label: str


def pairwise_comparisons(
    fit: FitResult,
    mm: ModelMatrix,
    variable: str,
    data: dict[str, NDArray],
    pairs: list[tuple[dict[str, float], dict[str, float]]],
    *,
    n_points: int = 200,
    level: float = 0.95,
    unconditional: bool = False,
    V_beta: NDArray | None = None,
) -> list[ContrastResult]:
    """Compute pairwise contrasts between conditions.

    Each pair is `(condition1, condition2)` where each condition is a dict of covariate values. The
    contrast is `f(x | condition1) - f(x | condition2)` evaluated over a grid of the focal
    *variable*.

    This is the `emmeans`/`marginaleffects::comparisons()` equivalent for smooth terms.
    """
    from whittaker.model_matrix import predict_matrix

    beta = fit.coefficients

    x_var = data.get(variable)
    if x_var is None:
        raise ValueError(f"Variable {variable!r} not found in data.")

    x_grid = np.linspace(float(x_var.min()), float(x_var.max()), n_points)

    if V_beta is None:
        W = fit.weights
        if fit.prior_weights is not None and W is None:
            W = fit.prior_weights
        V_beta = _bayesian_covariance(mm.X, mm.penalties, fit.smoothing_params, fit.scale, W=W)

    z_crit = norm.ppf(1.0 - (1.0 - level) / 2.0)

    results = []
    for cond1, cond2 in pairs:
        data1: dict[str, NDArray] = {}
        data2: dict[str, NDArray] = {}
        for k, v in data.items():
            if k == variable:
                data1[k] = x_grid
                data2[k] = x_grid
            else:
                data1[k] = np.full(n_points, cond1.get(k, np.mean(v)))
                data2[k] = np.full(n_points, cond2.get(k, np.mean(v)))

        X1 = predict_matrix(mm, data1)
        X2 = predict_matrix(mm, data2)

        for info in mm.smooths:
            if variable not in info.term.variables:
                continue

            cs, ce = info.col_start, info.col_end
            beta_j = beta[cs:ce]
            V_j = V_beta[cs:ce, cs:ce]

            label_term = repr(info.term)
            if info.by_level is not None:
                label_term = f"{label_term}:{info.by_level}"

            dX = X1[:, cs:ce] - X2[:, cs:ce]
            diff = dX @ beta_j
            var_diff = np.sum(dX * (dX @ V_j), axis=1)
            se_diff = np.sqrt(np.maximum(var_diff, 0.0))

            lower = diff - z_crit * se_diff
            upper = diff + z_crit * se_diff

            cond1_str = ", ".join(f"{k}={v}" for k, v in sorted(cond1.items()))
            cond2_str = ", ".join(f"{k}={v}" for k, v in sorted(cond2.items()))
            comparison_label = f"({cond1_str}) - ({cond2_str})"

            results.append(
                ContrastResult(
                    term=label_term,
                    x=x_grid.copy(),
                    difference=diff,
                    se=se_diff,
                    lower=lower,
                    upper=upper,
                    level=level,
                    label=comparison_label,
                )
            )

    return results
