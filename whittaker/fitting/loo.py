"""PSIS-LOO cross-validation for Bayesian GAM fits.

Implements Pareto-Smoothed Importance Sampling Leave-One-Out cross-validation (Vehtari, Gelman &
Gabry, 2017; Vehtari et al., 2022).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import genpareto


@dataclass
class LOOResult:
    """Result of PSIS-LOO cross-validation on a fitted GAM.

    Attributes
    ----------
    elpd_loo : float
        Expected log predictive density (ELPD_LOO), summed over observations. Higher is better.
    se_elpd_loo : float
        Approximate standard error of `elpd_loo`, computed as `sqrt(n * var(pointwise))`.
    p_loo : float
        Effective number of parameters (LOO penalty). Computed as `lpd_full - elpd_loo` where
        `lpd_full` is the log-likelihood at the posterior mean. Large `p_loo` relative to the actual
        parameter count suggests model misspecification.
    pointwise : NDArray
        Per-observation LOO log predictive density values, shape `(n,)`.
    pareto_k : NDArray
        Per-observation Pareto `k` diagnostic, shape `(n,)`. Values above 0.7 indicate that the PSIS
        approximation is unreliable for that observation; values above 1.0 indicate the importance
        weights have infinite variance and LOO is invalid.
    n_bad_k : int
        Number of observations with `pareto_k > 0.7`.
    """

    elpd_loo: float
    se_elpd_loo: float
    p_loo: float
    pointwise: NDArray
    pareto_k: NDArray
    n_bad_k: int

    def __repr__(self) -> str:
        lines = [
            "LOOResult",
            f"  ELPD_LOO:    {self.elpd_loo:.2f}  (SE {self.se_elpd_loo:.2f})",
            f"  p_LOO:       {self.p_loo:.2f}",
            f"  Bad k > 0.7: {self.n_bad_k} / {len(self.pareto_k)} observations",
        ]
        return "\n".join(lines)


@dataclass
class LOOComparison:
    """Comparison of two PSIS-LOO results on the same data.

    Attributes
    ----------
    elpd_diff : float
        Difference in ELPD_LOO: `result1.elpd_loo - result2.elpd_loo`. Positive means `result1` is
        preferred; negative means `result2`.
    se_diff : float
        Standard error of `elpd_diff`, computed from the pointwise differences using
        `sqrt(n * var(pointwise1 - pointwise2))`.
    """

    elpd_diff: float
    se_diff: float

    def __repr__(self) -> str:
        direction = "model 1 preferred" if self.elpd_diff > 0 else "model 2 preferred"
        return (
            f"LOOComparison\n"
            f"  ELPD diff: {self.elpd_diff:+.2f}  (SE {self.se_diff:.2f})\n"
            f"  {direction}"
        )


# ---------------------------------------------------------------------------
# PSIS smoothing internals
# ---------------------------------------------------------------------------


def _fit_gpd_tail(z: NDArray) -> tuple[float, float]:
    """Fit a Generalized Pareto Distribution to positive exceedances `z`.

    Uses scipy MLE with `loc` fixed at 0 (pure exceedances). Returns `(k_hat, sigma_hat)`.
    """
    try:
        k, _, sigma = genpareto.fit(z, floc=0)
        return float(k), float(sigma)
    except Exception:
        return 0.0, float(np.mean(z)) if len(z) > 0 else 1.0


def _psis_smooth_one(lw: NDArray) -> tuple[NDArray, float]:
    """Apply PSIS smoothing to the log importance weights for one observation.

    Parameters
    ----------
    lw : NDArray
        Log importance weights, shape `(S,)`. These are *raw* (not yet normalized), and equal to
        `-log_lik[i, :]`.

    Returns
    -------
    lw_smooth : NDArray
        Smoothed log weights, shape `(S,)`.
    k_hat : float
        Estimated Pareto shape parameter. Values above 0.7 indicate an unreliable approximation.
    """
    S = len(lw)
    M = min(S // 5, int(3 * np.sqrt(S)))
    M = max(M, 5)

    # Sort ascending; top M are at the end.
    order = np.argsort(lw)
    sorted_lw = lw[order]

    # Numerically stabilize by subtracting the max (= sorted_lw[-1]).
    sorted_lw = sorted_lw - sorted_lw[-1]

    # Threshold: the value just below the tail.
    cutoff_idx = max(S - M - 1, 0)
    cutoff = sorted_lw[cutoff_idx]

    # Exceedances on the weight scale.
    exp_cutoff = np.exp(cutoff)
    lw_tail = sorted_lw[-M:]  # ascending, shape (M,)
    z = np.exp(lw_tail) - exp_cutoff
    z = np.maximum(z, 0.0)

    if z[-1] == 0.0:
        return lw, 0.0

    k_hat, sigma = _fit_gpd_tail(z)

    # Smoothed quantiles at (j - 0.5) / M for j = 1, ..., M (ascending).
    p_j = (np.arange(1, M + 1) - 0.5) / M
    z_smooth = genpareto.ppf(p_j, c=k_hat, loc=0, scale=sigma)
    lw_tail_smooth = np.log(np.maximum(z_smooth + exp_cutoff, 1e-300))

    sorted_lw_smooth = sorted_lw.copy()
    sorted_lw_smooth[-M:] = np.sort(lw_tail_smooth)

    # Un-sort back to the original draw order.
    lw_out = np.empty(S)
    lw_out[order] = sorted_lw_smooth
    return lw_out, k_hat


def _psis_smooth(log_lik: NDArray) -> tuple[NDArray, NDArray]:
    """Apply PSIS smoothing to a `(n, S)` array of log-likelihoods.

    Parameters
    ----------
    log_lik : NDArray
        Per-observation, per-draw log-likelihoods, shape `(n, S)`.

    Returns
    -------
    lw_smooth : NDArray
        Smoothed log importance weights, shape `(n, S)`.
    pareto_k : NDArray
        Per-observation Pareto `k` diagnostics, shape `(n,)`.
    """
    n, S = log_lik.shape
    lw_smooth = np.empty_like(log_lik)
    pareto_k = np.empty(n)

    for i in range(n):
        lw_smooth[i, :], pareto_k[i] = _psis_smooth_one(-log_lik[i, :])

    return lw_smooth, pareto_k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_loo(
    log_lik: NDArray,
    lpd_full: NDArray,
) -> LOOResult:
    """Compute PSIS-LOO from a pre-computed `(n, S)` log-likelihood matrix.

    Parameters
    ----------
    log_lik : NDArray
        Per-observation, per-draw log-likelihoods, shape `(n, S)`.
    lpd_full : NDArray
        Per-observation log-likelihood at the posterior mean, shape `(n,)`. Used to compute `p_loo`.

    Returns
    -------
    LOOResult
    """
    lw_smooth, pareto_k = _psis_smooth(log_lik)

    # elpd_i = logsumexp(lw_i + ll_i) - logsumexp(lw_i)
    elpd_pointwise = logsumexp(lw_smooth + log_lik, axis=1) - logsumexp(lw_smooth, axis=1)

    n = len(elpd_pointwise)
    elpd_loo = float(np.sum(elpd_pointwise))
    se_elpd_loo = float(np.sqrt(n * np.var(elpd_pointwise, ddof=1)))
    p_loo = float(np.sum(lpd_full) - elpd_loo)
    n_bad_k = int(np.sum(pareto_k > 0.7))

    if n_bad_k > 0:
        warnings.warn(
            f"{n_bad_k} observation(s) have Pareto k > 0.7. The PSIS-LOO approximation "
            "may be unreliable. Consider moment matching or exact LOO for these points.",
            stacklevel=3,
        )

    return LOOResult(
        elpd_loo=elpd_loo,
        se_elpd_loo=se_elpd_loo,
        p_loo=p_loo,
        pointwise=elpd_pointwise,
        pareto_k=pareto_k,
        n_bad_k=n_bad_k,
    )


def loo_compare(result1: LOOResult, result2: LOOResult) -> LOOComparison:
    """Compare two PSIS-LOO results computed on the same observations.

    The standard error of the difference uses the pointwise LOO values from both models, which gives
    a paired comparison that accounts for correlation across observations (Vehtari et al., 2017).

    Parameters
    ----------
    result1, result2 : LOOResult
        LOO results from two models fitted to the same data. Must have the same number of
        observations.

    Returns
    -------
    LOOComparison
        Contains `elpd_diff = result1.elpd_loo - result2.elpd_loo` and its standard error.
    """
    n1, n2 = len(result1.pointwise), len(result2.pointwise)
    if n1 != n2:
        raise ValueError(
            f"Cannot compare LOO results with different numbers of observations "
            f"({n1} vs {n2}). Both models must be fitted to the same data."
        )
    diff_pw = result1.pointwise - result2.pointwise
    elpd_diff = float(np.sum(diff_pw))
    se_diff = float(np.sqrt(n1 * np.var(diff_pw, ddof=1)))
    return LOOComparison(elpd_diff=elpd_diff, se_diff=se_diff)
