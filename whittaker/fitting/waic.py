"""Widely Applicable Information Criterion (WAIC) for Bayesian GAM fits.

Implements WAIC (Watanabe, 2010; Gelman, Hwang & Vehtari, 2014) using the pointwise
log-likelihood matrix from posterior draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


@dataclass
class WAICResult:
    """Result of WAIC computation on a fitted GAM.

    Attributes
    ----------
    elpd_waic : float
        Expected log pointwise predictive density, summed over observations. Higher is better.
    se_elpd_waic : float
        Approximate standard error of `elpd_waic`, computed as `sqrt(n * var(pointwise))`.
    p_waic : float
        Effective number of parameters (WAIC penalty), computed as the sum of the per-observation
        variance of the log-likelihood across posterior draws.
    waic : float
        The WAIC value on the deviance scale: `-2 * elpd_waic`. Lower is better.
    pointwise : NDArray
        Per-observation ELPD contributions, shape `(n,)`.
    """

    elpd_waic: float
    se_elpd_waic: float
    p_waic: float
    waic: float
    pointwise: NDArray

    def __repr__(self) -> str:
        lines = [
            "WAICResult",
            f"  ELPD_WAIC: {self.elpd_waic:.2f}  (SE {self.se_elpd_waic:.2f})",
            f"  p_WAIC:    {self.p_waic:.2f}",
            f"  WAIC:      {self.waic:.2f}",
        ]
        return "\n".join(lines)


@dataclass
class WAICComparison:
    """Comparison of two WAIC results on the same data.

    Attributes
    ----------
    elpd_diff : float
        Difference in ELPD_WAIC: `result1.elpd_waic - result2.elpd_waic`. Positive means `result1`
        is preferred while negative means `result2`.
    se_diff : float
        Standard error of `elpd_diff`, computed from the pointwise differences using
        `sqrt(n * var(pointwise1 - pointwise2))`.
    """

    elpd_diff: float
    se_diff: float

    def __repr__(self) -> str:
        direction = "model 1 preferred" if self.elpd_diff > 0 else "model 2 preferred"
        return (
            f"WAICComparison\n"
            f"  ELPD diff: {self.elpd_diff:+.2f}  (SE {self.se_diff:.2f})\n"
            f"  {direction}"
        )


def compute_waic(log_lik: NDArray) -> WAICResult:
    """Compute WAIC from a pre-computed `(n, S)` log-likelihood matrix.

    Parameters
    ----------
    log_lik : NDArray
        Per-observation, per-draw log-likelihoods, shape `(n, S)`.

    Returns
    -------
    WAICResult
    """
    n, S = log_lik.shape

    # lppd_i = log(mean_s(exp(log_lik[i, s]))) = logsumexp(log_lik[i,:]) - log(S)
    lppd_pointwise = logsumexp(log_lik, axis=1) - np.log(S)

    # p_waic_i = var_s(log_lik[i, s])  (using ddof=1 per Gelman et al. 2014)
    p_waic_pointwise = np.var(log_lik, axis=1, ddof=1)

    pointwise = lppd_pointwise - p_waic_pointwise

    elpd_waic = float(np.sum(pointwise))
    p_waic = float(np.sum(p_waic_pointwise))
    se_elpd_waic = float(np.sqrt(n * np.var(pointwise, ddof=1)))
    waic = -2.0 * elpd_waic

    return WAICResult(
        elpd_waic=elpd_waic,
        se_elpd_waic=se_elpd_waic,
        p_waic=p_waic,
        waic=waic,
        pointwise=pointwise,
    )


def waic_compare(result1: WAICResult, result2: WAICResult) -> WAICComparison:
    """Compare two WAIC results computed on the same observations.

    The standard error of the difference uses the pointwise ELPD values from both models, giving
    a paired comparison that accounts for correlation across observations.

    Parameters
    ----------
    result1, result2 : WAICResult
        WAIC results from two models fitted to the same data. Must have the same number of
        observations.

    Returns
    -------
    WAICComparison
    """
    n1, n2 = len(result1.pointwise), len(result2.pointwise)
    if n1 != n2:
        raise ValueError(
            f"Cannot compare WAIC results with different numbers of observations "
            f"({n1} vs {n2}). Both models must be fitted to the same data."
        )
    diff_pw = result1.pointwise - result2.pointwise
    elpd_diff = float(np.sum(diff_pw))
    se_diff = float(np.sqrt(n1 * np.var(diff_pw, ddof=1)))
    return WAICComparison(elpd_diff=elpd_diff, se_diff=se_diff)
