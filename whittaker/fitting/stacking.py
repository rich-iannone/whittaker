"""Stacking weights for Bayesian model averaging.

Implements the stacking of predictive distributions to find optimal combination weights for multiple
models. Given pointwise log predictive density estimates from LOO or WAIC, the stacking weights
maximize the combined leave-one-out predictive density of the weighted mixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp

from whittaker.fitting.loo import LOOResult
from whittaker.fitting.waic import WAICResult


@dataclass
class StackingResult:
    """Result of stacking weight optimization.

    Attributes
    ----------
    weights : NDArray
        Optimal stacking weights, shape `(K,)`, summing to 1. Each weight is the contribution of the
        corresponding model to the predictive mixture.
    elpd_stacking : float
        Combined ELPD of the stacking mixture, summed over observations.
    se_elpd_stacking : float
        Approximate standard error of `elpd_stacking`.
    n_models : int
        Number of models in the comparison.
    n_obs : int
        Number of observations.
    method : str
        Whether the pointwise ELPD values came from `"loo"` or `"waic"`.
    """

    weights: NDArray
    elpd_stacking: float
    se_elpd_stacking: float
    n_models: int
    n_obs: int
    method: str

    def __repr__(self) -> str:
        lines = ["StackingResult"]
        lines.append(f"  Method:         {self.method.upper()}")
        lines.append(f"  Models:         {self.n_models}")
        lines.append(
            f"  ELPD (stacking): {self.elpd_stacking:.2f}  (SE {self.se_elpd_stacking:.2f})"
        )
        lines.append("  Weights:")
        for i, w in enumerate(self.weights):
            bar = "#" * int(round(w * 30))
            lines.append(f"    Model {i + 1}: {w:.3f}  {bar}")
        return "\n".join(lines)


def _validate_pointwise(
    results: list[LOOResult] | list[WAICResult],
) -> tuple[NDArray, str]:
    """Extract and validate the pointwise ELPD matrix from a list of results.

    Returns
    -------
    lpd_matrix : NDArray
        Shape `(n, K)` matrix of pointwise log predictive densities.
    method : str
        `"loo"` or `"waic"`.
    """
    if len(results) < 2:
        raise ValueError("Stacking requires at least 2 models.")

    if isinstance(results[0], LOOResult):
        method = "loo"
    elif isinstance(results[0], WAICResult):
        method = "waic"
    else:
        raise TypeError(f"Expected LOOResult or WAICResult, got {type(results[0]).__name__}.")

    for r in results:
        if not isinstance(r, type(results[0])):
            raise TypeError("All results must be the same type (all LOOResult or all WAICResult).")

    n = len(results[0].pointwise)
    for i, r in enumerate(results):
        if len(r.pointwise) != n:
            raise ValueError(
                f"All results must have the same number of observations. "
                f"Model 1 has {n}, model {i + 1} has {len(r.pointwise)}."
            )

    lpd_matrix = np.column_stack([r.pointwise for r in results])
    return lpd_matrix, method


def _stacking_weights(lpd_matrix: NDArray) -> NDArray:
    """Compute stacking weights by maximizing the combined log predictive density.

    Solves: maximize  sum_i log( sum_k w_k * exp(lpd[i, k]) )
            subject to  w_k >= 0,  sum_k w_k = 1

    Parameters
    ----------
    lpd_matrix : NDArray
        Shape `(n, K)` matrix of pointwise log predictive densities.

    Returns
    -------
    NDArray
        Optimal weights, shape `(K,)`.
    """
    n, K = lpd_matrix.shape

    if K == 1:
        return np.array([1.0])

    def neg_elpd(log_w: NDArray) -> float:
        log_w_norm = log_w - logsumexp(log_w)
        combined = logsumexp(lpd_matrix + log_w_norm[np.newaxis, :], axis=1)
        return -float(np.sum(combined))

    def neg_elpd_grad(log_w: NDArray) -> NDArray:
        log_w_norm = log_w - logsumexp(log_w)
        w = np.exp(log_w_norm)
        combined = logsumexp(lpd_matrix + log_w_norm[np.newaxis, :], axis=1)
        # d/d(log_w_k) = sum_i [ w_k * exp(lpd[i,k]) / sum_j w_j * exp(lpd[i,j]) ] - w_k
        #              = sum_i softmax_k(lpd[i,:] + log_w) - w_k
        log_softmax = (lpd_matrix + log_w_norm[np.newaxis, :]) - combined[:, np.newaxis]
        grad = np.sum(np.exp(log_softmax), axis=0) - w * n
        return -grad

    log_w0 = np.zeros(K)
    result = minimize(
        neg_elpd,
        log_w0,
        jac=neg_elpd_grad,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    log_w = result.x
    weights = np.exp(log_w - logsumexp(log_w))
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return weights


def stacking(
    *results: LOOResult | WAICResult,
) -> StackingResult:
    """Compute stacking weights for model averaging.

    Given LOO or WAIC results from multiple models fitted to the same data, finds the optimal
    combination weights that maximize the combined leave-one-out predictive density of the weighted
    mixture.

    Unlike pairwise `loo_compare()` or `waic_compare()`, stacking handles any number of models
    simultaneously and produces a single set of weights suitable for prediction averaging.

    Parameters
    ----------
    *results : LOOResult or WAICResult
        Two or more LOO or WAIC results. All must be the same type and computed on the same data
        (same number of observations).

    Returns
    -------
    StackingResult
        Contains the optimal weights, combined ELPD, and a display-friendly summary.

    Examples
    --------
    ```python
    loo1 = model1.loo()
    loo2 = model2.loo()
    loo3 = model3.loo()
    result = stacking(loo1, loo2, loo3)
    print(result.weights)  # e.g., array([0.62, 0.35, 0.03])
    ```
    """
    results_list = list(results)
    lpd_matrix, method = _validate_pointwise(results_list)
    n, K = lpd_matrix.shape

    weights = _stacking_weights(lpd_matrix)

    log_w = np.log(np.maximum(weights, 1e-300))
    pointwise_stacking = logsumexp(lpd_matrix + log_w[np.newaxis, :], axis=1)
    elpd_stacking = float(np.sum(pointwise_stacking))
    se_elpd_stacking = float(np.sqrt(n * np.var(pointwise_stacking, ddof=1)))

    return StackingResult(
        weights=weights,
        elpd_stacking=elpd_stacking,
        se_elpd_stacking=se_elpd_stacking,
        n_models=K,
        n_obs=n,
        method=method,
    )
