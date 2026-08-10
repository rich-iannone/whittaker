"""Monotone P-spline basis for shape-constrained smoothing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.pspline import PSpline, _diff_matrix


class MonotonePSpline(PSpline):
    """Shape-constrained P-spline: monotone increasing or decreasing.

    Uses a P-spline basis with an additional linear inequality constraint on the B-spline
    coefficients. Monotonicity of a B-spline curve is guaranteed when the coefficients are
    non-decreasing (increasing) or non-increasing (decreasing).

    The constraint is enforced during fitting via iterative coefficient projection (Pool Adjacent
    Violators).

    Parameters
    ----------
    k:
        Number of B-spline basis functions. The default is `20`.
    degree:
        B-spline polynomial degree. The default is `3` (cubic).
    m:
        Difference penalty order. The default is `2`.
    decreasing:
        If `True`, enforce monotone *decreasing*. The default is `False` (monotone increasing).
    """

    def __init__(
        self,
        k: int = 20,
        degree: int = 3,
        m: int = 2,
        decreasing: bool = False,
    ) -> None:
        super().__init__(k=k, degree=degree, m=m)
        self.decreasing = decreasing

    @property
    def constraint_direction(self) -> int:
        return -1 if self.decreasing else 1

    def null_space_dimension(self) -> int:
        return 0


class ConvexPSpline(PSpline):
    """Shape-constrained P-spline: convex or concave.

    Convexity of a B-spline curve is guaranteed when the second differences of the coefficients are
    non-negative.

    Parameters
    ----------
    k:
        Number of B-spline basis functions. The default is `20`.
    degree:
        B-spline polynomial degree. The default is `3` (cubic).
    m:
        Difference penalty order. The default is `2`.
    concave:
        If `True`, enforce concavity. The default is `False` (convex).
    """

    def __init__(
        self,
        k: int = 20,
        degree: int = 3,
        m: int = 2,
        concave: bool = False,
    ) -> None:
        super().__init__(k=k, degree=degree, m=m)
        self.concave = concave

    @property
    def constraint_direction(self) -> int:
        return -1 if self.concave else 1

    @property
    def constraint_order(self) -> int:
        return 2

    def null_space_dimension(self) -> int:
        return 0


def project_monotone(beta: NDArray, *, decreasing: bool = False) -> NDArray:
    """Project coefficients onto the monotone cone using PAVA."""
    if decreasing:
        return -_pava(-beta)
    return _pava(beta)


def project_convex(beta: NDArray, *, concave: bool = False) -> NDArray:
    """Project coefficients onto the convex (or concave) cone.

    Convexity requires non-negative second differences.  We project the second differences onto the
    non-negative cone, then reconstruct.
    """
    D1 = _diff_matrix(len(beta), 1)
    diffs = D1 @ beta
    if concave:
        diffs_proj = -_pava(-diffs)
    else:
        diffs_proj = _pava(diffs)
    out = np.empty_like(beta)
    out[0] = beta[0]
    out[1:] = out[0] + np.cumsum(diffs_proj)
    return out


def _pava(x: NDArray) -> NDArray:
    """Pool Adjacent Violators: isotonic regression (non-decreasing)."""
    n = len(x)
    result = x.copy()
    block_start = np.arange(n)
    block_size = np.ones(n, dtype=int)

    i = 0
    while i < n - 1:
        j = i + block_size[i]
        if j >= n:
            break
        if result[i] > result[j]:
            total = result[i] * block_size[i] + result[j] * block_size[j]
            new_size = block_size[i] + block_size[j]
            result[i] = total / new_size
            block_size[i] = new_size
            block_size[j] = 0
            while i > 0:
                prev = i - 1
                while prev >= 0 and block_size[prev] == 0:
                    prev -= 1
                if prev < 0:
                    break
                if result[prev] > result[i]:
                    total = result[prev] * block_size[prev] + result[i] * block_size[i]
                    new_size = block_size[prev] + block_size[i]
                    result[prev] = total / new_size
                    block_size[prev] = new_size
                    block_size[i] = 0
                    i = prev
                else:
                    break
        else:
            i = j

    out = np.empty(n)
    i = 0
    while i < n:
        if block_size[i] > 0:
            out[i : i + block_size[i]] = result[i]
            i += block_size[i]
        else:
            i += 1
    return out
