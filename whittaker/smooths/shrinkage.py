"""Shrinkage smooth bases: ShrinkageTPRS (ts) and ShrinkageCRS (cs).

These bases use the double-penalty approach (Marra & Wood, 2011) to allow smooths to be penalized to
zero entirely. Each shrinkage basis produces two penalty matrices: the original wiggliness penalty
and a second penalty on the null space. With separate smoothing parameters for each, the fitting
machinery can shrink a smooth out of the model altogether, enabling automatic variable selection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.cubic import CRS
from whittaker.smooths.tprs import TPRS

_EIG_TOL = 1e-10


def _null_space_penalty(S: NDArray) -> NDArray:
    """Construct a penalty matrix that penalizes the null space of S.

    Eigendecomposes S, identifies eigenvectors with zero eigenvalues, and returns their outer
    product as a projection penalty.
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    threshold = _EIG_TOL * max(eigvals.max(), 1.0)
    null_mask = eigvals < threshold
    U_null = eigvecs[:, null_mask]
    S_null = U_null @ U_null.T
    return (S_null + S_null.T) * 0.5


class ShrinkageTPRS(TPRS):
    """Shrinkage Thin Plate Regression Spline.

    Equivalent to mgcv's `bs="ts"` basis. Identical basis functions to TPRS (`bs="tp"`), but with an
    extra penalty on the polynomial null space so that the entire smooth can be shrunk to zero. This
    enables automatic variable selection via GCV or REML.

    The two penalties are:

    1. The original TPRS wiggliness penalty (zero in the null-space block).
    2. A projection penalty on the null space (identity in the null-space block).

    Each gets its own smoothing parameter during fitting.

    Parameters
    ----------
    k:
        Total number of basis functions. The default is `10`.
    m:
        Spline order. Must satisfy `2m > d`. The default is `2`.
    """

    def penalty_matrices(self) -> list[NDArray]:
        self._check_fitted()
        S_wiggle = super().penalty_matrix()

        k = self.k
        M = self._M
        S_null = np.zeros((k, k))
        S_null[:M, :M] = np.eye(M)

        return [S_wiggle, S_null]

    def null_space_dimension(self) -> int:
        return 0


class ShrinkageCRS(CRS):
    """Shrinkage Cubic Regression Spline.

    Equivalent to mgcv's `bs="cs"` basis. Identical basis functions to CRS (`bs="cr"`), but with an
    extra penalty on the 2-dimensional null space (constant + linear) so that the smooth can be
    shrunk to zero entirely.

    The two penalties are:

    1. The original CRS wiggliness penalty `S = Q R^{-1} Q'`.
    2. A projection penalty onto the null space of S.

    Each gets its own smoothing parameter during fitting.

    Parameters
    ----------
    k:
        Number of basis functions (= number of knots). Must be >= 3. The default is `10`.
    """

    def penalty_matrices(self) -> list[NDArray]:
        self._check_fitted()
        S_wiggle = super().penalty_matrix()
        S_null = _null_space_penalty(S_wiggle)
        return [S_wiggle, S_null]

    def null_space_dimension(self) -> int:
        return 0
