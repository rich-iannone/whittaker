"""Adaptive smooth basis via eigendecomposition of the penalty.

An adaptive smooth decomposes the single penalty matrix into its eigenvectors, creating one penalty
per eigenvector. Each gets its own smoothing parameter, allowing spatially varying smoothness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.tprs import TPRS

_EIG_TOL = 1e-10


class AdaptiveTPRS(TPRS):
    """Adaptive Thin Plate Regression Spline.

    Decomposes the TPRS penalty into its eigenvectors, assigning a separate smoothing parameter to
    each. This allows the amount of smoothing to vary across the covariate space.

    The number of adaptive penalty components is controlled by `n_penalties`. With `n_penalties=k-M`
    (the default), every eigenvector gets its own λ. Smaller values group eigenvectors into blocks
    for computational efficiency.

    Parameters
    ----------
    k:
        Total number of basis functions (the default is `10`).
    m:
        Spline order (the default is `2`).
    n_penalties:
        Number of adaptive penalty components (the default is `-1`, which uses all eigenvectors).
    """

    def __init__(self, k: int = 10, m: int = 2, n_penalties: int = -1) -> None:
        super().__init__(k=k, m=m)
        self._n_penalties = n_penalties

    def penalty_matrices(self) -> list[NDArray]:
        """Return decomposed penalty matrices, one per eigenvector group."""
        self._check_fitted()
        M = self._M
        r = self.k - M
        eigvals = self._eigenvalues

        n_pen = self._n_penalties if self._n_penalties > 0 else r

        if n_pen >= r:
            pens = []
            for i in range(r):
                S = np.zeros((self.k, self.k))
                S[M + i, M + i] = max(eigvals[i], _EIG_TOL)
                pens.append(S)
            return pens

        block_size = r / n_pen
        pens = []
        for b in range(n_pen):
            start = int(round(b * block_size))
            end = int(round((b + 1) * block_size))
            S = np.zeros((self.k, self.k))
            for i in range(start, min(end, r)):
                S[M + i, M + i] = max(eigvals[i], _EIG_TOL)
            pens.append(S)
        return pens

    def null_space_dimension(self) -> int:
        self._check_fitted()
        return self._M
