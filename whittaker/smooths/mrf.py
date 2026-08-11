"""Markov random field smooth basis (bs="mrf").

Implements a spatial smooth for areal/lattice data where the basis is an indicator per region and
the penalty is the graph Laplacian of the neighborhood structure. The quadratic form
beta' L beta = sum_{(i,j) neighbors} (beta_i - beta_j)^2 penalizes differences between neighboring
regions, inducing spatial smoothness.

The user supplies a neighborhood structure as either a dict mapping region labels to lists of
neighbor labels, or a symmetric adjacency matrix (ndarray).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis


class MRFBasis(SmoothBasis):
    """Markov random field basis for areal spatial data.

    Equivalent to mgcv's `bs="mrf"`. Each unique region gets one basis function (indicator column).
    The penalty matrix is the graph Laplacian L = D - A where A is the adjacency matrix and
    D = diag(row_sums(A)).

    Parameters
    ----------
    k:
        Maximum number of regions to retain. If `-1` (default), all observed levels are kept.
    neighborhood:
        The neighborhood structure. Either a dict mapping region labels to lists of neighbor labels,
        or a square symmetric adjacency matrix (ndarray). Required.
    """

    def __init__(self, k: int = -1, neighborhood: dict | NDArray | None = None) -> None:
        self._k_request = k
        self._neighborhood = neighborhood
        self._fitted: bool = False
        self._levels: NDArray
        self._level_to_idx: dict
        self._adjacency: NDArray
        self._laplacian: NDArray

    def fit(self, x: NDArray) -> MRFBasis:
        x1d = self._validate(x)
        levels = np.unique(x1d)

        if self._k_request != -1:
            if self._k_request < 2:
                raise ValueError(f"k must be at least 2 for MRFBasis, got {self._k_request}.")
            if self._k_request < len(levels):
                levels = levels[: self._k_request]

        if len(levels) < 2:
            raise ValueError(
                "MRFBasis requires at least 2 unique levels in the grouping variable, "
                f"got {len(levels)}."
            )

        if self._neighborhood is None:
            raise ValueError(
                "MRFBasis requires a neighborhood structure. Pass neighborhood= as a dict "
                "mapping region labels to lists of neighbor labels, or as a symmetric "
                "adjacency matrix."
            )

        self._levels = levels
        self._level_to_idx = {lev: i for i, lev in enumerate(levels)}
        k = len(levels)

        if isinstance(self._neighborhood, np.ndarray):
            A = np.asarray(self._neighborhood, dtype=float)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"Adjacency matrix must be square, got shape {A.shape}.")
            if A.shape[0] != k:
                raise ValueError(
                    f"Adjacency matrix has {A.shape[0]} rows but data has {k} unique levels."
                )
            A = (A + A.T) / 2.0
            np.fill_diagonal(A, 0.0)
        elif isinstance(self._neighborhood, dict):
            A = np.zeros((k, k))
            for region, neighbors in self._neighborhood.items():
                if region not in self._level_to_idx:
                    continue
                i = self._level_to_idx[region]
                for nb in neighbors:
                    if nb not in self._level_to_idx:
                        continue
                    j = self._level_to_idx[nb]
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        else:
            raise TypeError(
                f"neighborhood must be a dict or ndarray, got {type(self._neighborhood).__name__}."
            )

        self._adjacency = A
        D = np.diag(A.sum(axis=1))
        L = D - A
        self._laplacian = (L + L.T) / 2.0

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        self._check_fitted()
        x1d = self._validate(x)
        n = len(x1d)
        k = len(self._levels)
        B = np.zeros((n, k))
        for i, val in enumerate(x1d):
            idx = self._level_to_idx.get(val)
            if idx is not None:
                B[i, idx] = 1.0
        return B

    def penalty_matrix(self) -> NDArray:
        self._check_fitted()
        return self._laplacian.copy()

    def null_space_dimension(self) -> int:
        self._check_fitted()
        eigvals = np.linalg.eigvalsh(self._laplacian)
        threshold = 1e-10 * max(eigvals.max(), 1.0)
        return int(np.sum(eigvals < threshold))

    def identifiability_constraints(self) -> NDArray | None:
        self._check_fitted()
        k = len(self._levels)
        return np.ones((1, k)) / k

    @property
    def n_basis(self) -> int:
        if not self._fitted:
            raise RuntimeError("n_basis is not available until fit() is called.")
        return len(self._levels)

    @property
    def k(self) -> int:
        if self._fitted:
            return len(self._levels)
        return self._k_request

    @property
    def levels(self) -> NDArray:
        self._check_fitted()
        return self._levels.copy()

    @staticmethod
    def _validate(x: NDArray) -> NDArray:
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError(
                f"MRFBasis requires a 1-D grouping variable, got array with shape {x.shape}."
            )
        return x
