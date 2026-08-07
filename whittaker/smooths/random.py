"""Random effect smooth basis (bs="re").

Implements random intercepts as a GAM smooth term. Each level of a grouping factor gets its own
basis function (one-hot encoded), penalized by an identity matrix. This is equivalent to a random
intercept in a mixed model: the penalty λI shrinks all group deviations toward zero, with λ selected
by GCV or REML.

Since the penalty is a full-rank identity, the null space dimension is 0 — every coefficient is
penalized, so the entire random effect can be shrunk to zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis


class RandomEffectBasis(SmoothBasis):
    """Random effect basis (one-hot encoding with identity penalty).

    Equivalent to mgcv's `bs="re"` basis. Each unique level of the grouping variable becomes one
    basis function. The penalty matrix is the identity, so GCV/REML selects a single smoothing
    parameter that controls how much group-level variation is retained vs. shrunk toward the
    population mean.

    The covariate should be a 1-D array of group labels (strings, integers, or any hashable type).

    Parameters
    ----------
    k:
        Maximum number of levels to retain. If `-1` (the default), all observed levels are kept.
    """

    def __init__(self, k: int = -1) -> None:
        self._k_request = k
        self._fitted: bool = False
        self._levels: NDArray
        self._level_to_idx: dict

    def fit(self, x: NDArray) -> RandomEffectBasis:
        x1d = self._validate(x)
        levels = np.unique(x1d)

        if self._k_request != -1:
            if self._k_request < 2:
                raise ValueError(
                    f"k must be at least 2 for RandomEffectBasis, got {self._k_request}."
                )
            if self._k_request < len(levels):
                levels = levels[: self._k_request]

        if len(levels) < 2:
            raise ValueError(
                "RandomEffectBasis requires at least 2 unique levels in the grouping variable, "
                f"got {len(levels)}."
            )

        self._levels = levels
        self._level_to_idx = {lev: i for i, lev in enumerate(levels)}
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
        return np.eye(len(self._levels))

    def null_space_dimension(self) -> int:
        return 0

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
                "RandomEffectBasis requires a 1-D grouping variable, "
                f"got array with shape {x.shape}."
            )
        return x
