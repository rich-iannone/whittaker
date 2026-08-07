"""Factor-smooth interaction basis (bs="fs").

Implements per-level smooths with shared penalties, equivalent to mgcv's `bs="fs"`. Each level of
a grouping factor gets its own smooth curve for the numeric covariate, but all levels share the same
smoothing parameter(s). This is the standard approach for longitudinal / panel data where each
subject has its own trajectory.

The basis matrix is block-diagonal: one block per factor level, each block being the marginal smooth
basis multiplied by the level indicator. The penalty structure has `1 + M` penalties where `M` is
the null space dimension of the marginal basis:

1. The wiggliness penalty `I_L ⊗ S` (replicated across all `L` levels).
2. One penalty per null space component, penalizing that component's coefficients across levels
(effectively a random intercept/slope penalty).

Since every coefficient is penalized, `null_space_dimension = 0` and no identifiability constraints
are needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.cubic import CRS
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.tprs import TPRS

_MARGINAL_REGISTRY: dict[str, type[SmoothBasis]] = {
    "tp": TPRS,
    "cr": CRS,
    "ps": PSpline,
}


class FactorSmoothBasis(SmoothBasis):
    """Factor-smooth interaction basis (per-level smooth with shared penalties).

    Equivalent to mgcv's `bs="fs"` basis. Creates a separate smooth of the numeric covariate for
    each level of the grouping factor, with all levels sharing the same smoothing parameter(s).

    Parameters
    ----------
    k:
        Number of basis functions for the marginal smooth per level. The default is `10`.
    xt:
        Marginal basis type. One of `"tp"` (thin plate), `"cr"` (cubic regression), or
        `"ps"` (P-spline). The default is `"tp"`.
    m:
        Spline order for the marginal basis (passed through). Only used by `"tp"` and `"ps"`.
    """

    def __init__(self, k: int = 10, xt: str = "tp", **marginal_kwargs: Any) -> None:
        if xt not in _MARGINAL_REGISTRY:
            supported = ", ".join(sorted(_MARGINAL_REGISTRY))
            raise ValueError(f"Unknown marginal basis type xt={xt!r}. Supported: {supported}.")

        self._k_request = k
        self._xt = xt
        self._marginal_kwargs = marginal_kwargs
        self._fitted: bool = False

        self._marginal: SmoothBasis
        self._levels: NDArray
        self._level_to_idx: dict
        self._n_levels: int
        self._k_marginal: int
        self._marginal_nsd: int

    def fit(self, x_numeric: NDArray, factor: NDArray) -> FactorSmoothBasis:
        """Fit the factor-smooth basis.

        Parameters
        ----------
        x_numeric:
            Numeric covariate, shape `(n,)`.
        factor:
            Factor (grouping) variable, shape `(n,)`. Can be strings or integers.
        """
        x_num = self._as_1d_numeric(x_numeric)
        fac = self._as_1d_factor(factor)

        if len(x_num) != len(fac):
            raise ValueError(
                f"x_numeric and factor must have the same length, got {len(x_num)} and {len(fac)}."
            )

        self._levels = np.unique(fac)
        self._level_to_idx = {lev: i for i, lev in enumerate(self._levels)}
        self._n_levels = len(self._levels)

        if self._n_levels < 2:
            raise ValueError(
                f"FactorSmoothBasis requires at least 2 factor levels, got {self._n_levels}."
            )

        marginal_cls = _MARGINAL_REGISTRY[self._xt]
        kwargs: dict[str, Any] = {}
        if self._k_request != -1:
            kwargs["k"] = self._k_request
        kwargs.update(self._marginal_kwargs)
        self._marginal = marginal_cls(**kwargs)
        self._marginal.fit(x_num)

        self._k_marginal = self._marginal.n_basis
        self._marginal_nsd = self._marginal.null_space_dimension()

        self._fitted = True
        return self

    def basis_matrix(self, x_numeric: NDArray, factor: NDArray) -> NDArray:
        """Build the block-diagonal basis matrix.

        Parameters
        ----------
        x_numeric:
            Numeric covariate, shape `(n,)`.
        factor:
            Factor variable, shape `(n,)`.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, n_levels * k_marginal)`.
        """
        self._check_fitted()
        x_num = self._as_1d_numeric(x_numeric)
        fac = self._as_1d_factor(factor)

        B_marginal = self._marginal.basis_matrix(x_num)
        n = len(x_num)
        k_total = self._n_levels * self._k_marginal

        B = np.zeros((n, k_total))
        for lev, idx in self._level_to_idx.items():
            indicator = (fac == lev).astype(float)
            col_start = idx * self._k_marginal
            col_end = col_start + self._k_marginal
            B[:, col_start:col_end] = B_marginal * indicator[:, np.newaxis]

        return B

    def penalty_matrices(self) -> list[NDArray]:
        """Return the replicated penalty matrices.

        Returns `1 + M` penalties where `M` is the marginal null space dimension:

        1. Wiggliness penalty: `I_L ⊗ S_wiggle`
        2. One random-effect penalty per null space component
        """
        self._check_fitted()
        k_total = self._n_levels * self._k_marginal
        S_marginal = self._marginal.penalty_matrix()

        S_wiggle = np.zeros((k_total, k_total))
        for i in range(self._n_levels):
            cs = i * self._k_marginal
            ce = cs + self._k_marginal
            S_wiggle[cs:ce, cs:ce] = S_marginal
        S_wiggle = (S_wiggle + S_wiggle.T) * 0.5

        penalties = [S_wiggle]

        if self._marginal_nsd > 0:
            eigvals, eigvecs = np.linalg.eigh(S_marginal)
            threshold = max(eigvals.max(), 1.0) * 1e-10
            null_mask = eigvals < threshold
            null_vecs = eigvecs[:, null_mask]

            for j in range(null_vecs.shape[1]):
                v = null_vecs[:, j]
                S_null_comp = np.zeros((k_total, k_total))
                for i in range(self._n_levels):
                    cs = i * self._k_marginal
                    ce = cs + self._k_marginal
                    S_null_comp[cs:ce, cs:ce] = np.outer(v, v)
                S_null_comp = (S_null_comp + S_null_comp.T) * 0.5
                penalties.append(S_null_comp)

        return penalties

    def penalty_matrix(self) -> NDArray:
        self._check_fitted()
        return self.penalty_matrices()[0]

    def null_space_dimension(self) -> int:
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        return None

    @property
    def n_basis(self) -> int:
        if not self._fitted:
            raise RuntimeError("n_basis is not available until fit() is called.")
        return self._n_levels * self._k_marginal

    @property
    def k(self) -> int:
        if self._fitted:
            return self._k_marginal
        return self._k_request

    @property
    def levels(self) -> NDArray:
        self._check_fitted()
        return self._levels.copy()

    @property
    def n_levels(self) -> int:
        self._check_fitted()
        return self._n_levels

    @property
    def marginal_basis(self) -> SmoothBasis:
        self._check_fitted()
        return self._marginal

    @staticmethod
    def _as_1d_numeric(x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(f"Expected 1-D numeric array, got shape {x.shape}.")

    @staticmethod
    def _as_1d_factor(x: NDArray) -> NDArray:
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError(f"Expected 1-D factor array, got shape {x.shape}.")
        return x
