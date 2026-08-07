"""Cyclic (periodic) smooth bases: CyclicCRS (cc) and CyclicPSpline (cp)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.linalg import solve

from whittaker.smooths.base import SmoothBasis

# ---------------------------------------------------------------------------
# Cyclic CRS helpers
# ---------------------------------------------------------------------------


def _build_cyclic_Q(h: NDArray) -> NDArray:
    """Build the (m x m) cyclic second-difference matrix Q_c.

    For m knot spacings on a circle (m = k-1 free coefficients), the system
    R_c d = Q_c^T beta relates second derivatives d to coefficient values beta.
    """
    m = len(h)
    Q = np.zeros((m, m))
    for l in range(m):
        hp = h[(l - 1) % m]
        hc = h[l]
        Q[(l - 1) % m, l] += 1.0 / hp
        Q[l, l] += -(1.0 / hp + 1.0 / hc)
        Q[(l + 1) % m, l] += 1.0 / hc
    return Q


def _build_cyclic_R(h: NDArray) -> NDArray:
    """Build the (m x m) cyclic tridiagonal matrix R_c."""
    m = len(h)
    R = np.zeros((m, m))
    for l in range(m):
        hp = h[(l - 1) % m]
        hc = h[l]
        R[l, l] = (hp + hc) / 3.0
        R[l, (l + 1) % m] += hc / 6.0
        R[(l + 1) % m, l] += hc / 6.0
    return R


def _cyclic_crs_basis_matrix(
    x: NDArray,
    knots: NDArray,
    h: NDArray,
    A: NDArray,
    period: float,
) -> NDArray:
    """Evaluate the cyclic CRS basis at arbitrary x values.

    Maps x into the periodic range [knots[0], knots[-1]) and evaluates the cyclic cubic spline
    basis, returning an (n, k-1) design matrix.
    """
    n = len(x)
    m = len(h)  # k - 1 free coefficients

    x_mapped = knots[0] + (x - knots[0]) % period

    B = np.zeros((n, m))

    j = np.searchsorted(knots, x_mapped, side="right") - 1
    j = np.clip(j, 0, len(knots) - 2)

    u = (x_mapped - knots[j]) / h[j]

    left_idx = j
    right_idx = (j + 1) % m

    rows = np.arange(n)
    np.add.at(B, (rows, left_idx), 1.0 - u)
    np.add.at(B, (rows, right_idx), u)

    c1 = h[j] ** 2 / 6.0 * ((1.0 - u) ** 3 - (1.0 - u))
    c2 = h[j] ** 2 / 6.0 * (u**3 - u)

    B += c1[:, np.newaxis] * A[j, :]
    B += c2[:, np.newaxis] * A[(j + 1) % m, :]

    return B


# ---------------------------------------------------------------------------
# Cyclic P-spline helpers
# ---------------------------------------------------------------------------


def _periodic_bspline_knots(x_min: float, x_max: float, k: int, degree: int) -> NDArray:
    """Build a periodic knot vector for k periodic B-spline basis functions.

    Returns the augmented knot vector of length k + 2*degree + 1, constructed by extending k
    equally-spaced knots periodically in both directions.
    """
    period = x_max - x_min
    t_base = np.linspace(x_min, x_max, k, endpoint=False)

    left_ext = t_base[-degree:] - period
    right_ext = t_base[: degree + 1] + period

    return np.concatenate([left_ext, t_base, right_ext])


def _cyclic_diff_matrix(k: int, m: int) -> NDArray:
    """Circular m-th order difference matrix, shape (k, k).

    Built by repeated left-multiplication of the circular first-difference matrix D_1, where
    D_1[i, i] = -1 and D_1[i, (i+1) % k] = +1.
    """
    D1 = np.zeros((k, k))
    for i in range(k):
        D1[i, i] = -1.0
        D1[i, (i + 1) % k] = 1.0

    D = D1.copy()
    for _ in range(m - 1):
        D = D1 @ D
    return D


# ---------------------------------------------------------------------------
# CyclicCRS
# ---------------------------------------------------------------------------


class CyclicCRS(SmoothBasis):
    """Cyclic Cubic Regression Spline (periodic natural cubic spline).

    Equivalent to mgcv's `bs="cc"` basis. The spline is periodic over the range of the training
    data: f(x_min) = f(x_max), f'(x_min) = f'(x_max), and f''(x_min) = f''(x_max). Values outside
    the training range are mapped into the periodic domain via modular arithmetic.

    The cyclic constraint absorbs one degree of freedom, so k knots produce k-1 basis functions. The
    penalty null space is 1-dimensional (constant functions only as linear functions are no longer
    unpenalized under periodicity).

    Parameters
    ----------
    k:
        Number of knots. Must be >= 4 (yielding >= 3 basis functions). The default is `10`.
    """

    def __init__(self, k: int = 10) -> None:
        if k < 4:
            raise ValueError(f"k must be at least 4 for CyclicCRS, got {k}.")
        self.k = k
        self._fitted: bool = False

        self._knots: NDArray
        self._h: NDArray
        self._A: NDArray
        self._S: NDArray
        self._x_min: float
        self._x_max: float
        self._period: float

    def fit(self, x: NDArray) -> CyclicCRS:
        x1d = self._as_1d(x)
        n = len(x1d)

        if n < self.k:
            raise ValueError(
                f"Number of observations n={n} is smaller than k={self.k}. "
                f"Reduce k to at most {n - 1}."
            )

        knots = np.quantile(x1d, np.linspace(0.0, 1.0, self.k))
        if len(np.unique(knots)) < self.k:
            knots = np.linspace(float(x1d.min()), float(x1d.max()), self.k)

        self._knots = knots
        self._x_min = float(knots[0])
        self._x_max = float(knots[-1])
        self._period = self._x_max - self._x_min

        h = np.diff(knots)
        self._h = h

        Q = _build_cyclic_Q(h)
        R = _build_cyclic_R(h)

        self._A = solve(R, Q.T, assume_a="pos")

        S = Q @ self._A
        self._S = (S + S.T) * 0.5

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        self._check_fitted()
        x1d = self._as_1d(x)
        return _cyclic_crs_basis_matrix(x1d, self._knots, self._h, self._A, self._period)

    def penalty_matrix(self) -> NDArray:
        self._check_fitted()
        return self._S.copy()

    def null_space_dimension(self) -> int:
        return 1

    def identifiability_constraints(self) -> NDArray | None:
        self._check_fitted()
        B_train = self.basis_matrix(self._knots[:-1])
        return B_train.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        return self.k - 1

    @property
    def knots(self) -> NDArray:
        self._check_fitted()
        return self._knots.copy()

    @staticmethod
    def _as_1d(x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(
            "CyclicCRS is a univariate basis: pass a 1-D array or an (n, 1) "
            f"column vector, not an array with shape {x.shape}."
        )


# ---------------------------------------------------------------------------
# CyclicPSpline
# ---------------------------------------------------------------------------


class CyclicPSpline(SmoothBasis):
    """Cyclic P-Spline (periodic B-spline basis with circular difference penalty).

    Equivalent to mgcv's `bs="cp"` basis. The B-spline basis is constructed to be periodic over the
    training data range, so f(x_min) = f(x_max). Values outside the training range are mapped into
    the periodic domain.

    The circular difference penalty penalizes the m-th differences of adjacent coefficients with
    wrap-around at the boundaries. The penalty null space is 1-dimensional (constant functions
    only).

    Parameters
    ----------
    k:
        Number of basis functions. Must be >= degree + 1. The default is `10`.
    degree:
        B-spline polynomial degree. The default is `3` (cubic).
    m:
        Order of the circular difference penalty. The default is `2`.
    """

    def __init__(self, k: int = 10, degree: int = 3, m: int = 2) -> None:
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}.")
        if degree < 1:
            raise ValueError(f"degree must be at least 1, got {degree}.")
        if k < degree + 1:
            raise ValueError(
                f"k={k} is too small for degree={degree}: need k >= degree + 1 = {degree + 1}."
            )
        if m < 1:
            raise ValueError(f"Penalty order m must be at least 1, got {m}.")
        if m >= k:
            raise ValueError(f"Penalty order m={m} must be less than k={k}.")
        self.k = k
        self.degree = degree
        self.m = m
        self._fitted: bool = False

        self._t: NDArray
        self._x_min: float
        self._x_max: float
        self._period: float
        self._S: NDArray

    def fit(self, x: NDArray) -> CyclicPSpline:
        x1d = self._as_1d(x)
        n = len(x1d)

        if n < self.k:
            raise ValueError(
                f"Number of observations n={n} is smaller than k={self.k}. "
                f"Reduce k to at most {n - 1}."
            )

        self._x_min = float(x1d.min())
        self._x_max = float(x1d.max())

        if self._x_min == self._x_max:
            raise ValueError("All covariate values are identical; cannot fit a spline.")

        self._period = self._x_max - self._x_min
        self._t = _periodic_bspline_knots(self._x_min, self._x_max, self.k, self.degree)

        D = _cyclic_diff_matrix(self.k, self.m)
        S = D.T @ D
        self._S = (S + S.T) * 0.5

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        self._check_fitted()
        x1d = self._as_1d(x)

        x_mapped = self._x_min + (x1d - self._x_min) % self._period

        lo = self._t[self.degree]
        hi = self._t[-(self.degree + 1)]
        x_eval = np.clip(x_mapped, lo, hi - 1e-14 * (hi - lo + 1.0))

        B_full = BSpline.design_matrix(x_eval, self._t, self.degree).toarray()

        B = B_full[:, : self.k].copy()
        if self.degree > 0:
            B[:, : self.degree] += B_full[:, self.k :]

        return B

    def penalty_matrix(self) -> NDArray:
        self._check_fitted()
        return self._S.copy()

    def null_space_dimension(self) -> int:
        return 1

    def identifiability_constraints(self) -> NDArray | None:
        self._check_fitted()
        x_ref = np.linspace(self._x_min, self._x_max, max(self.k * 5, 100), endpoint=False)
        B_ref = self.basis_matrix(x_ref)
        return B_ref.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        return self.k

    @staticmethod
    def _as_1d(x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(
            "CyclicPSpline is a univariate basis: pass a 1-D array or an (n, 1) "
            f"column vector, not an array with shape {x.shape}."
        )
