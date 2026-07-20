"""Cubic Regression Splines (CRS)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

from whittaker.smooths.base import SmoothBasis

# ---------------------------------------------------------------------------
# Low-level construction helpers
# ---------------------------------------------------------------------------


def _build_Q(h: NDArray) -> NDArray:
    """Build the k x (k-2) second-difference matrix Q.

    For knot spacings h = [h_0, h_1, ..., h_{k-2}], column l (0-indexed) of Q has three non-zero
    entries:

        Q[l,   l] =  1 / h[l]
        Q[l+1, l] = -(1/h[l] + 1/h[l+1])
        Q[l+2, l] =  1 / h[l+1]

    Parameters
    ----------
    h:
        Knot spacings, shape `(k-1,)`. All entries must be positive.

    Returns
    -------
    NDArray
        Shape `(k, k-2)`.
    """
    k = len(h) + 1
    Q = np.zeros((k, k - 2))
    for col in range(k - 2):
        Q[col, col] = 1.0 / h[col]
        Q[col + 1, col] = -(1.0 / h[col] + 1.0 / h[col + 1])
        Q[col + 2, col] = 1.0 / h[col + 1]
    return Q


def _build_R(h: NDArray) -> NDArray:
    """Build the (k-2) x (k-2) symmetric tridiagonal matrix R.

    Diagonal and super/sub-diagonal entries:

        R[l, l]     = (h[l] + h[l+1]) / 3
        R[l, l+1]   = h[l+1] / 6
        R[l+1, l]   = h[l+1] / 6

    Parameters
    ----------
    h:
        Knot spacings, shape `(k-1,)`.

    Returns
    -------
    NDArray
        Shape `(k-2, k-2)`.
    """
    m = len(h) - 1  # = k - 2
    R = np.zeros((m, m))
    for row in range(m):
        R[row, row] = (h[row] + h[row + 1]) / 3.0
        if row < m - 1:
            R[row, row + 1] = h[row + 1] / 6.0
            R[row + 1, row] = h[row + 1] / 6.0
    return R


def _second_deriv_operator(Q: NDArray, R: NDArray) -> NDArray:
    """Return the k x k second-derivative operator A.

    The interior second derivatives of the spline satisfy:

    d[1:-1] = R⁻¹ Q' β

    Padding with zeros at the endpoints (natural BCs) gives the full operator A (shape k x k) such
    that d = A β, where d is the k-vector of second derivatives at all knots.

    Parameters
    ----------
    Q:
        Shape `(k, k-2)`.
    R:
        Shape `(k-2, k-2)`, symmetric positive definite.

    Returns
    -------
    NDArray
        Shape `(k, k)`.
    """
    k = Q.shape[0]
    # F = R⁻¹ Q'  →  (k-2, k)
    F = solve(R, Q.T, assume_a="pos")
    A = np.zeros((k, k))
    A[1:-1, :] = F
    return A


def _crs_basis_matrix(
    x: NDArray,
    knots: NDArray,
    h: NDArray,
    A: NDArray,
) -> NDArray:
    """Evaluate the CRS basis at arbitrary `x` values.

    Computes the n x k design matrix B where B[i, :] is the row corresponding to x[i]. Outside the
    knot range the spline is extended linearly (natural boundary conditions).

    Parameters
    ----------
    x:
        1-D evaluation points, shape `(n,)`.
    knots:
        Knot locations, shape `(k,)`, strictly increasing.
    h:
        Knot spacings `diff(knots)`, shape `(k-1,)`.
    A:
        Second-derivative operator, shape `(k, k)`.

    Returns
    -------
    NDArray
        Shape `(n, k)`.
    """
    n = len(x)
    k = len(knots)
    B = np.zeros((n, k))

    # Locate each x in the knot sequence.
    # j = index of the left-knot of the containing interval.
    j_raw = np.searchsorted(knots, x, side="right") - 1

    left_ext = j_raw < 0
    right_ext = j_raw >= k - 1
    interior = ~(left_ext | right_ext)

    # ------------------------------------------------------------------
    # Interior evaluation
    # ------------------------------------------------------------------
    if np.any(interior):
        rows = np.where(interior)[0]
        j = j_raw[rows]  # interval index in [0, k-2]
        xi = x[rows]
        u = (xi - knots[j]) / h[j]  # ∈ [0, 1)

        # Linear part.
        B[rows, j] += 1.0 - u
        B[rows, j + 1] += u

        # Cubic correction (via second derivatives).
        c1 = h[j] ** 2 / 6.0 * ((1.0 - u) ** 3 - (1.0 - u))
        c2 = h[j] ** 2 / 6.0 * (u**3 - u)
        B[rows, :] += c1[:, np.newaxis] * A[j, :]
        B[rows, :] += c2[:, np.newaxis] * A[j + 1, :]

    # ------------------------------------------------------------------
    # Left linear extrapolation  (x < knots[0])
    # ------------------------------------------------------------------
    if np.any(left_ext):
        rows = np.where(left_ext)[0]
        delta = x[rows] - knots[0]

        # f'(t₀) in terms of β: (e₁ - e₀)/h₀ - h₀/6 · A[1,:]
        # (f''(t₀) = 0 by natural BC; the A[1,:] term is f''(t₁).)
        e0 = np.zeros(k)
        e0[0] = 1.0
        e1 = np.zeros(k)
        e1[1] = 1.0
        deriv0 = (e1 - e0) / h[0] - h[0] / 6.0 * A[1, :]

        B[rows, :] = e0 + delta[:, np.newaxis] * deriv0

    # ------------------------------------------------------------------
    # Right linear extrapolation  (x ≥ knots[-1])
    # ------------------------------------------------------------------
    if np.any(right_ext):
        rows = np.where(right_ext)[0]
        delta = x[rows] - knots[-1]

        # f'(t_{k-1}) in terms of β: (e_{k-1} - e_{k-2})/h_{-1} + h_{-1}/6 · A[-2,:]
        ek1 = np.zeros(k)
        ek1[-1] = 1.0
        ek2 = np.zeros(k)
        ek2[-2] = 1.0
        derivk = (ek1 - ek2) / h[-1] + h[-1] / 6.0 * A[-2, :]

        B[rows, :] = ek1 + delta[:, np.newaxis] * derivk

    return B


# ---------------------------------------------------------------------------
# Public CRS class
# ---------------------------------------------------------------------------


class CRS(SmoothBasis):
    """Cubic Regression Splines (natural cubic splines with quantile knots).

    Equivalent to mgcv's `bs="cr"` basis. The basis is parameterized by the spline values at k knots
    placed at evenly-spaced quantiles of the training data. Only univariate covariates are
    supported.

    The first two columns of the basis matrix correspond to the linear null space of the penalty
    ({1, x}); the remaining k - 2 columns are penalized.

    Note that unlike TPRS, the null-space columns are **not** stored as the leading columns. The
    full k-column basis is used directly, with the penalty having a 2-dimensional null space.

    Parameters
    ----------
    k:
        Number of basis functions (= number of knots). Must be ≥ 3. The default is `10`.

    Examples
    --------
    >>> import numpy as np
    >>> from whittaker.smooths import CRS
    >>> x = np.linspace(0, 1, 100)
    >>> basis = CRS(k=10).fit(x)
    >>> B = basis.basis_matrix(x)
    >>> B.shape
    (100, 10)
    >>> S = basis.penalty_matrix()
    >>> S.shape
    (10, 10)
    """

    def __init__(self, k: int = 10) -> None:
        if k < 3:
            raise ValueError(f"k must be at least 3 for CRS, got {k}.")
        self.k = k
        self._fitted: bool = False

        # Set during fit():
        self._knots: NDArray
        self._h: NDArray
        self._A: NDArray  # (k, k) second-derivative operator
        self._S: NDArray  # (k, k) penalty matrix

    # ------------------------------------------------------------------
    # SmoothBasis interface
    # ------------------------------------------------------------------

    def fit(self, x: NDArray) -> CRS:
        """Fit the CRS to training data `x`.

        Places k knots at evenly-spaced quantiles of `x` and pre-computes the second-derivative
        operator and penalty matrix.

        Parameters
        ----------
        x:
            1-D training covariate, shape `(n,)` or `(n, 1)`.

        Returns
        -------
        CRS
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If `n < k`, or if `x` is not 1-D/`(n, 1)`.
        """
        x1d = self._as_1d(x)
        n = len(x1d)

        if n < self.k:
            raise ValueError(
                f"Number of observations n={n} is smaller than k={self.k}. "
                f"Reduce k to at most {n - 1}."
            )

        # Knots at evenly-spaced quantiles.
        knots = np.quantile(x1d, np.linspace(0.0, 1.0, self.k))

        # If there are ties (e.g. many equal values) fall back to linspace.
        if len(np.unique(knots)) < self.k:
            knots = np.linspace(float(x1d.min()), float(x1d.max()), self.k)

        self._knots = knots
        h = np.diff(knots)
        self._h = h

        Q = _build_Q(h)  # (k, k-2)
        R = _build_R(h)  # (k-2, k-2)

        self._A = _second_deriv_operator(Q, R)  # (k, k)

        S = Q @ self._A[1:-1, :]  # Q (k,k-2) @ R⁻¹Q' (k-2,k) = (k,k)
        self._S = (S + S.T) * 0.5  # symmetrize against floating-point drift

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the CRS basis at `x`.

        Parameters
        ----------
        x:
            1-D evaluation points, shape `(n,)` or `(n, 1)`.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)``.
        """
        self._check_fitted()
        x1d = self._as_1d(x)
        return _crs_basis_matrix(x1d, self._knots, self._h, self._A)

    def penalty_matrix(self) -> NDArray:
        """Return the k × k penalty matrix `S = Q R⁻¹ Q'`.

        `S` is positive semi-definite with rank `k - 2`. Its null space is spanned by constant and
        linear functions of the knots.

        Returns
        -------
        NDArray
            Shape `(k, k)``.
        """
        self._check_fitted()
        return self._S.copy()

    def null_space_dimension(self) -> int:
        """Return 2: the constant and linear functions are unpenalized."""
        return 2

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Returns a `(1, k)` matrix whose product with the coefficient vector is zero when the smooth
        has mean zero over the training data.
        """
        self._check_fitted()
        B_train = self.basis_matrix(self._knots)
        return B_train.mean(axis=0, keepdims=True)  # (1, k)

    @property
    def n_basis(self) -> int:
        """Total number of basis functions k."""
        return self.k

    @property
    def is_fitted(self) -> bool:
        """`True` after `fit()` has been called."""
        return self._fitted

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def knots(self) -> NDArray:
        """Knot locations set during `fit()`."""
        self._check_fitted()
        return self._knots.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_1d(x: NDArray) -> NDArray:
        """Coerce `x` to a 1-D float array.

        Accepts shape `(n,)` or `(n, 1)`. Raises `ValueError` for anything else (CRS is univariate).
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(
            "CRS is a univariate basis: pass a 1-D array or an (n, 1) "
            f"column vector, not an array with shape {x.shape}."
        )
