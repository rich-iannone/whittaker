"""P-Splines (B-spline basis with difference penalty)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from whittaker.smooths.base import SmoothBasis

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _bspline_knots(x_min: float, x_max: float, k: int, degree: int) -> NDArray:
    """Build the augmented knot vector for k B-spline basis functions.

    Uses `degree + 1` repeated boundary knots at each end (clamped B-spline) with `k - degree - 1`
    equally-spaced interior knots. The rightmost knot is nudged by a small epsilon so that x = x_max
    falls inside the half-open last interval `[t[-2], t[-1])`.

    Parameters
    ----------
    x_min, x_max:
        Data range endpoints.
    k:
        Number of basis functions.
    degree:
        B-spline polynomial degree.

    Returns
    -------
    NDArray
        Knot vector of length `k + degree + 1`.
    """
    n_interior = k - degree - 1
    if n_interior < 0:
        raise ValueError(
            f"k={k} is too small for degree={degree}: need k >= degree + 1 = {degree + 1}."
        )

    if n_interior == 0:
        interior: NDArray = np.empty(0)
    else:
        interior = np.linspace(x_min, x_max, n_interior + 2)[1:-1]

    t = np.concatenate([np.repeat(x_min, degree + 1), interior, np.repeat(x_max, degree + 1)])
    # Nudge the last knot right so x = x_max is inside the support.
    t[-1] += abs(t[-1]) * 1e-10 + 1e-14
    return t


def _diff_matrix(k: int, m: int) -> NDArray:
    """Return the (k-m) x k matrix of m-th order finite differences.

    Built by applying `np.diff` m times to the k x k identity matrix.

    Parameters
    ----------
    k:
        Number of columns (= number of B-spline coefficients).
    m:
        Difference order.  Must satisfy `0 < m < k`.

    Returns
    -------
    NDArray
        Shape `(k-m, k)`.
    """
    if m >= k:
        raise ValueError(f"Difference order m={m} must be less than k={k}.")
    return np.diff(np.eye(k), n=m, axis=0)


def _bspline_design(x: NDArray, t: NDArray, degree: int) -> NDArray:
    """Evaluate the B-spline design matrix at `x`.

    Parameters
    ----------
    x:
        1-D evaluation points, shape `(n,)`.
    t:
        Augmented knot vector, length `k + degree + 1`.
    degree:
        B-spline polynomial degree.

    Returns
    -------
    NDArray
        Dense design matrix of shape `(n, k)` where `k = len(t) - degree - 1`.
    """
    return BSpline.design_matrix(x, t, degree).toarray()


# ---------------------------------------------------------------------------
# Public PSpline class
# ---------------------------------------------------------------------------


class PSpline(SmoothBasis):
    """P-Spline: B-spline basis with m-th order difference penalty.

    Equivalent to mgcv's `bs="ps"` basis. The k basis functions are B-splines of degree `degree`
    with equally-spaced knots over the training data range. The penalty penalizes the m-th finite
    differences of adjacent B-spline coefficients.

    P-splines are:

    - **Cheap to construct**: no eigendecomposition or system solve.
    - **Data-location agnostic**: knots are equidistant, not at data quantiles.
    - **Well-suited to large k**: the penalty is banded (sparse).
    - **Naturally extrapolating**: B-splines extrapolate smoothly beyond the training range using
    the boundary B-spline functions.

    Parameters
    ----------
    k:
        Number of B-spline basis functions. Must satisfy `k >= degree + 1`. The default is `10`.
    degree:
        Polynomial degree of each B-spline piece. The default is `3` (cubic).
    m:
        Order of the difference penalty. The penalty penalizes the m-th differences of adjacent
        coefficients. The default is `2` (second differences: penalizes curvature). Must satisfy
        `0 < m < k`.

    Examples
    --------
    >>> import numpy as np
    >>> from whittaker.smooths import PSpline
    >>> x = np.linspace(0, 1, 100)
    >>> basis = PSpline(k=10).fit(x)
    >>> B = basis.basis_matrix(x)
    >>> B.shape
    (100, 10)
    >>> S = basis.penalty_matrix()
    >>> S.shape
    (10, 10)
    """

    def __init__(self, k: int = 10, degree: int = 3, m: int = 2) -> None:
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}.")
        if degree < 1:
            raise ValueError(f"degree must be at least 1, got {degree}.")
        if m < 1:
            raise ValueError(f"Penalty order m must be at least 1, got {m}.")
        if m >= k:
            raise ValueError(f"Penalty order m={m} must be less than k={k}.")
        self.k = k
        self.degree = degree
        self.m = m
        self._fitted: bool = False

        # Set by fit():
        self._t: NDArray  # augmented knot vector
        self._x_min: float
        self._x_max: float
        self._S: NDArray  # (k, k) penalty matrix

    # ------------------------------------------------------------------
    # SmoothBasis interface
    # ------------------------------------------------------------------

    def fit(self, x: NDArray) -> PSpline:
        """Fit the P-spline to training data `x`.

        Determines the knot vector from the data range and pre-computes the penalty matrix.
        No per-observation computation is stored as the B-spline design matrix is always evaluated
        on demand.

        Parameters
        ----------
        x:
            1-D training covariate, shape `(n,)` or `(n, 1)`.

        Returns
        -------
        PSpline
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If `n < k`, or if `x` is not 1-D / `(n, 1)`.
        """
        x1d = self._as_1d(x)
        n = len(x1d)

        if n < self.k:
            raise ValueError(
                f"Number of observations n={n} is smaller than k={self.k}. "
                f"Reduce k to at most {n - 1}."
            )
        if self.m >= self.k:
            raise ValueError(f"Penalty order m={self.m} must be less than k={self.k}.")

        self._x_min = float(x1d.min())
        self._x_max = float(x1d.max())

        if self._x_min == self._x_max:
            raise ValueError("All covariate values are identical; cannot fit a spline.")

        self._t = _bspline_knots(self._x_min, self._x_max, self.k, self.degree)

        # Penalty: S = D_m' D_m
        D = _diff_matrix(self.k, self.m)  # (k-m, k)
        S = D.T @ D  # (k, k)
        self._S = (S + S.T) * 0.5  # symmetrize against floating-point drift

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the B-spline basis at `x`.

        Values outside the training range `[x_min, x_max]` are extrapolated by the boundary B-spline
        functions (the basis smoothly extends beyond the knot range).

        Parameters
        ----------
        x:
            1-D evaluation points, shape `(n,)` or `(n, 1)`.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)`.
        """
        self._check_fitted()
        x1d = self._as_1d(x)

        # Clip to the knot support so BSpline.design_matrix doesn't raise.
        # The support is [t[degree], t[-(degree+1)]] ≈ [x_min, x_max + eps].
        x_eval = np.clip(x1d, self._t[self.degree], self._t[-(self.degree + 1)])

        return _bspline_design(x_eval, self._t, self.degree)

    def penalty_matrix(self) -> NDArray:
        """Return the `k x k` penalty matrix `S = D_m' D_m`.

        `S` is positive semi-definite with rank `k - m`.  Its null space is
        spanned by the `m` polynomial sequences
        `[1, 1, ..., 1]`, `[0, 1, ..., k-1]`, ...,
        `[0^(m-1), 1^(m-1), ..., (k-1)^(m-1)]`.

        Returns
        -------
        NDArray
            Shape `(k, k)`.
        """
        self._check_fitted()
        return self._S.copy()

    def null_space_dimension(self) -> int:
        """Return `m`: the difference penalty has an `m`-dimensional null space."""
        return self.m

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row.

        Returns a `(1, k)` matrix that enforces mean-zero contribution
        over the training knot range.
        """
        self._check_fitted()
        # Evaluate at equidistant points spanning the training range.
        x_ref = np.linspace(self._x_min, self._x_max, max(self.k * 5, 100))
        B_ref = self.basis_matrix(x_ref)
        return B_ref.mean(axis=0, keepdims=True)  # (1, k)

    @property
    def n_basis(self) -> int:
        """Total number of B-spline basis functions `k`."""
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
        """Full augmented knot vector (length `k + degree + 1`)."""
        self._check_fitted()
        return self._t.copy()

    @property
    def interior_knots(self) -> NDArray:
        """Interior knots only (excludes the repeated boundary knots)."""
        self._check_fitted()
        d = self.degree
        return self._t[d + 1 : -(d + 1)].copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_1d(x: NDArray) -> NDArray:
        """Coerce `x` to a 1-D float array.

        Accepts shape `(n,)` or `(n, 1)`. Raises for anything else (P-splines are univariate).
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(
            "PSpline is a univariate basis: pass a 1-D array or an (n, 1) "
            f"column vector, not an array with shape {x.shape}."
        )
