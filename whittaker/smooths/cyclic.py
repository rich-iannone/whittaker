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
    for idx in range(m):
        hp = h[(idx - 1) % m]
        hc = h[idx]
        Q[(idx - 1) % m, idx] += 1.0 / hp
        Q[idx, idx] += -(1.0 / hp + 1.0 / hc)
        Q[(idx + 1) % m, idx] += 1.0 / hc
    return Q


def _build_cyclic_R(h: NDArray) -> NDArray:
    """Build the (m x m) cyclic tridiagonal matrix R_c."""
    m = len(h)
    R = np.zeros((m, m))
    for idx in range(m):
        hp = h[(idx - 1) % m]
        hc = h[idx]
        R[idx, idx] = (hp + hc) / 3.0
        R[idx, (idx + 1) % m] += hc / 6.0
        R[(idx + 1) % m, idx] += hc / 6.0
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
    r"""Cyclic Cubic Regression Spline (periodic natural cubic spline).

    A cyclic (periodic) cubic regression spline is the natural extension of `CRS` to covariates
    that wrap around, such as time of day, day of year, or wind direction, where the value and
    slope at the end of the range must match the value and slope at the start. Equivalent to mgcv's
    `bs="cc"` basis. The spline is periodic over the range of the training data:
    `f(x_min) = f(x_max)`, `f'(x_min) = f'(x_max)`, and `f''(x_min) = f''(x_max)`. Values outside
    the training range are mapped into the periodic domain via modular arithmetic, so predictions
    remain well-defined for any `x`. Choose `CyclicCRS` (rather than plain `CRS`) whenever the
    covariate is inherently circular; using a non-cyclic basis on such data would otherwise produce
    an artificial discontinuity at the wrap-around point.

    The cyclic constraint absorbs one degree of freedom, so k knots produce k-1 basis functions. The
    penalty null space is 1-dimensional (constant functions only as linear functions are no longer
    unpenalized under periodicity).

    Parameters
    ----------
    k:
        Number of knots placed around the periodic domain. Must be at least `4`, which yields at
        least `3` basis functions (`n_basis = k - 1`) after the cyclic constraint removes one degree
        of freedom. As with `CRS`, larger `k` gives more local flexibility, with overall smoothness
        controlled by the penalty and `lambda` rather than by `k` alone. The default is `10`.

    Notes
    -----
    Knots are placed at evenly-spaced quantiles of the training data (falling back to an
    evenly-spaced grid when quantile ties would produce duplicates), exactly as in `CRS`, but the
    second-derivative relationship between the coefficient vector and the knot second derivatives
    is built on a *circular* tridiagonal system: `_build_cyclic_Q()` and `_build_cyclic_R()`
    construct the periodic analogues of `CRS`'s `Q` and `R` matrices, wrapping the
    sub/super-diagonal entries around modulo `m = k - 1` (the number of free coefficients once
    `f(x_min) = f(x_max)` is imposed). Concretely, the periodic constraint identifies knot
    `t_{k-1}` with `t_0`, so `\boldsymbol{\beta}` has only `m = k - 1` free entries, and second
    derivatives satisfy `\mathbf{R}_c \mathbf{d} = \mathbf{Q}_c^\top \boldsymbol{\beta}` with
    `Q_c`, `R_c` both
    `m \times m` and *circulant-tridiagonal* (each row's neighbors wrap around index `m`). Unlike
    `CRS`, there are no natural boundary conditions — periodicity itself replaces them — so the
    penalty null space drops from dimension 2 (constant and linear) to dimension 1 (constant only):
    a periodic function cannot be a nonzero linear function of `x`, since a nonzero slope is
    incompatible with `f(x_min) = f(x_max)`.

    The penalty is the same quadratic form as in `CRS`, computed from the circular matrices,

    $$
    \mathbf{S} = \mathbf{Q}_c \mathbf{R}_c^{-1} \mathbf{Q}_c^\top,
    $$

    an `m \times m` (`m = k - 1`) positive semi-definite matrix of rank `m - 1`, whose
    one-dimensional null space is the constant sequence. At evaluation time, `basis_matrix()`
    maps any `x` into the periodic domain `[x_min, x_max)` via
    `knots[0] + (x - knots[0]) % period` before evaluating the
    same piecewise-cubic construction as `CRS`, so extrapolation is not linear (as in `CRS`) but
    exactly periodic.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import CyclicCRS

    x = np.linspace(0, 2 * np.pi, 100)
    basis = CyclicCRS(k=10).fit(x)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
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
        """Fit the cyclic CRS to training data `x`.

        Places `k` knots at evenly-spaced quantiles of `x` (spanning one full period), then
        pre-computes the circular second-derivative operator and penalty matrix.

        Parameters
        ----------
        x:
            1-D training covariate, shape `(n,)` or `(n, 1)`. The period is inferred as
            `x.max() - x.min()`.

        Returns
        -------
        CyclicCRS
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
        """Evaluate the cyclic CRS basis at `x`.

        Any `x` outside the training range is first mapped into the periodic domain
        `[x_min, x_max)` via modular arithmetic, so the returned basis values are exactly periodic
        with period `x_max - x_min`.

        Parameters
        ----------
        x:
            1-D evaluation points, shape `(n,)` or `(n, 1)`.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k - 1)`.
        """
        self._check_fitted()
        x1d = self._as_1d(x)
        return _cyclic_crs_basis_matrix(x1d, self._knots, self._h, self._A, self._period)

    def penalty_matrix(self) -> NDArray:
        """Return the `(k - 1) x (k - 1)` cyclic penalty matrix `S = Q_c R_c^{-1} Q_c'`.

        `S` is positive semi-definite with rank `k - 2`. Its one-dimensional null space is spanned
        by the constant function (linear functions are no longer unpenalized under periodicity).

        Returns
        -------
        NDArray
            Shape `(k - 1, k - 1)`.
        """
        self._check_fitted()
        return self._S.copy()

    def null_space_dimension(self) -> int:
        """Return the dimension of the penalty null space.

        Always `1` for a cyclic CRS: only constant functions have zero penalty, since a nonzero
        linear term would violate the periodicity constraint `f(x_min) = f(x_max)`.

        Returns
        -------
        int
            Always `1`.
        """
        return 1

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Evaluates the fitted basis at the free knots (excluding the duplicated wrap-around knot)
        and averages each column, giving a `(1, k - 1)` row `C` such that `C @ beta == 0`
        constrains the smooth to have mean zero over the training data.

        Returns
        -------
        NDArray
            Shape `(1, k - 1)` constraint matrix.
        """
        self._check_fitted()
        B_train = self.basis_matrix(self._knots[:-1])
        return B_train.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        """Total number of basis functions.

        Equal to `k - 1`: the periodicity constraint `f(x_min) = f(x_max)`
        identifies the last knot with the first, so only `k - 1` of the `k`
        knot values are free coefficients.

        Returns
        -------
        int
            The basis dimension `k - 1`.
        """
        return self.k - 1

    @property
    def knots(self) -> NDArray:
        """Knot locations used by the fitted basis.

        The `k` locations, placed at evenly-spaced quantiles of the training
        data (or an evenly-spaced grid as a fallback), spanning one full
        period `[x_min, x_max]`. The last knot is identified with the first
        under the periodicity constraint, leaving `k - 1` free coefficients.

        Returns
        -------
        NDArray
            Strictly increasing knot locations, shape `(k,)`.
        """
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
    r"""Cyclic P-Spline (periodic B-spline basis with circular difference penalty).

    A cyclic P-spline is the periodic counterpart of `PSpline`: a B-spline basis whose knots and
    coefficients wrap around a circle, combined with a *circular* finite-difference penalty that
    also wraps around, so smoothness is enforced across the boundary rather than just within it.
    Equivalent to mgcv's `bs="cp"` basis. The B-spline basis is constructed to be periodic over the
    training data range, so `f(x_min) = f(x_max)` and, because the penalty ties coefficients across
    the wrap point, the fit avoids any artificial kink at the seam. Values outside the training
    range are mapped into the periodic domain via modular arithmetic. Prefer `CyclicPSpline` over
    `CyclicCRS` for the same reasons `PSpline` is often preferred over `CRS`: cheaper construction,
    equally-spaced (rather than quantile) knots, and a sparse banded penalty that scales well to
    larger `k`; prefer `CyclicCRS` when knot-based interpretability or an integrated-derivative
    penalty is wanted.

    The circular difference penalty penalizes the m-th differences of adjacent coefficients with
    wrap-around at the boundaries. The penalty null space is 1-dimensional (constant functions
    only).

    Parameters
    ----------
    k:
        Number of periodic B-spline basis functions. Must satisfy `k >= degree + 1`. Larger `k`
        gives more local flexibility around the cycle; overall smoothness is governed by the
        penalty and `lambda`. The default is `10`.
    degree:
        Polynomial degree of each B-spline piece. `degree=3` (cubic) is conventional; lower degrees
        give coarser, more locally-supported bases. The default is `3` (cubic).
    m:
        Order of the circular difference penalty applied to adjacent coefficients, with
        wrap-around so that the coefficients at the end of the cycle are treated as adjacent to
        those at the start. `m=2` (the default) penalizes circular second differences, the
        periodic analogue of curvature. The default is `2`.

    Notes
    -----
    `_periodic_bspline_knots()` builds a periodic knot vector by taking `k` equally-spaced knots
    spanning one period `[x_min, x_max)` and extending them periodically by `degree` knots on each
    side, so that the B-spline basis functions near the boundary have support that wraps smoothly
    around the cycle. `basis_matrix()` evaluates the (non-periodic) B-spline design on this extended
    knot vector and then folds the trailing `degree` "wrapped" columns back onto the leading
    `degree` columns (`B[:, :degree] += B_full[:, k:]`), so the returned matrix has exactly `k`
    periodic basis functions.

    The penalty replaces the ordinary finite-difference matrix used by `PSpline` with a *circular*
    difference matrix `\mathbf{D}_m` (built by `_cyclic_diff_matrix()` via repeated
    left-multiplication by the circular first-difference operator `D_1[i, i] = -1`,
    `D_1[i, (i+1) \bmod k] = 1`), giving a square `k \times k` penalty

    $$
    \mathbf{S} = \mathbf{D}_m^\top \mathbf{D}_m,
    $$

    positive semi-definite with rank `k - 1`. Its one-dimensional null space is the constant
    coefficient sequence: because the difference operator wraps around, no nonconstant polynomial
    sequence in the coefficient index can have zero circular difference (unlike the non-cyclic
    `PSpline`, where degree-`< m` polynomial sequences are all unpenalized). As with `PSpline`, the
    penalty is banded/sparse (with corner entries from the wrap-around) and cheap to factorize even
    for large `k`.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import CyclicPSpline

    x = np.linspace(0, 2 * np.pi, 100)
    basis = CyclicPSpline(k=10).fit(x)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
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
        """Fit the cyclic P-spline to training data `x`.

        Determines the period from the data range, builds the periodic knot vector, and
        pre-computes the circular difference penalty matrix.

        Parameters
        ----------
        x:
            1-D training covariate, shape `(n,)` or `(n, 1)`. The period is inferred as
            `x.max() - x.min()`.

        Returns
        -------
        CyclicPSpline
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If `n < k`, if all values of `x` are identical, or if `x` is not 1-D / `(n, 1)`.
        """
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
        """Evaluate the periodic B-spline basis at `x`.

        Any `x` outside the training range is first mapped into the periodic domain
        `[x_min, x_max)` via modular arithmetic. The trailing `degree` columns of the raw B-spline
        design (from the periodic knot extension) are folded back onto the leading `degree`
        columns so the result has exactly `k` periodic basis functions.

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
        """Return the `k x k` circular penalty matrix `S = D_m' D_m`.

        `S` is positive semi-definite with rank `k - 1`. Its one-dimensional null space is spanned
        by the constant coefficient sequence; unlike the non-cyclic `PSpline`, no nonconstant
        polynomial sequence in the coefficient index is unpenalized because the difference operator
        wraps around the boundary.

        Returns
        -------
        NDArray
            Shape `(k, k)`.
        """
        self._check_fitted()
        return self._S.copy()

    def null_space_dimension(self) -> int:
        """Return the dimension of the penalty null space.

        Always `1` for a cyclic P-spline: only the constant coefficient sequence has zero circular
        difference.

        Returns
        -------
        int
            Always `1`.
        """
        return 1

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Evaluates the basis on a fine equally-spaced grid spanning one period and averages each
        column, giving a `(1, k)` row `C` such that `C @ beta == 0` constrains the smooth to have
        mean zero over the training range.

        Returns
        -------
        NDArray
            Shape `(1, k)` constraint matrix.
        """
        self._check_fitted()
        x_ref = np.linspace(self._x_min, self._x_max, max(self.k * 5, 100), endpoint=False)
        B_ref = self.basis_matrix(x_ref)
        return B_ref.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        """Total number of periodic basis functions.

        Equal to `k`: unlike `CyclicCRS`, the periodic B-spline construction
        folds the extra "wrapped" columns back into the leading `degree`
        columns during basis construction, so no coefficients are dropped and
        the basis dimension is exactly `k`.

        Returns
        -------
        int
            The basis dimension `k`.
        """
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
