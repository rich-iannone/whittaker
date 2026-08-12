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
    r"""P-Spline: B-spline basis with m-th order difference penalty.

    A P-spline (Eilers & Marx 1996) combines a rich B-spline basis on equally-spaced knots with a
    discrete difference penalty directly on the coefficients, rather than an integrated-derivative
    penalty on the fitted curve. This decouples basis richness from smoothness — you can use many
    more basis functions than you would dare with an unpenalized spline, because the difference
    penalty (together with an appropriately chosen `lambda`) does the work of controlling wiggliness.
    Equivalent to mgcv's `bs="ps"` basis. P-splines are a good default choice for a single
    continuous covariate: they are cheap to construct (no eigendecomposition or dense linear solve
    is needed to build the penalty), the penalty matrix is banded/sparse which keeps large-`k` fits
    fast, and B-splines extrapolate smoothly beyond the training range via their boundary basis
    functions. Unlike `CRS`, knots are equidistant rather than placed at data quantiles, so P-splines
    are somewhat less efficient when the data are very unevenly distributed but are simpler and
    faster to set up.

    Parameters
    ----------
    k:
        Number of B-spline basis functions. Must satisfy `k >= degree + 1`. Larger `k` gives a
        richer basis (more, narrower B-splines) and lets the fit follow finer local structure; as
        with other penalized bases, wiggliness is ultimately governed by the penalty and `lambda`,
        not by `k` alone, so `k` can usually be set generously (e.g. `k=20-40`) without much
        downside. The default is `10`.
    degree:
        Polynomial degree of each B-spline piece. `degree=3` (cubic) is the conventional choice and
        matches most GAM software; `degree=1` gives a piecewise-linear basis useful for less smooth
        phenomena, `degree=0` a step-function basis. The default is `3` (cubic).
    m:
        Order of the difference penalty applied to adjacent B-spline coefficients. `m=2` (the
        default) penalizes second differences, which is the discrete analogue of penalizing
        curvature and is by far the most common choice; `m=1` penalizes changes in level (shrinks
        toward a constant), `m=3` penalizes changes in slope-of-slope for extra-smooth fits. Must
        satisfy `0 < m < k`. The default is `2` (second differences: penalizes curvature).

    Notes
    -----
    The knot vector is built by `_bspline_knots()`: `degree + 1` repeated (clamped) knots at each
    of `x_min` and `x_max`, plus `k - degree - 1` interior knots equally spaced between them,
    giving `k` B-spline basis functions of the requested `degree` via `scipy`'s `BSpline`
    machinery (de Boor's algorithm). Because the knots are equally spaced rather than data-adaptive,
    the design matrix `basis_matrix(x)` can be evaluated in closed form for any `x`, including
    points beyond `[x_min, x_max]`, which the boundary B-splines extend smoothly.

    The penalty acts directly on the coefficient vector `\boldsymbol{\beta}` through the `m`-th
    order finite-difference matrix `\mathbf{D}_m` (shape `(k - m, k)`, built by applying
    `numpy.diff` `m` times to the identity):

    $$
    \mathbf{S} = \mathbf{D}_m^\top \mathbf{D}_m,
    $$

    a `k \times k` positive semi-definite matrix of rank `k - m`. Its `m`-dimensional null space is
    spanned by the discrete polynomial sequences `[1, 1, \ldots, 1]`, `[0, 1, \ldots, k-1]`, ...,
    up to degree `m - 1` in the coefficient index — i.e. coefficient vectors that are themselves
    polynomial in index have zero penalty, mirroring how polynomials of degree `< m` have zero
    `m`-th derivative in the continuous case. Because `S` is banded (bandwidth `m`), it is sparse
    and cheap to factorize even for large `k`; the main numerical caveat is that `basis_matrix()`
    clips evaluation points to the B-spline's knot support before calling `scipy`'s
    `BSpline.design_matrix`, since points exactly at or beyond the padded boundary can otherwise
    trigger an out-of-support error.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import PSpline

    x = np.linspace(0, 1, 100)
    basis = PSpline(k=10).fit(x)
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
        """Return the dimension of the penalty null space.

        For a P-spline with difference order `m`, the null space of `D_m^T D_m` is exactly
        `m`-dimensional: it is spanned by the coefficient sequences that are themselves polynomial
        in the coefficient index up to degree `m - 1`, since these have zero `m`-th finite
        difference.

        Returns
        -------
        int
            The difference order `m`.
        """
        return self.m

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Evaluates the basis on a fine equally-spaced grid spanning the training range and averages
        each column, giving a `(1, k)` row `C` such that `C @ beta == 0` constrains the smooth to
        have mean zero over the training range.

        Returns
        -------
        NDArray
            Shape `(1, k)` matrix that enforces mean-zero contribution over the training knot
            range.
        """
        self._check_fitted()
        # Evaluate at equidistant points spanning the training range.
        x_ref = np.linspace(self._x_min, self._x_max, max(self.k * 5, 100))
        B_ref = self.basis_matrix(x_ref)
        return B_ref.mean(axis=0, keepdims=True)  # (1, k)

    @property
    def n_basis(self) -> int:
        """Total number of B-spline basis functions.

        Equal to `k`, the requested basis dimension: every B-spline
        coefficient column is retained (no null-space or wrap-around columns
        are dropped, unlike some of the other smooth bases).

        Returns
        -------
        int
            The basis dimension `k`.
        """
        return self.k

    @property
    def is_fitted(self) -> bool:
        """Whether the basis has been fitted.

        Returns
        -------
        bool
            `True` once `fit()` has been called and the knot vector and
            penalty matrix have been computed; `False` otherwise.
        """
        return self._fitted

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def knots(self) -> NDArray:
        """Full augmented B-spline knot vector.

        Includes the `degree + 1` repeated boundary knots at each end
        (needed for the clamped B-spline construction) as well as the
        equally-spaced interior knots.

        Returns
        -------
        NDArray
            Knot vector, shape `(k + degree + 1,)`.
        """
        self._check_fitted()
        return self._t.copy()

    @property
    def interior_knots(self) -> NDArray:
        """Interior knot locations only.

        Excludes the repeated boundary knots at `x_min` and `x_max`, leaving
        just the `k - degree - 1` equally-spaced knots strictly between them.

        Returns
        -------
        NDArray
            Interior knot locations, shape `(k - degree - 1,)`.
        """
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
