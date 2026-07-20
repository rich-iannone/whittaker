"""Thin Plate Regression Splines (TPRS)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis

# ---------------------------------------------------------------------------
# Low-level kernel functions
# ---------------------------------------------------------------------------


def _radial_basis(r: NDArray, d: int, m: int) -> NDArray:
    """Evaluate the TPS radial basis function η_m(r) for dimension d.

    For order m and dimension d:

    * If `2m - d` is **odd**: η(r) = r^(2m-d)
    * If `2m - d` is **even**: η(r) = r^(2m-d) ⋅ log(r), with η(0) = 0

    Parameters
    ----------
    r:
        Non-negative radii. Any shape.
    d:
        Covariate dimension.
    m:
        Spline order. Only m=2 is supported.

    Returns
    -------
    NDArray
        Same shape as `r`.
    """
    power = 2 * m - d
    if power <= 0:
        raise ValueError(f"Spline order m={m} is too low for dimension d={d}: need 2m > d.")
    if power % 2 == 1:
        # Odd power — no logarithm needed.
        return r**power
    else:
        # Even power — use r^p * log(r), convention 0 * log(0) = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r == 0.0, 0.0, r**power * np.log(r))


def _kernel_matrix(x1: NDArray, x2: NDArray, d: int, m: int) -> NDArray:
    """Build the n1 x n2 TPS kernel matrix E with E_{ij} = η_m(||x1_i - x2_j||).

    Parameters
    ----------
    x1:
        Array of shape `(n1, d)`.
    x2:
        Array of shape `(n2, d)`.
    d:
        Covariate dimension (must match last axis of `x1` and `x2`).
    m:
        Spline order.

    Returns
    -------
    NDArray
        Shape `(n1, n2)`.
    """
    # Pairwise squared distances via broadcasting.
    diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]  # (n1, n2, d)
    r = np.sqrt(np.sum(diff**2, axis=-1))  # (n1, n2)
    return _radial_basis(r, d=d, m=m)


def _polynomial_null_space(x: NDArray, m: int) -> NDArray:
    """Build the n x M polynomial null-space matrix T for order m.

    For m=2 (supported only), T = [1, x_1, …, x_d].

    Parameters
    ----------
    x:
        Shape `(n, d)`.
    m:
        Spline order (must be 2).

    Returns
    -------
    NDArray
        Shape `(n, M)` where M = d + 1 for m = 2.
    """
    if m != 2:
        raise NotImplementedError(
            f"Polynomial null space for m={m} is not yet implemented. "
            "Only m=2 is currently supported."
        )
    n, d = x.shape
    # T = [1, x1, x2, ..., xd]
    return np.column_stack([np.ones(n), x])


# ---------------------------------------------------------------------------
# Public TPRS class
# ---------------------------------------------------------------------------


class TPRS(SmoothBasis):
    """Thin Plate Regression Splines (TPRS).

    Constructs a rank-k approximation to the full TPS basis using the leading eigenvectors of the
    projected kernel matrix (Wood 2003).

    The basis has two parts:

    * **Polynomial null space** (first M = d + 1 columns): unpenalised affine functions
    `[1, x_1, …, x_d]`.
    * **Spline part** (remaining k - M columns): penalised, constructed from the leading
    eigenvectors of the projected TPS kernel.

    The penalty matrix is block-diagonal:

    S = diag(0, …, 0,  λ_1, …, λ_{k - M})

    Parameters
    ----------
    k:
        Total number of basis functions (including the M null-space columns). Must satisfy
        `k > d + 1`. The default is `10`.
    m:
        Spline order. Controls the order of derivative penalised. Only `m=2` (penalise squared
        second derivatives / curvature) is supported. The default is `2`.

    Notes
    -----
    Covariates with very different scales can cause numerical issues. It is advisable to center
    and/or standardize each column of `x` before fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from whittaker.smooths import TPRS
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 1, 100)
    >>> basis = TPRS(k=10).fit(x)
    >>> B = basis.basis_matrix(x)
    >>> B.shape
    (100, 10)
    >>> S = basis.penalty_matrix()
    >>> S.shape
    (10, 10)
    """

    def __init__(self, k: int = 10, m: int = 2) -> None:
        if m != 2:
            raise NotImplementedError(
                f"TPRS with m={m} is not yet supported. Only m=2 is implemented."
            )
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}.")
        self.k = k
        self.m = m
        self._fitted: bool = False

        # Attributes populated by fit():
        self._d: int
        self._M: int
        self._x_train: NDArray
        self._QU: NDArray  # (n_train, k - M) — spline weight matrix
        self._eigenvalues: NDArray  # (k - M,)

    # ------------------------------------------------------------------
    # SmoothBasis interface
    # ------------------------------------------------------------------

    def fit(self, x: NDArray) -> TPRS:
        """Fit the TPRS to training data `x`.

        Parameters
        ----------
        x:
            Training covariates. Shape `(n,)` for univariate or `(n, d)` for multivariate.

        Returns
        -------
        TPRS
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If `k` is too large for the number of observations or too small for the covariate
            dimension.
        """
        x2d = self._as_2d(x)
        n, d = x2d.shape

        M = d + 1  # null-space dimension for m=2
        r = self.k - M  # number of spline basis functions

        if r < 1:
            raise ValueError(
                f"k={self.k} is too small for d={d}: need k > {M} "
                f"(the null-space dimension M = d + 1 = {M})."
            )
        if n < self.k:
            raise ValueError(
                f"Number of observations n={n} is smaller than k={self.k}. "
                f"Reduce k to at most {n - 1}."
            )

        self._d = d
        self._M = M
        self._x_train = x2d.copy()

        # 1. Build n × n kernel matrix E.
        E = _kernel_matrix(x2d, x2d, d=d, m=self.m)
        E = (E + E.T) * 0.5  # symmetrize to remove floating-point drift

        # 2. Build n × M polynomial null-space matrix T.
        T = _polynomial_null_space(x2d, m=self.m)

        # 3. Full QR of T: T = Q * R.  Q is n × n orthogonal.
        #    Q[:, :M]  spans T  (Q_1)
        #    Q[:, M:]  is ⊥ to T  (Q_2, shape n × (n - M))
        Q_full, _ = np.linalg.qr(T, mode="complete")
        Q2 = Q_full[:, M:]  # (n, n-M)

        # 4. Project E onto the complement of T's column space.
        #    G = Q2' E Q2  — symmetric (n-M) × (n-M)
        G = Q2.T @ E @ Q2
        G = (G + G.T) * 0.5  # symmetrize

        # 5. Eigen-decompose G (symmetric → use eigh for stability).
        #    np.linalg.eigh returns eigenvalues in *ascending* order.
        eigenvalues, eigenvectors = np.linalg.eigh(G)

        # Sort in *descending* order to pick the r largest eigenvalues.
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 6. Truncate to the r leading eigenvectors.
        U_r = eigenvectors[:, :r]  # (n-M, r)
        D_r = eigenvalues[:r]  # (r,)

        # 7. Spline weight matrix for prediction.
        #    ψ_l(x*) = Σ_j QU[j, l] * η(||x* - x_j||)
        self._QU = Q2 @ U_r  # (n, r)
        self._eigenvalues = D_r

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the TPRS basis at `x`.

        Parameters
        ----------
        x:
            Covariate values. Shape `(n,)` or `(n, d)` where d must match the training dimension.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)`. The first M = d + 1 columns are the polynomial
            null-space functions; the remaining k - M columns are the truncated spline functions.
        """
        self._check_fitted()
        x2d = self._as_2d(x)

        if x2d.shape[1] != self._d:
            raise ValueError(f"Expected {self._d} covariate column(s), got {x2d.shape[1]}.")

        # Polynomial null-space columns.
        T_new = _polynomial_null_space(x2d, m=self.m)  # (n_new, M)

        # Kernel matrix of new points vs. training points.
        E_new = _kernel_matrix(x2d, self._x_train, d=self._d, m=self.m)  # (n_new, n)

        # Spline columns: E_new @ QU  (n_new, r)
        spline_cols = E_new @ self._QU

        return np.column_stack([T_new, spline_cols])  # (n_new, k)

    def penalty_matrix(self) -> NDArray:
        """Return the k x k penalty matrix S.

        S is block-diagonal:

        * First M rows/cols: all zeros (null-space, unpenalised).
        * Remaining k - M rows/cols: diagonal entries equal to the leading eigenvalues of the
        projected kernel matrix.

        Returns
        -------
        NDArray
            Shape `(k, k)`.
        """
        self._check_fitted()
        S = np.zeros((self.k, self.k))
        M = self._M
        S[M:, M:] = np.diag(self._eigenvalues)
        return S

    def null_space_dimension(self) -> int:
        """Return M = d + 1, the dimension of the polynomial null space."""
        self._check_fitted()
        return self._M

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Returns a `(1, k)` matrix whose product with the coefficient vector equals zero when the
        smooth has mean zero over the training data.
        """
        self._check_fitted()
        B_train = self.basis_matrix(self._x_train)
        # Constraint: colMeans(B_train) @ beta = 0
        C = B_train.mean(axis=0, keepdims=True)  # (1, k)
        return C

    @property
    def n_basis(self) -> int:
        """Total number of basis functions k."""
        return self.k

    @property
    def is_fitted(self) -> bool:
        """`True` after `fit()` has been called."""
        return self._fitted

    # ------------------------------------------------------------------
    # Convenience properties (available after fit)
    # ------------------------------------------------------------------

    @property
    def d(self) -> int:
        """Covariate dimension (set during `fit()`)."""
        self._check_fitted()
        return self._d

    @property
    def null_space_dim(self) -> int:
        """Alias for `null_space_dimension()`."""
        return self.null_space_dimension()

    @property
    def eigenvalues(self) -> NDArray:
        """Eigenvalues of the projected kernel matrix (k - M values)."""
        self._check_fitted()
        return self._eigenvalues.copy()
