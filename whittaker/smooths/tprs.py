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
        Spline order. Must satisfy `2m > d`.

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


def _null_space_dimension(d: int, m: int) -> int:
    """Number of monomials of total degree ≤ m-1 in d variables: C(m-1+d, d)."""
    from math import comb

    return comb(m - 1 + d, d)


def _polynomial_null_space(x: NDArray, m: int) -> NDArray:
    """Build the n x M polynomial null-space matrix T for order m.

    The null space of the m-th order thin plate spline penalty consists of all polynomials of total
    degree ≤ m-1 in d variables. The number of such monomials is M = C(m-1+d, d).

    Parameters
    ----------
    x:
        Shape `(n, d)`.
    m:
        Spline order (≥ 2).

    Returns
    -------
    NDArray
        Shape `(n, M)` where `M = C(m-1+d, d)`.
    """
    n, d = x.shape
    max_deg = m - 1

    if max_deg == 1:
        return np.column_stack([np.ones(n), x])

    from itertools import combinations_with_replacement

    cols: list[NDArray] = [np.ones(n)]
    for deg in range(1, max_deg + 1):
        for idx in combinations_with_replacement(range(d), deg):
            col = np.ones(n)
            for j in idx:
                col = col * x[:, j]
            cols.append(col)
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Public TPRS class
# ---------------------------------------------------------------------------


class TPRS(SmoothBasis):
    r"""Thin Plate Regression Splines (TPRS).

    A thin plate spline is the function that minimizes squared error subject to a penalty on
    total curvature, with no need to choose knot locations — it is the natural multivariate
    generalization of the cubic smoothing spline. The full thin plate spline has one basis
    function per unique data point, which is computationally impractical for anything beyond a
    few hundred observations, so TPRS instead constructs a low-rank approximation using the
    leading eigenvectors of the (nullspace-projected) thin plate spline kernel matrix (Wood 2003).
    Because it works for any number of covariate dimensions `d` and requires no knot placement,
    TPRS is a good default basis for smooth terms of one or more continuous, non-cyclic
    covariates, especially in more than two dimensions where tensor-product alternatives become
    unwieldy.

    Parameters
    ----------
    k:
        Total number of basis functions (including the `M` null-space columns). Must satisfy
        `k > M`. Larger `k` allows more wiggly fits at the cost of more computation; the penalty
        (not `k`) ultimately controls smoothness once `lambda` is chosen. The default is `10`.
    m:
        Spline order. Controls the order of derivative penalized: `m=2` penalizes (squared)
        second derivatives, the classic "thin plate" bending energy. Must satisfy `2m > d` where
        `d` is the covariate dimension. Common choices: `m=2` for `d <= 3` (the default), `m=3`
        for `d` in `{4, 5}`. The default is `2`.

    Notes
    -----
    The full thin plate spline basis uses the radial kernel

    $$
    \eta_m(r) = \begin{cases} r^{2m - d} & 2m - d \text{ odd} \\ r^{2m-d} \log(r) & 2m - d \text{ even} \end{cases},
    $$

    evaluated at pairwise distances `r = ||x_i - x_j||` between data points, plus a polynomial
    null space of all monomials of total degree at most `m - 1`, which has dimension
    `M = C(m - 1 + d, d)`. TPRS builds the basis in two stages:

    1. **Polynomial null space** (first `M` columns): the unpenalized low-degree polynomials,
       for which the roughness penalty is identically zero (e.g. any straight line has zero
       bending energy under `m=2`).
    2. **Truncated spline part** (remaining `k - M` columns): the full kernel matrix `E` is
       projected onto the orthogonal complement of the polynomial null space and eigen-decomposed;
       the `k - M` eigenvectors with the largest eigenvalues give the best rank-`(k - M)`
       approximation to the full thin plate spline in the sense of minimizing the change in the
       penalty for a given basis dimension.

    The resulting penalty matrix is block-diagonal,

    $$
    \mathbf{S} = \operatorname{diag}(0, \ldots, 0, \lambda_1, \ldots, \lambda_{k-M}),
    $$

    with the `M` null-space rows/columns exactly zero and the remaining diagonal entries equal to
    the retained eigenvalues of the projected kernel matrix. Because the basis is derived from an
    eigendecomposition of a matrix that mixes all covariate scales, columns of `x` with very
    different scales can cause numerical issues; centering and/or standardizing each column of
    `x` before fitting is advisable.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import TPRS

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)

    basis = TPRS(k=10).fit(x)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
    """

    def __init__(self, k: int = 10, m: int = 2) -> None:
        if m < 2:
            raise ValueError(f"Spline order m must be at least 2, got {m}.")
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

        M = _null_space_dimension(d, self.m)
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

        # 6. Truncate to the r leading eigenvectors and clamp eigenvalues
        #    (small negatives are numerical artifacts of the projection).
        U_r = eigenvectors[:, :r]  # (n-M, r)
        D_r = np.maximum(eigenvalues[:r], 0.0)  # (r,)

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
        """Return the dimension of the unpenalized polynomial null space.

        The TPRS penalty is exactly zero on polynomials of total degree at
        most `m - 1`, which span an `M`-dimensional null space with
        `M = C(m - 1 + d, d)` (the number of monomials of degree
        `<= m - 1` in `d` variables). These occupy the first `M` columns
        of the basis matrix.

        Returns
        -------
        int
            The null-space dimension `M`.
        """
        self._check_fitted()
        return self._M

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        When a TPRS term is combined with other terms sharing an
        intercept, it must be constrained so its contribution has zero
        mean over the training data to remain identifiable. The
        constraint is `C @ beta = 0` where `C` is the column mean of the
        training basis matrix, i.e. `colMeans(B_train) @ beta = 0`.

        Returns
        -------
        NDArray
            Constraint matrix of shape `(1, k)`.
        """
        self._check_fitted()
        B_train = self.basis_matrix(self._x_train)
        # Constraint: colMeans(B_train) @ beta = 0
        C = B_train.mean(axis=0, keepdims=True)  # (1, k)
        return C

    @property
    def n_basis(self) -> int:
        """Total number of basis functions `k`.

        Equal to the `k` argument supplied at construction, i.e. the
        combined size of the polynomial null space (`M` columns) and the
        truncated spline part (`k - M` columns).
        """
        return self.k

    @property
    def is_fitted(self) -> bool:
        """`True` after `fit()` has been called.

        `basis_matrix()`, `penalty_matrix()`, and the convenience
        properties (`d`, `eigenvalues`, etc.) all require the basis to
        have been fitted first and raise `RuntimeError` otherwise.
        """
        return self._fitted

    # ------------------------------------------------------------------
    # Convenience properties (available after fit)
    # ------------------------------------------------------------------

    @property
    def d(self) -> int:
        """Covariate dimension inferred from the training data.

        Set during `fit()` from the number of columns of the (reshaped)
        training covariates `x`; used to validate that new data passed to
        `basis_matrix()` has a matching number of columns.
        """
        self._check_fitted()
        return self._d

    @property
    def null_space_dim(self) -> int:
        r"""Dimension of the unpenalized null space of the TPRS penalty.

        This is a convenience property equivalent to calling `null_space_dimension()`.
        For a thin plate regression spline of order $m$ on $d$ covariates, the null
        space has dimension $M = \binom{m + d - 1}{d}$, corresponding to the polynomial
        terms of degree less than $m$ that are left unpenalized.

        Returns
        -------
        int
            Number of null-space basis functions $M$.
        """
        return self.null_space_dimension()

    @property
    def eigenvalues(self) -> NDArray:
        """Eigenvalues of the projected thin plate spline kernel matrix.

        These are the `k - M` largest eigenvalues retained from the
        eigendecomposition performed in `fit()`; they populate the
        diagonal of the penalized block of `penalty_matrix()` and
        determine how strongly each retained spline basis function is
        penalized.

        Returns
        -------
        NDArray
            Array of shape `(k - M,)`, sorted in descending order and
            clamped to be non-negative.
        """
        self._check_fitted()
        return self._eigenvalues.copy()
