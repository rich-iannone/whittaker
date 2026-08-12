"""Duchon splines: a generalisation of thin plate regression splines.

Duchon splines (Duchon, 1977) extend TPRS by decoupling the radial basis exponent from the covariate
dimension. The standard TPRS uses a radial basis with exponent `2m - d` (where `m` is the derivative
order and `d` is the covariate dimension). Duchon splines replace this with a separate parameter `s`
so the exponent is `2s`, giving finer control over smoothness.

In mgcv notation, `m = c(s, m)` where `s ≥ 0` is a real-valued order parameter and `m` is the
null-space (polynomial) order. When `s = m - d/2` and `m` is integer, this recovers the standard
TPRS.

Usage in a formula:

    s(x, bs="ds")                 # defaults: s=1, m=2
    s(x, bs="ds", m=[0.5, 1])     # s=0.5, null-space order=1
    s(x1, x2, bs="ds", m=[1, 2])  # 2-D Duchon spline
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.tprs import _polynomial_null_space


def _duchon_radial_basis(r: NDArray, s: float) -> NDArray:
    """Evaluate the Duchon radial basis η_s(r).

    * If `2s` is **not** an even integer: `η(r) = r^(2s)`
    * If `2s` is an **even** integer: `η(r) = r^(2s) · log(r)`, with `η(0) = 0`
    """
    power = 2.0 * s
    two_s_int = round(power)
    is_even_int = abs(power - two_s_int) < 1e-10 and two_s_int % 2 == 0

    if is_even_int:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r == 0.0, 0.0, r**power * np.log(r))
    else:
        return r**power


def _duchon_kernel_matrix(x1: NDArray, x2: NDArray, s: float) -> NDArray:
    diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    return _duchon_radial_basis(r, s)


class DuchonSpline(SmoothBasis):
    r"""Duchon spline basis.

    Duchon splines (Duchon, 1977) generalize thin plate splines by decoupling the radial basis
    exponent from the covariate dimension. Ordinary TPRS ties the exponent of its radial kernel to
    both the derivative order `m` being penalized and the covariate dimension `d` (exponent
    `2m - d`), which means that for high-dimensional covariates the derivative order actually
    penalized can end up being uncomfortably high just to keep the kernel well-defined. `DuchonSpline`
    introduces an independent exponent parameter `s`, so the radial kernel and the polynomial
    null-space order can be chosen separately. Like TPRS, it is built as a low-rank
    eigen-approximation to the full spline (Wood 2003's construction, generalized to the Duchon
    kernel), so it requires no knot placement and works for any covariate dimension. Choose
    `DuchonSpline` over `TPRS` when you want explicit control over the smoothness/exponent trade-off
    independent of dimension — for example to match a specific derivative-penalty order in higher
    dimensions without also inflating the null-space order.

    Parameters
    ----------
    k:
        Total number of basis functions, including the `M` polynomial null-space columns. Must
        satisfy `k > M` where `M = C(m_order - 1 + d, d)`. Larger `k` allows more wiggly fits at the
        cost of more computation; the roughness penalty (not `k`) ultimately controls smoothness once
        `lambda` is chosen. The default is `10`.
    m:
        Order specification, either:

        * a single integer, interpreted as the polynomial null-space order `m_order`, with the
          radial exponent parameter `s` defaulting to `1.0`; or
        * a two-element list/tuple `[s, m_order]`, where `s >= 0` is the real-valued radial basis
          exponent (the kernel behaves like `r^(2s)`, optionally with a log factor) and
          `m_order >= 1` is the polynomial null-space order (the null space consists of all
          monomials of total degree `<= m_order - 1`).

        Setting `s = m_order - d / 2` for integer `m_order` recovers the ordinary TPRS basis for
        that order. The default is `2` (i.e. `s=1.0`, `m_order=2`).

    Notes
    -----
    The Duchon radial kernel is

    $$
    \eta_s(r) = \begin{cases} r^{2s} & 2s \text{ is not an even integer} \\ r^{2s} \log(r) & 2s \text{ is an even integer} \end{cases},
    $$

    evaluated at pairwise distances `r = ||x_i - x_j||`, with the convention `η(0) = 0`. Together
    with the polynomial null space of all monomials of total degree at most `m_order - 1`
    (dimension `M = C(m_order - 1 + d, d)`), the basis is constructed in the same two stages as
    `TPRS`:

    1. **Polynomial null space** (first `M` columns): unpenalized low-degree polynomials.
    2. **Truncated spline part** (remaining `k - M` columns): the full `n x n` kernel matrix is
       projected onto the orthogonal complement of the null space (via a QR decomposition of the
       null-space design matrix) and eigendecomposed; the `k - M` leading eigenvectors give the
       best rank-`(k - M)` approximation to the full Duchon spline for that basis dimension.

    The resulting penalty matrix is block-diagonal,

    $$
    \mathbf{S} = \operatorname{diag}(0, \ldots, 0, \lambda_1, \ldots, \lambda_{k-M}),
    $$

    with the first `M` rows/columns exactly zero (unpenalized null space) and the remainder equal
    to the retained eigenvalues of the projected kernel matrix. As with `TPRS`, columns of `x` with
    very different scales can cause numerical issues in the eigendecomposition, so centering and/or
    standardizing covariates before fitting is advisable. Non-integer or large `s` values can also
    make the kernel matrix increasingly ill-conditioned; if fitting becomes numerically unstable,
    try a smaller `s` or standardized covariates.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import DuchonSpline

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)

    basis = DuchonSpline(k=10, m=[1.0, 2]).fit(x)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
    """

    def __init__(self, k: int = 10, m: int | list | tuple = 2) -> None:
        if isinstance(m, (list, tuple)):
            if len(m) != 2:
                raise ValueError(
                    f"m must be an integer or a two-element [s, m_order] list, got length {len(m)}."
                )
            self._s = float(m[0])
            self._m_order = int(m[1])
        else:
            self._s = 1.0
            self._m_order = int(m)

        if self._s < 0:
            raise ValueError(f"Duchon s parameter must be ≥ 0, got {self._s}.")
        if self._m_order < 1:
            raise ValueError(f"Null-space order must be ≥ 1, got {self._m_order}.")
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}.")

        self.k = k
        self._fitted = False

        self._d: int
        self._M: int
        self._x_train: NDArray
        self._QU: NDArray
        self._eigenvalues: NDArray

    def fit(self, x: NDArray) -> DuchonSpline:
        """Fit the Duchon spline to training data `x`.

        Parameters
        ----------
        x:
            Training covariates. Shape `(n,)` for univariate or `(n, d)` for multivariate.

        Returns
        -------
        DuchonSpline
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If `k` is too small for the null-space dimension `M`, or too large for the number of
            observations.
        """
        x2d = self._as_2d(x)
        n, d = x2d.shape

        M = comb(self._m_order - 1 + d, d)
        r = self.k - M

        if r < 1:
            raise ValueError(
                f"k={self.k} is too small for d={d}, m_order={self._m_order}: "
                f"need k > {M} (null-space dimension)."
            )
        if n < self.k:
            raise ValueError(f"n={n} < k={self.k}. Reduce k to at most {n - 1}.")

        self._d = d
        self._M = M
        self._x_train = x2d.copy()

        E = _duchon_kernel_matrix(x2d, x2d, self._s)
        E = (E + E.T) * 0.5

        T = _polynomial_null_space(x2d, m=self._m_order)

        Q_full, _ = np.linalg.qr(T, mode="complete")
        Q2 = Q_full[:, M:]

        G = Q2.T @ E @ Q2
        G = (G + G.T) * 0.5

        eigenvalues, eigenvectors = np.linalg.eigh(G)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        U_r = eigenvectors[:, :r]
        D_r = np.maximum(eigenvalues[:r], 0.0)

        self._QU = Q2 @ U_r
        self._eigenvalues = D_r

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the Duchon spline basis at `x`.

        Parameters
        ----------
        x:
            Covariate values. Shape `(n,)` or `(n, d)` where `d` must match the training dimension.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)`. The first `M` columns are the polynomial null-space
            functions; the remaining `k - M` columns are the truncated spline functions.

        Raises
        ------
        ValueError
            If the covariate dimension of `x` does not match the training dimension.
        """
        self._check_fitted()
        x2d = self._as_2d(x)

        if x2d.shape[1] != self._d:
            raise ValueError(f"Expected {self._d} covariate(s), got {x2d.shape[1]}.")

        T_new = _polynomial_null_space(x2d, m=self._m_order)
        E_new = _duchon_kernel_matrix(x2d, self._x_train, self._s)
        spline_cols = E_new @ self._QU

        return np.column_stack([T_new, spline_cols])

    def penalty_matrix(self) -> NDArray:
        """Return the `k x k` penalty matrix `S`.

        `S` is block-diagonal: the first `M` rows/columns (the polynomial null space) are exactly
        zero, and the remaining `k - M` rows/columns hold the retained eigenvalues of the projected
        Duchon kernel matrix on the diagonal.

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
        """Return `M`, the dimension of the polynomial null space.

        Returns
        -------
        int
            `M = C(m_order - 1 + d, d)`, the number of monomials of total degree at most
            `m_order - 1` in `d` variables.
        """
        self._check_fitted()
        return self._M

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Returns
        -------
        NDArray
            A `(1, k)` matrix whose product with the coefficient vector is zero when the smooth has
            mean zero over the training data.
        """
        self._check_fitted()
        B_train = self.basis_matrix(self._x_train)
        return B_train.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        """Total number of basis functions.

        Equal to `k`: the `M` polynomial null-space columns plus the
        `k - M` truncated-spline columns retained from the eigendecomposition
        of the projected Duchon kernel matrix.

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
            `True` once `fit()` has been called and the null-space basis,
            eigenvectors, and eigenvalues have been computed; `False`
            otherwise.
        """
        return self._fitted

    def __repr__(self) -> str:
        return f"DuchonSpline(k={self.k}, s={self._s}, m={self._m_order})"
