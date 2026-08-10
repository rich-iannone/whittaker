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
    """Duchon spline basis.

    Parameters
    ----------
    k:
        Total number of basis functions (including the polynomial null space).
    m:
        Order specification. Either a single integer (interpreted as the null-space polynomial
        order, with `s` defaulting to `1.0`), or a two-element list/tuple `[s, m_order]` where
        `s ≥ 0` is the radial basis exponent parameter and `m_order ≥ 1` is the polynomial
        null-space order.
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
        self._check_fitted()
        x2d = self._as_2d(x)

        if x2d.shape[1] != self._d:
            raise ValueError(f"Expected {self._d} covariate(s), got {x2d.shape[1]}.")

        T_new = _polynomial_null_space(x2d, m=self._m_order)
        E_new = _duchon_kernel_matrix(x2d, self._x_train, self._s)
        spline_cols = E_new @ self._QU

        return np.column_stack([T_new, spline_cols])

    def penalty_matrix(self) -> NDArray:
        self._check_fitted()
        S = np.zeros((self.k, self.k))
        M = self._M
        S[M:, M:] = np.diag(self._eigenvalues)
        return S

    def null_space_dimension(self) -> int:
        self._check_fitted()
        return self._M

    def identifiability_constraints(self) -> NDArray | None:
        self._check_fitted()
        B_train = self.basis_matrix(self._x_train)
        return B_train.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        return self.k

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        return f"DuchonSpline(k={self.k}, s={self._s}, m={self._m_order})"
