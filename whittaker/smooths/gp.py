"""Gaussian process smooth basis.

Implements smooths based on Gaussian process covariance functions. The basis is constructed from
the leading eigenfunctions of the covariance matrix evaluated at training knot locations.

Supported covariance functions (selected via the ``cov`` parameter):

* `"exp"`: exponential (Matérn ν=½): `σ² exp(-r/ρ)`
* `"matern32"`: Matérn ν=3/2: `σ² (1 + √3 r/ρ) exp(-√3 r/ρ)`
* `"matern52"`: Matérn ν=5/2: `σ² (1 + √5 r/ρ + 5r²/(3ρ²)) exp(-√5 r/ρ)`
* `"sqexp"`: squared exponential (RBF): `σ² exp(-r²/(2ρ²))`

Usage in a formula::

    s(x, bs="gp")
    s(x1, x2, bs="gp", xt="matern32")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis

_EPS = np.finfo(float).eps

_COV_FUNCTIONS = {"exp", "matern32", "matern52", "sqexp"}


def _pairwise_distances(x1: NDArray, x2: NDArray) -> NDArray:
    diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1) + _EPS)


def _cov_matrix(x1: NDArray, x2: NDArray, cov: str, rho: float) -> NDArray:
    r = _pairwise_distances(x1, x2)
    scaled = r / rho

    if cov == "exp":
        return np.exp(-scaled)
    elif cov == "matern32":
        s3 = np.sqrt(3.0) * scaled
        return (1.0 + s3) * np.exp(-s3)
    elif cov == "matern52":
        s5 = np.sqrt(5.0) * scaled
        return (1.0 + s5 + s5**2 / 3.0) * np.exp(-s5)
    elif cov == "sqexp":
        return np.exp(-0.5 * scaled**2)
    else:
        raise ValueError(
            f"Unknown covariance function {cov!r}. Supported: {sorted(_COV_FUNCTIONS)}"
        )


class GaussianProcess(SmoothBasis):
    """Gaussian process smooth basis.

    Parameters
    ----------
    k:
        Number of basis functions (leading eigenfunctions of the covariance matrix).
    cov:
        Covariance function. One of `"exp"`, `"matern32"`, `"matern52"`, or `"sqexp"`.
    """

    def __init__(self, k: int = 10, cov: str = "matern32") -> None:
        if cov not in _COV_FUNCTIONS:
            raise ValueError(f"Unknown covariance {cov!r}. Supported: {sorted(_COV_FUNCTIONS)}")
        self._k = k
        self._cov = cov
        self._fitted = False

        self._d: int = 0
        self._x_train: NDArray
        self._rho: float = 1.0
        self._U: NDArray  # (n_train, k) eigenvectors
        self._eigenvalues: NDArray  # (k,) eigenvalues

    def fit(self, x: NDArray) -> GaussianProcess:
        x2d = self._as_2d(x)
        n, d = x2d.shape

        if n < self._k:
            raise ValueError(f"n={n} < k={self._k}. Reduce k to at most {n - 1}.")

        self._d = d
        self._x_train = x2d.copy()

        ranges = x2d.max(axis=0) - x2d.min(axis=0)
        self._rho = float(np.mean(ranges[ranges > _EPS])) / 4.0
        if self._rho < _EPS:
            self._rho = 1.0

        C = _cov_matrix(x2d, x2d, self._cov, self._rho)
        C = (C + C.T) * 0.5

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k = min(self._k, n)
        self._U = eigenvectors[:, :k]
        self._eigenvalues = np.maximum(eigenvalues[:k], _EPS)
        self._k = k

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        self._check_fitted()
        x2d = self._as_2d(x)

        if x2d.shape[1] != self._d:
            raise ValueError(f"Expected {self._d} covariate(s), got {x2d.shape[1]}.")

        C_new = _cov_matrix(x2d, self._x_train, self._cov, self._rho)
        B = C_new @ self._U / self._eigenvalues[np.newaxis, :]
        return B

    def penalty_matrix(self) -> NDArray:
        self._check_fitted()
        return np.diag(1.0 / self._eigenvalues)

    def null_space_dimension(self) -> int:
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        self._check_fitted()
        B_train = self.basis_matrix(self._x_train)
        return B_train.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        return self._k

    def __repr__(self) -> str:
        return f"GaussianProcess(k={self._k}, cov={self._cov!r})"
