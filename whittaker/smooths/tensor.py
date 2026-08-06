"""Tensor product smooth basis (te)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis


def _row_tensor_product(B1: NDArray, B2: NDArray) -> NDArray:
    """Row-wise Kronecker product of two basis matrices.

    Given B1 (n, k1) and B2 (n, k2), returns (n, k1*k2) where
    result[i, j1*k2 + j2] = B1[i, j1] * B2[i, j2].
    """
    n, k1 = B1.shape
    k2 = B2.shape[1]
    return (B1[:, :, np.newaxis] * B2[:, np.newaxis, :]).reshape(n, k1 * k2)


class TensorProductBasis(SmoothBasis):
    """Tensor product of marginal smooth bases.

    Parameters
    ----------
    marginals:
        List of fitted or unfitted marginal basis objects.
    """

    def __init__(self, marginals: list[SmoothBasis]) -> None:
        if len(marginals) < 2:
            raise ValueError("TensorProductBasis requires at least 2 marginals.")
        self._marginals = marginals
        self._fitted = False

    def fit(self, x: NDArray) -> TensorProductBasis:
        x = self._as_2d(x)
        if x.shape[1] != len(self._marginals):
            raise ValueError(f"Expected {len(self._marginals)} columns, got {x.shape[1]}.")
        for j, basis in enumerate(self._marginals):
            basis.fit(x[:, j])
        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        self._check_fitted()
        x = self._as_2d(x)
        B = self._marginals[0].basis_matrix(x[:, 0])
        for j in range(1, len(self._marginals)):
            B = _row_tensor_product(B, self._marginals[j].basis_matrix(x[:, j]))
        return B

    def penalty_matrix(self) -> NDArray:
        """Sum of all marginal penalty matrices (for compatibility).

        For proper per-marginal penalization, use ``penalty_matrices()`` instead.
        """
        pens = self.penalty_matrices()
        return sum(pens)

    def penalty_matrices(self) -> list[NDArray]:
        """Return one penalty matrix per marginal direction.

        For d marginals with dimensions k_1, ..., k_d, penalty j is:
        I_{k_1} ⊗ ... ⊗ S_j ⊗ ... ⊗ I_{k_d}
        """
        self._check_fitted()
        dims = [m.n_basis for m in self._marginals]
        penalties = []
        for j, marginal in enumerate(self._marginals):
            S_j = marginal.penalty_matrix()
            P = S_j
            for i in range(j - 1, -1, -1):
                P = np.kron(np.eye(dims[i]), P)
            for i in range(j + 1, len(self._marginals)):
                P = np.kron(P, np.eye(dims[i]))
            penalties.append(P)
        return penalties

    def null_space_dimension(self) -> int:
        """Product of marginal null-space dimensions."""
        nsd = 1
        for m in self._marginals:
            nsd *= m.null_space_dimension()
        return nsd

    def identifiability_constraints(self) -> NDArray | None:
        """Sum-to-zero constraint on the tensor product basis."""
        self._check_fitted()
        n_train = None
        for m in self._marginals:
            if hasattr(m, "_x_train"):
                n_train = len(m._x_train)
                break
            if hasattr(m, "_knots"):
                n_train = len(m._knots)
                break

        if n_train is None:
            return None

        xs = []
        for m in self._marginals:
            if hasattr(m, "_x_train"):
                xs.append(m._x_train.ravel())
            elif hasattr(m, "_knots"):
                xs.append(m._knots.ravel())
            else:
                return None

        x_grid = np.column_stack(xs)
        B = self.basis_matrix(x_grid)
        return B.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        result = 1
        for m in self._marginals:
            result *= m.n_basis
        return result

    @property
    def marginals(self) -> list[SmoothBasis]:
        return self._marginals
