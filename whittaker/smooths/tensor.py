"""Tensor product smooth bases (te / ti / t2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis

_EIG_TOL = 1e-10


def _row_tensor_product(B1: NDArray, B2: NDArray) -> NDArray:
    """Row-wise Kronecker product of two basis matrices.

    Given B1 (n, k1) and B2 (n, k2), returns (n, k1*k2) where
    result[i, j1*k2 + j2] = B1[i, j1] * B2[i, j2].
    """
    n, k1 = B1.shape
    k2 = B2.shape[1]
    return (B1[:, :, np.newaxis] * B2[:, np.newaxis, :]).reshape(n, k1 * k2)


def _null_range_decompose(S: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Decompose a penalty into its null-space and range-space projections.

    Returns (U_range, eigenvalues_range, U_null) where:

    - U_range: eigenvectors spanning the penalized (range) space
    - eigenvalues_range: corresponding positive eigenvalues
    - U_null: eigenvectors spanning the penalty null space
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    null_mask = eigvals < _EIG_TOL * max(eigvals.max(), 1.0)
    return eigvecs[:, ~null_mask], eigvals[~null_mask], eigvecs[:, null_mask]


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

        For proper per-marginal penalization, use `penalty_matrices()` instead.
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


class TensorProductBasisT2(TensorProductBasis):
    """Tensor product basis with full penalty decomposition (t2).

    Uses the same row-wise Kronecker product basis as `te()`, but generates a richer penalty
    structure: one penalty for every non-empty subset of marginals. For *d* marginals this gives
    `2^d - 1` penalties (compared to *d* for `te()`), each with its own smoothing parameter.

    For 2 marginals with penalties S_1, S_2 and dimensions k_1, k_2::

        `te()` penalties:  S_1 ⊗ I,  I ⊗ S_2                       (2 penalties)
        `t2()` penalties:  S_1 ⊗ I,  I ⊗ S_2,  S_1 ⊗ S_2           (3 penalties)

    This gives each interaction order its own smoothing parameter, which can improve estimation when
    the interaction and main-effect smoothness differ substantially.
    """

    def penalty_matrices(self) -> list[NDArray]:
        """Return one penalty per non-empty subset of marginal directions."""
        self._check_fitted()
        d = len(self._marginals)
        dims = [m.n_basis for m in self._marginals]
        marginal_penalties = [m.penalty_matrix() for m in self._marginals]

        penalties = []
        for mask in range(1, 1 << d):
            P = None
            for j in range(d):
                if mask & (1 << j):
                    M_j = marginal_penalties[j]
                else:
                    M_j = np.eye(dims[j])
                P = M_j if P is None else np.kron(P, M_j)
            penalties.append(P)
        return penalties


class TensorInteractionBasis(SmoothBasis):
    """Tensor product interaction basis (ti).

    Like `TensorProductBasis` but with each marginal's penalty null space removed, so the basis
    spans only the pure interaction: no main effects. This allows ANOVA-style decompositions such as
    `s(x1) + s(x2) + ti(x1, x2)`.

    Parameters
    ----------
    marginals:
        List of marginal basis objects (unfitted).
    """

    def __init__(self, marginals: list[SmoothBasis]) -> None:
        if len(marginals) < 2:
            raise ValueError("TensorInteractionBasis requires at least 2 marginals.")
        self._marginals = marginals
        self._fitted = False
        self._range_projections: list[NDArray] = []
        self._range_eigenvalues: list[NDArray] = []
        self._range_dims: list[int] = []

    def fit(self, x: NDArray) -> TensorInteractionBasis:
        x = self._as_2d(x)
        if x.shape[1] != len(self._marginals):
            raise ValueError(f"Expected {len(self._marginals)} columns, got {x.shape[1]}.")
        for j, basis in enumerate(self._marginals):
            basis.fit(x[:, j])

        self._range_projections = []
        self._range_eigenvalues = []
        self._range_dims = []
        for basis in self._marginals:
            S = basis.penalty_matrix()
            U_range, eig_range, _ = _null_range_decompose(S)
            self._range_projections.append(U_range)
            self._range_eigenvalues.append(eig_range)
            self._range_dims.append(U_range.shape[1])

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        self._check_fitted()
        x = self._as_2d(x)
        B = self._marginals[0].basis_matrix(x[:, 0]) @ self._range_projections[0]
        for j in range(1, len(self._marginals)):
            Bj = self._marginals[j].basis_matrix(x[:, j]) @ self._range_projections[j]
            B = _row_tensor_product(B, Bj)
        return B

    def penalty_matrix(self) -> NDArray:
        pens = self.penalty_matrices()
        return sum(pens)

    def penalty_matrices(self) -> list[NDArray]:
        """One penalty per marginal direction in the range-space basis."""
        self._check_fitted()
        penalties = []
        for j in range(len(self._marginals)):
            S_j = np.diag(self._range_eigenvalues[j])
            P = S_j
            for i in range(j - 1, -1, -1):
                P = np.kron(np.eye(self._range_dims[i]), P)
            for i in range(j + 1, len(self._marginals)):
                P = np.kron(P, np.eye(self._range_dims[i]))
            penalties.append(P)
        return penalties

    def null_space_dimension(self) -> int:
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        self._check_fitted()
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
        self._check_fitted()
        result = 1
        for d in self._range_dims:
            result *= d
        return result

    @property
    def marginals(self) -> list[SmoothBasis]:
        return self._marginals
