"""Abstract base class for smooth basis types."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SmoothBasis(ABC):
    """Abstract base class for all smooth basis types.

    Subclasses must implement `fit()`, `basis_matrix()`,
    `penalty_matrix()`, `null_space_dimension()`, and the
    `n_basis=` property.

    The typical workflow is::

        basis = MyBasis(k=10)
        basis.fit(x_train)
        B_train = basis.basis_matrix(x_train)   # (n_train, k)
        B_new   = basis.basis_matrix(x_new)     # (n_new,  k)
        S       = basis.penalty_matrix()        # (k, k)
    """

    @abstractmethod
    def fit(self, x: NDArray) -> SmoothBasis:
        """Fit the basis to training data.

        Stores any parameters required for later calls to
        `basis_matrix()` (e.g. knot locations, transformation matrices).

        Parameters
        ----------
        x:
            Training covariates. Shape `(n,)` for univariate smooths or
            `(n, d)` for multivariate smooths.

        Returns
        -------
        self
            Returns the instance to allow method chaining.
        """
        ...

    @abstractmethod
    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate basis functions at `x`.

        Parameters
        ----------
        x:
            Covariate values. Shape `(n,)` for univariate or `(n, d)`
            for multivariate.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)` where `k = self.n_basis`.
        """
        ...

    @abstractmethod
    def penalty_matrix(self) -> NDArray:
        """Return the `(k, k)` penalty matrix S.

        The total penalty contribution for smoothing parameter λ is
        `λ * β @ S @ β`.
        """
        ...

    @abstractmethod
    def null_space_dimension(self) -> int:
        """Return the dimension M of the penalty null space.

        The first M columns of the basis matrix correspond to the
        unpenalized polynomial null space.
        """
        ...

    def identifiability_constraints(self) -> NDArray | None:
        """Return constraint matrix C such that `C @ β = 0`, or `None`.

        Subclasses may override this to supply sum-to-zero or other
        identifiability constraints for use during model fitting.
        """
        return None

    @property
    @abstractmethod
    def n_basis(self) -> int:
        """Number of basis functions k."""
        ...

    @property
    def is_fitted(self) -> bool:
        """`True` after `fit()` has been called."""
        return bool(getattr(self, "_fitted", False))

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise `RuntimeError` if the basis has not been fitted yet."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling this method. "
                "Call fit(x) first."
            )

    @staticmethod
    def _as_2d(x: NDArray) -> NDArray:
        """Return `x` reshaped to `(n, d)` with `d ≥ 1`."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x[:, np.newaxis]
        if x.ndim == 2:
            return x
        raise ValueError(f"Expected 1-D or 2-D array for covariate x, got {x.ndim}-D array.")
