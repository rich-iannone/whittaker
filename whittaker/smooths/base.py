r"""Abstract base class for smooth basis types."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SmoothBasis(ABC):
    r"""Abstract base class for all smooth basis types.

    Every smooth term in a GAM — a thin plate spline, a cubic regression spline, a P-spline, a
    Gaussian process smooth, and so on — is represented internally as a linear basis expansion
    `f(x) = B(x) @ beta` together with a quadratic roughness penalty `beta.T @ S @ beta`. This
    class fixes the contract that every basis type must satisfy so that the fitting machinery
    (penalized least squares, smoothing-parameter selection, prediction) can treat all basis
    types uniformly. Concrete subclasses differ only in how `B` and `S` are constructed, not in
    how they are consumed.

    Subclasses must implement `fit()`, `basis_matrix()`, `penalty_matrix()`,
    `null_space_dimension()`, and the `n_basis` property.

    The typical workflow is::

        basis = MyBasis(k=10)
        basis.fit(x_train)
        B_train = basis.basis_matrix(x_train)   # (n_train, k)
        B_new   = basis.basis_matrix(x_new)     # (n_new,  k)
        S       = basis.penalty_matrix()        # (k, k)

    Notes
    -----
    A penalized basis decomposes its `k` basis functions into an unpenalized *null space* of
    dimension `M` (typically low-order polynomials, for which the penalty contributes nothing —
    e.g. a straight line has zero roughness under a second-derivative penalty) and a penalized
    *range space* of dimension `k - M`. The total penalty for smoothing parameter `lambda` is

    $$
    \lambda \, \boldsymbol{\beta}^\top \mathbf{S} \boldsymbol{\beta},
    $$

    where `S` is symmetric positive semi-definite with `null_space_dimension()` zero eigenvalues.
    Basis types built on this contract may additionally supply identifiability constraints (via
    `identifiability_constraints()`) when the raw basis is not full rank on its own — for example
    when several smooth terms share an intercept and must each be constrained to have mean zero
    over the training data.

    Examples
    --------
    Any concrete subclass follows this pattern:

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
        r"""Return the `(k, k)` roughness penalty matrix `S`.

        `S` encodes the quadratic wiggliness penalty applied to the basis
        coefficients: the total penalty contribution for smoothing
        parameter `lambda` is `lambda * beta.T @ S @ beta`. `S` is always
        symmetric positive semi-definite, with `null_space_dimension()`
        zero eigenvalues corresponding to the unpenalized functions (e.g.
        low-order polynomials) that the basis can represent for free.

        Returns
        -------
        NDArray
            Symmetric positive semi-definite matrix of shape `(k, k)`
            where `k = self.n_basis`.
        """
        ...

    @abstractmethod
    def null_space_dimension(self) -> int:
        """Return the dimension `M` of the penalty null space.

        The null space is the subspace of basis coefficients for which
        the roughness penalty is exactly zero — typically the low-order
        polynomials (e.g. constants and straight lines under a
        second-derivative penalty). By convention the first `M` columns of
        the basis matrix span this null space, so `S[:M, :M]` and the
        corresponding rows/columns of `S` involving those columns vanish.

        Returns
        -------
        int
            The null space dimension `M`, satisfying `0 <= M < k`.
        """
        ...

    def identifiability_constraints(self) -> NDArray | None:
        """Return a constraint matrix `C` such that `C @ beta = 0`, or `None`.

        Some bases are not identifiable on their own once combined with
        other terms in a model (for example, several smooths sharing an
        intercept), and must be constrained — typically to have mean zero
        over the training data — before fitting. Subclasses may override
        this method to supply such sum-to-zero or other linear
        identifiability constraints; the default implementation returns
        `None`, meaning no constraint is required.

        Returns
        -------
        NDArray or None
            Constraint matrix of shape `(n_constraints, k)`, or `None` if
            the basis needs no identifiability constraint.
        """
        return None

    @property
    @abstractmethod
    def n_basis(self) -> int:
        """Number of basis functions `k`.

        This is the number of columns returned by `basis_matrix()` and the
        size of the (square) `penalty_matrix()`. It is fixed at
        construction time (e.g. via a `k` argument) and does not change
        after `fit()` is called.
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """`True` after `fit()` has been called.

        Used internally (via `_check_fitted()`) to guard methods such as
        `basis_matrix()` and `penalty_matrix()` that depend on state
        computed during `fit()` (e.g. knot locations or transformation
        matrices), raising a clear error instead of failing on missing
        attributes when called out of order.
        """
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
