r"""Markov random field smooth basis (bs="mrf").

Implements a spatial smooth for areal/lattice data where the basis is an indicator per region and
the penalty is the graph Laplacian of the neighborhood structure. The quadratic form
beta' L beta = sum_{(i,j) neighbors} (beta_i - beta_j)^2 penalizes differences between neighboring
regions, inducing spatial smoothness.

The user supplies a neighborhood structure as either a dict mapping region labels to lists of
neighbor labels, or a symmetric adjacency matrix (ndarray).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis


class MRFBasis(SmoothBasis):
    r"""Markov random field basis for areal spatial data.

    A Markov random field (MRF) smooth represents spatial structure over discrete areal units —
    counties, districts, postcodes, grid cells on a lattice — where the covariate is a categorical
    label rather than a continuous coordinate, and the only spatial information available is which
    units are adjacent to which. It is equivalent to mgcv's `bs="mrf"` basis. Each unique region
    gets its own basis function (an indicator column), and smoothness across the map is enforced
    directly through the neighborhood graph rather than through any distance metric: the penalty
    discourages the fitted values of neighboring regions from differing, so choose this basis over
    `TPRS`-style continuous smooths whenever the domain is a set of discrete areas linked by an
    adjacency structure (e.g. shared borders) instead of by Euclidean coordinates.

    Parameters
    ----------
    k:
        Maximum number of regions to retain. If `-1` (the default), all observed levels of the
        grouping variable are kept as basis functions. If a positive integer smaller than the
        number of observed levels, only the first `k` (in sorted level order) are used; this is
        rarely what a user wants for MRF smooths (unlike continuous bases, reducing `k` does not
        give a lower-rank approximation of the same structure — it silently drops regions), so in
        most workflows the default of `-1` should be left alone.
    neighborhood:
        The neighborhood structure that defines which regions are considered adjacent. Either a
        `dict` mapping region labels to lists of neighbor labels (only pairs need to be listed once;
        the adjacency is symmetrized automatically), or a square, symmetric adjacency matrix
        (`ndarray`) whose row/column order matches the sorted unique levels of the fitted grouping
        variable. This argument is required — there is no sensible default neighborhood structure.

    Notes
    -----
    Let there be `k` unique regions after `fit()`. The basis matrix `B` is the `n x k` matrix of
    region indicators, `B[i, j] = 1` if observation `i` belongs to region `j` and `0` otherwise —
    identical in structure to `RandomEffectBasis`. What distinguishes an MRF smooth is its penalty.
    Writing `A` for the symmetric adjacency matrix (`A[i, j] = 1` if regions `i` and `j` are
    neighbors) and `D = \operatorname{diag}(A \mathbf{1})` for the diagonal matrix of neighbor
    counts, the penalty matrix is the graph Laplacian

    $$
    \mathbf{L} = \mathbf{D} - \mathbf{A},
    $$

    so that the roughness penalty takes the form

    $$
    \boldsymbol{\beta}^\top \mathbf{L} \boldsymbol{\beta}
    = \sum_{(i,j) \, \in \, \text{neighbors}} (\beta_i - \beta_j)^2 .
    $$

    This penalizes exactly the pairwise differences between the fitted level for each region and
    the fitted levels of its geographic neighbors, pulling adjacent regions toward a common value as
    `lambda` grows, while leaving regions that are far apart on the map free to differ. The graph
    Laplacian of a connected neighborhood graph has exactly one zero eigenvalue, with eigenvector
    proportional to the all-ones vector; `null_space_dimension()` therefore returns `1` for a fully
    connected graph (the penalty cannot shrink a common overall level, only differences between
    neighbors), but can be larger if the neighborhood graph has multiple disconnected components,
    since each component then has its own unpenalized constant. Because the raw indicator basis
    shares an unpenalized constant with the model intercept, a sum-to-zero constraint (returned by
    `identifiability_constraints()`) is needed for identifiability when fitting alongside an
    intercept term. `fit()` raises if fewer than two regions are present, since a spatial smooth is
    meaningless with only one area, and raises if no `neighborhood` is supplied.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths.mrf import MRFBasis

    rng = np.random.default_rng(0)
    regions = np.array(["A", "B", "C", "D"])
    x = rng.choice(regions, size=40)

    neighborhood = {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B", "D"],
        "D": ["C"],
    }

    basis = MRFBasis(neighborhood=neighborhood).fit(x)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
    """

    def __init__(self, k: int = -1, neighborhood: dict | NDArray | None = None) -> None:
        self._k_request = k
        self._neighborhood = neighborhood
        self._fitted: bool = False
        self._levels: NDArray
        self._level_to_idx: dict
        self._adjacency: NDArray
        self._laplacian: NDArray

    def fit(self, x: NDArray) -> MRFBasis:
        """Fit the MRF basis to training data.

        Determines the set of unique region levels present in `x`, builds the symmetric adjacency
        matrix from the supplied `neighborhood` structure, and computes the graph Laplacian penalty.

        Parameters
        ----------
        x:
            Training grouping variable (region labels). Shape `(n,)`, of any hashable dtype
            (strings, integers, etc.).

        Returns
        -------
        MRFBasis
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If fewer than 2 unique levels are present in `x`, if `neighborhood` was not supplied, if
            an adjacency matrix is provided with the wrong shape, or if `k` is requested but is less
            than 2.
        TypeError
            If `neighborhood` is neither a `dict` nor an `ndarray`.
        """
        x1d = self._validate(x)
        levels = np.unique(x1d)

        if self._k_request != -1:
            if self._k_request < 2:
                raise ValueError(f"k must be at least 2 for MRFBasis, got {self._k_request}.")
            if self._k_request < len(levels):
                levels = levels[: self._k_request]

        if len(levels) < 2:
            raise ValueError(
                "MRFBasis requires at least 2 unique levels in the grouping variable, "
                f"got {len(levels)}."
            )

        if self._neighborhood is None:
            raise ValueError(
                "MRFBasis requires a neighborhood structure. Pass neighborhood= as a dict "
                "mapping region labels to lists of neighbor labels, or as a symmetric "
                "adjacency matrix."
            )

        self._levels = levels
        self._level_to_idx = {lev: i for i, lev in enumerate(levels)}
        k = len(levels)

        if isinstance(self._neighborhood, np.ndarray):
            A = np.asarray(self._neighborhood, dtype=float)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"Adjacency matrix must be square, got shape {A.shape}.")
            if A.shape[0] != k:
                raise ValueError(
                    f"Adjacency matrix has {A.shape[0]} rows but data has {k} unique levels."
                )
            A = (A + A.T) / 2.0
            np.fill_diagonal(A, 0.0)
        elif isinstance(self._neighborhood, dict):
            A = np.zeros((k, k))
            for region, neighbors in self._neighborhood.items():
                if region not in self._level_to_idx:
                    continue
                i = self._level_to_idx[region]
                for nb in neighbors:
                    if nb not in self._level_to_idx:
                        continue
                    j = self._level_to_idx[nb]
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        else:
            raise TypeError(
                f"neighborhood must be a dict or ndarray, got {type(self._neighborhood).__name__}."
            )

        self._adjacency = A
        D = np.diag(A.sum(axis=1))
        L = D - A
        self._laplacian = (L + L.T) / 2.0

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the MRF indicator basis at `x`.

        Parameters
        ----------
        x:
            Grouping variable values. Shape `(n,)`. Values not seen during `fit()` produce an
            all-zero row (no region indicator is set).

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)` where `k` is the number of region levels retained
            during `fit()`. Each row has at most one entry equal to `1.0`.
        """
        self._check_fitted()
        x1d = self._validate(x)
        n = len(x1d)
        k = len(self._levels)
        B = np.zeros((n, k))
        for i, val in enumerate(x1d):
            idx = self._level_to_idx.get(val)
            if idx is not None:
                B[i, idx] = 1.0
        return B

    def penalty_matrix(self) -> NDArray:
        """Return the `(k, k)` graph Laplacian penalty matrix `L = D - A`.

        Returns
        -------
        NDArray
            Symmetric positive semi-definite matrix of shape `(k, k)`, where `k` is the number of
            region levels.
        """
        self._check_fitted()
        return self._laplacian.copy()

    def null_space_dimension(self) -> int:
        """Return the number of zero eigenvalues of the graph Laplacian.

        This equals the number of connected components of the neighborhood graph: `1` for a fully
        connected map, and more than `1` if some regions have no path of neighbors linking them to
        the rest, since each disconnected component then carries its own unpenalized constant.

        Returns
        -------
        int
            Dimension of the penalty null space.
        """
        self._check_fitted()
        eigvals = np.linalg.eigvalsh(self._laplacian)
        threshold = 1e-10 * max(eigvals.max(), 1.0)
        return int(np.sum(eigvals < threshold))

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Returns
        -------
        NDArray
            A `(1, k)` matrix of equal weights `1 / k` whose product with the coefficient vector
            forces the mean fitted region effect to be zero, resolving the confound between the MRF
            smooth's unpenalized constant and the model intercept.
        """
        self._check_fitted()
        k = len(self._levels)
        return np.ones((1, k)) / k

    @property
    def n_basis(self) -> int:
        """Number of basis functions, i.e. the number of retained region levels."""
        if not self._fitted:
            raise RuntimeError("n_basis is not available until fit() is called.")
        return len(self._levels)

    @property
    def k(self) -> int:
        """Requested (before `fit()`) or actual (after `fit()`) number of region levels."""
        if self._fitted:
            return len(self._levels)
        return self._k_request

    @property
    def levels(self) -> NDArray:
        """Sorted array of unique region labels retained during `fit()`."""
        self._check_fitted()
        return self._levels.copy()

    @staticmethod
    def _validate(x: NDArray) -> NDArray:
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError(
                f"MRFBasis requires a 1-D grouping variable, got array with shape {x.shape}."
            )
        return x
