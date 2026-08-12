r"""Random effect smooth basis (bs="re").

Implements random intercepts as a GAM smooth term. Each level of a grouping factor gets its own
basis function (one-hot encoded), penalized by an identity matrix. This is equivalent to a random
intercept in a mixed model: the penalty λI shrinks all group deviations toward zero, with λ selected
by GCV or REML.

Since the penalty is a full-rank identity, the null space dimension is 0 — every coefficient is
penalized, so the entire random effect can be shrunk to zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis


class RandomEffectBasis(SmoothBasis):
    r"""Random effect basis (one-hot encoding with identity penalty).

    A random effect basis represents a grouping factor — subject ID, site, batch, cluster — as a
    penalized random intercept rather than as a fixed factor with one unconstrained parameter per
    level. It is equivalent to mgcv's `bs="re"` basis and to the random-intercept term of a linear
    mixed model: each unique level of the grouping variable gets its own one-hot column, and an
    identity penalty shrinks the estimated level effects toward their common mean, with the amount
    of shrinkage controlled by a single smoothing parameter selected by GCV or REML (equivalent to
    the group-level variance in a mixed model). Choose this basis over an ordinary fixed factor
    whenever the number of levels is large, levels have unbalanced sample sizes, or the goal is to
    borrow strength across levels rather than to estimate every level's effect independently; choose
    it over `MRFBasis` or `FactorSmoothBasis` when the grouping levels have no spatial/ordering
    structure to exploit and only an exchangeable random intercept is wanted.

    The covariate should be a 1-D array of group labels (strings, integers, or any hashable type).

    Parameters
    ----------
    k:
        Maximum number of levels to retain. If `-1` (the default), all observed levels are kept as
        basis functions. If a positive integer smaller than the number of observed levels, only the
        first `k` (in sorted level order) are used and any other levels are silently dropped from
        the basis (their rows become all-zero); this is rarely desirable for a random effect, so the
        default of `-1` should normally be left as-is unless there is a specific reason to cap the
        number of levels.

    Notes
    -----
    Let there be `k` unique levels of the grouping factor after `fit()`. The basis matrix `B` is
    the `n x k` matrix of level indicators, `B[i, j] = 1` if observation `i` belongs to level `j`
    and `0` otherwise. The penalty matrix is the `k x k` identity,

    $$
    \mathbf{S} = \mathbf{I}_k,
    $$

    so the roughness penalty is simply the sum of squared level effects,

    $$
    \boldsymbol{\beta}^\top \mathbf{S} \boldsymbol{\beta} = \sum_{j=1}^{k} \beta_j^2 ,
    $$

    exactly the ridge-type penalty that shrinks every group deviation toward zero as `lambda`
    grows, with no level treated differently from any other. Because the identity matrix is full
    rank, `null_space_dimension()` is always `0` — there is no unpenalized subspace, and in
    principle the entire random effect can be shrunk away as `lambda \to \infty`, recovering a model
    with no group-level variation at all. Because the raw indicator columns are collinear with an
    overall intercept (every row sums to `1`), `identifiability_constraints()` returns a sum-to-zero
    constraint that should be enforced when the random effect is fit alongside a fixed intercept
    term. `fit()` raises `ValueError` if fewer than two unique levels are present, since a random
    effect with a single level carries no information about between-group variation.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths.random import RandomEffectBasis

    rng = np.random.default_rng(0)
    groups = rng.choice(["a", "b", "c", "d"], size=50)

    basis = RandomEffectBasis().fit(groups)
    B = basis.basis_matrix(groups)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
    """

    def __init__(self, k: int = -1) -> None:
        self._k_request = k
        self._fitted: bool = False
        self._levels: NDArray
        self._level_to_idx: dict

    def fit(self, x: NDArray) -> RandomEffectBasis:
        """Fit the random effect basis to training data.

        Determines the set of unique group levels present in `x`; these define the columns of the
        basis matrix produced by `basis_matrix()`.

        Parameters
        ----------
        x:
            Training grouping variable. Shape `(n,)`, of any hashable dtype (strings, integers,
            etc.).

        Returns
        -------
        RandomEffectBasis
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If fewer than 2 unique levels are present in `x`, or if `k` is requested but is less
            than 2.
        """
        x1d = self._validate(x)
        levels = np.unique(x1d)

        if self._k_request != -1:
            if self._k_request < 2:
                raise ValueError(
                    f"k must be at least 2 for RandomEffectBasis, got {self._k_request}."
                )
            if self._k_request < len(levels):
                levels = levels[: self._k_request]

        if len(levels) < 2:
            raise ValueError(
                "RandomEffectBasis requires at least 2 unique levels in the grouping variable, "
                f"got {len(levels)}."
            )

        self._levels = levels
        self._level_to_idx = {lev: i for i, lev in enumerate(levels)}
        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the one-hot random effect basis at `x`.

        Parameters
        ----------
        x:
            Grouping variable values. Shape `(n,)`. Values not seen during `fit()` produce an
            all-zero row (no level indicator is set).

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)` where `k` is the number of levels retained during
            `fit()`. Each row has at most one entry equal to `1.0`.
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
        """Return the `(k, k)` identity penalty matrix.

        Returns
        -------
        NDArray
            The `k x k` identity matrix, where `k` is the number of retained levels. Every
            coefficient is penalized equally.
        """
        self._check_fitted()
        return np.eye(len(self._levels))

    def null_space_dimension(self) -> int:
        """Return `0`: the identity penalty has no unpenalized subspace.

        Because the penalty matrix is the full-rank identity `I_k`, every basis
        coefficient is penalized and there is no unpenalized null space, unlike
        smooths with derivative-based penalties (e.g. TPRS) that leave polynomial
        trends unpenalized.

        Returns
        -------
        int
            Always `0`.
        """
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Returns
        -------
        NDArray
            A `(1, k)` matrix of equal weights `1 / k` whose product with the coefficient vector
            forces the mean fitted level effect to be zero, resolving the confound between the
            random effect's implicit constant and a fixed model intercept.
        """
        self._check_fitted()
        k = len(self._levels)
        return np.ones((1, k)) / k

    @property
    def n_basis(self) -> int:
        """Number of basis functions, i.e. the number of retained group levels.

        Returns
        -------
        int
            The number `k` of unique group levels retained during `fit()`.

        Raises
        ------
        RuntimeError
            If accessed before `fit()` has been called.
        """
        if not self._fitted:
            raise RuntimeError("n_basis is not available until fit() is called.")
        return len(self._levels)

    @property
    def k(self) -> int:
        """Requested or actual number of group levels.

        Before `fit()` is called, returns the `k` value passed at construction
        (the requested cap on the number of levels, or `-1` for "all levels").
        After `fit()`, returns the actual number of group levels retained.

        Returns
        -------
        int
            The requested or actual level count.
        """
        if self._fitted:
            return len(self._levels)
        return self._k_request

    @property
    def levels(self) -> NDArray:
        """Sorted array of unique group labels retained during `fit()`.

        Returns
        -------
        NDArray
            A copy of the sorted group labels, shape `(k,)`.
        """
        self._check_fitted()
        return self._levels.copy()

    @staticmethod
    def _validate(x: NDArray) -> NDArray:
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError(
                "RandomEffectBasis requires a 1-D grouping variable, "
                f"got array with shape {x.shape}."
            )
        return x
