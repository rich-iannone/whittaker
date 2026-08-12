"""Shrinkage smooth bases: ShrinkageTPRS (ts) and ShrinkageCRS (cs).

These bases use the double-penalty approach (Marra & Wood, 2011) to allow smooths to be penalized to
zero entirely. Each shrinkage basis produces two penalty matrices: the original wiggliness penalty
and a second penalty on the null space. With separate smoothing parameters for each, the fitting
machinery can shrink a smooth out of the model altogether, enabling automatic variable selection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.cubic import CRS
from whittaker.smooths.tprs import TPRS

_EIG_TOL = 1e-10


def _null_space_penalty(S: NDArray) -> NDArray:
    """Construct a penalty matrix that penalizes the null space of S.

    Eigendecomposes S, identifies eigenvectors with zero eigenvalues, and returns their outer
    product as a projection penalty.
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    threshold = _EIG_TOL * max(eigvals.max(), 1.0)
    null_mask = eigvals < threshold
    U_null = eigvecs[:, null_mask]
    S_null = U_null @ U_null.T
    return (S_null + S_null.T) * 0.5


class ShrinkageTPRS(TPRS):
    r"""Shrinkage Thin Plate Regression Spline.

    Equivalent to mgcv's `bs="ts"` basis. Ordinary penalized smooths (such as `TPRS`) always leave
    a low-dimensional null space — typically the constant and linear terms — completely
    unpenalized, so even a very large smoothing parameter `lambda` cannot remove the term from the
    model entirely: the fit can be flattened to a straight line, but not to exactly zero.
    `ShrinkageTPRS` fixes this by adding a second penalty that acts specifically on the null space,
    using the double-penalty construction of Marra & Wood (2011). With two independently-chosen
    smoothing parameters, the fitting machinery can drive both the wiggly part and the null-space
    part of the smooth to zero simultaneously, so the whole term can be shrunk out of the model.
    This makes `ShrinkageTPRS` a good choice over plain `TPRS` whenever a term's inclusion is
    itself uncertain and you want automatic variable selection via GCV or REML rather than a
    separate hypothesis test.

    The basis functions are identical to `TPRS` (`bs="tp"`); only the penalty structure differs.

    Parameters
    ----------
    k:
        Total number of basis functions, including the `M` null-space columns. The default is
        `10`. See `TPRS` for guidance on choosing `k`.
    m:
        Spline order. Must satisfy `2m > d` where `d` is the covariate dimension. The default is
        `2`. See `TPRS` for guidance on choosing `m`.

    Notes
    -----
    `ShrinkageTPRS` reuses the exact basis construction of `TPRS`: the first `M` columns span the
    polynomial null space (degree `<= m - 1` monomials) and the remaining `k - M` columns are the
    truncated eigenbasis of the projected thin-plate kernel. What changes is the penalty. Instead
    of a single penalty matrix, `penalty_matrices()` returns **two** matrices:

    $$
    \mathbf{S}_{\text{wiggle}} = \operatorname{diag}(0, \ldots, 0, \lambda_1, \ldots,
    \lambda_{k-M}),
    \qquad
    \mathbf{S}_{\text{null}} = \begin{pmatrix} \mathbf{I}_M & \mathbf{0} \\ \mathbf{0} &
    \mathbf{0} \end{pmatrix},
    $$

    where `S_wiggle` is exactly the ordinary `TPRS` penalty (zero on the null-space block) and
    `S_null` is a projection matrix that is the identity on the null-space block and zero elsewhere.
    During fitting, each matrix is scaled by its own smoothing parameter, `lambda_wiggle` and
    `lambda_null`, and the two contributions are added together:

    $$
    \lambda_{\text{wiggle}} \, \boldsymbol{\beta}^\top \mathbf{S}_{\text{wiggle}} \boldsymbol{\beta}
    + \lambda_{\text{null}} \, \boldsymbol{\beta}^\top \mathbf{S}_{\text{null}} \boldsymbol{\beta}.
    $$

    Because `S_wiggle + S_null` is strictly positive definite (it has no zero eigenvalues once
    both penalties act together), the combined penalty null space is empty, which is why
    `null_space_dimension()` returns `0` for this basis — none of the `k` basis functions is
    exempt from penalization once both smoothing parameters are positive. If `lambda_null` is
    estimated to be very large during fitting, the null-space coefficients are effectively zeroed
    and the term is excluded from the model, which is the mechanism behind automatic term
    selection.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import ShrinkageTPRS

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)

    basis = ShrinkageTPRS(k=10).fit(x)
    B = basis.basis_matrix(x)
    S_wiggle, S_null = basis.penalty_matrices()
    B.shape, S_wiggle.shape, S_null.shape
    ```
    """

    def penalty_matrices(self) -> list[NDArray]:
        """Return the two penalty matrices used for shrinkage.

        Returns
        -------
        list[NDArray]
            A two-element list `[S_wiggle, S_null]`, each of shape `(k, k)`. `S_wiggle` is the
            ordinary `TPRS` wiggliness penalty (zero on the null-space block). `S_null` is a
            projection matrix that is the identity on the `M` null-space columns and zero elsewhere.
            Each matrix is intended to be paired with its own independently-estimated smoothing
            parameter during fitting.
        """
        self._check_fitted()
        S_wiggle = super().penalty_matrix()

        k = self.k
        M = self._M
        S_null = np.zeros((k, k))
        S_null[:M, :M] = np.eye(M)

        return [S_wiggle, S_null]

    def null_space_dimension(self) -> int:
        """Return `0`: the combined double penalty leaves no unpenalized null space.

        Returns
        -------
        int
            Always `0` for `ShrinkageTPRS`, since `S_wiggle + S_null` is positive definite.
        """
        return 0


class ShrinkageCRS(CRS):
    r"""Shrinkage Cubic Regression Spline.

    Equivalent to mgcv's `bs="cs"` basis. `CRS` penalizes only the curvature of a natural cubic
    spline, leaving its 2-dimensional null space of constant and linear functions completely free
    — a large smoothing parameter flattens the fit to a line, but never removes it from the model.
    `ShrinkageCRS` adds a second penalty, built directly from the eigenstructure of the ordinary
    CRS penalty, that specifically targets this null space. Following the double-penalty approach
    of Marra & Wood (2011), the two penalties are given independent smoothing parameters during
    fitting, so that both the wiggly and the linear/constant parts of the term can be shrunk
    simultaneously, letting the term drop out of the model entirely. Prefer `ShrinkageCRS` over
    plain `CRS` for univariate terms whose presence in the model is uncertain and where automatic
    selection via GCV or REML is preferred over an explicit inclusion/exclusion test.

    The basis functions are identical to `CRS` (`bs="cr"`); only the penalty structure differs.

    Parameters
    ----------
    k:
        Number of basis functions (equal to the number of knots). Must be at least `3`. The
        default is `10`. See `CRS` for guidance on choosing `k`.

    Notes
    -----
    `ShrinkageCRS` reuses the exact basis construction of `CRS`: `k` knots at evenly-spaced
    quantiles of the training data, with the design matrix built from natural-cubic-spline basis
    functions (see `CRS` for the full construction). What changes is the penalty. The ordinary CRS
    penalty is

    $$
    \mathbf{S}_{\text{wiggle}} = \mathbf{Q} \mathbf{R}^{-1} \mathbf{Q}^\top,
    $$

    which is positive semi-definite with rank `k - 2` and a 2-dimensional null space spanned by the
    constant and linear functions evaluated at the knots. `ShrinkageCRS` eigendecomposes
    `S_wiggle`, identifies the eigenvectors `U_null` whose eigenvalues are numerically zero
    (below `1e-10` times the largest eigenvalue), and builds a second penalty matrix from their
    outer product:

    $$
    \mathbf{S}_{\text{null}} = \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^\top.
    $$

    `S_null` is positive semi-definite with rank `2`, non-zero exactly on the subspace that
    `S_wiggle` leaves unpenalized. During fitting each matrix is scaled by its own smoothing
    parameter and the two contributions are summed:

    $$
    \lambda_{\text{wiggle}} \, \boldsymbol{\beta}^\top \mathbf{S}_{\text{wiggle}} \boldsymbol{\beta}
    + \lambda_{\text{null}} \, \boldsymbol{\beta}^\top \mathbf{S}_{\text{null}} \boldsymbol{\beta}.
    $$

    Because `S_wiggle + S_null` is strictly positive definite, the combined penalty has no null
    space, so `null_space_dimension()` returns `0`: none of the `k` basis functions escapes
    penalization once both smoothing parameters are positive.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import ShrinkageCRS

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)

    basis = ShrinkageCRS(k=10).fit(x)
    B = basis.basis_matrix(x)
    S_wiggle, S_null = basis.penalty_matrices()
    B.shape, S_wiggle.shape, S_null.shape
    ```
    """

    def penalty_matrices(self) -> list[NDArray]:
        """Return the two penalty matrices used for shrinkage.

        Returns
        -------
        list[NDArray]
            A two-element list `[S_wiggle, S_null]`, each of shape `(k, k)`. `S_wiggle` is the
            ordinary `CRS` wiggliness penalty `Q R^{-1} Q'`. `S_null` is a projection penalty built
            from the eigenvectors of `S_wiggle` with (numerically) zero eigenvalue, targeting the
            constant and linear null space. Each matrix is intended to be paired with its own
            independently-estimated smoothing parameter during fitting.
        """
        self._check_fitted()
        S_wiggle = super().penalty_matrix()
        S_null = _null_space_penalty(S_wiggle)
        return [S_wiggle, S_null]

    def null_space_dimension(self) -> int:
        """Return `0`: the combined double penalty leaves no unpenalized null space.

        Returns
        -------
        int
            Always `0` for `ShrinkageCRS`, since `S_wiggle + S_null` is positive definite.
        """
        return 0
