r"""Adaptive smooth basis via eigendecomposition of the penalty.

An adaptive smooth decomposes the single penalty matrix into its eigenvectors, creating one penalty
per eigenvector. Each gets its own smoothing parameter, allowing spatially varying smoothness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.tprs import TPRS

_EIG_TOL = 1e-10


class AdaptiveTPRS(TPRS):
    r"""Adaptive Thin Plate Regression Spline.

    An ordinary `TPRS` uses a single smoothing parameter `lambda` to control wiggliness uniformly
    across the whole covariate domain, which is a poor fit when the true function is smooth in some
    regions and rapidly varying in others (e.g. a signal with a sharp local feature embedded in an
    otherwise flat trend). `AdaptiveTPRS` addresses this by decomposing the ordinary TPRS penalty
    matrix into its eigenvectors and turning each eigenvector (or a contiguous block of them) into
    its own separate penalty term with its own smoothing parameter, following the "adaptive
    smoothing" construction of Wood (2000, 2017, sec. 5.4.2). Because each eigenvector of the
    original penalty corresponds to a distinct spatial pattern of wiggliness, giving each its own
    `lambda` lets the fitted smoothing-parameter-selection procedure impose more smoothing where the
    data support a flat fit and less where they support local structure. Choose `AdaptiveTPRS` over
    plain `TPRS` when there is a-priori reason to expect the required smoothness to vary spatially
    and the extra smoothing parameters (and their computational cost during fitting) can be
    afforded; otherwise, use `TPRS`.

    The number of adaptive penalty components is controlled by `n_penalties`. With
    `n_penalties=k-M` (the default), every eigenvector gets its own λ. Smaller values group
    eigenvectors into blocks for computational efficiency.

    Parameters
    ----------
    k:
        Total number of basis functions, exactly as in `TPRS` (the default is `10`). Since
        `AdaptiveTPRS` uses the same underlying basis functions as `TPRS` and only changes how the
        penalty is structured, the same guidance for choosing `k` applies.
    m:
        Spline order, exactly as in `TPRS` (the default is `2`). Controls the order of derivative
        that the *unweighted* thin plate penalty targets before it is decomposed into adaptive
        components.
    n_penalties:
        Number of adaptive penalty components to construct from the eigendecomposition of the base
        TPRS penalty. If `-1` (the default), one penalty is created per retained eigenvector
        (`k - M` penalties total, the maximum granularity and maximum flexibility for spatially
        varying smoothness). If a smaller positive integer is given, the eigenvectors (already
        sorted by decreasing eigenvalue during `TPRS.fit()`) are grouped into that many contiguous
        blocks of roughly equal size, each sharing one smoothing parameter; this reduces the number
        of smoothing parameters that must be estimated, trading some spatial adaptivity for faster,
        more stable fitting.

    Notes
    -----
    After `fit()` (inherited unchanged from `TPRS`), the basis matrix and its first `M` null-space
    columns are identical to plain `TPRS`; only the penalty differs. Let `D_r =
    \operatorname{diag}(d_1, \ldots, d_r)` be the diagonal matrix of the `r = k - M` eigenvalues
    retained by `TPRS.fit()` (in descending order), so that the ordinary TPRS penalty restricted to
    the range space is `D_r` itself. `AdaptiveTPRS` partitions the index set `\{1, \ldots, r\}` into
    `n_penalties` contiguous blocks `B_1, \ldots, B_{n_penalties}` of (approximately) equal size and
    returns one `k x k` penalty matrix per block,

    $$
    \mathbf{S}_b = \operatorname{diag}\bigl(0, \ldots, 0,\; \max(d_i, \epsilon) \cdot [i \in B_b],
    \ldots\bigr), \qquad b = 1, \ldots, n_\text{penalties},
    $$

    where each `S_b` is zero everywhere except in the diagonal entries corresponding to
    eigenvectors in its block, and `\epsilon = 10^{-10}` guards against the (numerically possible)
    tiny negative eigenvalues that arise from the projection step in `TPRS.fit()`. The total
    penalty used during model fitting is the weighted sum
    `\sum_b \lambda_b \, \boldsymbol{\beta}^\top \mathbf{S}_b \boldsymbol{\beta}`, with one
    smoothing parameter `\lambda_b` per block selected by the outer GCV/REML procedure — this is
    what allows the fitted smoothness to vary spatially: blocks corresponding to slowly varying
    eigenvectors can
    receive small `\lambda_b` (little shrinkage, high local flexibility) while blocks corresponding
    to rapidly varying eigenvectors receive large `\lambda_b` (heavy shrinkage), or vice versa,
    according to what the data support in different parts of the covariate space. Since the block
    matrices `S_b` are mutually orthogonal (each touches disjoint diagonal entries) and together
    cover exactly the `k - M` range-space coefficients, `null_space_dimension()` is unchanged from
    `TPRS` and equals `M`. Because more penalties mean more smoothing parameters to estimate,
    `n_penalties` close to `r` can make outer optimization slower and, on small or noisy datasets,
    less numerically stable; the default of `n_penalties=-1` should be reduced if fitting becomes
    unreliable.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths.adaptive import AdaptiveTPRS

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)

    basis = AdaptiveTPRS(k=10, n_penalties=3).fit(x)
    B = basis.basis_matrix(x)
    penalties = basis.penalty_matrices()
    B.shape, len(penalties), penalties[0].shape
    ```
    """

    def __init__(self, k: int = 10, m: int = 2, n_penalties: int = -1) -> None:
        super().__init__(k=k, m=m)
        self._n_penalties = n_penalties

    def penalty_matrices(self) -> list[NDArray]:
        """Return the decomposed penalty matrices, one per eigenvector group.

        Each returned matrix is `k x k` and zero everywhere except on the diagonal entries
        belonging to its group of eigenvectors of the underlying TPRS penalty, so that summing all
        returned matrices reconstructs the ordinary (non-adaptive) TPRS penalty restricted to the
        range space.

        Returns
        -------
        list[NDArray]
            List of `n_penalties` matrices (or `k - M` matrices if `n_penalties=-1`), each of shape
            `(k, k)`. Intended to be combined with one smoothing parameter per matrix during model
            fitting.
        """
        self._check_fitted()
        M = self._M
        r = self.k - M
        eigvals = self._eigenvalues

        n_pen = self._n_penalties if self._n_penalties > 0 else r

        if n_pen >= r:
            pens = []
            for i in range(r):
                S = np.zeros((self.k, self.k))
                S[M + i, M + i] = max(eigvals[i], _EIG_TOL)
                pens.append(S)
            return pens

        block_size = r / n_pen
        pens = []
        for b in range(n_pen):
            start = int(round(b * block_size))
            end = int(round((b + 1) * block_size))
            S = np.zeros((self.k, self.k))
            for i in range(start, min(end, r)):
                S[M + i, M + i] = max(eigvals[i], _EIG_TOL)
            pens.append(S)
        return pens

    def null_space_dimension(self) -> int:
        """Return `M`, the dimension of the (unchanged) TPRS polynomial null space.

        Splitting the range-space penalty into blocks does not add or remove any unpenalized
        basis functions, so this is identical to the value returned by the parent `TPRS` class.

        Returns
        -------
        int
            The polynomial null-space dimension `M = C(m - 1 + d, d)`.
        """
        self._check_fitted()
        return self._M
