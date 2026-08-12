r"""Factor-smooth interaction basis (bs="fs").

Implements per-level smooths with shared penalties, equivalent to mgcv's `bs="fs"`. Each level of
a grouping factor gets its own smooth curve for the numeric covariate, but all levels share the same
smoothing parameter(s). This is the standard approach for longitudinal / panel data where each
subject has its own trajectory.

The basis matrix is block-diagonal: one block per factor level, each block being the marginal smooth
basis multiplied by the level indicator. The penalty structure has `1 + M` penalties where `M` is
the null space dimension of the marginal basis:

1. The wiggliness penalty `I_L ⊗ S` (replicated across all `L` levels).
2. One penalty per null space component, penalizing that component's coefficients across levels
(effectively a random intercept/slope penalty).

Since every coefficient is penalized, `null_space_dimension = 0` and no identifiability constraints
are needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.cubic import CRS
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.tprs import TPRS

_MARGINAL_REGISTRY: dict[str, type[SmoothBasis]] = {
    "tp": TPRS,
    "cr": CRS,
    "ps": PSpline,
}


class FactorSmoothBasis(SmoothBasis):
    r"""Factor-smooth interaction basis (per-level smooth with shared penalties).

    A factor-smooth interaction fits a separate curve of a numeric covariate for every level of a
    grouping factor — one trajectory per subject in a longitudinal study, one seasonal curve per
    site, one dose-response curve per batch — while pooling information across levels by sharing
    the same wiggliness penalty (and, optionally, coefficient-level shrinkage) across all of them.
    It is equivalent to mgcv's `bs="fs"` basis and is the natural GAM analogue of a random-slope
    mixed model. Choose this basis over fitting `k` independent smooths (one per level) when the
    levels are numerous, some levels have little data, and it is desirable to borrow strength across
    levels via a shared smoothing parameter; choose it over a single shared smooth (with an
    additional `RandomEffectBasis` for level differences) when each level's mean *shape*, not merely
    its overall level, is expected to differ appreciably.

    Parameters
    ----------
    k:
        Number of basis functions for the marginal smooth per level. Applies to every level
        identically — all per-level smooths share the same basis dimension. Larger `k` allows more
        wiggly per-level curves at the cost of more coefficients (`n_levels * k` total); since the
        smoothing parameter (not `k`) ultimately controls how wiggly the fitted curves are, `k`
        mainly needs to be large enough not to unduly constrain the shape. The default is `10`.
    xt:
        Marginal basis type used for each level's smooth. One of `"tp"` (thin plate regression
        spline, a good general-purpose default), `"cr"` (cubic regression spline, cheaper for a
        single covariate with many knots), or `"ps"` (P-spline, useful when a difference penalty on
        B-spline coefficients is preferred). The default is `"tp"`.
    **marginal_kwargs:
        Additional keyword arguments forwarded to the marginal basis constructor for the chosen
        `xt` (e.g. `m` for the spline order used by `"tp"` and `"ps"` bases).

    Notes
    -----
    Let there be `L` factor levels and let the fitted marginal basis (shared in form, but evaluated
    per level) have `k_m` basis functions with null-space dimension `M`. The full basis matrix is
    block-diagonal by level:

    $$
    \mathbf{B} = \begin{bmatrix} \mathbf{1}_{[\text{level}=1]} \odot \mathbf{B}_m &
    \mathbf{1}_{[\text{level}=2]} \odot \mathbf{B}_m & \cdots &
    \mathbf{1}_{[\text{level}=L]} \odot \mathbf{B}_m \end{bmatrix},
    $$

    where `B_m` is the marginal basis matrix evaluated at the numeric covariate and
    `\mathbf{1}_{[\text{level}=\ell]}` is the indicator for observations belonging to level `\ell`
    (each row of `B` is nonzero only in the block for its own level); the total number of columns
    is `L * k_m`. The penalty structure has `1 + M` components:

    1. A shared **wiggliness** penalty, replicated identically across all levels via the Kronecker
       structure

       $$
       \mathbf{S}_{\text{wiggle}} = \mathbf{I}_L \otimes \mathbf{S}_m ,
       $$

       where `S_m` is the marginal basis's own penalty matrix. A single smoothing parameter
       controls how wiggly *every* level's curve is allowed to be.
    2. **One penalty per marginal null-space component** (there are `M` of them, e.g. `M=2` for a
       thin plate spline with `m=2` — the constant and linear components). For each null-space
       eigenvector `v` of `S_m`, the corresponding penalty places `\operatorname{outer}(v, v)` in
       every level's diagonal block, so that this penalty shrinks that low-order component of the
       curve (e.g. each level's intercept, or each level's linear trend) toward a common value
       across levels — a random-intercept/random-slope penalty for exactly the marginal basis's
       unpenalized directions.

    Because every basis coefficient is touched by at least one of these `1 + M` penalties (the
    wiggliness penalty covers the range space and the null-space penalties cover what the
    wiggliness penalty leaves unpenalized), `null_space_dimension()` is always `0` and no
    `identifiability_constraints()` are required — the basis is fully penalized and does not
    collide with a fixed intercept. `fit()` raises `ValueError` if fewer than 2 factor levels are
    present, since factor-smooth interactions require differentiating between levels.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths.factor_smooth import FactorSmoothBasis

    rng = np.random.default_rng(0)
    n = 100
    subject = rng.choice(["s1", "s2", "s3"], size=n)
    x_numeric = rng.uniform(0, 1, n)

    basis = FactorSmoothBasis(k=8).fit(x_numeric, subject)
    B = basis.basis_matrix(x_numeric, subject)
    penalties = basis.penalty_matrices()
    B.shape, len(penalties)
    ```
    """

    def __init__(self, k: int = 10, xt: str = "tp", **marginal_kwargs: Any) -> None:
        if xt not in _MARGINAL_REGISTRY:
            supported = ", ".join(sorted(_MARGINAL_REGISTRY))
            raise ValueError(f"Unknown marginal basis type xt={xt!r}. Supported: {supported}.")

        self._k_request = k
        self._xt = xt
        self._marginal_kwargs = marginal_kwargs
        self._fitted: bool = False

        self._marginal: SmoothBasis
        self._levels: NDArray
        self._level_to_idx: dict
        self._n_levels: int
        self._k_marginal: int
        self._marginal_nsd: int

    def fit(  # type: ignore[override]
        self, x_numeric: NDArray, factor: NDArray | None = None, **kwargs: Any,
    ) -> FactorSmoothBasis:
        """Fit the factor-smooth basis.

        Parameters
        ----------
        x_numeric:
            Numeric covariate, shape `(n,)`.
        factor:
            Factor (grouping) variable, shape `(n,)`. Can be strings or integers.
        """
        if factor is None:
            raise ValueError(
                "FactorSmoothBasis.fit() requires both x_numeric and factor arguments."
            )
        x_num = self._as_1d_numeric(x_numeric)
        fac = self._as_1d_factor(factor)

        if len(x_num) != len(fac):
            raise ValueError(
                f"x_numeric and factor must have the same length, got {len(x_num)} and {len(fac)}."
            )

        self._levels = np.unique(fac)
        self._level_to_idx = {lev: i for i, lev in enumerate(self._levels)}
        self._n_levels = len(self._levels)

        if self._n_levels < 2:
            raise ValueError(
                f"FactorSmoothBasis requires at least 2 factor levels, got {self._n_levels}."
            )

        marginal_cls = _MARGINAL_REGISTRY[self._xt]
        marginal_kw: dict[str, Any] = {}
        if self._k_request != -1:
            marginal_kw["k"] = self._k_request
        marginal_kw.update(self._marginal_kwargs)
        self._marginal = marginal_cls(**marginal_kw)
        self._marginal.fit(x_num)

        self._k_marginal = self._marginal.n_basis
        self._marginal_nsd = self._marginal.null_space_dimension()

        self._fitted = True
        return self

    def basis_matrix(  # type: ignore[override]
        self, x_numeric: NDArray, factor: NDArray | None = None, **kwargs: Any,
    ) -> NDArray:
        """Build the block-diagonal basis matrix.

        Parameters
        ----------
        x_numeric:
            Numeric covariate, shape `(n,)`.
        factor:
            Factor variable, shape `(n,)`.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, n_levels * k_marginal)`.
        """
        self._check_fitted()
        if factor is None:
            raise ValueError(
                "FactorSmoothBasis.basis_matrix() requires both x_numeric and factor arguments."
            )
        x_num = self._as_1d_numeric(x_numeric)
        fac = self._as_1d_factor(factor)

        B_marginal = self._marginal.basis_matrix(x_num)
        n = len(x_num)
        k_total = self._n_levels * self._k_marginal

        B = np.zeros((n, k_total))
        for lev, idx in self._level_to_idx.items():
            indicator = (fac == lev).astype(float)
            col_start = idx * self._k_marginal
            col_end = col_start + self._k_marginal
            B[:, col_start:col_end] = B_marginal * indicator[:, np.newaxis]

        return B

    def penalty_matrices(self) -> list[NDArray]:
        """Return the full set of `1 + M` penalty matrices.

        Returns
        -------
        list[NDArray]
            A list of `k_total x k_total` matrices (`k_total = n_levels * k_marginal`): the first
            is the shared wiggliness penalty `I_L ⊗ S_marginal`, and the remaining `M` (the marginal
            basis's null-space dimension) are random-effect-style penalties, one per null-space
            component of the marginal penalty, each shrinking that component's coefficients toward a
            common value across levels.
        """
        self._check_fitted()
        k_total = self._n_levels * self._k_marginal
        S_marginal = self._marginal.penalty_matrix()

        S_wiggle = np.zeros((k_total, k_total))
        for i in range(self._n_levels):
            cs = i * self._k_marginal
            ce = cs + self._k_marginal
            S_wiggle[cs:ce, cs:ce] = S_marginal
        S_wiggle = (S_wiggle + S_wiggle.T) * 0.5

        penalties = [S_wiggle]

        if self._marginal_nsd > 0:
            eigvals, eigvecs = np.linalg.eigh(S_marginal)
            threshold = max(eigvals.max(), 1.0) * 1e-10
            null_mask = eigvals < threshold
            null_vecs = eigvecs[:, null_mask]

            for j in range(null_vecs.shape[1]):
                v = null_vecs[:, j]
                S_null_comp = np.zeros((k_total, k_total))
                for i in range(self._n_levels):
                    cs = i * self._k_marginal
                    ce = cs + self._k_marginal
                    S_null_comp[cs:ce, cs:ce] = np.outer(v, v)
                S_null_comp = (S_null_comp + S_null_comp.T) * 0.5
                penalties.append(S_null_comp)

        return penalties

    def penalty_matrix(self) -> NDArray:
        """Return only the shared wiggliness penalty (first entry of `penalty_matrices()`).

        This exists for compatibility with the single-penalty `SmoothBasis` interface. It omits the
        `M` null-space (random-intercept/slope) penalties; use `penalty_matrices()` to access the
        full penalty structure needed for a proper fit.

        Returns
        -------
        NDArray
            The `(k_total, k_total)` wiggliness penalty `I_L ⊗ S_marginal`.
        """
        self._check_fitted()
        return self.penalty_matrices()[0]

    def null_space_dimension(self) -> int:
        """Return the dimension of the basis's unpenalized null space.

        The factor-smooth basis has no unpenalized null space: the shared wiggliness penalty
        `I_L ⊗ S_marginal` covers the range space of the marginal penalty in every level, and the
        `M` additional null-space (random-intercept/slope) penalties returned by
        `penalty_matrices()` cover exactly the directions the wiggliness penalty leaves
        unpenalized. Since every basis coefficient is touched by at least one penalty, this is
        always `0`.

        Returns
        -------
        int
            Always `0`.
        """
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        """Return `None`: no additional identifiability constraint is needed.

        Unlike `RandomEffectBasis` or `MRFBasis`, the null-space penalties here already shrink each
        level's low-order components toward a common value, so there is no separate unpenalized
        constant that collides with a model intercept.
        """
        return None

    @property
    def n_basis(self) -> int:
        """Total number of basis functions.

        Equal to `n_levels * k_marginal`: the marginal basis dimension repeated once per factor
        level, since the full basis matrix is block-diagonal by level.

        Returns
        -------
        int
            Total basis dimension.

        Raises
        ------
        RuntimeError
            If accessed before `fit()` has been called.
        """
        if not self._fitted:
            raise RuntimeError("n_basis is not available until fit() is called.")
        return self._n_levels * self._k_marginal

    @property
    def k(self) -> int:
        """Marginal basis dimension per factor level.

        Before `fit()` is called, returns the requested value passed to `__init__` (which may be
        `-1` to defer to the marginal basis's own default). After `fit()`, returns the marginal
        basis's actual, fitted dimension.

        Returns
        -------
        int
            Requested or fitted marginal basis dimension.
        """
        if self._fitted:
            return self._k_marginal
        return self._k_request

    @property
    def levels(self) -> NDArray:
        """Sorted array of unique factor levels retained during `fit()`.

        Each level corresponds to one diagonal block of the basis matrix and one column range of
        `k_marginal` coefficients.

        Returns
        -------
        NDArray
            Copy of the sorted, unique factor levels seen during `fit()`.
        """
        self._check_fitted()
        return self._levels.copy()

    @property
    def n_levels(self) -> int:
        """Number of factor (grouping) levels seen during `fit()`.

        Determines the number of diagonal blocks in the basis matrix and, together with
        `k_marginal`, the total basis dimension `n_basis`.

        Returns
        -------
        int
            Number of distinct factor levels, always `>= 2`.
        """
        self._check_fitted()
        return self._n_levels

    @property
    def marginal_basis(self) -> SmoothBasis:
        """The fitted marginal basis object shared, in form, across all levels.

        This single fitted basis (e.g. a `TPRS`, `CRS`, or `PSpline` instance) supplies the shape
        of the per-level smooths; it is evaluated once per level, each time multiplied by that
        level's indicator, to build the block-diagonal `basis_matrix()`.

        Returns
        -------
        SmoothBasis
            The fitted marginal basis instance.
        """
        self._check_fitted()
        return self._marginal

    @staticmethod
    def _as_1d_numeric(x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(f"Expected 1-D numeric array, got shape {x.shape}.")

    @staticmethod
    def _as_1d_factor(x: NDArray) -> NDArray:
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError(f"Expected 1-D factor array, got shape {x.shape}.")
        return x
