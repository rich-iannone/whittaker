r"""Tensor product smooth bases (te / ti / t2)."""

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
    r"""Tensor product of marginal smooth bases (`te()`-style interaction smooth).

    A tensor product smooth builds a multivariate smooth function of two or more covariates out of
    one-dimensional (or otherwise lower-dimensional) marginal smooths, one per covariate, by taking
    the row-wise outer product of their basis matrices. This is the standard way to represent an
    interaction between covariates that are measured on very different scales or units (e.g. a
    spatial coordinate combined with time, or a covariate in meters combined with one in years) —
    unlike an isotropic basis such as `TPRS`, which assumes all covariates share a common notion of
    distance, a tensor product smooth applies a separate marginal penalty (and, in principle, a
    separate smoothing parameter) to each covariate direction, so that the anisotropic scaling of
    the covariates does not distort the fitted surface. Choose `TensorProductBasis` over `TPRS`
    whenever the covariates involved are not naturally on comparable scales, and reach for
    `TensorInteractionBasis` instead when a decomposition into separate main-effect and pure-
    interaction terms (an ANOVA-style model) is wanted.

    Parameters
    ----------
    marginals:
        List of (typically unfitted) marginal basis objects, one per covariate dimension to be
        combined, e.g. `[TPRS(k=10), TPRS(k=8)]` for a bivariate smooth. Each marginal is fit
        independently to its own column of `x` inside `fit()`. At least 2 marginals are required;
        for a single covariate, use the marginal basis directly instead of wrapping it in a tensor
        product.

    Notes
    -----
    Given `d` marginal bases with basis matrices `B_1, \ldots, B_d` (each `B_j` of shape
    `(n, k_j)`), the tensor product basis matrix is the row-wise Kronecker product

    $$
    \mathbf{B}[i, :] = \mathbf{B}_1[i, :] \otimes \mathbf{B}_2[i, :] \otimes \cdots \otimes
    \mathbf{B}_d[i, :], \qquad i = 1, \ldots, n,
    $$

    which has `k = k_1 k_2 \cdots k_d` columns in total — every combination of one marginal basis
    function from each dimension. This basis represents *all* smooth functions expressible as sums
    of products of the marginal bases, including both pure main effects and their interaction, so
    unlike `TensorInteractionBasis`, no separate main-effect terms need to be added to the model for
    identifiability of low-order structure (though it is common practice in mgcv-style formulas to
    do so anyway for a cleaner ANOVA decomposition). For each marginal direction `j` with own
    penalty `S_j`, the tensor product carries one whole-basis penalty per marginal direction,

    $$
    \mathbf{S}_j^{\text{tensor}} = \mathbf{I}_{k_1} \otimes \cdots \otimes \mathbf{I}_{k_{j-1}}
    \otimes \mathbf{S}_j \otimes \mathbf{I}_{k_{j+1}} \otimes \cdots \otimes \mathbf{I}_{k_d},
    $$

    returned in order by `penalty_matrices()`; `penalty_matrix()` sums these into a single matrix
    only for compatibility with the base `SmoothBasis` interface — for a proper anisotropic fit
    (one smoothing parameter per marginal direction), use `penalty_matrices()` directly rather than
    `penalty_matrix()`. The null-space dimension of the combined penalty is the product of the
    marginal null-space dimensions, `M = M_1 \cdot M_2 \cdots M_d` (the multivariate polynomials
    that are simultaneously in the null space of every marginal penalty). Because `k` grows
    multiplicatively with the number of marginals and their individual sizes, tensor products
    become expensive quickly in more than two or three dimensions; for higher-dimensional smooths
    of covariates on comparable scales, an isotropic basis such as `TPRS` is usually preferable.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths.tensor import TensorProductBasis
    from whittaker.smooths.tprs import TPRS

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (100, 2))

    basis = TensorProductBasis([TPRS(k=6), TPRS(k=5)]).fit(x)
    B = basis.basis_matrix(x)
    penalties = basis.penalty_matrices()
    B.shape, len(penalties)
    ```
    """

    def __init__(self, marginals: list[SmoothBasis]) -> None:
        if len(marginals) < 2:
            raise ValueError("TensorProductBasis requires at least 2 marginals.")
        self._marginals = marginals
        self._fitted = False

    def fit(self, x: NDArray) -> TensorProductBasis:
        """Fit each marginal basis to its corresponding column of `x`.

        Parameters
        ----------
        x:
            Training covariates, shape `(n, d)` where `d` is the number of marginals. Column `j`
            is passed to `marginals[j].fit()`.

        Returns
        -------
        TensorProductBasis
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If the number of columns in `x` does not match the number of marginal bases.
        """
        x = self._as_2d(x)
        if x.shape[1] != len(self._marginals):
            raise ValueError(f"Expected {len(self._marginals)} columns, got {x.shape[1]}.")
        for j, basis in enumerate(self._marginals):
            basis.fit(x[:, j])
        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the tensor product basis at `x`.

        Builds each marginal's basis matrix independently and combines them with the row-wise
        Kronecker (tensor) product, so that column `j1 * k_2 * ... * k_d + j2 * k_3 * ... + ...`
        of the result is the product of the corresponding marginal basis function values.

        Parameters
        ----------
        x:
            Covariate values, shape `(n, d)` where `d` matches the number of marginals.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)` where `k = self.n_basis` is the product of the
            marginal basis dimensions.
        """
        self._check_fitted()
        x = self._as_2d(x)
        B = self._marginals[0].basis_matrix(x[:, 0])
        for j in range(1, len(self._marginals)):
            B = _row_tensor_product(B, self._marginals[j].basis_matrix(x[:, j]))
        return B

    def penalty_matrix(self) -> NDArray:
        """Sum of all marginal penalty matrices, for single-penalty compatibility.

        This collapses the per-direction penalties returned by `penalty_matrices()` into one
        matrix, which is convenient when a caller only supports a single penalty but discards the
        ability to give each marginal direction its own smoothing parameter. Prefer
        `penalty_matrices()` for a proper anisotropic fit.

        Returns
        -------
        NDArray
            The `(k, k)` sum of all per-marginal penalty matrices.
        """
        pens = self.penalty_matrices()
        return sum(pens)

    def penalty_matrices(self) -> list[NDArray]:
        """Return one penalty matrix per marginal direction.

        For `d` marginals with basis dimensions `k_1, ..., k_d` and marginal penalties
        `S_1, ..., S_d`, penalty `j` is the Kronecker product
        `I_{k_1} ⊗ ... ⊗ S_j ⊗ ... ⊗ I_{k_d}` (the marginal penalty in position `j`, identity
        elsewhere), so that penalizing with matrix `j` alone penalizes exactly the roughness of the
        tensor surface along covariate direction `j`, holding the other directions fixed.

        Returns
        -------
        list[NDArray]
            List of `d` matrices, each of shape `(k, k)` where `k` is the total basis dimension.
            Intended to be combined with one smoothing parameter per marginal direction during
            model fitting.
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
        """Return the product of the marginal null-space dimensions.

        The multivariate null space of a tensor product penalty is spanned by the tensor products
        of each marginal's own null-space basis functions (e.g. for two thin plate marginals with
        linear null spaces, the constant, and each marginal's linear term, giving a bilinear null
        space), so its dimension multiplies rather than adds across marginals.

        Returns
        -------
        int
            `M = M_1 * M_2 * ... * M_d`.
        """
        nsd = 1
        for m in self._marginals:
            nsd *= m.null_space_dimension()
        return nsd

    def identifiability_constraints(self) -> NDArray | None:
        """Return a sum-to-zero constraint on the tensor product basis, if derivable.

        Attempts to recover each marginal's training covariate values (via a `_x_train` or
        `_knots` attribute) to build a constraint that forces the tensor surface to have mean zero
        over the training data. If no marginal exposes its training data in a recognized form,
        no constraint can be derived and `None` is returned.

        Returns
        -------
        NDArray or None
            A `(1, k)` matrix of the mean basis-function values over the training grid, or `None`
            if the training covariates could not be recovered from the marginals.
        """
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
        """Total number of basis functions, the product of the marginal basis dimensions."""
        result = 1
        for m in self._marginals:
            result *= m.n_basis
        return result

    @property
    def marginals(self) -> list[SmoothBasis]:
        """The list of (fitted, after `fit()`) marginal basis objects."""
        return self._marginals


class TensorProductBasisT2(TensorProductBasis):
    r"""Tensor product basis with full penalty decomposition (`t2()`-style interaction smooth).

    `TensorProductBasisT2` builds exactly the same basis matrix as `TensorProductBasis` (`te()`) —
    the row-wise Kronecker product of the marginal bases — but replaces `te()`'s `d` per-direction
    penalties with a richer decomposition that has one penalty for *every* non-empty subset of the
    marginal directions, following the "type 2" tensor product construction of Wood, Scheipl and
    Faraway (2013). Because it uses ordinary quadratic penalties throughout (rather than the
    null-space/range-space projections used internally by `te()`), the resulting penalties are
    positive semi-definite by construction and combine cleanly with random-effects representations
    of the smooth, and it never has negative smoothing-parameter degeneracies that can occasionally
    affect `te()`. Prefer `TensorProductBasisT2` over plain `TensorProductBasis` when a strictly
    additive quadratic-penalty structure is needed (e.g. for mixed-model / REML-based fitting), or
    when the interaction component of the surface is believed to need noticeably different
    smoothing from any single marginal direction on its own and a dedicated smoothing parameter for
    that pure-interaction subset is wanted.

    Notes
    -----
    For `d` marginals with penalties `S_1, \ldots, S_d` and basis dimensions `k_1, \ldots, k_d`,
    every non-empty subset `\sigma \subseteq \{1, \ldots, d\}` of marginal directions gets its own
    penalty

    $$
    \mathbf{S}_\sigma = \mathbf{M}_1 \otimes \mathbf{M}_2 \otimes \cdots \otimes \mathbf{M}_d,
    \qquad \mathbf{M}_j = \begin{cases} \mathbf{S}_j & j \in \sigma \\ \mathbf{I}_{k_j} & j \notin
    \sigma \end{cases},
    $$

    giving `2^d - 1` penalties in total (compared to the `d` penalties of `TensorProductBasis`).
    For two marginals with penalties `S_1, S_2`, this is

    $$
    \{\, \mathbf{S}_1 \otimes \mathbf{I},\ \ \mathbf{I} \otimes \mathbf{S}_2,\ \ \mathbf{S}_1
    \otimes \mathbf{S}_2 \,\},
    $$

    the usual two `te()`-style main-effect-direction penalties plus one additional penalty,
    `S_1 ⊗ S_2`, that penalizes roughness jointly in *both* directions at once — this extra term is
    what lets the pure two-way-interaction component of the surface have its own smoothing
    parameter, separate from either marginal direction's smoothness. As the number of marginals `d`
    grows, the number of penalties grows exponentially (`2^d - 1`), so this construction is
    practical mainly for `d = 2` or `d = 3`; `null_space_dimension()` and `identifiability_constraints()`
    are inherited unchanged from `TensorProductBasis`, since the basis matrix itself does not
    change — only the penalty is decomposed further.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths.tensor import TensorProductBasisT2
    from whittaker.smooths.tprs import TPRS

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (100, 2))

    basis = TensorProductBasisT2([TPRS(k=6), TPRS(k=5)]).fit(x)
    B = basis.basis_matrix(x)
    penalties = basis.penalty_matrices()
    B.shape, len(penalties)  # 3 penalties for 2 marginals (2^2 - 1)
    ```
    """

    def penalty_matrices(self) -> list[NDArray]:
        """Return one penalty per non-empty subset of marginal directions.

        Returns
        -------
        list[NDArray]
            List of `2^d - 1` matrices, each of shape `(k, k)` where `d` is the number of
            marginals and `k` is the total basis dimension. Ordered by increasing subset bitmask
            (single-direction penalties first, higher-order interaction penalties last).
        """
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
    r"""Tensor product interaction basis (`ti()`-style pure interaction smooth).

    `TensorInteractionBasis` builds a tensor product smooth like `TensorProductBasis`, but first
    projects each marginal basis onto its own penalty *range space* (removing the marginal's
    null-space components, such as the constant and linear terms) before taking the tensor product.
    The result spans only the pure interaction between the covariates — none of the lower-order
    main-effect structure that a plain tensor product basis would otherwise reintroduce. This makes
    it the right building block for ANOVA-style decompositions of a smooth surface into orthogonal
    pieces, e.g. `s(x1) + s(x2) + ti(x1, x2)`, where `s(x1)` and `s(x2)` already carry the main
    effects and `ti(x1, x2)` is meant to add only what a sum of the two one-dimensional smooths
    cannot represent. Use `TensorInteractionBasis` instead of `TensorProductBasis` whenever main
    effects are (or will be) modeled by separate marginal smooths and double-counting of the
    main-effect structure inside the interaction term must be avoided; use `TensorProductBasis` or
    `TensorProductBasisT2` when the interaction term is meant to stand alone and include the main
    effects itself.

    Parameters
    ----------
    marginals:
        List of (unfitted) marginal basis objects, one per covariate dimension, e.g.
        `[TPRS(k=10), TPRS(k=8)]`. At least 2 marginals are required.

    Notes
    -----
    For each marginal basis with penalty `S_j`, `fit()` eigendecomposes `S_j` and keeps only the
    eigenvectors `U_j` with (numerically) positive eigenvalues — the penalized *range space* — while
    discarding the null-space eigenvectors (the unpenalized polynomials, dimension `M_j`). Each
    marginal's basis matrix is then reprojected onto this reduced space, `B_j' = B_j U_j`, of
    dimension `r_j = k_j - M_j` rather than the original `k_j`, before the marginals are combined
    with the same row-wise Kronecker product used by `TensorProductBasis`:

    $$
    \mathbf{B}[i, :] = \mathbf{B}_1'[i, :] \otimes \mathbf{B}_2'[i, :] \otimes \cdots \otimes
    \mathbf{B}_d'[i, :].
    $$

    Because every marginal contributes only its range space, none of the columns of `B` correspond
    to a main-effect direction, and the total basis dimension is the product of the range-space
    dimensions, `k = r_1 \cdot r_2 \cdots r_d`, smaller than the `k_1 \cdots k_d` used by
    `TensorProductBasis` on the same marginals. In this reduced basis, each marginal's penalty
    is already diagonal (it is expressed in its own eigenbasis), `\operatorname{diag}(d_{j,1},
    \ldots, d_{j,r_j})` for the retained eigenvalues `d_{j,i}`, and the per-direction penalty
    matrices are the Kronecker products

    $$
    \mathbf{S}_j^{\text{ti}} = \mathbf{I}_{r_1} \otimes \cdots \otimes \operatorname{diag}(d_{j,
    1}, \ldots, d_{j, r_j}) \otimes \cdots \otimes \mathbf{I}_{r_d},
    $$

    exactly analogous to `TensorProductBasis.penalty_matrices()` but operating in the range-space
    coordinates. Since every retained coefficient direction is, by construction, in some marginal's
    range space and therefore penalized by at least one of these matrices, `null_space_dimension()`
    is always `0` for the interaction basis itself. Note that the eigendecomposition used to find
    each marginal's range space assumes the marginal penalty has a well-separated null space (a
    numerical tolerance of `1e-10` relative to the largest eigenvalue is used to distinguish
    "zero"); for marginal bases with unusual or nearly-singular penalties this tolerance may need
    revisiting.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths.tensor import TensorInteractionBasis
    from whittaker.smooths.tprs import TPRS

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (100, 2))

    basis = TensorInteractionBasis([TPRS(k=6), TPRS(k=5)]).fit(x)
    B = basis.basis_matrix(x)
    penalties = basis.penalty_matrices()
    B.shape, len(penalties)
    ```
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
        """Fit each marginal basis, then decompose its penalty into null and range spaces.

        For each marginal, fits the marginal basis to its column of `x`, eigendecomposes the
        marginal's penalty matrix, and retains the eigenvectors with positive eigenvalues (the
        range space) for use by `basis_matrix()` and `penalty_matrices()`.

        Parameters
        ----------
        x:
            Training covariates, shape `(n, d)` where `d` is the number of marginals.

        Returns
        -------
        TensorInteractionBasis
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If the number of columns in `x` does not match the number of marginal bases.
        """
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
        """Evaluate the pure-interaction tensor basis at `x`.

        Projects each marginal's basis matrix onto its own penalty range space (dropping the
        marginal's null-space/main-effect columns) before combining them with the row-wise
        Kronecker product.

        Parameters
        ----------
        x:
            Covariate values, shape `(n, d)` where `d` matches the number of marginals.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)` where `k = self.n_basis` is the product of the
            marginal range-space dimensions.
        """
        self._check_fitted()
        x = self._as_2d(x)
        B = self._marginals[0].basis_matrix(x[:, 0]) @ self._range_projections[0]
        for j in range(1, len(self._marginals)):
            Bj = self._marginals[j].basis_matrix(x[:, j]) @ self._range_projections[j]
            B = _row_tensor_product(B, Bj)
        return B

    def penalty_matrix(self) -> NDArray:
        """Sum of all per-direction penalty matrices, for single-penalty compatibility.

        Returns
        -------
        NDArray
            The `(k, k)` sum of all matrices returned by `penalty_matrices()`.
        """
        pens = self.penalty_matrices()
        return sum(pens)

    def penalty_matrices(self) -> list[NDArray]:
        """Return one penalty matrix per marginal direction, in range-space coordinates.

        Because each marginal's basis has already been reprojected onto its penalty range space
        during `fit()`, the marginal penalty in that space is simply the diagonal matrix of its
        retained eigenvalues; this method Kronecker-expands that diagonal matrix into the full
        interaction basis exactly as `TensorProductBasis.penalty_matrices()` does for the ordinary
        (non-reprojected) basis.

        Returns
        -------
        list[NDArray]
            List of `d` matrices, each of shape `(k, k)` where `d` is the number of marginals and
            `k` is the total (range-space) basis dimension.
        """
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
        """Return `0`: every retained coefficient lies in some marginal's penalty range space."""
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        """Return a sum-to-zero constraint on the interaction basis, if derivable.

        Attempts to recover each marginal's training covariate values (via a `_x_train` or
        `_knots` attribute) to build a constraint that forces the interaction surface to have mean
        zero over the training data. If any marginal does not expose its training data in a
        recognized form, no constraint can be derived and `None` is returned.

        Returns
        -------
        NDArray or None
            A `(1, k)` matrix of the mean basis-function values over the training grid, or `None`
            if the training covariates could not be recovered from the marginals.
        """
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
        """Total number of basis functions, the product of the marginal range-space dimensions."""
        self._check_fitted()
        result = 1
        for d in self._range_dims:
            result *= d
        return result

    @property
    def marginals(self) -> list[SmoothBasis]:
        """The list of (fitted, after `fit()`) marginal basis objects."""
        return self._marginals
