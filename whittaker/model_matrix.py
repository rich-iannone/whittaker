"""Model matrix construction: formula + data -> design matrix + penalties.

This module bridges the parsed formula and smooth basis implementations into the full design matrix
and combined penalty structure required by the GAM fitting engine.

Given a parsed `~whittaker.formula.Formula` and a data dictionary, the `build_model_matrix()`
function:

1. Constructs the intercept column (if present).
2. Extracts columns for parametric (linear) terms.
3. Resolves each `~whittaker.formula.SmoothTerm` to its `~whittaker.smooths.SmoothBasis`, fits it,
builds the basis matrix, and applies identifiability constraints (sum-to-zero absorption).
4. Assembles everything into a single design matrix `X`.
5. Builds one block-diagonal penalty matrix `S_j` per smooth term, each expanded to the full model
dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from whittaker.formula.terms import (
    Formula,
    InteractionTerm,
    LinearTerm,
    OffsetTerm,
    SmoothTerm,
)
from whittaker.smooths.adaptive import AdaptiveTPRS
from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.cubic import CRS
from whittaker.smooths.cyclic import CyclicCRS, CyclicPSpline
from whittaker.smooths.duchon import DuchonSpline
from whittaker.smooths.factor_smooth import FactorSmoothBasis
from whittaker.smooths.gp import GaussianProcess
from whittaker.smooths.monotone import ConvexPSpline, MonotonePSpline
from whittaker.smooths.mrf import MRFBasis
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.random import RandomEffectBasis
from whittaker.smooths.shrinkage import ShrinkageCRS, ShrinkageTPRS
from whittaker.smooths.soap_film import SoapFilm
from whittaker.smooths.tensor import (
    TensorInteractionBasis,
    TensorProductBasis,
    TensorProductBasisT2,
)
from whittaker.smooths.tprs import TPRS

_BS_REGISTRY: dict[str, type[SmoothBasis]] = {
    "tp": TPRS,
    "cr": CRS,
    "cc": CyclicCRS,
    "ps": PSpline,
    "cp": CyclicPSpline,
    "ts": ShrinkageTPRS,
    "cs": ShrinkageCRS,
    "re": RandomEffectBasis,
    "ad": AdaptiveTPRS,
    "so": SoapFilm,
    "gp": GaussianProcess,
    "ds": DuchonSpline,
    "mrf": MRFBasis,
    "mpi": MonotonePSpline,
    "mpd": lambda **kw: MonotonePSpline(decreasing=True, **kw),
    "cx": ConvexPSpline,
    "cv": lambda **kw: ConvexPSpline(concave=True, **kw),
}


def _resolve_basis(term: SmoothTerm) -> SmoothBasis:
    """Instantiate a `SmoothBasis` from a parsed `SmoothTerm`."""
    if term.bs == "fs":
        return _resolve_fs_basis(term)

    cls = _BS_REGISTRY.get(term.bs)
    if cls is None:
        supported = ", ".join(sorted(_BS_REGISTRY) + ["fs"])
        raise ValueError(
            f"Unknown basis type bs={term.bs!r} in {term!r}. Supported types: {supported}."
        )

    kwargs: dict[str, Any] = {}
    if term.k != -1:
        kwargs["k"] = term.k

    if term.bs in ("ps", "cp", "mpi", "mpd", "cx", "cv"):
        if "degree" in term.extra:
            kwargs["degree"] = term.extra["degree"]
        if "m" in term.extra:
            kwargs["m"] = term.extra["m"]
    elif term.bs in ("tp", "ts", "ds"):
        if "m" in term.extra:
            kwargs["m"] = term.extra["m"]
    elif term.bs == "ad":
        if "m" in term.extra:
            kwargs["m"] = term.extra["m"]
        if "n_penalties" in term.extra:
            kwargs["n_penalties"] = term.extra["n_penalties"]
    elif term.bs == "so":
        xt = term.extra.get("xt", {})
        if isinstance(xt, dict):
            if "boundary" in xt:
                kwargs["boundary"] = xt["boundary"]
            if "knots" in xt:
                kwargs["knots"] = xt["knots"]
    elif term.bs == "gp":
        xt = term.extra.get("xt", None)
        if isinstance(xt, str):
            kwargs["cov"] = xt
    elif term.bs == "mrf":
        xt = term.extra.get("xt", {})
        if isinstance(xt, dict) and "neighborhood" in xt:
            kwargs["neighborhood"] = xt["neighborhood"]

    return cls(**kwargs)


def _resolve_fs_basis(term: SmoothTerm) -> FactorSmoothBasis:
    """Instantiate a `FactorSmoothBasis` from a parsed `SmoothTerm`."""
    if len(term.variables) < 2:
        raise ValueError(
            f"bs='fs' requires at least 2 variables (numeric + factor), "
            f"got {len(term.variables)} in {term!r}."
        )

    kwargs: dict[str, Any] = {}
    if term.k != -1:
        kwargs["k"] = term.k

    xt = term.extra.get("xt", "tp")
    kwargs["xt"] = xt

    if "m" in term.extra:
        kwargs["m"] = term.extra["m"]

    return FactorSmoothBasis(**kwargs)


def _resolve_tensor_basis(term: SmoothTerm) -> TensorProductBasis:
    """Instantiate a `TensorProductBasis` from a `te()` term."""
    d = len(term.variables)
    if d < 2:
        raise ValueError(f"te() requires at least 2 variables, got {d}.")

    k_list = term.extra.get("k")
    if k_list is None:
        k_per = [term.k] * d if term.k != -1 else [-1] * d
    elif isinstance(k_list, list):
        if len(k_list) != d:
            raise ValueError(
                f"te() got k={k_list} but has {d} variables. "
                f"Length of k must match the number of variables."
            )
        k_per = k_list
    else:
        k_per = [int(k_list)] * d

    bs_spec = term.extra.get("bs")
    if bs_spec is None:
        bs_per = [term.bs] * d
    elif isinstance(bs_spec, list):
        if len(bs_spec) != d:
            raise ValueError(
                f"te() got bs={bs_spec} but has {d} variables. "
                f"Length of bs must match the number of variables."
            )
        bs_per = bs_spec
    else:
        bs_per = [str(bs_spec)] * d

    marginals: list[SmoothBasis] = []
    for j in range(d):
        marginal_term = SmoothTerm(
            variables=(term.variables[j],),
            smooth_type="s",
            bs=bs_per[j],
            k=k_per[j],
            extra={k: v for k, v in term.extra.items() if k not in ("k", "bs")},
        )
        marginals.append(_resolve_basis(marginal_term))

    return TensorProductBasis(marginals)


def _resolve_tensor_interaction_basis(term: SmoothTerm) -> TensorInteractionBasis:
    """Instantiate a `TensorInteractionBasis` from a `ti()` term."""
    d = len(term.variables)
    if d < 2:
        raise ValueError(f"ti() requires at least 2 variables, got {d}.")

    k_list = term.extra.get("k")
    if k_list is None:
        k_per = [term.k] * d if term.k != -1 else [-1] * d
    elif isinstance(k_list, list):
        if len(k_list) != d:
            raise ValueError(
                f"ti() got k={k_list} but has {d} variables. "
                f"Length of k must match the number of variables."
            )
        k_per = k_list
    else:
        k_per = [int(k_list)] * d

    bs_spec = term.extra.get("bs")
    if bs_spec is None:
        bs_per = [term.bs] * d
    elif isinstance(bs_spec, list):
        if len(bs_spec) != d:
            raise ValueError(
                f"ti() got bs={bs_spec} but has {d} variables. "
                f"Length of bs must match the number of variables."
            )
        bs_per = bs_spec
    else:
        bs_per = [str(bs_spec)] * d

    marginals: list[SmoothBasis] = []
    for j in range(d):
        marginal_term = SmoothTerm(
            variables=(term.variables[j],),
            smooth_type="s",
            bs=bs_per[j],
            k=k_per[j],
            extra={k: v for k, v in term.extra.items() if k not in ("k", "bs")},
        )
        marginals.append(_resolve_basis(marginal_term))

    return TensorInteractionBasis(marginals)


def _resolve_t2_basis(term: SmoothTerm) -> TensorProductBasisT2:
    """Instantiate a `TensorProductBasisT2` from a `t2()` term."""
    d = len(term.variables)
    if d < 2:
        raise ValueError(f"t2() requires at least 2 variables, got {d}.")

    k_list = term.extra.get("k")
    if k_list is None:
        k_per = [term.k] * d if term.k != -1 else [-1] * d
    elif isinstance(k_list, list):
        if len(k_list) != d:
            raise ValueError(
                f"t2() got k={k_list} but has {d} variables. "
                f"Length of k must match the number of variables."
            )
        k_per = k_list
    else:
        k_per = [int(k_list)] * d

    bs_spec = term.extra.get("bs")
    if bs_spec is None:
        bs_per = [term.bs] * d
    elif isinstance(bs_spec, list):
        if len(bs_spec) != d:
            raise ValueError(
                f"t2() got bs={bs_spec} but has {d} variables. "
                f"Length of bs must match the number of variables."
            )
        bs_per = bs_spec
    else:
        bs_per = [str(bs_spec)] * d

    marginals: list[SmoothBasis] = []
    for j in range(d):
        marginal_term = SmoothTerm(
            variables=(term.variables[j],),
            smooth_type="s",
            bs=bs_per[j],
            k=k_per[j],
            extra={k: v for k, v in term.extra.items() if k not in ("k", "bs")},
        )
        marginals.append(_resolve_basis(marginal_term))

    return TensorProductBasisT2(marginals)


def _is_factor(arr: NDArray) -> bool:
    """Check if an array should be treated as a factor (categorical) variable."""
    arr = np.asarray(arr)
    return arr.dtype.kind in ("U", "S", "O")


def _extract_by_column(data: dict[str, NDArray], name: str) -> NDArray:
    """Get a by-variable column from *data* without coercing dtype."""
    if name not in data:
        available = ", ".join(sorted(data))
        raise KeyError(
            f"Column {name!r} required by the formula is not in the data. "
            f"Available columns: {available}."
        )
    col = np.asarray(data[name])
    if col.ndim != 1:
        raise ValueError(f"Column {name!r} must be 1-D, got shape {col.shape}.")
    return col


def _extract_column(data: dict[str, NDArray], name: str) -> NDArray:
    """Get a 1-D float array from *data*, raising a clear error if missing."""
    if name not in data:
        available = ", ".join(sorted(data))
        raise KeyError(
            f"Column {name!r} required by the formula is not in the data. "
            f"Available columns: {available}."
        )
    col = np.asarray(data[name], dtype=float)
    if col.ndim != 1:
        raise ValueError(f"Column {name!r} must be 1-D, got shape {col.shape}.")
    return col


def _apply_constraint(B: NDArray, C: NDArray) -> NDArray:
    """Absorb identifiability constraints into the basis matrix.

    Given an `(n, k)` basis matrix *B* and a `(1, k)` constraint row *C* (representing
    `C @ β = 0`), this returns an `(n, k - 1)` matrix whose column space satisfies the constraint.

    The method uses QR decomposition of `C.T` to find a `(k, k - 1)` null-space matrix *Z* such that
    `C @ Z = 0`, then returns `B @ Z`.
    """
    Q, _ = np.linalg.qr(C.T, mode="complete")
    Z = Q[:, C.shape[0] :]  # (k, k - 1)
    return B @ Z


def _apply_constraint_to_penalty(S: NDArray, C: NDArray) -> NDArray:
    """Project the penalty matrix through the same constraint null space."""
    Q, _ = np.linalg.qr(C.T, mode="complete")
    Z = Q[:, C.shape[0] :]  # (k, k - 1)
    return Z.T @ S @ Z


@dataclass
class SmoothInfo:
    """Metadata describing where one smooth term lives inside a `ModelMatrix`.

    Each `~whittaker.formula.terms.SmoothTerm` in a formula (an `s()`, `te()`, `ti()`, or `t2()`
    call) expands into several columns of the full design matrix `X` — one per basis function of
    the fitted `SmoothBasis`. A `SmoothInfo` records the bookkeeping needed to make sense of that
    expansion after the fact: which contiguous slice `X[:, col_start:col_end]` belongs to this
    term, the fitted `SmoothBasis` instance itself (so the exact same knots and constraints can be
    reused later), and which entries of `ModelMatrix.penalties` hold this term's own penalty
    block(s). `build_model_matrix` creates one `SmoothInfo` per smooth term (or per `by=` level,
    for factor-by smooths) and stores the list on `ModelMatrix.smooths`; `predict_matrix` later
    reads these back to evaluate each smooth's basis on new data without needing the original
    `Formula` object.

    Attributes
    ----------
    term : SmoothTerm
        The parsed `SmoothTerm` this info belongs to.
    basis : SmoothBasis
        The fitted `SmoothBasis` instance for this term. Retained so that `predict_matrix` can
        call `basis.basis_matrix()` on new covariate values using the exact knots, degrees of
        freedom, and identifiability constraints learned during fitting, rather than re-fitting a
        new basis.
    col_start : int
        Start column index (inclusive) in the full model matrix `X`.
    col_end : int
        End column index (exclusive) in the full model matrix `X`.
    null_space_dim : int
        Dimension of the penalty null space for this smooth, i.e. the number of basis directions
        (after identifiability constraints) that the penalty does not shrink at all. For an
        ordinary cubic or thin-plate spline this is typically the "linear" part of the smooth;
        for shrinkage bases (`bs="ts"`, `bs="cs"`) and random-effect/factor-smooth bases
        (`bs="re"`, `bs="fs"`) it is `0` because every direction is already penalized.
    penalty_indices : list of int
        Indices into `ModelMatrix.penalties` that belong to this smooth. For `s()` terms this is
        usually a single index (or two, if `select=True` added a null-space penalty); for `te()`
        terms it is one index per marginal direction.
    by_var : str or None
        Name of the `by=` variable for this term, or `None` if the term has no `by=` modifier.
    by_level : str or None
        For factor `by=` variables, the specific level this `SmoothInfo` corresponds to (one
        `SmoothInfo` is created per level); `None` for continuous `by=` variables or terms without a
        `by=` modifier.
    """

    term: SmoothTerm
    basis: SmoothBasis
    col_start: int
    col_end: int
    null_space_dim: int
    penalty_indices: list[int] = field(default_factory=list)
    by_var: str | None = None
    by_level: str | None = None


@dataclass
class ModelMatrix:
    """Numeric design matrix, penalties, and metadata produced by :func:`build_model_matrix`.

    A `ModelMatrix` is the numeric form of a fitted `~whittaker.gam.GAM`'s formula: the design
    matrix `X` such that the linear predictor is `eta = X @ beta` (plus an optional offset),
    together with one quadratic penalty matrix per smooth term and enough metadata (`smooths`,
    `column_names`) to later build a matching prediction matrix or attribute coefficients back to
    individual terms. `GAM.fit()` calls `build_model_matrix` once and stores the result so that
    `predict()`, `summary()`, and `plot()` can all refer back to the same column layout.

    Attributes
    ----------
    X : numpy.ndarray
        Full design matrix, shape `(n, p)` where *n* is the number of observations and *p* is the
        total number of columns (intercept + parametric + all constrained smooth bases).
    penalties : list of numpy.ndarray
        List of `(p, p)` penalty matrices, one per smooth term (or per marginal/null-space
        penalty within a term), each containing that term's penalty embedded in the appropriate
        block of the full model dimension so `beta.T @ S_j @ beta` depends only on that term's
        coefficients.
    smooths : list of SmoothInfo
        Per-smooth metadata (`SmoothInfo`) in formula order.
    column_names : list of str
        Human-readable label for each column of `X`, in order, e.g. `"(Intercept)"`, a covariate
        name for a linear term, or `"s(x)[0]"` for the first basis function of a smooth term.
    has_intercept : bool
        Whether column 0 is the intercept.
    n_parametric : int
        Number of parametric (linear + interaction) columns, not counting the intercept.
    offset : numpy.ndarray or None
        Offset vector of shape `(n,)`, or `None` if no offset term.
    offset_expressions : list of str
        Column expressions summed to form `offset`, retained so `predict_offset` can reconstruct the
        offset for new data.
    response : numpy.ndarray
        The response column as a 1-D float array.
    """

    X: NDArray
    penalties: list[NDArray]
    smooths: list[SmoothInfo] = field(default_factory=list)
    column_names: list[str] = field(default_factory=list)
    has_intercept: bool = True
    n_parametric: int = 0
    offset: NDArray | None = None
    offset_expressions: list[str] = field(default_factory=list)
    response: NDArray = field(default_factory=lambda: np.empty(0))

    @property
    def n_obs(self) -> int:
        """Number of observations used to build the design matrix.

        Returns
        -------
        int
            The number of rows of `X`, i.e. `X.shape[0]`.
        """
        return self.X.shape[0]

    @property
    def n_coefs(self) -> int:
        """Total number of model coefficients.

        Counts every column of `X`: the intercept (if present), all parametric
        (linear/interaction) columns, and every basis function of every smooth term after
        identifiability constraints have been applied.

        Returns
        -------
        int
            The number of columns of `X`, i.e. `X.shape[1]`.
        """
        return self.X.shape[1]

    @property
    def penalty_matrix(self) -> NDArray:
        r"""Combined penalty `S_total = sum(S_j)` (unweighted by `lambda`).

        Sums every entry of `penalties` element-wise into a single `(n_coefs, n_coefs)` matrix,
        without applying any per-smooth smoothing parameter. This is a convenience for inspecting
        the overall penalty structure; it is *not* what the fitting engine
        (`~whittaker.pirls.pirls_fit`) actually optimizes against, since each term's contribution
        should be weighted by its own
        `lambda_j` before being summed. Use `penalties` directly, combined with the fitted
        smoothing parameters, for anything that needs the weighted penalty.

        Returns
        -------
        numpy.ndarray
            An `(n_coefs, n_coefs)` matrix, or an all-zeros matrix of that shape if there are no
            penalized (smooth) terms.
        """
        if not self.penalties:
            return np.zeros((self.n_coefs, self.n_coefs))
        return sum(self.penalties)  # type: ignore[return-value]


def _null_space_penalty(S: NDArray) -> NDArray:
    """Construct a penalty that penalizes the null space of *S*."""
    eigvals, eigvecs = np.linalg.eigh(S)
    threshold = 1e-10 * max(eigvals.max(), 1.0)
    null_mask = eigvals < threshold
    U_null = eigvecs[:, null_mask]
    S_null = U_null @ U_null.T
    return (S_null + S_null.T) * 0.5


def build_model_matrix(
    formula: Formula,
    data: dict[str, NDArray],
    *,
    apply_constraints: bool = True,
    select: bool = False,
) -> ModelMatrix:
    r"""Assemble the full design matrix and penalty structure from a formula.

    A generalized additive model is fit by turning the covariates into a purely numeric *model
    matrix* (also called a *design matrix*) `X` — an `(n, p)` array of `n` observations by `p`
    coefficients — such that the linear predictor is simply a matrix-vector product:

    $$
    \eta = X \beta + \text{offset}
    $$

    Every term in the formula contributes one or more columns to `X`. An intercept contributes a
    single column of ones; a linear term `x3` contributes the column `x3` itself; an interaction
    `x1:x2` contributes the elementwise product `x1 * x2`. A smooth term such as `s(x)`, by
    contrast, expands into *several* columns: the basis functions of the underlying spline (or
    other smooth) basis, evaluated at each observation's value of `x`. Fitting a GAM is then no
    different from fitting a (penalized) linear model in these expanded columns — the
    non-linearity of `f(x)` lives entirely in how the basis functions are constructed, not in how
    the coefficients enter the model.

    `build_model_matrix` is the bridge between a parsed `~whittaker.formula.terms.Formula` and the
    numeric matrices the P-IRLS fitting engine (`~whittaker.pirls.pirls_fit`) actually needs: the
    design matrix `X` plus, for each smooth term, a quadratic penalty matrix `S_j` that penalizes
    wiggliness via `beta.T @ S_j @ beta`. Each `S_j` starts out sized to just that term's own basis
    functions but is expanded ("embedded") to the full `(p, p)` model dimension, with zeros
    everywhere outside that term's column block, so that summing `lambda_j * S_j` over all terms
    gives a single penalty matrix that can be added directly to the unpenalized normal equations.

    Parameters
    ----------
    formula : Formula
        A parsed `~whittaker.formula.terms.Formula`, typically produced by
        `~whittaker.formula.parser.parse`.
    data : dict[str, numpy.ndarray]
        Column-oriented data as `{name: 1-D array}`. Every column referenced by the formula must be
        present. All arrays must have the same length.
    apply_constraints : bool
        If `True` (the default), apply sum-to-zero identifiability constraints to each smooth term
        so the intercept is identifiable.
    select : bool
        If `True`, add an extra penalty on each smooth's null space so that terms can be penalized
        to zero entirely (double penalty approach, Marra & Wood 2011). This enables automatic smooth
        selection via GCV or REML. Smooths that already have `null_space_dim == 0` (e.g. `bs="ts"`,
        `bs="cs"`, `bs="re"`, `bs="fs"`) are unaffected.

    Returns
    -------
    ModelMatrix
        Bundled design matrix, penalties, and metadata.

    Raises
    ------
    KeyError
        If a required column is missing from *data*.
    ValueError
        If an unsupported basis type is requested.

    Notes
    -----
    An ordinary smoothing penalty `S` for a term like `s(x)` typically has a non-trivial null
    space — directions in coefficient space that the penalty does not shrink at all (e.g. the
    linear component of a cubic spline). This means increasing that term's smoothing parameter
    `lambda_j` can flatten the smooth toward a straight line, but never all the way to zero, so
    ordinary GCV/REML smoothing-parameter selection cannot remove an irrelevant term from the
    model entirely.

    When `select=True`, `build_model_matrix` adds a second penalty matrix per term (see
    `_null_space_penalty`) built from the eigendecomposition of the term's own penalty `S`: the
    eigenvectors with (numerically) zero eigenvalue span exactly the null space of `S`, and
    projecting onto that eigenspace gives a penalty `S_null` that penalizes only those
    previously-unpenalized directions. With both `S` and `S_null` present (each with its own
    smoothing parameter), driving both `lambda_j` and the null-space smoothing parameter to large
    values shrinks the *entire* term, including its linear component, to zero — allowing REML- or
    GCV-based fitting to perform automatic term selection much like a lasso penalty does for
    linear models. See Marra, G. and Wood, S.N. (2011), "Practical variable selection for
    generalized additive models", *Computational Statistics & Data Analysis*, 55(7), 2372-2387.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.formula.parser import parse
    from whittaker.model_matrix import build_model_matrix

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 50))
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=50)

    formula = parse("y ~ s(x)")
    model_matrix = build_model_matrix(formula, {"x": x, "y": y})

    model_matrix.X.shape
    ```

    ```{python}
    model_matrix.column_names[:5]
    ```
    """
    # --- validate lengths ---------------------------------------------------
    lengths = {name: len(arr) for name, arr in data.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        detail = ", ".join(f"{k}: {v}" for k, v in lengths.items())
        raise ValueError(f"All data columns must have the same length. Got: {detail}.")
    n = next(iter(lengths.values())) if lengths else 0

    # --- response -----------------------------------------------------------
    response = _extract_column(data, formula.response)

    # --- intercept ----------------------------------------------------------
    blocks: list[NDArray] = []
    col_names: list[str] = []

    if formula.intercept:
        blocks.append(np.ones((n, 1)))
        col_names.append("(Intercept)")

    # --- parametric terms ---------------------------------------------------
    n_parametric = 0
    offset: NDArray | None = None
    offset_expressions: list[str] = []

    for term in formula.terms:
        if isinstance(term, LinearTerm):
            col = _extract_column(data, term.variable)
            blocks.append(col[:, np.newaxis])
            col_names.append(term.variable)
            n_parametric += 1

        elif isinstance(term, InteractionTerm):
            left = _extract_column(data, term.left)
            right = _extract_column(data, term.right)

            if term.full:
                blocks.append(left[:, np.newaxis])
                col_names.append(term.left)
                blocks.append(right[:, np.newaxis])
                col_names.append(term.right)
                n_parametric += 2

            interaction = left * right
            blocks.append(interaction[:, np.newaxis])
            col_names.append(f"{term.left}:{term.right}")
            n_parametric += 1

        elif isinstance(term, OffsetTerm):
            offset_col = _extract_column(data, term.expression)
            offset_expressions.append(term.expression)
            offset = offset_col.copy() if offset is None else offset + offset_col

    # --- smooth terms -------------------------------------------------------
    smooth_infos: list[SmoothInfo] = []
    penalty_blocks: list[tuple[int, int, NDArray]] = []

    for term in formula.terms:
        if not isinstance(term, SmoothTerm):
            continue

        if term.smooth_type == "te":
            basis = _resolve_tensor_basis(term)
        elif term.smooth_type == "ti":
            basis = _resolve_tensor_interaction_basis(term)
        elif term.smooth_type == "t2":
            basis = _resolve_t2_basis(term)
        elif term.smooth_type == "s":
            basis = _resolve_basis(term)
        else:
            raise NotImplementedError(
                f"Smooth type {term.smooth_type!r} is not yet supported. "
                "Only s(), te(), ti(), and t2() are implemented."
            )

        if isinstance(basis, FactorSmoothBasis):
            x_numeric = _extract_column(data, term.variables[0])
            x_factor = _extract_by_column(data, term.variables[1])
            basis.fit(x_numeric, x_factor)
            basis_mat = basis.basis_matrix(x_numeric, x_factor)
            nsd = basis.null_space_dimension()
        elif isinstance(basis, (RandomEffectBasis, MRFBasis)):
            x = _extract_by_column(data, term.variables[0])
            basis.fit(x)
            basis_mat = basis.basis_matrix(x)
            nsd = basis.null_space_dimension()
        else:
            if len(term.variables) == 1:
                x = _extract_column(data, term.variables[0])
            else:
                x = np.column_stack([_extract_column(data, v) for v in term.variables])
            basis.fit(x)
            basis_mat = basis.basis_matrix(x)
            nsd = basis.null_space_dimension()

        if hasattr(basis, "penalty_matrices"):
            pen_mats = basis.penalty_matrices()
        else:
            pen_mats = [basis.penalty_matrix()]

        # by= variables skip identifiability constraints (following mgcv convention)
        has_by = term.by is not None
        if apply_constraints and not has_by:
            constraint = basis.identifiability_constraints()
            if constraint is not None:
                basis_mat = _apply_constraint(basis_mat, constraint)
                pen_mats = [_apply_constraint_to_penalty(pm, constraint) for pm in pen_mats]
                nsd = max(nsd - constraint.shape[0], 0)

        if select and nsd > 0:
            S_null = _null_space_penalty(pen_mats[0])
            pen_mats.append(S_null)
            nsd = 0

        if has_by:
            by_col = _extract_by_column(data, term.by)

            if _is_factor(by_col):
                levels = sorted(np.unique(by_col))
                for level in levels:
                    indicator = (by_col == level).astype(float)
                    basis_mat_level = basis_mat * indicator[:, np.newaxis]

                    k_eff = basis_mat_level.shape[1]
                    col_start = sum(b.shape[1] for b in blocks)
                    col_end = col_start + k_eff

                    blocks.append(basis_mat_level)
                    for j in range(k_eff):
                        col_names.append(f"{term!r}:{level}[{j}]")

                    pen_start_idx = len(penalty_blocks)
                    for pm in pen_mats:
                        penalty_blocks.append((col_start, col_end, pm))
                    pen_indices = list(range(pen_start_idx, pen_start_idx + len(pen_mats)))

                    smooth_infos.append(
                        SmoothInfo(
                            term=term,
                            basis=basis,
                            col_start=col_start,
                            col_end=col_end,
                            null_space_dim=nsd,
                            penalty_indices=pen_indices,
                            by_var=term.by,
                            by_level=str(level),
                        )
                    )
            else:
                by_vals = np.asarray(by_col, dtype=float)
                basis_mat_by = basis_mat * by_vals[:, np.newaxis]

                k_eff = basis_mat_by.shape[1]
                col_start = sum(b.shape[1] for b in blocks)
                col_end = col_start + k_eff

                blocks.append(basis_mat_by)
                for j in range(k_eff):
                    col_names.append(f"{term!r}[{j}]")

                pen_start_idx = len(penalty_blocks)
                for pm in pen_mats:
                    penalty_blocks.append((col_start, col_end, pm))
                pen_indices = list(range(pen_start_idx, pen_start_idx + len(pen_mats)))

                smooth_infos.append(
                    SmoothInfo(
                        term=term,
                        basis=basis,
                        col_start=col_start,
                        col_end=col_end,
                        null_space_dim=nsd,
                        penalty_indices=pen_indices,
                        by_var=term.by,
                    )
                )
        else:
            k_eff = basis_mat.shape[1]
            col_start = sum(b.shape[1] for b in blocks)
            col_end = col_start + k_eff

            blocks.append(basis_mat)
            for j in range(k_eff):
                label = f"{term!r}[{j}]"
                col_names.append(label)

            pen_start_idx = len(penalty_blocks)
            for pm in pen_mats:
                penalty_blocks.append((col_start, col_end, pm))
            pen_indices = list(range(pen_start_idx, pen_start_idx + len(pen_mats)))

            smooth_infos.append(
                SmoothInfo(
                    term=term,
                    basis=basis,
                    col_start=col_start,
                    col_end=col_end,
                    null_space_dim=nsd,
                    penalty_indices=pen_indices,
                )
            )

    # --- assemble X ---------------------------------------------------------
    if not blocks:
        raise ValueError("The formula produces no model columns.")

    X = np.column_stack(blocks)  # (n, p)
    p = X.shape[1]

    # --- expand penalties to full model size ---------------------------------
    penalties: list[NDArray] = []
    for col_start, col_end, S_block in penalty_blocks:
        S_full = np.zeros((p, p))
        S_full[col_start:col_end, col_start:col_end] = S_block
        penalties.append(S_full)

    return ModelMatrix(
        X=X,
        penalties=penalties,
        smooths=smooth_infos,
        column_names=col_names,
        has_intercept=formula.intercept,
        n_parametric=n_parametric,
        offset=offset,
        offset_expressions=offset_expressions,
        response=response,
    )


def predict_matrix(
    model: ModelMatrix,
    new_data: dict[str, NDArray],
) -> NDArray:
    """Build the prediction design matrix for new data.

    Predicting from a fitted GAM requires evaluating each term at new covariate values in
    *exactly* the same numeric representation used during fitting. For a linear term this is
    trivial (just read off the new column), but for a smooth term it is not: simply re-fitting a
    fresh basis on the new data would choose different knots, degrees of freedom, and
    identifiability constraints, producing a matrix whose columns don't line up with the fitted
    coefficients `beta` at all. `predict_matrix` avoids this by reusing the exact `SmoothBasis`
    objects stored on `model.smooths[i].basis` — the same knots and constraints the model was
    fitted with — and simply evaluating `basis.basis_matrix()` at the new covariate values, so the
    resulting columns are directly compatible with the training-time coefficients.

    Because a `ModelMatrix` does not retain the original `~whittaker.formula.terms.Formula` object
    (only its numeric consequences), `predict_matrix` first calls `_reconstruct_formula` to rebuild
    an equivalent `Formula` from `model.column_names` and `model.smooths`, then walks that
    formula's terms in order — extracting linear/interaction columns directly from `new_data` and
    re-evaluating each smooth's stored basis — to assemble a new design matrix with the same
    column layout as `model.X`.

    Parameters
    ----------
    model : ModelMatrix
        A `ModelMatrix` previously returned by `build_model_matrix()`.
    new_data : dict[str, numpy.ndarray]
        Column-oriented new data. Must contain every covariate column referenced by the original
        formula (the response column is not needed).

    Returns
    -------
    numpy.ndarray
        Design matrix of shape `(n_new, p)` with the same column layout as `model.X`.

    Notes
    -----
    `_reconstruct_formula` infers term boundaries purely from `column_names` and the stored
    `SmoothInfo` objects, not from any record of the literal original formula string. In practice
    this reconstruction always agrees with the training matrix's column layout, since it is
    derived from the very structures that layout was built from — but the reconstructed
    `Formula`'s term order and string labels are *inferred*, and should not be treated as a
    faithful reproduction of the original formula object passed to `build_model_matrix` (for
    example, `full=True` interactions are not reconstructed exactly as originally specified).

    Examples
    --------
    Continuing from the `build_model_matrix` example above, predict at new values of `x` using the
    same fitted basis:

    ```{python}
    import numpy as np
    from whittaker.formula.parser import parse
    from whittaker.model_matrix import build_model_matrix, predict_matrix

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 50))
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=50)

    model_matrix = build_model_matrix(parse("y ~ s(x)"), {"x": x, "y": y})

    new_x = np.linspace(0, 1, 5)
    X_new = predict_matrix(model_matrix, {"x": new_x})
    X_new.shape
    ```
    """
    lengths = {name: len(arr) for name, arr in new_data.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        detail = ", ".join(f"{k}: {v}" for k, v in lengths.items())
        raise ValueError(f"All data columns must have the same length. Got: {detail}.")
    n_new = next(iter(lengths.values())) if lengths else 0

    formula = _reconstruct_formula(model)

    blocks: list[NDArray] = []

    if model.has_intercept:
        blocks.append(np.ones((n_new, 1)))

    for term in formula.terms:
        if isinstance(term, LinearTerm):
            col = _extract_column(new_data, term.variable)
            blocks.append(col[:, np.newaxis])

        elif isinstance(term, InteractionTerm):
            left = _extract_column(new_data, term.left)
            right = _extract_column(new_data, term.right)
            if term.full:
                blocks.append(left[:, np.newaxis])
                blocks.append(right[:, np.newaxis])
            blocks.append((left * right)[:, np.newaxis])

    for info in model.smooths:
        term = info.term
        if isinstance(info.basis, FactorSmoothBasis):
            x_numeric = _extract_column(new_data, term.variables[0])
            x_factor = _extract_by_column(new_data, term.variables[1])
            basis_mat = info.basis.basis_matrix(x_numeric, x_factor)
        elif isinstance(info.basis, (RandomEffectBasis, MRFBasis)):
            x = _extract_by_column(new_data, term.variables[0])
            basis_mat = info.basis.basis_matrix(x)
        elif len(term.variables) == 1:
            x = _extract_column(new_data, term.variables[0])
            basis_mat = info.basis.basis_matrix(x)
        else:
            x = np.column_stack([_extract_column(new_data, v) for v in term.variables])
            basis_mat = info.basis.basis_matrix(x)

        has_by = info.by_var is not None
        if not has_by:
            constraint = info.basis.identifiability_constraints()
            if constraint is not None and (info.col_end - info.col_start) < info.basis.n_basis:
                basis_mat = _apply_constraint(basis_mat, constraint)

        if info.by_level is not None:
            by_col = _extract_by_column(new_data, info.by_var)
            indicator = (by_col == info.by_level).astype(float)
            basis_mat = basis_mat * indicator[:, np.newaxis]
        elif info.by_var is not None:
            by_vals = _extract_column(new_data, info.by_var)
            basis_mat = basis_mat * by_vals[:, np.newaxis]

        blocks.append(basis_mat)

    return np.column_stack(blocks)


def predict_offset(
    model: ModelMatrix,
    new_data: dict[str, NDArray],
) -> NDArray | None:
    """Extract the offset vector for new data.

    Parameters
    ----------
    model:
        A `ModelMatrix` previously returned by `build_model_matrix()`.
    new_data:
        Column-oriented new data.

    Returns
    -------
    NDArray | None
        Offset vector of shape `(n_new,)`, or `None` if the model has no offset terms.
    """
    if not model.offset_expressions:
        return None

    offset: NDArray | None = None
    for expr in model.offset_expressions:
        col = _extract_column(new_data, expr)
        offset = col.copy() if offset is None else offset + col
    return offset


def _reconstruct_formula(model: ModelMatrix) -> Formula:
    """Recover the original formula terms from a ModelMatrix.

    This is a helper for `predict_matrix()`: it reads `column_names` and smooths to rebuild the term
    list in the correct order.
    """
    terms = []
    col_idx = 1 if model.has_intercept else 0

    names = model.column_names
    smooth_starts = {s.col_start for s in model.smooths}
    smooth_by_start = {s.col_start: s for s in model.smooths}

    while col_idx < len(names):
        if col_idx in smooth_starts:
            info = smooth_by_start[col_idx]
            terms.append(info.term)
            col_idx = info.col_end
        else:
            name = names[col_idx]
            if ":" in name:
                left, right = name.split(":", 1)
                terms.append(InteractionTerm(left=left, right=right, full=False))
                col_idx += 1
            else:
                terms.append(LinearTerm(variable=name))
                col_idx += 1

    return Formula(response="y", terms=terms, intercept=model.has_intercept)
