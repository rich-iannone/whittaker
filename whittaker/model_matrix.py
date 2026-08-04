"""Model matrix construction: formula + data → design matrix + penalties.

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
from whittaker.smooths.base import SmoothBasis
from whittaker.smooths.cubic import CRS
from whittaker.smooths.pspline import PSpline
from whittaker.smooths.tprs import TPRS

_BS_REGISTRY: dict[str, type[SmoothBasis]] = {
    "tp": TPRS,
    "cr": CRS,
    "ps": PSpline,
}


def _resolve_basis(term: SmoothTerm) -> SmoothBasis:
    """Instantiate a :class:`SmoothBasis` from a parsed :class:`SmoothTerm`."""
    cls = _BS_REGISTRY.get(term.bs)
    if cls is None:
        supported = ", ".join(sorted(_BS_REGISTRY))
        raise ValueError(
            f"Unknown basis type bs={term.bs!r} in {term!r}. Supported types: {supported}."
        )

    kwargs: dict[str, Any] = {}
    if term.k != -1:
        kwargs["k"] = term.k

    if term.bs == "ps":
        if "degree" in term.extra:
            kwargs["degree"] = term.extra["degree"]
        if "m" in term.extra:
            kwargs["m"] = term.extra["m"]
    elif term.bs == "tp":
        if "m" in term.extra:
            kwargs["m"] = term.extra["m"]

    return cls(**kwargs)


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

    Given an `(n, k)` basis matrix *B* and a `(1, k)` constraint row *C* (representing `C @ β = 0`),
    this returns an `(n, k - 1)` matrix whose column space satisfies the constraint.

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
    """Metadata and matrices for a single smooth term in the model.

    Attributes
    ----------
    term:
        The parsed `SmoothTerm` this info belongs to.
    basis:
        The fitted `SmoothBasis` instance.
    col_start:
        Start column index (inclusive) in the full model matrix `X`.
    col_end:
        End column index (exclusive) in the full model matrix `X`.
    null_space_dim:
        Dimension of the penalty null space for this smooth.
    """

    term: SmoothTerm
    basis: SmoothBasis
    col_start: int
    col_end: int
    null_space_dim: int


@dataclass
class ModelMatrix:
    """Result of :func:`build_model_matrix`.

    Attributes
    ----------
    X:
        Full design matrix, shape `(n, p)` where *p* is the total number of columns (intercept +
        parametric + all constrained smooth bases).
    penalties:
        List of `(p, p)` penalty matrices, one per smooth term, each containing that term's penalty
        embedded in the appropriate block of the full model dimension.
    smooths:
        Per-smooth metadata (`SmoothInfo`) in formula order.
    column_names:
        Human-readable label for each column of `X`, in order.
    has_intercept:
        Whether column 0 is the intercept.
    n_parametric:
        Number of parametric (linear + interaction) columns, not counting the intercept.
    offset:
        Offset vector of shape `(n,)`, or `None` if no offset term.
    response:
        The response column as a 1-D float array (`numpy.ndarray`).
    """

    X: NDArray
    penalties: list[NDArray]
    smooths: list[SmoothInfo] = field(default_factory=list)
    column_names: list[str] = field(default_factory=list)
    has_intercept: bool = True
    n_parametric: int = 0
    offset: NDArray | None = None
    response: NDArray = field(default_factory=lambda: np.empty(0))

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.X.shape[0]

    @property
    def n_coefs(self) -> int:
        """Total number of model coefficients (columns of ``X``)."""
        return self.X.shape[1]

    @property
    def penalty_matrix(self) -> NDArray:
        """Combined penalty `S_total = sum(S_j)` (unweighted by λ).

        Useful as a quick reference; the fitting engine should use `penalties` with per-smooth λ
        weights.
        """
        if not self.penalties:
            return np.zeros((self.n_coefs, self.n_coefs))
        return sum(self.penalties)  # type: ignore[return-value]


def build_model_matrix(
    formula: Formula,
    data: dict[str, NDArray],
    *,
    apply_constraints: bool = True,
) -> ModelMatrix:
    """Assemble the full design matrix and penalty structure from a formula.

    Parameters
    ----------
    formula:
        A parsed `~whittaker.formula.Formula`.
    data:
        Column-oriented data as `{name: 1-D array}`. Every column referenced by the formula must be
        present. All arrays must have the same length.
    apply_constraints:
        If `True` (the default), apply sum-to-zero identifiability constraints to each smooth term
        so the intercept is identifiable.

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
            if offset is None:
                offset = offset_col.copy()
            else:
                offset = offset + offset_col

    # --- smooth terms -------------------------------------------------------
    smooth_infos: list[SmoothInfo] = []
    penalty_blocks: list[tuple[int, int, NDArray]] = []

    for term in formula.terms:
        if not isinstance(term, SmoothTerm):
            continue

        if term.smooth_type != "s":
            raise NotImplementedError(
                f"Smooth type {term.smooth_type!r} is not yet supported. Only s() is implemented."
            )

        basis = _resolve_basis(term)

        if len(term.variables) == 1:
            x = _extract_column(data, term.variables[0])
        else:
            x = np.column_stack([_extract_column(data, v) for v in term.variables])

        basis.fit(x)

        basis_mat = basis.basis_matrix(x)  # (n, k)
        pen_mat = basis.penalty_matrix()  # (k, k)
        nsd = basis.null_space_dimension()

        if apply_constraints:
            constraint = basis.identifiability_constraints()
            if constraint is not None:
                basis_mat = _apply_constraint(basis_mat, constraint)
                pen_mat = _apply_constraint_to_penalty(pen_mat, constraint)
                nsd = max(nsd - constraint.shape[0], 0)

        k_eff = basis_mat.shape[1]
        col_start = sum(b.shape[1] for b in blocks)
        col_end = col_start + k_eff

        blocks.append(basis_mat)
        for j in range(k_eff):
            label = f"{term!r}[{j}]"
            col_names.append(label)

        smooth_infos.append(
            SmoothInfo(
                term=term,
                basis=basis,
                col_start=col_start,
                col_end=col_end,
                null_space_dim=nsd,
            )
        )
        penalty_blocks.append((col_start, col_end, pen_mat))

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
        response=response,
    )


def predict_matrix(
    model: ModelMatrix,
    new_data: dict[str, NDArray],
) -> NDArray:
    """Build the prediction design matrix for new data.

    Re-uses the fitted smooth bases stored in *model* to evaluate basis matrices at new covariate
    values, then assembles the same column structure as the training matrix.

    Parameters
    ----------
    model:
        A `ModelMatrix` previously returned by `build_model_matrix()`.
    new_data:
        Column-oriented new data.

    Returns
    -------
    NDArray
        Design matrix of shape `(n_new, p)` with the same column layout as `model.X`.
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
        if len(term.variables) == 1:
            x = _extract_column(new_data, term.variables[0])
        else:
            x = np.column_stack([_extract_column(new_data, v) for v in term.variables])

        basis_mat = info.basis.basis_matrix(x)

        constraint = info.basis.identifiability_constraints()
        if constraint is not None and (info.col_end - info.col_start) < info.basis.n_basis:
            basis_mat = _apply_constraint(basis_mat, constraint)

        blocks.append(basis_mat)

    return np.column_stack(blocks)


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
