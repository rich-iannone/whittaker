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


