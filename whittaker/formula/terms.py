"""Term representations for model formulas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class LinearTerm:
    """A plain linear (parametric) term, e.g. `x1` or `group`."""

    variable: str

    def __repr__(self) -> str:
        return self.variable


@dataclass
class InteractionTerm:
    """A two-way interaction term.

    `full=True` (`*`) includes both main effects and the interaction.
    `full=False` (`:`) includes the interaction only.
    """

    left: str
    right: str
    full: bool = True

    def __repr__(self) -> str:
        op = "*" if self.full else ":"
        return f"{self.left} {op} {self.right}"


@dataclass
class OffsetTerm:
    """An offset term, e.g. `offset(log_exposure)`.

    The `expression` is the raw string inside `offset()`.
    """

    expression: str

    def __repr__(self) -> str:
        return f"offset({self.expression})"


@dataclass
class SmoothTerm:
    """A smooth term, e.g. `s(x1, bs='cr', k=10)` or `te(x1, x2)`.

    Parameters
    ----------
    variables:
        Column names that are arguments to the smooth function.
    smooth_type:
        One of `"s"`, `"te"`, `"ti"`, `"t2"`.
    bs:
        Basis type (e.g. `"tp"`, `"cr"`, `"ps"`, `"cc"`).
        Defaults to `"tp"` (thin plate regression splines).
    k:
        Number of basis functions.  `-1` means auto-select.
    by:
        Name of a factor or numeric column for factor-by or varying-coefficient
        smooths (the `by=` argument in mgcv).
    extra:
        Any additional keyword arguments passed to the smooth constructor.
    """

    variables: tuple[str, ...]
    smooth_type: str = "s"
    bs: str = "tp"
    k: int = -1
    by: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts: list[str] = list(self.variables)
        if self.bs != "tp":
            parts.append(f"bs={self.bs!r}")
        if self.k != -1:
            parts.append(f"k={self.k}")
        if self.by is not None:
            parts.append(f"by={self.by!r}")
        for key, val in self.extra.items():
            parts.append(f"{key}={val!r}")
        return f"{self.smooth_type}({', '.join(parts)})"


#: Union type covering all concrete term types.
Term = Union[LinearTerm, SmoothTerm, InteractionTerm, OffsetTerm]


@dataclass
class Formula:
    """A parsed model formula.

    Parameters
    ----------
    response:
        Name of the response variable (left of `~`).
    terms:
        Ordered list of model terms (right of `~`).
    intercept:
        Whether the model includes an intercept.  Suppress with `0 +` or
        `- 1` in the formula string.
    """

    response: str
    terms: list[Term]
    intercept: bool = True

    def required_columns(self) -> list[str]:
        """Return every column name referenced in the formula, deduplicated."""
        seen: dict[str, None] = {self.response: None}
        for term in self.terms:
            if isinstance(term, LinearTerm):
                seen[term.variable] = None
            elif isinstance(term, SmoothTerm):
                for v in term.variables:
                    seen[v] = None
                if term.by is not None:
                    seen[term.by] = None
            elif isinstance(term, InteractionTerm):
                seen[term.left] = None
                seen[term.right] = None
            # OffsetTerm: may be an expression, not a bare column name — skip
        return list(seen)

    def __repr__(self) -> str:
        rhs_parts: list[str] = []
        if not self.intercept:
            rhs_parts.append("0")
        rhs_parts.extend(repr(t) for t in self.terms)
        return f"{self.response} ~ {' + '.join(rhs_parts)}"
