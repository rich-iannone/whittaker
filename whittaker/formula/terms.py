r"""Term representations for model formulas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class LinearTerm:
    """A plain linear (parametric) term, e.g. `x1` or `group`.

    Represents a bare covariate name on the right-hand side of a formula — anything that is not
    wrapped in `s()`, `te()`, `ti()`, `t2()`, or `offset()`, and does not use `*`/`:` interaction
    syntax. The column is entered into the model matrix unmodified (numeric columns as a single
    linear column; string/categorical columns are expanded to dummy indicator columns) and
    contributes a single unpenalized coefficient (or one per non-reference level, for a factor)
    to the linear predictor. Use a `LinearTerm` for effects you want to assume are linear, or for
    categorical covariates; use a `SmoothTerm` when you want the data to determine the shape of
    the relationship.

    Parameters
    ----------
    variable : str
        Name of the data column this term refers to.
    """

    variable: str

    def __repr__(self) -> str:
        return self.variable


@dataclass
class InteractionTerm:
    """A two-way parametric interaction between two bare covariates, e.g. `x1 * x2`.

    Represents a crossing of two columns in the model matrix (numeric-by-numeric products,
    numeric-by-factor varying slopes, or factor-by-factor cell means, depending on the column
    types). Use this when you want an interaction *without* smoothing — for a smooth interaction
    surface between two continuous covariates, use a tensor-product `SmoothTerm` (`te()`, `ti()`,
    or `t2()`) instead, or a factor-`by` `SmoothTerm` for "one smooth per group".

    `full=True` corresponds to `*`-style crossing (R's convention): both main effects (`x1` and
    `x2` individually) plus the interaction column(s) are included. `full=False` corresponds to
    `:`-style crossing: the interaction only, with no accompanying main effects. Note that the
    parser (`whittaker.formula.parser.parse`) currently only produces terms with `full=True`
    (`x1 * x2` syntax); `x1:x2` colon syntax is not valid Python and is rejected at parse time, so
    `full=False` terms must be constructed directly if needed.

    Parameters
    ----------
    left : str
        Name of the first covariate.
    right : str
        Name of the second covariate.
    full : bool
        If `True` (default), include both main effects and the interaction (`*`-style). If
        `False`, include only the interaction, dropping the main effects (`:`-style).
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

    An offset is a covariate whose coefficient is fixed at exactly `1` rather than estimated —
    it is added directly onto the linear predictor unpenalized and unscaled. The canonical use
    case is modeling rates with a count-based family: for a Poisson model of event counts with
    varying exposure times, `offset(log(exposure))` lets the model fit the rate per unit exposure
    while accounting for the fact that longer exposures mechanically produce more events (since
    `log(mu) = eta + log(exposure)` implies `mu / exposure = exp(eta)`, the rate).

    The `expression` attribute stores the raw, unparsed text found inside `offset(...)` in the
    formula string (e.g. `"log(exposure)"` or `"log_exposure"`) — expression evaluation, if any,
    happens later when the model matrix is built.

    Parameters
    ----------
    expression : str
        Raw expression text inside `offset(...)`.
    """

    expression: str

    def __repr__(self) -> str:
        return f"offset({self.expression})"


@dataclass
class SmoothTerm:
    r"""A smooth term, e.g. `s(x1, bs='cr', k=10)` or `te(x1, x2)`.

    Represents an unspecified smooth function $f(\cdot)$ of one or more covariates, entered
    into the model matrix as a spline basis expansion with an associated wiggliness penalty. The
    penalty's strength (the smoothing parameter $\lambda$) is estimated automatically when the
    `GAM` is fit, rather than being a free choice like the degree of a polynomial term — this is
    what makes the term "smooth" in the GAM sense rather than a fixed parametric basis expansion.

    `smooth_type` controls how multiple `variables` are combined:

    - `"s"`: a single (marginal) smooth. Most common for one variable; for two or more it fits a
      single isotropic basis (e.g. a thin-plate spline over `(x1, x2)` jointly), appropriate when
      the covariates share the same scale/units.
    - `"te"`: a full tensor-product smooth. Builds a separate marginal basis for each variable
      and forms their tensor (outer) product, with one smoothing parameter per marginal
      direction. Appropriate for interactions between covariates on different scales, and
      implicitly includes the main effects of each variable.
    - `"ti"`: a tensor-product *interaction* smooth. Like `"te"`, but each marginal's penalty
      null space (e.g. the linear component) is projected out first, so the term captures only
      the pure interaction with no main-effect content. Used for an ANOVA-style decomposition,
      e.g. `s(x1) + s(x2) + ti(x1, x2)` separates main effects from their interaction.
    - `"t2"`: an alternative tensor-product parameterization to `"te"`, decomposing the penalty
      over every non-empty subset of the marginal directions ($2^d - 1$ penalties for $d$
      variables) so each interaction order gets its own smoothing parameter.

    Parameters
    ----------
    variables : tuple[str, ...]
        Column names that are arguments to the smooth function. A single name for `"s"` (or two
        or more for a multivariate `"s"`); two or more names are required for `"te"`, `"ti"`, and
        `"t2"`.
    smooth_type : str
        One of `"s"`, `"te"`, `"ti"`, `"t2"` (see above). Defaults to `"s"`.
    bs : str
        Basis type. Common values include:

        - `"tp"` (default): thin plate regression spline — a good general-purpose default with
          no need to place knots.
        - `"cr"`: cubic regression spline.
        - `"cc"`: cyclic (periodic) cubic regression spline, for covariates such as day-of-year
          or angle where the ends of the range should meet smoothly.
        - `"ps"`: P-spline (B-spline basis with a discrete difference penalty).
        - `"cp"`: cyclic P-spline.
        - `"ts"` / `"cs"`: shrinkage versions of `"tp"` / `"cr"` with an extra penalty on the
          null space, useful for automatic term selection without `select=True`.
        - `"re"`: random-effect basis (one ridge-penalized column per factor level), for
          smooth-random-intercept terms.
        - `"fs"`: factor-smooth interaction (a separate smooth per factor level, sharing one
          smoothing parameter).
        - `"ad"`, `"gp"`, `"ds"`, `"so"`, `"mrf"`, `"mpi"`/`"mpd"`, `"cx"`/`"cv"`: adaptive,
          Gaussian-process, Duchon spline, soap-film, Markov-random-field, and
          monotone/convex-constrained bases respectively, for more specialized use cases.
    k : int
        Number of basis functions (an upper bound on the effective degrees of freedom the smooth
        can use). `-1` (default) means auto-select a sensible default for the basis type.
    by : str or None
        Name of a factor or numeric column for a factor-by smooth (a separate curve estimated per
        factor level) or a varying-coefficient smooth (the smooth's value multiplies a numeric
        `by` column) — mirroring the `by=` argument in R's mgcv.
    extra : dict[str, Any]
        Any additional keyword arguments passed through to the underlying basis constructor
        (e.g. `xt=`, `m=`), for basis types with extra configuration options.
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
    """A parsed model formula: the structured representation of a `GAM`'s right-hand side.

    A `Formula` is what `whittaker.formula.parser.parse` produces from a formula string such as
    `"y ~ s(x1) + s(x2, bs='cr', k=15) + te(x3, x4) + group"`, and is what `GAM.__init__` accepts
    either as that raw string or as an already-parsed `Formula` object. It separates the response
    column name from an ordered list of `Term` objects (`LinearTerm`, `SmoothTerm`,
    `InteractionTerm`, `OffsetTerm`) describing the right-hand side, plus whether an intercept is
    included. Downstream code (`whittaker.model_matrix.build_model_matrix`) consumes a `Formula`
    to construct the actual numeric design matrix and penalty structure used for fitting.

    Parameters
    ----------
    response : str
        Name of the response variable (the left-hand side, before `~`).
    terms : list[Term]
        Ordered list of model terms (the right-hand side, after `~`), where `Term` is a union of
        `LinearTerm`, `SmoothTerm`, `InteractionTerm`, and `OffsetTerm`.
    intercept : bool
        Whether the model includes an intercept column. `True` by default; suppress it by
        including `0 +` or `- 1` on the right-hand side of the formula string (e.g.
        `"y ~ 0 + s(x)"`).
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
