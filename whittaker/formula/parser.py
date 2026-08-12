"""Formula string parser for Whittaker GAMs."""

from __future__ import annotations

import ast
from typing import Any

from whittaker.formula.terms import (
    Formula,
    InteractionTerm,
    LinearTerm,
    OffsetTerm,
    SmoothTerm,
    Term,
)

# Smooth function names recognised in formula strings.
_SMOOTH_FUNCS: frozenset[str] = frozenset({"s", "te", "ti", "t2"})

# Default basis type per smooth function.
_DEFAULT_BS: dict[str, str] = {
    "s": "tp",
    "te": "tp",
    "ti": "tp",
    "t2": "tp",
}


# Sentinel for "-1" / "+0" intercept suppression.
class _DropIntercept:
    pass


_DROP = _DropIntercept()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(formula: str) -> Formula:
    """Parse *formula* into a `~whittaker.formula.terms.Formula`.

    Turns an `mgcv`-style R formula string such as `"y ~ s(x1) + s(x2, bs='cr') + x3"` into a
    structured `Formula` object: a response column name, an ordered list of `Term` objects (one per
    right-hand-side entry), and a flag indicating whether an intercept is included. `whittaker.gam.GAM`
    calls `parse()` internally when constructed from a formula string, and the resulting `Formula` is
    later consumed by `~whittaker.model_matrix.build_model_matrix` to build the numeric design matrix
    and penalty structure. This lets users specify a GAM the same way they would in R's `mgcv`, but
    written in Python. Rather than a hand-written grammar or regular expression, `parse()` uses
    Python's standard-library `ast` module to parse the right-hand side as a Python expression and
    walks the resulting syntax tree; consequently, only formula constructs that are expressible as
    plain Python expressions are supported — bare names, function calls, `+`/`-`/`*` binary operators,
    and the integer literals `0`, `1`, and `-1` for intercept control. Anything else raises a
    `ValueError` naming the unsupported construct.

    The following right-hand-side syntax is recognised:

    - A bare column name, e.g. `x3`, becomes a `~whittaker.formula.terms.LinearTerm`.
    - A call to `s()`, `te()`, `ti()`, or `t2()` becomes a
      `~whittaker.formula.terms.SmoothTerm`. Positional arguments name the smooth's variable(s) (more
      than one for `te()`/`ti()`/`t2()` tensor products). Recognised keyword arguments are `bs=` (the
      basis type, a string such as `"cr"`, `"mpi"`, or `"cx"`), `k=` (the basis dimension, an `int`, or
      a `list[int]` giving one dimension per marginal for tensor terms), and `by=` (a bare column name
      for a by-variable interaction). Any other keyword, e.g. `xt=`, `m=`, or `degree=`, is collected
      into the term's `extra` dict and passed through unevaluated (as a literal or bare-name string).
    - A call to `offset()`, e.g. `offset(log(n))`, becomes an
      `~whittaker.formula.terms.OffsetTerm` whose `expression` is the unparsed argument text.
    - `x1 * x2` becomes a full interaction (`~whittaker.formula.terms.InteractionTerm` with
      `full=True`): both main effects plus the interaction column. Both operands of `*` must be bare
      column names. Note that `x1:x2`-style colon syntax for a *reduced* (interaction-only) term is
      **not** recognised by this parser — `:` is not a valid Python binary operator between
      identifiers, so `ast.parse` rejects it, and a formula string using `x1:x2` raises a `ValueError`
      rather than producing a reduced interaction term.
    - `0`, `-1`, or `+0` on their own suppresses the intercept (`Formula.intercept` becomes `False`).
      `+1` is accepted as a no-op, since the intercept is already included by default.

    Parameters
    ----------
    formula:
        A model formula string such as `"y ~ s(x1) + s(x2, bs='cr') + x3"`.

    Returns
    -------
    Formula
        Structured representation of the formula.

    Raises
    ------
    ValueError
        If the formula string is malformed or contains unsupported syntax.

    Notes
    -----
    This is not a full formula-parsing DSL like R's `formula()` — there is no `.` shorthand for "all
    other columns", no `poly()` or other in-formula transformations, and no arbitrary nesting beyond
    what is listed above. The supported grammar is exactly what can be expressed as a restricted
    `ast.parse(rhs, mode="eval")` walk over names, calls, and `+`/`-`/`*` binary operators; any
    construct outside that grammar raises a `ValueError` with a message describing what was found and
    what is supported.

    Examples
    --------
    ```{python}
    from whittaker.formula.parser import parse

    formula = parse("y ~ s(x1) + s(x2, bs='cr', k=15) + x3")
    formula.response
    ```

    ```{python}
    [(term.__class__.__name__, term) for term in formula.terms]
    ```
    """
    if "~" not in formula:
        raise ValueError(
            f"Formula {formula!r} is missing the '~' separator.\n"
            "Expected format: 'response ~ terms'."
        )

    lhs, rhs = formula.split("~", 1)
    response = lhs.strip()
    rhs = rhs.strip()

    if not response.isidentifier():
        raise ValueError(
            f"Response variable {response!r} is not a valid Python identifier.\n"
            "The left-hand side of '~' must be a single column name."
        )

    intercept, terms = _parse_rhs(rhs)
    return Formula(response=response, terms=terms, intercept=intercept)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_rhs(rhs: str) -> tuple[bool, list[Term]]:
    """Parse the right-hand side and return `(intercept, terms)`."""
    try:
        tree = ast.parse(rhs, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Could not parse formula right-hand side {rhs!r}.\nSyntax error: {exc}"
        ) from exc

    raw_items = _collect_additive_terms(tree.body)

    intercept = True
    terms: list[Term] = []

    for node, negated in raw_items:
        result = _convert_node(node, negated)
        if result is None:
            pass  # no-op (e.g. explicit "+1")
        elif isinstance(result, _DropIntercept):
            intercept = False
        else:
            terms.append(result)

    return intercept, terms


def _collect_additive_terms(
    node: ast.expr,
) -> list[tuple[ast.expr, bool]]:
    """Recursively flatten a `+`/`-` expression into `(node, negated)` pairs."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _collect_additive_terms(node.left) + _collect_additive_terms(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        left = _collect_additive_terms(node.left)
        # Negate each item on the right-hand side of the subtraction.
        right = [(n, not neg) for n, neg in _collect_additive_terms(node.right)]
        return left + right
    return [(node, False)]


def _convert_node(
    node: ast.expr,
    negated: bool,
) -> Term | _DropIntercept | None:
    """Convert one AST node into a `~whittaker.formula.terms.Term`."""

    # ---- numeric literals (intercept control) ----
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        value: int = node.value
        if negated:
            value = -value
        if value == -1 or value == 0:
            return _DROP
        if value == 1:
            return None  # explicit "+1" — intercept stays, nothing to add
        raise ValueError(
            f"Unexpected numeric constant {node.value!r} in formula right-hand side.\n"
            "Only 0, 1, or -1 are meaningful as standalone numeric terms."
        )

    # ---- plain column name ----
    if isinstance(node, ast.Name):
        if negated:
            raise ValueError(
                f"Cannot negate a linear term {node.id!r} in a formula.\n"
                "Use '- 1' or '+ 0' to suppress the intercept."
            )
        return LinearTerm(variable=node.id)

    # ---- function calls: s(), te(), ti(), t2(), offset() ----
    if isinstance(node, ast.Call):
        if negated:
            raise ValueError(
                f"Cannot negate a smooth term {ast.unparse(node)!r} in a formula.\n"
                "Use '- 1' or '+ 0' to suppress the intercept."
            )
        return _convert_call(node)

    # ---- x1 * x2 (full interaction with main effects) ----
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if negated:
            raise ValueError(
                f"Cannot negate an interaction term {ast.unparse(node)!r} in a formula."
            )
        left_name = _require_name(node.left, context="left side of '*'")
        right_name = _require_name(node.right, context="right side of '*'")
        return InteractionTerm(left=left_name, right=right_name, full=True)

    raise ValueError(
        f"Unsupported formula term: {ast.unparse(node)!r}.\n"
        "Supported terms: column names, s(), te(), ti(), t2(), offset(), "
        "column * column interactions, and 0 / 1 / -1 for intercept control."
    )


def _convert_call(node: ast.Call) -> Term:
    """Convert an `ast.Call` node to a :data:`~whittaker.formula.terms.Term`."""
    if not isinstance(node.func, ast.Name):
        raise ValueError(
            f"Unsupported callable in formula: {ast.unparse(node.func)!r}.\n"
            "Only s(), te(), ti(), t2(), and offset() are supported."
        )

    func_name: str = node.func.id

    # ---- offset() ----
    if func_name == "offset":
        if len(node.args) != 1 or node.keywords:
            raise ValueError(
                "offset() requires exactly one positional argument and no keyword "
                f"arguments, got: {ast.unparse(node)!r}."
            )
        return OffsetTerm(expression=ast.unparse(node.args[0]))

    # ---- smooth functions ----
    if func_name not in _SMOOTH_FUNCS:
        raise ValueError(
            f"Unknown function {func_name!r} in formula.\n"
            f"Recognised smooth functions: {sorted(_SMOOTH_FUNCS)}.\n"
            "For a linear term use a plain column name; for an offset use offset()."
        )

    # Collect positional variable names.
    variables: list[str] = []
    for arg in node.args:
        if not isinstance(arg, ast.Name):
            raise ValueError(
                f"Positional arguments to {func_name}() must be bare column names, "
                f"got {ast.unparse(arg)!r}."
            )
        variables.append(arg.id)

    if not variables:
        raise ValueError(
            f"{func_name}() requires at least one variable argument, got: {ast.unparse(node)!r}."
        )

    # Collect keyword arguments.
    bs: str = _DEFAULT_BS[func_name]
    k: int = -1
    by: str | None = None
    extra: dict[str, Any] = {}

    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError(
                f"**kwargs expansion is not supported in formula terms "
                f"(found in {ast.unparse(node)!r})."
            )
        key: str = kw.arg
        val = _eval_kwarg_value(kw.value, key, func_name)

        if key == "bs":
            if not isinstance(val, str):
                raise ValueError(f"bs= must be a string, got {val!r} in {ast.unparse(node)!r}.")
            bs = val
        elif key == "k":
            if isinstance(val, int):
                k = val
            elif isinstance(val, list) and all(isinstance(v, int) for v in val):
                # k=[c1, c2, ...] is valid for tensor products — store in extra.
                extra["k"] = val
            else:
                raise ValueError(
                    f"k= must be an integer or a list of integers, "
                    f"got {val!r} in {ast.unparse(node)!r}."
                )
        elif key == "by":
            if not isinstance(val, str):
                raise ValueError(
                    f"by= must be a column name (string), got {val!r} in {ast.unparse(node)!r}."
                )
            by = val
        else:
            extra[key] = val

    return SmoothTerm(
        variables=tuple(variables),
        smooth_type=func_name,
        bs=bs,
        k=k,
        by=by,
        extra=extra,
    )


def _eval_kwarg_value(node: ast.expr, key: str, func_name: str) -> Any:
    """Evaluate a keyword-argument value node.

    Supports literals (strings, ints, floats, bools, None, lists, tuples) and
    bare names (interpreted as column name strings, as `by=group` is common).
    """
    # Bare identifier: treat as a column name string.
    if isinstance(node, ast.Name):
        return node.id

    try:
        return ast.literal_eval(node)
    except ValueError:
        pass

    raise ValueError(
        f"Could not evaluate keyword argument {key}={ast.unparse(node)!r} "
        f"in {func_name}().\n"
        "Only literals (strings, numbers, lists, …) and bare column names are "
        "supported as keyword-argument values."
    )


def _require_name(node: ast.expr, context: str) -> str:
    """Return the identifier from a `ast.Name` node or raise a clear error."""
    if isinstance(node, ast.Name):
        return node.id
    raise ValueError(
        f"Expected a bare column name for {context}, "
        f"got {ast.unparse(node)!r}.\n"
        "Both sides of a '*' interaction must be simple column names."
    )
