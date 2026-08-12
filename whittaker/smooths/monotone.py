"""Shape-constrained P-spline bases: monotone and convex/concave smoothing.

This module provides two variants of the P-spline basis (see `whittaker.smooths.pspline`) that
constrain the *shape* of the fitted curve rather than only its smoothness: `MonotonePSpline` enforces
a non-decreasing or non-increasing fit, and `ConvexPSpline` enforces a convex or concave fit. Both
constraints are linear inequality conditions on the B-spline coefficients, and both are enforced
during fitting by projecting the coefficients onto the corresponding constraint cone after each
penalized least-squares update, using the Pool Adjacent Violators Algorithm (PAVA, `_pava`) as the
underlying projection mechanism. `project_monotone` and `project_convex` expose that projection
directly for the monotone and convex/concave cases, respectively.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.pspline import PSpline, _diff_matrix


class MonotonePSpline(PSpline):
    r"""Shape-constrained P-spline: monotone increasing or decreasing.

    Uses the same B-spline basis and difference penalty as `~whittaker.smooths.pspline.PSpline`, but
    additionally requires the fitted curve `f(x) = \sum_j \beta_j B_j(x)` to be non-decreasing (or,
    with `decreasing=True`, non-increasing) over the whole domain. This is enforced as a linear
    inequality constraint on the coefficients rather than on `f` directly: because each B-spline basis
    function `B_j` is non-negative and has local support (a "bump" that overlaps only its
    neighbours), a non-decreasing sequence of coefficients `\beta_1 \le \beta_2 \le \dots \le \beta_k`
    guarantees a non-decreasing curve. Intuitively, moving `x` to the right shifts the basis functions'
    weight away from earlier, smaller-or-equal coefficients and onto later, larger-or-equal ones, so
    the weighted sum can only stay flat or increase. Use `MonotonePSpline` (via `s(x, bs="mpi")` for
    increasing or `s(x, bs="mpd")` for decreasing in a formula) whenever domain knowledge says the
    relationship must be monotone — e.g. a dose-response curve, a cumulative distribution, or a growth
    curve — and an unconstrained smooth would otherwise wiggle non-monotonically due to noise.

    Parameters
    ----------
    k:
        Number of B-spline basis functions. The default is `20`.
    degree:
        B-spline polynomial degree. The default is `3` (cubic).
    m:
        Difference penalty order. The default is `2`.
    decreasing:
        If `True`, enforce monotone *decreasing*. The default is `False` (monotone increasing).

    Notes
    -----
    The monotonicity constraint is enforced during fitting, not by direct constrained optimization.
    Instead, at each penalized iteratively reweighted least squares (P-IRLS) iteration the ordinary
    (unconstrained) coefficient update is projected onto the monotone cone: `whittaker.fitting.pirls`
    detects any smooth term whose basis is a `MonotonePSpline` and passes its coefficient block through
    `project_monotone` before the next iteration's linear predictor is formed. This projection uses the
    Pool Adjacent Violators Algorithm (PAVA), the standard algorithm for isotonic regression (Barlow,
    Bartholomew, Bremner & Brunk, 1972; Best & Chakravarti, 1990). Given an arbitrary vector, PAVA finds
    the closest non-decreasing vector to it in the least-squares sense by scanning for adjacent
    "violations" (a value followed by a smaller one) and replacing each violating block with its mean,
    merging blocks until no violations remain. Because this is an orthogonal projection onto the convex
    cone of non-decreasing sequences, iterating it alongside the P-IRLS coefficient update drives the
    fit toward the coefficient vector, within that cone, that best balances the penalized deviance and
    the constraint.

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wt

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 200))
    y = 3 * x + rng.normal(scale=0.15, size=200)

    model = wt.GAM("y ~ s(x, bs='mpi')").fit({"x": x, "y": y})

    new_x = np.linspace(0, 1, 50)
    fitted = model.predict({"x": new_x}).values
    np.all(np.diff(fitted) >= -1e-8)
    ```
    """

    def __init__(
        self,
        k: int = 20,
        degree: int = 3,
        m: int = 2,
        decreasing: bool = False,
    ) -> None:
        super().__init__(k=k, degree=degree, m=m)
        self.decreasing = decreasing

    @property
    def constraint_direction(self) -> int:
        """Sign convention used by the PAVA projection.

        Returns `-1` when `decreasing=True` (the coefficients are projected onto the
        non-increasing cone by negating, applying PAVA, and negating back) or `1` for the default
        non-decreasing case. Consulted by `whittaker.fitting.pirls` when applying
        `project_monotone` to this term's coefficient block.

        Returns
        -------
        int
            `-1` for monotone decreasing, `1` for monotone increasing.
        """
        return -1 if self.decreasing else 1

    def null_space_dimension(self) -> int:
        """Return the dimension of the basis's unpenalized null space.

        The monotonicity constraint is enforced by post-hoc projection rather than by removing
        degrees of freedom from the penalty, so the underlying P-spline penalty null space is
        treated as fully absorbed elsewhere in the model; this basis reports `0` so it does not
        additionally compete with a model intercept for identifiability.

        Returns
        -------
        int
            Always `0`.
        """
        return 0


class ConvexPSpline(PSpline):
    r"""Shape-constrained P-spline: convex or concave.

    Uses the same B-spline basis and difference penalty as `~whittaker.smooths.pspline.PSpline`, but
    additionally requires the fitted curve `f(x) = \sum_j \beta_j B_j(x)` to be convex (or, with
    `concave=True`, concave) over the whole domain. As with `MonotonePSpline`, this is enforced as a
    linear inequality on the coefficients: for an equally-spaced B-spline basis, the curve is convex
    whenever the second differences of the coefficients, `\Delta^2 \beta_j = \beta_j - 2\beta_{j-1} +
    \beta_{j-2}`, are all non-negative. Use `ConvexPSpline` (via `s(x, bs="cx")` for convex or
    `s(x, bs="cv")` for concave in a formula) when the relationship is known to have a single bend of
    consistent curvature — e.g. a cost curve, a learning curve, or a concave production function —
    and an unconstrained smooth would otherwise produce spurious inflection points from noise.

    Parameters
    ----------
    k:
        Number of B-spline basis functions. The default is `20`.
    degree:
        B-spline polynomial degree. The default is `3` (cubic).
    m:
        Difference penalty order. The default is `2`.
    concave:
        If `True`, enforce concavity. The default is `False` (convex).

    Notes
    -----
    As with `MonotonePSpline`, the constraint is enforced during P-IRLS fitting by projecting the
    ordinary coefficient update onto the convex (or concave) cone after each iteration: smooth terms
    whose basis is a `ConvexPSpline` have their coefficient block passed through `project_convex`
    before the next iteration's linear predictor is formed (see `whittaker.fitting.pirls`). The
    projection extends the monotone PAVA projection by one order of differencing: convexity requires
    the *first* differences of the coefficients, `d_j = \beta_j - \beta_{j-1}`, to form a non-decreasing
    sequence (equivalently, that the second differences of `\beta` are non-negative), so `project_convex`
    computes the first differences, projects *them* onto the monotone cone with PAVA (see the `Notes`
    on `MonotonePSpline` for the algorithm), and then reconstructs `\beta` by cumulatively summing the
    projected differences back up from `\beta_0`. Concavity is handled by negating the differences
    before and after the PAVA projection, mirroring `decreasing=True` for `MonotonePSpline`.

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wt

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-1, 1, 200))
    y = x**2 + rng.normal(scale=0.1, size=200)

    model = wt.GAM("y ~ s(x, bs='cx')").fit({"x": x, "y": y})

    new_x = np.linspace(-1, 1, 50)
    fitted = model.predict({"x": new_x}).values
    second_diff = np.diff(fitted, n=2)
    # Second differences are non-negative up to fitting/projection tolerance.
    np.all(second_diff >= -1e-2)
    ```
    """

    def __init__(
        self,
        k: int = 20,
        degree: int = 3,
        m: int = 2,
        concave: bool = False,
    ) -> None:
        super().__init__(k=k, degree=degree, m=m)
        self.concave = concave

    @property
    def constraint_direction(self) -> int:
        """Sign convention used by the second-difference PAVA projection.

        Returns `-1` when `concave=True` (the first differences of the coefficients are negated
        before and after the PAVA projection, yielding non-positive second differences) or `1`
        for the default convex case. Consulted by `project_convex` and by
        `whittaker.fitting.pirls` when projecting this term's coefficient block each iteration.

        Returns
        -------
        int
            `-1` for concave, `1` for convex.
        """
        return -1 if self.concave else 1

    @property
    def constraint_order(self) -> int:
        """Order of the difference constraint enforced by projection.

        Always `2`: convexity/concavity is a constraint on the *second* differences of the
        coefficients (equivalently, the first differences must form a monotone sequence), as
        opposed to `MonotonePSpline`'s first-order (`1`) constraint.

        Returns
        -------
        int
            Always `2`.
        """
        return 2

    def null_space_dimension(self) -> int:
        """Return the dimension of the basis's unpenalized null space.

        As with `MonotonePSpline`, the shape constraint is enforced by post-hoc projection of the
        coefficients rather than by removing degrees of freedom from the penalty, so this basis
        reports `0` unpenalized dimensions.

        Returns
        -------
        int
            Always `0`.
        """
        return 0


def project_monotone(beta: NDArray, *, decreasing: bool = False) -> NDArray:
    """Project a coefficient vector onto the monotone cone using PAVA.

    Returns the closest (in least-squares distance) non-decreasing, or if `decreasing=True`,
    non-increasing vector to `beta`. See `MonotonePSpline` for details on the PAVA algorithm and how
    this projection is used to enforce monotonicity during fitting.

    Parameters
    ----------
    beta:
        Coefficient vector to project, shape `(k,)`.
    decreasing:
        If `True`, project onto the non-increasing cone instead of the non-decreasing cone. The
        default is `False`.

    Returns
    -------
    NDArray
        The projected coefficient vector, shape `(k,)`.
    """
    if decreasing:
        return -_pava(-beta)
    return _pava(beta)


def project_convex(beta: NDArray, *, concave: bool = False) -> NDArray:
    """Project a coefficient vector onto the convex (or concave) cone.

    Convexity requires non-negative second differences of `beta`. This is achieved by projecting the
    *first* differences of `beta` onto the monotone (non-decreasing) cone with PAVA, then
    reconstructing the coefficient vector by cumulative summation. See `ConvexPSpline` for the full
    derivation and how this projection is used to enforce convexity during fitting.

    Parameters
    ----------
    beta:
        Coefficient vector to project, shape `(k,)`.
    concave:
        If `True`, project onto the concave cone (non-positive second differences) instead of the
        convex cone. The default is `False`.

    Returns
    -------
    NDArray
        The projected coefficient vector, shape `(k,)`.
    """
    D1 = _diff_matrix(len(beta), 1)
    diffs = D1 @ beta
    if concave:
        diffs_proj = -_pava(-diffs)
    else:
        diffs_proj = _pava(diffs)
    out = np.empty_like(beta)
    out[0] = beta[0]
    out[1:] = out[0] + np.cumsum(diffs_proj)
    return out


def _pava(x: NDArray) -> NDArray:
    """Pool Adjacent Violators Algorithm: isotonic (non-decreasing) regression on `x`."""
    n = len(x)
    result = x.copy()
    block_start = np.arange(n)
    block_size = np.ones(n, dtype=int)

    i = 0
    while i < n - 1:
        j = i + block_size[i]
        if j >= n:
            break
        if result[i] > result[j]:
            total = result[i] * block_size[i] + result[j] * block_size[j]
            new_size = block_size[i] + block_size[j]
            result[i] = total / new_size
            block_size[i] = new_size
            block_size[j] = 0
            while i > 0:
                prev = i - 1
                while prev >= 0 and block_size[prev] == 0:
                    prev -= 1
                if prev < 0:  # pragma: no cover - block_size[0] is never zeroed, so this
                    break  # defensive guard against walking past the start is unreachable.
                if result[prev] > result[i]:
                    total = result[prev] * block_size[prev] + result[i] * block_size[i]
                    new_size = block_size[prev] + block_size[i]
                    result[prev] = total / new_size
                    block_size[prev] = new_size
                    block_size[i] = 0
                    i = prev
                else:
                    break
        else:
            i = j

    out = np.empty(n)
    i = 0
    while i < n:
        if block_size[i] > 0:
            out[i : i + block_size[i]] = result[i]
            i += block_size[i]
        else:  # pragma: no cover - merged blocks are always contiguous, so a zero-size
            i += 1  # entry not immediately consumed by the previous block never occurs.
    return out
