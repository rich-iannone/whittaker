r"""Altair-based plotting for fitted GAM objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import altair as alt

    from whittaker.gam import GAM


def _check_altair() -> None:
    try:
        import altair as alt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "Plotting requires the 'altair' optional dependency. "
            "Install it with: pip install 'whittaker[altair]'"
        ) from exc


def partial_effects(
    model: GAM,
    *,
    n_points: int = 200,
    level: float = 0.95,
) -> alt.VConcatChart | alt.Chart | alt.LayerChart | alt.FacetChart | alt.HConcatChart:
    """Plot partial effects with confidence bands for each smooth term.

    For univariate `s()` terms, produces a line plot with a shaded confidence band. For bivariate
    `te()` and `ti()` terms, produces a heatmap of the partial effect with a companion SE panel.

    Parameters
    ----------
    model:
        A fitted GAM.
    n_points:
        Number of evenly spaced points at which to evaluate each smooth. For 2-D smooths, the grid
        has approximately `sqrt(n_points)` points per side.
    level:
        Confidence level for the bands (the default is `0.95` -> ±1.96 SE).

    Returns
    -------
    altair.VConcatChart or altair.Chart
        One panel per smooth term (two for 2-D smooths), vertically concatenated.
    """
    _check_altair()
    import altair as alt

    model._check_fitted()

    from scipy.stats import norm

    z_val = float(norm.ppf(1.0 - (1.0 - level) / 2.0))

    mm = model._model_matrix
    beta = model._fit_result.coefficients
    sp = model._fit_result.smoothing_params
    scale = model._fit_result.scale

    X_train = mm.X
    p = X_train.shape[1]
    XtX = X_train.T @ X_train
    S_total = np.zeros_like(XtX)
    for lam, pen in zip(sp, mm.penalties, strict=False):
        S_total += lam * pen
    A = XtX + S_total
    A = (A + A.T) * 0.5

    eigvals, eigvecs = np.linalg.eigh(A)
    tol = np.max(eigvals) * p * np.finfo(float).eps
    keep = eigvals > tol
    eigvals_inv = np.zeros_like(eigvals)
    eigvals_inv[keep] = 1.0 / eigvals[keep]

    charts: list[alt.Chart | alt.LayerChart | alt.FacetChart | alt.HConcatChart] = []

    for idx, info in enumerate(mm.smooths):
        is_2d = len(info.term.variables) >= 2

        if is_2d:
            chart = _partial_effect_2d(
                info, idx, model, beta, eigvecs, eigvals_inv, scale, z_val, n_points
            )
            charts.append(chart)
        else:
            chart = _partial_effect_1d(
                info, idx, model, beta, eigvecs, eigvals_inv, scale, z_val, n_points
            )
            charts.append(chart)

    if len(charts) == 1:
        return charts[0]
    return alt.vconcat(*charts)


def _partial_effect_1d(
    info: object,
    idx: int,
    model: GAM,
    beta: np.ndarray,
    eigvecs: np.ndarray,
    eigvals_inv: np.ndarray,
    scale: float,
    z_val: float,
    n_points: int,
) -> alt.Chart | alt.LayerChart | alt.FacetChart:
    """Build a 1-D partial-effect line chart with confidence band."""
    import altair as alt

    from whittaker.model_matrix import SmoothInfo, _apply_constraint

    assert isinstance(info, SmoothInfo)

    var_name = info.term.variables[0]
    x_grid = _smooth_grid(info, n_points)

    B_grid = info.basis.basis_matrix(x_grid)

    has_by = info.by_var is not None
    if not has_by:
        constraint = info.basis.identifiability_constraints()
        n_constrained = info.col_end - info.col_start
        if constraint is not None and n_constrained < info.basis.n_basis:
            B_grid = _apply_constraint(B_grid, constraint)

    beta_j = beta[info.col_start : info.col_end]
    f_j = B_grid @ beta_j

    p_full = len(beta)
    X_partial = np.zeros((n_points, p_full))
    X_partial[:, info.col_start : info.col_end] = B_grid

    Xp_V = X_partial @ eigvecs
    var_diag = np.sum(Xp_V**2 * eigvals_inv[np.newaxis, :], axis=1) * scale
    se_j = np.sqrt(np.maximum(var_diag, 0.0))

    edf_j = model._fit_result.edf[idx]
    title_str = _smooth_title(info, edf_j)

    data_dict = {
        var_name: x_grid.tolist(),
        "effect": f_j.tolist(),
        "lower": (f_j - z_val * se_j).tolist(),
        "upper": (f_j + z_val * se_j).tolist(),
    }
    source = alt.Data(
        values=[
            dict(zip(data_dict, t, strict=False)) for t in zip(*data_dict.values(), strict=False)
        ]
    )

    band = (
        alt.Chart(source)
        .mark_area(opacity=0.25, color="#4682B4")
        .encode(
            x=alt.X(f"{var_name}:Q").title(var_name),
            y=alt.Y("lower:Q").title(title_str),
            y2="upper:Q",
        )
    )

    line = (
        alt.Chart(source)
        .mark_line(color="#4682B4", strokeWidth=2)
        .encode(
            x=alt.X(f"{var_name}:Q"),
            y=alt.Y("effect:Q"),
        )
    )

    zero_rule = (
        alt.Chart(alt.Data(values=[{}]))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y=alt.datum(0))
    )

    return (band + line + zero_rule).properties(width="container", height=300, title=title_str)


def _partial_effect_2d(
    info: object,
    idx: int,
    model: GAM,
    beta: np.ndarray,
    eigvecs: np.ndarray,
    eigvals_inv: np.ndarray,
    scale: float,
    z_val: float,
    n_points: int,
) -> alt.Chart | alt.LayerChart | alt.HConcatChart:
    """Build a 2-D partial-effect heatmap with an SE companion panel."""
    import altair as alt

    from whittaker.model_matrix import SmoothInfo, _apply_constraint

    assert isinstance(info, SmoothInfo)

    var1 = info.term.variables[0]
    var2 = info.term.variables[1]

    n_side = max(int(np.ceil(np.sqrt(n_points))), 15)
    x1_grid, x2_grid, x_flat = _smooth_grid_2d(info, n_side)

    B_grid = info.basis.basis_matrix(x_flat)

    has_by = info.by_var is not None
    if not has_by:
        constraint = info.basis.identifiability_constraints()
        n_constrained = info.col_end - info.col_start
        if constraint is not None and n_constrained < info.basis.n_basis:
            B_grid = _apply_constraint(B_grid, constraint)

    beta_j = beta[info.col_start : info.col_end]
    f_flat = B_grid @ beta_j

    n_grid = len(f_flat)
    p_full = len(beta)
    X_partial = np.zeros((n_grid, p_full))
    X_partial[:, info.col_start : info.col_end] = B_grid

    Xp_V = X_partial @ eigvecs
    var_diag = np.sum(Xp_V**2 * eigvals_inv[np.newaxis, :], axis=1) * scale
    se_flat = np.sqrt(np.maximum(var_diag, 0.0))

    edf_j = model._fit_result.edf[idx]
    title_str = _smooth_title(info, edf_j)

    dx1 = float(x1_grid[1] - x1_grid[0]) if len(x1_grid) > 1 else 1.0
    dx2 = float(x2_grid[1] - x2_grid[0]) if len(x2_grid) > 1 else 1.0

    records = []
    k = 0
    for i in range(len(x1_grid)):
        for j in range(len(x2_grid)):
            records.append(
                {
                    var1: float(x1_grid[i]),
                    var2: float(x2_grid[j]),
                    f"{var1}_lo": float(x1_grid[i] - dx1 / 2),
                    f"{var1}_hi": float(x1_grid[i] + dx1 / 2),
                    f"{var2}_lo": float(x2_grid[j] - dx2 / 2),
                    f"{var2}_hi": float(x2_grid[j] + dx2 / 2),
                    "effect": float(f_flat[k]),
                    "se": float(se_flat[k]),
                }
            )
            k += 1

    source = alt.Data(values=records)

    eff_max = float(max(abs(np.min(f_flat)), abs(np.max(f_flat))))
    if eff_max < 1e-10:
        eff_max = 1.0

    effect_chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X(f"{var1}_lo:Q", title=var1),
            x2=f"{var1}_hi:Q",
            y=alt.Y(f"{var2}_lo:Q", title=var2),
            y2=f"{var2}_hi:Q",
            color=alt.Color(
                "effect:Q",
                scale=alt.Scale(scheme="blueorange", domainMid=0, domain=[-eff_max, eff_max]),
                title="Effect",
            ),
            tooltip=[
                alt.Tooltip(f"{var1}:Q", format=".3f"),
                alt.Tooltip(f"{var2}:Q", format=".3f"),
                alt.Tooltip("effect:Q", format=".3f"),
                alt.Tooltip("se:Q", format=".3f"),
            ],
        )
        .properties(width=350, height=300, title=title_str)
    )

    se_chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X(f"{var1}_lo:Q", title=var1),
            x2=f"{var1}_hi:Q",
            y=alt.Y(f"{var2}_lo:Q", title=var2),
            y2=f"{var2}_hi:Q",
            color=alt.Color(
                "se:Q",
                scale=alt.Scale(scheme="reds"),
                title="SE",
            ),
            tooltip=[
                alt.Tooltip(f"{var1}:Q", format=".3f"),
                alt.Tooltip(f"{var2}:Q", format=".3f"),
                alt.Tooltip("se:Q", format=".3f"),
            ],
        )
        .properties(width=350, height=300, title=f"SE: {title_str}")
    )

    return alt.hconcat(effect_chart, se_chart).resolve_scale(color="independent")


def _smooth_title(info: object, edf: float) -> str:
    """Build a descriptive title string for a smooth term."""
    from whittaker.model_matrix import SmoothInfo

    assert isinstance(info, SmoothInfo)
    term = info.term
    var_str = ", ".join(term.variables)
    prefix = term.smooth_type

    if info.by_level is not None:
        return f"{prefix}({var_str}, by={info.by_var}):{info.by_level}, edf={edf:.1f}"
    if info.by_var is not None:
        return f"{prefix}({var_str}, by={info.by_var}), edf={edf:.1f}"
    return f"{prefix}({var_str}, edf={edf:.1f})"


def _marginal_range(basis: object) -> tuple[float, float]:
    """Return (min, max) of a univariate basis's training domain."""
    if hasattr(basis, "_x_min") and hasattr(basis, "_x_max"):
        return float(basis._x_min), float(basis._x_max)  # type: ignore[attr-defined]
    if hasattr(basis, "_knots") and basis._knots is not None:  # type: ignore[attr-defined]
        return float(np.min(basis._knots)), float(np.max(basis._knots))  # type: ignore[attr-defined]
    if hasattr(basis, "_x_train") and basis._x_train is not None:  # type: ignore[attr-defined]
        return float(np.min(basis._x_train)), float(np.max(basis._x_train))  # type: ignore[attr-defined]
    return 0.0, 1.0


def _smooth_grid(
    info: object,
    n_points: int,
) -> np.ndarray:
    """Generate an evaluation grid for a univariate smooth term."""
    from whittaker.model_matrix import SmoothInfo

    assert isinstance(info, SmoothInfo)
    lo, hi = _marginal_range(info.basis)
    return np.linspace(lo, hi, n_points)


def _smooth_grid_2d(
    info: object,
    n_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 2-D meshgrid for a tensor product smooth.

    Returns (x1_grid, x2_grid, x_flat) where x1_grid and x2_grid are 1-D arrays of length n_side and
    x_flat is (n_side*n_side, 2) for basis evaluation.
    """
    from whittaker.model_matrix import SmoothInfo
    from whittaker.smooths.tensor import TensorInteractionBasis, TensorProductBasis

    assert isinstance(info, SmoothInfo)

    basis = info.basis
    if isinstance(basis, (TensorProductBasis, TensorInteractionBasis)):
        m1, m2 = basis.marginals[0], basis.marginals[1]
        lo1, hi1 = _marginal_range(m1)
        lo2, hi2 = _marginal_range(m2)
    else:
        lo1, hi1 = 0.0, 1.0
        lo2, hi2 = 0.0, 1.0

    x1_grid = np.linspace(lo1, hi1, n_side)
    x2_grid = np.linspace(lo2, hi2, n_side)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid, indexing="ij")
    x_flat = np.column_stack([xx1.ravel(), xx2.ravel()])
    return x1_grid, x2_grid, x_flat


_CHECK_PLOT_NAMES = ("qq", "residuals", "histogram", "response")


def check(
    model: GAM,
    plots: tuple[str, ...] | list[str] | None = None,
) -> list[alt.Chart | alt.LayerChart | alt.FacetChart]:
    r"""Produce GAM diagnostic plots.

    Provides the standard suite of residual diagnostics used to assess GAM fit quality, analogous to
    `mgcv::gam.check()` in R. Rather than a single composite figure, each requested diagnostic is
    returned as its own full-width Altair chart, so callers can lay them out, select a subset, or
    display them individually (e.g. one per tab in an interactive app). Available plots (selected
    via `plots=`):

    - `"qq"`: QQ plot of deviance residuals against theoretical normal quantiles. Systematic
      curvature away from the reference line suggests the response distribution (family) may be
      misspecified.
    - `"residuals"`: Pearson residuals vs fitted values. A even, patternless scatter around zero is
      the target; funnel shapes suggest heteroscedasticity (consider a location-scale family), and
      curvature suggests a missing or under-smoothed term.
    - `"histogram"`: Histogram of deviance residuals, for checking overall symmetry and shape.
    - `"response"`: Observed response vs fitted values, with a 1:1 reference line, for an overall
      sense of fit quality and to spot outliers.

    Parameters
    ----------
    model:
        A fitted GAM.
    plots:
        Which diagnostic plots to include. Pass a list of names (e.g., `["qq", "residuals"]`) or
        `None` (default) for all four, in the order `"qq"`, `"residuals"`, `"histogram"`,
        `"response"`.

    Returns
    -------
    list[altair.Chart]
        One chart per requested diagnostic plot, in the order selected.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.gam import GAM
    from whittaker.plotting import check

    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=n)

    model = GAM("y ~ s(x)").fit({"x": x, "y": y})
    charts = check(model, plots=["qq", "residuals"])
    print(len(charts))
    ```
    """
    _check_altair()
    import altair as alt

    model._check_fitted()

    selected = list(plots) if plots is not None else list(_CHECK_PLOT_NAMES)
    for name in selected:
        if name not in _CHECK_PLOT_NAMES:
            raise ValueError(f"Unknown check plot {name!r}. Choose from {_CHECK_PLOT_NAMES}")

    deviance_resid = model.get_residuals("deviance")
    pearson_resid = model.get_residuals("pearson")
    fitted = model.fitted_values
    response = model._model_matrix.response

    result: list[alt.Chart | alt.LayerChart | alt.FacetChart] = []

    if "qq" in selected:
        sorted_resid = np.sort(deviance_resid)
        n = len(sorted_resid)
        from scipy.stats import norm

        theoretical_q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)

        qq_data = alt.Data(
            values=[
                {"theoretical": float(t), "observed": float(o)}
                for t, o in zip(theoretical_q, sorted_resid, strict=False)
            ]
        )

        qq_range = max(abs(float(theoretical_q[0])), abs(float(theoretical_q[-1])))
        qq_line_data = alt.Data(
            values=[{"x": -qq_range, "y": -qq_range}, {"x": qq_range, "y": qq_range}]
        )

        qq_points = (
            alt.Chart(qq_data)
            .mark_circle(size=15, color="#4682B4", opacity=0.6)
            .encode(
                x=alt.X("theoretical:Q").title("Theoretical quantiles"),
                y=alt.Y("observed:Q").title("Deviance residuals"),
            )
        )
        qq_ref = (
            alt.Chart(qq_line_data)
            .mark_line(color="gray", strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        result.append(
            (qq_points + qq_ref).properties(
                width="container", height=250, title="QQ plot of deviance residuals"
            )
        )

    if "residuals" in selected:
        resid_fit_data = alt.Data(
            values=[
                {"fitted": float(f), "residual": float(r)}
                for f, r in zip(fitted, pearson_resid, strict=False)
            ]
        )

        resid_fit_plot = (
            alt.Chart(resid_fit_data)
            .mark_circle(size=15, color="#4682B4", opacity=0.5)
            .encode(
                x=alt.X("fitted:Q").title("Fitted values"),
                y=alt.Y("residual:Q").title("Pearson residuals"),
            )
            .properties(width="container", height=250, title="Pearson residuals vs fitted")
        )
        resid_zero = (
            alt.Chart(alt.Data(values=[{}]))
            .mark_rule(strokeDash=[4, 4], color="gray")
            .encode(y=alt.datum(0))
        )
        result.append(resid_fit_plot + resid_zero)

    if "histogram" in selected:
        hist_data = alt.Data(values=[{"residual": float(r)} for r in deviance_resid])
        result.append(
            alt.Chart(hist_data)
            .mark_bar(color="#4682B4", opacity=0.7)
            .encode(
                x=alt.X("residual:Q").bin(maxbins=30).title("Deviance residuals"),
                y=alt.Y("count()").title("Frequency"),
            )
            .properties(width="container", height=250, title="Histogram of deviance residuals")
        )

    if "response" in selected:
        resp_fit_data = alt.Data(
            values=[
                {"fitted": float(f), "response": float(y)}
                for f, y in zip(fitted, response, strict=False)
            ]
        )

        resp_range_min = float(min(np.min(fitted), np.min(response)))
        resp_range_max = float(max(np.max(fitted), np.max(response)))
        ref_line_data = alt.Data(
            values=[
                {"x": resp_range_min, "y": resp_range_min},
                {"x": resp_range_max, "y": resp_range_max},
            ]
        )

        resp_points = (
            alt.Chart(resp_fit_data)
            .mark_circle(size=15, color="#4682B4", opacity=0.5)
            .encode(
                x=alt.X("fitted:Q").title("Fitted values"),
                y=alt.Y("response:Q").title("Response"),
            )
        )
        resp_ref = (
            alt.Chart(ref_line_data)
            .mark_line(color="gray", strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        result.append(
            (resp_points + resp_ref).properties(
                width="container", height=250, title="Response vs fitted"
            )
        )

    return result
