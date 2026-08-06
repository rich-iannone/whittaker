"""Altair-based plotting for fitted GAM objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import altair as alt

    from whittaker.gam import GAM


def _check_altair() -> None:
    try:
        import altair as alt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Plotting requires the 'altair' optional dependency. "
            "Install it with: pip install 'whittaker[altair]'"
        ) from exc


def partial_effects(
    model: GAM,
    *,
    n_points: int = 200,
    level: float = 0.95,
) -> alt.VConcatChart | alt.Chart:
    """Plot partial effects with confidence bands for each smooth term.

    Parameters
    ----------
    model:
        A fitted GAM.
    n_points:
        Number of evenly spaced points at which to evaluate each smooth.
    level:
        Confidence level for the bands (the default is `0.95` -> ±1.96 SE).

    Returns
    -------
    altair.VConcatChart or altair.Chart
        One panel per smooth term, vertically concatenated.
    """
    _check_altair()
    import altair as alt

    model._check_fitted()

    from scipy.stats import norm

    z_val = norm.ppf(1.0 - (1.0 - level) / 2.0)

    mm = model._model_matrix
    beta = model._fit_result.coefficients
    sp = model._fit_result.smoothing_params
    scale = model._fit_result.scale

    X_train = mm.X
    p = X_train.shape[1]
    XtX = X_train.T @ X_train
    S_total = np.zeros_like(XtX)
    for lam, pen in zip(sp, mm.penalties):
        S_total += lam * pen
    A = XtX + S_total
    A = (A + A.T) * 0.5

    # Eigendecomposition-based pseudoinverse (robust to ill-conditioning)
    eigvals, eigvecs = np.linalg.eigh(A)
    tol = np.max(eigvals) * p * np.finfo(float).eps
    keep = eigvals > tol
    eigvals_inv = np.zeros_like(eigvals)
    eigvals_inv[keep] = 1.0 / eigvals[keep]

    charts: list[alt.Chart] = []

    for idx, info in enumerate(mm.smooths):
        var_name = info.term.variables[0]

        x_grid = _smooth_grid(info, n_points)

        B_grid = info.basis.basis_matrix(x_grid)

        has_by = info.by_var is not None
        if not has_by:
            constraint = info.basis.identifiability_constraints()
            n_constrained = info.col_end - info.col_start
            if constraint is not None and n_constrained < info.basis.n_basis:
                from whittaker.model_matrix import _apply_constraint

                B_grid = _apply_constraint(B_grid, constraint)

        beta_j = beta[info.col_start : info.col_end]
        f_j = B_grid @ beta_j

        X_partial = np.zeros((n_points, X_train.shape[1]))
        X_partial[:, info.col_start : info.col_end] = B_grid

        Xp_V = X_partial @ eigvecs
        var_diag = np.sum(Xp_V**2 * eigvals_inv[np.newaxis, :], axis=1) * scale
        se_j = np.sqrt(np.maximum(var_diag, 0.0))

        edf_j = model._fit_result.edf[idx]

        if info.by_level is not None:
            title_str = f"s({var_name}, by={info.by_var}):{info.by_level}, edf={edf_j:.1f}"
        elif info.by_var is not None:
            title_str = f"s({var_name}, by={info.by_var}), edf={edf_j:.1f}"
        else:
            title_str = f"s({var_name}, edf={edf_j:.1f})"

        data_dict = {
            var_name: x_grid.tolist(),
            "effect": f_j.tolist(),
            "lower": (f_j - z_val * se_j).tolist(),
            "upper": (f_j + z_val * se_j).tolist(),
        }
        source = alt.Data(values=[dict(zip(data_dict, t)) for t in zip(*data_dict.values())])

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

        chart = (band + line + zero_rule).properties(
            width=450,
            height=300,
            title=title_str,
        )
        charts.append(chart)

    if len(charts) == 1:
        return charts[0]
    return alt.vconcat(*charts)


def _smooth_grid(
    info: object,
    n_points: int,
) -> np.ndarray:
    """Generate an evaluation grid for a smooth term from its training knot range."""
    from whittaker.model_matrix import SmoothInfo

    assert isinstance(info, SmoothInfo)

    basis = info.basis

    # PSpline stores explicit range
    if hasattr(basis, "_x_min") and hasattr(basis, "_x_max"):
        return np.linspace(float(basis._x_min), float(basis._x_max), n_points)

    # CRS stores knots
    if hasattr(basis, "_knots") and basis._knots is not None:
        return np.linspace(float(np.min(basis._knots)), float(np.max(basis._knots)), n_points)

    # TPRS stores training data
    if hasattr(basis, "_x_train") and basis._x_train is not None:
        return np.linspace(float(np.min(basis._x_train)), float(np.max(basis._x_train)), n_points)

    return np.linspace(0.0, 1.0, n_points)


def check(
    model: GAM,
) -> alt.VConcatChart:
    """Produce GAM diagnostic plots.

    Returns a 2×2 panel:

    - QQ plot of deviance residuals
    - Residuals vs fitted values
    - Histogram of residuals
    - Response vs fitted values

    Parameters
    ----------
    model:
        A fitted GAM.

    Returns
    -------
    altair.VConcatChart
        A 2×2 diagnostic panel.
    """
    _check_altair()
    import altair as alt

    model._check_fitted()

    residuals = model.residuals
    fitted = model.fitted_values
    response = model._model_matrix.response

    sorted_resid = np.sort(residuals)
    n = len(sorted_resid)
    from scipy.stats import norm

    theoretical_q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    qq_data = alt.Data(
        values=[
            {"theoretical": float(t), "observed": float(o)}
            for t, o in zip(theoretical_q, sorted_resid)
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
            y=alt.Y("observed:Q").title("Observed quantiles"),
        )
    )
    qq_ref = (
        alt.Chart(qq_line_data).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    )
    qq_plot = (qq_points + qq_ref).properties(width=300, height=250, title="QQ plot of residuals")

    resid_fit_data = alt.Data(
        values=[{"fitted": float(f), "residual": float(r)} for f, r in zip(fitted, residuals)]
    )

    resid_fit_plot = (
        alt.Chart(resid_fit_data)
        .mark_circle(size=15, color="#4682B4", opacity=0.5)
        .encode(
            x=alt.X("fitted:Q").title("Fitted values"),
            y=alt.Y("residual:Q").title("Residuals"),
        )
        .properties(width=300, height=250, title="Residuals vs fitted")
    )
    resid_zero = (
        alt.Chart(alt.Data(values=[{}]))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y=alt.datum(0))
    )
    resid_fit_plot = resid_fit_plot + resid_zero

    hist_data = alt.Data(values=[{"residual": float(r)} for r in residuals])
    hist_plot = (
        alt.Chart(hist_data)
        .mark_bar(color="#4682B4", opacity=0.7)
        .encode(
            x=alt.X("residual:Q").bin(maxbins=30).title("Residuals"),
            y=alt.Y("count()").title("Frequency"),
        )
        .properties(width=300, height=250, title="Histogram of residuals")
    )

    resp_fit_data = alt.Data(
        values=[{"fitted": float(f), "response": float(y)} for f, y in zip(fitted, response)]
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
        alt.Chart(ref_line_data).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    )
    resp_fit_plot = (resp_points + resp_ref).properties(
        width=300, height=250, title="Response vs fitted"
    )

    top_row = alt.hconcat(qq_plot, resid_fit_plot)
    bottom_row = alt.hconcat(hist_plot, resp_fit_plot)
    return alt.vconcat(top_row, bottom_row).resolve_scale(x="independent", y="independent")
