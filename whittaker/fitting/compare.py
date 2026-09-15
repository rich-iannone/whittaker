"""Model comparison table: compare multiple fitted GAMs side by side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whittaker.gam import GAM


@dataclass
class ComparisonRow:
    """One row of a model comparison table.

    Attributes
    ----------
    label : str
        Formula string identifying the model.
    aic : float
        Akaike Information Criterion.
    delta_aic : float
        AIC difference from the best (lowest-AIC) model.
    bic : float
        Bayesian Information Criterion.
    deviance_explained : float
        Proportion of null deviance explained.
    r_squared_adj : float
        Adjusted R-squared.
    edf_total : float
        Total effective degrees of freedom.
    gcv_score : float or None
        GCV score, or `None` for Bayesian fits.
    scale : float
        Estimated scale (dispersion) parameter.
    n_obs : int
        Number of observations.
    """

    label: str
    aic: float
    delta_aic: float
    bic: float
    deviance_explained: float
    r_squared_adj: float
    edf_total: float
    gcv_score: float | None
    scale: float
    n_obs: int


@dataclass
class ComparisonResult:
    """Result of comparing multiple fitted GAMs.

    Rows are sorted by AIC (best first). The `delta_aic` field on each row gives the AIC difference
    from the best model, making it easy to see which models are competitive.

    Attributes
    ----------
    rows : list[ComparisonRow]
        One row per model, sorted by AIC (ascending).
    """

    rows: list[ComparisonRow]

    @property
    def n_models(self) -> int:
        """Number of models compared."""
        return len(self.rows)

    @property
    def best(self) -> ComparisonRow:
        """The model with the lowest AIC."""
        return self.rows[0]

    def __repr__(self) -> str:
        has_gcv = any(r.gcv_score is not None for r in self.rows)

        hdr_parts = [
            f"{'#':>3}",
            f"{'Formula':<30}",
            f"{'AIC':>10}",
            f"{'ΔAIC':>8}",
            f"{'BIC':>10}",
            f"{'Dev.Expl.':>10}",
            f"{'Adj.R²':>8}",
            f"{'EDF':>7}",
        ]
        if has_gcv:
            hdr_parts.append(f"{'GCV':>10}")
        header = " ".join(hdr_parts)

        sep_parts = [
            "-" * 3,
            "-" * 30,
            "-" * 10,
            "-" * 8,
            "-" * 10,
            "-" * 10,
            "-" * 8,
            "-" * 7,
        ]
        if has_gcv:
            sep_parts.append("-" * 10)
        sep = " ".join(sep_parts)

        lines = [
            f"Model Comparison ({self.n_models} models, {self.rows[0].n_obs} observations)",
            "",
            header,
            sep,
        ]

        for i, row in enumerate(self.rows):
            label = row.label if len(row.label) <= 30 else row.label[:27] + "..."
            parts = [
                f"{i + 1:>3}",
                f"{label:<30}",
                f"{row.aic:>10.2f}",
                f"{row.delta_aic:>+8.2f}",
                f"{row.bic:>10.2f}",
                f"{row.deviance_explained:>9.1%}",
                f"{row.r_squared_adj:>8.4f}",
                f"{row.edf_total:>7.1f}",
            ]
            if has_gcv:
                gcv_str = f"{row.gcv_score:>10.6f}" if row.gcv_score is not None else f"{'---':>10}"
                parts.append(gcv_str)
            lines.append(" ".join(parts))

        return "\n".join(lines)


def compare(*models: GAM) -> ComparisonResult:
    """Compare multiple fitted GAMs in a summary table.

    Collects AIC, BIC, deviance explained, adjusted R-squared, EDF, GCV (when available), and scale
    from each model, sorts by AIC, and computes delta-AIC from the best model.

    Parameters
    ----------
    *models : GAM
        Two or more fitted GAM objects.

    Returns
    -------
    ComparisonResult
        Comparison table sorted by AIC (best first).

    Raises
    ------
    ValueError
        If fewer than 2 models are provided.

    Examples
    --------
    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + rng.normal(0, 0.3, 200)
    data = {"x": x, "y": y}

    m1 = wk.GAM("y ~ x").fit(data)
    m2 = wk.GAM("y ~ s(x, k=5)").fit(data)
    m3 = wk.GAM("y ~ s(x, k=15)").fit(data)

    print(wk.compare(m1, m2, m3))
    ```
    """
    if len(models) < 2:
        raise ValueError("compare() requires at least 2 models.")

    for m in models:
        if not m._fitted:
            raise RuntimeError(
                f"Model {m.formula!r} has not been fitted yet. Call .fit(data) first."
            )

    rows: list[ComparisonRow] = []
    for m in models:
        gof = m.goodness_of_fit()
        rows.append(
            ComparisonRow(
                label=repr(m.formula),
                aic=gof.aic,
                delta_aic=0.0,
                bic=gof.bic,
                deviance_explained=gof.deviance_explained,
                r_squared_adj=gof.r_squared_adj,
                edf_total=gof.edf_total,
                gcv_score=gof.gcv_score,
                scale=gof.scale,
                n_obs=gof.n_obs,
            )
        )

    rows.sort(key=lambda r: r.aic)
    best_aic = rows[0].aic
    for row in rows:
        row.delta_aic = row.aic - best_aic

    return ComparisonResult(rows=rows)
