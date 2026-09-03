"""Posterior predictive checks for fitted GAMs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class PPCResult:
    """Result of a posterior predictive check on a fitted GAM.

    Attributes
    ----------
    y_rep : NDArray
        Posterior predictive draws on the response scale, shape `(n, n_sim)`. Each column is one
        draw from the posterior predictive distribution (a plausible dataset the model could have
        generated).
    observed : NDArray
        Observed response values used to fit the model, shape `(n,)`.
    """

    y_rep: NDArray
    observed: NDArray
    # Mapping: stat_name -> (observed_value, array of per-rep values shape (n_sim,))
    _stats: dict[str, tuple[float, NDArray]] = field(default_factory=dict, repr=False)

    def p_value(self, name: str) -> float:
        """Bayesian p-value for a named test statistic.

        The Bayesian p-value is the proportion of replicated datasets for which the test statistic
        equals or exceeds the observed value: $p = P(T(y^\\text{rep}) \\ge T(y^\\text{obs}))$.

        Values near 0.5 indicate good calibration. Values near 0 or 1 indicate systematic
        discrepancy between the model and the data.

        Parameters
        ----------
        name : str
            One of `"mean"`, `"sd"`, `"min"`, `"max"`, `"prop_zero"`.
        """
        obs_val, rep_vals = self._stats[name]
        return float(np.mean(rep_vals >= obs_val))

    @property
    def stat_names(self) -> list[str]:
        """Names of the computed test statistics."""
        return list(self._stats.keys())

    def stat(self, name: str) -> tuple[float, NDArray]:
        """Return `(observed_value, rep_values)` for a named statistic."""
        return self._stats[name]

    def __repr__(self) -> str:
        n, n_sim = self.y_rep.shape
        lines = [f"PPCResult (n={n}, n_sim={n_sim})", ""]
        col_w = (16, 10, 10, 10)
        header = (
            f"  {'Statistic':<{col_w[0]}} "
            f"{'Observed':>{col_w[1]}} "
            f"{'Mean(rep)':>{col_w[2]}} "
            f"{'p-value':>{col_w[3]}}"
        )
        lines.append(header)
        lines.append("  " + "-" * (sum(col_w) + 3))
        for name, (obs_val, rep_vals) in self._stats.items():
            p = float(np.mean(rep_vals >= obs_val))
            lines.append(
                f"  {name:<{col_w[0]}} "
                f"{obs_val:>{col_w[1]}.3f} "
                f"{float(np.mean(rep_vals)):>{col_w[2]}.3f} "
                f"{p:>{col_w[3]}.3f}"
            )
        return "\n".join(lines)


def compute_ppc(y_rep: NDArray, observed: NDArray) -> PPCResult:
    """Compute a posterior predictive check from a draw matrix and observed response.

    Parameters
    ----------
    y_rep : NDArray
        Posterior predictive draws, shape `(n, n_sim)`.
    observed : NDArray
        Observed response values, shape `(n,)`.

    Returns
    -------
    PPCResult
    """
    n_sim = y_rep.shape[1]

    def _rep_stat(fn: object) -> NDArray:
        return np.array([fn(y_rep[:, s]) for s in range(n_sim)])  # type: ignore[operator]

    stat_fns: dict[str, object] = {
        "mean": lambda v: float(np.mean(v)),
        "sd": lambda v: float(np.std(v, ddof=1)),
        "min": lambda v: float(np.min(v)),
        "max": lambda v: float(np.max(v)),
        "prop_zero": lambda v: float(np.mean(v == 0)),
    }

    stats: dict[str, tuple[float, NDArray]] = {}
    for name, fn in stat_fns.items():
        obs_val = fn(observed)  # type: ignore[operator]
        stats[name] = (obs_val, _rep_stat(fn))

    return PPCResult(y_rep=y_rep, observed=observed, _stats=stats)
