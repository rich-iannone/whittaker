"""Causal GAMs: partially linear models and double machine learning.

Implements causal inference with GAM-based nuisance estimation:

- **Partially linear model**: `Y = theta * D + f(X) + eps` where `D` is a treatment, `f(X)` is a
GAM for confounding, and `theta` is the ATE.
- **Interactive model**: `Y = g(D, X) + eps` where treatment effects can vary with covariates (CATE
estimation).
- **Double/Debiased ML (DML)**: Orthogonal moment estimation that remains valid even when the
nuisance GAM is regularised (Chernozhukov et al. 2018).
- **Mediation analysis**: Decomposes total effect into direct and indirect pathways through a
mediator variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from whittaker.data import InputData, prepare_data
from whittaker.families.base import Family
from whittaker.families.gaussian import Gaussian
from whittaker.gam import GAM


@dataclass
class TreatmentEffect:
    """Average treatment effect estimate with inference.

    Attributes
    ----------
    ate:
        Estimated average treatment effect.
    se:
        Standard error of the ATE estimate.
    ci_lower:
        Lower bound of the confidence interval.
    ci_upper:
        Upper bound of the confidence interval.
    level:
        Confidence level.
    p_value:
        Two-sided p-value for H0: ATE = 0.
    method:
        Estimation method used.
    n_obs:
        Number of observations.
    """

    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    level: float
    p_value: float
    method: str
    n_obs: int

    def __repr__(self) -> str:
        return (
            f"TreatmentEffect(ate={self.ate:.4f}, se={self.se:.4f}, "
            f"p={self.p_value:.4f}, "
            f"{self.level:.0%} CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )


@dataclass
class CATEResult:
    """Conditional average treatment effect estimates.

    Attributes
    ----------
    x:
        Covariate values at which CATE is evaluated.
    cate:
        CATE estimates.
    se:
        Standard errors.
    lower:
        Lower CI bounds.
    upper:
        Upper CI bounds.
    variable:
        Name of the conditioning variable.
    level:
        Confidence level.
    """

    x: NDArray
    cate: NDArray
    se: NDArray
    lower: NDArray
    upper: NDArray
    variable: str
    level: float


@dataclass
class MediationResult:
    """Mediation analysis results.

    Attributes
    ----------
    total_effect:
        Total effect of treatment on outcome.
    direct_effect:
        Direct effect (not through mediator).
    indirect_effect:
        Indirect effect (through mediator).
    proportion_mediated:
        Fraction of total effect mediated.
    total_se:
        SE of total effect.
    direct_se:
        SE of direct effect.
    indirect_se:
        SE of indirect effect.
    n_obs:
        Number of observations.
    """

    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    total_se: float
    direct_se: float
    indirect_se: float
    n_obs: int

    def __repr__(self) -> str:
        return (
            f"MediationResult(\n"
            f"  total={self.total_effect:.4f} (SE={self.total_se:.4f})\n"
            f"  direct={self.direct_effect:.4f} (SE={self.direct_se:.4f})\n"
            f"  indirect={self.indirect_effect:.4f} (SE={self.indirect_se:.4f})\n"
            f"  proportion_mediated={self.proportion_mediated:.2%}\n"
            f")"
        )


class CausalGAM:
    """Causal GAM for treatment effect estimation.

    Uses double/debiased machine learning (DML) with GAM nuisance functions to estimate treatment
    effects that are robust to regularization bias.

    Parameters
    ----------
    outcome:
        Name of the outcome variable.
    treatment:
        Name of the treatment variable.
    confounders:
        List of confounder variable names.
    method:
        `"partially_linear"` (default) for constant ATE, or `"interactive"` for heterogeneous
        treatment effects.
    family:
        Response distribution for the outcome model.
    n_folds:
        Number of cross-fitting folds for DML (default `5`).
    """

    def __init__(
        self,
        outcome: str,
        treatment: str,
        confounders: list[str],
        *,
        method: str = "partially_linear",
        family: Family | None = None,
        n_folds: int = 5,
    ) -> None:
        if method not in ("partially_linear", "interactive"):
            raise ValueError(f"method must be 'partially_linear' or 'interactive', got {method!r}")
        self._outcome = outcome
        self._treatment = treatment
        self._confounders = list(confounders)
        self._method = method
        self._family = family if family is not None else Gaussian()
        self._n_folds = n_folds
        self._fitted = False

        self._ate: float | None = None
        self._ate_se: float | None = None
        self._residuals_y: NDArray | None = None
        self._residuals_d: NDArray | None = None
        self._outcome_models: list[GAM] = []
        self._treatment_models: list[GAM] = []
        self._cate_model: GAM | None = None
        self._data: dict[str, NDArray] | None = None

    @property
    def outcome(self) -> str:
        return self._outcome

    @property
    def treatment(self) -> str:
        return self._treatment

    @property
    def confounders(self) -> list[str]:
        return list(self._confounders)

    @property
    def method(self) -> str:
        return self._method

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        data: InputData,
        *,
        fit_method: str = "REML",
        select: bool = False,
        seed: int | None = None,
    ) -> CausalGAM:
        """Fit the causal GAM via cross-fitted DML.

        Parameters
        ----------
        data:
            Column-oriented data containing outcome, treatment, and confounders.
        fit_method:
            Smoothing parameter method for nuisance GAMs.
        select:
            Enable variable selection in nuisance GAMs.
        seed:
            Random seed for fold assignment.

        Returns
        -------
        CausalGAM
            Returns `self` for method chaining.
        """
        arrays = prepare_data(data)
        self._data = arrays
        y = arrays[self._outcome]
        d = arrays[self._treatment]
        n = len(y)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        fold_ids = np.zeros(n, dtype=int)
        for i in range(n):
            fold_ids[perm[i]] = i * self._n_folds // n

        smooth_terms = " + ".join(f"s({c})" for c in self._confounders)
        outcome_formula = f"{self._outcome} ~ {smooth_terms}"
        treatment_formula = f"{self._treatment} ~ {smooth_terms}"

        residuals_y = np.zeros(n)
        residuals_d = np.zeros(n)
        self._outcome_models = []
        self._treatment_models = []

        for fold in range(self._n_folds):
            train = fold_ids != fold
            test = fold_ids == fold

            train_data = {k: v[train] for k, v in arrays.items()}
            test_data = {k: v[test] for k, v in arrays.items()}

            y_model = GAM(outcome_formula, family=self._family)
            y_model.fit(train_data, method=fit_method, select=select)
            self._outcome_models.append(y_model)

            d_model = GAM(treatment_formula, family=Gaussian())
            d_model.fit(train_data, method=fit_method, select=select)
            self._treatment_models.append(d_model)

            residuals_y[test] = y[test] - y_model.predict(test_data).values
            residuals_d[test] = d[test] - d_model.predict(test_data).values

        self._residuals_y = residuals_y
        self._residuals_d = residuals_d

        denom = np.sum(residuals_d * residuals_d)
        self._ate = float(np.sum(residuals_d * residuals_y) / denom)

        psi = residuals_d * (residuals_y - self._ate * residuals_d)
        self._ate_se = float(np.sqrt(np.mean(psi**2) / denom))

        if self._method == "interactive":
            self._fit_cate(arrays, fit_method, select)

        self._fitted = True
        return self

    def _fit_cate(
        self,
        arrays: dict[str, NDArray],
        fit_method: str,
        select: bool,
    ) -> None:
        pseudo_outcome = self._residuals_y / self._residuals_d
        finite_mask = np.isfinite(pseudo_outcome) & (np.abs(self._residuals_d) > 1e-6)

        if np.sum(finite_mask) < 20:
            self._cate_model = None
            return

        smooth_terms = " + ".join(f"s({c})" for c in self._confounders)
        cate_formula = f"_pseudo_y ~ {smooth_terms}"

        cate_data = {c: arrays[c][finite_mask] for c in self._confounders}
        cate_data["_pseudo_y"] = pseudo_outcome[finite_mask]

        weights = self._residuals_d[finite_mask] ** 2

        self._cate_model = GAM(cate_formula, family=Gaussian())
        self._cate_model.fit(cate_data, method=fit_method, select=select, weights=weights)

    def treatment_effect(self, level: float = 0.95) -> TreatmentEffect:
        """Compute the average treatment effect with inference.

        Parameters
        ----------
        level:
            Confidence level (default `0.95`).

        Returns
        -------
        TreatmentEffect
        """
        self._check_fitted()

        z = stats.norm.ppf(1.0 - (1.0 - level) / 2)
        ci_lower = self._ate - z * self._ate_se
        ci_upper = self._ate + z * self._ate_se

        t_stat = self._ate / self._ate_se if self._ate_se > 0 else np.inf
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

        return TreatmentEffect(
            ate=self._ate,
            se=self._ate_se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            level=level,
            p_value=p_value,
            method=self._method,
            n_obs=len(self._residuals_y),
        )

    def cate(
        self,
        new_data: InputData | None = None,
        *,
        variable: str | None = None,
        n_points: int = 100,
        level: float = 0.95,
    ) -> CATEResult:
        """Estimate conditional average treatment effects.

        Requires `method="interactive"`. Returns CATE as a function of a chosen confounder variable.

        Parameters
        ----------
        new_data:
            Covariate data for prediction. If `None`, evaluates on a grid of the specified variable.
        variable:
            Confounder to condition on. Required if `new_data` is `None`. Defaults to the first
            confounder.
        n_points:
            Number of grid points (used when `new_data` is `None`).
        level:
            Confidence level for intervals.

        Returns
        -------
        CATEResult
        """
        self._check_fitted()

        if self._method != "interactive":
            raise ValueError(
                "CATE estimation requires method='interactive'. "
                "Refit with CausalGAM(..., method='interactive')."
            )

        if self._cate_model is None:
            raise RuntimeError("CATE model could not be fitted (too few valid pseudo-outcomes).")

        if variable is None:
            variable = self._confounders[0]

        if variable not in self._confounders:
            raise ValueError(
                f"Variable {variable!r} is not among the confounders: {self._confounders}"
            )

        if new_data is None:
            x_var = self._data[variable]
            x_grid = np.linspace(x_var.min(), x_var.max(), n_points)
            new_data = {}
            for c in self._confounders:
                if c == variable:
                    new_data[c] = x_grid
                else:
                    new_data[c] = np.full(n_points, np.mean(self._data[c]))
        else:
            new_data = prepare_data(new_data)
            x_grid = new_data[variable]
            n_points = len(x_grid)

        pred = self._cate_model.predict(new_data, se=True)
        cate_vals = pred.values
        se_vals = pred.se

        z = stats.norm.ppf(1.0 - (1.0 - level) / 2)
        lower = cate_vals - z * se_vals
        upper = cate_vals + z * se_vals

        return CATEResult(
            x=x_grid,
            cate=cate_vals,
            se=se_vals,
            lower=lower,
            upper=upper,
            variable=variable,
            level=level,
        )

    def residuals(self) -> tuple[NDArray, NDArray]:
        """Return the orthogonalized residuals.

        Returns
        -------
        tuple[NDArray, NDArray]
            `(residuals_y, residuals_d)`, a residualized outcome and treatment after partialling out
            confounders.
        """
        self._check_fitted()
        return self._residuals_y.copy(), self._residuals_d.copy()

    def summary(self) -> str:
        """Text summary of the causal GAM."""
        self._check_fitted()

        te = self.treatment_effect()
        lines = [
            "CausalGAM summary",
            "=" * 60,
            f"Outcome:     {self._outcome}",
            f"Treatment:   {self._treatment}",
            f"Confounders: {', '.join(self._confounders)}",
            f"Method:      {self._method}",
            f"N folds:     {self._n_folds}",
            f"N obs:       {te.n_obs}",
            "",
            "Treatment effect:",
            f"  ATE = {te.ate:.4f} (SE = {te.se:.4f})",
            f"  95% CI: [{te.ci_lower:.4f}, {te.ci_upper:.4f}]",
            f"  p-value: {te.p_value:.4f}",
        ]

        if self._method == "interactive" and self._cate_model is not None:
            lines.append("")
            lines.append("CATE model fitted (use .cate() for estimates)")

        return "\n".join(lines)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("This CausalGAM has not been fitted yet. Call .fit(data) first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"CausalGAM(outcome={self._outcome!r}, "
            f"treatment={self._treatment!r}, "
            f"method={self._method!r}, {status})"
        )


def mediation_analysis(
    outcome: str,
    treatment: str,
    mediator: str,
    confounders: list[str],
    data: InputData,
    *,
    family: Family | None = None,
    fit_method: str = "REML",
    select: bool = False,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> MediationResult:
    """Causal mediation analysis with GAM nuisance models.

    Estimates direct and indirect effects of `treatment` on `outcome` through a `mediator`,
    controlling for `confounders`. Uses the simulation-based approach (Imai, Keele, Tingley 2010).

    Parameters
    ----------
    outcome:
        Name of the outcome variable.
    treatment:
        Name of the treatment variable (binary 0/1).
    mediator:
        Name of the mediator variable.
    confounders:
        List of confounder variable names.
    data:
        Column-oriented data.
    family:
        Response distribution for the outcome model.
    fit_method:
        Smoothing parameter method.
    select:
        Enable variable selection.
    n_simulations:
        Number of Monte Carlo draws for inference.
    seed:
        Random seed.

    Returns
    -------
    MediationResult
    """
    if family is None:
        family = Gaussian()

    arrays = prepare_data(data)
    rng = np.random.default_rng(seed)
    n = len(arrays[outcome])
    y = arrays[outcome]
    d = arrays[treatment]

    smooth_terms_conf = " + ".join(f"s({c})" for c in confounders)

    mediator_formula = f"{mediator} ~ {treatment} + {smooth_terms_conf}"
    mediator_model = GAM(mediator_formula, family=Gaussian())
    mediator_model.fit(arrays, method=fit_method, select=select)

    outcome_formula = f"{outcome} ~ {treatment} + {mediator} + {smooth_terms_conf}"
    outcome_model = GAM(outcome_formula, family=family)
    outcome_model.fit(arrays, method=fit_method, select=select)

    d1_data = {k: v.copy() for k, v in arrays.items()}
    d0_data = {k: v.copy() for k, v in arrays.items()}
    d1_data[treatment] = np.ones(n)
    d0_data[treatment] = np.zeros(n)

    m1 = mediator_model.predict(d1_data).values
    m0 = mediator_model.predict(d0_data).values

    def _compute_effects(d1_d, d0_d, m1_v, m0_v):
        d1_m1_data = {k: v.copy() for k, v in arrays.items()}
        d1_m1_data[treatment] = np.ones(n)
        d1_m1_data[mediator] = m1_v

        d1_m0_data = {k: v.copy() for k, v in arrays.items()}
        d1_m0_data[treatment] = np.ones(n)
        d1_m0_data[mediator] = m0_v

        d0_m0_data = {k: v.copy() for k, v in arrays.items()}
        d0_m0_data[treatment] = np.zeros(n)
        d0_m0_data[mediator] = m0_v

        y_d1_m1 = outcome_model.predict(d1_m1_data).values
        y_d1_m0 = outcome_model.predict(d1_m0_data).values
        y_d0_m0 = outcome_model.predict(d0_m0_data).values

        indirect = float(np.mean(y_d1_m1 - y_d1_m0))
        direct = float(np.mean(y_d1_m0 - y_d0_m0))
        total = indirect + direct
        return total, direct, indirect

    total, direct, indirect = _compute_effects(d1_data, d0_data, m1, m0)

    totals = np.zeros(n_simulations)
    directs = np.zeros(n_simulations)
    indirects = np.zeros(n_simulations)

    for sim in range(n_simulations):
        boot_idx = rng.integers(0, n, size=n)
        boot_arrays = {k: v[boot_idx] for k, v in arrays.items()}

        m_model_b = GAM(mediator_formula, family=Gaussian())
        m_model_b.fit(boot_arrays, method=fit_method, select=select)

        o_model_b = GAM(outcome_formula, family=family)
        o_model_b.fit(boot_arrays, method=fit_method, select=select)

        d1_b = {k: v.copy() for k, v in boot_arrays.items()}
        d0_b = {k: v.copy() for k, v in boot_arrays.items()}
        d1_b[treatment] = np.ones(n)
        d0_b[treatment] = np.zeros(n)

        m1_b = m_model_b.predict(d1_b).values
        m0_b = m_model_b.predict(d0_b).values

        d1_m1_b = {k: v.copy() for k, v in boot_arrays.items()}
        d1_m1_b[treatment] = np.ones(n)
        d1_m1_b[mediator] = m1_b

        d1_m0_b = {k: v.copy() for k, v in boot_arrays.items()}
        d1_m0_b[treatment] = np.ones(n)
        d1_m0_b[mediator] = m0_b

        d0_m0_b = {k: v.copy() for k, v in boot_arrays.items()}
        d0_m0_b[treatment] = np.zeros(n)
        d0_m0_b[mediator] = m0_b

        y_d1_m1_b = o_model_b.predict(d1_m1_b).values
        y_d1_m0_b = o_model_b.predict(d1_m0_b).values
        y_d0_m0_b = o_model_b.predict(d0_m0_b).values

        indirects[sim] = float(np.mean(y_d1_m1_b - y_d1_m0_b))
        directs[sim] = float(np.mean(y_d1_m0_b - y_d0_m0_b))
        totals[sim] = indirects[sim] + directs[sim]

    proportion = indirect / total if abs(total) > 1e-10 else 0.0

    return MediationResult(
        total_effect=total,
        direct_effect=direct,
        indirect_effect=indirect,
        proportion_mediated=proportion,
        total_se=float(np.std(totals, ddof=1)),
        direct_se=float(np.std(directs, ddof=1)),
        indirect_se=float(np.std(indirects, ddof=1)),
        n_obs=n,
    )
