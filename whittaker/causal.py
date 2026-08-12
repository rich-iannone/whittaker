r"""Causal GAMs: partially linear models and double machine learning.

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
    r"""Average treatment effect estimate with inference.

    Returned by `CausalGAM.treatment_effect()`, this holds the debiased/double-machine-learning
    estimate of the average treatment effect (ATE) together with its standard error, confidence
    interval, and a Wald test against the null of no effect.

    Attributes
    ----------
    ate:
        Estimated average treatment effect: the coefficient `theta` in the partially linear model
        `Y = theta * D + f(X) + eps`, or its interactive-model analogue.
    se:
        Standard error of the ATE estimate, computed from the influence function of the DML moment
        condition.
    ci_lower:
        Lower bound of the `level`-confidence interval, `ate - z * se`.
    ci_upper:
        Upper bound of the `level`-confidence interval, `ate + z * se`.
    level:
        Confidence level used to construct the interval (e.g. `0.95`).
    p_value:
        Two-sided p-value for `H0: ATE = 0`, from a normal (Wald) approximation.
    method:
        Estimation method used (`"partially_linear"` or `"interactive"`).
    n_obs:
        Number of observations used in estimation.
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
    r"""Conditional average treatment effect estimates.

    Returned by `CausalGAM.cate()` when the model was fit with `method="interactive"`. Represents
    the treatment effect as a smooth function of one confounder variable, evaluated on a grid (or on
    user-supplied covariate data), together with pointwise confidence bands.

    Attributes
    ----------
    x:
        Covariate values of `variable` at which CATE is evaluated.
    cate:
        CATE estimates, `tau(x) = E[Y(1) - Y(0) | X = x]`, at each value of `x`.
    se:
        Standard errors of the CATE estimates.
    lower:
        Lower pointwise confidence bounds, `cate - z * se`.
    upper:
        Upper pointwise confidence bounds, `cate + z * se`.
    variable:
        Name of the conditioning (confounder) variable that CATE is plotted against.
    level:
        Confidence level used for the bands.
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
    r"""Mediation analysis results.

    Returned by `mediation_analysis()`. Decomposes the total effect of a treatment on an outcome
    into a direct component (not passing through the mediator) and an indirect component (passing
    through the mediator), following the potential-outcomes framework of Imai, Keele, & Tingley
    (2010). Standard errors are obtained by a nonparametric bootstrap over the whole estimation
    procedure (refitting both the mediator and outcome GAMs on each resample).

    Attributes
    ----------
    total_effect:
        Total effect of treatment on outcome, `direct_effect + indirect_effect`.
    direct_effect:
        Direct effect of treatment on outcome, holding the mediator fixed at its value under
        treatment (natural direct effect).
    indirect_effect:
        Indirect effect of treatment on outcome operating through the mediator (natural indirect
        effect).
    proportion_mediated:
        Fraction of the total effect attributable to the mediator, `indirect_effect / total_effect`.
    total_se:
        Bootstrap standard error of the total effect.
    direct_se:
        Bootstrap standard error of the direct effect.
    indirect_se:
        Bootstrap standard error of the indirect effect.
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
    r"""Causal GAM for treatment effect estimation.

    Estimates the causal effect of a treatment `D` on an outcome `Y`, controlling for a set of
    confounders `X`, using double/debiased machine learning (DML; Chernozhukov et al. 2018) with GAM
    nuisance models. Two structural forms are supported:

    - **Partially linear** (`method="partially_linear"`): `Y = theta * D + f(X) + eps`, giving a
      single constant average treatment effect (ATE) `theta`.
    - **Interactive** (`method="interactive"`): `Y = g(D, X) + eps`, allowing the treatment effect
      to vary smoothly with `X` (conditional average treatment effect, CATE).

    DML addresses the regularization bias that arises when flexible, penalized nuisance models
    (here, GAMs for `E[Y | X]` and `E[D | X]`) are plugged directly into a naive treatment-effect
    estimator: the smoothing bias in the nuisance fits would otherwise leak into the treatment
    effect estimate. Cross-fitting (fitting nuisance models on one subset of folds and evaluating
    residuals on the held-out fold) together with a Neyman-orthogonal moment condition makes the
    resulting ATE estimate root-n consistent and asymptotically normal even though the nuisance GAMs
    converge at slower nonparametric rates.

    Use `CausalGAM` for observational-data effect estimation where confounding is plausibly
    captured by smooth functions of observed covariates, and you want valid inference (standard
    errors, confidence intervals) on the treatment effect rather than just a point prediction.

    Parameters
    ----------
    outcome:
        Name of the outcome variable.
    treatment:
        Name of the treatment variable.
    confounders:
        List of confounder variable names. Both the outcome and treatment nuisance GAMs use
        `s(c)` smooth terms for each confounder `c`.
    method:
        `"partially_linear"` (default) for constant ATE, or `"interactive"` for heterogeneous
        treatment effects (enables `.cate()`).
    family:
        Response distribution for the outcome nuisance model. Defaults to `Gaussian()`. The
        treatment nuisance model always uses `Gaussian()` regardless of this setting, since DML
        residualizes the treatment via its conditional mean.
    n_folds:
        Number of cross-fitting folds for DML (default `5`). Each fold's nuisance models are fit on
        the other `n_folds - 1` folds and evaluated on the held-out fold to avoid overfitting bias.

    Notes
    -----
    Fitting proceeds in three steps. First, cross-fitted residuals are formed for both outcome and
    treatment:

    $$\hat\varepsilon_{Y,i} = Y_i - \hat m_Y(X_i), \qquad
    \hat\varepsilon_{D,i} = D_i - \hat m_D(X_i)$$

    where `\hat m_Y` and `\hat m_D` are GAM estimates of `E[Y \mid X]` and `E[D \mid X]`, each fit on
    folds excluding observation `i`. Second, the ATE is estimated by the residual-on-residual
    regression (the partialling-out estimator):

    $$\hat\theta = \frac{\sum_i \hat\varepsilon_{D,i} \, \hat\varepsilon_{Y,i}}
    {\sum_i \hat\varepsilon_{D,i}^2}$$

    Third, its standard error is derived from the empirical variance of the Neyman-orthogonal score
    $\psi_i = \hat\varepsilon_{D,i}(\hat\varepsilon_{Y,i} - \hat\theta \hat\varepsilon_{D,i})$:

    $$\widehat{\mathrm{se}}(\hat\theta) = \sqrt{\frac{\overline{\psi^2}}
    {\left(\sum_i \hat\varepsilon_{D,i}^2\right)^{2} / n}}$$

    When `method="interactive"`, a further GAM is fit on the pseudo-outcome
    `\hat\varepsilon_{Y,i} / \hat\varepsilon_{D,i}`, weighted by `\hat\varepsilon_{D,i}^2`, to recover
    the CATE as a smooth function of the confounders.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.causal import CausalGAM

    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(0, 1, n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(2 * x - 1))), n).astype(float)
    y = 1.5 * d + np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=n)

    model = CausalGAM(outcome="y", treatment="d", confounders=["x"], n_folds=5)
    model.fit({"x": x, "d": d, "y": y}, seed=0)
    print(model.treatment_effect())
    ```
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
        r"""Fit the causal GAM via cross-fitted DML.

        Randomly assigns observations to `n_folds` folds, then for each fold fits an outcome GAM
        `E[Y | X]` and a treatment GAM `E[D | X]` on the remaining folds and predicts residuals on
        the held-out fold. The pooled cross-fitted residuals are combined into the partialling-out
        ATE estimate and its standard error (see class Notes). If `method="interactive"`, an
        additional CATE model is fit on the residual ratio.

        Parameters
        ----------
        data:
            Column-oriented data containing outcome, treatment, and confounder columns.
        fit_method:
            Smoothing parameter selection method for the outcome and treatment nuisance GAMs
            (default `"REML"`).
        select:
            Enable double-penalty variable selection in the nuisance GAMs.
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
        r"""Compute the average treatment effect with inference.

        Builds a `TreatmentEffect` from the ATE and standard error computed during `fit()`, adding a
        normal-approximation confidence interval and two-sided Wald p-value for `H0: ATE = 0`.

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
        r"""Estimate conditional average treatment effects.

        Requires `method="interactive"`. Evaluates the fitted CATE model (a GAM regressed on the
        pseudo-outcome `\hat\varepsilon_Y / \hat\varepsilon_D`) either on user-supplied `new_data` or
        on a grid over one confounder, holding the other confounders at their training-data means.
        Returns CATE as a function of a chosen confounder variable, together with pointwise
        confidence bands derived from the CATE model's own standard errors.

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
    r"""Causal mediation analysis with GAM nuisance models.

    Estimates how much of the total effect of `treatment` on `outcome` operates through an
    intermediate `mediator` variable, versus acting directly, controlling for `confounders`. Uses
    the simulation-based approach of Imai, Keele, & Tingley (2010): a mediator GAM
    `E[M | D, X]` and an outcome GAM `E[Y | D, M, X]` are fit on the observed data, and then used to
    predict the outcome under counterfactual combinations of treatment and mediator status that
    isolate the direct and indirect pathways.

    Use this when you have a hypothesized causal chain `treatment -> mediator -> outcome` (plus a
    possible direct `treatment -> outcome` path) and want to decompose the total causal effect into
    how much passes through the mediator versus how much does not.

    Parameters
    ----------
    outcome:
        Name of the outcome variable.
    treatment:
        Name of the treatment variable (binary 0/1).
    mediator:
        Name of the mediator variable.
    confounders:
        List of confounder variable names, entered as smooth terms in both the mediator and outcome
        models.
    data:
        Column-oriented data containing outcome, treatment, mediator, and confounder columns.
    family:
        Response distribution for the outcome model. Defaults to `Gaussian()`. The mediator model
        always uses `Gaussian()`.
    fit_method:
        Smoothing parameter selection method for both nuisance models.
    select:
        Enable double-penalty variable selection in the nuisance models.
    n_simulations:
        Number of bootstrap resamples used to estimate standard errors for the total, direct, and
        indirect effects.
    seed:
        Random seed for the bootstrap.

    Notes
    -----
    Natural direct and indirect effects are computed by contrasting predicted outcomes under three
    counterfactual scenarios, holding treatment fixed at `d \in \{0, 1\}` and setting the mediator to
    its predicted value under either treatment level:

    $$\text{indirect} = \frac{1}{n}\sum_i \left[\hat Y_i(1, \hat M_i(1)) - \hat Y_i(1, \hat
    M_i(0))\right], \qquad
    \text{direct} = \frac{1}{n}\sum_i \left[\hat Y_i(1, \hat M_i(0)) - \hat Y_i(0, \hat M_i(0))\right]$$

    where `\hat Y_i(d, m)` is the outcome GAM's prediction with treatment set to `d` and mediator set
    to `m`, and `\hat M_i(d)` is the mediator GAM's prediction with treatment set to `d`. The total
    effect is `indirect + direct`, and `proportion_mediated = indirect / total`. Standard errors for
    all three quantities come from re-running the full procedure (refitting both GAMs) on
    `n_simulations` bootstrap resamples of the data.

    Returns
    -------
    MediationResult
        The total, direct, and indirect effects with bootstrap standard errors, and the proportion
        of the total effect that is mediated.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.causal import mediation_analysis

    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0, 1, n)
    d = rng.binomial(1, 0.5, n).astype(float)
    m = 0.5 * d + 0.3 * x + rng.normal(scale=0.2, size=n)
    y = 0.4 * d + 0.8 * m + np.sin(2 * np.pi * x) + rng.normal(scale=0.3, size=n)

    result = mediation_analysis(
        outcome="y", treatment="d", mediator="m", confounders=["x"],
        data={"x": x, "d": d, "m": m, "y": y}, n_simulations=200, seed=0,
    )
    print(result)
    ```
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
